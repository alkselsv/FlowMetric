#!/usr/bin/env python3
"""Query GPT-6 Astra (or another VLM) on sampled frames once per sampling variant; cache to JSON.

The responses are tracker-independent, so every detector/tracker combination reuses them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from astra import astra_reid_people, make_client, scale_astra_boxes_to_original  # noqa: E402
from track import make_batches  # noqa: E402

OUT = ROOT / "data" / "experiments" / "astra"


def sample_frames(video: Path, step: int) -> tuple[list[int], list]:
    cap = cv2.VideoCapture(str(video))
    idx, frames, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            idx.append(i)
            frames.append(f)
        i += 1
    cap.release()
    return idx, frames


def run_variant(video: Path, name: str, *, fps_sample: float, model: str | None, effort: str, batch: int) -> None:
    out = OUT / f"{name}.json"
    if out.exists():
        print(f"[{name}] cached")
        return
    client, default_model = make_client()
    model = model or default_model
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / fps_sample))
    idx, frames = sample_frames(video, step)
    batches = make_batches(len(frames), batch)
    print(f"[{name}] {len(frames)} frames, {len(batches)} call(s), model {model}")
    t0 = time.time()

    def call(b: list[int]) -> dict:
        r = astra_reid_people([frames[i] for i in b], model=model, reasoning_effort=effort, client=client)
        r = scale_astra_boxes_to_original(r)
        r["_meta"]["video_frames"] = [idx[i] for i in b]
        return r

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            reids = list(pool.map(call, batches))
    except Exception as e:  # keep other variants running
        (OUT / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}")
        print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
        return
    meta = {"model": model, "effort": effort, "sample_fps": fps_sample, "frames": len(frames),
            "calls": len(batches), "wall_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps({"meta": meta, "reids": reids}, ensure_ascii=False))
    print(f"[{name}] done in {meta['wall_s']}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=next((ROOT / "data").glob("*.mp4")))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1 fps Astra response from the production run (same sampling: every 24th frame).
    prod = ROOT / "data" / "output" / "astra_reid.json"
    if prod.exists() and not (OUT / "astra_1fps.json").exists():
        reids = json.loads(prod.read_text())
        meta = {"model": "openai/gpt-6-astra", "effort": "low", "sample_fps": 1.0,
                "frames": sum(len(r["_meta"]["video_frames"]) for r in reids), "calls": len(reids), "wall_s": None}
        (OUT / "astra_1fps.json").write_text(json.dumps({"meta": meta, "reids": reids}, ensure_ascii=False))

    # Repeats (_r2, _r3) measure run-to-run variance of the VLM answer.
    variants = [
        ("astra_0.5fps", 0.5, None, "low"),
        ("astra_2fps", 2.0, None, "low"),
        ("luna_1fps", 1.0, "openai/gpt-6-luna", "low"),
        ("astra_1fps_r2", 1.0, None, "low"),
        ("astra_1fps_r3", 1.0, None, "low"),
        ("astra_2fps_r2", 2.0, None, "low"),
        ("astra_1fps_medium", 1.0, None, "medium"),
    ]
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        list(pool.map(lambda v: run_variant(args.video, v[0], fps_sample=v[1], model=v[2], effort=v[3], batch=16), variants))


if __name__ == "__main__":
    main()
