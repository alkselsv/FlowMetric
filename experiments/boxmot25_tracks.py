#!/usr/bin/env python3
"""BoxMOT 25.x trackers (GitHub master: OccluBoost, SFSORT, ...) on cached detections.

BoxMOT 25 cannot share a process with the pinned BoxMOT 12, so run it from its source tree:
    git clone --depth 1 https://github.com/mikel-brostrom/boxmot .boxmot25
    PYTHONPATH=.boxmot25 python experiments/boxmot25_tracks.py
Writes data/experiments/tracks/{det}__bm25-{tracker}[-u].w3.pkl, which run_experiments.py --stages eval scores.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "experiments"
WARMUP_FRAMES = 3  # same as run_experiments.py
DEVICE = "mps"
REID = ROOT / "weights" / "osnet_x0_25_msmt17.pt"

# Our shared thresholds (same as the BoxMOT 12 grid): new tracks at >= 0.4, high/det at 0.3, low at 0.1.
UNIFIED = {
    "occluboost": {"det_thresh": 0.3, "new_track_thresh": 0.4, "instant_confirm_thresh": 0.5},
    "botsort": {"track_high_thresh": 0.3, "track_low_thresh": 0.1, "new_track_thresh": 0.4, "det_thresh": 0.3},
    "hybridsort": {"det_thresh": 0.3, "track_thresh": 0.3},
    "boosttrack": {"det_thresh": 0.3},
    "bytetrack": {"track_thresh": 0.3},
    "ocsort": {"det_thresh": 0.3},
    "sfsort": {"high_th": 0.3, "new_track_th": 0.4, "low_th": 0.1},
}
WITH_REID = {"occluboost", "botsort", "hybridsort", "boosttrack"}
CLASSES = {
    "occluboost": ("OccluBoost", "OccluBoostConfig"),
    "botsort": ("BotSort", "BotSortConfig"),
    "hybridsort": ("HybridSort", "HybridSortConfig"),
    "boosttrack": ("BoostTrack", "BoostTrackConfig"),
    "bytetrack": ("ByteTrack", "ByteTrackConfig"),
    "ocsort": ("OcSort", "OcSortConfig"),
    "sfsort": ("SFSORT", "SFSORTConfig"),
}


def make_tracker(name: str, unified: bool, fps: float):
    import dataclasses

    import boxmot

    cls_name, cfg_name = CLASSES[name]
    cfg_cls = getattr(boxmot, cfg_name)
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    kw = {"use_cmc": False} if "use_cmc" in fields else {}  # static camera
    if unified:
        kw |= UNIFIED[name]
    cfg = cfg_cls(**{k: v for k, v in kw.items() if k in fields})
    cls = getattr(boxmot, cls_name)
    if name in WITH_REID:
        return cls(cfg, reid=boxmot.ReIDConfig(model=str(REID), device=DEVICE))
    return cls(cfg)


def run(frames, fps, dets, name: str, unified: bool) -> dict:
    t = make_tracker(name, unified, fps)

    def d6(d: np.ndarray) -> np.ndarray:
        return np.hstack([d, np.zeros((len(d), 1), np.float32)]) if len(d) else np.empty((0, 6), np.float32)

    for _ in range(WARMUP_FRAMES):
        t.update(d6(dets[0]), frames[0])
    out: dict[int, list] = {}
    t0 = time.perf_counter()
    for fi, (frame, d) in enumerate(zip(frames, dets)):
        arr = np.asarray(t.update(d6(d), frame)).reshape(-1, 8)
        out[fi] = [(int(r[4]), r[:4].astype(float), float(r[5])) for r in arr]
    ms = (time.perf_counter() - t0) / len(frames) * 1000
    h, w = frames[0].shape[:2]
    return {"lp_dets": out, "fps": fps, "w": w, "h": h, "n": len(frames), "ms_trk": ms}


def main() -> None:
    import boxmot

    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=next((ROOT / "data").glob("*.mp4")))
    p.add_argument("--detectors", nargs="*",
                   default=["yolo11s@1280", "yolo11m@1280", "yolo26s@1280", "yolo26m@1280", "rfdetr-m+cal",
                            "rfdetr-l@1024+cal"])
    p.add_argument("--trackers", nargs="*", default=list(CLASSES))
    args = p.parse_args()
    print(f"boxmot {boxmot.__version__} from {Path(boxmot.__file__).parent}", flush=True)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    for det in args.detectors:
        dets = pickle.loads((EXP / "dets" / f"{det}.pkl").read_bytes())["dets"]
        for name in args.trackers:
            for unified in (False, True):
                key = f"bm25-{name}{'-u' if unified else ''}"
                path = EXP / "tracks" / f"{det}__{key}.w{WARMUP_FRAMES}.pkl"
                if path.exists():
                    continue
                try:
                    rec = run(frames, fps, dets, name, unified)
                except Exception as e:
                    print(f"[bm25] {det} + {key} FAILED: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    continue
                path.write_bytes(pickle.dumps(rec))
                ids = len({t for v in rec["lp_dets"].values() for t, _, _ in v})
                print(f"[bm25] {det} + {key}: {rec['ms_trk']:.0f} ms/frame, {ids} ids", flush=True)


if __name__ == "__main__":
    sys.exit(main())
