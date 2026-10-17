#!/usr/bin/env python3
"""FlowMetric hybrid people tracking.

~95% local: YOLO + BoT-SORT/ByteTrack on every frame (boxes, frame-to-frame IDs).
~5% GPT-6 Astra: sampled frames (default 1 fps, 16-frame batches) for ReID + role,
via VseLLM (VSELLM_API_KEY) or OpenAI directly (OPENAI_API_KEY).
Local tracks are synced to Astra identities via IoU matching + constrained union-find,
which merges fragmented tracker IDs into stable person IDs. Service events are derived
from the trajectories (events.py) and scored against benchmark/ground_truth.json.
Output: annotated video, tracks.json, events.json, benchmark_report.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Pass 1: local detection + tracking
# ---------------------------------------------------------------------------


@dataclass
class LocalPass:
    fps: float
    width: int
    height: int
    total: int
    # frame_idx -> list of (tracker_id, xyxy np.array[4], conf)
    dets: dict[int, list[tuple[int, np.ndarray, float]]] = field(default_factory=dict)
    sample_idx: list[int] = field(default_factory=list)
    sample_frames: list[np.ndarray] = field(default_factory=list)


def local_track(
    source: Path,
    *,
    model_name: str,
    conf: float,
    iou: float,
    tracker: str,
    sample_fps: float,
    imgsz: int = 1280,
) -> LocalPass:
    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    lp = LocalPass(
        fps=fps,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            imgsz=imgsz,  # 640 misses distant seated people on wide CCTV frames
            classes=[0],  # COCO person
            tracker=tracker,
            verbose=False,
        )[0]
        frame_dets: list[tuple[int, np.ndarray, float]] = []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid, score in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.id.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
            ):
                frame_dets.append((int(tid), box.astype(float), float(score)))
        lp.dets[idx] = frame_dets

        if step and idx % step == 0:
            lp.sample_idx.append(idx)
            lp.sample_frames.append(frame.copy())
        idx += 1
        if idx % 50 == 0:
            print(f"  local: {idx}/{lp.total} frames")

    cap.release()
    lp.total = idx
    return lp


# ---------------------------------------------------------------------------
# Pass 2: GPT-6 Astra ReID on sampled frames
# ---------------------------------------------------------------------------


def make_batches(n: int, batch_size: int, overlap: int = 1) -> list[list[int]]:
    """Index batches over sampled frames with overlap so IDs chain across batches."""
    if n == 0:
        return []
    batches: list[list[int]] = []
    start = 0
    stride = max(1, batch_size - overlap)
    while start < n:
        batches.append(list(range(start, min(start + batch_size, n))))
        if start + batch_size >= n:
            break
        start += stride
    return batches


def astra_pass(
    lp: LocalPass,
    *,
    model: str | None,
    effort: str,
    batch_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    from astra import astra_reid_people, make_client, scale_astra_boxes_to_original

    client, default_model = make_client()
    model = model or default_model
    batches = make_batches(len(lp.sample_frames), batch_size)
    print(
        f"  astra: {len(lp.sample_frames)} sampled frames → {len(batches)} call(s) "
        f"to {model} @ {client.base_url}"
    )

    def run(b: list[int]) -> dict[str, Any]:
        reid = astra_reid_people(
            [lp.sample_frames[i] for i in b],
            model=model,
            reasoning_effort=effort,
            client=client,
        )
        reid = scale_astra_boxes_to_original(reid)
        reid["_meta"]["video_frames"] = [lp.sample_idx[i] for i in b]
        return reid

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(run, batches))


# ---------------------------------------------------------------------------
# Sync: local tracker IDs ↔ Astra identities
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class ConstrainedUnionFind:
    """Union-find where a component may hold at most one Astra id per batch."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.astra: dict[str, dict[int, int]] = {}  # root -> {batch: astra_id}

    def add(self, node: str, batch: int | None = None, aid: int | None = None) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.astra[node] = {batch: aid} if batch is not None else {}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        ma, mb = self.astra[ra], self.astra[rb]
        for batch, aid in mb.items():
            if batch in ma and ma[batch] != aid:
                return False  # Astra says these are different people
        self.parent[rb] = ra
        ma.update(mb)
        del self.astra[rb]
        return True


def sync_ids(
    lp: LocalPass,
    reids: list[dict[str, Any]],
    *,
    iou_thr: float,
) -> tuple[dict[int, int], dict[int, str], dict[str, Any]]:
    """Return tracker_id -> global_id, global_id -> role, sync stats."""
    votes: dict[int, Counter] = defaultdict(Counter)  # tid -> Counter[(batch, aid)]
    roles: dict[tuple[int, int], str] = {}
    matched = total_local = 0

    for b, reid in enumerate(reids):
        vframes = reid["_meta"]["video_frames"]
        per_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for person in reid.get("people", []):
            roles[(b, person["id"])] = person["role"]
            for det in person["detections"]:
                fi = det["frame_index"]
                if 0 <= fi < len(vframes):
                    per_frame[vframes[fi]].append((person["id"], np.asarray(det["xyxy"], float)))

        for vf in vframes:
            local = lp.dets.get(vf, [])
            astra = per_frame.get(vf, [])
            total_local += len(local)
            if not local or not astra:
                continue
            m = iou_matrix(
                np.stack([d[1] for d in local]),
                np.stack([a[1] for a in astra]),
            )
            rows, cols = linear_sum_assignment(-m)
            for r, c in zip(rows, cols):
                if m[r, c] >= iou_thr:
                    votes[local[r][0]][(b, astra[c][0])] += 1
                    matched += 1

    uf = ConstrainedUnionFind()
    all_tids = {tid for dets in lp.dets.values() for tid, _, _ in dets}
    for tid in all_tids:
        uf.add(f"T{tid}")

    rejected = 0
    for tid, cnt in votes.items():
        best_per_batch: dict[int, tuple[int, int]] = {}
        for (b, aid), n in cnt.items():
            if b not in best_per_batch or n > best_per_batch[b][1]:
                best_per_batch[b] = (aid, n)
        for b, (aid, _) in best_per_batch.items():
            node = f"A{b}:{aid}"
            uf.add(node, b, aid)
            if not uf.union(f"T{tid}", node):
                rejected += 1

    # Stable display numbers ordered by first appearance
    first_seen: dict[str, int] = {}
    for fi in sorted(lp.dets):
        for tid, _, _ in lp.dets[fi]:
            root = uf.find(f"T{tid}")
            first_seen.setdefault(root, fi)
    root_to_gid = {r: i + 1 for i, r in enumerate(sorted(first_seen, key=first_seen.get))}

    tid_to_gid = {tid: root_to_gid[uf.find(f"T{tid}")] for tid in all_tids}
    gid_role: dict[int, str] = {}
    for root, gid in root_to_gid.items():
        rvotes = Counter(roles.get((b, aid), "unknown") for b, aid in uf.astra.get(root, {}).items())
        rvotes.pop("unknown", None)
        gid_role[gid] = rvotes.most_common(1)[0][0] if rvotes else ""

    stats = {
        "local_tracks": len(all_tids),
        "global_ids": len(root_to_gid),
        "astra_verified": sum(1 for r in root_to_gid if uf.astra.get(r)),
        "matched_boxes": matched,
        "local_boxes_on_samples": total_local,
        "rejected_merges": rejected,
    }
    return tid_to_gid, gid_role, stats


# ---------------------------------------------------------------------------
# Pass 3: render
# ---------------------------------------------------------------------------

ROLE_BGR = {"staff": (219, 152, 52), "customer": (113, 204, 46)}


def id_color(gid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(gid * 9973)
    c = rng.integers(64, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


EVENT_LABEL = {
    "enter": "entered",
    "exit": "left frame",
    "leave_kitchen": "left kitchen",
    "seated": "seated",
    "service_start": "service start",
    "service_end": "service end",
}


def _event_text(e: dict[str, Any]) -> str:
    who = f"ID {e['staff']} -> ID {e['customer']}" if "staff" in e else f"ID {e['person']}"
    return f"{e['t']:5.1f}s  {EVENT_LABEL.get(e['type'], e['type'])}: {who}"


def render(
    source: Path,
    dest: Path,
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
    events: dict[str, Any] | None = None,
    trail_len: int = 30,
    event_hold_s: float = 3.0,
) -> None:
    cap = cv2.VideoCapture(str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), lp.fps, (lp.width, lp.height)
    )
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen: set[int] = set()
    sampled = set(lp.sample_idx) if mode == "hybrid" else set()
    max_jump = 0.08 * max(lp.width, lp.height)
    ev_list = (events or {}).get("events", [])
    waiting = [s for s in (events or {}).get("states", []) if s["type"] == "waiting"]
    roles = {**{str(k): v for k, v in gid_role.items()}, **(events or {}).get("roles", {})}

    for idx in range(lp.total):
        ok, frame = cap.read()
        if not ok:
            break
        now = idx / lp.fps
        dets = lp.dets.get(idx, [])
        drawn: set[int] = set()
        for tid, box, score in dets:
            gid = tid_to_gid.get(tid, tid)
            if gid in drawn:  # two local fragments mapped to same person
                continue
            drawn.add(gid)
            seen.add(gid)
            x1, y1, x2, y2 = map(int, box)
            role = roles.get(str(gid), "")
            color = ROLE_BGR.get(role, id_color(gid))
            if any(s["person"] == str(gid) and s["start"] <= now <= s["end"] for s in waiting):
                role = f"{role} waiting".strip()

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            prev = trails[gid][-1] if trails[gid] else None
            if prev and np.hypot(center[0] - prev[0], center[1] - prev[1]) > max_jump:
                trails[gid] = []  # ReID merge after a gap — don't draw a teleport line
            trails[gid].append(center)
            trails[gid] = trails[gid][-trail_len:]
            pts = trails[gid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {gid}" + (f" {role}" if role else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
            )

        hud = f"FlowMetric [{mode}]  frame {idx + 1}/{lp.total}  people {len(drawn)}  ids {len(seen)}"
        if idx in sampled:
            hud += "  * astra"
        cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        recent = [e for e in ev_list if now - event_hold_s <= e["t"] <= now][-6:]
        for k, e in enumerate(reversed(recent)):
            y = lp.height - 20 - 34 * k
            text = _event_text(e)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()


def build_tracks(
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
) -> dict[str, Any]:
    """Per-person trajectories keyed by final ID (input for events + benchmark)."""
    per_frame: dict[int, dict[int, tuple[int, np.ndarray, float]]] = defaultdict(dict)
    for fi, dets in lp.dets.items():
        for tid, box, score in dets:
            gid = tid_to_gid.get(tid, tid)
            prev = per_frame[gid].get(fi)
            if prev is None or score > prev[2]:
                per_frame[gid][fi] = (tid, box, score)

    tracks: dict[str, Any] = {}
    for gid, frames in per_frame.items():
        fis = sorted(frames)
        tracks[str(gid)] = {
            "role": gid_role.get(gid, ""),
            "tids": sorted({frames[f][0] for f in fis}),
            "frames": fis,
            "boxes": [[round(float(v), 1) for v in frames[f][1]] for f in fis],
            "conf": [round(frames[f][2], 3) for f in fis],
        }
    return {
        "mode": mode,
        "fps": lp.fps,
        "width": lp.width,
        "height": lp.height,
        "n_frames": lp.total,
        "tracks": tracks,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_hybrid(
    source: Path,
    dest: Path,
    *,
    model_name: str = "yolo11s.pt",
    conf: float = 0.2,
    iou: float = 0.5,
    imgsz: int = 1280,
    tracker: str = "botsort.yaml",
    camera: Path | None = None,
    gt: Path | None = None,
    use_astra: bool = True,
    astra_model: str | None = None,
    astra_effort: str = "low",
    sample_fps: float = 1.0,
    batch_size: int = 16,
    workers: int = 4,
    match_iou: float = 0.3,
    astra_cache: Path | None = None,
) -> dict[str, Any]:
    from events import extract_events, load_camera

    print(f"[1/4] local detect+track: {model_name} @ {imgsz} + {tracker}")
    lp = local_track(
        source, model_name=model_name, conf=conf, iou=iou, imgsz=imgsz,
        tracker=tracker, sample_fps=sample_fps if use_astra else 0,
    )

    tid_to_gid: dict[int, int] = {}
    gid_role: dict[int, str] = {}
    sync_stats: dict[str, Any] = {}
    mode = "local"

    if use_astra:
        if astra_cache and astra_cache.exists():
            print(f"[2/4] astra: loading cache {astra_cache}")
            reids = json.loads(astra_cache.read_text(encoding="utf-8"))
        else:
            print("[2/4] astra ReID on sampled frames")
            reids = astra_pass(
                lp, model=astra_model, effort=astra_effort,
                batch_size=batch_size, workers=workers,
            )
            if astra_cache:
                astra_cache.parent.mkdir(parents=True, exist_ok=True)
                astra_cache.write_text(json.dumps(reids, ensure_ascii=False, indent=2), encoding="utf-8")
        tid_to_gid, gid_role, sync_stats = sync_ids(lp, reids, iou_thr=match_iou)
        mode = "hybrid"
        print(f"      sync: {sync_stats}")
    else:
        print("[2/4] astra: skipped (--no-astra)")

    out_dir = dest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = build_tracks(lp, tid_to_gid, gid_role, mode=mode)
    ev = extract_events(tracks, load_camera(camera))
    (out_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/4] events: {len(ev['events'])} events, {len(ev['states'])} states → {out_dir / 'events.json'}")
    for e in ev["events"]:
        print(f"      {_event_text(e)}")

    print(f"[4/4] render → {dest}")
    render(source, dest, lp, tid_to_gid, gid_role, mode=mode, events=ev)

    bench = None
    if gt is not None and gt.exists():
        from benchmark import print_report, run_benchmark

        bench = run_benchmark(tracks, ev, json.loads(gt.read_text(encoding="utf-8")))
        (out_dir / "benchmark_report.json").write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBenchmark vs {gt.name} [{mode}]")
        print_report(bench)

    local_ids = {tid for d in lp.dets.values() for tid, _, _ in d}
    return {
        "benchmark": bench,
        "events": ev,
        "mode": mode,
        "frames": lp.total,
        "fps": lp.fps,
        "size": (lp.width, lp.height),
        "local_tracks": len(local_ids),
        "final_ids": len(set(tid_to_gid.values())) if tid_to_gid else len(local_ids),
        "astra_samples": len(lp.sample_idx) if use_astra else 0,
        "sync": sync_stats,
        "output": dest,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=next((ROOT / "data").glob("*.mp4"), None))
    p.add_argument("--out", type=Path, default=ROOT / "data" / "output" / "tracked.mp4")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1280, help="detector input size")
    p.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--camera", type=Path, default=ROOT / "config" / "camera.json", help="zones config")
    p.add_argument(
        "--gt", type=Path, default=ROOT / "benchmark" / "ground_truth.json",
        help="ground truth for the benchmark (skipped if the file is missing)",
    )
    p.add_argument("--no-astra", action="store_true", help="local-only tracking")
    p.add_argument(
        "--astra-model", default=None,
        help="default: openai/gpt-6-astra via VseLLM, gpt-6-astra via OpenAI",
    )
    p.add_argument("--astra-effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--sample-fps", type=float, default=1.0, help="frames/sec sent to Astra")
    p.add_argument("--batch-size", type=int, default=16, help="frames per Astra call")
    p.add_argument("--workers", type=int, default=4, help="parallel Astra calls")
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument(
        "--astra-cache", type=Path, default=ROOT / "data" / "output" / "astra_reid.json",
        help="reuse Astra responses (delete file to re-query)",
    )
    args = p.parse_args()

    if args.source is None or not args.source.exists():
        raise SystemExit("No input video. Pass --source path/to/video.mp4")

    from astra import has_api_key

    cache_ready = args.astra_cache is not None and args.astra_cache.exists()
    if not args.no_astra and not cache_ready and not has_api_key():
        raise SystemExit("Set VSELLM_API_KEY (or OPENAI_API_KEY), or pass --no-astra")

    stats = run_hybrid(
        args.source, args.out,
        model_name=args.model, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        tracker=args.tracker, camera=args.camera, gt=args.gt,
        use_astra=not args.no_astra, astra_model=args.astra_model,
        astra_effort=args.astra_effort, sample_fps=args.sample_fps,
        batch_size=args.batch_size, workers=args.workers,
        match_iou=args.match_iou, astra_cache=args.astra_cache,
    )
    print(
        f"done [{stats['mode']}]: {stats['frames']} frames, "
        f"local tracks {stats['local_tracks']} → final IDs {stats['final_ids']}, "
        f"astra samples {stats['astra_samples']} → {stats['output']}"
    )


if __name__ == "__main__":
    main()
