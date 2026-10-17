#!/usr/bin/env python3
"""Static table map for a fixed camera: YOLO 'dining table' boxes aggregated over sampled frames.

People occlude tables, so single-frame detections flicker; boxes are clustered across frames
and a table is kept if it is seen in enough samples (median box per cluster).
Output: config/tables.json, used by events.py for table-anchored seating/service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from track import ROOT, iou_matrix, weights_path

COCO_DINING_TABLE = 60
DEFAULT_OUT = ROOT / "config" / "tables.json"


def detect_tables(
    video: Path,
    *,
    model_name: str = "yolo11s.pt",
    imgsz: int = 1280,
    conf: float = 0.1,  # tables are mostly occluded by diners; support across frames filters noise
    step: int = 12,
    min_support: float = 0.3,
    match_iou: float = 0.4,
) -> list[dict]:
    if step <= 0:
        raise ValueError("step must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    from ultralytics import YOLO

    model = YOLO(weights_path(model_name))
    clusters: list[list[np.ndarray]] = []
    n_samples, i = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[COCO_DINING_TABLE], verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            n_samples += 1
            if clusters and len(boxes):
                ref = np.stack([np.median(c, axis=0) for c in clusters])
                m = iou_matrix(boxes, ref)
                rows, cols = linear_sum_assignment(-m)
                used = set()
                for r_, c_ in zip(rows, cols):
                    if m[r_, c_] >= match_iou:
                        clusters[c_].append(boxes[r_])
                        used.add(r_)
                clusters += [[b] for j, b in enumerate(boxes) if j not in used]
            else:
                clusters += [[b] for b in boxes]
        i += 1
    cap.release()
    if n_samples == 0:
        raise RuntimeError(f"No frames decoded from {video}")

    tables = [
        {"box": np.median(c, axis=0).round(1).tolist(), "support": round(len(c) / n_samples, 2)}
        for c in clusters
        if len(c) / n_samples >= min_support
    ]
    # one physical table split into clusters (e.g. top only vs top+legs) → keep the better-supported one
    tables.sort(key=lambda t: -t["support"])
    kept: list[dict] = []
    for t in tables:
        if not any(_containment(t["box"], k["box"]) > 0.7 for k in kept):
            kept.append(t)
    return kept


def _containment(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller box."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / small if small > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=next((ROOT / "data").glob("*.mp4"), None))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preview", type=Path, default=ROOT / "data" / "output" / "tables.jpg")
    args = p.parse_args()

    if args.video is None or not args.video.exists():
        raise SystemExit("No input video. Pass --video path/to/video.mp4")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    resolution = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()

    tables = detect_tables(args.video)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"source": args.video.name, "resolution": resolution, "tables": tables}, indent=2),
        encoding="utf-8",
    )
    print(f"{len(tables)} tables → {args.out}")

    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if ok:
        for k, t in enumerate(tables):
            x1, y1, x2, y2 = map(int, t["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(frame, f"T{k} {t['support']:.0%}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 165, 255), 2, cv2.LINE_AA)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), frame)
        print(f"preview → {args.preview}")


if __name__ == "__main__":
    main()
