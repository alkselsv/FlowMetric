"""Frame-local person detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from flowmetric.paths import weights_path


@dataclass
class LocalPass:
    fps: float
    width: int
    height: int
    total: int
    dets: dict[int, list[tuple[int, np.ndarray, float]]] = field(default_factory=dict)
    sample_idx: list[int] = field(default_factory=list)
    sample_frames: list[np.ndarray] = field(default_factory=list)
    _pair_constraints: tuple[set[tuple[int, int]], set[tuple[int, int]]] | None = field(
        default=None, repr=False
    )


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
    """Run the local detector/tracker and retain optional ReID sample frames."""
    model = YOLO(weights_path(model_name))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    local_pass = LocalPass(
        fps=fps,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 0

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(
                frame,
                persist=True,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=[0],
                tracker=tracker,
                verbose=False,
            )[0]
            detections: list[tuple[int, np.ndarray, float]] = []
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id, score in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.id.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    detections.append(
                        (int(track_id), box.astype(float), float(score))
                    )
            local_pass.dets[frame_index] = detections

            if step and frame_index % step == 0:
                local_pass.sample_idx.append(frame_index)
                local_pass.sample_frames.append(frame.copy())
            frame_index += 1
            if frame_index % 50 == 0:
                print(f"  local: {frame_index}/{local_pass.total} frames")
    finally:
        capture.release()

    local_pass.total = frame_index
    return local_pass
