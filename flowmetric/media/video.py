"""Provider-neutral helpers for validated trajectories and video crops."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def xyxy_box(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain four xyxy coordinates")
    box = tuple(finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"{label} must be strictly ordered xyxy coordinates")
    return box


def track_box_map(
    tracks: dict[str, Any], track_id: str
) -> dict[int, tuple[float, float, float, float]]:
    collection = tracks.get("tracks")
    if not isinstance(collection, dict):
        raise ValueError("tracks must contain a 'tracks' object")
    track = collection.get(track_id)
    if not isinstance(track, dict):
        raise ValueError(f"tracks has no trajectory for id {track_id!r}")
    frames, boxes = track.get("frames"), track.get("boxes")
    if not isinstance(frames, list) or not isinstance(boxes, list):
        raise ValueError(f"track {track_id!r} frames and boxes must be arrays")
    if len(frames) != len(boxes):
        raise ValueError(f"track {track_id!r} frames and boxes have different lengths")
    result: dict[int, tuple[float, float, float, float]] = {}
    for index, (frame, raw_box) in enumerate(zip(frames, boxes)):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(
                f"track {track_id!r} frame {index} must be non-negative integer"
            )
        if frame in result:
            raise ValueError(f"track {track_id!r} repeats frame {frame}")
        result[frame] = xyxy_box(raw_box, f"track {track_id!r} box {index}")
    return result


def sample_evenly(values: Sequence[int], count: int) -> tuple[int, ...]:
    ordered = tuple(sorted(set(values)))
    if len(ordered) <= count:
        return ordered
    last = len(ordered) - 1
    return tuple(ordered[(position * last) // (count - 1)] for position in range(count))


def service_crop_bounds(
    frame_shape: Sequence[int],
    *,
    staff_boxes: Sequence[Sequence[float]],
    customer_boxes: Sequence[Sequence[float]],
    table_box: Sequence[float] | None,
    context_margin: float = 0.15,
) -> tuple[int, int, int, int]:
    """Return a fixed ROI enclosing an interaction across sampled frames."""
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = frame_shape[:2]
    if (
        isinstance(height, bool)
        or not isinstance(height, (int, np.integer))
        or int(height) <= 0
        or isinstance(width, bool)
        or not isinstance(width, (int, np.integer))
        or int(width) <= 0
    ):
        raise ValueError("frame_shape must have positive integer dimensions")
    height, width = int(height), int(width)
    margin = finite_number(context_margin, "context_margin")
    if margin < 0:
        raise ValueError("context_margin must be non-negative")
    if len(staff_boxes) == 0 or len(customer_boxes) == 0:
        raise ValueError("staff_boxes and customer_boxes must be non-empty")
    staff = [xyxy_box(box, f"staff_boxes[{index}]") for index, box in enumerate(staff_boxes)]
    customer = [
        xyxy_box(box, f"customer_boxes[{index}]")
        for index, box in enumerate(customer_boxes)
    ]
    table = xyxy_box(table_box, "table_box") if table_box is not None else None
    boxes = staff + customer + ([table] if table is not None else [])
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    dx = max(x2 - x1, 1.0) * margin
    dy = max(y2 - y1, 1.0) * margin
    left = max(0, int(math.floor(x1 - dx)))
    top = max(0, int(math.floor(y1 - dy)))
    right = min(width, int(math.ceil(x2 + dx)))
    bottom = min(height, int(math.ceil(y2 + dy)))
    if left >= right or top >= bottom:
        raise ValueError("candidate geometry does not intersect the video frame")
    return left, top, right, bottom


def load_video_frames(source: Path, frame_indexes: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Decode requested source frames once through a single capture handle."""
    requested = tuple(sorted(set(frame_indexes)))
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in requested
    ):
        raise ValueError("frame indexes must be non-negative integers")
    if not requested:
        return {}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open source video: {source}")
    frames: dict[int, np.ndarray] = {}
    try:
        next_position = 0
        for index in requested:
            if next_position != index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"cannot seek source video to frame {index}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode source video frame {index}")
            frames[index] = frame
            next_position = index + 1
    finally:
        capture.release()
    return frames
