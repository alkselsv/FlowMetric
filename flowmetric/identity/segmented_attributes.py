"""Foreground-masked clothing attributes for conservative ReID consensus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowmetric.media.video import load_video_frames
from flowmetric.tracking import LocalPass


def _masked_histogram(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.size == 0 or not np.any(mask):
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], mask.astype(np.uint8), [12, 4], [0, 180, 0, 256]
    ).reshape(-1).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1e-12)
    histogram = np.sqrt(histogram)
    return histogram / max(float(np.linalg.norm(histogram)), 1e-12)


def masked_clothing_descriptor(
    crop: np.ndarray, foreground: np.ndarray
) -> np.ndarray:
    """Describe upper/lower clothing after removing the local background."""
    foreground = np.asarray(foreground, dtype=bool)
    if crop.ndim != 3 or crop.shape[:2] != foreground.shape:
        raise ValueError("foreground must match the crop height and width")
    height = crop.shape[0]
    upper_end = max(1, int(round(0.58 * height)))
    lower_start = min(height - 1, int(round(0.45 * height)))
    upper = _masked_histogram(crop[:upper_end], foreground[:upper_end])
    lower = _masked_histogram(crop[lower_start:], foreground[lower_start:])
    descriptor = np.concatenate([np.sqrt(0.55) * upper, np.sqrt(0.45) * lower])
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)


def segment_person_attributes(
    frame: np.ndarray,
    box: np.ndarray,
    *,
    iterations: int = 2,
) -> tuple[np.ndarray, float] | None:
    """Segment one person with GrabCut and return clothing attributes and quality."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    height, width = frame.shape[:2]
    box = np.asarray(box, dtype=float)
    box_width = max(float(box[2] - box[0]), 1.0)
    box_height = max(float(box[3] - box[1]), 1.0)
    pad_x, pad_y = 0.10 * box_width, 0.06 * box_height
    x1 = max(0, int(np.floor(box[0] - pad_x)))
    y1 = max(0, int(np.floor(box[1] - pad_y)))
    x2 = min(width, int(np.ceil(box[2] + pad_x)))
    y2 = min(height, int(np.ceil(box[3] + pad_y)))
    if x2 - x1 < 12 or y2 - y1 < 20:
        return None
    crop = frame[y1:y2, x1:x2]
    rectangle = (
        max(1, int(np.floor(box[0])) - x1),
        max(1, int(np.floor(box[1])) - y1),
        min(crop.shape[1] - 2, int(np.ceil(box[2])) - x1),
        min(crop.shape[0] - 2, int(np.ceil(box[3])) - y1),
    )
    rect_x, rect_y, rect_right, rect_bottom = rectangle
    rect_width, rect_height = rect_right - rect_x, rect_bottom - rect_y
    if rect_width < 3 or rect_height < 3:
        return None
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            crop,
            mask,
            (rect_x, rect_y, rect_width, rect_height),
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None
    foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    kernel = np.ones((3, 3), dtype=np.uint8)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    person_area = max(rect_width * rect_height, 1)
    fraction = float(foreground.sum()) / person_area
    if not 0.20 <= fraction <= 0.95:
        return None
    border = np.concatenate(
        [foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]]
    )
    border_leak = float(np.mean(border))
    if border_leak > 0.15:
        return None
    quality = float(np.clip((1.0 - border_leak) * min(fraction / 0.45, 1.0), 0, 1))
    return masked_clothing_descriptor(crop, foreground), quality


def encode_segmented_attribute_tracklets(
    source: Path,
    local_pass: LocalPass,
    *,
    max_samples_per_tracklet: int = 4,
) -> dict[str, Any]:
    """Build foreground-masked attributes on the strongest OSNet observations."""
    if (
        isinstance(max_samples_per_tracklet, bool)
        or not isinstance(max_samples_per_tracklet, int)
        or max_samples_per_tracklet < 2
    ):
        raise ValueError("max_samples_per_tracklet must be an integer of at least 2")
    selected_quality: dict[tuple[int, int], float] = {}
    for tracklet_id, samples in local_pass.appearance_embeddings.items():
        strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[
            :max_samples_per_tracklet
        ]
        for frame, _, quality in strongest:
            selected_quality[(int(tracklet_id), int(frame))] = float(quality)
    observations: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame, detections in local_pass.dets.items():
        selected = [
            (int(tracklet_id), np.asarray(box, dtype=float))
            for tracklet_id, box, _ in detections
            if (int(tracklet_id), int(frame)) in selected_quality
        ]
        if selected:
            observations[int(frame)] = selected
    frame_indexes = tuple(sorted(observations))
    frames = load_video_frames(Path(source), frame_indexes)
    local_pass.segmented_attribute_embeddings = {}
    requested = accepted = rejected = 0
    for frame_index in frame_indexes:
        for tracklet_id, box in observations[frame_index]:
            requested += 1
            result = segment_person_attributes(frames[frame_index], box)
            if result is None:
                rejected += 1
                continue
            descriptor, mask_quality = result
            quality = selected_quality[(tracklet_id, frame_index)] * mask_quality
            local_pass.segmented_attribute_embeddings.setdefault(tracklet_id, []).append(
                (frame_index, descriptor, quality)
            )
            accepted += 1
    return {
        "method": "grabcut_person_box_v1",
        "frames": len(frame_indexes),
        "requested_observations": requested,
        "accepted_observations": accepted,
        "rejected_masks": rejected,
        "tracklets": len(local_pass.segmented_attribute_embeddings),
    }
