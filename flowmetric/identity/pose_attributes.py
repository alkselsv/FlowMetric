"""Pose-guided clothing and body attributes for conservative ReID consensus."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from flowmetric.media.video import load_video_frames
from flowmetric.service.local import PoseDetection, UltralyticsPoseBackend
from flowmetric.tracking import LocalPass

PoseBackend = Callable[[list[np.ndarray]], list[list[PoseDetection]]]


def _valid_point(
    keypoints: np.ndarray, confidence: np.ndarray, index: int, threshold: float = 0.25
) -> np.ndarray | None:
    if index >= len(keypoints) or index >= len(confidence):
        return None
    point = np.asarray(keypoints[index], dtype=float)
    if confidence[index] < threshold or not np.all(np.isfinite(point)):
        return None
    return point


def _region_histogram(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    histogram = histogram.reshape(-1).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1e-12)
    histogram = np.sqrt(histogram)
    return histogram / max(float(np.linalg.norm(histogram)), 1e-12)


def pose_guided_attribute_descriptor(
    frame: np.ndarray,
    box: np.ndarray,
    keypoints: np.ndarray,
    keypoint_confidence: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return clothing histograms and pose-normalized body proportions."""
    box = np.asarray(box, dtype=float)
    keypoints = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(keypoint_confidence, dtype=float)
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(np.floor(box[0]))))
    top = max(0, min(height - 1, int(np.floor(box[1]))))
    right = max(left + 1, min(width, int(np.ceil(box[2]))))
    bottom = max(top + 1, min(height, int(np.ceil(box[3]))))
    box_width = max(float(right - left), 1.0)
    box_height = max(float(bottom - top), 1.0)

    shoulders = [_valid_point(keypoints, confidence, index) for index in (5, 6)]
    hips = [_valid_point(keypoints, confidence, index) for index in (11, 12)]
    knees = [_valid_point(keypoints, confidence, index) for index in (13, 14)]
    ankles = [_valid_point(keypoints, confidence, index) for index in (15, 16)]
    pose_points = [point for point in (*shoulders, *hips, *knees, *ankles) if point is not None]
    pose_quality = float(
        np.mean([confidence[index] for index in (5, 6, 11, 12, 13, 14, 15, 16)])
    ) if len(confidence) > 16 else 0.0

    shoulder_y = (
        float(np.mean([point[1] for point in shoulders if point is not None]))
        if all(point is not None for point in shoulders)
        else top + 0.20 * box_height
    )
    hip_y = (
        float(np.mean([point[1] for point in hips if point is not None]))
        if all(point is not None for point in hips)
        else top + 0.55 * box_height
    )
    ankle_y = (
        float(np.mean([point[1] for point in ankles if point is not None]))
        if any(point is not None for point in ankles)
        else top + 0.95 * box_height
    )
    inner_left = int(round(left + 0.12 * box_width))
    inner_right = int(round(right - 0.12 * box_width))
    torso = (
        inner_left,
        max(top, int(round(shoulder_y))),
        max(inner_left + 1, inner_right),
        min(bottom, max(int(round(shoulder_y)) + 1, int(round(hip_y)))),
    )
    lower = (
        inner_left,
        max(top, int(round(hip_y))),
        max(inner_left + 1, inner_right),
        min(bottom, max(int(round(hip_y)) + 1, int(round(ankle_y)))),
    )
    torso_hist = _region_histogram(frame, torso)
    lower_hist = _region_histogram(frame, lower)

    def pair_width(points: list[np.ndarray | None]) -> float:
        if any(point is None for point in points):
            return 0.0
        return abs(float(points[0][0] - points[1][0])) / box_width  # type: ignore[index]

    shoulder_width = pair_width(shoulders)
    hip_width = pair_width(hips)
    torso_length = max(0.0, hip_y - shoulder_y) / box_height
    leg_length = max(0.0, ankle_y - hip_y) / box_height
    proportions = np.asarray(
        [shoulder_width, hip_width, torso_length, leg_length], dtype=np.float32
    )
    proportions /= max(float(np.linalg.norm(proportions)), 1e-12)
    descriptor = np.concatenate(
        [
            np.sqrt(0.45) * torso_hist,
            np.sqrt(0.40) * lower_hist,
            np.sqrt(0.15) * proportions,
        ]
    )
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    coverage = len(pose_points) / 8.0
    quality = float(np.clip(pose_quality * (0.5 + 0.5 * coverage), 0.0, 1.0))
    return descriptor.astype(np.float32), quality


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(float((left[2] - left[0]) * (left[3] - left[1])), 0.0)
    right_area = max(float((right[2] - right[0]) * (right[3] - right[1])), 0.0)
    return intersection / max(left_area + right_area - intersection, 1e-12)


def encode_pose_attribute_tracklets(
    source: Path,
    local_pass: LocalPass,
    *,
    model: str | Path = "yolo11n-pose.pt",
    device: str = "cpu",
    min_iou: float = 0.40,
    min_pose_quality: float = 0.20,
    backend: PoseBackend | None = None,
) -> dict[str, Any]:
    """Encode pose-guided attributes on the existing OSNet observation frames."""
    if not 0 <= min_iou <= 1 or not 0 <= min_pose_quality <= 1:
        raise ValueError("min_iou and min_pose_quality must be in [0, 1]")
    sample_quality = {
        (int(tracklet_id), int(frame)): float(quality)
        for tracklet_id, samples in local_pass.appearance_embeddings.items()
        for frame, _, quality in samples
    }
    observations: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame, detections in local_pass.dets.items():
        selected = [
            (int(tracklet_id), np.asarray(box, dtype=float))
            for tracklet_id, box, _ in detections
            if (int(tracklet_id), int(frame)) in sample_quality
        ]
        if selected:
            observations[int(frame)] = selected
    frame_indexes = tuple(sorted(observations))
    frames = load_video_frames(Path(source), frame_indexes)
    pose_backend = backend or UltralyticsPoseBackend(model, device=device, imgsz=640)
    pose_by_frame: dict[int, list[PoseDetection]] = {}
    for offset in range(0, len(frame_indexes), 8):
        batch_indexes = frame_indexes[offset : offset + 8]
        results = pose_backend([frames[index] for index in batch_indexes])
        if len(results) != len(batch_indexes):
            raise RuntimeError("pose backend returned a malformed batch")
        pose_by_frame.update(zip(batch_indexes, results))

    local_pass.pose_attribute_embeddings = {}
    requested = matched = low_quality = 0
    for frame_index in frame_indexes:
        local = observations[frame_index]
        poses = pose_by_frame.get(frame_index, [])
        requested += len(local)
        if not poses:
            continue
        overlaps = np.asarray(
            [
                [_iou(box, np.asarray(pose.box, dtype=float)) for pose in poses]
                for _, box in local
            ],
            dtype=float,
        )
        rows, columns = linear_sum_assignment(-overlaps)
        for row, column in zip(rows, columns):
            if overlaps[row, column] < min_iou:
                continue
            tracklet_id, box = local[row]
            pose = poses[column]
            descriptor, pose_quality = pose_guided_attribute_descriptor(
                frames[frame_index], box, pose.keypoints, pose.keypoint_confidence
            )
            if pose_quality < min_pose_quality:
                low_quality += 1
                continue
            quality = sample_quality[(tracklet_id, frame_index)] * pose_quality
            local_pass.pose_attribute_embeddings.setdefault(tracklet_id, []).append(
                (frame_index, descriptor, quality)
            )
            matched += 1
    return {
        "model": Path(model).name,
        "frames": len(frame_indexes),
        "requested_observations": requested,
        "matched_observations": matched,
        "low_quality_observations": low_quality,
        "tracklets": len(local_pass.pose_attribute_embeddings),
    }
