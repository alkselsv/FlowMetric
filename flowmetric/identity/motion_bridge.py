"""Appearance-confirmed, bidirectional motion bridges for broken tracklets.

The production local stitcher deliberately uses a very narrow geometric gate.
This experimental layer looks across slightly longer gaps, but requires motion
to agree in both directions and an independent appearance confirmation.  It is
kept in shadow mode until its proposed links can be validated on labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flowmetric.tracking import LocalPass


@dataclass(frozen=True)
class MotionBridge:
    earlier_id: int
    later_id: int
    gap_s: float
    forward_distance: float
    backward_distance: float
    height_ratio: float
    appearance_similarity: float
    appearance_support: int
    cost: float
    earlier_margin: float | None = None
    later_margin: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "earlier_tracklet_id": self.earlier_id,
            "later_tracklet_id": self.later_id,
            "gap_s": round(self.gap_s, 4),
            "forward_distance": round(self.forward_distance, 4),
            "backward_distance": round(self.backward_distance, 4),
            "height_ratio": round(self.height_ratio, 4),
            "appearance_similarity": round(self.appearance_similarity, 4),
            "appearance_support": self.appearance_support,
            "cost": round(self.cost, 4),
            "earlier_margin": (
                None if self.earlier_margin is None else round(self.earlier_margin, 4)
            ),
            "later_margin": (
                None if self.later_margin is None else round(self.later_margin, 4)
            ),
        }


@dataclass(frozen=True)
class _Track:
    tracklet_id: int
    frames: np.ndarray
    feet: np.ndarray
    heights: np.ndarray

    @property
    def first_frame(self) -> int:
        return int(self.frames[0])

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1])


def _tracks(local_pass: LocalPass) -> list[_Track]:
    observations: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for frame, detections in local_pass.dets.items():
        strongest: dict[int, tuple[np.ndarray, float]] = {}
        for tracklet_id, box, confidence in detections:
            previous = strongest.get(int(tracklet_id))
            if previous is None or confidence > previous[1]:
                strongest[int(tracklet_id)] = (np.asarray(box, dtype=float), confidence)
        for tracklet_id, (box, confidence) in strongest.items():
            observations.setdefault(tracklet_id, []).append((int(frame), box, confidence))

    result: list[_Track] = []
    for tracklet_id, items in observations.items():
        items.sort(key=lambda item: item[0])
        frames = np.asarray([item[0] for item in items], dtype=int)
        boxes = np.stack([item[1] for item in items])
        result.append(
            _Track(
                tracklet_id=tracklet_id,
                frames=frames,
                feet=np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]],
                heights=np.maximum(boxes[:, 3] - boxes[:, 1], 1.0),
            )
        )
    return result


def _velocity(track: _Track, fps: float, *, at_end: bool) -> np.ndarray:
    """Median endpoint velocity in pixels/second, resistant to box jitter."""
    if len(track.frames) < 2:
        return np.zeros(2, dtype=float)
    subset = slice(max(0, len(track.frames) - 6), None) if at_end else slice(0, 6)
    frames = track.frames[subset]
    feet = track.feet[subset]
    deltas = np.diff(frames).astype(float)
    valid = deltas > 0
    if not np.any(valid):
        return np.zeros(2, dtype=float)
    return np.median(np.diff(feet, axis=0)[valid] / deltas[valid, None] * fps, axis=0)


def _bounded_velocity(velocity: np.ndarray, height: float, gap_s: float) -> np.ndarray:
    # A noisy endpoint must not extrapolate by more than two body heights.
    result = velocity.copy()
    speed = float(np.linalg.norm(result))
    limit = 2.0 * height / max(gap_s, 1e-6)
    if speed > limit:
        result *= limit / speed
    return result


def _prototype(samples: list[tuple[int, np.ndarray, float]]) -> np.ndarray:
    strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[:8]
    matrix = np.stack([np.asarray(item[1], dtype=float) for item in strongest])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    weights = np.asarray([max(float(item[2]), 0.05) for item in strongest])
    vector = np.average(matrix, axis=0, weights=weights)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def propose_motion_bridges(
    local_pass: LocalPass,
    *,
    cluster_by_tracklet: dict[int, str] | None = None,
    blocked_pairs: set[tuple[int, int]] | None = None,
    max_gap_s: float = 3.0,
    max_motion_distance: float = 1.0,
    max_height_ratio: float = 1.4,
    min_appearance_similarity: float = 0.80,
    ambiguity_margin: float = 0.08,
) -> list[MotionBridge]:
    """Globally assign conservative motion+appearance links across short gaps."""
    if not local_pass.appearance_embeddings:
        return []
    cluster_by_tracklet = cluster_by_tracklet or {}
    blocked_pairs = {
        tuple(sorted((int(left), int(right)))) for left, right in (blocked_pairs or set())
    }
    for label, value in {
        "max_gap_s": max_gap_s,
        "max_motion_distance": max_motion_distance,
        "max_height_ratio": max_height_ratio,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if not np.isfinite(min_appearance_similarity) or not -1 <= min_appearance_similarity <= 1:
        raise ValueError("min_appearance_similarity must be in [-1, 1]")

    tracks = _tracks(local_pass)
    prototypes = {
        int(tracklet_id): _prototype(samples)
        for tracklet_id, samples in local_pass.appearance_embeddings.items()
        if samples
    }
    candidates: list[MotionBridge] = []
    for earlier in tracks:
        if earlier.tracklet_id not in prototypes:
            continue
        for later in tracks:
            if later.tracklet_id not in prototypes:
                continue
            if (
                cluster_by_tracklet.get(earlier.tracklet_id) is not None
                and cluster_by_tracklet.get(earlier.tracklet_id)
                == cluster_by_tracklet.get(later.tracklet_id)
            ):
                continue
            if tuple(sorted((earlier.tracklet_id, later.tracklet_id))) in blocked_pairs:
                continue
            gap_frames = later.first_frame - earlier.last_frame
            if gap_frames <= 0:
                continue
            gap_s = gap_frames / local_pass.fps
            if gap_s > max_gap_s:
                continue
            earlier_height = float(np.median(earlier.heights[-min(5, len(earlier.heights)) :]))
            later_height = float(np.median(later.heights[: min(5, len(later.heights))]))
            height_ratio = max(earlier_height, later_height) / max(
                min(earlier_height, later_height), 1.0
            )
            if height_ratio > max_height_ratio:
                continue
            scale = max((earlier_height + later_height) / 2, 1.0)
            forward_velocity = _bounded_velocity(
                _velocity(earlier, local_pass.fps, at_end=True), earlier_height, gap_s
            )
            backward_velocity = _bounded_velocity(
                _velocity(later, local_pass.fps, at_end=False), later_height, gap_s
            )
            forward = float(
                np.linalg.norm(earlier.feet[-1] + forward_velocity * gap_s - later.feet[0])
                / scale
            )
            backward = float(
                np.linalg.norm(later.feet[0] - backward_velocity * gap_s - earlier.feet[-1])
                / scale
            )
            if max(forward, backward) > max_motion_distance:
                continue
            similarity = float(
                np.dot(prototypes[earlier.tracklet_id], prototypes[later.tracklet_id])
            )
            if similarity < min_appearance_similarity:
                continue
            support = min(
                len(local_pass.appearance_embeddings[earlier.tracklet_id]),
                len(local_pass.appearance_embeddings[later.tracklet_id]),
            )
            if support < 2:
                continue
            motion = (forward + backward) / (2 * max_motion_distance)
            appearance = (1.0 - similarity) / max(1.0 - min_appearance_similarity, 1e-6)
            cost = (
                0.55 * motion
                + 0.25 * appearance
                + 0.10 * ((height_ratio - 1.0) / (max_height_ratio - 1.0))
                + 0.10 * (gap_s / max_gap_s)
            )
            candidates.append(
                MotionBridge(
                    earlier_id=earlier.tracklet_id,
                    later_id=later.tracklet_id,
                    gap_s=gap_s,
                    forward_distance=forward,
                    backward_distance=backward,
                    height_ratio=height_ratio,
                    appearance_similarity=similarity,
                    appearance_support=support,
                    cost=cost,
                )
            )

    if not candidates:
        return []
    earlier_ids = sorted({item.earlier_id for item in candidates})
    later_ids = sorted({item.later_id for item in candidates})
    row = {tracklet_id: index for index, tracklet_id in enumerate(earlier_ids)}
    column = {tracklet_id: index for index, tracklet_id in enumerate(later_ids)}
    # Dummy rows/columns let the assignment leave a weak candidate unmatched.
    size = len(earlier_ids) + len(later_ids)
    unmatched_cost = 1.0
    matrix = np.full((size, size), 1e6, dtype=float)
    for index in range(len(earlier_ids)):
        matrix[index, len(later_ids) + index] = unmatched_cost
    for index in range(len(later_ids)):
        matrix[len(earlier_ids) + index, index] = unmatched_cost
    matrix[len(earlier_ids) :, len(later_ids) :] = 0.0
    by_pair = {(item.earlier_id, item.later_id): item for item in candidates}
    for item in candidates:
        matrix[row[item.earlier_id], column[item.later_id]] = min(item.cost, unmatched_cost + 1.0)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is an ultralytics dependency
        raise RuntimeError("motion bridge global assignment requires scipy") from exc
    rows, columns = linear_sum_assignment(matrix)

    selected: list[MotionBridge] = []
    for row_index, column_index in zip(rows, columns):
        if row_index >= len(earlier_ids) or column_index >= len(later_ids):
            continue
        pair = (earlier_ids[row_index], later_ids[column_index])
        item = by_pair.get(pair)
        if item is None or item.cost >= unmatched_cost:
            continue
        earlier_costs = [
            candidate.cost
            for candidate in candidates
            if candidate.earlier_id == item.earlier_id and candidate != item
        ]
        later_costs = [
            candidate.cost
            for candidate in candidates
            if candidate.later_id == item.later_id and candidate != item
        ]
        earlier_margin = None if not earlier_costs else min(earlier_costs) - item.cost
        later_margin = None if not later_costs else min(later_costs) - item.cost
        if (earlier_margin is not None and earlier_margin < ambiguity_margin) or (
            later_margin is not None and later_margin < ambiguity_margin
        ):
            continue
        selected.append(
            MotionBridge(
                **{
                    **item.__dict__,
                    "earlier_margin": earlier_margin,
                    "later_margin": later_margin,
                }
            )
        )
    return sorted(selected, key=lambda item: (item.cost, item.earlier_id, item.later_id))
