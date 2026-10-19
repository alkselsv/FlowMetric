"""Conservative motion-based proposals for stitching local tracklets.

The tracker is the best source of evidence for very short identity gaps on a
static camera.  This module only proposes mutually unambiguous endpoint links;
the constrained identity memory still has final authority to reject a link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from flowmetric.tracking import LocalPass


@dataclass(frozen=True)
class TrackletStitch:
    earlier_id: int
    later_id: int
    gap_s: float
    predicted_distance: float
    height_ratio: float
    score: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "earlier_tracklet_id": self.earlier_id,
            "later_tracklet_id": self.later_id,
            "gap_s": round(self.gap_s, 4),
            "predicted_distance": round(self.predicted_distance, 4),
            "height_ratio": round(self.height_ratio, 4),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class _EndpointTrack:
    tracklet_id: int
    frames: np.ndarray
    centers: np.ndarray
    heights: np.ndarray

    @property
    def first_frame(self) -> int:
        return int(self.frames[0])

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1])


def _endpoint_tracks(local_pass: LocalPass) -> list[_EndpointTrack]:
    observations: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for frame, detections in local_pass.dets.items():
        best: dict[int, tuple[np.ndarray, float]] = {}
        for tracklet_id, box, confidence in detections:
            previous = best.get(int(tracklet_id))
            if previous is None or confidence > previous[1]:
                best[int(tracklet_id)] = (np.asarray(box, dtype=float), confidence)
        for tracklet_id, (box, confidence) in best.items():
            observations.setdefault(tracklet_id, []).append((int(frame), box, confidence))

    tracks: list[_EndpointTrack] = []
    for tracklet_id, items in observations.items():
        items.sort(key=lambda item: item[0])
        frames = np.asarray([item[0] for item in items], dtype=int)
        boxes = np.stack([item[1] for item in items])
        tracks.append(
            _EndpointTrack(
                tracklet_id=tracklet_id,
                frames=frames,
                centers=np.c_[
                    (boxes[:, 0] + boxes[:, 2]) / 2,
                    (boxes[:, 1] + boxes[:, 3]) / 2,
                ],
                heights=np.maximum(boxes[:, 3] - boxes[:, 1], 1.0),
            )
        )
    return tracks


def _end_velocity(track: _EndpointTrack, fps: float) -> np.ndarray:
    """Robust recent velocity in pixels/second."""
    if len(track.frames) < 2:
        return np.zeros(2, dtype=float)
    start = max(0, len(track.frames) - 6)
    frame_deltas = np.diff(track.frames[start:]).astype(float)
    valid = frame_deltas > 0
    if not np.any(valid):
        return np.zeros(2, dtype=float)
    velocities = (
        np.diff(track.centers[start:], axis=0)[valid]
        / frame_deltas[valid, None]
        * fps
    )
    return np.median(velocities, axis=0)


def _candidate(
    earlier: _EndpointTrack,
    later: _EndpointTrack,
    *,
    fps: float,
    max_gap_s: float,
    max_distance: float,
    max_height_ratio: float,
) -> TrackletStitch | None:
    gap_frames = later.first_frame - earlier.last_frame
    if gap_frames <= 0:
        return None
    gap_s = gap_frames / fps
    if gap_s > max_gap_s:
        return None

    earlier_height = float(np.median(earlier.heights[-min(5, len(earlier.heights)) :]))
    later_height = float(np.median(later.heights[: min(5, len(later.heights))]))
    height_ratio = max(earlier_height, later_height) / max(
        min(earlier_height, later_height), 1.0
    )
    if height_ratio > max_height_ratio:
        return None

    velocity = _end_velocity(earlier, fps)
    # Do not let one jittery endpoint extrapolate farther than one body height
    # during this deliberately short gap.
    speed = float(np.linalg.norm(velocity))
    max_speed = earlier_height / max(gap_s, 1.0 / fps)
    if speed > max_speed:
        velocity *= max_speed / speed
    predicted = earlier.centers[-1] + velocity * gap_s
    scale = max((earlier_height + later_height) / 2, 1.0)
    distance = float(np.linalg.norm(predicted - later.centers[0]) / scale)
    if distance > max_distance:
        return None

    score = distance + 0.35 * (height_ratio - 1.0) + 0.1 * (gap_s / max_gap_s)
    return TrackletStitch(
        earlier_id=earlier.tracklet_id,
        later_id=later.tracklet_id,
        gap_s=gap_s,
        predicted_distance=distance,
        height_ratio=height_ratio,
        score=score,
    )


def propose_tracklet_stitches(
    local_pass: LocalPass,
    *,
    max_gap_s: float = 0.75,
    max_distance: float = 0.35,
    max_height_ratio: float = 1.25,
    ambiguity_margin: float = 0.12,
) -> list[TrackletStitch]:
    """Return high-precision, mutual-best links across short tracking gaps."""
    for label, value in {
        "max_gap_s": max_gap_s,
        "max_distance": max_distance,
        "max_height_ratio": max_height_ratio,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")

    tracks = _endpoint_tracks(local_pass)
    candidates = [
        candidate
        for earlier in tracks
        for later in tracks
        if (
            candidate := _candidate(
                earlier,
                later,
                fps=local_pass.fps,
                max_gap_s=max_gap_s,
                max_distance=max_distance,
                max_height_ratio=max_height_ratio,
            )
        )
        is not None
    ]
    by_earlier: dict[int, list[TrackletStitch]] = {}
    by_later: dict[int, list[TrackletStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: TrackletStitch, choices: Iterable[TrackletStitch]) -> bool:
        ranked = list(choices)
        return ranked[0] == candidate and (
            len(ranked) == 1 or ranked[1].score - candidate.score >= ambiguity_margin
        )

    return sorted(
        (
            candidate
            for candidate in candidates
            if unambiguous(candidate, by_earlier[candidate.earlier_id])
            and unambiguous(candidate, by_later[candidate.later_id])
        ),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )
