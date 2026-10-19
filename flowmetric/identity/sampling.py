"""Detection-aware frame selection for the Astra ReID gallery."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from flowmetric.tracking import LocalPass


def select_adaptive_reid_frames(
    local_pass: LocalPass,
    *,
    count: int,
    confirmations: int = 3,
) -> list[int]:
    """Choose a fixed-size, temporally spread set that covers rare tracklets."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if (
        isinstance(confirmations, bool)
        or not isinstance(confirmations, int)
        or confirmations <= 0
    ):
        raise ValueError("confirmations must be a positive integer")
    candidates = sorted(frame for frame, dets in local_pass.dets.items() if dets)
    if len(candidates) <= count:
        return candidates

    frames_by_tid: dict[int, set[int]] = defaultdict(set)
    tids_by_frame: dict[int, set[int]] = {}
    for frame in candidates:
        tids = {int(tid) for tid, _, _ in local_pass.dets[frame]}
        tids_by_frame[frame] = tids
        for tid in tids:
            frames_by_tid[tid].add(frame)

    targets = {
        tid: min(confirmations, len(frames)) for tid, frames in frames_by_tid.items()
    }
    # Rare fragments receive more weight; long tracks still earn repeated views.
    weights = {
        tid: 1.0 / max(len(frames), 1) ** 0.5
        for tid, frames in frames_by_tid.items()
    }
    selected: list[int] = []
    coverage: Counter[int] = Counter()
    occupied_bins: set[int] = set()

    while len(selected) < count:
        remaining = [frame for frame in candidates if frame not in selected]
        if not remaining:
            break

        def score(frame: int) -> tuple[float, int, int]:
            temporal_bin = min(count - 1, frame * count // max(local_pass.total, 1))
            gain = sum(
                weights[tid] / (1.0 + coverage[tid])
                for tid in tids_by_frame[frame]
                if coverage[tid] < targets[tid]
            )
            if temporal_bin not in occupied_bins:
                gain += 0.2
            # Prefer the centre of a temporal bin for deterministic tie-breaking.
            centre = (temporal_bin + 0.5) * local_pass.total / count
            return gain, -round(abs(frame - centre)), -frame

        chosen = max(remaining, key=score)
        selected.append(chosen)
        coverage.update(tids_by_frame[chosen])
        occupied_bins.add(
            min(count - 1, chosen * count // max(local_pass.total, 1))
        )

    return sorted(selected)


def select_targeted_resample_frames(
    local_pass: LocalPass,
    *,
    query_tracklet_ids: list[int],
    base_frames: list[int],
    additional_count: int = 16,
    confirmations: int = 3,
) -> list[int]:
    """Choose extra high-quality frames only for unresolved query tracklets."""
    if (
        isinstance(additional_count, bool)
        or not isinstance(additional_count, int)
        or additional_count < 0
    ):
        raise ValueError("additional_count must be a non-negative integer")
    if (
        isinstance(confirmations, bool)
        or not isinstance(confirmations, int)
        or confirmations <= 0
    ):
        raise ValueError("confirmations must be a positive integer")
    if any(
        isinstance(tracklet_id, bool)
        or not isinstance(tracklet_id, int)
        or tracklet_id < 0
        for tracklet_id in query_tracklet_ids
    ):
        raise ValueError("query_tracklet_ids must be non-negative integers")
    queries = set(query_tracklet_ids)
    if len(queries) != len(query_tracklet_ids):
        raise ValueError("query_tracklet_ids must be unique")
    if not queries or additional_count == 0:
        return []

    observations: dict[int, dict[int, tuple[np.ndarray, float]]] = defaultdict(dict)
    quality_by_frame_tid: dict[tuple[int, int], float] = {}
    frame_area = max(float(local_pass.width * local_pass.height), 1.0)
    for frame, detections in local_pass.dets.items():
        boxes = [np.asarray(box, dtype=float) for _, box, _ in detections]
        for index, (tracklet_id, box, confidence) in enumerate(detections):
            tracklet_id = int(tracklet_id)
            if tracklet_id not in queries:
                continue
            box = np.asarray(box, dtype=float)
            observations[tracklet_id][int(frame)] = (box, float(confidence))
            box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
            size_quality = min(np.sqrt(box_area / frame_area) / 0.2, 1.0)
            occlusion = 0.0
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                x1, y1 = np.maximum(box[:2], other[:2])
                x2, y2 = np.minimum(box[2:], other[2:])
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                occlusion = max(occlusion, intersection / max(box_area, 1.0))
            visibility = 1.0 - min(occlusion, 1.0)
            quality_by_frame_tid[(int(frame), tracklet_id)] = (
                0.5 * float(confidence)
                + 0.3 * float(size_quality)
                + 0.2 * visibility
            )

    missing = queries - set(observations)
    if missing:
        raise ValueError(f"query tracklets are absent from local detections: {sorted(missing)}")

    base = set(base_frames)
    selected: list[int] = []
    covered_frames = {
        tracklet_id: sorted(base & set(frames))
        for tracklet_id, frames in observations.items()
    }
    targets = {
        tracklet_id: min(confirmations, len(frames))
        for tracklet_id, frames in observations.items()
    }
    rarity = {
        tracklet_id: 1.0 / max(len(frames), 1) ** 0.5
        for tracklet_id, frames in observations.items()
    }
    candidate_frames = sorted(
        {
            frame
            for frames in observations.values()
            for frame in frames
            if frame not in base
        }
    )

    while len(selected) < additional_count:
        remaining = [frame for frame in candidate_frames if frame not in selected]
        if not remaining or all(
            len(covered_frames[tracklet_id]) >= target
            for tracklet_id, target in targets.items()
        ):
            break

        def score(frame: int) -> tuple[float, int]:
            gain = 0.0
            for tracklet_id, frames in observations.items():
                if frame not in frames or len(covered_frames[tracklet_id]) >= targets[tracklet_id]:
                    continue
                history = covered_frames[tracklet_id]
                span = max(max(frames) - min(frames), 1)
                diversity = (
                    min(abs(frame - previous) for previous in history) / span
                    if history
                    else 1.0
                )
                gain += rarity[tracklet_id] * (
                    1.0
                    + quality_by_frame_tid[(frame, tracklet_id)]
                    + 0.25 * min(diversity, 1.0)
                ) / (1.0 + len(history))
            return gain, -frame

        chosen = max(remaining, key=score)
        if score(chosen)[0] <= 0:
            break
        selected.append(chosen)
        for tracklet_id, frames in observations.items():
            if chosen in frames:
                covered_frames[tracklet_id].append(chosen)

    return sorted(selected)
