"""Conservative suppression of nested, unverified person tracklets.

Low-threshold person detectors occasionally track a head, torso, or carried
object as a second person inside an existing person box.  Such tracklets must
not be merged because both detections coexist.  Instead, this module identifies
short, unsupported duplicates that stay inside one stronger track.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from flowmetric.tracking import LocalPass


@dataclass(frozen=True)
class NestedTrackletSuppression:
    tracklet_id: int
    host_global_id: int
    observations: int
    nested_observations: int
    nested_fraction: float
    mean_containment: float
    mean_area_ratio: float
    mean_confidence: float
    host_mean_confidence: float
    duration_s: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "tracklet_id": self.tracklet_id,
            "host_global_id": self.host_global_id,
            "observations": self.observations,
            "nested_observations": self.nested_observations,
            "nested_fraction": round(self.nested_fraction, 4),
            "mean_containment": round(self.mean_containment, 4),
            "mean_area_ratio": round(self.mean_area_ratio, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "host_mean_confidence": round(self.host_mean_confidence, 4),
            "duration_s": round(self.duration_s, 4),
            "decision": "suppress",
            "reason": "short_unverified_nested_duplicate",
        }


def _intersection_over_candidate(candidate: np.ndarray, host: np.ndarray) -> float:
    x1, y1 = np.maximum(candidate[:2], host[:2])
    x2, y2 = np.minimum(candidate[2:], host[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(
        float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])),
        1.0,
    )
    return float(intersection / area)


def propose_nested_tracklet_suppressions(
    local_pass: LocalPass,
    tid_to_gid: dict[int, int],
    track_identity: dict[str, dict[str, Any]],
    *,
    max_duration_s: float = 0.5,
    min_containment: float = 0.55,
    max_area_ratio: float = 0.55,
    min_nested_fraction: float = 0.5,
    max_mean_confidence: float = 0.75,
) -> list[NestedTrackletSuppression]:
    """Return short unsupported tracklets consistently nested in one host.

    The host must be longer and more confident than the candidate.  A
    candidate already accepted or stitched by ReID is never suppressed.
    """
    for label, value in {
        "max_duration_s": max_duration_s,
        "min_containment": min_containment,
        "max_area_ratio": max_area_ratio,
        "min_nested_fraction": min_nested_fraction,
        "max_mean_confidence": max_mean_confidence,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    for label, value in {
        "min_containment": min_containment,
        "max_area_ratio": max_area_ratio,
        "min_nested_fraction": min_nested_fraction,
        "max_mean_confidence": max_mean_confidence,
    }.items():
        if value > 1:
            raise ValueError(f"{label} must be at most 1")

    observations: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    by_frame: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    for frame, detections in local_pass.dets.items():
        best: dict[int, tuple[np.ndarray, float]] = {}
        for tracklet_id, box, confidence in detections:
            tracklet_id = int(tracklet_id)
            if tracklet_id not in best or confidence > best[tracklet_id][1]:
                best[tracklet_id] = (np.asarray(box, dtype=float), float(confidence))
        by_frame[int(frame)] = best
        for tracklet_id, (box, confidence) in best.items():
            observations[tracklet_id].append((int(frame), box, confidence))

    gid_members: dict[int, set[int]] = defaultdict(set)
    for tracklet_id, global_id in tid_to_gid.items():
        gid_members[int(global_id)].add(int(tracklet_id))

    def duration(tracklet_id: int) -> float:
        frames = [item[0] for item in observations[tracklet_id]]
        return (max(frames) - min(frames)) / local_pass.fps if frames else 0.0

    mean_confidence = {
        tracklet_id: float(np.mean([item[2] for item in items]))
        for tracklet_id, items in observations.items()
    }
    proposals: list[NestedTrackletSuppression] = []
    protected_statuses = {"merged", "appearance_stitched", "local_stitched"}

    for tracklet_id, items in sorted(observations.items()):
        identity = track_identity.get(str(tracklet_id), {})
        if identity.get("status") in protected_statuses:
            continue
        # A short track at either clip boundary is censored: its true duration
        # is unknown, and reference-frame anchoring may depend on it.
        if items[0][0] == 0 or items[-1][0] == local_pass.total - 1:
            continue
        global_id = tid_to_gid.get(tracklet_id)
        if global_id is None or len(gid_members[global_id]) != 1:
            continue
        candidate_duration = duration(tracklet_id)
        candidate_confidence = mean_confidence[tracklet_id]
        if (
            candidate_duration > max_duration_s
            or candidate_confidence > max_mean_confidence
        ):
            continue

        nested_by_gid: Counter[int] = Counter()
        evidence_by_gid: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for frame, candidate_box, _ in items:
            candidate_area = max(
                float(
                    (candidate_box[2] - candidate_box[0])
                    * (candidate_box[3] - candidate_box[1])
                ),
                1.0,
            )
            best_host: tuple[float, int, float, float] | None = None
            for host_tid, (host_box, _) in by_frame[frame].items():
                host_gid = tid_to_gid.get(host_tid)
                if host_tid == tracklet_id or host_gid in {None, global_id}:
                    continue
                host_area = max(
                    float((host_box[2] - host_box[0]) * (host_box[3] - host_box[1])),
                    1.0,
                )
                containment = _intersection_over_candidate(candidate_box, host_box)
                area_ratio = candidate_area / host_area
                if containment < min_containment or area_ratio > max_area_ratio:
                    continue
                score = containment - 0.1 * area_ratio
                if best_host is None or score > best_host[0]:
                    best_host = (score, int(host_gid), containment, area_ratio)
            if best_host is not None:
                _, host_gid, containment, area_ratio = best_host
                host_confidences = [
                    mean_confidence[host_tid]
                    for host_tid in gid_members[host_gid]
                    if host_tid in mean_confidence
                ]
                host_mean = float(np.mean(host_confidences))
                nested_by_gid[host_gid] += 1
                evidence_by_gid[host_gid].append(
                    (containment, area_ratio, host_mean)
                )

        if not nested_by_gid:
            continue
        host_gid, nested_count = nested_by_gid.most_common(1)[0]
        nested_fraction = nested_count / len(items)
        evidence = evidence_by_gid[host_gid]
        host_tids = gid_members[host_gid]
        host_duration = max(duration(host_tid) for host_tid in host_tids)
        host_mean = float(np.mean([item[2] for item in evidence]))
        if (
            nested_fraction < min_nested_fraction
            or host_duration <= candidate_duration
            or host_mean <= candidate_confidence
        ):
            continue
        proposals.append(
            NestedTrackletSuppression(
                tracklet_id=tracklet_id,
                host_global_id=host_gid,
                observations=len(items),
                nested_observations=nested_count,
                nested_fraction=nested_fraction,
                mean_containment=float(np.mean([item[0] for item in evidence])),
                mean_area_ratio=float(np.mean([item[1] for item in evidence])),
                mean_confidence=candidate_confidence,
                host_mean_confidence=host_mean,
                duration_s=candidate_duration,
            )
        )
    return proposals


def apply_tracklet_suppressions(
    local_pass: LocalPass,
    tracklet_ids: set[int],
) -> None:
    """Remove selected local detections after gallery validation is complete."""
    if not tracklet_ids:
        return
    local_pass.dets = {
        frame: [item for item in detections if int(item[0]) not in tracklet_ids]
        for frame, detections in local_pass.dets.items()
    }
    for tracklet_id in tracklet_ids:
        local_pass.appearance_embeddings.pop(tracklet_id, None)
        local_pass.part_appearance_embeddings.pop(tracklet_id, None)
    local_pass._pair_constraints = None
