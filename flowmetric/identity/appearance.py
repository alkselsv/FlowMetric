"""Conservative OSNet appearance links for unresolved local tracklets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from flowmetric.media.video import load_video_frames
from flowmetric.tracking import LocalPass


class LocalAppearanceEncoder:
    """Small adapter around the optional BoxMOT OSNet inference backend."""

    def __init__(self, weights: Path, *, device: str) -> None:
        if not Path(weights).is_file():
            raise ValueError(f"appearance model does not exist: {weights}")
        try:
            import torch
            from boxmot.appearance.reid_auto_backend import ReidAutoBackend
        except ImportError as exc:
            raise RuntimeError(
                "OSNet appearance stitching requires the optional BoxMOT dependencies; "
                "install requirements-experiments.txt and boxmot==12.0.2"
            ) from exc
        backend = ReidAutoBackend(
            weights=Path(weights),
            device=torch.device(device),
            half=False,
        )
        self._model = backend.model

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        features = np.asarray(self._model.get_features(boxes, frame), dtype=np.float32)
        if features.ndim != 2 or len(features) != len(boxes):
            raise RuntimeError("appearance backend returned malformed embeddings")
        return features

    def encode_part_aware(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        occluders: Sequence[Sequence[np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode full, upper and lower crops with visibility-aware weights."""
        _, combined, visibility = self.encode_full_and_part_aware(
            frame, boxes, occluders
        )
        return combined, visibility

    def encode_full_and_part_aware(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        occluders: Sequence[Sequence[np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode full crops and their part-aware counterparts in one inference."""
        if len(boxes) != len(occluders):
            raise ValueError("occluders must contain one collection per box")
        regions = np.stack([region for box in boxes for region in _part_boxes(box)])
        features = np.asarray(
            self._model.get_features(regions, frame), dtype=np.float32
        )
        if features.ndim != 2 or len(features) != 3 * len(boxes):
            raise RuntimeError("appearance backend returned malformed part embeddings")
        features = features.reshape(len(boxes), 3, features.shape[1])
        visibility = np.stack(
            [part_visibility(box, others) for box, others in zip(boxes, occluders)]
        )
        full = features[:, 0]
        full = full / np.maximum(np.linalg.norm(full, axis=1, keepdims=True), 1e-12)
        return (
            full,
            compose_part_embeddings(features, visibility),
            visibility.mean(axis=1),
        )


class TrackerAppearanceEncoder:
    """Adapt the local OSNet encoder to Ultralytics BoT-SORT's xywh API."""

    def __init__(self, encoder: LocalAppearanceEncoder) -> None:
        self._encoder = encoder

    def __call__(self, frame: np.ndarray, detections: np.ndarray) -> list[np.ndarray]:
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] < 4:
            raise ValueError("tracker detections must have shape N x >=4")
        boxes = detections[:, :4].copy()
        boxes[:, 0] = detections[:, 0] - detections[:, 2] / 2
        boxes[:, 1] = detections[:, 1] - detections[:, 3] / 2
        boxes[:, 2] = detections[:, 0] + detections[:, 2] / 2
        boxes[:, 3] = detections[:, 1] + detections[:, 3] / 2
        return list(self._encoder.encode(frame, boxes))


def _part_boxes(box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = np.asarray(box, dtype=np.float32)
    height = max(float(box[3] - box[1]), 1.0)
    upper = box.copy()
    upper[3] = box[1] + 0.55 * height
    lower = box.copy()
    lower[1] = box[1] + 0.45 * height
    return box.copy(), upper, lower


def _intersection_fraction(region: np.ndarray, occluder: np.ndarray) -> float:
    x1, y1 = np.maximum(region[:2], occluder[:2])
    x2, y2 = np.minimum(region[2:], occluder[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area = max(
        float((region[2] - region[0]) * (region[3] - region[1])),
        1.0,
    )
    return min(intersection / area, 1.0)


def part_visibility(
    box: np.ndarray, occluders: Sequence[np.ndarray]
) -> np.ndarray:
    """Return visibility for full, upper and lower body regions."""
    regions = _part_boxes(box)
    visibility = [1.0]
    for region in regions[1:]:
        covered = max(
            (
                _intersection_fraction(region, np.asarray(other, dtype=float))
                for other in occluders
            ),
            default=0.0,
        )
        visibility.append(max(0.05, 1.0 - covered))
    return np.asarray(visibility, dtype=np.float32)


def compose_part_embeddings(
    features: np.ndarray, visibility: np.ndarray
) -> np.ndarray:
    """Combine N×3×D part features so cosine uses visible regions most."""
    features = np.asarray(features, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32)
    if features.ndim != 3 or features.shape[1] != 3:
        raise ValueError("features must have shape N x 3 x D")
    if visibility.shape != features.shape[:2]:
        raise ValueError("visibility must have shape N x 3")
    normalized = features / np.maximum(
        np.linalg.norm(features, axis=2, keepdims=True), 1e-12
    )
    base_weights = np.asarray([0.50, 0.25, 0.25], dtype=np.float32)
    effective = np.clip(visibility, 0.0, 1.0) * base_weights[None, :]
    combined = (normalized * np.sqrt(effective)[:, :, None]).reshape(
        len(features), -1
    )
    return combined / np.maximum(
        np.linalg.norm(combined, axis=1, keepdims=True), 1e-12
    )


@dataclass(frozen=True)
class AppearanceStitch:
    earlier_id: int
    later_id: int
    gap_s: float
    endpoint_distance: float
    height_ratio: float
    similarity: float
    score: float
    selection: str = "mutual_best"
    global_similarity: float | None = None
    primary_similarity: float | None = None
    secondary_similarity: float | None = None
    support: int | None = None

    def as_dict(self) -> dict[str, float | int | str]:
        result: dict[str, float | int | str] = {
            "earlier_tracklet_id": self.earlier_id,
            "later_tracklet_id": self.later_id,
            "gap_s": round(self.gap_s, 4),
            "endpoint_distance": round(self.endpoint_distance, 4),
            "height_ratio": round(self.height_ratio, 4),
            "similarity": round(self.similarity, 4),
            "score": round(self.score, 4),
            "selection": self.selection,
        }
        if self.global_similarity is not None:
            result["global_similarity"] = round(self.global_similarity, 4)
        if self.primary_similarity is not None:
            result["primary_similarity"] = round(self.primary_similarity, 4)
        if self.secondary_similarity is not None:
            result["secondary_similarity"] = round(self.secondary_similarity, 4)
        if self.support is not None:
            result["support"] = self.support
        return result


@dataclass(frozen=True)
class _AppearanceTrack:
    tracklet_id: int
    first_frame: int
    last_frame: int
    first_center: np.ndarray
    last_center: np.ndarray
    first_height: float
    last_height: float
    prototype: np.ndarray


def _prototype(
    samples: list[tuple[int, np.ndarray, float]], *, max_samples: int = 8
) -> np.ndarray:
    strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[:max_samples]
    matrix = np.stack([np.asarray(item[1], dtype=float) for item in strongest])
    vector = np.mean(matrix, axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def robust_temporal_prototype(
    samples: list[tuple[int, np.ndarray, float]],
    *,
    max_samples: int = 8,
    min_inlier_similarity: float = 0.65,
) -> tuple[np.ndarray, int]:
    """Aggregate a tracklet around its medoid while down-weighting weak crops."""
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not 0 <= min_inlier_similarity <= 1:
        raise ValueError("min_inlier_similarity must be in [0, 1]")
    if not samples:
        raise ValueError("samples must be non-empty")
    strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[:max_samples]
    matrix = np.stack([np.asarray(item[1], dtype=float) for item in strongest])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarities = matrix @ matrix.T
    medoid_index = int(np.argmax(np.median(similarities, axis=1)))
    inliers = similarities[medoid_index] >= min_inlier_similarity
    if not np.any(inliers):
        inliers[medoid_index] = True
    quality = np.asarray([max(float(item[2]), 0.05) for item in strongest])
    weights = quality[inliers]
    vector = np.average(matrix[inliers], axis=0, weights=weights)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector, int(np.count_nonzero(inliers))


def summarize_robust_appearance(local_pass: LocalPass) -> dict[str, int]:
    """Return audit counts for robust aggregation without exposing embeddings."""
    tracklets = samples = inliers = 0
    for tracklet_samples in local_pass.appearance_embeddings.values():
        if not tracklet_samples:
            continue
        tracklets += 1
        considered = min(len(tracklet_samples), 8)
        samples += considered
        _, supported = robust_temporal_prototype(tracklet_samples)
        inliers += supported
    return {
        "tracklets": tracklets,
        "samples": samples,
        "inliers": inliers,
        "outliers": samples - inliers,
    }


def calibrate_appearance_threshold(
    local_pass: LocalPass,
    *,
    cluster_by_tracklet: dict[int, str],
    verified_clusters: set[str],
    cannot_link: set[tuple[int, int]],
    default_threshold: float = 0.82,
    minimum_threshold: float = 0.80,
    separation_margin: float = 0.04,
    min_positive_pairs: int = 4,
    min_negative_pairs: int = 20,
) -> dict[str, float | int | bool | str]:
    """Calibrate a conservative cosine threshold from trusted scene evidence."""
    for label, value in {
        "default_threshold": default_threshold,
        "minimum_threshold": minimum_threshold,
        "separation_margin": separation_margin,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if not 0 < minimum_threshold <= default_threshold <= 1:
        raise ValueError(
            "thresholds must satisfy 0 < minimum_threshold <= default_threshold <= 1"
        )
    if separation_margin <= 0:
        raise ValueError("separation_margin must be positive")
    for label, value in {
        "min_positive_pairs": min_positive_pairs,
        "min_negative_pairs": min_negative_pairs,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")

    prototypes = {
        tracklet_id: _prototype(samples)
        for tracklet_id, samples in local_pass.appearance_embeddings.items()
        if samples
    }
    by_cluster: dict[str, list[int]] = {}
    for tracklet_id, cluster in cluster_by_tracklet.items():
        if cluster in verified_clusters and tracklet_id in prototypes:
            by_cluster.setdefault(cluster, []).append(tracklet_id)
    positives: list[float] = []
    for tracklet_ids in by_cluster.values():
        ordered = sorted(tracklet_ids)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                positives.append(float(np.dot(prototypes[left], prototypes[right])))
    negatives = [
        float(np.dot(prototypes[left], prototypes[right]))
        for left, right in cannot_link
        if left in prototypes and right in prototypes
    ]
    result: dict[str, float | int | bool | str] = {
        "enabled": True,
        "applied": False,
        "threshold": float(default_threshold),
        "default_threshold": float(default_threshold),
        "minimum_threshold": float(minimum_threshold),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "reason": "insufficient_trusted_pairs",
    }
    if len(positives) < min_positive_pairs or len(negatives) < min_negative_pairs:
        return result
    positive_floor = float(np.quantile(positives, 0.20))
    negative_ceiling = float(np.quantile(negatives, 0.99))
    result.update(
        {
            "positive_floor": positive_floor,
            "negative_ceiling": negative_ceiling,
            "separation": positive_floor - negative_ceiling,
        }
    )
    if positive_floor - negative_ceiling < separation_margin:
        result["reason"] = "distributions_not_separated"
        return result
    threshold = max(
        minimum_threshold,
        min(default_threshold, (positive_floor + negative_ceiling) / 2),
    )
    result.update(
        {
            "applied": threshold < default_threshold,
            "threshold": threshold,
            "reason": "separated_trusted_pairs",
        }
    )
    return result


def _tracks(
    local_pass: LocalPass,
    appearance_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] | None = None,
) -> list[_AppearanceTrack]:
    observations: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame, detections in local_pass.dets.items():
        for tracklet_id, box, _ in detections:
            observations.setdefault(int(tracklet_id), []).append(
                (int(frame), np.asarray(box, dtype=float))
            )
    tracks: list[_AppearanceTrack] = []
    embedding_source = appearance_embeddings or local_pass.appearance_embeddings
    for tracklet_id, samples in embedding_source.items():
        boxes = sorted(observations.get(tracklet_id, []), key=lambda item: item[0])
        if not boxes or not samples:
            continue
        first_frame, first_box = boxes[0]
        last_frame, last_box = boxes[-1]
        tracks.append(
            _AppearanceTrack(
                tracklet_id=tracklet_id,
                first_frame=first_frame,
                last_frame=last_frame,
                first_center=np.asarray(
                    [(first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2]
                ),
                last_center=np.asarray(
                    [(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2]
                ),
                first_height=max(float(first_box[3] - first_box[1]), 1.0),
                last_height=max(float(last_box[3] - last_box[1]), 1.0),
                prototype=_prototype(samples),
            )
        )
    return tracks


def _appearance_candidates(
    local_pass: LocalPass,
    *,
    max_gap_s: float,
    max_endpoint_distance: float,
    max_height_ratio: float,
    min_similarity: float,
    appearance_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] | None = None,
) -> list[AppearanceStitch]:
    candidates: list[AppearanceStitch] = []
    tracks = _tracks(local_pass, appearance_embeddings)
    for earlier in tracks:
        for later in tracks:
            gap_frames = later.first_frame - earlier.last_frame
            if gap_frames <= 0:
                continue
            gap_s = gap_frames / local_pass.fps
            if gap_s > max_gap_s:
                continue
            height_ratio = max(earlier.last_height, later.first_height) / max(
                min(earlier.last_height, later.first_height), 1.0
            )
            if height_ratio > max_height_ratio:
                continue
            scale = max((earlier.last_height + later.first_height) / 2, 1.0)
            endpoint_distance = float(
                np.linalg.norm(earlier.last_center - later.first_center) / scale
            )
            if endpoint_distance > max_endpoint_distance:
                continue
            similarity = float(np.dot(earlier.prototype, later.prototype))
            if similarity < min_similarity:
                continue
            score = (
                1.0 - similarity
                + 0.04 * endpoint_distance
                + 0.02 * gap_s / max_gap_s
            )
            candidates.append(
                AppearanceStitch(
                    earlier_id=earlier.tracklet_id,
                    later_id=later.tracklet_id,
                    gap_s=gap_s,
                    endpoint_distance=endpoint_distance,
                    height_ratio=height_ratio,
                    similarity=similarity,
                    score=score,
                )
            )
    return candidates


def propose_appearance_stitches(
    local_pass: LocalPass,
    *,
    max_gap_s: float = 4.0,
    max_endpoint_distance: float = 1.5,
    max_height_ratio: float = 1.8,
    min_similarity: float = 0.82,
    ambiguity_margin: float = 0.06,
    global_assignment: bool = False,
    global_min_similarity: float = 0.90,
    global_unmatched_cost: float = 0.16,
    endpoint_rescue: bool = False,
    endpoint_min_similarity: float = 0.90,
    endpoint_global_floor: float = 0.70,
) -> list[AppearanceStitch]:
    """Return mutual-best links plus optional residual global assignment."""
    for label, value in {
        "max_gap_s": max_gap_s,
        "max_endpoint_distance": max_endpoint_distance,
        "max_height_ratio": max_height_ratio,
        "min_similarity": min_similarity,
        "ambiguity_margin": ambiguity_margin,
        "global_min_similarity": global_min_similarity,
        "global_unmatched_cost": global_unmatched_cost,
        "endpoint_min_similarity": endpoint_min_similarity,
        "endpoint_global_floor": endpoint_global_floor,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")

    candidates = _appearance_candidates(
        local_pass,
        max_gap_s=max_gap_s,
        max_endpoint_distance=max_endpoint_distance,
        max_height_ratio=max_height_ratio,
        min_similarity=min_similarity,
    )

    if endpoint_rescue:
        tracks = _tracks(local_pass)
        existing_pairs = {
            (candidate.earlier_id, candidate.later_id) for candidate in candidates
        }
        for earlier in tracks:
            for later in tracks:
                pair = (earlier.tracklet_id, later.tracklet_id)
                if pair in existing_pairs:
                    continue
                gap_frames = later.first_frame - earlier.last_frame
                if gap_frames <= 0:
                    continue
                gap_s = gap_frames / local_pass.fps
                if gap_s > max_gap_s:
                    continue
                height_ratio = max(earlier.last_height, later.first_height) / max(
                    min(earlier.last_height, later.first_height), 1.0
                )
                if height_ratio > max_height_ratio:
                    continue
                scale = max((earlier.last_height + later.first_height) / 2, 1.0)
                endpoint_distance = float(
                    np.linalg.norm(earlier.last_center - later.first_center) / scale
                )
                if endpoint_distance > max_endpoint_distance:
                    continue
                earlier_samples = sorted(
                    local_pass.appearance_embeddings.get(earlier.tracklet_id, []),
                    key=lambda item: item[0],
                )
                later_samples = sorted(
                    local_pass.appearance_embeddings.get(later.tracklet_id, []),
                    key=lambda item: item[0],
                )
                if len(earlier_samples) < 2 or len(later_samples) < 2:
                    continue
                earlier_endpoint = _prototype(earlier_samples[-2:], max_samples=2)
                later_endpoint = _prototype(later_samples[:2], max_samples=2)
                endpoint_similarity = float(
                    np.dot(earlier_endpoint, later_endpoint)
                )
                global_similarity = float(
                    np.dot(earlier.prototype, later.prototype)
                )
                if (
                    endpoint_similarity < endpoint_min_similarity
                    or global_similarity < endpoint_global_floor
                ):
                    continue
                score = (
                    1.0 - endpoint_similarity
                    + 0.04 * endpoint_distance
                    + 0.02 * gap_s / max_gap_s
                )
                candidates.append(
                    AppearanceStitch(
                        earlier_id=earlier.tracklet_id,
                        later_id=later.tracklet_id,
                        gap_s=gap_s,
                        endpoint_distance=endpoint_distance,
                        height_ratio=height_ratio,
                        similarity=endpoint_similarity,
                        score=score,
                        selection="endpoint_temporal_rescue",
                        global_similarity=global_similarity,
                    )
                )

    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: Iterable[AppearanceStitch]) -> bool:
        ranked = list(choices)
        return ranked[0] == candidate and (
            len(ranked) == 1 or ranked[1].score - candidate.score >= ambiguity_margin
        )

    mutual = list(
        candidate
        for candidate in candidates
        if unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    )
    if not global_assignment:
        return sorted(
            mutual,
            key=lambda item: (item.score, item.earlier_id, item.later_id),
        )

    mutual_pairs = {
        (candidate.earlier_id, candidate.later_id) for candidate in mutual
    }
    residual = [
        candidate
        for candidate in candidates
        if candidate.similarity >= global_min_similarity
        and candidate.score < global_unmatched_cost
        and (candidate.earlier_id, candidate.later_id) not in mutual_pairs
    ]
    if not residual:
        return sorted(
            mutual,
            key=lambda item: (item.score, item.earlier_id, item.later_id),
        )

    left_ids = sorted({candidate.earlier_id for candidate in residual})
    right_ids = sorted({candidate.later_id for candidate in residual})
    left_index = {tracklet_id: index for index, tracklet_id in enumerate(left_ids)}
    right_index = {tracklet_id: index for index, tracklet_id in enumerate(right_ids)}
    costs = np.full(
        (len(left_ids), len(right_ids) + len(left_ids)),
        1e6,
        dtype=float,
    )
    costs[:, len(right_ids) :] = global_unmatched_cost
    by_pair: dict[tuple[int, int], AppearanceStitch] = {}
    for candidate in residual:
        row = left_index[candidate.earlier_id]
        column = right_index[candidate.later_id]
        if candidate.score < costs[row, column]:
            costs[row, column] = candidate.score
            by_pair[(row, column)] = candidate
    rows, columns = linear_sum_assignment(costs)
    global_links = [
        AppearanceStitch(
            **{
                **by_pair[(row, column)].__dict__,
                "selection": "global_supplemental_assignment",
            }
        )
        for row, column in zip(rows, columns)
        if column < len(right_ids)
        and (row, column) in by_pair
        and costs[row, column] < global_unmatched_cost
    ]
    return sorted(
        (
            *mutual,
            *global_links,
        ),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def propose_appearance_graph_edges(
    local_pass: LocalPass,
    *,
    min_similarity: float = 0.94,
    max_score: float = 0.10,
    max_gap_s: float = 4.0,
    max_endpoint_distance: float = 1.5,
    max_height_ratio: float = 1.8,
) -> list[AppearanceStitch]:
    """Return strict appearance edges without mutual-best pruning for graph use."""
    for label, value in {
        "max_score": max_score,
        "max_gap_s": max_gap_s,
        "max_endpoint_distance": max_endpoint_distance,
        "max_height_ratio": max_height_ratio,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be a finite number in [-1, 1]")
    return sorted(
        (
            candidate
            for candidate in _appearance_candidates(
                local_pass,
                max_gap_s=max_gap_s,
                max_endpoint_distance=max_endpoint_distance,
                max_height_ratio=max_height_ratio,
                min_similarity=min_similarity,
            )
            if candidate.score <= max_score
        ),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def propose_ensemble_appearance_stitches(
    local_pass: LocalPass,
    *,
    full_min_similarity: float = 0.80,
    part_min_similarity: float = 0.86,
    ambiguity_margin: float = 0.04,
) -> list[AppearanceStitch]:
    """Preserve full-body links and add only full/part consensus links."""
    for label, value in {
        "full_min_similarity": full_min_similarity,
        "part_min_similarity": part_min_similarity,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if full_min_similarity > 1 or part_min_similarity > 1:
        raise ValueError("ensemble similarity thresholds must not exceed 1")
    full_mutual = propose_appearance_stitches(local_pass)
    if not local_pass.part_appearance_embeddings:
        return full_mutual

    full_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=full_min_similarity,
    )
    part_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=part_min_similarity,
        appearance_embeddings=local_pass.part_appearance_embeddings,
    )
    full_by_pair = {
        (item.earlier_id, item.later_id): item for item in full_candidates
    }
    consensus: list[AppearanceStitch] = []
    for part in part_candidates:
        pair = (part.earlier_id, part.later_id)
        full = full_by_pair.get(pair)
        if full is None:
            continue
        similarity = 0.60 * full.similarity + 0.40 * part.similarity
        geometry_cost = full.score - (1.0 - full.similarity)
        consensus.append(
            AppearanceStitch(
                earlier_id=full.earlier_id,
                later_id=full.later_id,
                gap_s=full.gap_s,
                endpoint_distance=full.endpoint_distance,
                height_ratio=full.height_ratio,
                similarity=similarity,
                score=1.0 - similarity + geometry_cost,
                selection="full_part_consensus",
                primary_similarity=full.similarity,
                secondary_similarity=part.similarity,
            )
        )

    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in consensus:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: list[AppearanceStitch]) -> bool:
        return choices[0] == candidate and (
            len(choices) == 1
            or choices[1].score - candidate.score >= ambiguity_margin
        )

    existing = {(item.earlier_id, item.later_id) for item in full_mutual}
    additions = [
        candidate
        for candidate in consensus
        if (candidate.earlier_id, candidate.later_id) not in existing
        and unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    ]
    return sorted(
        (*full_mutual, *additions),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def propose_pose_attribute_stitches(
    local_pass: LocalPass,
    *,
    full_min_similarity: float = 0.78,
    attribute_min_similarity: float = 0.90,
    ambiguity_margin: float = 0.04,
    min_support: int = 2,
) -> list[AppearanceStitch]:
    """Preserve OSNet links and add only pose-guided attribute consensus."""
    for label, value in {
        "full_min_similarity": full_min_similarity,
        "attribute_min_similarity": attribute_min_similarity,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if full_min_similarity > 1 or attribute_min_similarity > 1:
        raise ValueError("pose-attribute similarity thresholds must not exceed 1")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 2:
        raise ValueError("min_support must be an integer of at least 2")
    baseline = propose_appearance_stitches(local_pass)
    if not local_pass.pose_attribute_embeddings:
        return baseline

    full_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=full_min_similarity,
    )
    attribute_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=attribute_min_similarity,
        appearance_embeddings=local_pass.pose_attribute_embeddings,
    )
    full_by_pair = {
        (item.earlier_id, item.later_id): item for item in full_candidates
    }
    candidates: list[AppearanceStitch] = []
    for attribute in attribute_candidates:
        pair = (attribute.earlier_id, attribute.later_id)
        full = full_by_pair.get(pair)
        if full is None:
            continue
        support = min(
            len(local_pass.pose_attribute_embeddings.get(full.earlier_id, [])),
            len(local_pass.pose_attribute_embeddings.get(full.later_id, [])),
        )
        if support < min_support:
            continue
        similarity = 0.65 * full.similarity + 0.35 * attribute.similarity
        geometry_cost = full.score - (1.0 - full.similarity)
        candidates.append(
            AppearanceStitch(
                earlier_id=full.earlier_id,
                later_id=full.later_id,
                gap_s=full.gap_s,
                endpoint_distance=full.endpoint_distance,
                height_ratio=full.height_ratio,
                similarity=similarity,
                score=1.0 - similarity + geometry_cost,
                selection="pose_attribute_consensus",
                primary_similarity=full.similarity,
                secondary_similarity=attribute.similarity,
                support=support,
            )
        )

    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: list[AppearanceStitch]) -> bool:
        return choices[0] == candidate and (
            len(choices) == 1
            or choices[1].score - candidate.score >= ambiguity_margin
        )

    existing = {(item.earlier_id, item.later_id) for item in baseline}
    additions = [
        candidate
        for candidate in candidates
        if (candidate.earlier_id, candidate.later_id) not in existing
        and unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    ]
    return sorted(
        (*baseline, *additions),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def propose_segmented_attribute_stitches(
    local_pass: LocalPass,
    *,
    full_min_similarity: float = 0.78,
    attribute_min_similarity: float = 0.92,
    ambiguity_margin: float = 0.04,
    min_support: int = 2,
) -> list[AppearanceStitch]:
    """Preserve OSNet links and add foreground-masked clothing consensus."""
    for label, value in {
        "full_min_similarity": full_min_similarity,
        "attribute_min_similarity": attribute_min_similarity,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if full_min_similarity > 1 or attribute_min_similarity > 1:
        raise ValueError("segmented-attribute thresholds must not exceed 1")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 2:
        raise ValueError("min_support must be an integer of at least 2")
    baseline = propose_appearance_stitches(local_pass)
    if not local_pass.segmented_attribute_embeddings:
        return baseline
    full_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=full_min_similarity,
    )
    attribute_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=attribute_min_similarity,
        appearance_embeddings=local_pass.segmented_attribute_embeddings,
    )
    full_by_pair = {
        (item.earlier_id, item.later_id): item for item in full_candidates
    }
    candidates: list[AppearanceStitch] = []
    for attribute in attribute_candidates:
        pair = (attribute.earlier_id, attribute.later_id)
        full = full_by_pair.get(pair)
        if full is None:
            continue
        support = min(
            len(local_pass.segmented_attribute_embeddings.get(full.earlier_id, [])),
            len(local_pass.segmented_attribute_embeddings.get(full.later_id, [])),
        )
        if support < min_support:
            continue
        similarity = 0.65 * full.similarity + 0.35 * attribute.similarity
        geometry_cost = full.score - (1.0 - full.similarity)
        candidates.append(
            AppearanceStitch(
                earlier_id=full.earlier_id,
                later_id=full.later_id,
                gap_s=full.gap_s,
                endpoint_distance=full.endpoint_distance,
                height_ratio=full.height_ratio,
                similarity=similarity,
                score=1.0 - similarity + geometry_cost,
                selection="segmented_attribute_consensus",
                primary_similarity=full.similarity,
                secondary_similarity=attribute.similarity,
                support=support,
            )
        )
    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: list[AppearanceStitch]) -> bool:
        return choices[0] == candidate and (
            len(choices) == 1
            or choices[1].score - candidate.score >= ambiguity_margin
        )

    existing = {(item.earlier_id, item.later_id) for item in baseline}
    additions = [
        candidate
        for candidate in candidates
        if (candidate.earlier_id, candidate.later_id) not in existing
        and unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    ]
    return sorted(
        (*baseline, *additions),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def _independent_view_similarity(
    left: list[tuple[int, np.ndarray, float]],
    right: list[tuple[int, np.ndarray, float]],
    *,
    min_pair_similarity: float,
    min_support: int,
    max_samples: int = 8,
) -> float | None:
    """Match distinct crop observations and require repeated visual support."""
    if len(left) < min_support or len(right) < min_support:
        return None
    left_best = sorted(left, key=lambda item: (-item[2], item[0]))[:max_samples]
    right_best = sorted(right, key=lambda item: (-item[2], item[0]))[:max_samples]
    left_matrix = np.stack([np.asarray(item[1], dtype=float) for item in left_best])
    right_matrix = np.stack([np.asarray(item[1], dtype=float) for item in right_best])
    left_matrix /= np.maximum(np.linalg.norm(left_matrix, axis=1, keepdims=True), 1e-12)
    right_matrix /= np.maximum(np.linalg.norm(right_matrix, axis=1, keepdims=True), 1e-12)
    similarities = left_matrix @ right_matrix.T
    rows, columns = linear_sum_assignment(-similarities)
    matched = sorted(
        (float(similarities[row, column]) for row, column in zip(rows, columns)),
        reverse=True,
    )
    supported = matched[:min_support]
    if len(supported) < min_support or supported[-1] < min_pair_similarity:
        return None
    return float(np.mean(supported))


def propose_multiview_appearance_stitches(
    local_pass: LocalPass,
    *,
    min_pair_similarity: float = 0.88,
    min_support: int = 2,
    ambiguity_margin: float = 0.04,
) -> list[AppearanceStitch]:
    """Preserve mean-prototype links and add repeated independent view matches."""
    if not 0 < min_pair_similarity <= 1:
        raise ValueError("min_pair_similarity must be in (0, 1]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 2:
        raise ValueError("min_support must be an integer of at least 2")
    if not np.isfinite(ambiguity_margin) or ambiguity_margin <= 0:
        raise ValueError("ambiguity_margin must be a positive finite number")

    baseline = propose_appearance_stitches(local_pass)
    geometric = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=-1.0,
    )
    candidates: list[AppearanceStitch] = []
    for candidate in geometric:
        similarity = _independent_view_similarity(
            local_pass.appearance_embeddings.get(candidate.earlier_id, []),
            local_pass.appearance_embeddings.get(candidate.later_id, []),
            min_pair_similarity=min_pair_similarity,
            min_support=min_support,
        )
        if similarity is None:
            continue
        geometry_cost = candidate.score - (1.0 - candidate.similarity)
        candidates.append(
            AppearanceStitch(
                earlier_id=candidate.earlier_id,
                later_id=candidate.later_id,
                gap_s=candidate.gap_s,
                endpoint_distance=candidate.endpoint_distance,
                height_ratio=candidate.height_ratio,
                similarity=similarity,
                score=1.0 - similarity + geometry_cost,
                selection="multiview_consensus",
                primary_similarity=candidate.similarity,
                support=min_support,
            )
        )

    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: list[AppearanceStitch]) -> bool:
        return choices[0] == candidate and (
            len(choices) == 1
            or choices[1].score - candidate.score >= ambiguity_margin
        )

    existing = {(item.earlier_id, item.later_id) for item in baseline}
    additions = [
        candidate
        for candidate in candidates
        if (candidate.earlier_id, candidate.later_id) not in existing
        and unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    ]
    return sorted(
        (*baseline, *additions),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def propose_robust_appearance_stitches(
    local_pass: LocalPass,
    *,
    robust_min_similarity: float = 0.88,
    mean_similarity_floor: float = 0.70,
    ambiguity_margin: float = 0.04,
    min_support: int = 2,
) -> list[AppearanceStitch]:
    """Preserve mean links and add only unambiguous robust-prototype links."""
    for label, value in {
        "robust_min_similarity": robust_min_similarity,
        "ambiguity_margin": ambiguity_margin,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")
    if robust_min_similarity > 1:
        raise ValueError("robust_min_similarity must not exceed 1")
    if not np.isfinite(mean_similarity_floor) or not -1 <= mean_similarity_floor <= 1:
        raise ValueError("mean_similarity_floor must be in [-1, 1]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 2:
        raise ValueError("min_support must be an integer of at least 2")

    baseline = propose_appearance_stitches(local_pass)
    robust_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    support: dict[int, int] = {}
    for tracklet_id, samples in local_pass.appearance_embeddings.items():
        if not samples:
            continue
        prototype, inlier_count = robust_temporal_prototype(samples)
        robust_embeddings[tracklet_id] = [(samples[0][0], prototype, 1.0)]
        support[tracklet_id] = inlier_count

    mean_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=mean_similarity_floor,
    )
    robust_candidates = _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=robust_min_similarity,
        appearance_embeddings=robust_embeddings,
    )
    mean_by_pair = {
        (item.earlier_id, item.later_id): item for item in mean_candidates
    }
    candidates: list[AppearanceStitch] = []
    for robust in robust_candidates:
        pair = (robust.earlier_id, robust.later_id)
        mean = mean_by_pair.get(pair)
        if mean is None:
            continue
        pair_support = min(support[mean.earlier_id], support[mean.later_id])
        if pair_support < min_support:
            continue
        geometry_cost = mean.score - (1.0 - mean.similarity)
        candidates.append(
            AppearanceStitch(
                earlier_id=mean.earlier_id,
                later_id=mean.later_id,
                gap_s=mean.gap_s,
                endpoint_distance=mean.endpoint_distance,
                height_ratio=mean.height_ratio,
                similarity=robust.similarity,
                score=1.0 - robust.similarity + geometry_cost,
                selection="robust_temporal_consensus",
                primary_similarity=mean.similarity,
                secondary_similarity=robust.similarity,
                support=pair_support,
            )
        )

    by_earlier: dict[int, list[AppearanceStitch]] = {}
    by_later: dict[int, list[AppearanceStitch]] = {}
    for candidate in candidates:
        by_earlier.setdefault(candidate.earlier_id, []).append(candidate)
        by_later.setdefault(candidate.later_id, []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (item.score, item.earlier_id, item.later_id))

    def unambiguous(candidate: AppearanceStitch, choices: list[AppearanceStitch]) -> bool:
        return choices[0] == candidate and (
            len(choices) == 1
            or choices[1].score - candidate.score >= ambiguity_margin
        )

    existing = {(item.earlier_id, item.later_id) for item in baseline}
    additions = [
        candidate
        for candidate in candidates
        if (candidate.earlier_id, candidate.later_id) not in existing
        and unambiguous(candidate, by_earlier[candidate.earlier_id])
        and unambiguous(candidate, by_later[candidate.later_id])
    ]
    return sorted(
        (*baseline, *additions),
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )


def select_multiview_resample_tracklets(
    local_pass: LocalPass,
    unresolved_tracklets: set[int],
    *,
    target_samples: int = 3,
) -> set[int]:
    """Select unresolved tracklets and only their feasible appearance neighbours."""
    if isinstance(target_samples, bool) or not isinstance(target_samples, int):
        raise ValueError("target_samples must be an integer")
    if target_samples < 2:
        raise ValueError("target_samples must be at least 2")
    unresolved = {int(tracklet_id) for tracklet_id in unresolved_tracklets}
    eligible = set(unresolved)
    for candidate in _appearance_candidates(
        local_pass,
        max_gap_s=4.0,
        max_endpoint_distance=1.5,
        max_height_ratio=1.8,
        min_similarity=-1.0,
    ):
        pair = {candidate.earlier_id, candidate.later_id}
        if pair & unresolved:
            eligible.update(pair)
    return {
        tracklet_id
        for tracklet_id in eligible
        if len(local_pass.appearance_embeddings.get(tracklet_id, [])) < target_samples
    }


def _crop_sharpness(frame: np.ndarray, box: np.ndarray) -> float:
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(np.floor(box[0]))))
    y1 = max(0, min(height, int(np.floor(box[1]))))
    x2 = max(0, min(width, int(np.ceil(box[2]))))
    y2 = max(0, min(height, int(np.ceil(box[3]))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def densify_multiview_appearance(
    source: Path,
    local_pass: LocalPass,
    weights: Path,
    *,
    device: str,
    tracklet_ids: set[int],
    target_samples: int = 3,
    min_sharpness: float = 20.0,
    min_visibility: float = 0.55,
) -> dict[str, int | list[int]]:
    """Add diverse, sharp OSNet observations without rerunning detection."""
    if isinstance(target_samples, bool) or not isinstance(target_samples, int):
        raise ValueError("target_samples must be an integer")
    if target_samples < 2:
        raise ValueError("target_samples must be at least 2")
    if not np.isfinite(min_sharpness) or min_sharpness < 0:
        raise ValueError("min_sharpness must be a non-negative finite number")
    if not np.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be in [0, 1]")

    requested = {int(tracklet_id) for tracklet_id in tracklet_ids}
    existing_frames = {
        tracklet_id: {
            int(frame) for frame, _, _ in local_pass.appearance_embeddings.get(tracklet_id, [])
        }
        for tracklet_id in requested
    }
    observations: dict[int, list[tuple[int, np.ndarray, float, list[np.ndarray]]]] = {
        tracklet_id: [] for tracklet_id in requested
    }
    for frame_index, detections in local_pass.dets.items():
        for tracklet_id, box, confidence in detections:
            tracklet_id = int(tracklet_id)
            if (
                tracklet_id not in requested
                or frame_index in existing_frames[tracklet_id]
            ):
                continue
            observations[tracklet_id].append(
                (
                    int(frame_index),
                    np.asarray(box, dtype=float),
                    float(confidence),
                    [
                        np.asarray(other_box, dtype=float)
                        for other_id, other_box, _ in detections
                        if int(other_id) != tracklet_id
                    ],
                )
            )

    requested_frames = tuple(
        sorted(
            {
                frame_index
                for tracklet_observations in observations.values()
                for frame_index, _, _, _ in tracklet_observations
            }
        )
    )
    frames = load_video_frames(Path(source), requested_frames)
    rejected_blur = 0
    rejected_occlusion = 0
    selected: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for tracklet_id, candidates in observations.items():
        need = max(
            0,
            target_samples
            - len(local_pass.appearance_embeddings.get(tracklet_id, [])),
        )
        if not need:
            continue
        scored: list[tuple[float, int, np.ndarray]] = []
        for frame_index, box, confidence, occluders in candidates:
            visibility = float(part_visibility(box, occluders).mean())
            if visibility < min_visibility:
                rejected_occlusion += 1
                continue
            sharpness = _crop_sharpness(frames[frame_index], box)
            if sharpness < min_sharpness:
                rejected_blur += 1
                continue
            frame_area = max(float(local_pass.width * local_pass.height), 1.0)
            box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
            size_quality = min(np.sqrt(box_area / frame_area) / 0.2, 1.0)
            sharpness_quality = min(sharpness / 100.0, 1.0)
            quality = float(
                confidence
                * (0.5 + 0.5 * size_quality)
                * visibility
                * (0.5 + 0.5 * sharpness_quality)
            )
            scored.append((quality, frame_index, box))
        if not scored:
            continue

        all_frames = sorted(frame for _, frame, _ in scored)
        span = all_frames[-1] - all_frames[0] if len(all_frames) > 1 else 0
        min_separation = max(1, span // max(2 * target_samples, 1))
        occupied = set(existing_frames[tracklet_id])
        chosen: list[tuple[int, np.ndarray, float]] = []
        for quality, frame_index, box in sorted(scored, reverse=True):
            if all(abs(frame_index - other) >= min_separation for other in occupied):
                chosen.append((frame_index, box, quality))
                occupied.add(frame_index)
                if len(chosen) == need:
                    break
        if len(chosen) < need:
            used = {frame for frame, _, _ in chosen}
            for quality, frame_index, box in sorted(scored, reverse=True):
                if frame_index in used:
                    continue
                chosen.append((frame_index, box, quality))
                used.add(frame_index)
                if len(chosen) == need:
                    break
        if chosen:
            selected[tracklet_id] = chosen

    encoder = LocalAppearanceEncoder(Path(weights), device=device) if selected else None
    added = 0
    for tracklet_id, chosen in selected.items():
        for frame_index, box, quality in sorted(chosen):
            embedding = encoder.encode(  # type: ignore[union-attr]
                frames[frame_index], np.stack([box])
            )[0]
            local_pass.appearance_embeddings.setdefault(tracklet_id, []).append(
                (frame_index, embedding, quality)
            )
            added += 1
        local_pass.appearance_embeddings[tracklet_id].sort(key=lambda item: item[0])

    return {
        "target_tracklets": sorted(requested),
        "decoded_frames": len(requested_frames),
        "added_samples": added,
        "densified_tracklets": sorted(selected),
        "rejected_blur": rejected_blur,
        "rejected_occlusion": rejected_occlusion,
    }


def propose_anchored_appearance_stitches(
    local_pass: LocalPass,
    *,
    cluster_by_tracklet: dict[int, str],
    verified_clusters: set[str],
    max_gap_s: float = 4.0,
    max_endpoint_distance: float = 1.5,
    max_height_ratio: float = 1.8,
    min_similarity: float = 0.90,
    ambiguity_margin: float = 0.06,
    max_score: float = 0.14,
) -> list[AppearanceStitch]:
    """Attach an unresolved cluster to its unambiguous Astra-verified neighbour.

    Unlike mutual-best tracklet matching, an identity cluster may legitimately
    receive several fragments.  Candidates are therefore ranked on the
    unresolved side only, while the verified cluster acts as a multi-view
    appearance gallery.  Hard merge constraints are still enforced by the
    caller's union-find.
    """
    for label, value in {
        "max_gap_s": max_gap_s,
        "max_endpoint_distance": max_endpoint_distance,
        "max_height_ratio": max_height_ratio,
        "min_similarity": min_similarity,
        "ambiguity_margin": ambiguity_margin,
        "max_score": max_score,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite number")

    candidates = _appearance_candidates(
        local_pass,
        max_gap_s=max_gap_s,
        max_endpoint_distance=max_endpoint_distance,
        max_height_ratio=max_height_ratio,
        min_similarity=min_similarity,
    )
    best_by_cluster_pair: dict[tuple[str, str], AppearanceStitch] = {}
    for candidate in candidates:
        earlier_cluster = cluster_by_tracklet.get(candidate.earlier_id)
        later_cluster = cluster_by_tracklet.get(candidate.later_id)
        if earlier_cluster is None or later_cluster is None or earlier_cluster == later_cluster:
            continue
        earlier_verified = earlier_cluster in verified_clusters
        later_verified = later_cluster in verified_clusters
        if earlier_verified == later_verified:
            continue
        query_cluster, reference_cluster = (
            (later_cluster, earlier_cluster)
            if earlier_verified
            else (earlier_cluster, later_cluster)
        )
        key = (query_cluster, reference_cluster)
        previous = best_by_cluster_pair.get(key)
        if previous is None or (
            candidate.score,
            candidate.earlier_id,
            candidate.later_id,
        ) < (
            previous.score,
            previous.earlier_id,
            previous.later_id,
        ):
            best_by_cluster_pair[key] = candidate

    by_query: dict[str, list[tuple[str, AppearanceStitch]]] = {}
    for (query_cluster, reference_cluster), candidate in best_by_cluster_pair.items():
        if candidate.score <= max_score:
            by_query.setdefault(query_cluster, []).append((reference_cluster, candidate))

    selected: list[AppearanceStitch] = []
    for choices in by_query.values():
        choices.sort(
            key=lambda item: (
                item[1].score,
                item[0],
                item[1].earlier_id,
                item[1].later_id,
            )
        )
        best = choices[0][1]
        if len(choices) > 1 and choices[1][1].score - best.score < ambiguity_margin:
            continue
        selected.append(
            AppearanceStitch(
                **{
                    **best.__dict__,
                    "selection": "astra_anchored_cluster",
                }
            )
        )
    return sorted(
        selected,
        key=lambda item: (item.score, item.earlier_id, item.later_id),
    )
