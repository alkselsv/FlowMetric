"""Scene-context pairwise ReID reranker evaluated without mutating identities."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from flowmetric.identity.scenario import (
    assess_scenario_link,
    build_tracklet_lifecycles,
)
from flowmetric.tracking import LocalPass


FEATURE_NAMES = (
    "osnet_similarity",
    "max_crop_similarity",
    "minimum_quality",
    "minimum_support",
    "gap_s",
    "endpoint_distance",
    "height_log_ratio",
    "direction_consistency",
    "same_zone",
    "same_boundary",
    "scenario_support",
    "scenario_compatible",
    "role_match",
    "role_conflict",
    "pose_attribute_similarity",
    "pose_attribute_available",
    "segmented_attribute_similarity",
    "segmented_attribute_available",
)


def _prototype(samples: list[tuple[int, np.ndarray, float]]) -> np.ndarray:
    strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[:8]
    matrix = np.stack([np.asarray(item[1], dtype=float) for item in strongest])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    quality = np.asarray([max(float(item[2]), 0.05) for item in strongest])
    vector = np.average(matrix, axis=0, weights=quality)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _attribute_similarity(
    source: dict[int, list[tuple[int, np.ndarray, float]]], left: int, right: int
) -> tuple[float, float]:
    if not source.get(left) or not source.get(right):
        return 0.0, 0.0
    return float(np.dot(_prototype(source[left]), _prototype(source[right]))), 1.0


def _share_conflicting_astra_batch(
    left: str,
    right: str,
    astra_by_cluster: dict[str, dict[int, int]],
) -> bool:
    left_labels = astra_by_cluster.get(left, {})
    right_labels = astra_by_cluster.get(right, {})
    return any(
        batch in right_labels and identity != right_labels[batch]
        for batch, identity in left_labels.items()
    )


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def run_context_reranker_shadow(
    local_pass: LocalPass,
    *,
    training_cluster_by_tracklet: dict[int, str],
    current_cluster_by_tracklet: dict[int, str],
    astra_by_cluster: dict[str, dict[int, int]],
    cannot_link: set[tuple[int, int]],
    role_by_tracklet: dict[int, str],
    scene_config: dict[str, Any],
    probability_threshold: float = 0.98,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    """Fit on trusted pseudo-labels and audit residual pairs without merging."""
    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be in (0, 1)")
    if not 0 < ambiguity_margin < 1:
        raise ValueError("ambiguity_margin must be in (0, 1)")
    if not isinstance(scene_config, dict):
        raise ValueError("scene_config must be a dictionary")

    embeddings = {
        int(tracklet_id): samples
        for tracklet_id, samples in local_pass.appearance_embeddings.items()
        if samples
    }
    prototypes = {
        tracklet_id: _prototype(samples) for tracklet_id, samples in embeddings.items()
    }
    lifecycles = build_tracklet_lifecycles(local_pass, scene_config)
    cannot = {tuple(sorted((int(left), int(right)))) for left, right in cannot_link}

    def ordered_pair(left: int, right: int) -> tuple[int, int]:
        left_life, right_life = lifecycles[left], lifecycles[right]
        return (right, left) if right_life.last_frame < left_life.first_frame else (left, right)

    def features(left: int, right: int) -> tuple[np.ndarray, dict[str, float]]:
        earlier_id, later_id = ordered_pair(left, right)
        earlier, later = lifecycles[earlier_id], lifecycles[later_id]
        left_samples, right_samples = embeddings[left], embeddings[right]
        osnet = float(np.dot(prototypes[left], prototypes[right]))
        left_matrix = np.stack([np.asarray(item[1], dtype=float) for item in left_samples])
        right_matrix = np.stack([np.asarray(item[1], dtype=float) for item in right_samples])
        left_matrix /= np.maximum(np.linalg.norm(left_matrix, axis=1, keepdims=True), 1e-12)
        right_matrix /= np.maximum(np.linalg.norm(right_matrix, axis=1, keepdims=True), 1e-12)
        max_crop = float(np.max(left_matrix @ right_matrix.T))
        minimum_quality = min(
            float(np.mean([item[2] for item in left_samples])),
            float(np.mean([item[2] for item in right_samples])),
        )
        minimum_support = min(len(left_samples), len(right_samples)) / 8.0
        gap_s = (later.first_frame - earlier.last_frame) / local_pass.fps
        scale = max((earlier.last_height + later.first_height) / 2, 1.0)
        displacement = later.first_foot - earlier.last_foot
        endpoint_distance = float(np.linalg.norm(displacement) / scale)
        height_ratio = max(earlier.last_height, later.first_height) / max(
            min(earlier.last_height, later.first_height), 1.0
        )
        velocity_norm = float(np.linalg.norm(earlier.end_velocity))
        displacement_norm = float(np.linalg.norm(displacement))
        direction = (
            float(np.dot(earlier.end_velocity, displacement) / (velocity_norm * displacement_norm))
            if velocity_norm > 1e-6 and displacement_norm > 1e-6
            else 0.0
        )
        assessment = assess_scenario_link(
            earlier_id, later_id, lifecycles=lifecycles, fps=local_pass.fps
        )
        left_role, right_role = role_by_tracklet.get(left, ""), role_by_tracklet.get(right, "")
        pose_similarity, pose_available = _attribute_similarity(
            local_pass.pose_attribute_embeddings, left, right
        )
        segment_similarity, segment_available = _attribute_similarity(
            local_pass.segmented_attribute_embeddings, left, right
        )
        values = np.asarray(
            [
                osnet,
                max_crop,
                minimum_quality,
                min(minimum_support, 1.0),
                float(np.clip(gap_s / 4.0, -1.0, 2.0)),
                min(endpoint_distance / 2.0, 2.0),
                min(float(np.log(max(height_ratio, 1.0))), 1.0),
                direction,
                float(earlier.last_zone == later.first_zone),
                float(
                    earlier.last_boundary is not None
                    and earlier.last_boundary == later.first_boundary
                ),
                min(len(assessment.signals) / 2.0, 1.0),
                float(assessment.compatible),
                float(bool(left_role) and left_role == right_role),
                float(bool(left_role) and bool(right_role) and left_role != right_role),
                pose_similarity,
                pose_available,
                segment_similarity,
                segment_available,
            ],
            dtype=float,
        )
        return values, dict(zip(FEATURE_NAMES, values.tolist()))

    tracklet_ids = sorted(set(prototypes) & set(lifecycles))
    labelled: list[tuple[int, int, int, np.ndarray]] = []
    for left, right in combinations(tracklet_ids, 2):
        left_cluster = training_cluster_by_tracklet.get(left)
        right_cluster = training_cluster_by_tracklet.get(right)
        if left_cluster is None or right_cluster is None:
            continue
        label: int | None = None
        if left_cluster == right_cluster and astra_by_cluster.get(left_cluster):
            label = 1
        elif (
            tuple(sorted((left, right))) in cannot
            or _share_conflicting_astra_batch(
                left_cluster, right_cluster, astra_by_cluster
            )
        ):
            label = 0
        if label is not None:
            vector, _ = features(left, right)
            labelled.append((left, right, label, vector))

    positives = [item for item in labelled if item[2] == 1]
    negatives = sorted(
        (item for item in labelled if item[2] == 0),
        key=lambda item: (-item[3][0], item[0], item[1]),
    )
    negatives = negatives[: max(200, 10 * len(positives))]
    training = positives + negatives
    result: dict[str, Any] = {
        "enabled": True,
        "mode": "shadow",
        "feature_names": list(FEATURE_NAMES),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "probability_threshold": probability_threshold,
        "ambiguity_margin": ambiguity_margin,
        "trained": False,
        "reason": None,
        "candidates": [],
        "high_confidence_candidates": 0,
    }
    if len(positives) < 8 or len(negatives) < 40:
        result["reason"] = "insufficient_trusted_pairs"
        return result

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "context reranker requires scikit-learn from requirements-experiments.txt"
        ) from exc
    matrix = np.stack([item[3] for item in training])
    labels = np.asarray([item[2] for item in training], dtype=int)
    def make_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                class_weight="balanced",
                max_iter=1000,
                random_state=0,
            ),
        )

    folds = min(5, len(positives), len(negatives))
    cross_validation = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=0
    )
    out_of_fold = cross_val_predict(
        make_model(),
        matrix,
        labels,
        cv=cross_validation,
        method="predict_proba",
    )[:, 1]
    model = make_model()
    model.fit(matrix, labels)
    training_probabilities = model.predict_proba(matrix)[:, 1]
    result["trained"] = True
    result["training_auc"] = _auc(labels, training_probabilities)
    result["cross_validated_auc"] = _auc(labels, out_of_fold)
    coefficients = model.named_steps["logisticregression"].coef_[0]
    result["standardized_coefficients"] = {
        name: float(coefficient)
        for name, coefficient in zip(FEATURE_NAMES, coefficients)
    }
    result["reason"] = "shadow_only"

    raw_candidates: list[dict[str, Any]] = []
    for left, right in combinations(tracklet_ids, 2):
        if current_cluster_by_tracklet.get(left) == current_cluster_by_tracklet.get(right):
            continue
        left_verified = bool(astra_by_cluster.get(training_cluster_by_tracklet.get(left, "")))
        right_verified = bool(astra_by_cluster.get(training_cluster_by_tracklet.get(right, "")))
        if left_verified and right_verified:
            continue
        earlier_id, later_id = ordered_pair(left, right)
        earlier, later = lifecycles[earlier_id], lifecycles[later_id]
        gap_s = (later.first_frame - earlier.last_frame) / local_pass.fps
        if gap_s <= 0 or gap_s > 4.0:
            continue
        scale = max((earlier.last_height + later.first_height) / 2, 1.0)
        endpoint_distance = float(
            np.linalg.norm(later.first_foot - earlier.last_foot) / scale
        )
        height_ratio = max(earlier.last_height, later.first_height) / max(
            min(earlier.last_height, later.first_height), 1.0
        )
        if endpoint_distance > 1.8 or height_ratio > 2.0:
            continue
        vector, feature_values = features(left, right)
        probability = float(model.predict_proba(vector[None, :])[0, 1])
        raw_candidates.append(
            {
                "earlier_tracklet_id": earlier_id,
                "later_tracklet_id": later_id,
                "probability": probability,
                "features": feature_values,
            }
        )

    by_earlier: dict[int, list[dict[str, Any]]] = {}
    by_later: dict[int, list[dict[str, Any]]] = {}
    for candidate in raw_candidates:
        by_earlier.setdefault(candidate["earlier_tracklet_id"], []).append(candidate)
        by_later.setdefault(candidate["later_tracklet_id"], []).append(candidate)
    for choices in (*by_earlier.values(), *by_later.values()):
        choices.sort(key=lambda item: (-item["probability"], item["earlier_tracklet_id"], item["later_tracklet_id"]))

    for candidate in raw_candidates:
        earlier_choices = by_earlier[candidate["earlier_tracklet_id"]]
        later_choices = by_later[candidate["later_tracklet_id"]]
        earlier_margin = (
            candidate["probability"] - earlier_choices[1]["probability"]
            if len(earlier_choices) > 1 and earlier_choices[0] is candidate
            else 1.0 if earlier_choices[0] is candidate else -1.0
        )
        later_margin = (
            candidate["probability"] - later_choices[1]["probability"]
            if len(later_choices) > 1 and later_choices[0] is candidate
            else 1.0 if later_choices[0] is candidate else -1.0
        )
        mutual = earlier_choices[0] is candidate and later_choices[0] is candidate
        high_confidence = (
            candidate["probability"] >= probability_threshold
            and mutual
            and earlier_margin >= ambiguity_margin
            and later_margin >= ambiguity_margin
        )
        candidate.update(
            {
                "mutual_best": mutual,
                "earlier_margin": round(float(earlier_margin), 6),
                "later_margin": round(float(later_margin), 6),
                "decision": "shadow_high_confidence" if high_confidence else "shadow_abstain",
            }
        )
    raw_candidates.sort(
        key=lambda item: (-item["probability"], item["earlier_tracklet_id"], item["later_tracklet_id"])
    )
    result["candidates"] = raw_candidates
    result["high_confidence_candidates"] = sum(
        item["decision"] == "shadow_high_confidence" for item in raw_candidates
    )
    return result
