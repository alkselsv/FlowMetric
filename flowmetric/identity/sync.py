"""Weighted identity association with confidence and hard constraints."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from flowmetric.identity.memory import (
    ConstrainedUnionFind,
    IdentityEvidence,
    RetrospectiveIdentityMemory,
)
from flowmetric.identity.appearance import (
    calibrate_appearance_threshold,
    propose_anchored_appearance_stitches,
    propose_appearance_graph_edges,
    propose_appearance_stitches,
    propose_ensemble_appearance_stitches,
    propose_multiview_appearance_stitches,
    propose_pose_attribute_stitches,
    propose_segmented_attribute_stitches,
    propose_robust_appearance_stitches,
    summarize_robust_appearance,
)
from flowmetric.identity.graph import build_identity_graph_edges
from flowmetric.identity.scenario import (
    assess_scenario_link,
    build_tracklet_lifecycles,
)
from flowmetric.identity.reranker import run_context_reranker_shadow
from flowmetric.identity.motion_bridge import propose_motion_bridges
from flowmetric.identity.stitch import propose_tracklet_stitches
from flowmetric.tracking import LocalPass

ADAPTIVE_SINGLETON_CONFIDENCE = 0.90

# ---------------------------------------------------------------------------
# Sync: local tracker IDs ↔ Astra identities
# ---------------------------------------------------------------------------


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _spatially_distinct(a: np.ndarray, b: np.ndarray) -> bool:
    """Strong evidence that simultaneous boxes belong to different people.

    Trackers sometimes emit two nearly identical/nested boxes for one person.
    Mere coexistence is therefore insufficient for a cannot-link constraint.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    containment = ix * iy / max(min(area_a, area_b), 1.0)
    center_dx = (a[0] + a[2] - b[0] - b[2]) / 2
    center_dy = (a[1] + a[3] - b[1] - b[3]) / 2
    body_height = max(min(a[3] - a[1], b[3] - b[1]), 1.0)
    center_gap = math.hypot(center_dx, center_dy) / body_height
    return containment < 0.1 and center_gap > 0.35


def _local_pair_constraints(
    lp: LocalPass,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Compute local-track coexistence constraints once per immutable local pass."""
    if lp._pair_constraints is not None:
        return lp._pair_constraints

    simultaneous_pairs: set[tuple[int, int]] = set()
    incompatible_pairs: set[tuple[int, int]] = set()
    for dets in lp.dets.values():
        boxes_by_tid: dict[int, tuple[np.ndarray, float]] = {}
        for tid, box, score in dets:
            if tid not in boxes_by_tid or score > boxes_by_tid[tid][1]:
                boxes_by_tid[tid] = (box, score)
        frame_tids = sorted(boxes_by_tid)
        for i in range(len(frame_tids)):
            for j in range(i + 1, len(frame_tids)):
                pair = (frame_tids[i], frame_tids[j])
                simultaneous_pairs.add(pair)
                if _spatially_distinct(
                    boxes_by_tid[pair[0]][0], boxes_by_tid[pair[1]][0]
                ):
                    incompatible_pairs.add(pair)
    lp._pair_constraints = simultaneous_pairs, incompatible_pairs
    return lp._pair_constraints


def _normalize_legacy_reids_for_sync(
    lp: LocalPass, reids: list[dict[str, Any]]
) -> list[int]:
    """Adapt historical experiment responses whose boxes were already scaled.

    Old ``experiments/astra/*.json`` files were saved after scaling boxes to the
    original video, but their metadata still described the resized input. This
    adapter only repairs that known metadata shape. Versioned production-cache
    loading remains strict and rejects these files before reaching ``sync_ids``.
    """
    from flowmetric.infrastructure.astra import AstraResponseError

    inferred_sample_frames: set[int] = set()
    for batch_i, reid in enumerate(reids):
        if not isinstance(reid, dict) or not isinstance(reid.get("_meta"), dict):
            continue  # the common validator below will provide the diagnostic
        meta = reid["_meta"]
        video_frames = meta.get("video_frames")
        if isinstance(video_frames, list):
            inferred_sample_frames.update(
                frame
                for frame in video_frames
                if isinstance(frame, int) and not isinstance(frame, bool)
            )
        if "coordinate_space" in meta:
            continue

        resized_width = meta.get("width")
        resized_height = meta.get("height")
        scale = meta.get("scale")
        if (
            isinstance(resized_width, bool)
            or not isinstance(resized_width, int)
            or resized_width <= 0
            or isinstance(resized_height, bool)
            or not isinstance(resized_height, int)
            or resized_height <= 0
            or isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise AstraResponseError(
                f"legacy Astra batch {batch_i} has invalid resize metadata"
            )
        inferred_width = round(resized_width / float(scale))
        inferred_height = round(resized_height / float(scale))
        if inferred_width != lp.width or inferred_height != lp.height:
            raise AstraResponseError(
                f"legacy Astra batch {batch_i} dimensions infer "
                f"{inferred_width}x{inferred_height}, expected {lp.width}x{lp.height}"
            )
        meta.update(
            {
                "coordinate_space": "original",
                "width": lp.width,
                "height": lp.height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "original_width": lp.width,
                "original_height": lp.height,
                "scale_x": resized_width / lp.width,
                "scale_y": resized_height / lp.height,
                "legacy_metadata_normalized": True,
            }
        )
    return lp.sample_idx or sorted(inferred_sample_frames)


def validate_reids_against_local(
    lp: LocalPass, reids: list[dict[str, Any]]
) -> None:
    """Reject structurally complete batches that omitted every visible person.

    Metadata alone cannot prove that a vision response was complete.  This
    deliberately conservative guard only rejects the unambiguous failure mode:
    a batch contains no Astra detections although the local detector saw people
    on its sampled frames.
    """
    from flowmetric.infrastructure.astra import AstraResponseError

    for batch_i, reid in enumerate(reids):
        meta = reid.get("_meta", {}) if isinstance(reid, dict) else {}
        video_frames = meta.get("video_frames", [])
        if meta.get("protocol") == "tracklet_gallery_v1":
            from flowmetric.identity.gallery import validate_normalized_gallery_metadata

            validate_normalized_gallery_metadata(meta)
            local_ids = {
                tid for frame in video_frames for tid, _, _ in lp.dets.get(frame, [])
            }
            gallery_ids = set(meta["gallery_tracklet_ids"])
            coverage_valid = (
                gallery_ids <= local_ids
                if meta.get("gallery_scope", "full") == "targeted"
                else gallery_ids == local_ids
            )
            if not coverage_valid:
                raise AstraResponseError(
                    f"Astra gallery batch {batch_i} local tracklet coverage mismatch"
                )
            continue
        local_boxes = sum(len(lp.dets.get(frame, [])) for frame in video_frames)
        astra_boxes = sum(
            len(person.get("detections", []))
            for person in reid.get("people", [])
            if isinstance(person, dict)
        ) if isinstance(reid, dict) else 0
        if local_boxes and not astra_boxes:
            raise AstraResponseError(
                f"Astra batch {batch_i} returned no detections for "
                f"{local_boxes} local boxes on sampled frames"
            )


def _reinforced_singleton_evidence(
    reids: list[dict[str, Any]],
    *,
    merge_confidence: float,
    min_support: int,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    """Find one-crop queries independently confirmed through a strong reference.

    A targeted association qualifies only when an earlier full gallery put the
    same query and reference together, both responses are highly confident,
    and the reference satisfies the normal global support threshold in both
    passes.  The returned value is ``(reference_tracklet_id, confirming_batch)``.
    """
    if min_support <= 1:
        return {}
    confidence_threshold = max(
        merge_confidence, ADAPTIVE_SINGLETON_CONFIDENCE
    )
    full_associations: list[tuple[int, dict[str, Any]]] = []
    reinforced: dict[tuple[int, int, int], tuple[int, int]] = {}

    def strongly_supported(association: dict[str, Any], tid: int) -> bool:
        confidence = float(association.get("confidence", 0.0))
        support = int(association.get("support", {}).get(str(tid), 0))
        return (
            support >= min_support
            and confidence >= merge_confidence
            and confidence * support >= min_support * merge_confidence
        )

    for batch, reid in enumerate(reids):
        meta = reid.get("_meta", {})
        associations = meta.get("gallery_associations", [])
        if meta.get("gallery_scope", "full") != "targeted":
            full_associations.extend((batch, association) for association in associations)
            continue
        queries = {int(value) for value in meta.get("targeted_query_ids", [])}
        for association in associations:
            confidence = float(association.get("confidence", 0.0))
            if confidence < confidence_threshold:
                continue
            identity_id = int(association["id"])
            tracklet_ids = {int(value) for value in association.get("tracklet_ids", [])}
            support = association.get("support", {})
            references = sorted(
                tid
                for tid in tracklet_ids - queries
                if strongly_supported(association, tid)
            )
            for query in sorted(tracklet_ids & queries):
                if int(support.get(str(query), 0)) != 1:
                    continue
                confirmations: list[tuple[int, int, int]] = []
                for prior_batch, prior in full_associations:
                    prior_ids = {
                        int(value) for value in prior.get("tracklet_ids", [])
                    }
                    prior_support = prior.get("support", {})
                    prior_confidence = float(prior.get("confidence", 0.0))
                    if (
                        query not in prior_ids
                        or int(prior_support.get(str(query), 0)) != 1
                        or prior_confidence < confidence_threshold
                    ):
                        continue
                    for reference in references:
                        if reference in prior_ids and strongly_supported(prior, reference):
                            confirmations.append(
                                (
                                    int(prior_support.get(str(reference), 0)),
                                    reference,
                                    prior_batch,
                                )
                            )
                if confirmations:
                    _, reference, prior_batch = max(confirmations)
                    reinforced[(query, batch, identity_id)] = (
                        reference,
                        prior_batch,
                    )
    return reinforced


def sync_ids(
    lp: LocalPass,
    reids: list[dict[str, Any]],
    *,
    iou_thr: float,
    merge_confidence: float = 0.55,
    min_support: int = 2,
    local_stitch: bool = False,
    adaptive_singletons: bool = True,
    appearance_stitch: bool = False,
    global_appearance: bool = False,
    anchored_appearance: bool = False,
    appearance_min_similarity: float = 0.82,
    appearance_source: str = "local_osnet_stitch",
    endpoint_appearance: bool = False,
    offline_graph: bool = False,
    scenario_constraints: bool = False,
    scene_config: dict[str, Any] | None = None,
    ensemble_appearance: bool = False,
    multiview_appearance: bool = False,
    robust_appearance: bool = False,
    calibrated_appearance: bool = False,
    pose_attribute_appearance: bool = False,
    segmented_attribute_appearance: bool = False,
    context_reranker_shadow: bool = False,
    motion_bridge_shadow: bool = False,
) -> tuple[dict[int, int], dict[int, str], dict[str, Any]]:
    """Return IDs using weighted evidence and explicit merge/abstain decisions."""
    from flowmetric.infrastructure.astra import validate_reid_collection

    if not math.isfinite(merge_confidence) or not 0 <= merge_confidence <= 1:
        raise ValueError("merge_confidence must be a finite number in [0, 1]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support <= 0:
        raise ValueError("min_support must be a positive integer")
    if not isinstance(adaptive_singletons, bool):
        raise ValueError("adaptive_singletons must be a boolean")
    if not isinstance(appearance_stitch, bool):
        raise ValueError("appearance_stitch must be a boolean")
    if not isinstance(global_appearance, bool):
        raise ValueError("global_appearance must be a boolean")
    if not isinstance(anchored_appearance, bool):
        raise ValueError("anchored_appearance must be a boolean")
    if not isinstance(offline_graph, bool):
        raise ValueError("offline_graph must be a boolean")
    if not isinstance(scenario_constraints, bool):
        raise ValueError("scenario_constraints must be a boolean")
    if scenario_constraints and not offline_graph:
        raise ValueError("scenario_constraints requires offline_graph")
    if scenario_constraints and not isinstance(scene_config, dict):
        raise ValueError("scenario_constraints requires a scene_config")
    if not isinstance(ensemble_appearance, bool):
        raise ValueError("ensemble_appearance must be a boolean")
    if not isinstance(multiview_appearance, bool):
        raise ValueError("multiview_appearance must be a boolean")
    if not isinstance(robust_appearance, bool):
        raise ValueError("robust_appearance must be a boolean")
    if not isinstance(calibrated_appearance, bool):
        raise ValueError("calibrated_appearance must be a boolean")
    if not isinstance(pose_attribute_appearance, bool):
        raise ValueError("pose_attribute_appearance must be a boolean")
    if not isinstance(segmented_attribute_appearance, bool):
        raise ValueError("segmented_attribute_appearance must be a boolean")
    if not isinstance(context_reranker_shadow, bool):
        raise ValueError("context_reranker_shadow must be a boolean")
    if context_reranker_shadow and not isinstance(scene_config, dict):
        raise ValueError("context_reranker_shadow requires a scene_config")
    if not isinstance(motion_bridge_shadow, bool):
        raise ValueError("motion_bridge_shadow must be a boolean")
    if (
        isinstance(appearance_min_similarity, bool)
        or not isinstance(appearance_min_similarity, (int, float))
        or not math.isfinite(float(appearance_min_similarity))
        or not -1 <= appearance_min_similarity <= 1
    ):
        raise ValueError("appearance_min_similarity must be in [-1, 1]")
    if not isinstance(appearance_source, str) or not appearance_source:
        raise ValueError("appearance_source must be a non-empty string")

    expected_sample_frames = _normalize_legacy_reids_for_sync(lp, reids)
    validate_reid_collection(
        reids,
        expected_sample_frames=expected_sample_frames,
        original_width=lp.width,
        original_height=lp.height,
    )
    validate_reids_against_local(lp, reids)
    reinforced_singletons = (
        _reinforced_singleton_evidence(
            reids,
            merge_confidence=merge_confidence,
            min_support=min_support,
        )
        if adaptive_singletons
        else {}
    )
    edge_scores: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    edge_sources: dict[tuple[int, int, int], str] = {}
    model_abstentions: set[tuple[int, int]] = set()
    roles: dict[tuple[int, int], str] = {}
    matched = total_local = 0

    for b, reid in enumerate(reids):
        vframes = reid["_meta"]["video_frames"]
        per_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for person in reid.get("people", []):
            roles[(b, person["id"])] = person["role"]
            for det in person["detections"]:
                fi = det["frame_index"]
                if 0 <= fi < len(vframes):
                    per_frame[vframes[fi]].append((person["id"], np.asarray(det["xyxy"], float)))

        if reid["_meta"].get("protocol") == "tracklet_gallery_v1":
            model_abstentions.update(
                (int(tracklet_id), b)
                for tracklet_id in reid["_meta"].get("abstained_tracklet_ids", [])
            )
            for association in reid["_meta"].get("gallery_associations", []):
                aid = int(association["id"])
                confidence = float(association["confidence"])
                support_by_tid = association.get("support", {})
                for tid in association.get("tracklet_ids", []):
                    support = int(support_by_tid.get(str(tid), 1))
                    key = (int(tid), b, aid)
                    edge_scores[key].extend([confidence] * support)
                    edge_sources[key] = "tracklet_gallery"
                    matched += sum(
                        1
                        for vf in vframes
                        if any(local_tid == tid for local_tid, _, _ in lp.dets.get(vf, []))
                    )
            total_local += sum(len(lp.dets.get(vf, [])) for vf in vframes)
            continue

        for vf in vframes:
            local = lp.dets.get(vf, [])
            astra = per_frame.get(vf, [])
            total_local += len(local)
            if not local or not astra:
                continue
            m = iou_matrix(
                np.stack([d[1] for d in local]),
                np.stack([a[1] for a in astra]),
            )
            # Gate impossible edges *before* assignment.  Otherwise one very
            # strong match plus one sub-threshold match can beat two valid
            # medium-IoU matches, after which thresholding silently loses one.
            cost = np.where(m >= iou_thr, -m, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if m[r, c] >= iou_thr:
                    key = (local[r][0], b, astra[c][0])
                    edge_scores[key].append(float(m[r, c]))
                    edge_sources[key] = "box_iou"
                    matched += 1

    all_tids = {tid for dets in lp.dets.values() for tid, _, _ in dets}
    simultaneous_pairs, incompatible_pairs = _local_pair_constraints(lp)
    observed_tids = {key[0] for key in edge_scores}
    edges_by_tid_batch: dict[tuple[int, int], list[tuple[int, list[float]]]] = defaultdict(list)
    for (tid, batch, aid), scores in edge_scores.items():
        edges_by_tid_batch[(tid, batch)].append((aid, scores))

    evidence_by_batch: dict[int, list[IdentityEvidence]] = defaultdict(list)
    for (tid, b), candidates in sorted(edges_by_tid_batch.items()):
        aid, scores = max(
            candidates,
            key=lambda item: (sum(item[1]), float(np.mean(item[1])), len(item[1])),
        )
        support = len(scores)
        confidence = float(np.average(scores))
        weighted_support = float(sum(scores))
        adaptive_confirmation = reinforced_singletons.get((tid, b, aid))
        evidence_by_batch[b].append(
            IdentityEvidence(
                tracklet_id=tid,
                batch=b,
                identity_id=aid,
                source=edge_sources[(tid, b, aid)],
                support=support,
                confidence=confidence,
                weighted_support=weighted_support,
                role=roles.get((b, aid), "unknown"),
                required_support=1 if adaptive_confirmation else None,
                required_confidence=(
                    max(merge_confidence, ADAPTIVE_SINGLETON_CONFIDENCE)
                    if adaptive_confirmation
                    else None
                ),
                policy=(
                    "reinforced_singleton_v1" if adaptive_confirmation else None
                ),
                reference_tracklet_id=(
                    adaptive_confirmation[0] if adaptive_confirmation else None
                ),
                confirming_batch=(
                    adaptive_confirmation[1] if adaptive_confirmation else None
                ),
            )
        )

    memory = RetrospectiveIdentityMemory(
        all_tids,
        cannot_link=incompatible_pairs,
        merge_confidence=merge_confidence,
        min_support=min_support,
    )
    for batch in sorted(evidence_by_batch):
        memory.observe_batch(evidence_by_batch[batch], batch=batch)
    projection = memory.resolve()
    uf = projection.union_find
    rejected = projection.rejected_reasons
    identity_decisions = projection.decisions

    def cluster_roles(root: str) -> set[str]:
        return {
            roles.get((batch, identity_id), "unknown")
            for batch, identity_id in uf.astra.get(root, {}).items()
        } - {"unknown"}

    reranker_training_clusters = {
        int(tracklet_id): uf.find(f"T{tracklet_id}") for tracklet_id in all_tids
    }
    reranker_training_astra = {
        root: dict(uf.astra.get(root, {}))
        for root in set(reranker_training_clusters.values())
    }
    reranker_roles = {
        tracklet_id: next(iter(cluster_roles(root)), "")
        for tracklet_id, root in reranker_training_clusters.items()
    }

    appearance_stitch_decisions: list[dict[str, Any]] = []
    appearance_stitched_tids: set[int] = set()
    graph_stitched_tids: set[int] = set()
    offline_graph_decisions: list[dict[str, Any]] = []
    scenario_supported_edges = 0
    scenario_rejected_edges = 0
    anchored_appearance_candidates = 0
    anchored_appearance_merges = 0
    effective_appearance_min_similarity = float(appearance_min_similarity)
    appearance_calibration: dict[str, Any] | None = None
    if calibrated_appearance:
        calibration_clusters = {
            int(tracklet_id): uf.find(f"T{tracklet_id}")
            for tracklet_id in all_tids
        }
        calibration_verified = {
            root
            for root in set(calibration_clusters.values())
            if uf.astra.get(root)
        }
        appearance_calibration = calibrate_appearance_threshold(
            lp,
            cluster_by_tracklet=calibration_clusters,
            verified_clusters=calibration_verified,
            cannot_link=incompatible_pairs,
            default_threshold=effective_appearance_min_similarity,
        )
        effective_appearance_min_similarity = float(
            appearance_calibration["threshold"]
        )
    if offline_graph:
        mutual_appearance = (
            propose_ensemble_appearance_stitches(lp)
            if ensemble_appearance
            else propose_segmented_attribute_stitches(lp)
            if segmented_attribute_appearance
            else propose_pose_attribute_stitches(lp)
            if pose_attribute_appearance
            else propose_robust_appearance_stitches(lp)
            if robust_appearance
            else propose_multiview_appearance_stitches(lp)
            if multiview_appearance
            else propose_appearance_stitches(
                lp,
                min_similarity=effective_appearance_min_similarity,
                global_assignment=global_appearance,
                endpoint_rescue=endpoint_appearance,
            )
        )
        supplemental_appearance = propose_appearance_graph_edges(lp)
        scenario_assessments = {}
        scenario_appearance = []
        lifecycles = {}
        if scenario_constraints:
            lifecycles = build_tracklet_lifecycles(lp, scene_config or {})
            for proposal in propose_appearance_graph_edges(
                lp,
                min_similarity=0.90,
                max_score=0.16,
                max_endpoint_distance=1.25,
            ):
                pair = tuple(sorted((proposal.earlier_id, proposal.later_id)))
                assessment = assess_scenario_link(
                    proposal.earlier_id,
                    proposal.later_id,
                    lifecycles=lifecycles,
                    fps=lp.fps,
                )
                scenario_assessments[pair] = assessment
                if assessment.compatible and assessment.supported:
                    scenario_appearance.append(proposal)
        motion_proposals = propose_tracklet_stitches(lp) if local_stitch else []
        graph_edges = build_identity_graph_edges(
            mutual_appearance,
            supplemental_appearance,
            motion_proposals,
            scenario_appearance,
        )
        if scenario_constraints:
            for edge in graph_edges:
                pair = (edge.left_id, edge.right_id)
                scenario_assessments.setdefault(
                    pair,
                    assess_scenario_link(
                        edge.left_id,
                        edge.right_id,
                        lifecycles=lifecycles,
                        fps=lp.fps,
                    ),
                )
            graph_edges.sort(
                key=lambda edge: (
                    -len(edge.sources),
                    -(
                        edge.confidence
                        + scenario_assessments[(edge.left_id, edge.right_id)].confidence_bonus
                    ),
                    edge.left_id,
                    edge.right_id,
                )
            )
        for edge in graph_edges:
            left_node = f"T{edge.left_id}"
            right_node = f"T{edge.right_id}"
            left_root, right_root = uf.find(left_node), uf.find(right_node)
            decision = {**edge.as_dict(), "source": "offline_identity_graph"}
            scenario_assessment = scenario_assessments.get(
                (edge.left_id, edge.right_id)
            )
            if scenario_assessment is not None:
                decision["scenario"] = scenario_assessment.as_dict()
            if left_root == right_root:
                decision.update({"decision": "already_merged"})
            else:
                left_roles = cluster_roles(left_root)
                right_roles = cluster_roles(right_root)
                both_verified = bool(uf.astra.get(left_root)) and bool(
                    uf.astra.get(right_root)
                )
                strict_cross_verified = (
                    "appearance_strict" in edge.sources
                    and edge.appearance_similarity is not None
                    and edge.appearance_similarity >= 0.94
                )
                if scenario_assessment is not None and not scenario_assessment.compatible:
                    decision.update(
                        {"decision": "reject", "reason": scenario_assessment.reason}
                    )
                    scenario_rejected_edges += 1
                elif left_roles and right_roles and left_roles.isdisjoint(right_roles):
                    decision.update({"decision": "reject", "reason": "role_conflict"})
                elif both_verified and "motion" not in edge.sources and not strict_cross_verified:
                    decision.update(
                        {"decision": "reject", "reason": "both_astra_verified"}
                    )
                elif uf.union(left_node, right_node):
                    decision.update({"decision": "merge"})
                    graph_stitched_tids.update((edge.left_id, edge.right_id))
                    if scenario_assessment is not None and scenario_assessment.supported:
                        scenario_supported_edges += 1
                else:
                    decision.update(
                        {"decision": "reject", "reason": uf.last_rejection or "unknown"}
                    )
            offline_graph_decisions.append(decision)

    if appearance_stitch and not offline_graph:
        proposals = (
            propose_ensemble_appearance_stitches(lp)
            if ensemble_appearance
            else propose_segmented_attribute_stitches(lp)
            if segmented_attribute_appearance
            else propose_pose_attribute_stitches(lp)
            if pose_attribute_appearance
            else propose_robust_appearance_stitches(lp)
            if robust_appearance
            else propose_multiview_appearance_stitches(lp)
            if multiview_appearance
            else propose_appearance_stitches(
                lp,
                min_similarity=effective_appearance_min_similarity,
                global_assignment=global_appearance,
                endpoint_rescue=endpoint_appearance,
            )
        )
        for proposal in proposals:
            left_node = f"T{proposal.earlier_id}"
            right_node = f"T{proposal.later_id}"
            left_root, right_root = uf.find(left_node), uf.find(right_node)
            decision = {**proposal.as_dict(), "source": appearance_source}
            if left_root == right_root:
                decision.update({"decision": "already_merged"})
            elif uf.astra.get(left_root) and uf.astra.get(right_root):
                decision.update(
                    {"decision": "reject", "reason": "both_astra_verified"}
                )
            elif (
                cluster_roles(left_root)
                and cluster_roles(right_root)
                and cluster_roles(left_root).isdisjoint(cluster_roles(right_root))
            ):
                decision.update({"decision": "reject", "reason": "role_conflict"})
            elif uf.union(left_node, right_node):
                decision.update({"decision": "merge"})
                appearance_stitched_tids.update(
                    (proposal.earlier_id, proposal.later_id)
                )
            else:
                decision.update(
                    {"decision": "reject", "reason": uf.last_rejection or "unknown"}
                )
            appearance_stitch_decisions.append(decision)

        if anchored_appearance:
            cluster_by_tracklet = {
                int(tid): uf.find(f"T{tid}") for tid in all_tids
            }
            verified_clusters = {
                root for root in set(cluster_by_tracklet.values()) if uf.astra.get(root)
            }
            anchored_proposals = propose_anchored_appearance_stitches(
                lp,
                cluster_by_tracklet=cluster_by_tracklet,
                verified_clusters=verified_clusters,
            )
            anchored_appearance_candidates = len(anchored_proposals)
            for proposal in anchored_proposals:
                left_node = f"T{proposal.earlier_id}"
                right_node = f"T{proposal.later_id}"
                left_root, right_root = uf.find(left_node), uf.find(right_node)
                decision = {
                    **proposal.as_dict(),
                    "source": "local_osnet_astra_anchor",
                }
                if left_root == right_root:
                    decision.update({"decision": "already_merged"})
                elif uf.astra.get(left_root) and uf.astra.get(right_root):
                    decision.update(
                        {"decision": "reject", "reason": "both_astra_verified"}
                    )
                elif (
                    cluster_roles(left_root)
                    and cluster_roles(right_root)
                    and cluster_roles(left_root).isdisjoint(cluster_roles(right_root))
                ):
                    decision.update({"decision": "reject", "reason": "role_conflict"})
                elif uf.union(left_node, right_node):
                    decision.update({"decision": "merge"})
                    appearance_stitched_tids.update(
                        (proposal.earlier_id, proposal.later_id)
                    )
                    anchored_appearance_merges += 1
                else:
                    decision.update(
                        {
                            "decision": "reject",
                            "reason": uf.last_rejection or "unknown",
                        }
                    )
                appearance_stitch_decisions.append(decision)

    local_stitch_decisions: list[dict[str, Any]] = []
    locally_stitched_tids: set[int] = set()
    if local_stitch and not offline_graph:
        for proposal in propose_tracklet_stitches(lp):
            left_node = f"T{proposal.earlier_id}"
            right_node = f"T{proposal.later_id}"
            left_root, right_root = uf.find(left_node), uf.find(right_node)
            decision = {**proposal.as_dict(), "source": "local_motion_stitch"}
            if left_root == right_root:
                decision.update({"decision": "already_merged"})
            else:
                left_roles = cluster_roles(left_root)
                right_roles = cluster_roles(right_root)
                if left_roles and right_roles and left_roles.isdisjoint(right_roles):
                    decision.update({"decision": "reject", "reason": "role_conflict"})
                elif uf.union(left_node, right_node):
                    decision.update({"decision": "merge"})
                    locally_stitched_tids.update(
                        (proposal.earlier_id, proposal.later_id)
                    )
                else:
                    decision.update(
                        {
                            "decision": "reject",
                            "reason": uf.last_rejection or "unknown",
                        }
                    )
            local_stitch_decisions.append(decision)

    context_reranker = None
    if context_reranker_shadow:
        context_reranker = run_context_reranker_shadow(
            lp,
            training_cluster_by_tracklet=reranker_training_clusters,
            current_cluster_by_tracklet={
                int(tracklet_id): uf.find(f"T{tracklet_id}")
                for tracklet_id in all_tids
            },
            astra_by_cluster=reranker_training_astra,
            cannot_link=incompatible_pairs,
            role_by_tracklet=reranker_roles,
            scene_config=scene_config or {},
        )

    motion_bridge_decisions: list[dict[str, Any]] = []
    if motion_bridge_shadow:
        motion_clusters = {
            int(tracklet_id): uf.find(f"T{tracklet_id}") for tracklet_id in all_tids
        }
        motion_blocked = set(incompatible_pairs)
        motion_tracklet_ids = sorted(all_tids)
        for left_index, left_id in enumerate(motion_tracklet_ids):
            left_root = motion_clusters[int(left_id)]
            for right_id in motion_tracklet_ids[left_index + 1 :]:
                right_root = motion_clusters[int(right_id)]
                left_roles = cluster_roles(left_root)
                right_roles = cluster_roles(right_root)
                if (
                    (left_roles and right_roles and left_roles.isdisjoint(right_roles))
                    or (uf.astra.get(left_root) and uf.astra.get(right_root))
                ):
                    motion_blocked.add(tuple(sorted((int(left_id), int(right_id)))))
        for proposal in propose_motion_bridges(
            lp,
            cluster_by_tracklet=motion_clusters,
            blocked_pairs=motion_blocked,
        ):
            left_node = f"T{proposal.earlier_id}"
            right_node = f"T{proposal.later_id}"
            left_root, right_root = uf.find(left_node), uf.find(right_node)
            decision = {
                **proposal.as_dict(),
                "source": "bidirectional_motion_appearance_bridge",
                "mode": "shadow",
            }
            if left_root == right_root:
                decision.update({"decision": "already_merged"})
            else:
                decision.update({"decision": "candidate"})
            motion_bridge_decisions.append(decision)

    evidence_by_key = {
        evidence.key: evidence
        for batch_evidence in evidence_by_batch.values()
        for evidence in batch_evidence
    }
    accepted_confidence: dict[int, list[float]] = defaultdict(list)
    accepted_support: Counter[int] = Counter()
    for decision in identity_decisions:
        if decision["decision"] != "merge":
            continue
        key = (
            int(decision["tracklet_id"]),
            int(decision["batch"]),
            int(decision["identity_id"]),
        )
        evidence = evidence_by_key[key]
        accepted_confidence[evidence.tracklet_id].append(evidence.confidence)
        accepted_support[evidence.tracklet_id] += evidence.support

    decided_tid_batches = {
        (int(decision["tracklet_id"]), int(decision["batch"]))
        for decision in identity_decisions
    }
    for tid, batch in sorted(model_abstentions - decided_tid_batches):
        identity_decisions.append(
            {
                "tracklet_id": tid,
                "batch": batch,
                "identity_id": None,
                "source": "tracklet_gallery",
                "support": 0,
                "confidence": None,
                "weighted_support": 0.0,
                "decision": "abstain",
                "reason": "model_abstained",
            }
        )
    identity_decisions.sort(key=lambda item: (item["batch"], item["tracklet_id"]))
    rejected_tids = {
        int(decision["tracklet_id"])
        for decision in identity_decisions
        if decision["decision"] == "reject"
    }

    # Stable display numbers ordered by first appearance
    first_seen: dict[str, int] = {}
    for fi in sorted(lp.dets):
        for tid, _, _ in lp.dets[fi]:
            root = uf.find(f"T{tid}")
            first_seen.setdefault(root, fi)
    root_to_gid = {r: i + 1 for i, r in enumerate(sorted(first_seen, key=first_seen.get))}

    tid_to_gid = {tid: root_to_gid[uf.find(f"T{tid}")] for tid in all_tids}
    gid_role: dict[int, str] = {}
    for root, gid in root_to_gid.items():
        rvotes = Counter(roles.get((b, aid), "unknown") for b, aid in uf.astra.get(root, {}).items())
        rvotes.pop("unknown", None)
        gid_role[gid] = rvotes.most_common(1)[0][0] if rvotes else ""

    track_identity = {
        str(tid): {
            "status": (
                "merged"
                if accepted_confidence.get(tid)
                else "graph_stitched"
                if tid in graph_stitched_tids
                else "appearance_stitched"
                if tid in appearance_stitched_tids
                else "local_stitched"
                if tid in locally_stitched_tids
                else "rejected"
                if tid in rejected_tids
                else "abstained"
                if tid in observed_tids or any(item[0] == tid for item in model_abstentions)
                else "local_only"
            ),
            "confidence": (
                round(float(np.mean(accepted_confidence[tid])), 4)
                if accepted_confidence.get(tid)
                else None
            ),
            "accepted_confirmations": len(accepted_confidence.get(tid, [])),
            "evidence_support": accepted_support.get(tid, 0),
        }
        for tid in sorted(all_tids)
    }

    stats = {
        "local_tracks": len(all_tids),
        "global_ids": len(root_to_gid),
        "astra_verified": sum(1 for r in root_to_gid if uf.astra.get(r)),
        "matched_boxes": matched,
        "local_boxes_on_samples": total_local,
        "rejected_merges": sum(rejected.values()),
        "rejected_merge_reasons": dict(rejected),
        "simultaneous_local_pairs": len(simultaneous_pairs),
        "spatial_cannot_link_pairs": len(incompatible_pairs),
        "merge_confidence": merge_confidence,
        "min_support": min_support,
        "adaptive_singletons_enabled": adaptive_singletons,
        "adaptive_singleton_confidence": ADAPTIVE_SINGLETON_CONFIDENCE,
        "adaptive_singleton_candidates": len(reinforced_singletons),
        "adaptive_singleton_merges": sum(
            item.get("policy") == "reinforced_singleton_v1"
            and item["decision"] == "merge"
            for item in identity_decisions
        ),
        "appearance_stitch_enabled": appearance_stitch,
        "appearance_min_similarity": effective_appearance_min_similarity,
        "appearance_default_min_similarity": appearance_min_similarity,
        "calibrated_appearance_enabled": calibrated_appearance,
        "appearance_calibration": appearance_calibration,
        "appearance_source": appearance_source,
        "endpoint_appearance_enabled": endpoint_appearance,
        "global_appearance_enabled": global_appearance,
        "anchored_appearance_enabled": anchored_appearance,
        "appearance_stitch_decisions": appearance_stitch_decisions,
        "appearance_stitch_candidates": len(appearance_stitch_decisions),
        "appearance_stitch_merges": sum(
            item["decision"] == "merge" for item in appearance_stitch_decisions
        ),
        "endpoint_appearance_candidates": sum(
            item.get("selection") == "endpoint_temporal_rescue"
            for item in appearance_stitch_decisions
        ),
        "endpoint_appearance_merges": sum(
            item.get("selection") == "endpoint_temporal_rescue"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "global_appearance_candidates": sum(
            item.get("selection") == "global_supplemental_assignment"
            for item in appearance_stitch_decisions
        ),
        "global_appearance_merges": sum(
            item.get("selection") == "global_supplemental_assignment"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "anchored_appearance_candidates": anchored_appearance_candidates,
        "anchored_appearance_merges": anchored_appearance_merges,
        "ensemble_appearance_enabled": ensemble_appearance,
        "ensemble_consensus_candidates": sum(
            item.get("selection") == "full_part_consensus"
            for item in appearance_stitch_decisions
        ),
        "ensemble_consensus_merges": sum(
            item.get("selection") == "full_part_consensus"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "multiview_appearance_enabled": multiview_appearance,
        "multiview_consensus_candidates": sum(
            item.get("selection") == "multiview_consensus"
            for item in appearance_stitch_decisions
        ),
        "multiview_consensus_merges": sum(
            item.get("selection") == "multiview_consensus"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "robust_appearance_enabled": robust_appearance,
        "robust_consensus_candidates": sum(
            item.get("selection") == "robust_temporal_consensus"
            for item in appearance_stitch_decisions
        ),
        "robust_consensus_merges": sum(
            item.get("selection") == "robust_temporal_consensus"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "robust_aggregation": (
            summarize_robust_appearance(lp) if robust_appearance else None
        ),
        "pose_attribute_appearance_enabled": pose_attribute_appearance,
        "pose_attribute_consensus_candidates": sum(
            item.get("selection") == "pose_attribute_consensus"
            for item in appearance_stitch_decisions
        ),
        "pose_attribute_consensus_merges": sum(
            item.get("selection") == "pose_attribute_consensus"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "segmented_attribute_appearance_enabled": segmented_attribute_appearance,
        "segmented_attribute_consensus_candidates": sum(
            item.get("selection") == "segmented_attribute_consensus"
            for item in appearance_stitch_decisions
        ),
        "segmented_attribute_consensus_merges": sum(
            item.get("selection") == "segmented_attribute_consensus"
            and item["decision"] == "merge"
            for item in appearance_stitch_decisions
        ),
        "context_reranker_shadow_enabled": context_reranker_shadow,
        "context_reranker": context_reranker,
        "motion_bridge_shadow_enabled": motion_bridge_shadow,
        "motion_bridge_decisions": motion_bridge_decisions,
        "motion_bridge_candidates": sum(
            item["decision"] == "candidate" for item in motion_bridge_decisions
        ),
        "offline_graph_enabled": offline_graph,
        "offline_graph_nodes": len(all_tids),
        "offline_graph_decisions": offline_graph_decisions,
        "offline_graph_edges": len(offline_graph_decisions),
        "offline_graph_merges": sum(
            item["decision"] == "merge" for item in offline_graph_decisions
        ),
        "scenario_constraints_enabled": scenario_constraints,
        "scenario_supported_edges": sum(
            item.get("scenario", {}).get("supported", False)
            for item in offline_graph_decisions
        ),
        "scenario_conflict_edges": sum(
            not item.get("scenario", {}).get("compatible", True)
            for item in offline_graph_decisions
        ),
        "scenario_supported_merges": scenario_supported_edges,
        "scenario_rejected_edges": scenario_rejected_edges,
        "memory_strategy": "retrospective_strongest_first_v1",
        "revision_log": memory.revision_log,
        "identity_decisions": identity_decisions,
        "local_stitch_enabled": local_stitch,
        "local_stitch_decisions": local_stitch_decisions,
        "local_stitch_candidates": len(local_stitch_decisions),
        "local_stitch_merges": sum(
            item["decision"] == "merge" for item in local_stitch_decisions
        ),
        "track_identity": track_identity,
        "targeted_batches": sum(
            reid.get("_meta", {}).get("gallery_scope") == "targeted"
            for reid in reids
            if isinstance(reid, dict)
        ),
        "targeted_query_tracklets": sum(
            len(reid.get("_meta", {}).get("targeted_query_ids", []))
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
        "targeted_reference_tracklets": sum(
            len(
                set(reid.get("_meta", {}).get("gallery_tracklet_ids", []))
                - set(reid.get("_meta", {}).get("targeted_query_ids", []))
            )
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
        "targeted_shortlist_queries": sum(
            len(
                reid.get("_meta", {}).get(
                    "targeted_candidate_shortlists", {}
                )
            )
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
        "targeted_shortlist_options": sum(
            sum(
                len(references)
                for references in reid.get("_meta", {})
                .get("targeted_candidate_shortlists", {})
                .values()
            )
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
        "targeted_shortlist_rejections": sum(
            reid.get("_meta", {}).get("targeted_shortlist_rejections", 0)
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
        "targeted_resampled_frames": sum(
            len(reid.get("_meta", {}).get("targeted_resampled_frames", []))
            for reid in reids
            if isinstance(reid, dict)
            and reid.get("_meta", {}).get("gallery_scope") == "targeted"
        ),
    }
    stats["track_identity_status_counts"] = dict(
        Counter(item["status"] for item in track_identity.values())
    )
    tracklet_frames: dict[int, list[int]] = defaultdict(list)
    for frame, detections in lp.dets.items():
        for tid, _, _ in detections:
            tracklet_frames[int(tid)].append(int(frame))
    duration_bins = Counter()
    for frames in tracklet_frames.values():
        duration = (max(frames) - min(frames)) / lp.fps
        duration_bins[
            "lt_0.5s"
            if duration < 0.5
            else "0.5_to_1s"
            if duration < 1.0
            else "1_to_3s"
            if duration < 3.0
            else "ge_3s"
        ] += 1
    stats["local_tracklet_duration_bins"] = {
        label: duration_bins.get(label, 0)
        for label in ("lt_0.5s", "0.5_to_1s", "1_to_3s", "ge_3s")
    }
    return tid_to_gid, gid_role, stats
