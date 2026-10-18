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
from flowmetric.tracking import LocalPass

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
            if set(meta["gallery_tracklet_ids"]) != local_ids:
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


def sync_ids(
    lp: LocalPass,
    reids: list[dict[str, Any]],
    *,
    iou_thr: float,
    merge_confidence: float = 0.55,
    min_support: int = 2,
) -> tuple[dict[int, int], dict[int, str], dict[str, Any]]:
    """Return IDs using weighted evidence and explicit merge/abstain decisions."""
    from flowmetric.infrastructure.astra import validate_reid_collection

    if not math.isfinite(merge_confidence) or not 0 <= merge_confidence <= 1:
        raise ValueError("merge_confidence must be a finite number in [0, 1]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support <= 0:
        raise ValueError("min_support must be a positive integer")

    expected_sample_frames = _normalize_legacy_reids_for_sync(lp, reids)
    validate_reid_collection(
        reids,
        expected_sample_frames=expected_sample_frames,
        original_width=lp.width,
        original_height=lp.height,
    )
    validate_reids_against_local(lp, reids)
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
        "memory_strategy": "retrospective_strongest_first_v1",
        "revision_log": memory.revision_log,
        "identity_decisions": identity_decisions,
        "track_identity": track_identity,
    }
    return tid_to_gid, gid_role, stats
