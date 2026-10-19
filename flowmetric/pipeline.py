#!/usr/bin/env python3
"""FlowMetric hybrid people tracking and CLI orchestration.

~95% local: YOLO + BoT-SORT/ByteTrack on every frame (boxes, frame-to-frame IDs).
~5% GPT-6 Astra: numbered local-tracklet crop galleries (default 1 fps sampling,
16-frame batches) for ReID + role, via VseLLM or OpenAI. Local tracks are synced
through weighted multi-crop evidence, explicit abstention, and constrained union-find.
Service events are derived
from the trajectories (events.py). Optional Astra or fully local pose/object stages
verify only ambiguous service candidates without changing IDs, boxes, or interval
boundaries. Results are scored against benchmark/ground_truth.json (metrics.py).
Output: annotated video, tracks.json, events.json, benchmark_report.json.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowmetric.config import IdentityConfig, PipelineConfig, ServiceConfig, TrackingConfig
from flowmetric.identity.sync import iou_matrix, sync_ids, validate_reids_against_local
from flowmetric.output import build_tracks, event_text as _event_text, render
from flowmetric.paths import PROJECT_ROOT as ROOT, weights_path
from flowmetric.tracking import LocalPass, local_track

# ---------------------------------------------------------------------------
# Pass 2: GPT-6 Astra ReID on sampled frames
# ---------------------------------------------------------------------------


def make_batches(n: int, batch_size: int, overlap: int = 1) -> list[list[int]]:
    """Index batches over sampled frames with overlap so IDs chain across batches."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= batch_size:
        raise ValueError("overlap must be smaller than batch_size")
    if n == 0:
        return []
    batches: list[list[int]] = []
    start = 0
    stride = max(1, batch_size - overlap)
    while start < n:
        batches.append(list(range(start, min(start + batch_size, n))))
        if start + batch_size >= n:
            break
        start += stride
    return batches


def validate_astra_options(
    *,
    sample_fps: float,
    batch_size: int,
    workers: int,
    match_iou: float,
    batch_overlap: int = 1,
    targeted_pass: bool = False,
    targeted_references_per_identity: int = 1,
    targeted_resampling: bool = False,
    targeted_resample_budget: int = 16,
    targeted_shortlist: bool = False,
    targeted_shortlist_size: int = 5,
) -> None:
    """Validate Astra orchestration options before decoding or model work."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be a positive finite number")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    make_batches(0, batch_size, overlap=batch_overlap)
    if not isinstance(targeted_pass, bool):
        raise ValueError("targeted_pass must be a boolean")
    if not isinstance(targeted_resampling, bool):
        raise ValueError("targeted_resampling must be a boolean")
    if not isinstance(targeted_shortlist, bool):
        raise ValueError("targeted_shortlist must be a boolean")
    if (
        isinstance(targeted_resample_budget, bool)
        or not isinstance(targeted_resample_budget, int)
        or targeted_resample_budget < 0
    ):
        raise ValueError("targeted_resample_budget must be a non-negative integer")
    if (
        isinstance(targeted_shortlist_size, bool)
        or not isinstance(targeted_shortlist_size, int)
        or not 3 <= targeted_shortlist_size <= 5
    ):
        raise ValueError("targeted_shortlist_size must be in [3, 5]")
    if (
        isinstance(targeted_references_per_identity, bool)
        or not isinstance(targeted_references_per_identity, int)
        or not 1 <= targeted_references_per_identity <= 3
    ):
        raise ValueError("targeted_references_per_identity must be in [1, 3]")
    if not math.isfinite(match_iou) or not 0 <= match_iou <= 1:
        raise ValueError("match_iou must be a finite number in [0, 1]")


def astra_pass(
    lp: LocalPass,
    *,
    source: Path | None = None,
    model: str | None,
    effort: str,
    batch_size: int,
    batch_overlap: int,
    workers: int,
    targeted_pass: bool,
    merge_confidence: float,
    min_support: int,
    targeted_references_per_identity: int = 1,
    targeted_resampling: bool = False,
    targeted_resample_budget: int = 16,
    targeted_shortlist: bool = False,
    targeted_shortlist_size: int = 5,
) -> list[dict[str, Any]]:
    from flowmetric.infrastructure.astra import (
        AstraResponseError,
        make_client,
        validate_reid_batches,
    )
    from flowmetric.identity.gallery import (
        astra_reid_tracklet_gallery,
        build_tracklet_gallery,
        normalize_gallery_response,
        targeted_gallery_inventory,
        build_targeted_candidate_shortlists,
    )

    client, default_model = make_client()
    model = model or default_model
    batches = make_batches(len(lp.sample_frames), batch_size, overlap=batch_overlap)
    expected_batches = [[lp.sample_idx[i] for i in batch] for batch in batches]
    print(
        f"  astra: {len(lp.sample_frames)} sampled frames → {len(batches)} call(s) "
        f"to {model} @ {client.base_url}"
    )

    def run(b: list[int]) -> dict[str, Any]:
        frames = [lp.sample_frames[i] for i in b]
        video_frames = [lp.sample_idx[i] for i in b]
        if not any(lp.dets.get(frame) for frame in video_frames):
            return normalize_gallery_response(
                {"identities": [], "abstained_tracklet_ids": []},
                tracklets=[],
                video_frames=video_frames,
                detections=lp.dets,
                original_width=lp.width,
                original_height=lp.height,
                requested_model=model,
                actual_model=None,
            )
        gallery, tracklets = build_tracklet_gallery(
            frames,
            video_frames,
            lp.dets,
        )
        for attempt in range(2):
            try:
                response, actual_model = astra_reid_tracklet_gallery(
                    gallery,
                    tracklets,
                    model=model,
                    reasoning_effort=effort,
                    client=client,
                )
                break
            except AstraResponseError:
                if attempt == 1:
                    raise
        return normalize_gallery_response(
            response,
            tracklets=tracklets,
            video_frames=video_frames,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model=model,
            actual_model=actual_model,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        reids = list(pool.map(run, batches))
    validate_reid_batches(
        reids,
        expected_batches=expected_batches,
        original_width=lp.width,
        original_height=lp.height,
    )
    validate_reids_against_local(lp, reids)

    if targeted_pass:
        frames_by_tid: dict[int, list[int]] = defaultdict(list)
        for frame, detections in lp.dets.items():
            for tracklet_id, _, _ in detections:
                frames_by_tid[int(tracklet_id)].append(int(frame))
        tracklet_midpoints = {
            tracklet_id: (min(frames) + max(frames)) / 2
            for tracklet_id, frames in frames_by_tid.items()
        }
        query_ids, reference_ids = targeted_gallery_inventory(
            reids,
            merge_confidence=merge_confidence,
            min_support=min_support,
            representatives_per_group=targeted_references_per_identity,
            tracklet_midpoints=tracklet_midpoints,
        )
        if query_ids and reference_ids:
            candidate_shortlists: dict[int, list[int]] = {}
            if targeted_shortlist:
                candidate_shortlists = build_targeted_candidate_shortlists(
                    lp,
                    reids,
                    query_tracklet_ids=query_ids,
                    reference_tracklet_ids=reference_ids,
                    limit=targeted_shortlist_size,
                )
                shortlisted_references = sorted(
                    {
                        reference
                        for references in candidate_shortlists.values()
                        for reference in references
                    }
                )
                reference_ids = shortlisted_references
            if not reference_ids:
                return reids
            targeted_frame_indexes = list(lp.sample_idx)
            targeted_frames = list(lp.sample_frames)
            resampled_frames: list[int] = []
            if targeted_resampling:
                if source is None:
                    raise ValueError("source is required for targeted resampling")
                from flowmetric.identity.sampling import select_targeted_resample_frames
                from flowmetric.media.video import load_video_frames

                resampled_frames = select_targeted_resample_frames(
                    lp,
                    query_tracklet_ids=query_ids,
                    base_frames=lp.sample_idx,
                    additional_count=targeted_resample_budget,
                )
                if resampled_frames:
                    decoded = load_video_frames(source, tuple(resampled_frames))
                    frames_by_index = dict(zip(lp.sample_idx, lp.sample_frames))
                    frames_by_index.update(decoded)
                    targeted_frame_indexes = sorted(frames_by_index)
                    targeted_frames = [
                        frames_by_index[index] for index in targeted_frame_indexes
                    ]
            selected_ids = set(query_ids) | set(reference_ids)
            targeted_detections = {
                frame: [
                    detection
                    for detection in lp.dets.get(frame, [])
                    if int(detection[0]) in selected_ids
                ]
                for frame in targeted_frame_indexes
            }
            gallery, tracklets = build_tracklet_gallery(
                targeted_frames,
                targeted_frame_indexes,
                targeted_detections,
            )
            response: dict[str, Any]
            actual_model: str | None
            for attempt in range(2):
                try:
                    response, actual_model = astra_reid_tracklet_gallery(
                        gallery,
                        tracklets,
                        model=model,
                        reasoning_effort=effort,
                        client=client,
                        target_tracklet_ids=query_ids,
                        candidate_shortlists=(
                            candidate_shortlists if targeted_shortlist else None
                        ),
                        repair_shortlist_violations=attempt == 1,
                    )
                    break
                except AstraResponseError:
                    if attempt == 1:
                        raise
            targeted_reid = normalize_gallery_response(
                response,
                tracklets=tracklets,
                video_frames=targeted_frame_indexes,
                detections=targeted_detections,
                original_width=lp.width,
                original_height=lp.height,
                requested_model=model,
                actual_model=actual_model,
                gallery_scope="targeted",
                targeted_query_ids=query_ids,
                targeted_candidate_shortlists=candidate_shortlists,
            )
            targeted_reid["_meta"]["targeted_resampled_frames"] = resampled_frames
            reids.append(targeted_reid)
            print(
                f"  astra targeted: {len(query_ids)} abstained tracklets + "
                f"{len(reference_ids)} references + {len(resampled_frames)} "
                f"resampled frames + shortlist={targeted_shortlist} → 1 call"
            )
            validate_reid_batches(
                reids,
                expected_batches=expected_batches,
                original_width=lp.width,
                original_height=lp.height,
            )
            validate_reids_against_local(lp, reids)
    return reids


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_hybrid(
    source: Path,
    dest: Path,
    *,
    model_name: str = "yolo11s.pt",
    conf: float = 0.1,
    iou: float = 0.5,
    imgsz: int = 1280,
    tracker: str = str(ROOT / "config" / "botsort_recall.yaml"),
    device: str = "auto",
    camera: Path | None = None,
    tables: Path | None = None,
    gt: Path | None = None,
    use_astra: bool = True,
    astra_model: str | None = None,
    astra_effort: str = "low",
    sample_fps: float = 1.0,
    batch_size: int = 16,
    batch_overlap: int = 4,
    workers: int = 4,
    match_iou: float = 0.3,
    reid_merge_confidence: float = 0.55,
    reid_min_support: int = 2,
    reid_local_stitch: bool = True,
    reid_targeted_pass: bool = True,
    reid_targeted_references: int = 1,
    reid_targeted_resampling: bool = False,
    reid_targeted_resample_budget: int = 16,
    reid_targeted_shortlist: bool = False,
    reid_targeted_shortlist_size: int = 5,
    reid_adaptive_singletons: bool = True,
    reid_appearance_stitch: bool = False,
    reid_global_appearance: bool = False,
    reid_anchored_appearance: bool = False,
    reid_nested_suppression: bool = False,
    reid_endpoint_appearance: bool = False,
    reid_part_aware_appearance: bool = False,
    reid_in_tracker: bool = False,
    reid_offline_graph: bool = False,
    reid_scenario_constraints: bool = False,
    reid_ensemble_appearance: bool = False,
    reid_multiview_appearance: bool = False,
    reid_dense_multiview_appearance: bool = False,
    reid_robust_appearance: bool = False,
    reid_calibrated_appearance: bool = False,
    reid_pose_attribute_appearance: bool = False,
    reid_segmented_attribute_appearance: bool = False,
    reid_context_reranker_shadow: bool = False,
    reid_motion_bridge_shadow: bool = False,
    reid_appearance_backend: str = "osnet",
    reid_appearance_model: Path = ROOT / "weights" / "osnet_x0_25_msmt17.pt",
    reid_pose_model: str = "yolo11n-pose.pt",
    reid_sinet_model: Path = ROOT / "weights" / "sinet_mars.pth.tar",
    reid_adaptive_sampling: bool = True,
    astra_cache: Path | None = None,
    service_verifier: str = "off",
    service_verifier_model: str | None = None,
    service_verifier_effort: str = "low",
    service_verifier_frames: int = 6,
    service_verifier_cache: Path | None = None,
    local_pose_model: str = "yolo11n-pose.pt",
    local_object_model: str = "yolo11n.pt",
    local_service_model: Path = ROOT / "config" / "service_mlp_bootstrap.json",
    local_service_sample_fps: float = 4.0,
    local_service_threshold: float | None = None,
    local_service_imgsz: int = 640,
    local_service_device: str | None = None,
) -> dict[str, Any]:
    from flowmetric.device import resolve_device
    from flowmetric.events import extract_events
    from scene import load_scene_config

    compute_device = resolve_device(device)
    service_compute_device = compute_device
    reid_offline_graph = reid_offline_graph or reid_scenario_constraints
    reid_multiview_appearance = (
        reid_multiview_appearance or reid_dense_multiview_appearance
    )
    reid_appearance_stitch = (
        reid_appearance_stitch
        or reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_part_aware_appearance
        or reid_offline_graph
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_calibrated_appearance
        or reid_pose_attribute_appearance
        or reid_segmented_attribute_appearance
        or reid_context_reranker_shadow
        or reid_motion_bridge_shadow
    )
    if reid_appearance_backend not in {"osnet", "sinet"}:
        raise ValueError("reid_appearance_backend must be 'osnet' or 'sinet'")
    if reid_appearance_backend == "sinet" and (
        reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_part_aware_appearance
        or reid_offline_graph
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_calibrated_appearance
        or reid_pose_attribute_appearance
        or reid_segmented_attribute_appearance
        or reid_context_reranker_shadow
        or reid_motion_bridge_shadow
    ):
        raise ValueError(
            "SINet is calibrated only for mutual-best appearance stitching; "
            "global, anchored, endpoint, part-aware, ensemble, multiview, robust, calibrated, pose/segmented-attribute, context-reranker, motion-bridge, and offline-graph modes require OSNet"
        )
    if reid_part_aware_appearance and reid_ensemble_appearance:
        raise ValueError(
            "part-aware replacement and full/part ensemble modes are mutually exclusive"
        )
    if reid_ensemble_appearance and (
        reid_global_appearance or reid_endpoint_appearance
    ):
        raise ValueError(
            "full/part ensemble cannot be combined with global or endpoint appearance"
        )
    if reid_multiview_appearance and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_global_appearance
        or reid_endpoint_appearance
    ):
        raise ValueError(
            "multiview appearance cannot be combined with part-aware, ensemble, "
            "global, or endpoint appearance"
        )
    if reid_robust_appearance and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_global_appearance
        or reid_endpoint_appearance
    ):
        raise ValueError(
            "robust appearance cannot be combined with part-aware, ensemble, "
            "multiview, global, or endpoint appearance"
        )
    if reid_calibrated_appearance and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_offline_graph
    ):
        raise ValueError(
            "calibrated appearance cannot be combined with other experimental "
            "appearance or offline-graph modes"
        )
    if reid_pose_attribute_appearance and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_calibrated_appearance
        or reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_offline_graph
    ):
        raise ValueError(
            "pose-attribute appearance cannot be combined with other experimental "
            "appearance or offline-graph modes"
        )
    if reid_segmented_attribute_appearance and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_calibrated_appearance
        or reid_pose_attribute_appearance
        or reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_offline_graph
    ):
        raise ValueError(
            "segmented-attribute appearance cannot be combined with other "
            "experimental appearance or offline-graph modes"
        )
    if reid_context_reranker_shadow and (
        reid_part_aware_appearance
        or reid_ensemble_appearance
        or reid_multiview_appearance
        or reid_robust_appearance
        or reid_calibrated_appearance
        or reid_pose_attribute_appearance
        or reid_segmented_attribute_appearance
        or reid_global_appearance
        or reid_anchored_appearance
        or reid_endpoint_appearance
        or reid_offline_graph
    ):
        raise ValueError(
            "context reranker shadow mode currently requires baseline OSNet appearance"
        )

    if use_astra:
        validate_astra_options(
            sample_fps=sample_fps,
            batch_size=batch_size,
            batch_overlap=batch_overlap,
            targeted_pass=reid_targeted_pass,
            targeted_references_per_identity=reid_targeted_references,
            targeted_resampling=reid_targeted_resampling,
            targeted_resample_budget=reid_targeted_resample_budget,
            targeted_shortlist=reid_targeted_shortlist,
            targeted_shortlist_size=reid_targeted_shortlist_size,
            workers=workers,
            match_iou=match_iou,
        )
    if service_verifier not in {"off", "astra", "local"}:
        raise ValueError("service_verifier must be 'off', 'astra', or 'local'")
    if service_verifier == "local" and local_service_device is not None:
        service_compute_device = resolve_device(local_service_device)
    if not math.isfinite(reid_merge_confidence) or not 0 <= reid_merge_confidence <= 1:
        raise ValueError("reid_merge_confidence must be in [0, 1]")
    if (
        isinstance(reid_min_support, bool)
        or not isinstance(reid_min_support, int)
        or reid_min_support <= 0
    ):
        raise ValueError("reid_min_support must be a positive integer")
    if service_verifier == "astra" and (
        isinstance(service_verifier_frames, bool)
        or not isinstance(service_verifier_frames, int)
        or not 4 <= service_verifier_frames <= 8
    ):
        raise ValueError("service_verifier_frames must be an integer in [4, 8]")
    if service_verifier == "local":
        if not Path(local_service_model).is_file():
            raise ValueError(f"local service MLP does not exist: {local_service_model}")
        from flowmetric.service.local import LocalServiceMLP

        LocalServiceMLP.load(Path(local_service_model))
        if (
            isinstance(local_service_sample_fps, bool)
            or not isinstance(local_service_sample_fps, (int, float))
            or not math.isfinite(float(local_service_sample_fps))
            or local_service_sample_fps <= 0
        ):
            raise ValueError("local_service_sample_fps must be positive")
        if local_service_threshold is not None and (
            isinstance(local_service_threshold, bool)
            or not isinstance(local_service_threshold, (int, float))
            or not math.isfinite(float(local_service_threshold))
            or not 0 <= local_service_threshold <= 1
        ):
            raise ValueError("local_service_threshold must be in [0, 1]")
        if (
            isinstance(local_service_imgsz, bool)
            or not isinstance(local_service_imgsz, int)
            or local_service_imgsz <= 0
        ):
            raise ValueError("local_service_imgsz must be a positive integer")

    probe = cv2.VideoCapture(str(source))
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Cannot read video dimensions from {source}")
    scene = load_scene_config(
        camera or ROOT / "config" / "camera.json",
        width=source_width,
        height=source_height,
        source=source,
        tables_path=tables,
    )

    effective_tracker = (
        str(ROOT / "config" / "botsort_reid.yaml")
        if reid_in_tracker
        else tracker
    )
    print(
        f"[1/4] local detect+track: {model_name} @ {imgsz} + {effective_tracker} "
        f"on {compute_device}"
    )
    lp = local_track(
        source, model_name=model_name, conf=conf, iou=iou, imgsz=imgsz,
        tracker=effective_tracker, sample_fps=sample_fps if use_astra else 0,
        device=compute_device,
        appearance_model=(
            reid_appearance_model
            if reid_appearance_stitch and reid_appearance_backend == "osnet"
            else None
        ),
        part_aware_appearance=reid_part_aware_appearance,
        ensemble_appearance=reid_ensemble_appearance,
        tracker_reid_model=(
            reid_appearance_model if reid_in_tracker else None
        ),
    )
    appearance_stats: dict[str, Any] = {}
    appearance_min_similarity = 0.82
    appearance_source = "local_osnet_stitch"
    if reid_part_aware_appearance:
        appearance_source = "local_osnet_part_aware_stitch"
    elif reid_ensemble_appearance:
        appearance_source = "local_osnet_full_part_ensemble"
    elif reid_dense_multiview_appearance:
        appearance_source = "local_osnet_dense_multiview_consensus"
    elif reid_multiview_appearance:
        appearance_source = "local_osnet_multiview_consensus"
    elif reid_robust_appearance:
        appearance_source = "local_osnet_robust_temporal_consensus"
    elif reid_calibrated_appearance:
        appearance_source = "local_osnet_scene_calibrated"
    elif reid_pose_attribute_appearance:
        appearance_source = "local_osnet_pose_attribute_consensus"
    elif reid_segmented_attribute_appearance:
        appearance_source = "local_osnet_segmented_attribute_consensus"
    elif reid_context_reranker_shadow:
        appearance_source = "local_osnet_context_reranker_shadow"
    elif reid_motion_bridge_shadow:
        appearance_source = "local_osnet_motion_bridge_shadow"
    if reid_pose_attribute_appearance:
        from flowmetric.identity.pose_attributes import encode_pose_attribute_tracklets

        print("      attribute ReID: pose-guided torso/lower-body descriptors")
        appearance_stats = encode_pose_attribute_tracklets(
            source,
            lp,
            model=reid_pose_model,
            device=compute_device,
        )
    if reid_segmented_attribute_appearance:
        from flowmetric.identity.segmented_attributes import (
            encode_segmented_attribute_tracklets,
        )

        print("      attribute ReID: foreground-masked clothing descriptors")
        appearance_stats = encode_segmented_attribute_tracklets(source, lp)
    if reid_appearance_stitch and reid_appearance_backend == "sinet":
        from flowmetric.identity.video_appearance import (
            SINET_MARS_MIN_SIMILARITY,
            encode_sinet_tracklets,
        )

        print("      video ReID: SINet-MARS, 4 frames per tracklet")
        appearance_stats = encode_sinet_tracklets(
            source,
            lp,
            reid_sinet_model,
            device=compute_device,
        )
        appearance_min_similarity = SINET_MARS_MIN_SIMILARITY
        appearance_source = "local_sinet_stitch"
    if use_astra and reid_adaptive_sampling:
        from flowmetric.identity.sampling import select_adaptive_reid_frames
        from flowmetric.media.video import load_video_frames

        sample_count = max(1, len(lp.sample_idx))
        adaptive_indexes = select_adaptive_reid_frames(lp, count=sample_count)
        adaptive_frames = load_video_frames(source, tuple(adaptive_indexes))
        lp.sample_idx = adaptive_indexes
        lp.sample_frames = [adaptive_frames[index] for index in adaptive_indexes]
        print(
            f"      adaptive ReID sampling: {sample_count} frames, "
            f"{len({tid for frame in adaptive_indexes for tid, _, _ in lp.dets[frame]})} tracklets covered"
        )

    tid_to_gid: dict[int, int] = {}
    gid_role: dict[int, str] = {}
    sync_stats: dict[str, Any] = {}
    mode = "local"

    if use_astra:
        from flowmetric.infrastructure.astra import (
            AstraCacheError,
            AstraResponseError,
            build_astra_cache_manifest,
            configured_default_model_id,
            has_api_key,
            load_astra_cache,
            save_astra_cache,
        )

        batch_positions = make_batches(
            len(lp.sample_frames), batch_size, overlap=batch_overlap
        )
        expected_batches = [
            [lp.sample_idx[i] for i in batch] for batch in batch_positions
        ]
        cache_manifest = build_astra_cache_manifest(
            source,
            fps=lp.fps,
            width=lp.width,
            height=lp.height,
            frame_count=lp.total,
            sample_frame_indexes=lp.sample_idx,
            sample_fps=sample_fps,
            batch_size=batch_size,
            batch_overlap=batch_overlap,
            targeted_pass=reid_targeted_pass,
            targeted_references_per_identity=reid_targeted_references,
            targeted_resampling=reid_targeted_resampling,
            targeted_resample_budget=reid_targeted_resample_budget,
            targeted_shortlist=reid_targeted_shortlist,
            targeted_shortlist_size=reid_targeted_shortlist_size,
            part_aware_appearance=reid_part_aware_appearance,
            tracker_reid=reid_in_tracker,
            offline_graph=reid_offline_graph,
            requested_model=astra_model,
            effort=astra_effort,
            resolved_models=[model_id]
            if (model_id := astra_model or configured_default_model_id())
            else [],
        )
        reids: list[dict[str, Any]] | None = None
        if astra_cache and astra_cache.exists():
            try:
                reids = load_astra_cache(
                    astra_cache,
                    expected_manifest=cache_manifest,
                    expected_batches=expected_batches,
                    original_width=lp.width,
                    original_height=lp.height,
                )
                validate_reids_against_local(lp, reids)
                print(f"[2/4] astra: validated cache {astra_cache}")
            except (AstraCacheError, AstraResponseError) as exc:
                reids = None
                if not has_api_key():
                    raise RuntimeError(
                        f"Cannot reuse Astra cache {astra_cache}: {exc}. "
                        "Set VSELLM_API_KEY/OPENAI_API_KEY to refresh it or use --no-astra."
                    ) from exc
                print(f"[2/4] astra: cache rejected ({exc}); querying Astra again")
        if reids is None:
            print("[2/4] astra ReID on sampled frames")
            reids = astra_pass(
                lp, source=source, model=astra_model, effort=astra_effort,
                batch_size=batch_size, batch_overlap=batch_overlap, workers=workers,
                targeted_pass=reid_targeted_pass,
                merge_confidence=reid_merge_confidence,
                min_support=reid_min_support,
                targeted_references_per_identity=reid_targeted_references,
                targeted_resampling=reid_targeted_resampling,
                targeted_resample_budget=reid_targeted_resample_budget,
                targeted_shortlist=reid_targeted_shortlist,
                targeted_shortlist_size=reid_targeted_shortlist_size,
            )
            if astra_cache:
                cache_manifest["astra"]["resolved_models"] = sorted(
                    {
                        str(reid["_meta"]["requested_model"])
                        for reid in reids
                        if reid["_meta"].get("requested_model")
                    }
                )
                cache_manifest["astra"]["actual_models"] = sorted(
                    {
                        str(reid["_meta"]["actual_model"])
                        for reid in reids
                        if reid["_meta"].get("actual_model")
                    }
                )
                if len(cache_manifest["astra"]["actual_models"]) == 1:
                    cache_manifest["astra"]["actual_model"] = cache_manifest["astra"][
                        "actual_models"
                    ][0]
                save_astra_cache(
                    astra_cache,
                    manifest=cache_manifest,
                    reids=reids,
                )
        sync_options = dict(
            iou_thr=match_iou,
            merge_confidence=reid_merge_confidence,
            min_support=reid_min_support,
            local_stitch=reid_local_stitch,
            adaptive_singletons=reid_adaptive_singletons,
            appearance_stitch=reid_appearance_stitch,
            global_appearance=reid_global_appearance,
            anchored_appearance=reid_anchored_appearance,
            appearance_min_similarity=appearance_min_similarity,
            appearance_source=appearance_source,
            endpoint_appearance=reid_endpoint_appearance,
            offline_graph=reid_offline_graph,
            scenario_constraints=reid_scenario_constraints,
            scene_config=scene,
            ensemble_appearance=reid_ensemble_appearance,
            multiview_appearance=reid_multiview_appearance,
            robust_appearance=reid_robust_appearance,
            calibrated_appearance=reid_calibrated_appearance,
            pose_attribute_appearance=reid_pose_attribute_appearance,
            segmented_attribute_appearance=reid_segmented_attribute_appearance,
            context_reranker_shadow=reid_context_reranker_shadow,
            motion_bridge_shadow=reid_motion_bridge_shadow,
        )
        tid_to_gid, gid_role, sync_stats = sync_ids(lp, reids, **sync_options)
        if reid_dense_multiview_appearance:
            from flowmetric.identity.appearance import (
                densify_multiview_appearance,
                select_multiview_resample_tracklets,
            )

            unresolved = {
                int(tracklet_id)
                for tracklet_id, identity in sync_stats["track_identity"].items()
                if identity["status"] in {"abstained", "local_only", "rejected"}
            }
            resample_tracklets = select_multiview_resample_tracklets(
                lp, unresolved, target_samples=3
            )
            dense_stats = densify_multiview_appearance(
                source,
                lp,
                reid_appearance_model,
                device=compute_device,
                tracklet_ids=resample_tracklets,
                target_samples=3,
            )
            if dense_stats["added_samples"]:
                tid_to_gid, gid_role, sync_stats = sync_ids(
                    lp, reids, **sync_options
                )
            sync_stats["dense_multiview_enabled"] = True
            sync_stats["dense_multiview_unresolved_tracklets"] = sorted(unresolved)
            sync_stats["dense_multiview_resampling"] = dense_stats
        sync_stats["appearance_backend"] = (
            "osnet_part_aware"
            if reid_part_aware_appearance
            else "osnet_full_part_ensemble"
            if reid_ensemble_appearance
            else "osnet_dense_multiview"
            if reid_dense_multiview_appearance
            else "osnet_multiview"
            if reid_multiview_appearance
            else "osnet_robust_temporal"
            if reid_robust_appearance
            else "osnet_scene_calibrated"
            if reid_calibrated_appearance
            else "osnet_pose_attribute"
            if reid_pose_attribute_appearance
            else "osnet_segmented_attribute"
            if reid_segmented_attribute_appearance
            else "osnet_context_reranker_shadow"
            if reid_context_reranker_shadow
            else reid_appearance_backend
        )
        sync_stats["tracker_reid_enabled"] = reid_in_tracker
        if appearance_stats:
            sync_stats["appearance_embedding"] = appearance_stats
        sync_stats["sampling_strategy"] = (
            "rare_tracklet_coverage_v1"
            if reid_adaptive_sampling
            else "uniform_cadence_v1"
        )
        sync_stats["sampled_frames"] = list(lp.sample_idx)
        sync_stats["sampled_tracklets"] = len(
            {
                int(tid)
                for frame in lp.sample_idx
                for tid, _, _ in lp.dets.get(frame, [])
            }
        )
        sync_stats["nested_suppression_enabled"] = reid_nested_suppression
        sync_stats["nested_suppression_decisions"] = []
        sync_stats["nested_suppressed_tracklets"] = 0
        if reid_nested_suppression:
            from flowmetric.identity.suppression import (
                apply_tracklet_suppressions,
                propose_nested_tracklet_suppressions,
            )

            suppressions = propose_nested_tracklet_suppressions(
                lp,
                tid_to_gid,
                sync_stats.get("track_identity", {}),
            )
            suppressed_tids = {item.tracklet_id for item in suppressions}
            apply_tracklet_suppressions(lp, suppressed_tids)
            for tracklet_id in suppressed_tids:
                tid_to_gid.pop(tracklet_id, None)
                identity = sync_stats["track_identity"].get(str(tracklet_id))
                if identity is not None:
                    identity["status"] = "suppressed_nested"
            surviving_gids = set(tid_to_gid.values())
            gid_role = {
                gid: role for gid, role in gid_role.items() if gid in surviving_gids
            }
            sync_stats["global_ids"] = len(surviving_gids)
            sync_stats["nested_suppression_decisions"] = [
                item.as_dict() for item in suppressions
            ]
            sync_stats["nested_suppressed_tracklets"] = len(suppressions)
            sync_stats["track_identity_status_counts"] = dict(
                Counter(
                    item["status"]
                    for item in sync_stats["track_identity"].values()
                )
            )
        mode = "hybrid"
        decision_counts = Counter(
            item["decision"] for item in sync_stats.get("identity_decisions", [])
        )
        print(
            "      sync: "
            f"{sync_stats['local_tracks']} local → {sync_stats['global_ids']} IDs, "
            f"verified {sync_stats['astra_verified']}, decisions {dict(decision_counts)}, "
            f"local stitches {sync_stats['local_stitch_merges']}, "
            f"graph merges {sync_stats['offline_graph_merges']}, "
            f"rejected merges {sync_stats['rejected_merges']}, "
            f"retrospective revisions {len(sync_stats.get('revision_log', []))}"
        )
    else:
        print("[2/4] astra: skipped (--no-astra)")

    out_dir = dest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    service_suffix = {
        "off": "",
        "astra": "+service-astra",
        "local": "+service-local",
    }[service_verifier]
    pipeline_mode = f"{mode}{service_suffix}"
    tracks = build_tracks(lp, tid_to_gid, gid_role, mode=pipeline_mode)
    tracks["device"] = compute_device
    if service_verifier == "local":
        tracks["service_device"] = service_compute_device
    if sync_stats:
        tracks["identity_sync"] = sync_stats
        identity_by_tid = sync_stats.get("track_identity", {})
        for track in tracks["tracks"].values():
            evidence = [identity_by_tid.get(str(tid), {}) for tid in track["tids"]]
            confidences = [
                item["confidence"]
                for item in evidence
                if isinstance(item.get("confidence"), (int, float))
            ]
            track["identity_status"] = (
                "merged"
                if any(item.get("status") == "merged" for item in evidence)
                else "appearance_stitched"
                if any(
                    item.get("status") == "appearance_stitched" for item in evidence
                )
                else "local_stitched"
                if any(item.get("status") == "local_stitched" for item in evidence)
                else "rejected"
                if any(item.get("status") == "rejected" for item in evidence)
                else "abstained"
                if any(item.get("status") == "abstained" for item in evidence)
                else "local_only"
            )
            track["identity_confidence"] = (
                round(float(np.mean(confidences)), 4) if confidences else None
            )
    if (lp.width, lp.height) != (source_width, source_height):
        raise RuntimeError(
            "Video dimensions changed between calibration validation and processing: "
            f"{source_width}x{source_height} -> {lp.width}x{lp.height}"
        )
    verify_service = None
    if service_verifier == "astra":
        from flowmetric.service.astra import AstraServiceVerifier

        print("      service verifier: Astra on ambiguous FSM candidates")
        verify_service = AstraServiceVerifier(
            source,
            tracks,
            cache_path=service_verifier_cache,
            model=service_verifier_model,
            effort=service_verifier_effort,
            frames_per_candidate=service_verifier_frames,
        )
    elif service_verifier == "local":
        from flowmetric.service.local import LocalServiceVerifier

        print("      service verifier: local pose + objects + MLP")
        verify_service = LocalServiceVerifier(
            source,
            tracks,
            classifier_path=local_service_model,
            pose_model=local_pose_model,
            object_model=local_object_model,
            decision_threshold=local_service_threshold,
            sample_fps=local_service_sample_fps,
            imgsz=local_service_imgsz,
            device=service_compute_device,
        )
    ev = extract_events(tracks, scene, service_verifier=verify_service)
    (out_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/4] events: {len(ev['events'])} events, {len(ev['states'])} states → {out_dir / 'events.json'}")
    for e in ev["events"]:
        print(f"      {_event_text(e)}")

    print(f"[4/4] render → {dest}")
    render(source, dest, lp, tid_to_gid, gid_role, mode=pipeline_mode, events=ev)

    bench = None
    if gt is not None and gt.exists():
        from metrics import print_report, run_benchmark

        bench = run_benchmark(tracks, ev, json.loads(gt.read_text(encoding="utf-8")))
        (out_dir / "benchmark_report.json").write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBenchmark vs {gt.name} [{pipeline_mode}]")
        print_report(bench)

    local_ids = {tid for d in lp.dets.values() for tid, _, _ in d}
    stats = {
        "benchmark": bench,
        "events": ev,
        "mode": pipeline_mode,
        "frames": lp.total,
        "fps": lp.fps,
        "size": (lp.width, lp.height),
        "local_tracks": len(local_ids),
        "final_ids": len(set(tid_to_gid.values())) if tid_to_gid else len(local_ids),
        "astra_samples": len(lp.sample_idx) if use_astra else 0,
        "tables": len(scene.get("tables", [])),
        "sync": sync_stats,
        "output": dest,
        "device": compute_device,
    }
    if service_verifier != "off":
        stats["service_verifier"] = service_verifier
        stats["service_candidates"] = len(ev.get("service_candidates", []))
    if service_verifier == "local":
        stats["service_device"] = service_compute_device
    return stats


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run FlowMetric from grouped configuration while preserving ``run_hybrid``.

    ``run_hybrid`` remains the compatibility API used by existing experiments;
    new callers should construct a ``PipelineConfig`` instead of passing a long
    list of loosely related keyword arguments.
    """
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig")
    tracking = config.tracking
    identity = config.identity
    service = config.service
    return run_hybrid(
        config.source,
        config.destination,
        model_name=tracking.model_name,
        conf=tracking.confidence,
        iou=tracking.iou,
        imgsz=tracking.image_size,
        tracker=tracking.tracker,
        device=tracking.device,
        camera=config.camera,
        tables=config.tables,
        gt=config.ground_truth,
        use_astra=identity.enabled,
        astra_model=identity.model,
        astra_effort=identity.effort,
        sample_fps=identity.sample_fps,
        batch_size=identity.batch_size,
        batch_overlap=identity.batch_overlap,
        workers=identity.workers,
        match_iou=identity.match_iou,
        reid_merge_confidence=identity.merge_confidence,
        reid_min_support=identity.minimum_support,
        reid_local_stitch=identity.local_stitch,
        reid_targeted_pass=identity.targeted_pass,
        reid_targeted_references=identity.targeted_references_per_identity,
        reid_targeted_resampling=identity.targeted_resampling,
        reid_targeted_resample_budget=identity.targeted_resample_budget,
        reid_targeted_shortlist=identity.targeted_shortlist,
        reid_targeted_shortlist_size=identity.targeted_shortlist_size,
        reid_adaptive_singletons=identity.adaptive_singletons,
        reid_appearance_stitch=identity.appearance_stitch,
        reid_global_appearance=identity.global_appearance,
        reid_anchored_appearance=identity.anchored_appearance,
        reid_nested_suppression=identity.nested_suppression,
        reid_endpoint_appearance=identity.endpoint_appearance,
        reid_part_aware_appearance=identity.part_aware_appearance,
        reid_in_tracker=identity.in_tracker,
        reid_offline_graph=identity.offline_graph,
        reid_scenario_constraints=identity.scenario_constraints,
        reid_ensemble_appearance=identity.ensemble_appearance,
        reid_multiview_appearance=identity.multiview_appearance,
        reid_dense_multiview_appearance=identity.dense_multiview_appearance,
        reid_robust_appearance=identity.robust_appearance,
        reid_calibrated_appearance=identity.calibrated_appearance,
        reid_pose_attribute_appearance=identity.pose_attribute_appearance,
        reid_segmented_attribute_appearance=identity.segmented_attribute_appearance,
        reid_context_reranker_shadow=identity.context_reranker_shadow,
        reid_motion_bridge_shadow=identity.motion_bridge_shadow,
        reid_appearance_backend=identity.appearance_backend,
        reid_appearance_model=identity.appearance_model,
        reid_pose_model=identity.pose_model,
        reid_sinet_model=identity.sinet_model,
        reid_adaptive_sampling=identity.adaptive_sampling,
        astra_cache=identity.cache,
        service_verifier=service.verifier,
        service_verifier_model=service.astra_model,
        service_verifier_effort=service.astra_effort,
        service_verifier_frames=service.astra_frames,
        service_verifier_cache=service.astra_cache,
        local_pose_model=service.pose_model,
        local_object_model=service.object_model,
        local_service_model=service.classifier,
        local_service_sample_fps=service.sample_fps,
        local_service_threshold=service.threshold,
        local_service_imgsz=service.image_size,
        local_service_device=service.device,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=next((ROOT / "data").glob("*.mp4"), None))
    p.add_argument("--out", type=Path, default=ROOT / "data" / "output" / "tracked.mp4")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1280, help="detector input size")
    p.add_argument(
        "--tracker",
        default=str(ROOT / "config" / "botsort_recall.yaml"),
        help="Ultralytics tracker YAML; the default keeps low-confidence distant people",
    )
    p.add_argument(
        "--device", default="auto",
        help="compute device: auto, cpu, mps, cuda, cuda:N or N",
    )
    p.add_argument("--camera", type=Path, default=ROOT / "config" / "camera.json", help="zones config")
    p.add_argument(
        "--tables", type=Path, default=None,
        help="optional static table map (for example config/tables.json)",
    )
    p.add_argument(
        "--gt", type=Path, default=ROOT / "benchmark" / "ground_truth.json",
        help="ground truth for the benchmark (skipped if the file is missing)",
    )
    p.add_argument(
        "--no-astra", action="store_true",
        help="skip Astra ReID (service verification is controlled separately)",
    )
    p.add_argument(
        "--astra-model", default=None,
        help="default: openai/gpt-6-astra via VseLLM, gpt-6-astra via OpenAI",
    )
    p.add_argument("--astra-effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--sample-fps", type=float, default=1.0, help="frames/sec sent to Astra")
    p.add_argument("--batch-size", type=int, default=16, help="frames per Astra call")
    p.add_argument(
        "--batch-overlap", type=int, default=4,
        help="sampled frames repeated between adjacent Astra batches",
    )
    p.add_argument("--workers", type=int, default=4, help="parallel Astra calls")
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument(
        "--reid-merge-confidence", type=float, default=0.55,
        help="minimum weighted confidence for a tracklet identity merge",
    )
    p.add_argument(
        "--reid-min-support", type=int, default=2,
        help="minimum crop/frame confirmations for a ReID merge",
    )
    p.add_argument(
        "--no-reid-local-stitch", action="store_true",
        help="disable conservative motion-based stitching across short tracking gaps",
    )
    p.add_argument(
        "--no-reid-targeted-pass", action="store_true",
        help="disable the second Astra pass over abstained tracklets and references",
    )
    p.add_argument(
        "--reid-targeted-references", type=int, default=1,
        help="temporally diverse reference tracklets per confirmed identity (1-3)",
    )
    p.add_argument(
        "--reid-targeted-resampling", action="store_true",
        help="add extra full-track frames for unresolved targeted-pass queries",
    )
    p.add_argument(
        "--reid-targeted-resample-frames", type=int, default=16,
        help="maximum extra frames decoded for unresolved targeted-pass queries",
    )
    p.add_argument(
        "--reid-targeted-shortlist", action="store_true",
        help="limit each unresolved targeted query to compatible reference identities",
    )
    p.add_argument(
        "--reid-targeted-shortlist-size", type=int, default=5,
        help="candidate reference identities per targeted query (3-5)",
    )
    p.add_argument(
        "--no-reid-adaptive-singletons", action="store_true",
        help="disable repeated high-confidence one-crop ReID associations",
    )
    p.add_argument(
        "--reid-appearance-stitch", action="store_true",
        help="enable conservative local appearance stitching for unresolved tracklets",
    )
    p.add_argument(
        "--reid-appearance-backend", choices=["osnet", "sinet"], default="osnet",
        help="appearance model: frame-based OSNet or four-frame video SINet",
    )
    p.add_argument(
        "--reid-global-appearance", action="store_true",
        help="globally assign residual high-confidence OSNet links",
    )
    p.add_argument(
        "--reid-anchored-appearance", action="store_true",
        help="enable high-confidence OSNet rescue into Astra-verified clusters",
    )
    p.add_argument(
        "--reid-nested-suppression", action="store_true",
        help="suppress short unsupported person tracklets nested in a stronger track",
    )
    p.add_argument(
        "--reid-endpoint-appearance", action="store_true",
        help="rescue OSNet links supported by matching trajectory endpoints",
    )
    p.add_argument(
        "--reid-part-aware-appearance", action="store_true",
        help="encode full, upper and lower body with occlusion-aware OSNet weights",
    )
    p.add_argument(
        "--reid-in-tracker", action="store_true",
        help="enable native appearance association inside BoT-SORT",
    )
    p.add_argument(
        "--reid-offline-graph", action="store_true",
        help="globally rank motion and appearance edges under hard ReID constraints",
    )
    p.add_argument(
        "--reid-scenario-constraints", action="store_true",
        help="apply restaurant zone, direction, occlusion and exit lifecycle evidence",
    )
    p.add_argument(
        "--reid-ensemble-appearance", action="store_true",
        help="preserve full-body OSNet links and add full/part consensus links",
    )
    p.add_argument(
        "--reid-multiview-appearance", action="store_true",
        help="add links supported by at least two independent OSNet crop matches",
    )
    p.add_argument(
        "--reid-dense-multiview-appearance", action="store_true",
        help="resample unresolved short tracklets before multi-view OSNet consensus",
    )
    p.add_argument(
        "--reid-robust-appearance", action="store_true",
        help="add OSNet links from quality-weighted outlier-resistant prototypes",
    )
    p.add_argument(
        "--reid-calibrated-appearance", action="store_true",
        help="calibrate OSNet threshold from trusted Astra and cannot-link pairs",
    )
    p.add_argument(
        "--reid-pose-attribute-appearance", action="store_true",
        help="add pose-guided clothing attribute consensus to OSNet ReID",
    )
    p.add_argument(
        "--reid-segmented-attribute-appearance", action="store_true",
        help="add foreground-masked clothing attribute consensus to OSNet ReID",
    )
    p.add_argument(
        "--reid-context-reranker-shadow", action="store_true",
        help="audit a learned context pair reranker without changing identity merges",
    )
    p.add_argument(
        "--reid-motion-bridge-shadow", action="store_true",
        help="audit globally assigned bidirectional motion+appearance bridges",
    )
    p.add_argument(
        "--reid-appearance-model", type=Path,
        default=ROOT / "weights" / "osnet_x0_25_msmt17.pt",
        help="local person-ReID weights used by appearance stitching",
    )
    p.add_argument(
        "--reid-pose-model", default="yolo11n-pose.pt",
        help="pose model used for pose-guided ReID attributes",
    )
    p.add_argument(
        "--reid-sinet-model", type=Path,
        default=ROOT / "weights" / "sinet_mars.pth.tar",
        help="trusted official SINet-MARS checkpoint (.pth.tar)",
    )
    p.add_argument(
        "--no-reid-adaptive-sampling", action="store_true",
        help="use fixed-cadence Astra frames instead of rare-tracklet coverage",
    )
    p.add_argument(
        "--astra-cache", type=Path, default=ROOT / "data" / "output" / "astra_reid.json",
        help="reuse a matching versioned cache; stale/legacy caches are refreshed safely",
    )
    p.add_argument(
        "--service-verifier", choices=["off", "astra", "local"], default="off",
        help="verify ambiguous service candidates with Astra or local pose/object MLP",
    )
    p.add_argument(
        "--service-verifier-model", default=None,
        help="model for service verification (default: provider Astra model)",
    )
    p.add_argument(
        "--service-verifier-effort", default="low",
        choices=["low", "medium", "high"],
    )
    p.add_argument(
        "--service-verifier-frames", type=int, default=6,
        help="annotated crop frames per ambiguous Astra candidate (4-8)",
    )
    p.add_argument(
        "--service-verifier-cache", type=Path,
        default=ROOT / "data" / "output" / "astra_service_verification.json",
        help="separate versioned cache for Astra service-verification verdicts",
    )
    p.add_argument(
        "--local-pose-model", default="yolo11n-pose.pt",
        help="Ultralytics pose model for the fully local service verifier",
    )
    p.add_argument(
        "--local-object-model", default="yolo11n.pt",
        help="object model; custom weights may add tray/plate classes",
    )
    p.add_argument(
        "--local-service-model", type=Path,
        default=ROOT / "config" / "service_mlp_bootstrap.json",
        help="versioned local MLP JSON checkpoint",
    )
    p.add_argument(
        "--local-service-sample-fps", type=float, default=4.0,
        help="pose/object sampling rate inside candidate intervals",
    )
    p.add_argument(
        "--local-service-threshold", type=float, default=None,
        help="override the decision threshold stored in the MLP checkpoint",
    )
    p.add_argument(
        "--local-service-imgsz", type=int, default=640,
        help="pose/object inference size for candidate crops",
    )
    p.add_argument(
        "--local-service-device", default=None,
        help="optional local-verifier override: auto, cpu, mps, cuda, cuda:N or N",
    )
    args = p.parse_args()

    if args.source is None or not args.source.exists():
        raise SystemExit("No input video. Pass --source path/to/video.mp4")

    from flowmetric.infrastructure.astra import has_api_key

    if not args.no_astra:
        try:
            validate_astra_options(
                sample_fps=args.sample_fps,
                batch_size=args.batch_size,
                batch_overlap=args.batch_overlap,
                targeted_pass=not args.no_reid_targeted_pass,
                targeted_references_per_identity=args.reid_targeted_references,
                targeted_resampling=args.reid_targeted_resampling,
                targeted_resample_budget=args.reid_targeted_resample_frames,
                targeted_shortlist=args.reid_targeted_shortlist,
                targeted_shortlist_size=args.reid_targeted_shortlist_size,
                workers=args.workers,
                match_iou=args.match_iou,
            )
        except ValueError as exc:
            p.error(str(exc))
    if (
        args.service_verifier == "astra"
        and not 4 <= args.service_verifier_frames <= 8
    ):
        p.error("--service-verifier-frames must be in [4, 8]")
    if args.service_verifier == "local":
        if not args.local_service_model.is_file():
            p.error(f"--local-service-model does not exist: {args.local_service_model}")
        if not math.isfinite(args.local_service_sample_fps) or args.local_service_sample_fps <= 0:
            p.error("--local-service-sample-fps must be positive")
        if args.local_service_threshold is not None and not 0 <= args.local_service_threshold <= 1:
            p.error("--local-service-threshold must be in [0, 1]")
        if args.local_service_imgsz <= 0:
            p.error("--local-service-imgsz must be positive")
    if not math.isfinite(args.reid_merge_confidence) or not 0 <= args.reid_merge_confidence <= 1:
        p.error("--reid-merge-confidence must be in [0, 1]")
    if args.reid_min_support <= 0:
        p.error("--reid-min-support must be positive")

    cache_ready = args.astra_cache is not None and args.astra_cache.exists()
    if not args.no_astra and not has_api_key():
        if not cache_ready:
            raise SystemExit("Set VSELLM_API_KEY (or OPENAI_API_KEY), or pass --no-astra")
        if args.no_reid_adaptive_sampling:
            # Reject an obviously stale/legacy uniform-sampling cache before
            # the expensive local pass. Detection-aware indexes are only known
            # after tracking and are validated later in run_hybrid.
            from flowmetric.infrastructure.astra import (
                AstraCacheError,
                build_astra_cache_manifest,
                load_astra_cache,
            )

            probe = cv2.VideoCapture(str(args.source))
            if not probe.isOpened():
                raise SystemExit(f"Cannot open {args.source}")
            fps = probe.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            probe.release()
            step = max(1, int(round(fps / args.sample_fps)))
            sample_indexes = list(range(0, total, step))
            batch_positions = make_batches(
                len(sample_indexes), args.batch_size, overlap=args.batch_overlap
            )
            expected_batches = [
                [sample_indexes[i] for i in batch] for batch in batch_positions
            ]
            expected_manifest = build_astra_cache_manifest(
                args.source,
                fps=fps,
                width=width,
                height=height,
                frame_count=total,
                sample_frame_indexes=sample_indexes,
                sample_fps=args.sample_fps,
                batch_size=args.batch_size,
                batch_overlap=args.batch_overlap,
                targeted_pass=not args.no_reid_targeted_pass,
                targeted_references_per_identity=args.reid_targeted_references,
                targeted_resampling=args.reid_targeted_resampling,
                targeted_resample_budget=args.reid_targeted_resample_frames,
                targeted_shortlist=args.reid_targeted_shortlist,
                targeted_shortlist_size=args.reid_targeted_shortlist_size,
                part_aware_appearance=args.reid_part_aware_appearance,
                tracker_reid=args.reid_in_tracker,
                offline_graph=(
                    args.reid_offline_graph or args.reid_scenario_constraints
                ),
                requested_model=args.astra_model,
                effort=args.astra_effort,
            )
            try:
                load_astra_cache(
                    args.astra_cache,
                    expected_manifest=expected_manifest,
                    expected_batches=expected_batches,
                    original_width=width,
                    original_height=height,
                )
            except (AstraCacheError, OSError, ValueError) as exc:
                raise SystemExit(
                    f"Cannot reuse Astra cache {args.astra_cache}: {exc}. "
                    "Set VSELLM_API_KEY/OPENAI_API_KEY to refresh it or use --no-astra."
                ) from exc

    stats = run_pipeline(
        PipelineConfig(
            source=args.source,
            destination=args.out,
            camera=args.camera,
            tables=args.tables,
            ground_truth=args.gt,
            tracking=TrackingConfig(
                model_name=args.model,
                confidence=args.conf,
                iou=args.iou,
                image_size=args.imgsz,
                tracker=args.tracker,
                device=args.device,
            ),
            identity=IdentityConfig(
                enabled=not args.no_astra,
                model=args.astra_model,
                effort=args.astra_effort,
                sample_fps=args.sample_fps,
                batch_size=args.batch_size,
                batch_overlap=args.batch_overlap,
                workers=args.workers,
                match_iou=args.match_iou,
                merge_confidence=args.reid_merge_confidence,
                minimum_support=args.reid_min_support,
                local_stitch=not args.no_reid_local_stitch,
                targeted_pass=not args.no_reid_targeted_pass,
                targeted_references_per_identity=args.reid_targeted_references,
                targeted_resampling=args.reid_targeted_resampling,
                targeted_resample_budget=args.reid_targeted_resample_frames,
                targeted_shortlist=args.reid_targeted_shortlist,
                targeted_shortlist_size=args.reid_targeted_shortlist_size,
                adaptive_singletons=not args.no_reid_adaptive_singletons,
                appearance_stitch=args.reid_appearance_stitch,
                global_appearance=args.reid_global_appearance,
                anchored_appearance=args.reid_anchored_appearance,
                nested_suppression=args.reid_nested_suppression,
                endpoint_appearance=args.reid_endpoint_appearance,
                part_aware_appearance=args.reid_part_aware_appearance,
                in_tracker=args.reid_in_tracker,
                offline_graph=args.reid_offline_graph,
                scenario_constraints=args.reid_scenario_constraints,
                ensemble_appearance=args.reid_ensemble_appearance,
                multiview_appearance=args.reid_multiview_appearance,
                dense_multiview_appearance=args.reid_dense_multiview_appearance,
                robust_appearance=args.reid_robust_appearance,
                calibrated_appearance=args.reid_calibrated_appearance,
                pose_attribute_appearance=args.reid_pose_attribute_appearance,
                segmented_attribute_appearance=args.reid_segmented_attribute_appearance,
                context_reranker_shadow=args.reid_context_reranker_shadow,
                motion_bridge_shadow=args.reid_motion_bridge_shadow,
                appearance_backend=args.reid_appearance_backend,
                appearance_model=args.reid_appearance_model,
                pose_model=args.reid_pose_model,
                sinet_model=args.reid_sinet_model,
                adaptive_sampling=not args.no_reid_adaptive_sampling,
                cache=args.astra_cache,
            ),
            service=ServiceConfig(
                verifier=args.service_verifier,
                astra_model=args.service_verifier_model,
                astra_effort=args.service_verifier_effort,
                astra_frames=args.service_verifier_frames,
                astra_cache=args.service_verifier_cache,
                pose_model=args.local_pose_model,
                object_model=args.local_object_model,
                classifier=args.local_service_model,
                sample_fps=args.local_service_sample_fps,
                threshold=args.local_service_threshold,
                image_size=args.local_service_imgsz,
                device=args.local_service_device,
            ),
        )
    )
    print(
        f"done [{stats['mode']}] on {stats['device']}: {stats['frames']} frames, "
        f"local tracks {stats['local_tracks']} → final IDs {stats['final_ids']}, "
        f"astra samples {stats['astra_samples']} → {stats['output']}"
    )


if __name__ == "__main__":
    main()
