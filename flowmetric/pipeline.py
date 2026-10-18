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
from collections import Counter
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
    *, sample_fps: float, batch_size: int, workers: int, match_iou: float
) -> None:
    """Validate Astra orchestration options before decoding or model work."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be a positive finite number")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    make_batches(0, batch_size)
    if not math.isfinite(match_iou) or not 0 <= match_iou <= 1:
        raise ValueError("match_iou must be a finite number in [0, 1]")


def astra_pass(
    lp: LocalPass,
    *,
    model: str | None,
    effort: str,
    batch_size: int,
    workers: int,
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
    )

    client, default_model = make_client()
    model = model or default_model
    batches = make_batches(len(lp.sample_frames), batch_size)
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
    workers: int = 4,
    match_iou: float = 0.3,
    reid_merge_confidence: float = 0.55,
    reid_min_support: int = 2,
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

    if use_astra:
        validate_astra_options(
            sample_fps=sample_fps,
            batch_size=batch_size,
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

    print(
        f"[1/4] local detect+track: {model_name} @ {imgsz} + {tracker} "
        f"on {compute_device}"
    )
    lp = local_track(
        source, model_name=model_name, conf=conf, iou=iou, imgsz=imgsz,
        tracker=tracker, sample_fps=sample_fps if use_astra else 0,
        device=compute_device,
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

        batch_positions = make_batches(len(lp.sample_frames), batch_size)
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
                lp, model=astra_model, effort=astra_effort,
                batch_size=batch_size, workers=workers,
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
        tid_to_gid, gid_role, sync_stats = sync_ids(
            lp,
            reids,
            iou_thr=match_iou,
            merge_confidence=reid_merge_confidence,
            min_support=reid_min_support,
        )
        mode = "hybrid"
        decision_counts = Counter(
            item["decision"] for item in sync_stats.get("identity_decisions", [])
        )
        print(
            "      sync: "
            f"{sync_stats['local_tracks']} local → {sync_stats['global_ids']} IDs, "
            f"verified {sync_stats['astra_verified']}, decisions {dict(decision_counts)}, "
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
        workers=identity.workers,
        match_iou=identity.match_iou,
        reid_merge_confidence=identity.merge_confidence,
        reid_min_support=identity.minimum_support,
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
        # Reject an obviously stale/legacy cache before the expensive local pass.
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
        batch_positions = make_batches(len(sample_indexes), args.batch_size)
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
                workers=args.workers,
                match_iou=args.match_iou,
                merge_confidence=args.reid_merge_confidence,
                minimum_support=args.reid_min_support,
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
