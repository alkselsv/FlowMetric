"""Typed configuration groups for the FlowMetric pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flowmetric.paths import PROJECT_ROOT, WEIGHTS_DIR


@dataclass(frozen=True)
class TrackingConfig:
    model_name: str = "yolo11s.pt"
    confidence: float = 0.1
    iou: float = 0.5
    image_size: int = 1280
    tracker: str = str(PROJECT_ROOT / "config" / "botsort_recall.yaml")
    device: str = "auto"


@dataclass(frozen=True)
class IdentityConfig:
    enabled: bool = True
    model: str | None = None
    effort: str = "low"
    sample_fps: float = 1.0
    batch_size: int = 16
    batch_overlap: int = 4
    workers: int = 4
    match_iou: float = 0.3
    merge_confidence: float = 0.55
    minimum_support: int = 2
    local_stitch: bool = True
    targeted_pass: bool = True
    targeted_references_per_identity: int = 1
    targeted_resampling: bool = False
    targeted_resample_budget: int = 16
    targeted_shortlist: bool = False
    targeted_shortlist_size: int = 5
    adaptive_singletons: bool = True
    appearance_stitch: bool = False
    global_appearance: bool = False
    anchored_appearance: bool = False
    nested_suppression: bool = False
    endpoint_appearance: bool = False
    part_aware_appearance: bool = False
    in_tracker: bool = False
    offline_graph: bool = False
    scenario_constraints: bool = False
    ensemble_appearance: bool = False
    multiview_appearance: bool = False
    dense_multiview_appearance: bool = False
    robust_appearance: bool = False
    calibrated_appearance: bool = False
    pose_attribute_appearance: bool = False
    segmented_attribute_appearance: bool = False
    context_reranker_shadow: bool = False
    motion_bridge_shadow: bool = False
    appearance_backend: str = "osnet"
    appearance_model: Path = WEIGHTS_DIR / "osnet_x0_25_msmt17.pt"
    pose_model: str = "yolo11n-pose.pt"
    sinet_model: Path = WEIGHTS_DIR / "sinet_mars.pth.tar"
    adaptive_sampling: bool = True
    cache: Path | None = None


@dataclass(frozen=True)
class ServiceConfig:
    verifier: str = "off"
    astra_model: str | None = None
    astra_effort: str = "low"
    astra_frames: int = 6
    astra_cache: Path | None = None
    pose_model: str = "yolo11n-pose.pt"
    object_model: str = "yolo11n.pt"
    classifier: Path = PROJECT_ROOT / "config" / "service_mlp_bootstrap.json"
    sample_fps: float = 4.0
    threshold: float | None = None
    image_size: int = 640
    device: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    destination: Path
    camera: Path | None = None
    tables: Path | None = None
    ground_truth: Path | None = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
