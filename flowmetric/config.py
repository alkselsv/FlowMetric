"""Typed configuration groups for the FlowMetric pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flowmetric.paths import PROJECT_ROOT


@dataclass(frozen=True)
class TrackingConfig:
    model_name: str = "yolo11s.pt"
    confidence: float = 0.1
    iou: float = 0.5
    image_size: int = 1280
    tracker: str = str(PROJECT_ROOT / "config" / "botsort_recall.yaml")


@dataclass(frozen=True)
class IdentityConfig:
    enabled: bool = True
    model: str | None = None
    effort: str = "low"
    sample_fps: float = 1.0
    batch_size: int = 16
    workers: int = 4
    match_iou: float = 0.3
    merge_confidence: float = 0.55
    minimum_support: int = 2
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
