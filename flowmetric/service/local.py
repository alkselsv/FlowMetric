"""Private, fully local verification of proposed restaurant service episodes.

The event FSM owns identities and interval boundaries.  This module samples a
denser local sequence for ambiguous candidates, extracts pose/object evidence,
and applies a small versioned MLP without sending frames outside the machine.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flowmetric.domain import ServiceCandidate, ServiceDecision
from flowmetric.media import (
    load_video_frames,
    sample_evenly as _sample_evenly,
    service_crop_bounds,
    track_box_map as _track_box_map,
)
from .astra import (
    DEFAULT_RULE_HIGH,
    DEFAULT_RULE_LOW,
)

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights"

LOCAL_MLP_FORMAT_VERSION = 1
DEFAULT_LOCAL_SAMPLE_FPS = 4.0
DEFAULT_LOCAL_CONTEXT_SECONDS = 0.5
DEFAULT_LOCAL_MAX_FRAMES = 48
DEFAULT_MIN_LOCAL_FRAMES = 3
DEFAULT_MIN_POSE_COVERAGE = 0.34
DEFAULT_SERVICE_OBJECTS = frozenset(
    {
        "tray",
        "serving tray",
        "plate",
        "dish",
        "cup",
        "bowl",
        "wine glass",
        "bottle",
        "fork",
        "knife",
        "spoon",
    }
)

LOCAL_FEATURE_NAMES = (
    "fsm_confidence",
    "pose_coverage",
    "facing_toward_ratio",
    "mean_facing_score",
    "hand_customer_ratio",
    "hand_table_ratio",
    "mean_hand_proximity",
    "lean_toward_ratio",
    "service_object_ratio",
    "object_hand_ratio",
    "contact_duration_ratio",
    "contact_duration_s_scaled",
)

FrameLoader = Callable[[Path, tuple[int, ...]], Mapping[int, np.ndarray]]


@dataclass(frozen=True)
class PoseDetection:
    box: tuple[float, float, float, float]
    keypoints: np.ndarray
    keypoint_confidence: np.ndarray


@dataclass(frozen=True)
class ObjectDetection:
    box: tuple[float, float, float, float]
    label: str
    confidence: float


PoseBackend = Callable[[list[np.ndarray]], list[list[PoseDetection]]]
ObjectBackend = Callable[[list[np.ndarray]], list[list[ObjectDetection]]]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _probability(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _box(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain four xyxy coordinates")
    box = tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"{label} must be strictly ordered xyxy coordinates")
    return box


def select_local_service_frames(
    candidate: ServiceCandidate,
    tracks: dict[str, Any],
    *,
    sample_fps: float = DEFAULT_LOCAL_SAMPLE_FPS,
    context_seconds: float = DEFAULT_LOCAL_CONTEXT_SECONDS,
    max_frames: int = DEFAULT_LOCAL_MAX_FRAMES,
) -> tuple[int, ...]:
    """Sample a dense, chronological, jointly visible local sequence."""
    fps = _finite_number(tracks.get("fps"), "tracks fps")
    sample_fps = _finite_number(sample_fps, "sample_fps")
    context_seconds = _finite_number(context_seconds, "context_seconds")
    if fps <= 0 or sample_fps <= 0:
        raise ValueError("fps and sample_fps must be positive")
    if context_seconds < 0:
        raise ValueError("context_seconds must be non-negative")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 3:
        raise ValueError("max_frames must be an integer >= 3")

    interval = candidate.interval
    context = int(round(context_seconds * fps))
    lower = max(0, interval.start_frame - context)
    upper = interval.end_frame + context
    n_frames = tracks.get("n_frames")
    if isinstance(n_frames, int) and not isinstance(n_frames, bool) and n_frames > 0:
        upper = min(upper, n_frames - 1)
    staff = _track_box_map(tracks, interval.staff)
    customer = _track_box_map(tracks, interval.customer)
    joint = sorted(frame for frame in staff.keys() & customer.keys() if lower <= frame <= upper)
    if not joint:
        return ()

    step = max(1, int(round(fps / sample_fps)))
    sampled = [joint[0]]
    for frame in joint[1:]:
        if frame - sampled[-1] >= step:
            sampled.append(frame)
    if sampled[-1] != joint[-1]:
        sampled.append(joint[-1])
    return _sample_evenly(sampled, max_frames)


def _weights_path(name: str | Path) -> str:
    path = Path(name)
    return str(WEIGHTS / path) if path.parent == Path(".") else str(path)


class UltralyticsPoseBackend:
    """Lazy YOLO pose inference on candidate crops."""

    def __init__(
        self,
        model: str | Path = "yolo11n-pose.pt",
        *,
        confidence: float = 0.15,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = _weights_path(model)
        self.model_id = Path(self.model_path).name
        self.confidence = _probability(confidence, "pose confidence")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("pose imgsz must be a positive integer")
        self.imgsz = imgsz
        self.device = device
        self._model: Any = None

    def __call__(self, images: list[np.ndarray]) -> list[list[PoseDetection]]:
        if not images:
            return []
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        results = self._model.predict(
            images,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        output: list[list[PoseDetection]] = []
        for result in results:
            detections: list[PoseDetection] = []
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                points = result.keypoints.xy.detach().cpu().numpy()
                raw_conf = result.keypoints.conf
                confidence = (
                    raw_conf.detach().cpu().numpy()
                    if raw_conf is not None
                    else np.ones(points.shape[:2], dtype=float)
                )
                for box, keypoints, keypoint_confidence in zip(boxes, points, confidence):
                    detections.append(
                        PoseDetection(
                            box=tuple(float(value) for value in box),
                            keypoints=np.asarray(keypoints, dtype=float),
                            keypoint_confidence=np.asarray(keypoint_confidence, dtype=float),
                        )
                    )
            output.append(detections)
        return output


class UltralyticsObjectBackend:
    """Lazy YOLO object inference for cups, dishes, trays, and utensils."""

    def __init__(
        self,
        model: str | Path = "yolo11n.pt",
        *,
        confidence: float = 0.15,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = _weights_path(model)
        self.model_id = Path(self.model_path).name
        self.confidence = _probability(confidence, "object confidence")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("object imgsz must be a positive integer")
        self.imgsz = imgsz
        self.device = device
        self._model: Any = None

    def __call__(self, images: list[np.ndarray]) -> list[list[ObjectDetection]]:
        if not images:
            return []
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        results = self._model.predict(
            images,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        output: list[list[ObjectDetection]] = []
        for result in results:
            detections: list[ObjectDetection] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                confidence = result.boxes.conf.detach().cpu().numpy()
                names = result.names
                for box, class_id, score in zip(boxes, classes, confidence):
                    label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
                    detections.append(
                        ObjectDetection(
                            box=tuple(float(value) for value in box),
                            label=str(label).strip().lower().replace("_", " "),
                            confidence=float(score),
                        )
                    )
            output.append(detections)
        return output


class LocalServiceMLP:
    """Strict, NumPy-only runtime for a small versioned binary MLP."""

    def __init__(self, payload: dict[str, Any], *, model_id: str) -> None:
        required = {
            "format_version",
            "feature_names",
            "mean",
            "scale",
            "layers",
            "threshold",
            "metadata",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("local service MLP has an unknown or malformed format")
        if type(payload["format_version"]) is not int or payload["format_version"] != LOCAL_MLP_FORMAT_VERSION:
            raise ValueError(
                f"unsupported local service MLP version {payload['format_version']!r}"
            )
        feature_names = payload["feature_names"]
        if not isinstance(feature_names, list) or tuple(feature_names) != LOCAL_FEATURE_NAMES:
            raise ValueError("local service MLP feature_names do not match runtime features")
        width = len(LOCAL_FEATURE_NAMES)
        self.mean = np.asarray(payload["mean"], dtype=float)
        self.scale = np.asarray(payload["scale"], dtype=float)
        if self.mean.shape != (width,) or self.scale.shape != (width,):
            raise ValueError("local service MLP mean/scale shape mismatch")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("local service MLP normalization must be finite with positive scale")
        if not isinstance(payload["metadata"], dict):
            raise ValueError("local service MLP metadata must be an object")
        self.metadata = payload["metadata"]
        self.threshold = _probability(payload["threshold"], "local service MLP threshold")
        self.model_id = model_id
        self.layers: list[tuple[np.ndarray, np.ndarray, str]] = []
        input_width = width
        raw_layers = payload["layers"]
        if not isinstance(raw_layers, list) or len(raw_layers) < 2:
            raise ValueError("local service MLP must contain at least two layers")
        for index, raw_layer in enumerate(raw_layers):
            if not isinstance(raw_layer, dict) or set(raw_layer) != {"weights", "bias", "activation"}:
                raise ValueError(f"local service MLP layer {index} is malformed")
            weights = np.asarray(raw_layer["weights"], dtype=float)
            bias = np.asarray(raw_layer["bias"], dtype=float)
            activation = raw_layer["activation"]
            if weights.ndim != 2 or bias.ndim != 1 or weights.shape != (len(bias), input_width):
                raise ValueError(f"local service MLP layer {index} shape mismatch")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError(f"local service MLP layer {index} contains non-finite values")
            if activation not in {"relu", "sigmoid", "linear"}:
                raise ValueError(f"local service MLP layer {index} has unknown activation")
            self.layers.append((weights, bias, activation))
            input_width = len(bias)
        if input_width != 1 or self.layers[-1][2] != "sigmoid":
            raise ValueError("local service MLP must end in one sigmoid output")

    @classmethod
    def load(cls, path: Path) -> "LocalServiceMLP":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load local service MLP {path}: {exc}") from exc
        return cls(payload, model_id=path.name)

    def predict_proba(self, features: Mapping[str, float]) -> float:
        if set(features) != set(LOCAL_FEATURE_NAMES):
            raise ValueError("local service feature set does not match the MLP model")
        vector = np.asarray([features[name] for name in LOCAL_FEATURE_NAMES], dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError("local service features must be finite")
        vector = (vector - self.mean) / self.scale
        for weights, bias, activation in self.layers:
            vector = weights @ vector + bias
            if activation == "relu":
                vector = np.maximum(vector, 0.0)
            elif activation == "sigmoid":
                vector = 1.0 / (1.0 + np.exp(-np.clip(vector, -60.0, 60.0)))
        return _probability(float(vector[0]), "local service MLP probability")


def _intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(min(a, b), 1e-9)


def _point_box_distance(point: np.ndarray, box: tuple[float, float, float, float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return float(math.hypot(dx, dy))


def _center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def _shift_box(
    box: tuple[float, float, float, float], bounds: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    return box[0] - bounds[0], box[1] - bounds[1], box[2] - bounds[0], box[3] - bounds[1]


def _valid_keypoint(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    index: int,
    *,
    threshold: float = 0.2,
) -> np.ndarray | None:
    if keypoints.ndim != 2 or keypoints.shape[1] != 2 or index >= len(keypoints):
        return None
    if confidence.ndim != 1 or index >= len(confidence) or confidence[index] < threshold:
        return None
    point = np.asarray(keypoints[index], dtype=float)
    return point if np.isfinite(point).all() else None


def _mean_points(points: Sequence[np.ndarray | None]) -> np.ndarray | None:
    valid = [point for point in points if point is not None]
    return np.mean(valid, axis=0) if valid else None


def _match_staff_pose(
    staff_box: tuple[float, float, float, float],
    detections: Sequence[PoseDetection],
) -> PoseDetection | None:
    matches = [
        (_intersection_over_smaller(staff_box, _box(detection.box, "pose box")), detection)
        for detection in detections
    ]
    if not matches:
        return None
    overlap, detection = max(matches, key=lambda item: item[0])
    return detection if overlap >= 0.45 else None


def _expanded_interaction_box(
    staff: tuple[float, float, float, float],
    customer: tuple[float, float, float, float],
    table: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    boxes = [staff, customer] + ([table] if table is not None else [])
    height = max(staff[3] - staff[1], 1.0)
    margin = 0.25 * height
    return (
        min(box[0] for box in boxes) - margin,
        min(box[1] for box in boxes) - margin,
        max(box[2] for box in boxes) + margin,
        max(box[3] for box in boxes) + margin,
    )


def extract_local_service_features(
    candidate: ServiceCandidate,
    frames: tuple[int, ...],
    staff_boxes: Mapping[int, tuple[float, float, float, float]],
    customer_boxes: Mapping[int, tuple[float, float, float, float]],
    table_box: tuple[float, float, float, float] | None,
    pose_detections: Mapping[int, Sequence[PoseDetection]],
    object_detections: Mapping[int, Sequence[ObjectDetection]],
    *,
    fps: float,
    service_objects: frozenset[str] = DEFAULT_SERVICE_OBJECTS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Aggregate pose, hand, object, lean, and contact-duration evidence."""
    fps = _finite_number(fps, "fps")
    if fps <= 0 or not frames:
        raise ValueError("feature extraction needs positive fps and sampled frames")

    pose_frames = 0
    facing_scores: list[float] = []
    hand_customer: list[bool] = []
    hand_table: list[bool] = []
    hand_proximity: list[float] = []
    lean_scores: list[float] = []
    service_object_frames = 0
    object_hand_frames = 0
    contact_by_frame: list[tuple[int, bool]] = []
    label_counts: Counter[str] = Counter()

    for frame in frames:
        staff = staff_boxes[frame]
        customer = customer_boxes[frame]
        table = table_box
        staff_height = max(staff[3] - staff[1], 1.0)
        target = _center(table) if table is not None else np.asarray(
            [(customer[0] + customer[2]) / 2, customer[1] + 0.25 * (customer[3] - customer[1])]
        )
        pose = _match_staff_pose(staff, pose_detections.get(frame, ()))
        wrists: list[np.ndarray] = []
        facing_score = 0.5
        lean_score = 0.0
        near_customer = False
        near_table = False
        proximity = 0.0
        if pose is not None:
            keypoints = np.asarray(pose.keypoints, dtype=float)
            confidence = np.asarray(pose.keypoint_confidence, dtype=float)
            shoulders = _mean_points(
                [
                    _valid_keypoint(keypoints, confidence, 5),
                    _valid_keypoint(keypoints, confidence, 6),
                ]
            )
            hips = _mean_points(
                [
                    _valid_keypoint(keypoints, confidence, 11),
                    _valid_keypoint(keypoints, confidence, 12),
                ]
            )
            nose = _valid_keypoint(keypoints, confidence, 0)
            wrists = [
                point
                for point in (
                    _valid_keypoint(keypoints, confidence, 9),
                    _valid_keypoint(keypoints, confidence, 10),
                )
                if point is not None
            ]
            pose_frames += 1
            if shoulders is not None and nose is not None:
                face_dx = float(nose[0] - shoulders[0])
                target_dx = float(target[0] - shoulders[0])
                direction = 0.0 if abs(target_dx) < 1e-6 else math.copysign(1.0, target_dx)
                facing_score = float(
                    np.clip(0.5 + 0.5 * direction * face_dx / (0.10 * staff_height), 0.0, 1.0)
                )
            if shoulders is not None and hips is not None:
                target_dx = float(target[0] - hips[0])
                direction = 0.0 if abs(target_dx) < 1e-6 else math.copysign(1.0, target_dx)
                lean_score = float(
                    np.clip(direction * (shoulders[0] - hips[0]) / (0.12 * staff_height), 0.0, 1.0)
                )
            if wrists:
                customer_distance = min(_point_box_distance(wrist, customer) for wrist in wrists)
                table_distance = (
                    min(_point_box_distance(wrist, table) for wrist in wrists)
                    if table is not None
                    else math.inf
                )
                minimum_distance = min(customer_distance, table_distance)
                near_customer = customer_distance <= 0.20 * staff_height
                near_table = table_distance <= 0.20 * staff_height
                proximity = float(math.exp(-minimum_distance / (0.25 * staff_height)))
            facing_scores.append(facing_score)
            lean_scores.append(lean_score)
            hand_customer.append(near_customer)
            hand_table.append(near_table)
            hand_proximity.append(proximity)

        interaction_box = _expanded_interaction_box(staff, customer, table)
        relevant: list[ObjectDetection] = []
        for detection in object_detections.get(frame, ()):
            label = detection.label.strip().lower().replace("_", " ")
            if label not in service_objects:
                continue
            box = _box(detection.box, "object box")
            if _point_box_distance(_center(box), interaction_box) > 0:
                continue
            relevant.append(detection)
            label_counts[label] += 1
        object_present = bool(relevant)
        if object_present:
            service_object_frames += 1
        object_near_hand = False
        if wrists and relevant:
            object_near_hand = any(
                min(_point_box_distance(wrist, _box(item.box, "object box")) for wrist in wrists)
                <= 0.12 * staff_height
                for item in relevant
            )
        if object_near_hand:
            object_hand_frames += 1
        contact = (near_customer or near_table) and (
            facing_score >= 0.55 or lean_score >= 0.15 or object_near_hand
        )
        contact_by_frame.append((frame, contact))

    sample_count = len(frames)
    pose_denominator = max(pose_frames, 1)
    nominal_step = max(1.0, float(np.median(np.diff(frames))) if len(frames) > 1 else fps)
    max_gap = max(1, int(round(1.75 * nominal_step)))
    longest_start: int | None = None
    longest_end: int | None = None
    current_start: int | None = None
    previous_frame: int | None = None
    for frame, active in contact_by_frame:
        if active and (current_start is None or previous_frame is None or frame - previous_frame <= max_gap):
            current_start = frame if current_start is None else current_start
        elif active:
            current_start = frame
        else:
            current_start = None
        if active and current_start is not None:
            if longest_start is None or frame - current_start > longest_end - longest_start:
                longest_start, longest_end = current_start, frame
        previous_frame = frame if active else None
    contact_duration = (
        (longest_end - longest_start + nominal_step) / fps
        if longest_start is not None and longest_end is not None
        else 0.0
    )
    interval_duration = max(
        (candidate.interval.end_frame - candidate.interval.start_frame + 1) / fps,
        nominal_step / fps,
    )

    features = {
        "fsm_confidence": _probability(candidate.interval.confidence, "FSM confidence"),
        "pose_coverage": pose_frames / sample_count,
        "facing_toward_ratio": sum(score >= 0.55 for score in facing_scores) / pose_denominator,
        "mean_facing_score": float(np.mean(facing_scores)) if facing_scores else 0.0,
        "hand_customer_ratio": sum(hand_customer) / pose_denominator,
        "hand_table_ratio": sum(hand_table) / pose_denominator,
        "mean_hand_proximity": float(np.mean(hand_proximity)) if hand_proximity else 0.0,
        "lean_toward_ratio": sum(score >= 0.15 for score in lean_scores) / pose_denominator,
        "service_object_ratio": service_object_frames / sample_count,
        "object_hand_ratio": object_hand_frames / pose_denominator,
        "contact_duration_ratio": float(np.clip(contact_duration / interval_duration, 0.0, 1.0)),
        "contact_duration_s_scaled": float(np.clip(contact_duration / 2.0, 0.0, 1.0)),
    }
    features = {name: round(_probability(value, name), 6) for name, value in features.items()}
    evidence = {
        "sampled_frame_count": sample_count,
        "matched_pose_frames": pose_frames,
        "service_object_frames": service_object_frames,
        "object_hand_frames": object_hand_frames,
        "contact_duration_s": round(contact_duration, 3),
        "objects": dict(sorted(label_counts.items())),
    }
    return features, evidence


class LocalServiceVerifier:
    """Triage candidates and confirm ambiguous episodes entirely on-device."""

    def __init__(
        self,
        source: Path,
        tracks: dict[str, Any],
        *,
        classifier_path: Path,
        pose_model: str | Path = "yolo11n-pose.pt",
        object_model: str | Path = "yolo11n.pt",
        rule_low: float = DEFAULT_RULE_LOW,
        rule_high: float = DEFAULT_RULE_HIGH,
        decision_threshold: float | None = None,
        sample_fps: float = DEFAULT_LOCAL_SAMPLE_FPS,
        context_seconds: float = DEFAULT_LOCAL_CONTEXT_SECONDS,
        max_frames: int = DEFAULT_LOCAL_MAX_FRAMES,
        imgsz: int = 640,
        device: str | None = None,
        pose_confidence: float = 0.15,
        object_confidence: float = 0.15,
        min_pose_coverage: float = DEFAULT_MIN_POSE_COVERAGE,
        service_objects: frozenset[str] = DEFAULT_SERVICE_OBJECTS,
        pose_backend: PoseBackend | None = None,
        object_backend: ObjectBackend | None = None,
        frame_loader: FrameLoader | None = None,
    ) -> None:
        self.source = Path(source)
        if not isinstance(tracks, dict):
            raise TypeError("tracks must be the tracks.json object")
        self.tracks = tracks
        self.classifier = LocalServiceMLP.load(Path(classifier_path))
        self.rule_low = _probability(rule_low, "rule_low")
        self.rule_high = _probability(rule_high, "rule_high")
        if self.rule_low > self.rule_high:
            raise ValueError("rule_low must not exceed rule_high")
        self.threshold = (
            self.classifier.threshold
            if decision_threshold is None
            else _probability(decision_threshold, "decision_threshold")
        )
        self.sample_fps = _finite_number(sample_fps, "sample_fps")
        self.context_seconds = _finite_number(context_seconds, "context_seconds")
        self.min_pose_coverage = _probability(min_pose_coverage, "min_pose_coverage")
        if self.sample_fps <= 0 or self.context_seconds < 0:
            raise ValueError("sample_fps must be positive and context_seconds non-negative")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 3:
            raise ValueError("max_frames must be an integer >= 3")
        self.max_frames = max_frames
        if not isinstance(service_objects, frozenset) or not service_objects:
            raise ValueError("service_objects must be a non-empty frozenset")
        self.service_objects = frozenset(
            label.strip().lower().replace("_", " ") for label in service_objects
        )
        self.pose_backend = pose_backend or UltralyticsPoseBackend(
            pose_model,
            confidence=pose_confidence,
            imgsz=imgsz,
            device=device,
        )
        self.object_backend = object_backend or UltralyticsObjectBackend(
            object_model,
            confidence=object_confidence,
            imgsz=imgsz,
            device=device,
        )
        self.pose_model_id = getattr(self.pose_backend, "model_id", str(pose_model))
        self.object_model_id = getattr(self.object_backend, "model_id", str(object_model))
        self.frame_loader = frame_loader or load_video_frames

    def __call__(self, candidates: list[ServiceCandidate]) -> list[ServiceDecision]:
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")
        decisions: list[ServiceDecision | None] = [None] * len(candidates)
        sampled: list[tuple[int, ...]] = [()] * len(candidates)
        request_indexes: list[int] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, ServiceCandidate):
                raise TypeError("candidates must contain ServiceCandidate instances")
            confidence = _probability(candidate.interval.confidence, "candidate rule confidence")
            if confidence < self.rule_low:
                decisions[index] = ServiceDecision(
                    False,
                    candidate.interval.staff,
                    candidate.interval.customer,
                    1.0 - confidence,
                    "rules_low_confidence",
                )
            elif confidence > self.rule_high:
                decisions[index] = ServiceDecision(
                    True,
                    candidate.interval.staff,
                    candidate.interval.customer,
                    confidence,
                    "rules_high_confidence",
                )
            else:
                frames = select_local_service_frames(
                    candidate,
                    self.tracks,
                    sample_fps=self.sample_fps,
                    context_seconds=self.context_seconds,
                    max_frames=self.max_frames,
                )
                sampled[index] = frames
                if len(frames) < DEFAULT_MIN_LOCAL_FRAMES:
                    decisions[index] = ServiceDecision(
                        True,
                        candidate.interval.staff,
                        candidate.interval.customer,
                        confidence,
                        "insufficient_local_frames",
                        sampled_frames=frames,
                    )
                else:
                    request_indexes.append(index)

        if not request_indexes:
            return [decision for decision in decisions if decision is not None]

        union_frames = tuple(
            sorted({frame for index in request_indexes for frame in sampled[index]})
        )
        loaded = self.frame_loader(self.source, union_frames)
        if not isinstance(loaded, Mapping):
            raise TypeError("frame_loader must return a frame-index mapping")
        missing = set(union_frames) - set(loaded)
        if missing:
            raise RuntimeError(f"frame_loader omitted requested frames: {sorted(missing)}")

        images: list[np.ndarray] = []
        image_owner: list[tuple[int, int]] = []
        shifted_staff: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        shifted_customer: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        shifted_table: dict[int, tuple[float, float, float, float] | None] = {}
        for candidate_index in request_indexes:
            candidate = candidates[candidate_index]
            frames = sampled[candidate_index]
            staff_map = _track_box_map(self.tracks, candidate.interval.staff)
            customer_map = _track_box_map(self.tracks, candidate.interval.customer)
            shapes = {tuple(loaded[frame].shape) for frame in frames}
            if len(shapes) != 1:
                raise ValueError("all sampled local-service frames must have the same shape")
            bounds = service_crop_bounds(
                loaded[frames[0]].shape,
                staff_boxes=[staff_map[frame] for frame in frames],
                customer_boxes=[customer_map[frame] for frame in frames],
                table_box=candidate.table_box,
                context_margin=0.15,
            )
            left, top, right, bottom = bounds
            shifted_staff[candidate_index] = {
                frame: _shift_box(staff_map[frame], bounds) for frame in frames
            }
            shifted_customer[candidate_index] = {
                frame: _shift_box(customer_map[frame], bounds) for frame in frames
            }
            shifted_table[candidate_index] = (
                _shift_box(candidate.table_box, bounds)
                if candidate.table_box is not None
                else None
            )
            for frame in frames:
                images.append(loaded[frame][top:bottom, left:right].copy())
                image_owner.append((candidate_index, frame))

        pose_results = self.pose_backend(images)
        object_results = self.object_backend(images)
        if len(pose_results) != len(images) or len(object_results) != len(images):
            raise RuntimeError("local pose/object backend result count mismatch")
        pose_by_candidate: dict[int, dict[int, Sequence[PoseDetection]]] = {
            index: {} for index in request_indexes
        }
        object_by_candidate: dict[int, dict[int, Sequence[ObjectDetection]]] = {
            index: {} for index in request_indexes
        }
        for owner, poses, objects in zip(image_owner, pose_results, object_results):
            candidate_index, frame = owner
            pose_by_candidate[candidate_index][frame] = poses
            object_by_candidate[candidate_index][frame] = objects

        fps = _finite_number(self.tracks.get("fps"), "tracks fps")
        for candidate_index in request_indexes:
            candidate = candidates[candidate_index]
            frames = sampled[candidate_index]
            features, evidence = extract_local_service_features(
                candidate,
                frames,
                shifted_staff[candidate_index],
                shifted_customer[candidate_index],
                shifted_table[candidate_index],
                pose_by_candidate[candidate_index],
                object_by_candidate[candidate_index],
                fps=fps,
                service_objects=self.service_objects,
            )
            details: dict[str, Any] = {
                "features": features,
                "evidence": evidence,
                "classifier_threshold": self.threshold,
                "pose_model": self.pose_model_id,
                "object_model": self.object_model_id,
            }
            if (
                evidence["matched_pose_frames"] < DEFAULT_MIN_LOCAL_FRAMES
                or features["pose_coverage"] < self.min_pose_coverage
            ):
                decisions[candidate_index] = ServiceDecision(
                    True,
                    candidate.interval.staff,
                    candidate.interval.customer,
                    candidate.interval.confidence,
                    "insufficient_pose_evidence",
                    sampled_frames=frames,
                    model=self.classifier.model_id,
                    details=details,
                )
                continue
            probability = self.classifier.predict_proba(features)
            is_service = probability >= self.threshold
            details["classifier_probability"] = round(probability, 6)
            decisions[candidate_index] = ServiceDecision(
                is_service,
                candidate.interval.staff,
                candidate.interval.customer,
                probability if is_service else 1.0 - probability,
                "local_mlp",
                sampled_frames=frames,
                model=self.classifier.model_id,
                details=details,
            )

        if any(decision is None for decision in decisions):
            raise RuntimeError("internal error: missing local service decision")
        return [decision for decision in decisions if decision is not None]
