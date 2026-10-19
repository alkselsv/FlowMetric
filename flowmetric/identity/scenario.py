"""Restaurant-scene lifecycle evidence for offline tracklet association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from flowmetric.tracking import LocalPass


@dataclass(frozen=True)
class TrackletLifecycle:
    tracklet_id: int
    first_frame: int
    last_frame: int
    first_foot: np.ndarray
    last_foot: np.ndarray
    first_height: float
    last_height: float
    end_velocity: np.ndarray
    first_zone: str
    last_zone: str
    first_boundary: str | None
    last_boundary: str | None
    exits_scene: bool


@dataclass(frozen=True)
class ScenarioAssessment:
    compatible: bool
    reason: str | None
    signals: tuple[str, ...]
    confidence_bonus: float
    gap_s: float | None
    required_speed: float | None
    residual_speed: float | None

    @property
    def supported(self) -> bool:
        return len(self.signals) >= 2

    def as_dict(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "reason": self.reason,
            "signals": list(self.signals),
            "supported": self.supported,
            "confidence_bonus": round(self.confidence_bonus, 4),
            "gap_s": round(self.gap_s, 4) if self.gap_s is not None else None,
            "required_speed": (
                round(self.required_speed, 4)
                if self.required_speed is not None
                else None
            ),
            "residual_speed": (
                round(self.residual_speed, 4)
                if self.residual_speed is not None
                else None
            ),
        }


def _foot(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2, box[3]], dtype=float)


def _in_zone(point: np.ndarray, polygon: object) -> bool:
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(
        np.asarray(polygon, np.float32),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0


def _zone(point: np.ndarray, scene: dict[str, Any]) -> str:
    for name in ("door", "kitchen"):
        if _in_zone(point, scene.get(name)):
            return name
    return "hall"


def _boundary(
    point: np.ndarray,
    *,
    width: int,
    height: int,
    margin: float,
    scene: dict[str, Any],
) -> str | None:
    if _in_zone(point, scene.get("door")):
        return "door"
    if point[0] < margin * width:
        return "left"
    if point[0] > (1.0 - margin) * width:
        return "right"
    if point[1] > (1.0 - margin) * height:
        return "bottom"
    return None


def _velocity(items: list[tuple[int, np.ndarray]], fps: float, height: float) -> np.ndarray:
    recent = items[-4:]
    if len(recent) < 2:
        return np.zeros(2, dtype=float)
    frame_delta = recent[-1][0] - recent[0][0]
    if frame_delta <= 0:
        return np.zeros(2, dtype=float)
    return (
        (_foot(recent[-1][1]) - _foot(recent[0][1]))
        * fps
        / frame_delta
        / max(height, 1.0)
    )


def build_tracklet_lifecycles(
    local_pass: LocalPass,
    scene: dict[str, Any],
) -> dict[int, TrackletLifecycle]:
    """Summarize endpoints, zones and exit state for every local tracklet."""
    margin = float(scene.get("edge_margin", 0.06))
    observations: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    for frame, detections in local_pass.dets.items():
        for tracklet_id, box, confidence in detections:
            by_frame = observations.setdefault(int(tracklet_id), {})
            previous = by_frame.get(int(frame))
            if previous is None or confidence > previous[1]:
                by_frame[int(frame)] = (np.asarray(box, dtype=float), confidence)

    result: dict[int, TrackletLifecycle] = {}
    for tracklet_id, by_frame in observations.items():
        items = [(frame, value[0]) for frame, value in sorted(by_frame.items())]
        first_frame, first_box = items[0]
        last_frame, last_box = items[-1]
        first_foot, last_foot = _foot(first_box), _foot(last_box)
        first_height = max(float(first_box[3] - first_box[1]), 1.0)
        last_height = max(float(last_box[3] - last_box[1]), 1.0)
        first_boundary = _boundary(
            first_foot,
            width=local_pass.width,
            height=local_pass.height,
            margin=margin,
            scene=scene,
        )
        last_boundary = _boundary(
            last_foot,
            width=local_pass.width,
            height=local_pass.height,
            margin=margin,
            scene=scene,
        )
        velocity = _velocity(items, local_pass.fps, last_height)
        outward = False
        if last_boundary == "left":
            outward = velocity[0] < -0.15
        elif last_boundary == "right":
            outward = velocity[0] > 0.15
        elif last_boundary == "bottom":
            outward = velocity[1] > 0.15
        elif last_boundary == "door" and len(items) >= 2:
            polygon = np.asarray(scene.get("door"), dtype=float)
            target = polygon.mean(axis=0)
            movement = last_foot - _foot(items[-2][1])
            outward = float(np.dot(movement, target - last_foot)) > 0
        result[tracklet_id] = TrackletLifecycle(
            tracklet_id=tracklet_id,
            first_frame=first_frame,
            last_frame=last_frame,
            first_foot=first_foot,
            last_foot=last_foot,
            first_height=first_height,
            last_height=last_height,
            end_velocity=velocity,
            first_zone=_zone(first_foot, scene),
            last_zone=_zone(last_foot, scene),
            first_boundary=first_boundary,
            last_boundary=last_boundary,
            exits_scene=last_boundary is not None and outward,
        )
    return result


def assess_scenario_link(
    left_id: int,
    right_id: int,
    *,
    lifecycles: dict[int, TrackletLifecycle],
    fps: float,
) -> ScenarioAssessment:
    """Return conservative restaurant-lifecycle support or a hard conflict."""
    left, right = lifecycles[left_id], lifecycles[right_id]
    earlier, later = (left, right)
    if right.last_frame < left.first_frame:
        earlier, later = right, left
    gap_frames = later.first_frame - earlier.last_frame
    if gap_frames <= 0:
        return ScenarioAssessment(True, None, (), 0.0, None, None, None)

    gap_s = gap_frames / fps
    scale = max((earlier.last_height + later.first_height) / 2, 1.0)
    displacement = later.first_foot - earlier.last_foot
    endpoint_distance = float(np.linalg.norm(displacement) / scale)
    required_speed = endpoint_distance / max(gap_s, 1e-6)
    normalized_displacement = displacement / scale
    residual_speed = float(
        np.linalg.norm(
            normalized_displacement - earlier.end_velocity * gap_s
        )
        / max(gap_s, 1e-6)
    )

    if required_speed > 3.0 and residual_speed > 2.0:
        return ScenarioAssessment(
            False,
            "scenario_implausible_speed",
            (),
            0.0,
            gap_s,
            required_speed,
            residual_speed,
        )
    if earlier.exits_scene and (
        later.first_boundary != earlier.last_boundary or gap_s > 0.5
    ):
        return ScenarioAssessment(
            False,
            "scenario_completed_exit",
            (),
            0.0,
            gap_s,
            required_speed,
            residual_speed,
        )
    if (
        {earlier.last_zone, later.first_zone} == {"door", "kitchen"}
        and required_speed > 1.5
    ):
        return ScenarioAssessment(
            False,
            "scenario_incompatible_zones",
            (),
            0.0,
            gap_s,
            required_speed,
            residual_speed,
        )

    signals: list[str] = []
    if earlier.last_zone == later.first_zone and earlier.last_zone != "hall":
        signals.append(f"same_{earlier.last_zone}_zone")
    if (
        earlier.last_boundary is not None
        and earlier.last_boundary == later.first_boundary
        and gap_s <= 1.0
        and endpoint_distance <= 0.8
    ):
        signals.append("same_boundary")
    if (
        earlier.last_zone == later.first_zone == "hall"
        and earlier.last_boundary is None
        and later.first_boundary is None
        and gap_s <= 1.5
        and endpoint_distance <= 0.65
    ):
        signals.append("interior_occlusion")
    velocity_norm = float(np.linalg.norm(earlier.end_velocity))
    displacement_norm = float(np.linalg.norm(displacement))
    if velocity_norm >= 0.10 and displacement_norm > 1e-6:
        direction = float(
            np.dot(earlier.end_velocity, displacement)
            / (velocity_norm * displacement_norm)
        )
        if direction >= 0.5:
            signals.append("direction_consistent")

    return ScenarioAssessment(
        True,
        None,
        tuple(signals),
        min(0.03, 0.015 * len(signals)),
        gap_s,
        required_speed,
        residual_speed,
    )
