"""Validated scene calibration for zones and optional static table geometry."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any


class SceneConfigError(ValueError):
    """Scene calibration cannot safely be applied to the input video."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneConfigError(f"Cannot read {label} config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SceneConfigError(f"{label} config {path} must contain a JSON object")
    return value


def _validate_resolution(config: dict[str, Any], path: Path, width: int, height: int) -> None:
    resolution = config.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in resolution)
    ):
        raise SceneConfigError(
            f"Scene config {path} must declare resolution as [width, height]"
        )
    if resolution != [width, height]:
        raise SceneConfigError(
            f"Scene config {path} is calibrated for {resolution[0]}x{resolution[1]}, "
            f"but the video is {width}x{height}"
        )


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneConfigError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SceneConfigError(f"{label} must be a finite number")
    return result


def _validate_polygon(value: Any, label: str, width: int, height: int) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise SceneConfigError(f"{label} must contain at least three points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise SceneConfigError(f"{label}[{index}] must be [x, y]")
        x = _number(point[0], f"{label}[{index}].x")
        y = _number(point[1], f"{label}[{index}].y")
        if not 0 <= x <= width or not 0 <= y <= height:
            raise SceneConfigError(f"{label}[{index}] lies outside the calibrated frame")
        points.append((x, y))
    twice_area = abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ))
    if twice_area <= 1e-6:
        raise SceneConfigError(f"{label} must enclose a non-zero area")


def _validate_table(value: Any, label: str, width: int, height: int) -> None:
    box = value.get("box") if isinstance(value, dict) else value
    if not isinstance(box, list) or len(box) != 4:
        raise SceneConfigError(f"{label}.box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = [_number(v, f"{label}.box") for v in box]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SceneConfigError(f"{label}.box is invalid or outside the calibrated frame")
    if isinstance(value, dict) and "support" in value:
        support = _number(value["support"], f"{label}.support")
        if not 0 <= support <= 1:
            raise SceneConfigError(f"{label}.support must be between 0 and 1")


def load_scene_config(
    camera_path: Path,
    *,
    width: int,
    height: int,
    source: Path,
    tables_path: Path | None = None,
) -> dict[str, Any]:
    """Load and validate pixel-space calibration, optionally merging table boxes."""
    camera_path = Path(camera_path)
    camera = _read_object(camera_path, "camera")
    _validate_resolution(camera, camera_path, width, height)
    for zone in ("door", "kitchen"):
        if zone in camera:
            _validate_polygon(camera[zone], f"camera.{zone}", width, height)
    margin = _number(camera.get("edge_margin", 0.06), "camera.edge_margin")
    if not 0 <= margin < 0.5:
        raise SceneConfigError("camera.edge_margin must be in [0, 0.5)")

    embedded_tables = camera.get("tables")
    if embedded_tables is not None:
        if not isinstance(embedded_tables, list):
            raise SceneConfigError(f"Camera config {camera_path} tables must be an array")
        for index, table in enumerate(embedded_tables):
            _validate_table(table, f"camera.tables[{index}]", width, height)

    if tables_path is None:
        return camera

    tables_path = Path(tables_path)
    table_config = _read_object(tables_path, "tables")
    _validate_resolution(table_config, tables_path, width, height)
    calibrated_source = table_config.get("source")
    if calibrated_source and calibrated_source != source.name:
        warnings.warn(
            f"Table map {tables_path} was generated from {calibrated_source!r}, "
            f"but is being applied to {source.name!r}; ensure the camera view is unchanged",
            stacklevel=2,
        )
    tables = table_config.get("tables")
    if not isinstance(tables, list):
        raise SceneConfigError(f"Tables config {tables_path} must contain a tables array")
    for index, table in enumerate(tables):
        _validate_table(table, f"tables[{index}]", width, height)
    camera["tables"] = tables
    camera["tables_source"] = str(tables_path)
    return camera
