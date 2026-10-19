"""GPT-6 Astra client for sparse-frame person ReID (hybrid FlowMetric stage)."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ASTRA_CACHE_VERSION = 1
ASTRA_PROMPT_VERSION = "2026-10-18.tracklet-gallery-targeted-v3"
ASTRA_SCHEMA_VERSION = "2"
VALID_ROLES = frozenset({"customer", "staff", "unknown"})


class AstraResponseError(RuntimeError):
    """A malformed Astra result that can be retried without side effects."""

    retry_safe = True


class AstraCacheError(RuntimeError):
    """An Astra cache that is unsafe to reuse for the current source/config."""


REID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Stable identity across all provided frames",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["customer", "staff", "unknown"],
                    },
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame_index": {
                                    "type": "integer",
                                    "description": "0-based index in the provided frame list",
                                },
                                "xyxy": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            },
                            "required": ["frame_index", "xyxy"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "role", "detections"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["people", "notes"],
    "additionalProperties": False,
}


def bgr_to_data_url(bgr: np.ndarray, max_side: int = 1280) -> tuple[str, float, int, int]:
    """Encode a BGR image as an inline JPEG URL, optionally resizing it.

    The returned tuple contains ``(url, scale, width, height)``.  This helper is
    shared by the ReID and service-verification Astra calls so both paths use
    the same bounded, deterministic image encoding.
    """
    if not isinstance(bgr, np.ndarray) or bgr.ndim not in (2, 3):
        raise ValueError("bgr must be a 2D/3D numpy array")
    if bgr.shape[0] <= 0 or bgr.shape[1] <= 0:
        raise ValueError("bgr must have non-empty image dimensions")
    if (
        isinstance(max_side, bool)
        or not isinstance(max_side, (int, float))
        or not math.isfinite(float(max_side))
        or max_side <= 0
    ):
        raise ValueError("max_side must be positive")
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    url = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"
    rh, rw = bgr.shape[:2]
    return url, scale, rw, rh


# Kept as an alias for callers that used the module-private helper.
_bgr_to_data_url = bgr_to_data_url


VSELLM_BASE_URL = "https://api.vsellm.ru/v1"
ASTRA_REQUEST_TIMEOUT_S = 90.0
ASTRA_TRANSPORT_RETRIES = 2


def configured_default_model_id() -> str | None:
    """Return the provider-specific default model without constructing a client."""
    if os.environ.get("VSELLM_API_KEY"):
        return "openai/gpt-6-astra"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-6-astra"
    return None


def make_client() -> tuple[OpenAI, str]:
    """Return (client, default Astra model id) for the configured provider.

    VSELLM_API_KEY → VseLLM (OpenAI-compatible proxy, model ``openai/gpt-6-astra``);
    otherwise OPENAI_API_KEY → OpenAI directly (``gpt-6-astra``).
    """
    if key := os.environ.get("VSELLM_API_KEY"):
        base_url = os.environ.get("VSELLM_BASE_URL", VSELLM_BASE_URL)
        return (
            OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=ASTRA_REQUEST_TIMEOUT_S,
                max_retries=ASTRA_TRANSPORT_RETRIES,
            ),
            "openai/gpt-6-astra",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return (
            OpenAI(
                timeout=ASTRA_REQUEST_TIMEOUT_S,
                max_retries=ASTRA_TRANSPORT_RETRIES,
            ),
            "gpt-6-astra",
        )
    raise RuntimeError("Set VSELLM_API_KEY (or OPENAI_API_KEY) for the Astra stage")


def has_api_key() -> bool:
    return bool(os.environ.get("VSELLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None,
    schema: dict[str, Any] = REID_SCHEMA,
    schema_name: str = "flowmetric_reid",
) -> tuple[dict[str, Any], str | None]:
    """Request JSON, preferring a strict schema when the provider supports it.

    OpenAI-compatible proxies differ in their support for ``json_schema`` and
    ``reasoning_effort``.  The fallback sequence preserves that compatibility;
    callers must still perform semantic validation on the returned object.
    """
    if not isinstance(schema, dict):
        raise TypeError("schema must be a JSON-schema object")
    if not isinstance(schema_name, str) or not schema_name:
        raise ValueError("schema_name must be a non-empty string")
    formats: list[dict[str, Any] | None] = [
        {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        {"type": "json_object"},
        None,
    ]
    last_err: Exception | None = None
    for fmt in formats:
        for effort in ([reasoning_effort, None] if reasoning_effort else [None]):
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if fmt is not None:
                kwargs["response_format"] = fmt
            if effort is not None:
                kwargs["reasoning_effort"] = effort
            try:
                resp = client.chat.completions.create(**kwargs)
                data = _parse_json(resp.choices[0].message.content or "")
                actual_model = getattr(resp, "model", None)
                return data, str(actual_model) if actual_model else None
            except BadRequestError as e:
                last_err = e
            except json.JSONDecodeError as e:
                last_err = e
    raise RuntimeError(f"Astra request failed: {last_err}") from last_err


def _chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None,
    schema: dict[str, Any] = REID_SCHEMA,
    schema_name: str = "flowmetric_reid",
) -> tuple[dict[str, Any], str | None]:
    """Backward-compatible alias for :func:`chat_json`."""
    return chat_json(
        client,
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        schema=schema,
        schema_name=schema_name,
    )


def _frame_shape(frames_bgr: list[np.ndarray]) -> tuple[int, int]:
    """Return the shared (height, width), rejecting mixed or invalid inputs."""
    first_shape: tuple[int, ...] | None = None
    for i, frame in enumerate(frames_bgr):
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError(f"frame {i} must be a 2D/3D numpy array")
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"frame {i} has an empty image shape: {frame.shape}")
        shape = tuple(int(v) for v in frame.shape)
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            raise ValueError(
                f"all Astra frames must have the same shape; frame 0 is "
                f"{first_shape}, frame {i} is {shape}"
            )
    assert first_shape is not None
    return first_shape[0], first_shape[1]


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AstraResponseError(f"{label} must be a positive integer")
    return value


def _person_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AstraResponseError(f"{label} must be a non-negative integer")
    return value


def validate_astra_response(
    reid: dict[str, Any],
    *,
    n_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize a response, clipping boxes to its coordinate space.

    This validation is intentionally independent of JSON-schema support because
    some OpenAI-compatible proxies only accept ``json_object`` or plain text.
    Invalid results raise :class:`AstraResponseError`, which is safe to retry.
    """
    if not isinstance(reid, dict):
        raise AstraResponseError("Astra response must be a JSON object")
    meta = reid.get("_meta")
    if meta is not None and not isinstance(meta, dict):
        raise AstraResponseError("Astra response _meta must be an object")
    meta = meta or {}

    n_frames = meta.get("n_frames") if n_frames is None else n_frames
    width = meta.get("width") if width is None else width
    height = meta.get("height") if height is None else height
    n_frames = _positive_int(n_frames, "n_frames")
    width = _positive_int(width, "coordinate width")
    height = _positive_int(height, "coordinate height")

    people = reid.get("people")
    if not isinstance(people, list):
        raise AstraResponseError("Astra response 'people' must be an array")
    if not isinstance(reid.get("notes"), str):
        raise AstraResponseError("Astra response 'notes' must be a string")

    seen_ids: set[int] = set()
    seen_id_frames: set[tuple[int, int]] = set()
    for person_i, person in enumerate(people):
        if not isinstance(person, dict):
            raise AstraResponseError(f"people[{person_i}] must be an object")
        person_id = _person_id(person.get("id"), f"people[{person_i}].id")
        if person_id in seen_ids:
            raise AstraResponseError(f"duplicate Astra person id: {person_id}")
        seen_ids.add(person_id)
        role = person.get("role")
        if role not in VALID_ROLES:
            raise AstraResponseError(
                f"person {person_id} has invalid role {role!r}; "
                f"expected one of {sorted(VALID_ROLES)}"
            )
        detections = person.get("detections")
        if not isinstance(detections, list):
            raise AstraResponseError(f"person {person_id} detections must be an array")
        for det_i, det in enumerate(detections):
            label = f"person {person_id} detection {det_i}"
            if not isinstance(det, dict):
                raise AstraResponseError(f"{label} must be an object")
            frame_index = det.get("frame_index")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise AstraResponseError(f"{label} frame_index must be an integer")
            if not 0 <= frame_index < n_frames:
                raise AstraResponseError(
                    f"{label} frame_index {frame_index} is outside [0, {n_frames})"
                )
            key = (person_id, frame_index)
            if key in seen_id_frames:
                raise AstraResponseError(
                    f"duplicate detection for Astra id {person_id}, frame {frame_index}"
                )
            seen_id_frames.add(key)

            box = det.get("xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise AstraResponseError(f"{label} xyxy must contain four numbers")
            coords: list[float] = []
            for coord in box:
                if isinstance(coord, bool) or not isinstance(coord, (int, float)):
                    raise AstraResponseError(f"{label} xyxy contains a non-number")
                number = float(coord)
                if not math.isfinite(number):
                    raise AstraResponseError(f"{label} xyxy contains a non-finite number")
                coords.append(number)
            x1, y1, x2, y2 = coords
            if x1 >= x2 or y1 >= y2:
                raise AstraResponseError(f"{label} xyxy is not strictly ordered: {coords}")
            clipped = [
                min(max(x1, 0.0), float(width)),
                min(max(y1, 0.0), float(height)),
                min(max(x2, 0.0), float(width)),
                min(max(y2, 0.0), float(height)),
            ]
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                raise AstraResponseError(f"{label} lies outside the image after clipping")
            det["xyxy"] = clipped
    return reid


def astra_reid_people(
    frames_bgr: list[np.ndarray],
    *,
    model: str | None = None,
    reasoning_effort: str | None = "low",
    max_side: int = 1280,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Detect people on sampled frames and assign stable IDs across the batch.

    Boxes are returned in *resized* pixel space; use ``scale`` to map back.
    """
    if not frames_bgr:
        raise ValueError("frames_bgr is empty")
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    original_height, original_width = _frame_shape(frames_bgr)
    if client is None:
        client, default_model = make_client()
        model = model or default_model
    if model is None:
        raise ValueError("model is required when passing a client")

    encoded: list[tuple[str, float, int, int]] = [
        bgr_to_data_url(f, max_side=max_side) for f in frames_bgr
    ]
    scale = encoded[0][1]
    rw, rh = encoded[0][2], encoded[0][3]

    prompt = f"""You are the ReID stage of FlowMetric (restaurant CCTV people-flow).

You receive {len(frames_bgr)} frames sampled over time from ONE camera.
Image size after resize: {rw}x{rh} px. Coordinates: absolute xyxy, origin top-left.

Tasks:
1. Detect every visible person in each frame.
2. Assign a single stable integer id to the SAME person across frames (re-identification).
3. Prefer numeric identity continuity over appearance guesses; if unsure a person is new, still pick the most likely existing id or a new one.
4. Label role: customer / staff / unknown (staff = serving/walking with tray/uniform cues; seated diners = customer).
5. Refer to people ONLY by their numeric id — never rename the same person between frames.
6. Omit people you cannot see; do not invent boxes.

Return ONLY a JSON object of this shape (no prose):
{{"people": [{{"id": int, "role": "customer"|"staff"|"unknown",
  "detections": [{{"frame_index": int, "xyxy": [x1, y1, x2, y2]}}]}}],
 "notes": str}}"""

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for i, (url, _, _, _) in enumerate(encoded):
        content.append({"type": "text", "text": f"frame_index={i}"})
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})

    data, actual_model = _chat_json(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(data, dict):
        raise AstraResponseError("Astra response must be a JSON object")
    data["_meta"] = {
        "coordinate_space": "resized",
        "scale": scale,
        "scale_x": rw / original_width,
        "scale_y": rh / original_height,
        "width": rw,
        "height": rh,
        "resized_width": rw,
        "resized_height": rh,
        "original_width": original_width,
        "original_height": original_height,
        "n_frames": len(frames_bgr),
        "requested_model": model,
        "actual_model": actual_model,
        "prompt_version": ASTRA_PROMPT_VERSION,
        "schema_version": ASTRA_SCHEMA_VERSION,
    }
    return validate_astra_response(data)


def scale_astra_boxes_to_original(reid: dict[str, Any]) -> dict[str, Any]:
    """Mutate resized boxes to original pixels exactly once.

    ``coordinate_space`` makes this operation idempotent. Legacy in-memory
    responses are supported when their old ``scale/width/height`` metadata is
    sufficient, but legacy cache files are deliberately rejected elsewhere.
    """
    meta = reid.get("_meta")
    if not isinstance(meta, dict):
        raise AstraResponseError("cannot scale Astra response without _meta")
    coordinate_space = meta.get("coordinate_space", "resized")
    if coordinate_space == "original":
        return validate_astra_response(reid)
    if coordinate_space != "resized":
        raise AstraResponseError(f"unknown Astra coordinate_space: {coordinate_space!r}")

    resized_width = _positive_int(
        meta.get("resized_width", meta.get("width")), "resized width"
    )
    resized_height = _positive_int(
        meta.get("resized_height", meta.get("height")), "resized height"
    )
    raw_scale = meta.get("scale", 1.0)
    if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
        raise AstraResponseError("invalid Astra coordinate scale")
    scale = float(raw_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise AstraResponseError("invalid Astra coordinate scale")
    original_width = _positive_int(
        meta["original_width"]
        if "original_width" in meta
        else round(resized_width / scale),
        "original width",
    )
    original_height = _positive_int(
        meta["original_height"]
        if "original_height" in meta
        else round(resized_height / scale),
        "original height",
    )
    raw_scale_x = meta.get("scale_x", resized_width / original_width)
    raw_scale_y = meta.get("scale_y", resized_height / original_height)
    if (
        isinstance(raw_scale_x, bool)
        or not isinstance(raw_scale_x, (int, float))
        or isinstance(raw_scale_y, bool)
        or not isinstance(raw_scale_y, (int, float))
    ):
        raise AstraResponseError("invalid Astra coordinate scale")
    scale_x = float(raw_scale_x)
    scale_y = float(raw_scale_y)
    if not math.isfinite(scale_x) or not math.isfinite(scale_y) or scale_x <= 0 or scale_y <= 0:
        raise AstraResponseError("invalid Astra coordinate scale")

    validate_astra_response(
        reid,
        width=resized_width,
        height=resized_height,
        n_frames=meta.get("n_frames"),
    )
    for person in reid.get("people", []):
        for det in person.get("detections", []):
            x1, y1, x2, y2 = det["xyxy"]
            det["xyxy"] = [x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y]
    meta.update(
        {
            "coordinate_space": "original",
            "width": original_width,
            "height": original_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "original_width": original_width,
            "original_height": original_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
    )
    return validate_astra_response(reid)


def source_fingerprint(path: Path) -> dict[str, Any]:
    """Return stable source identity without putting its path in the cache."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"source changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size": after.st_size}


def build_astra_cache_manifest(
    source: Path,
    *,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    sample_frame_indexes: list[int],
    sample_fps: float,
    batch_size: int,
    requested_model: str | None,
    effort: str,
    batch_overlap: int = 1,
    targeted_pass: bool = False,
    targeted_references_per_identity: int = 1,
    targeted_resampling: bool = False,
    targeted_resample_budget: int = 16,
    targeted_shortlist: bool = False,
    targeted_shortlist_size: int = 5,
    part_aware_appearance: bool = False,
    tracker_reid: bool = False,
    offline_graph: bool = False,
    resolved_models: list[str] | None = None,
    actual_models: list[str] | None = None,
) -> dict[str, Any]:
    """Build the complete, versioned identity/config manifest for a cache."""
    if resolved_models is None and requested_model:
        resolved_models = [requested_model]
    resolved_model_ids = sorted(set(resolved_models or []))
    actual_model_ids = sorted(set(actual_models or []))
    return {
        "source": source_fingerprint(source),
        "video": {
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_count),
            "sample_frame_indexes": [int(i) for i in sample_frame_indexes],
        },
        "astra": {
            "sample_fps": float(sample_fps),
            "batch_size": int(batch_size),
            "batch_overlap": int(batch_overlap),
            "targeted_pass": bool(targeted_pass),
            "targeted_references_per_identity": int(
                targeted_references_per_identity
            ),
            "targeted_resampling": bool(targeted_resampling),
            "targeted_resample_budget": int(targeted_resample_budget),
            "targeted_shortlist": bool(targeted_shortlist),
            "targeted_shortlist_size": int(targeted_shortlist_size),
            "part_aware_appearance": bool(part_aware_appearance),
            "tracker_reid": bool(tracker_reid),
            "offline_graph": bool(offline_graph),
            "requested_model": requested_model,
            "resolved_models": resolved_model_ids,
            "actual_model": actual_model_ids[0] if len(actual_model_ids) == 1 else None,
            "actual_models": actual_model_ids,
            "effort": effort,
            "prompt_version": ASTRA_PROMPT_VERSION,
            "schema_version": ASTRA_SCHEMA_VERSION,
        },
    }


def _manifest_value(manifest: dict[str, Any], section: str, field: str) -> Any:
    value = manifest.get(section)
    if not isinstance(value, dict) or field not in value:
        raise AstraCacheError(f"cache manifest is missing {section}.{field}")
    return value[field]


def validate_astra_cache_manifest(
    cached: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject stale or differently configured cache manifests."""
    exact_fields = {
        "source": ("sha256", "size"),
        "video": ("width", "height", "frame_count", "sample_frame_indexes"),
        "astra": (
            "batch_size",
            "batch_overlap",
            "targeted_pass",
            "targeted_references_per_identity",
            "targeted_resampling",
            "targeted_resample_budget",
            "targeted_shortlist",
            "targeted_shortlist_size",
            "requested_model",
            "effort",
            "prompt_version",
            "schema_version",
        ),
    }
    if expected.get("astra", {}).get("targeted_shortlist"):
        exact_fields["astra"] = (*exact_fields["astra"], "part_aware_appearance")
    if expected.get("astra", {}).get("tracker_reid"):
        exact_fields["astra"] = (*exact_fields["astra"], "tracker_reid")
    if expected.get("astra", {}).get("targeted_shortlist"):
        exact_fields["astra"] = (*exact_fields["astra"], "offline_graph")
    mismatches: list[str] = []
    for section, fields in exact_fields.items():
        for field in fields:
            cached_value = _manifest_value(cached, section, field)
            expected_value = _manifest_value(expected, section, field)
            if cached_value != expected_value:
                mismatches.append(
                    f"{section}.{field}: cached={cached_value!r}, expected={expected_value!r}"
                )
    for section, field in (("video", "fps"), ("astra", "sample_fps")):
        cached_value = _manifest_value(cached, section, field)
        expected_value = _manifest_value(expected, section, field)
        if (
            isinstance(cached_value, bool)
            or not isinstance(cached_value, (int, float))
            or not math.isclose(
                float(cached_value),
                float(expected_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            mismatches.append(
                f"{section}.{field}: cached={cached_value!r}, expected={expected_value!r}"
            )
    astra_meta = cached.get("astra")
    assert isinstance(astra_meta, dict)
    for field in ("resolved_models", "actual_models"):
        models = astra_meta.get(field)
        if not isinstance(models, list) or any(not isinstance(v, str) or not v for v in models):
            raise AstraCacheError(f"cache manifest astra.{field} must be an array of model ids")
    expected_models = expected.get("astra", {}).get("resolved_models", [])
    if expected_models and astra_meta["resolved_models"] != expected_models:
        mismatches.append(
            "astra.resolved_models: "
            f"cached={astra_meta['resolved_models']!r}, expected={expected_models!r}"
        )
    if "actual_model" not in astra_meta:
        raise AstraCacheError("cache manifest is missing astra.actual_model")
    actual_model = astra_meta["actual_model"]
    if actual_model is not None and (not isinstance(actual_model, str) or not actual_model):
        raise AstraCacheError("cache manifest astra.actual_model must be a model id or null")
    if mismatches:
        raise AstraCacheError("Astra cache manifest mismatch: " + "; ".join(mismatches))


def validate_reid_collection(
    reids: list[dict[str, Any]],
    *,
    expected_sample_frames: list[int],
    original_width: int,
    original_height: int,
) -> None:
    """Validate all batches and ensure every sampled video frame is covered."""
    if not isinstance(reids, list):
        raise AstraResponseError("Astra batches must be an array")
    covered: set[int] = set()
    expected_set = set(expected_sample_frames)
    for batch_i, reid in enumerate(reids):
        if not isinstance(reid, dict):
            raise AstraResponseError(f"Astra batch {batch_i} must be an object")
        meta = reid.get("_meta")
        if not isinstance(meta, dict):
            raise AstraResponseError(f"Astra batch {batch_i} has no _meta object")
        video_frames = meta.get("video_frames")
        if not isinstance(video_frames, list) or any(
            isinstance(v, bool) or not isinstance(v, int) for v in video_frames
        ):
            raise AstraResponseError(f"Astra batch {batch_i} video_frames must be integers")
        if len(video_frames) != len(set(video_frames)):
            raise AstraResponseError(f"Astra batch {batch_i} repeats a video frame")
        if meta.get("coordinate_space") != "original":
            raise AstraResponseError(
                f"Astra batch {batch_i} must use original coordinate_space before sync"
            )
        if (
            meta.get("original_width") != original_width
            or meta.get("original_height") != original_height
        ):
            raise AstraResponseError(
                f"Astra batch {batch_i} original dimensions do not match the video"
            )
        if meta.get("width") != original_width or meta.get("height") != original_height:
            raise AstraResponseError(f"Astra batch {batch_i} active dimensions are inconsistent")
        if meta.get("n_frames") != len(video_frames):
            raise AstraResponseError(
                f"Astra batch {batch_i} n_frames does not match video_frames"
            )
        unexpected = set(video_frames) - expected_set
        declared_resampled = meta.get("targeted_resampled_frames", [])
        if (
            not isinstance(declared_resampled, list)
            or any(
                isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                for frame in declared_resampled
            )
            or len(declared_resampled) != len(set(declared_resampled))
        ):
            raise AstraResponseError(
                f"Astra batch {batch_i} targeted_resampled_frames must be "
                "unique non-negative integers"
            )
        allowed_resampled = (
            set(declared_resampled)
            if meta.get("protocol") == "tracklet_gallery_v1"
            and meta.get("gallery_scope") == "targeted"
            else set()
        )
        if unexpected != allowed_resampled:
            raise AstraResponseError(
                f"Astra batch {batch_i} contains undeclared resampled video frames: "
                f"{sorted(unexpected - allowed_resampled)}"
            )
        validate_astra_response(
            reid,
            n_frames=len(video_frames),
            width=original_width,
            height=original_height,
        )
        covered.update(video_frames)
    missing = expected_set - covered
    if missing:
        raise AstraResponseError(
            f"incomplete Astra batch coverage; missing sampled frames: {sorted(missing)}"
        )
    if reids and not expected_sample_frames:
        raise AstraResponseError("Astra returned batches although no sample frames were requested")


def validate_reid_batches(
    reids: list[dict[str, Any]],
    *,
    expected_batches: list[list[int]],
    original_width: int,
    original_height: int,
) -> None:
    """Validate base batches plus optional explicitly targeted follow-up batches."""
    if len(reids) < len(expected_batches):
        raise AstraResponseError(
            f"incomplete Astra batch coverage: got {len(reids)} batches, "
            f"expected {len(expected_batches)}"
        )
    for batch_i, (reid, expected_frames) in enumerate(zip(reids, expected_batches)):
        meta = reid.get("_meta") if isinstance(reid, dict) else None
        actual_frames = meta.get("video_frames") if isinstance(meta, dict) else None
        if actual_frames != expected_frames:
            raise AstraResponseError(
                f"Astra batch {batch_i} frame coverage mismatch: "
                f"got {actual_frames!r}, expected {expected_frames!r}"
            )
    for batch_i, reid in enumerate(reids[len(expected_batches) :], len(expected_batches)):
        meta = reid.get("_meta") if isinstance(reid, dict) else None
        if (
            not isinstance(meta, dict)
            or meta.get("protocol") != "tracklet_gallery_v1"
            or meta.get("gallery_scope") != "targeted"
            or not meta.get("targeted_query_ids")
        ):
            raise AstraResponseError(
                f"unexpected extra Astra batch {batch_i}; only targeted follow-ups are allowed"
            )
    expected_samples = sorted({frame for batch in expected_batches for frame in batch})
    validate_reid_collection(
        reids,
        expected_sample_frames=expected_samples,
        original_width=original_width,
        original_height=original_height,
    )


def _atomic_write_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            json.dump(data, tmp, ensure_ascii=False, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass


def save_astra_cache(
    path: Path,
    *,
    manifest: dict[str, Any],
    reids: list[dict[str, Any]],
) -> None:
    validate_astra_cache_manifest(manifest, manifest)
    video = manifest["video"]
    validate_reid_collection(
        reids,
        expected_sample_frames=video["sample_frame_indexes"],
        original_width=video["width"],
        original_height=video["height"],
    )
    _atomic_write_json(
        {"cache_version": ASTRA_CACHE_VERSION, "manifest": manifest, "reids": reids},
        path,
    )


def load_astra_cache(
    path: Path,
    *,
    expected_manifest: dict[str, Any],
    expected_batches: list[list[int]],
    original_width: int,
    original_height: int,
) -> list[dict[str, Any]]:
    """Load a cache only after identity, config, response, and coverage checks."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AstraCacheError(f"cannot read Astra cache {path}: {exc}") from exc
    if not isinstance(payload, dict) or "cache_version" not in payload:
        raise AstraCacheError(
            "legacy Astra cache has no versioned manifest and cannot be reused safely"
        )
    cache_version = payload.get("cache_version")
    if isinstance(cache_version, bool) or cache_version != ASTRA_CACHE_VERSION:
        raise AstraCacheError(
            f"unsupported Astra cache version {cache_version!r}; "
            f"expected {ASTRA_CACHE_VERSION}"
        )
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise AstraCacheError("Astra cache manifest must be an object")
    validate_astra_cache_manifest(manifest, expected_manifest)
    reids = payload.get("reids")
    if not isinstance(reids, list):
        raise AstraCacheError("Astra cache reids must be an array")
    try:
        validate_reid_batches(
            reids,
            expected_batches=expected_batches,
            original_width=original_width,
            original_height=original_height,
        )
    except AstraResponseError as exc:
        raise AstraCacheError(f"invalid Astra cache responses: {exc}") from exc
    return reids


def save_reid_json(reid: dict[str, Any], path: Path) -> None:
    _atomic_write_json(reid, path)
