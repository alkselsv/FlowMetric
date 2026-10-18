"""Selective Astra verification for locally proposed service episodes.

The local FSM remains responsible for identities, geometry, and interval
boundaries.  This module sends only ambiguous candidates to Astra and asks for
one strict yes/no verdict for the already identified staff/customer pair.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from flowmetric.infrastructure.astra import (
    AstraCacheError,
    AstraResponseError,
    _atomic_write_json,
    bgr_to_data_url,
    chat_json,
    configured_default_model_id,
    has_api_key,
    make_client,
    source_fingerprint,
)
from flowmetric.domain import ServiceCandidate, ServiceDecision
from flowmetric.media import (
    finite_number as _finite_number,
    load_video_frames,
    sample_evenly as _sample_evenly,
    service_crop_bounds,
    track_box_map as _track_box_map,
    xyxy_box as _box,
)

SERVICE_CACHE_VERSION = 1
SERVICE_PROMPT_VERSION = "2026-10-18.3"
SERVICE_SCHEMA_VERSION = "1"
SERVICE_SAMPLING_VERSION = "2"
SERVICE_SCHEMA_NAME = "flowmetric_service_verdict"

DEFAULT_RULE_LOW = 0.3
DEFAULT_RULE_HIGH = 0.8
DEFAULT_MIN_FRAMES = 4
DEFAULT_FRAMES_PER_CANDIDATE = 6
DEFAULT_MAX_FRAMES = 8
DEFAULT_CONTEXT_SECONDS = 1.0
DEFAULT_CONTEXT_MARGIN = 0.15

SERVICE_RESPONSE_FIELDS = frozenset(
    {"is_service", "staff_id", "customer_id", "confidence"}
)
SERVICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_service": {"type": "boolean"},
        "staff_id": {"type": "integer", "minimum": 0},
        "customer_id": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["is_service", "staff_id", "customer_id", "confidence"],
    "additionalProperties": False,
}

FrameLoader = Callable[[Path, tuple[int, ...]], Mapping[int, np.ndarray]]
Requester = Callable[..., dict[str, Any] | tuple[dict[str, Any], str | None]]


def _probability(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _numeric_id(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a numeric string, got {value!r}")
    if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise ValueError(
            f"{label} must be a canonical non-negative numeric string, got {value!r}"
        )
    return int(value)


def validate_service_response(
    response: Any,
    *,
    expected_staff_id: str,
    expected_customer_id: str,
) -> dict[str, Any]:
    """Strictly validate and normalize an Astra service verdict.

    Semantic validation is always applied because some compatible providers
    fall back from JSON Schema to JSON-object or plain-text response modes.
    """
    expected_staff = _numeric_id(expected_staff_id, "expected staff_id")
    expected_customer = _numeric_id(expected_customer_id, "expected customer_id")
    if not isinstance(response, dict):
        raise AstraResponseError("Astra service response must be a JSON object")
    fields = set(response)
    if fields != SERVICE_RESPONSE_FIELDS:
        missing = sorted(SERVICE_RESPONSE_FIELDS - fields)
        extra = sorted(fields - SERVICE_RESPONSE_FIELDS)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise AstraResponseError(
            "Astra service response must contain exactly four fields: "
            + ", ".join(details)
        )

    is_service = response["is_service"]
    if type(is_service) is not bool:
        raise AstraResponseError("Astra service is_service must be a boolean")
    staff_id = response["staff_id"]
    customer_id = response["customer_id"]
    if type(staff_id) is not int or staff_id < 0:
        raise AstraResponseError("Astra service staff_id must be a non-negative integer")
    if type(customer_id) is not int or customer_id < 0:
        raise AstraResponseError("Astra service customer_id must be a non-negative integer")
    if staff_id != expected_staff or customer_id != expected_customer:
        raise AstraResponseError(
            "Astra service response changed candidate identities: "
            f"expected {expected_staff}->{expected_customer}, "
            f"got {staff_id}->{customer_id}"
        )
    try:
        confidence = _probability(response["confidence"], "Astra service confidence")
    except ValueError as exc:
        raise AstraResponseError(str(exc)) from exc
    return {
        "is_service": is_service,
        "staff_id": staff_id,
        "customer_id": customer_id,
        "confidence": confidence,
    }


def select_service_frames(
    candidate: ServiceCandidate,
    tracks: dict[str, Any],
    *,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    frames_per_candidate: int = DEFAULT_FRAMES_PER_CANDIDATE,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[int, ...]:
    """Choose chronological frames where both proposed identities are visible."""
    context_seconds = _finite_number(context_seconds, "context_seconds")
    if context_seconds < 0:
        raise ValueError("context_seconds must be non-negative")
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or not DEFAULT_MIN_FRAMES <= max_frames <= 8
        or isinstance(frames_per_candidate, bool)
        or not isinstance(frames_per_candidate, int)
        or not DEFAULT_MIN_FRAMES <= frames_per_candidate <= max_frames
    ):
        raise ValueError("frame sampling must use 4..8 frames with target <= maximum")
    interval = candidate.interval
    if interval.end_frame < interval.start_frame:
        raise ValueError("service candidate end_frame precedes start_frame")
    fps = _finite_number(tracks.get("fps"), "tracks fps")
    if fps <= 0:
        raise ValueError("tracks fps must be positive")
    context = int(round(context_seconds * fps))
    lower = max(0, interval.start_frame - context)
    upper = interval.end_frame + context
    n_frames = tracks.get("n_frames")
    if isinstance(n_frames, int) and not isinstance(n_frames, bool) and n_frames >= 0:
        upper = min(upper, max(n_frames - 1, 0))
    staff = _track_box_map(tracks, interval.staff)
    customer = _track_box_map(tracks, interval.customer)
    joint = sorted(frame for frame in staff.keys() & customer.keys() if lower <= frame <= upper)
    target = min(frames_per_candidate, max_frames)
    return _sample_evenly(joint, target)


def annotate_service_crop(
    frame: np.ndarray,
    *,
    staff_box: Sequence[float],
    customer_box: Sequence[float],
    table_box: Sequence[float] | None,
    staff_id: str,
    customer_id: str,
    context_margin: float = DEFAULT_CONTEXT_MARGIN,
    crop_bounds: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop the proposed interaction and draw only its already known geometry."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        raise ValueError("frame must be a 2D/3D numpy array")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame has empty image dimensions")
    staff = _box(staff_box, "staff_box")
    customer = _box(customer_box, "customer_box")
    table = _box(table_box, "table_box") if table_box is not None else None
    if crop_bounds is None:
        left, top, right, bottom = service_crop_bounds(
            frame.shape,
            staff_boxes=[staff],
            customer_boxes=[customer],
            table_box=table,
            context_margin=context_margin,
        )
    else:
        if not isinstance(crop_bounds, (list, tuple)) or len(crop_bounds) != 4:
            raise ValueError("crop_bounds must contain four integer coordinates")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in crop_bounds
        ):
            raise ValueError("crop_bounds must contain four integer coordinates")
        left, top, right, bottom = (int(value) for value in crop_bounds)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("crop_bounds must be ordered within the frame")
    crop = frame[top:bottom, left:right].copy()

    def draw(box: tuple[float, float, float, float], label: str, color: tuple[int, int, int]) -> None:
        ax1 = max(0, min(crop.shape[1] - 1, int(round(box[0] - left))))
        ay1 = max(0, min(crop.shape[0] - 1, int(round(box[1] - top))))
        ax2 = max(0, min(crop.shape[1] - 1, int(round(box[2] - left))))
        ay2 = max(0, min(crop.shape[0] - 1, int(round(box[3] - top))))
        cv2.rectangle(crop, (ax1, ay1), (ax2, ay2), color, 2)
        cv2.putText(
            crop,
            label,
            (ax1, max(12, ay1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    draw(staff, f"STAFF {staff_id}", (50, 220, 50))
    draw(customer, f"CUSTOMER {customer_id}", (0, 180, 255))
    if table is not None:
        draw(table, "TABLE", (255, 120, 20))
    return crop


def _tracks_digest(tracks: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            tracks,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tracks must be JSON-serializable without non-finite values: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AstraCacheError(f"{label} is not strict JSON: {exc}") from exc


def _candidate_manifest(
    candidate: ServiceCandidate, sampled_frames: tuple[int, ...]
) -> dict[str, Any]:
    interval = candidate.interval
    table = _box(candidate.table_box, "candidate table_box") if candidate.table_box else None
    return {
        "staff_id": interval.staff,
        "customer_id": interval.customer,
        "start_frame": interval.start_frame,
        "end_frame": interval.end_frame,
        "left_censored": interval.left_censored,
        "right_censored": interval.right_censored,
        "rule_confidence": float(interval.confidence),
        "table_box": list(table) if table is not None else None,
        "sampled_frames": list(sampled_frames),
    }


class AstraServiceVerifier:
    """Triage service candidates and ask Astra only about ambiguous episodes."""

    def __init__(
        self,
        source: Path,
        tracks: dict[str, Any],
        *,
        cache_path: Path | None = None,
        model: str | None = None,
        effort: str = "low",
        rule_low: float = DEFAULT_RULE_LOW,
        rule_high: float = DEFAULT_RULE_HIGH,
        frames_per_candidate: int = DEFAULT_FRAMES_PER_CANDIDATE,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        context_margin: float = DEFAULT_CONTEXT_MARGIN,
        max_side: int = 1280,
        requester: Requester | None = None,
        frame_loader: FrameLoader | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.source = Path(source)
        if not isinstance(tracks, dict):
            raise TypeError("tracks must be the tracks.json object")
        self.tracks = tracks
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.model = model
        self.resolved_model = model or configured_default_model_id()
        # The two supported providers spell the same default model as
        # ``gpt-6-astra`` and ``openai/gpt-6-astra``.  A canonical cache id
        # keeps an otherwise valid cache reusable when credentials are absent.
        resolved_for_cache = self.resolved_model or "gpt-6-astra"
        self.cache_model_id = resolved_for_cache.removeprefix("openai/")
        self.effort = effort
        self.rule_low = _probability(rule_low, "rule_low")
        self.rule_high = _probability(rule_high, "rule_high")
        if self.rule_low > self.rule_high:
            raise ValueError("rule_low must not exceed rule_high")
        if (
            isinstance(frames_per_candidate, bool)
            or not isinstance(frames_per_candidate, int)
            or not DEFAULT_MIN_FRAMES <= frames_per_candidate <= DEFAULT_MAX_FRAMES
        ):
            raise ValueError("frames_per_candidate must be an integer in [4, 8]")
        self.frames_per_candidate = frames_per_candidate
        self.context_seconds = _finite_number(context_seconds, "context_seconds")
        if self.context_seconds < 0:
            raise ValueError("context_seconds must be non-negative")
        self.context_margin = _finite_number(context_margin, "context_margin")
        if self.context_margin < 0:
            raise ValueError("context_margin must be non-negative")
        if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side <= 0:
            raise ValueError("max_side must be a positive integer")
        self.max_side = max_side
        if not isinstance(effort, str) or not effort:
            raise ValueError("effort must be a non-empty string")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError("model must be a non-empty string or None")
        if client is not None and model is None and requester is None:
            raise ValueError("model is required when passing an Astra client")
        self.requester = requester
        self.frame_loader = frame_loader or load_video_frames
        self.client = client

    def _validate_candidate(self, candidate: Any) -> float:
        if not isinstance(candidate, ServiceCandidate):
            raise TypeError("candidates must contain ServiceCandidate instances")
        interval = candidate.interval
        _numeric_id(interval.staff, "candidate staff_id")
        _numeric_id(interval.customer, "candidate customer_id")
        if (
            isinstance(interval.start_frame, bool)
            or not isinstance(interval.start_frame, int)
            or interval.start_frame < 0
            or isinstance(interval.end_frame, bool)
            or not isinstance(interval.end_frame, int)
            or interval.end_frame < interval.start_frame
        ):
            raise ValueError("candidate frame bounds must be ordered non-negative integers")
        if type(interval.left_censored) is not bool or type(interval.right_censored) is not bool:
            raise ValueError("candidate censoring flags must be booleans")
        if candidate.table_box is not None:
            _box(candidate.table_box, "candidate table_box")
        return _probability(interval.confidence, "candidate rule confidence")

    def _build_manifest(
        self,
        candidates: list[ServiceCandidate],
        sampled: list[tuple[int, ...]],
    ) -> dict[str, Any]:
        return {
            "source": source_fingerprint(self.source),
            "tracks_sha256": _tracks_digest(self.tracks),
            "candidates": [
                _candidate_manifest(candidate, frames)
                for candidate, frames in zip(candidates, sampled)
            ],
            "astra": {
                "requested_model": self.model,
                "resolved_model": self.cache_model_id,
                "effort": self.effort,
                "prompt_version": SERVICE_PROMPT_VERSION,
                "schema_version": SERVICE_SCHEMA_VERSION,
                "sampling_version": SERVICE_SAMPLING_VERSION,
                "rule_low": self.rule_low,
                "rule_high": self.rule_high,
                "minimum_frames": DEFAULT_MIN_FRAMES,
                "frames_per_candidate": self.frames_per_candidate,
                "maximum_frames": DEFAULT_MAX_FRAMES,
                "context_seconds": self.context_seconds,
                "context_margin": self.context_margin,
                "max_side": self.max_side,
            },
        }

    def _load_cache(
        self,
        expected_manifest: dict[str, Any],
        request_indexes: list[int],
        sampled: list[tuple[int, ...]],
        candidates: list[ServiceCandidate],
    ) -> dict[int, ServiceDecision]:
        assert self.cache_path is not None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AstraCacheError(f"cannot read service-verification cache: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "cache_version",
            "manifest",
            "results",
        }:
            raise AstraCacheError("service-verification cache has an unknown or malformed format")
        if type(payload["cache_version"]) is not int or payload["cache_version"] != SERVICE_CACHE_VERSION:
            raise AstraCacheError(
                "unsupported service-verification cache version "
                f"{payload['cache_version']!r}; expected {SERVICE_CACHE_VERSION}"
            )
        if not isinstance(payload["manifest"], dict) or _canonical_json(
            payload["manifest"], "service-verification cache manifest"
        ) != _canonical_json(expected_manifest, "expected service-verification manifest"):
            raise AstraCacheError("service-verification cache manifest mismatch")
        results = payload["results"]
        if not isinstance(results, list) or len(results) != len(request_indexes):
            raise AstraCacheError("service-verification cache result count mismatch")
        decisions: dict[int, ServiceDecision] = {}
        for position, (entry, candidate_index) in enumerate(zip(results, request_indexes)):
            if not isinstance(entry, dict) or set(entry) != {"metadata", "actual_model", "response"}:
                raise AstraCacheError(f"service-verification cache result {position} is malformed")
            metadata = entry["metadata"]
            if not isinstance(metadata, dict) or set(metadata) != {"candidate_index", "sampled_frames"}:
                raise AstraCacheError(f"service-verification cache metadata {position} is malformed")
            cached_index = metadata["candidate_index"]
            cached_frames = metadata["sampled_frames"]
            if (
                type(cached_index) is not int
                or not isinstance(cached_frames, list)
                or any(type(frame) is not int for frame in cached_frames)
                or cached_index != candidate_index
                or cached_frames != list(sampled[candidate_index])
            ):
                raise AstraCacheError(f"service-verification cache result {position} identity mismatch")
            actual_model = entry["actual_model"]
            if actual_model is not None and (not isinstance(actual_model, str) or not actual_model):
                raise AstraCacheError(f"service-verification cache actual_model {position} is invalid")
            interval = candidates[candidate_index].interval
            try:
                response = validate_service_response(
                    entry["response"],
                    expected_staff_id=interval.staff,
                    expected_customer_id=interval.customer,
                )
            except (AstraResponseError, ValueError) as exc:
                raise AstraCacheError(
                    f"service-verification cache response {position} is invalid: {exc}"
                ) from exc
            decisions[candidate_index] = ServiceDecision(
                is_service=response["is_service"],
                staff_id=interval.staff,
                customer_id=interval.customer,
                confidence=response["confidence"],
                source="astra",
                sampled_frames=sampled[candidate_index],
                model=actual_model,
            )
        return decisions

    def _messages(
        self,
        candidate: ServiceCandidate,
        sampled_frames: tuple[int, ...],
        crops: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        interval = candidate.interval
        prompt = f"""You verify one proposed restaurant service episode.

The local tracker has already assigned STAFF {interval.staff} and CUSTOMER {interval.customer}.
The local FSM proposes one episode with fixed inclusive boundaries frame
{interval.start_frame} through frame {interval.end_frame}.

Confirm only whether that proposed pair is interacting as staff serving that customer during
the proposed episode.  Judge the proposed interval using the before/after context: service is
a purposeful staff interaction with this customer or their table, not merely passing or
standing nearby.  Do not detect people, draw or return boxes, change either ID, choose a
different pair, or adjust the local boundaries.  The supplied crops already label STAFF,
CUSTOMER, and TABLE where available.  Return only one JSON object, without Markdown or prose,
with exactly is_service, staff_id, customer_id, and confidence.  Example shape (use your own
verdict and confidence): {{"is_service": true, "staff_id": {int(interval.staff)},
"customer_id": {int(interval.customer)}, "confidence": 0.91}}.  Confidence is confidence in
the returned verdict.
"""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame_index, crop in zip(sampled_frames, crops):
            url, _, _, _ = bgr_to_data_url(crop, max_side=self.max_side)
            content.append({"type": "text", "text": f"video_frame={frame_index}"})
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        return [{"role": "user", "content": content}]

    def _request(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
        if self.requester is not None:
            result = self.requester(
                messages=messages,
                model=self.resolved_model,
                reasoning_effort=self.effort,
                schema=SERVICE_SCHEMA,
                schema_name=SERVICE_SCHEMA_NAME,
            )
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise TypeError("service requester tuple must contain response and actual model")
                response, actual_model = result
            else:
                response, actual_model = result, None
            if actual_model is not None and (not isinstance(actual_model, str) or not actual_model):
                raise TypeError("service requester actual model must be a non-empty string or None")
            return response, actual_model

        client = self.client
        model = self.resolved_model
        if client is None:
            client, default_model = make_client()
            model = model or default_model
            self.client = client
            self.resolved_model = model
        assert model is not None
        return chat_json(
            client,
            model=model,
            messages=messages,
            reasoning_effort=self.effort,
            schema=SERVICE_SCHEMA,
            schema_name=SERVICE_SCHEMA_NAME,
        )

    def __call__(self, candidates: list[ServiceCandidate]) -> list[ServiceDecision]:
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")
        decisions: list[ServiceDecision | None] = [None] * len(candidates)
        sampled: list[tuple[int, ...]] = [()] * len(candidates)
        request_indexes: list[int] = []

        for index, candidate in enumerate(candidates):
            confidence = self._validate_candidate(candidate)
            interval = candidate.interval
            if confidence < self.rule_low:
                decisions[index] = ServiceDecision(
                    False,
                    interval.staff,
                    interval.customer,
                    1.0 - confidence,
                    "rules_low_confidence",
                )
                continue
            if confidence > self.rule_high:
                decisions[index] = ServiceDecision(
                    True,
                    interval.staff,
                    interval.customer,
                    confidence,
                    "rules_high_confidence",
                )
                continue
            frames = select_service_frames(
                candidate,
                self.tracks,
                context_seconds=self.context_seconds,
                frames_per_candidate=self.frames_per_candidate,
                max_frames=DEFAULT_MAX_FRAMES,
            )
            sampled[index] = frames
            if len(frames) < DEFAULT_MIN_FRAMES:
                decisions[index] = ServiceDecision(
                    True,
                    interval.staff,
                    interval.customer,
                    confidence,
                    "insufficient_visual_evidence",
                    sampled_frames=frames,
                )
                continue
            request_indexes.append(index)

        if not request_indexes:
            return [decision for decision in decisions if decision is not None]

        manifest = (
            self._build_manifest(candidates, sampled)
            if self.cache_path is not None
            else None
        )
        cache_problem: AstraCacheError | None = None
        if self.cache_path is not None and self.cache_path.exists():
            assert manifest is not None
            try:
                cached = self._load_cache(
                    manifest, request_indexes, sampled, candidates
                )
            except AstraCacheError as exc:
                cache_problem = exc
            else:
                for index, decision in cached.items():
                    decisions[index] = decision
                return [decision for decision in decisions if decision is not None]

        can_request = self.requester is not None or self.client is not None or has_api_key()
        if not can_request:
            detail = f"; cache rejected: {cache_problem}" if cache_problem else ""
            raise RuntimeError(
                "Astra service verification needs VSELLM_API_KEY or OPENAI_API_KEY "
                f"after a cache miss{detail}"
            )

        union_frames = tuple(
            sorted({frame for index in request_indexes for frame in sampled[index]})
        )
        loaded = self.frame_loader(self.source, union_frames)
        if not isinstance(loaded, Mapping):
            raise TypeError("frame_loader must return a frame-index mapping")
        missing = set(union_frames) - set(loaded)
        if missing:
            raise RuntimeError(f"frame_loader omitted requested frames: {sorted(missing)}")

        staff_boxes: dict[str, dict[int, tuple[float, float, float, float]]] = {}
        customer_boxes: dict[str, dict[int, tuple[float, float, float, float]]] = {}
        cache_results: list[dict[str, Any]] = []
        for candidate_index in request_indexes:
            candidate = candidates[candidate_index]
            interval = candidate.interval
            staff_map = staff_boxes.setdefault(
                interval.staff, _track_box_map(self.tracks, interval.staff)
            )
            customer_map = customer_boxes.setdefault(
                interval.customer, _track_box_map(self.tracks, interval.customer)
            )
            candidate_frames = sampled[candidate_index]
            frame_shapes = {tuple(loaded[frame].shape) for frame in candidate_frames}
            if len(frame_shapes) != 1:
                raise ValueError("all sampled service frames must have the same shape")
            crop_bounds = service_crop_bounds(
                loaded[candidate_frames[0]].shape,
                staff_boxes=[staff_map[frame] for frame in candidate_frames],
                customer_boxes=[customer_map[frame] for frame in candidate_frames],
                table_box=candidate.table_box,
                context_margin=self.context_margin,
            )
            crops = [
                annotate_service_crop(
                    loaded[frame_index],
                    staff_box=staff_map[frame_index],
                    customer_box=customer_map[frame_index],
                    table_box=candidate.table_box,
                    staff_id=interval.staff,
                    customer_id=interval.customer,
                    context_margin=self.context_margin,
                    crop_bounds=crop_bounds,
                )
                for frame_index in candidate_frames
            ]
            raw_response, actual_model = self._request(
                self._messages(candidate, sampled[candidate_index], crops)
            )
            response = validate_service_response(
                raw_response,
                expected_staff_id=interval.staff,
                expected_customer_id=interval.customer,
            )
            decisions[candidate_index] = ServiceDecision(
                is_service=response["is_service"],
                staff_id=interval.staff,
                customer_id=interval.customer,
                confidence=response["confidence"],
                source="astra",
                sampled_frames=sampled[candidate_index],
                model=actual_model,
            )
            cache_results.append(
                {
                    "metadata": {
                        "candidate_index": candidate_index,
                        "sampled_frames": list(sampled[candidate_index]),
                    },
                    "actual_model": actual_model,
                    "response": response,
                }
            )

        if self.cache_path is not None:
            assert manifest is not None
            _atomic_write_json(
                {
                    "cache_version": SERVICE_CACHE_VERSION,
                    "manifest": manifest,
                    "results": cache_results,
                },
                self.cache_path,
            )
        if any(decision is None for decision in decisions):
            raise RuntimeError("internal error: missing service verification decision")
        return [decision for decision in decisions if decision is not None]
