"""Privacy-reduced Astra ReID over numbered local-tracklet crops.

The vision model never draws boxes and never invents identifiers.  It receives a
contact sheet whose cells are labelled with existing local tracker IDs, then
partitions those IDs into identities or an explicit abstain set.  Local boxes
remain the only geometry used by the rest of the pipeline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from flowmetric.infrastructure.astra import (
    ASTRA_PROMPT_VERSION,
    ASTRA_SCHEMA_VERSION,
    AstraResponseError,
    bgr_to_data_url,
    chat_json,
)

GALLERY_PROTOCOL = "tracklet_gallery_v1"

TRACKLET_REID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "identities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0},
                    "role": {
                        "type": "string",
                        "enum": ["customer", "staff", "unknown"],
                    },
                    "tracklet_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "role", "tracklet_ids", "confidence"],
                "additionalProperties": False,
            },
        },
        "abstained_tracklet_ids": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
    },
    "required": ["identities", "abstained_tracklet_ids"],
    "additionalProperties": False,
}


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AstraResponseError(f"{label} must be a number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise AstraResponseError(f"{label} must be a finite number in [0, 1]")
    return number


def validate_tracklet_reid_response(
    response: Any,
    *,
    expected_tracklet_ids: Sequence[int],
) -> dict[str, Any]:
    """Validate an exact partition of the supplied local IDs."""
    if not isinstance(response, dict) or set(response) != {
        "identities",
        "abstained_tracklet_ids",
    }:
        raise AstraResponseError("tracklet ReID response has an unknown or malformed shape")
    identities = response["identities"]
    abstained = response["abstained_tracklet_ids"]
    if not isinstance(identities, list) or not isinstance(abstained, list):
        raise AstraResponseError("tracklet ReID identities and abstentions must be arrays")

    expected = set(expected_tracklet_ids)
    if len(expected) != len(expected_tracklet_ids) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected_tracklet_ids
    ):
        raise ValueError("expected_tracklet_ids must be unique non-negative integers")

    seen_identity_ids: set[int] = set()
    assigned: set[int] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, dict) or set(identity) != {
            "id",
            "role",
            "tracklet_ids",
            "confidence",
        }:
            raise AstraResponseError(f"identity {index} has an unknown or malformed shape")
        identity_id = identity["id"]
        if isinstance(identity_id, bool) or not isinstance(identity_id, int) or identity_id < 0:
            raise AstraResponseError(f"identity {index} id must be a non-negative integer")
        if identity_id in seen_identity_ids:
            raise AstraResponseError(f"duplicate tracklet identity id {identity_id}")
        seen_identity_ids.add(identity_id)
        if identity["role"] not in {"customer", "staff", "unknown"}:
            raise AstraResponseError(f"identity {identity_id} has invalid role")
        tracklet_ids = identity["tracklet_ids"]
        if not isinstance(tracklet_ids, list) or not tracklet_ids:
            raise AstraResponseError(f"identity {identity_id} tracklet_ids must be non-empty")
        _finite_probability(identity["confidence"], f"identity {identity_id} confidence")
        for tracklet_id in tracklet_ids:
            if (
                isinstance(tracklet_id, bool)
                or not isinstance(tracklet_id, int)
                or tracklet_id not in expected
            ):
                raise AstraResponseError(
                    f"identity {identity_id} returned unknown tracklet id {tracklet_id!r}"
                )
            if tracklet_id in assigned:
                raise AstraResponseError(f"tracklet id {tracklet_id} was assigned more than once")
            assigned.add(tracklet_id)

    for tracklet_id in abstained:
        if (
            isinstance(tracklet_id, bool)
            or not isinstance(tracklet_id, int)
            or tracklet_id not in expected
        ):
            raise AstraResponseError(f"unknown abstained tracklet id {tracklet_id!r}")
        if tracklet_id in assigned:
            raise AstraResponseError(f"tracklet id {tracklet_id} was assigned more than once")
        assigned.add(tracklet_id)
    missing = expected - assigned
    if missing:
        raise AstraResponseError(f"tracklet ReID omitted ids: {sorted(missing)}")
    return response


def validate_normalized_gallery_metadata(meta: Any) -> dict[str, Any]:
    """Validate cached gallery associations before they can affect identity sync."""
    if not isinstance(meta, dict) or meta.get("protocol") != GALLERY_PROTOCOL:
        raise AstraResponseError("invalid normalized tracklet gallery metadata")
    gallery_ids = meta.get("gallery_tracklet_ids")
    associations = meta.get("gallery_associations")
    abstained = meta.get("abstained_tracklet_ids")
    if not isinstance(gallery_ids, list) or not isinstance(associations, list) or not isinstance(abstained, list):
        raise AstraResponseError("gallery metadata arrays are missing")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gallery_ids):
        raise AstraResponseError("gallery_tracklet_ids must be non-negative integers")
    if len(gallery_ids) != len(set(gallery_ids)):
        raise AstraResponseError("gallery_tracklet_ids contains duplicates")
    expected = set(gallery_ids)
    assigned: set[int] = set()
    seen_identity_ids: set[int] = set()
    for index, association in enumerate(associations):
        if not isinstance(association, dict) or set(association) != {
            "id",
            "tracklet_ids",
            "confidence",
            "support",
        }:
            raise AstraResponseError(f"gallery association {index} is malformed")
        identity_id = association["id"]
        if isinstance(identity_id, bool) or not isinstance(identity_id, int) or identity_id < 0:
            raise AstraResponseError(f"gallery association {index} id is invalid")
        if identity_id in seen_identity_ids:
            raise AstraResponseError(f"duplicate gallery identity id {identity_id}")
        seen_identity_ids.add(identity_id)
        tracklet_ids = association["tracklet_ids"]
        support = association["support"]
        if not isinstance(tracklet_ids, list) or not tracklet_ids or not isinstance(support, dict):
            raise AstraResponseError(f"gallery association {identity_id} is incomplete")
        _finite_probability(association["confidence"], f"identity {identity_id} confidence")
        if set(support) != {str(tracklet_id) for tracklet_id in tracklet_ids}:
            raise AstraResponseError(f"gallery association {identity_id} support is incomplete")
        for tracklet_id in tracklet_ids:
            if (
                isinstance(tracklet_id, bool)
                or not isinstance(tracklet_id, int)
                or tracklet_id not in expected
                or tracklet_id in assigned
            ):
                raise AstraResponseError(
                    f"gallery association {identity_id} has invalid tracklet {tracklet_id!r}"
                )
            count = support[str(tracklet_id)]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise AstraResponseError(
                    f"gallery association {identity_id} support must be positive"
                )
            assigned.add(tracklet_id)
    for tracklet_id in abstained:
        if (
            isinstance(tracklet_id, bool)
            or not isinstance(tracklet_id, int)
            or tracklet_id not in expected
            or tracklet_id in assigned
        ):
            raise AstraResponseError(f"invalid abstained gallery tracklet {tracklet_id!r}")
        assigned.add(tracklet_id)
    if assigned != expected:
        raise AstraResponseError(
            f"normalized gallery metadata omitted tracklets: {sorted(expected - assigned)}"
        )
    return meta


def _evenly_spaced(values: Sequence[Any], count: int) -> list[Any]:
    if len(values) <= count:
        return list(values)
    last = len(values) - 1
    return [values[(index * last) // (count - 1)] for index in range(count)]


def _crop_with_context(frame: np.ndarray, box: np.ndarray, margin: float = 0.18) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    pad_x = margin * max(x2 - x1, 1.0)
    pad_y = margin * max(y2 - y1, 1.0)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if left >= right or top >= bottom:
        raise ValueError("tracklet crop is empty")
    return frame[top:bottom, left:right].copy()


def _render_cell(
    crop: np.ndarray,
    *,
    label: str,
    cell_width: int,
    cell_height: int,
) -> np.ndarray:
    header = 30
    canvas = np.full((cell_height, cell_width, 3), 24, dtype=np.uint8)
    available_height = cell_height - header
    scale = min(cell_width / crop.shape[1], available_height / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (cell_width - resized.shape[1]) // 2
    y = header + (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        label,
        (5, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_tracklet_gallery(
    sample_frames: Sequence[np.ndarray],
    video_frames: Sequence[int],
    detections: Mapping[int, Sequence[tuple[int, np.ndarray, float]]],
    *,
    crops_per_tracklet: int = 3,
    cell_width: int = 176,
    cell_height: int = 240,
    columns: int = 6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build one labelled contact sheet from local boxes only."""
    if len(sample_frames) != len(video_frames) or not sample_frames:
        raise ValueError("sample_frames and video_frames must be aligned and non-empty")
    if crops_per_tracklet < 1 or columns < 1 or cell_width < 64 or cell_height < 96:
        raise ValueError("invalid tracklet gallery layout")
    observations: dict[int, list[tuple[int, int, np.ndarray, float]]] = defaultdict(list)
    for sample_index, (frame, video_frame) in enumerate(zip(sample_frames, video_frames)):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            raise ValueError("gallery frames must be non-empty colour images")
        for tracklet_id, box, confidence in detections.get(video_frame, ()):
            observations[int(tracklet_id)].append(
                (sample_index, int(video_frame), np.asarray(box, dtype=float), float(confidence))
            )
    if not observations:
        raise ValueError("cannot build a tracklet gallery without local detections")

    cells: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for tracklet_id in sorted(observations):
        chosen = _evenly_spaced(observations[tracklet_id], crops_per_tracklet)
        metadata.append(
            {
                "tracklet_id": tracklet_id,
                "crop_count": len(chosen),
                "video_frames": [item[1] for item in chosen],
                "mean_detector_confidence": round(
                    float(np.mean([item[3] for item in chosen])), 4
                ),
            }
        )
        for sample_index, video_frame, box, _ in chosen:
            crop = _crop_with_context(sample_frames[sample_index], box)
            cells.append(
                _render_cell(
                    crop,
                    label=f"tracklet={tracklet_id} frame={video_frame}",
                    cell_width=cell_width,
                    cell_height=cell_height,
                )
            )

    rows = math.ceil(len(cells) / columns)
    gallery = np.full((rows * cell_height, columns * cell_width, 3), 24, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        y, x = row * cell_height, column * cell_width
        gallery[y : y + cell_height, x : x + cell_width] = cell
    return gallery, metadata


def astra_reid_tracklet_gallery(
    gallery_bgr: np.ndarray,
    tracklets: Sequence[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str | None,
    client: Any,
    max_side: int = 1600,
) -> tuple[dict[str, Any], str | None]:
    """Ask Astra only to associate numbered local tracklets."""
    tracklet_ids = [int(item["tracklet_id"]) for item in tracklets]
    # Keep labels/crops readable instead of shrinking one very tall sheet. The
    # pages remain in a single request, so identities may be compared across them.
    page_height = 5 * 240
    pages = [
        gallery_bgr[top : min(top + page_height, gallery_bgr.shape[0])]
        for top in range(0, gallery_bgr.shape[0], page_height)
    ]
    encoded_pages = [bgr_to_data_url(page, max_side=max_side) for page in pages]
    width = max(item[2] for item in encoded_pages)
    height = max(item[3] for item in encoded_pages)
    inventory = ", ".join(
        f"{item['tracklet_id']}({item['crop_count']} crops)" for item in tracklets
    )
    prompt = f"""You are the identity-association stage of FlowMetric restaurant CCTV.

The {len(encoded_pages)} image page(s) are contact sheets up to {width}x{height}. Every cell is a LOCAL tracker crop and
is labelled with an existing integer tracklet ID and video frame. Tracklet inventory:
{inventory}

Group tracklet IDs only when the crops provide convincing evidence that they show the
same physical person. Assign customer/staff/unknown from visible appearance and context.
Use confidence for the whole group. Put an uncertain tracklet in
abstained_tracklet_ids instead of forcing a merge. Every supplied tracklet ID must occur
exactly once, either in one identity.tracklet_ids array or in abstained_tracklet_ids.
Never create IDs, boxes, coordinates, detections, explanations, or extra fields.
Return ONLY one JSON object matching the supplied schema, without Markdown or prose."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for page_index, (url, _, _, _) in enumerate(encoded_pages):
        content.append({"type": "text", "text": f"gallery_page={page_index}"})
        content.append(
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
        )
    data, actual_model = chat_json(
        client,
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        reasoning_effort=reasoning_effort,
        schema=TRACKLET_REID_SCHEMA,
        schema_name="flowmetric_tracklet_reid",
    )
    return validate_tracklet_reid_response(
        data, expected_tracklet_ids=tracklet_ids
    ), actual_model


def normalize_gallery_response(
    response: dict[str, Any],
    *,
    tracklets: Sequence[dict[str, Any]],
    video_frames: Sequence[int],
    detections: Mapping[int, Sequence[tuple[int, np.ndarray, float]]],
    original_width: int,
    original_height: int,
    requested_model: str,
    actual_model: str | None,
) -> dict[str, Any]:
    """Convert associations to the established internal ReID collection shape."""
    by_tracklet: dict[int, dict[int, tuple[np.ndarray, float]]] = defaultdict(dict)
    for local_frame_index, video_frame in enumerate(video_frames):
        for tracklet_id, box, detector_confidence in detections.get(video_frame, ()):
            previous = by_tracklet[int(tracklet_id)].get(local_frame_index)
            if previous is None or detector_confidence > previous[1]:
                by_tracklet[int(tracklet_id)][local_frame_index] = (
                    np.asarray(box, dtype=float),
                    float(detector_confidence),
                )

    people: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    crop_counts = {int(item["tracklet_id"]): int(item["crop_count"]) for item in tracklets}
    for identity in response["identities"]:
        chosen_by_frame: dict[int, tuple[np.ndarray, float]] = {}
        for tracklet_id in identity["tracklet_ids"]:
            for frame_index, detection in by_tracklet[tracklet_id].items():
                previous = chosen_by_frame.get(frame_index)
                if previous is None or detection[1] > previous[1]:
                    chosen_by_frame[frame_index] = detection
        people.append(
            {
                "id": identity["id"],
                "role": identity["role"],
                "detections": [
                    {
                        "frame_index": frame_index,
                        "xyxy": [float(value) for value in chosen_by_frame[frame_index][0]],
                    }
                    for frame_index in sorted(chosen_by_frame)
                ],
            }
        )
        associations.append(
            {
                "id": identity["id"],
                "tracklet_ids": list(identity["tracklet_ids"]),
                "confidence": float(identity["confidence"]),
                "support": {
                    str(tracklet_id): crop_counts[tracklet_id]
                    for tracklet_id in identity["tracklet_ids"]
                },
            }
        )
    return {
        "people": people,
        "notes": "local tracklet gallery associations; geometry copied from local detections",
        "_meta": {
            "protocol": GALLERY_PROTOCOL,
            "coordinate_space": "original",
            "scale": 1.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "width": int(original_width),
            "height": int(original_height),
            "resized_width": int(original_width),
            "resized_height": int(original_height),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "n_frames": len(video_frames),
            "video_frames": [int(frame) for frame in video_frames],
            "requested_model": requested_model,
            "actual_model": actual_model,
            "prompt_version": ASTRA_PROMPT_VERSION,
            "schema_version": ASTRA_SCHEMA_VERSION,
            "gallery_tracklet_ids": [int(item["tracklet_id"]) for item in tracklets],
            "gallery_associations": associations,
            "abstained_tracklet_ids": list(response["abstained_tracklet_ids"]),
        },
    }
