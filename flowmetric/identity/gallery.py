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


def validate_targeted_shortlist_response(
    response: dict[str, Any],
    *,
    target_tracklet_ids: Sequence[int],
    candidate_shortlists: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Ensure a targeted response cannot bypass its per-query shortlist."""
    targets = {int(value) for value in target_tracklet_ids}
    normalized = {
        int(query): {int(reference) for reference in references}
        for query, references in candidate_shortlists.items()
    }
    if set(normalized) != targets:
        raise ValueError("candidate shortlists must cover every targeted query")
    for identity in response["identities"]:
        members = {int(value) for value in identity["tracklet_ids"]}
        queries = members & targets
        if len(queries) > 1:
            raise AstraResponseError(
                "targeted shortlist response merged multiple unresolved queries"
            )
        if queries:
            query = next(iter(queries))
            disallowed = members - targets - normalized[query]
            if disallowed:
                raise AstraResponseError(
                    f"targeted query {query} matched references outside its shortlist: "
                    f"{sorted(disallowed)}"
                )
    return response


def constrain_targeted_shortlist_response(
    response: dict[str, Any],
    *,
    target_tracklet_ids: Sequence[int],
    candidate_shortlists: Mapping[int, Sequence[int]],
) -> tuple[dict[str, Any], int]:
    """Safely abstain shortlist violations after a model retry has failed."""
    targets = {int(value) for value in target_tracklet_ids}
    allowed = {
        int(query): {int(reference) for reference in references}
        for query, references in candidate_shortlists.items()
    }
    if set(allowed) != targets:
        raise ValueError("candidate shortlists must cover every targeted query")

    identities: list[dict[str, Any]] = []
    abstained = {int(value) for value in response["abstained_tracklet_ids"]}
    next_identity_id = max(
        (int(identity["id"]) for identity in response["identities"]),
        default=-1,
    ) + 1
    rejected_queries = 0
    for identity in response["identities"]:
        members = [int(value) for value in identity["tracklet_ids"]]
        queries = [value for value in members if value in targets]
        references = [value for value in members if value not in targets]
        valid = (
            len(queries) <= 1
            and (
                not queries
                or not references
                or set(references) <= allowed[queries[0]]
            )
        )
        if valid:
            identities.append(identity)
            continue

        abstained.update(queries)
        rejected_queries += len(queries)
        for index, reference in enumerate(references):
            identities.append(
                {
                    "id": (
                        int(identity["id"])
                        if index == 0
                        else next_identity_id
                    ),
                    "role": identity["role"],
                    "tracklet_ids": [reference],
                    "confidence": float(identity["confidence"]),
                }
            )
            if index > 0:
                next_identity_id += 1

    constrained = {
        "identities": identities,
        "abstained_tracklet_ids": sorted(abstained),
    }
    validate_tracklet_reid_response(
        constrained,
        expected_tracklet_ids={
            int(value)
            for identity in response["identities"]
            for value in identity["tracklet_ids"]
        }
        | {int(value) for value in response["abstained_tracklet_ids"]},
    )
    validate_targeted_shortlist_response(
        constrained,
        target_tracklet_ids=target_tracklet_ids,
        candidate_shortlists=candidate_shortlists,
    )
    return constrained, rejected_queries


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
    gallery_scope = meta.get("gallery_scope", "full")
    targeted_query_ids = meta.get("targeted_query_ids", [])
    targeted_resampled_frames = meta.get("targeted_resampled_frames", [])
    targeted_candidate_shortlists = meta.get("targeted_candidate_shortlists", {})
    targeted_shortlist_rejections = meta.get("targeted_shortlist_rejections", 0)
    if gallery_scope not in {"full", "targeted"}:
        raise AstraResponseError("gallery_scope must be full or targeted")
    if (
        not isinstance(targeted_query_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value not in expected
            for value in targeted_query_ids
        )
        or len(targeted_query_ids) != len(set(targeted_query_ids))
    ):
        raise AstraResponseError("targeted_query_ids must be unique gallery tracklets")
    if gallery_scope == "targeted" and not targeted_query_ids:
        raise AstraResponseError("targeted gallery has no query tracklets")
    if gallery_scope == "full" and targeted_query_ids:
        raise AstraResponseError("full gallery cannot contain targeted queries")
    video_frames = meta.get("video_frames", [])
    if (
        not isinstance(targeted_resampled_frames, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in video_frames
            for value in targeted_resampled_frames
        )
        or len(targeted_resampled_frames) != len(set(targeted_resampled_frames))
    ):
        raise AstraResponseError(
            "targeted_resampled_frames must be unique targeted video frames"
        )
    if gallery_scope == "full" and targeted_resampled_frames:
        raise AstraResponseError("full gallery cannot contain targeted resampled frames")
    if not isinstance(targeted_candidate_shortlists, dict):
        raise AstraResponseError("targeted_candidate_shortlists must be an object")
    if (
        isinstance(targeted_shortlist_rejections, bool)
        or not isinstance(targeted_shortlist_rejections, int)
        or targeted_shortlist_rejections < 0
    ):
        raise AstraResponseError("targeted_shortlist_rejections must be non-negative")
    if gallery_scope == "full" and targeted_candidate_shortlists:
        raise AstraResponseError("full gallery cannot contain targeted candidate shortlists")
    expected_query_keys = {str(value) for value in targeted_query_ids}
    if (
        targeted_candidate_shortlists
        and set(targeted_candidate_shortlists) != expected_query_keys
    ):
        raise AstraResponseError("candidate shortlists must cover every targeted query")
    for query, references in targeted_candidate_shortlists.items():
        if (
            query not in expected_query_keys
            or not isinstance(references, list)
            or len(references) != len(set(references))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in expected
                or value in targeted_query_ids
                for value in references
            )
        ):
            raise AstraResponseError("candidate shortlist contains invalid references")
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


def _intersection_over_subject(box: np.ndarray, other: np.ndarray) -> float:
    x1 = max(float(box[0]), float(other[0]))
    y1 = max(float(box[1]), float(other[1]))
    x2 = min(float(box[2]), float(other[2]))
    y2 = min(float(box[3]), float(other[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
    return intersection / area


def _crop_quality(
    frame: np.ndarray,
    box: np.ndarray,
    confidence: float,
    other_boxes: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    crop = _crop_with_context(frame, box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 300.0, 1.0)
    frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    size = min(math.sqrt(box_area / frame_area) / 0.25, 1.0)
    occlusion = max(
        (_intersection_over_subject(box, other) for other in other_boxes),
        default=0.0,
    )
    visibility = 1.0 - min(occlusion, 1.0)
    quality = 0.4 * confidence + 0.25 * sharpness + 0.2 * visibility + 0.15 * size
    return crop, float(quality)


def _quality_diverse(
    values: Sequence[tuple[int, int, np.ndarray, float, np.ndarray, float]],
    count: int,
) -> list[tuple[int, int, np.ndarray, float, np.ndarray, float]]:
    """Pick the best crop in each chronological segment."""
    if len(values) <= count:
        return list(values)
    boundaries = np.linspace(0, len(values), count + 1, dtype=int)
    chosen = [
        max(
            values[int(boundaries[index]) : int(boundaries[index + 1])],
            key=lambda item: (item[5], item[3], -item[1]),
        )
        for index in range(count)
    ]
    return sorted(chosen, key=lambda item: item[1])


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
    observations: dict[
        int, list[tuple[int, int, np.ndarray, float, np.ndarray, float]]
    ] = defaultdict(list)
    for sample_index, (frame, video_frame) in enumerate(zip(sample_frames, video_frames)):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            raise ValueError("gallery frames must be non-empty colour images")
        frame_detections = list(detections.get(video_frame, ()))
        for detection_index, (tracklet_id, box, confidence) in enumerate(frame_detections):
            box_array = np.asarray(box, dtype=float)
            crop, quality = _crop_quality(
                frame,
                box_array,
                float(confidence),
                [
                    np.asarray(other_box, dtype=float)
                    for other_index, (_, other_box, _) in enumerate(frame_detections)
                    if other_index != detection_index
                ],
            )
            observations[int(tracklet_id)].append(
                (
                    sample_index,
                    int(video_frame),
                    box_array,
                    float(confidence),
                    crop,
                    quality,
                )
            )
    if not observations:
        raise ValueError("cannot build a tracklet gallery without local detections")

    cells: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for tracklet_id in sorted(observations):
        chosen = _quality_diverse(observations[tracklet_id], crops_per_tracklet)
        metadata.append(
            {
                "tracklet_id": tracklet_id,
                "crop_count": len(chosen),
                "video_frames": [item[1] for item in chosen],
                "crop_quality": [round(float(item[5]), 4) for item in chosen],
                "selection_strategy": "quality_by_temporal_segment_v1",
                "mean_detector_confidence": round(
                    float(np.mean([item[3] for item in chosen])), 4
                ),
            }
        )
        for _, video_frame, _, _, crop, _ in chosen:
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
    target_tracklet_ids: Sequence[int] | None = None,
    candidate_shortlists: Mapping[int, Sequence[int]] | None = None,
    repair_shortlist_violations: bool = False,
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
    target_ids = sorted({int(value) for value in target_tracklet_ids or []})
    if not set(target_ids) <= set(tracklet_ids):
        raise ValueError("target_tracklet_ids must be present in the gallery")
    targeted_instruction = ""
    if target_ids:
        shortlists = {
            int(query): sorted({int(reference) for reference in references})
            for query, references in (candidate_shortlists or {}).items()
        }
        if shortlists and set(shortlists) != set(target_ids):
            raise ValueError("candidate_shortlists must cover every target tracklet")
        if any(
            reference not in tracklet_ids or reference in target_ids
            for references in shortlists.values()
            for reference in references
        ):
            raise ValueError("candidate shortlist references must be gallery references")
        shortlist_instruction = ""
        if shortlists:
            shortlist_lines = "\n".join(
                f"- query {query}: allowed reference tracklets {references}"
                for query, references in sorted(shortlists.items())
            )
            shortlist_instruction = f"""
Candidate shortlists (hard constraints):
{shortlist_lines}
For each query, compare only with its allowed references. Never attach it to another
reference or to another unresolved query. A query may remain abstained or form a
singleton identity when it is a genuinely new person; do not force a listed match.
"""
        targeted_instruction = f"""
This is a targeted second pass. The unresolved QUERY tracklets are: {target_ids}.
All other tracklets are trusted reference prototypes from previously accepted groups.
Focus on attaching a query to a convincing reference of the same physical person.
Do not merge two reference prototypes unless the visual evidence is unmistakable.
If no reference convincingly matches a query, put that query in abstained_tracklet_ids.
{shortlist_instruction}
"""
    prompt = f"""You are the identity-association stage of FlowMetric restaurant CCTV.

The {len(encoded_pages)} image page(s) are contact sheets up to {width}x{height}. Every cell is a LOCAL tracker crop and
is labelled with an existing integer tracklet ID and video frame. Tracklet inventory:
{inventory}
{targeted_instruction}

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
    validated = validate_tracklet_reid_response(
        data, expected_tracklet_ids=tracklet_ids
    )
    if target_ids and candidate_shortlists:
        try:
            validate_targeted_shortlist_response(
                validated,
                target_tracklet_ids=target_ids,
                candidate_shortlists=candidate_shortlists,
            )
        except AstraResponseError:
            if not repair_shortlist_violations:
                raise
            validated, rejected_queries = constrain_targeted_shortlist_response(
                validated,
                target_tracklet_ids=target_ids,
                candidate_shortlists=candidate_shortlists,
            )
            validated["_targeted_shortlist_rejections"] = rejected_queries
    return validated, actual_model


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
    gallery_scope: str = "full",
    targeted_query_ids: Sequence[int] = (),
    targeted_candidate_shortlists: Mapping[int, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Convert associations to the established internal ReID collection shape."""
    if gallery_scope not in {"full", "targeted"}:
        raise ValueError("gallery_scope must be 'full' or 'targeted'")
    gallery_tracklet_ids = [int(item["tracklet_id"]) for item in tracklets]
    query_ids = sorted({int(value) for value in targeted_query_ids})
    if not set(query_ids) <= set(gallery_tracklet_ids):
        raise ValueError("targeted_query_ids must be present in tracklets")
    if gallery_scope == "targeted" and not query_ids:
        raise ValueError("targeted gallery must contain at least one query")
    shortlists = {
        str(int(query)): sorted({int(reference) for reference in references})
        for query, references in (targeted_candidate_shortlists or {}).items()
    }
    if shortlists and set(shortlists) != {str(value) for value in query_ids}:
        raise ValueError("targeted candidate shortlists must cover every query")
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
            "gallery_tracklet_ids": gallery_tracklet_ids,
            "gallery_scope": gallery_scope,
            "targeted_query_ids": query_ids,
            "targeted_candidate_shortlists": shortlists,
            "targeted_shortlist_rejections": int(
                response.get("_targeted_shortlist_rejections", 0)
            ),
            "gallery_associations": associations,
            "abstained_tracklet_ids": list(response["abstained_tracklet_ids"]),
        },
    }


def targeted_gallery_inventory(
    reids: Sequence[dict[str, Any]],
    *,
    merge_confidence: float = 0.55,
    min_support: int = 2,
    representatives_per_group: int = 1,
    tracklet_midpoints: Mapping[int, float] | None = None,
) -> tuple[list[int], list[int]]:
    """Return unresolved queries and strong, temporally diverse references."""
    if (
        isinstance(representatives_per_group, bool)
        or not isinstance(representatives_per_group, int)
        or representatives_per_group <= 0
    ):
        raise ValueError("representatives_per_group must be a positive integer")
    midpoints = tracklet_midpoints or {}
    accepted_tracklets: set[int] = set()
    unresolved_tracklets: set[int] = set()
    representatives: list[int] = []
    for reid in reids:
        meta = reid.get("_meta", {}) if isinstance(reid, dict) else {}
        if meta.get("protocol") != GALLERY_PROTOCOL:
            continue
        for association in meta.get("gallery_associations", []):
            tracklet_ids = [int(value) for value in association.get("tracklet_ids", [])]
            support = association.get("support", {})
            if not tracklet_ids:
                continue
            confidence = float(association.get("confidence", 0.0))
            eligible = [
                tracklet_id
                for tracklet_id in tracklet_ids
                if int(support.get(str(tracklet_id), 0)) >= min_support
                and confidence >= merge_confidence
                and confidence * int(support.get(str(tracklet_id), 0))
                >= min_support * merge_confidence
            ]
            accepted_tracklets.update(eligible)
            unresolved_tracklets.update(set(tracklet_ids) - set(eligible))
            if eligible:
                selected = [
                    max(
                        eligible,
                        key=lambda tracklet_id: (
                            int(support.get(str(tracklet_id), 0)),
                            -tracklet_id,
                        ),
                    )
                ]
                while (
                    len(selected) < representatives_per_group
                    and len(selected) < len(eligible)
                ):
                    remaining = [value for value in eligible if value not in selected]
                    selected.append(
                        max(
                            remaining,
                            key=lambda tracklet_id: (
                                min(
                                    abs(
                                        float(midpoints.get(tracklet_id, tracklet_id))
                                        - float(midpoints.get(chosen, chosen))
                                    )
                                    for chosen in selected
                                ),
                                int(support.get(str(tracklet_id), 0)),
                                -tracklet_id,
                            ),
                        )
                    )
                representatives.extend(selected)
        unresolved_tracklets.update(
            int(value) for value in meta.get("abstained_tracklet_ids", [])
        )
    queries = sorted(unresolved_tracklets - accepted_tracklets)
    references = sorted(set(representatives) - set(queries))
    return queries, references


def build_targeted_candidate_shortlists(
    local_pass: Any,
    reids: Sequence[dict[str, Any]],
    *,
    query_tracklet_ids: Sequence[int],
    reference_tracklet_ids: Sequence[int],
    limit: int = 5,
) -> dict[int, list[int]]:
    """Rank compatible reference prototypes for each unresolved tracklet."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer in [1, 10]")
    queries = sorted({int(value) for value in query_tracklet_ids})
    references = sorted({int(value) for value in reference_tracklet_ids})
    if (
        len(queries) != len(query_tracklet_ids)
        or len(references) != len(reference_tracklet_ids)
    ):
        raise ValueError("query and reference tracklet ids must be unique")

    roles: dict[int, str] = {}
    for reid in reids:
        meta = reid.get("_meta", {}) if isinstance(reid, dict) else {}
        role_by_identity = {
            int(person["id"]): str(person.get("role", "unknown"))
            for person in reid.get("people", [])
            if isinstance(person, dict) and isinstance(person.get("id"), int)
        }
        for association in meta.get("gallery_associations", []):
            role = role_by_identity.get(int(association.get("id", -1)), "unknown")
            for tracklet_id in association.get("tracklet_ids", []):
                roles[int(tracklet_id)] = role

    observations: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for frame, detections in local_pass.dets.items():
        for tracklet_id, box, _ in detections:
            tracklet_id = int(tracklet_id)
            if tracklet_id in queries or tracklet_id in references:
                observations[tracklet_id][int(frame)] = np.asarray(box, dtype=float)

    def prototype(tracklet_id: int) -> np.ndarray | None:
        samples = local_pass.appearance_embeddings.get(tracklet_id, [])
        if not samples:
            return None
        strongest = sorted(samples, key=lambda item: (-item[2], item[0]))[:8]
        vector = np.mean(
            np.stack([np.asarray(item[1], dtype=float) for item in strongest]), axis=0
        )
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    prototypes = {tracklet_id: prototype(tracklet_id) for tracklet_id in queries + references}

    def spatially_distinct(a: np.ndarray, b: np.ndarray) -> bool:
        x1, y1 = np.maximum(a[:2], b[:2])
        x2, y2 = np.minimum(a[2:], b[2:])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(float((a[2] - a[0]) * (a[3] - a[1])), 1.0)
        area_b = max(float((b[2] - b[0]) * (b[3] - b[1])), 1.0)
        containment = intersection / min(area_a, area_b)
        center_gap = float(
            np.linalg.norm(
                np.asarray(
                    [
                        (a[0] + a[2] - b[0] - b[2]) / 2,
                        (a[1] + a[3] - b[1] - b[3]) / 2,
                    ]
                )
            )
            / max(min(a[3] - a[1], b[3] - b[1]), 1.0)
        )
        return containment < 0.1 and center_gap > 0.35

    def motion_distance(query: int, reference: int) -> float:
        query_items = sorted(observations.get(query, {}).items())
        reference_items = sorted(observations.get(reference, {}).items())
        if not query_items or not reference_items:
            return float("inf")
        if query_items[0][0] > reference_items[-1][0]:
            left, right = reference_items[-1][1], query_items[0][1]
        elif reference_items[0][0] > query_items[-1][0]:
            left, right = query_items[-1][1], reference_items[0][1]
        else:
            shared = sorted(set(observations[query]) & set(observations[reference]))
            if not shared:
                return float("inf")
            distances = []
            for frame in shared:
                left = observations[query][frame]
                right = observations[reference][frame]
                scale = max(
                    ((left[3] - left[1]) + (right[3] - right[1])) / 2,
                    1.0,
                )
                distances.append(
                    float(
                        np.linalg.norm(
                            np.asarray(
                                [
                                    (left[0] + left[2] - right[0] - right[2]) / 2,
                                    (left[1] + left[3] - right[1] - right[3]) / 2,
                                ]
                            )
                        )
                        / scale
                    )
                )
            return float(np.median(distances))
        scale = max(
            ((left[3] - left[1]) + (right[3] - right[1])) / 2,
            1.0,
        )
        return float(
            np.linalg.norm(
                np.asarray(
                    [
                        (left[0] + left[2] - right[0] - right[2]) / 2,
                        (left[1] + left[3] - right[1] - right[3]) / 2,
                    ]
                )
            )
            / scale
        )

    result: dict[int, list[int]] = {}
    for query in queries:
        ranked: list[tuple[tuple[float, float, int], int]] = []
        query_role = roles.get(query, "unknown")
        for reference in references:
            reference_role = roles.get(reference, "unknown")
            if (
                query_role != "unknown"
                and reference_role != "unknown"
                and query_role != reference_role
            ):
                continue
            shared = set(observations.get(query, {})) & set(observations.get(reference, {}))
            if any(
                spatially_distinct(
                    observations[query][frame], observations[reference][frame]
                )
                for frame in shared
            ):
                continue
            query_prototype = prototypes.get(query)
            reference_prototype = prototypes.get(reference)
            similarity = (
                float(np.dot(query_prototype, reference_prototype))
                if query_prototype is not None and reference_prototype is not None
                else -1.0
            )
            distance = motion_distance(query, reference)
            ranked.append(((-similarity, distance, reference), reference))
        result[query] = [reference for _, reference in sorted(ranked)[:limit]]
    return result
