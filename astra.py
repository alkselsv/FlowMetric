"""GPT-6 Astra client for sparse-frame person ReID (hybrid FlowMetric stage)."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

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


def _bgr_to_data_url(bgr: np.ndarray, max_side: int = 1280) -> tuple[str, float, int, int]:
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


VSELLM_BASE_URL = "https://api.vsellm.ru/v1"


def make_client() -> tuple[OpenAI, str]:
    """Return (client, default Astra model id) for the configured provider.

    VSELLM_API_KEY → VseLLM (OpenAI-compatible proxy, model ``openai/gpt-6-astra``);
    otherwise OPENAI_API_KEY → OpenAI directly (``gpt-6-astra``).
    """
    if key := os.environ.get("VSELLM_API_KEY"):
        base_url = os.environ.get("VSELLM_BASE_URL", VSELLM_BASE_URL)
        return OpenAI(api_key=key, base_url=base_url), "openai/gpt-6-astra"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "gpt-6-astra"
    raise RuntimeError("Set VSELLM_API_KEY (or OPENAI_API_KEY) for the Astra stage")


def has_api_key() -> bool:
    return bool(os.environ.get("VSELLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def _chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Chat Completions with strict JSON schema, degrading for proxies that reject options."""
    formats: list[dict[str, Any] | None] = [
        {
            "type": "json_schema",
            "json_schema": {"name": "flowmetric_reid", "strict": True, "schema": REID_SCHEMA},
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
                return _parse_json(resp.choices[0].message.content or "")
            except BadRequestError as e:
                last_err = e
            except json.JSONDecodeError as e:
                last_err = e
    raise RuntimeError(f"Astra request failed: {last_err}") from last_err


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
    if client is None:
        client, default_model = make_client()
        model = model or default_model
    if model is None:
        raise ValueError("model is required when passing a client")

    encoded: list[tuple[str, float, int, int]] = [
        _bgr_to_data_url(f, max_side=max_side) for f in frames_bgr
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

    data = _chat_json(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        reasoning_effort=reasoning_effort,
    )
    data.setdefault("people", [])
    data.setdefault("notes", "")
    data["_meta"] = {"scale": scale, "width": rw, "height": rh, "n_frames": len(frames_bgr)}
    return data


def scale_astra_boxes_to_original(reid: dict[str, Any]) -> dict[str, Any]:
    """Mutate detections xyxy from resized → original full-frame pixels."""
    scale = float(reid.get("_meta", {}).get("scale", 1.0))
    if scale == 1.0:
        return reid
    inv = 1.0 / scale
    for person in reid.get("people", []):
        for det in person.get("detections", []):
            det["xyxy"] = [c * inv for c in det["xyxy"]]
    return reid


def save_reid_json(reid: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reid, ensure_ascii=False, indent=2), encoding="utf-8")
