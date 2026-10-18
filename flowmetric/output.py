"""Serialization and annotated-video rendering."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowmetric.tracking import LocalPass

# ---------------------------------------------------------------------------
# Pass 3: render
# ---------------------------------------------------------------------------

ROLE_BGR = {"staff": (219, 152, 52), "customer": (113, 204, 46)}


def id_color(gid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(gid * 9973)
    c = rng.integers(64, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


EVENT_LABEL = {
    "enter": "entered",
    "exit": "left frame",
    "leave_kitchen": "left kitchen",
    "seated": "seated",
    "service_start": "service start",
    "service_end": "service end",
}


def event_text(e: dict[str, Any]) -> str:
    who = f"ID {e['staff']} -> ID {e['customer']}" if "staff" in e else f"ID {e['person']}"
    return f"{e['t']:5.1f}s  {EVENT_LABEL.get(e['type'], e['type'])}: {who}"


def render(
    source: Path,
    dest: Path,
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
    events: dict[str, Any] | None = None,
    trail_len: int = 30,
    event_hold_s: float = 3.0,
) -> None:
    cap = cv2.VideoCapture(str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dest), cv2.VideoWriter_fourcc(*"mp4v"), lp.fps, (lp.width, lp.height)
    )
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen: set[int] = set()
    sampled = set(lp.sample_idx) if mode.startswith("hybrid") else set()
    max_jump = 0.08 * max(lp.width, lp.height)
    ev_list = (events or {}).get("events", [])
    waiting = [s for s in (events or {}).get("states", []) if s["type"] == "waiting"]
    roles = {**{str(k): v for k, v in gid_role.items()}, **(events or {}).get("roles", {})}

    for idx in range(lp.total):
        ok, frame = cap.read()
        if not ok:
            break
        now = idx / lp.fps
        dets = lp.dets.get(idx, [])
        drawn: set[int] = set()
        for tid, box, score in dets:
            gid = tid_to_gid.get(tid, tid)
            if gid in drawn:  # two local fragments mapped to same person
                continue
            drawn.add(gid)
            seen.add(gid)
            x1, y1, x2, y2 = map(int, box)
            role = roles.get(str(gid), "")
            color = ROLE_BGR.get(role, id_color(gid))
            if any(s["person"] == str(gid) and s["start"] <= now <= s["end"] for s in waiting):
                role = f"{role} waiting".strip()

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            prev = trails[gid][-1] if trails[gid] else None
            if prev and np.hypot(center[0] - prev[0], center[1] - prev[1]) > max_jump:
                trails[gid] = []  # ReID merge after a gap — don't draw a teleport line
            trails[gid].append(center)
            trails[gid] = trails[gid][-trail_len:]
            pts = trails[gid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {gid}" + (f" {role}" if role else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
            )

        hud = f"FlowMetric [{mode}]  frame {idx + 1}/{lp.total}  people {len(drawn)}  ids {len(seen)}"
        if idx in sampled:
            hud += "  * astra"
        cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        recent = [e for e in ev_list if now - event_hold_s <= e["t"] <= now][-6:]
        for k, e in enumerate(reversed(recent)):
            y = lp.height - 20 - 34 * k
            text = event_text(e)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()


def build_tracks(
    lp: LocalPass,
    tid_to_gid: dict[int, int],
    gid_role: dict[int, str],
    *,
    mode: str,
) -> dict[str, Any]:
    """Per-person trajectories keyed by final ID (input for events + benchmark)."""
    per_frame: dict[int, dict[int, tuple[int, np.ndarray, float]]] = defaultdict(dict)
    for fi, dets in lp.dets.items():
        for tid, box, score in dets:
            gid = tid_to_gid.get(tid, tid)
            prev = per_frame[gid].get(fi)
            if prev is None or score > prev[2]:
                per_frame[gid][fi] = (tid, box, score)

    tracks: dict[str, Any] = {}
    for gid, frames in per_frame.items():
        fis = sorted(frames)
        tracks[str(gid)] = {
            "role": gid_role.get(gid, ""),
            "tids": sorted({frames[f][0] for f in fis}),
            "frames": fis,
            "boxes": [[round(float(v), 1) for v in frames[f][1]] for f in fis],
            "conf": [round(frames[f][2], 3) for f in fis],
        }
    return {
        "mode": mode,
        "fps": lp.fps,
        "width": lp.width,
        "height": lp.height,
        "n_frames": lp.total,
        "tracks": tracks,
    }
