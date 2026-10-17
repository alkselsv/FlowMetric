"""Service events from person trajectories (tracks.json → events.json).

Events: enter, exit, leave_kitchen, seated, service_start, service_end.
States: waiting, consuming.
Roles come from Astra when available; otherwise a motion/zone heuristic fills them in.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMERA = ROOT / "config" / "camera.json"

MIN_TRACK_S = 0.5  # shorter tracks are detector noise / fragments
SMOOTH_S = 0.5
MOVE_SPEED = 0.2  # body heights per second
STATIONARY_SPEED = 0.12
SEATED_MIN_S = 2.0
REACQUIRE_S = 1.5  # a birth this soon after a nearby death is the same person
SERVICE_DIST = 0.9  # staff-customer center distance, in staff box heights
SERVICE_MIN_S = 1.0
SERVICE_GAP_S = 0.5


@dataclass
class Track:
    gid: str
    role: str
    frames: np.ndarray
    boxes: np.ndarray  # (n, 4) xyxy
    fps: float

    @property
    def t0(self) -> float:
        return self.frames[0] / self.fps

    @property
    def t1(self) -> float:
        return self.frames[-1] / self.fps

    @property
    def life(self) -> float:
        return self.t1 - self.t0

    @property
    def centers(self) -> np.ndarray:
        b = self.boxes
        return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]

    @property
    def heights(self) -> np.ndarray:
        return self.boxes[:, 3] - self.boxes[:, 1]

    def travel(self, a: int = 0, b: int | None = None) -> float:
        """Net displacement between indices a and b, in body heights."""
        c, h = self.centers, self.heights
        b = len(c) - 1 if b is None else b
        return float(np.linalg.norm(c[b] - c[a]) / max(np.median(h[a : b + 1]), 1.0))

    def reach(self) -> float:
        """Farthest distance from the starting point, in body heights (robust to box jitter)."""
        c = self.centers
        return float(np.max(np.linalg.norm(c - c[0], axis=1)) / max(np.median(self.heights), 1.0))

    def path(self) -> float:
        """Total smoothed path length, in body heights."""
        return float(np.sum(self.speed()[1:] * np.diff(self.frames) / self.fps))

    def speed(self) -> np.ndarray:
        """Smoothed center speed in body heights per second."""
        c = self.centers
        if len(c) < 2:
            return np.zeros(len(c))
        k = max(1, int(SMOOTH_S * self.fps))
        kernel = np.ones(k) / k
        cs = np.c_[
            np.convolve(np.pad(c[:, 0], (k // 2, k - 1 - k // 2), mode="edge"), kernel, "valid"),
            np.convolve(np.pad(c[:, 1], (k // 2, k - 1 - k // 2), mode="edge"), kernel, "valid"),
        ]
        dt = np.diff(self.frames) / self.fps
        v = np.linalg.norm(np.diff(cs, axis=0), axis=1) / np.maximum(dt, 1e-6)
        h = np.convolve(np.pad(self.heights, (k // 2, k - 1 - k // 2), mode="edge"), kernel, "valid")
        v = np.r_[v[0], v] / np.maximum(h, 1.0)
        return v


def load_tracks(data: dict[str, Any]) -> list[Track]:
    fps = float(data["fps"])
    return [
        Track(
            gid=gid,
            role=t.get("role", ""),
            frames=np.asarray(t["frames"], int),
            boxes=np.asarray(t["boxes"], float),
            fps=fps,
        )
        for gid, t in data["tracks"].items()
    ]


def in_zone(pt: np.ndarray, poly: list[list[float]] | None) -> bool:
    if not poly:
        return False
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def foot(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, box[3]])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


# ---------------------------------------------------------------------------
# Roles (fallback when Astra did not label a track)
# ---------------------------------------------------------------------------


def heuristic_role(tr: Track, cam: dict[str, Any], n_frames: int) -> str:
    if tr.life < 1.0:
        return ""
    feet = [foot(b) for b in tr.boxes]
    kitchen_frac = np.mean([in_zone(p, cam.get("kitchen")) for p in feet])
    if kitchen_frac > 0.3:
        return "staff"
    v = tr.speed()
    moving = v > MOVE_SPEED
    stationary_tail = 0
    for s in v[::-1]:
        if s >= STATIONARY_SPEED:
            break
        stationary_tail += 1
    walked = tr.path() >= 1.0 and tr.reach() >= 0.5
    arrived_and_sat = walked and stationary_tail / tr.fps >= 3.0 and tr.frames[-1] >= n_frames - 2
    if arrived_and_sat:
        return "customer"
    return "staff" if walked and moving.mean() >= 0.25 else "customer"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _seated_index(tr: Track, v: np.ndarray) -> int | None:
    """Box height drops (standing → seated) and the person stays put; else first long stop."""
    walking = v > MOVE_SPEED
    if not walking.any():
        return None
    hold = int(SEATED_MIN_S * tr.fps)
    win = int(tr.fps)  # compare with the last second: box height changes with perspective
    c, h = tr.centers, tr.heights
    for i in range(max(int(np.argmax(walking)), 1), len(c) - hold):
        if tr.travel(0, i) < 1.0 or h[i] >= 0.85 * h[max(0, i - win) : i].max():
            continue
        if np.max(np.linalg.norm(c[i : i + hold] - c[i], axis=1)) < 0.5 * h[i]:
            return i
    for a, b in runs(v < STATIONARY_SPEED):
        if a > 0 and (b - a + 1) / tr.fps >= SEATED_MIN_S and tr.travel(0, a) >= 1.0:
            return a
    return None


def _reacquired_birth(tr: Track, tracks: list[Track]) -> bool:
    c0, h0 = tr.centers[0], tr.heights[0]
    for o in tracks:
        if o is tr or o.life < 0.1:
            continue
        if 0 <= tr.t0 - o.t1 <= REACQUIRE_S and np.linalg.norm(o.centers[-1] - c0) < 1.0 * h0:
            return True
    return False


def _reacquired_death(tr: Track, tracks: list[Track]) -> bool:
    c1, h1 = tr.centers[-1], tr.heights[-1]
    for o in tracks:
        if o is tr or o.life < 0.1:
            continue
        if 0 <= o.t0 - tr.t1 <= REACQUIRE_S and np.linalg.norm(o.centers[0] - c1) < 1.0 * h1:
            return True
    return False


def extract_events(data: dict[str, Any], cam: dict[str, Any]) -> dict[str, Any]:
    fps = float(data["fps"])
    W, H, n_frames = data["width"], data["height"], data["n_frames"]
    t_end = (n_frames - 1) / fps
    all_tracks = load_tracks(data)
    tracks = [t for t in all_tracks if t.life >= MIN_TRACK_S]

    roles: dict[str, str] = {}
    for tr in tracks:
        tr.role = tr.role or heuristic_role(tr, cam, n_frames)
        roles[tr.gid] = tr.role

    events: list[dict[str, Any]] = []
    seated_at: dict[str, float] = {}
    margin = cam.get("edge_margin", 0.06)

    for tr in tracks:
        v = tr.speed()
        born_mid = tr.frames[0] > 0
        feet = [foot(b) for b in tr.boxes]

        # enter: stands in the doorway (feet behind the threshold), then moves into the hall
        if in_zone(feet[0], cam.get("door")) and not (born_mid and _reacquired_birth(tr, all_tracks)):
            left_door = [i for i, p in enumerate(feet) if not in_zone(p, cam.get("door"))]
            if left_door:
                moving = np.nonzero(v[: left_door[0] + 1] > MOVE_SPEED)[0]
                i = moving[0] if len(moving) else left_door[0]
                events.append({"t": round(tr.frames[i] / fps, 3), "type": "enter", "person": tr.gid})

        # leave_kitchen: staff appears out of the (unseen) kitchen
        if born_mid and in_zone(foot(tr.boxes[0]), cam.get("kitchen")) and not _reacquired_birth(tr, all_tracks):
            events.append({"t": round(tr.t0, 3), "type": "leave_kitchen", "person": tr.gid})

        # exit: track ends at the frame edge or in the doorway, and nobody picks it up
        if tr.frames[-1] < n_frames - 1:
            b = tr.boxes[-1]
            cx = (b[0] + b[2]) / 2
            at_edge = cx < margin * W or cx > (1 - margin) * W or b[3] > (1 - margin) * H and b[1] > 0.5 * H
            if (at_edge or in_zone(feet[-1], cam.get("door"))) and not _reacquired_death(tr, all_tracks):
                events.append({"t": round(tr.t1, 3), "type": "exit", "person": tr.gid})

        # seated: a customer who walked (≥1 body height) comes to rest
        if tr.role == "customer":
            i_seat = _seated_index(tr, v)
            if i_seat is not None:
                seated_at[tr.gid] = tr.frames[i_seat] / fps
                events.append({"t": round(seated_at[tr.gid], 3), "type": "seated", "person": tr.gid})

    # service: staff stops next to a seated customer
    customers = [t for t in tracks if t.role == "customer"]
    cust_pos = {  # head point: the table side a waiter actually approaches
        c.gid: dict(zip(
            c.frames.tolist(),
            np.c_[(c.boxes[:, 0] + c.boxes[:, 2]) / 2, c.boxes[:, 1] + 0.12 * c.heights],
        ))
        for c in customers
    }
    services: list[tuple[str, str, float, float]] = []
    for s in (t for t in tracks if t.role == "staff"):
        v = s.speed()
        near: list[str | None] = []
        for i, f in enumerate(s.frames.tolist()):
            best, best_d = None, SERVICE_DIST
            if v[i] < MOVE_SPEED:
                for gid, pos in cust_pos.items():
                    p = pos.get(f)
                    if p is None:
                        continue
                    d = np.linalg.norm(p - s.centers[i]) / max(s.heights[i], 1.0)
                    if d < best_d:
                        best, best_d = gid, d
            near.append(best)

        mask = np.array([n is not None for n in near])
        # bridge short gaps
        gap = int(SERVICE_GAP_S * fps)
        for a, b_ in runs(~mask):
            if 0 < a and b_ < len(mask) - 1 and (b_ - a + 1) <= gap:
                mask[a : b_ + 1] = True
        for a, b_ in runs(mask):
            if (s.frames[b_] - s.frames[a]) / fps < SERVICE_MIN_S:
                continue
            who = Counter(n for n in near[a : b_ + 1] if n).most_common(1)
            if not who:
                continue
            cust = who[0][0]
            ts, te = s.frames[a] / fps, s.frames[b_] / fps
            services.append((s.gid, cust, ts, te))
            if s.frames[a] > 0:
                events.append({"t": round(ts, 3), "type": "service_start", "staff": s.gid, "customer": cust})
            if s.frames[b_] < n_frames - 1:
                events.append({"t": round(te, 3), "type": "service_end", "staff": s.gid, "customer": cust})

    # states
    states: list[dict[str, Any]] = []
    first_service = {}
    for staff, cust, ts, _ in sorted(services, key=lambda x: x[2]):
        first_service.setdefault(cust, ts)
    for c in customers:
        start = seated_at.get(c.gid, c.t0 if c.frames[0] == 0 else None)
        if start is None:
            continue
        if c.gid in first_service and first_service[c.gid] > start:
            states.append({"type": "waiting", "person": c.gid, "start": round(start, 3), "end": round(first_service[c.gid], 3)})
        elif c.gid in seated_at and c.gid not in first_service:
            states.append({"type": "waiting", "person": c.gid, "start": round(start, 3), "end": round(c.t1, 3)})
        if c.frames[0] == 0 and c.gid not in seated_at:
            states.append({"type": "consuming", "person": c.gid, "start": round(c.t0, 3), "end": round(c.t1, 3)})

    events = _dedupe_exits(events, {t.gid: t for t in tracks})
    events.sort(key=lambda e: e["t"])
    return {"events": events, "states": states, "roles": roles, "duration_s": round(t_end, 3)}


def _dedupe_exits(events: list[dict[str, Any]], by_gid: dict[str, Track]) -> list[dict[str, Any]]:
    """Several fragments of one person leaving at the same spot → keep the longest track."""
    exits = sorted((e for e in events if e["type"] == "exit"), key=lambda e: -by_gid[e["person"]].life)
    kept: list[dict[str, Any]] = []
    for e in exits:
        tr = by_gid[e["person"]]
        dup = any(
            abs(e["t"] - k["t"]) <= 0.5
            and np.linalg.norm(tr.centers[-1] - by_gid[k["person"]].centers[-1]) < 1.0 * tr.heights[-1]
            for k in kept
        )
        if not dup:
            kept.append(e)
    return [e for e in events if e["type"] != "exit"] + kept


def load_camera(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DEFAULT_CAMERA).read_text(encoding="utf-8"))
