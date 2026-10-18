"""Service events from person trajectories (tracks.json → events.json).

Events: enter, exit, leave_kitchen, seated, service_start, service_end.
States: waiting, consuming.
Roles come from Astra when available; otherwise a motion/zone heuristic fills them in.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowmetric.domain import (
    ServiceCandidate,
    ServiceDecision,
    ServiceInterval,
    ServiceVerifier,
    Track,
)
from flowmetric.customer.states import (
    customer_activity_start as _customer_activity_start,
    customer_states as _customer_states,
)
from flowmetric.service.fsm import (
    MOVE_SPEED,
    SERVICE_ACQUIRE_S,
    SERVICE_APPROACH_MIN_DELTA,
    SERVICE_APPROACH_TRIGGER_DELTA,
    SERVICE_APPROACH_WINDOW_S,
    SERVICE_COOLDOWN_S,
    SERVICE_DEPARTURE_MIN_DELTA,
    SERVICE_DEPARTURE_S,
    SERVICE_DIST,
    SERVICE_GAP_S,
    SERVICE_MERGE_GAP_S,
    SERVICE_MIN_COVERAGE,
    SERVICE_MIN_OBSERVATIONS,
    SERVICE_MIN_S,
    SERVICE_RELEASE_DIST,
    SERVICE_STATIONARY_RATIO,
    STATIONARY_SPEED,
    TABLE_GAP,
    TABLE_RELEASE_GAP,
    extract_service_intervals as _extract_service_intervals,
    rect_gap,
)
from flowmetric.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_CAMERA = ROOT / "config" / "camera.json"

MIN_TRACK_S = 0.5  # shorter tracks are detector noise / fragments
SEATED_MIN_S = 2.0
REACQUIRE_S = 1.5  # a birth this soon after a nearby death is the same person
EXIT_MAX_GAP_S = 0.2  # larger terminal observation gaps are not a continuous exit track
EXIT_MIN_COVERAGE = 0.5
TABLE_REACH = 0.3  # seated "lap point" may lie this far (customer heights) outside a table box
# SERVICE_APPROACH_MIN_DELTA and SERVICE_STATIONARY_RATIO define the development
# benchmark operating point and require held-out validation before use as
# universal defaults.


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


def table_at(box: np.ndarray, tables: list[list[float]]) -> int | None:
    """Table a (seated) person is at: the one around their lap point, nearest to its center."""
    h = box[3] - box[1]
    lap = np.array([(box[0] + box[2]) / 2, box[1] + 0.6 * h])
    m = TABLE_REACH * h
    best, best_d = None, np.inf
    for k, t in enumerate(tables):
        if t[0] - m <= lap[0] <= t[2] + m and t[1] - m <= lap[1] <= t[3] + m:
            d = np.linalg.norm(lap - [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2]) / np.hypot(t[2] - t[0], t[3] - t[1])
            if d < best_d:
                best, best_d = k, d
    return best


def home_table(tr: "Track", tables: list[list[float]], start: int = 0) -> int | None:
    """Most frequent table of a customer from index `start` on (None if never at a table)."""
    votes = Counter(table_at(b, tables) for b in tr.boxes[start:])
    votes.pop(None, None)
    return votes.most_common(1)[0][0] if votes else None


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


def _apply_service_verifier(
    candidates: list[ServiceCandidate],
    verifier: ServiceVerifier | None,
) -> tuple[list[ServiceInterval], list[tuple[ServiceCandidate, ServiceDecision]]]:
    """Apply an optional verifier before point events and states are derived."""
    if verifier is None:
        return [candidate.interval for candidate in candidates], []

    decisions = verifier(candidates)
    if not isinstance(decisions, list) or len(decisions) != len(candidates):
        raise ValueError(
            "service verifier must return exactly one ServiceDecision per candidate"
        )

    accepted: list[ServiceInterval] = []
    audited: list[tuple[ServiceCandidate, ServiceDecision]] = []
    for index, (candidate, decision) in enumerate(zip(candidates, decisions)):
        if not isinstance(decision, ServiceDecision):
            raise TypeError(
                f"service verifier result {index} must be a ServiceDecision"
            )
        interval = candidate.interval
        if decision.staff_id != interval.staff or decision.customer_id != interval.customer:
            raise ValueError(
                "service verifier changed candidate identities: "
                f"expected {interval.staff}->{interval.customer}, got "
                f"{decision.staff_id}->{decision.customer_id}"
            )
        if not isinstance(decision.is_service, bool):
            raise TypeError("service verifier is_service must be boolean")
        if (
            isinstance(decision.confidence, bool)
            or not isinstance(decision.confidence, (int, float))
            or not math.isfinite(float(decision.confidence))
            or not 0 <= float(decision.confidence) <= 1
        ):
            raise ValueError("service verifier confidence must be finite and in [0, 1]")
        if not isinstance(decision.source, str) or not decision.source:
            raise ValueError("service verifier source must be a non-empty string")
        if any(
            isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
            for frame in decision.sampled_frames
        ):
            raise ValueError("service verifier sampled_frames must be non-negative integers")
        if decision.model is not None and (
            not isinstance(decision.model, str) or not decision.model
        ):
            raise ValueError("service verifier model must be a non-empty string or None")
        if decision.details is not None:
            if not isinstance(decision.details, dict):
                raise TypeError("service verifier details must be an object or None")
            try:
                json.dumps(decision.details, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"service verifier details must be strict JSON: {exc}"
                ) from exc

        audited.append((candidate, decision))
        if decision.is_service:
            accepted.append(interval)
    return accepted, audited


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


def _reliable_exit_tail(tr: Track) -> bool:
    """Whether the track ends with a dense, continuous segment long enough to trust."""
    max_gap_frames = max(1, int(EXIT_MAX_GAP_S * tr.fps))
    start = len(tr.frames) - 1
    while start > 0 and tr.frames[start] - tr.frames[start - 1] <= max_gap_frames:
        start -= 1

    span_frames = int(tr.frames[-1] - tr.frames[start])
    if span_frames / tr.fps < MIN_TRACK_S:
        return False
    coverage = (len(tr.frames) - start) / (span_frames + 1)
    return coverage >= EXIT_MIN_COVERAGE


def extract_events(
    data: dict[str, Any],
    cam: dict[str, Any],
    *,
    service_verifier: ServiceVerifier | None = None,
) -> dict[str, Any]:
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
    seat_idx: dict[str, int] = {}
    margin = cam.get("edge_margin", 0.06)
    tables = [t["box"] if isinstance(t, dict) else t for t in cam.get("tables", [])]

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
        if (
            tr.role == "staff"
            and born_mid
            and in_zone(foot(tr.boxes[0]), cam.get("kitchen"))
            and not _reacquired_birth(tr, all_tracks)
        ):
            events.append({"t": round(tr.t0, 3), "type": "leave_kitchen", "person": tr.gid})

        # exit: track ends at the frame edge or in the doorway, and nobody picks it up
        if tr.frames[-1] < n_frames - 1 and _reliable_exit_tail(tr):
            b = tr.boxes[-1]
            cx = (b[0] + b[2]) / 2
            at_edge = cx < margin * W or cx > (1 - margin) * W or b[3] > (1 - margin) * H and b[1] > 0.5 * H
            if (at_edge or in_zone(feet[-1], cam.get("door"))) and not _reacquired_death(tr, all_tracks):
                events.append({"t": round(tr.t1, 3), "type": "exit", "person": tr.gid})

        # seated: a customer who walked (≥1 body height) comes to rest
        if tr.role == "customer":
            i_seat = _seated_index(tr, v)
            if i_seat is not None and tables and home_table(tr, tables, i_seat) is None:
                i_seat = None  # came to rest away from any table (standing in the aisle, queueing)
            if i_seat is not None:
                seat_idx[tr.gid] = i_seat
                seated_at[tr.gid] = tr.frames[i_seat] / fps
                events.append({"t": round(seated_at[tr.gid], 3), "type": "seated", "person": tr.gid})

    # service: staff stops next to a seated customer
    customers = [t for t in tracks if t.role == "customer"]
    # A customer visible from the beginning may already be seated when the clip
    # starts.  A customer first seen mid-video must have an observed seated
    # transition before they are eligible for a service event.
    eligible_from = {
        c.gid: 0 if c.frames[0] == 0 else int(c.frames[seat_idx[c.gid]])
        for c in customers
        if c.frames[0] == 0 or c.gid in seat_idx
    }
    cust_table = {
        c.gid: home_table(c, tables, 0 if c.frames[0] == 0 else seat_idx[c.gid])
        for c in customers
        if c.gid in eligible_from
    } if tables else {}
    eligible_customers = [
        c for c in customers
        if c.gid in eligible_from and (not tables or cust_table[c.gid] is not None)
    ]
    cust_pos = {  # head point: the table side a waiter actually approaches
        c.gid: dict(zip(
            c.frames.tolist(),
            np.c_[(c.boxes[:, 0] + c.boxes[:, 2]) / 2, c.boxes[:, 1] + 0.12 * c.heights],
        ))
        for c in eligible_customers
    }
    local_service_intervals = _extract_service_intervals(
        [t for t in tracks if t.role == "staff"],
        eligible_customers,
        cust_pos,
        eligible_from,
        cust_table,
        tables,
        n_frames,
        fps,
    )
    service_candidates = [
        ServiceCandidate(
            interval=interval,
            table_box=(
                tuple(float(value) for value in tables[table_index])
                if (table_index := cust_table.get(interval.customer)) is not None
                else None
            ),
        )
        for interval in local_service_intervals
    ]
    service_intervals, verification_audit = _apply_service_verifier(
        service_candidates, service_verifier
    )

    def interval_key(interval: ServiceInterval) -> tuple[str, str, int, int]:
        return (
            interval.staff,
            interval.customer,
            interval.start_frame,
            interval.end_frame,
        )

    verification_by_interval = {
        interval_key(candidate.interval): (candidate, decision)
        for candidate, decision in verification_audit
    }
    services: list[tuple[str, str, float, float]] = []
    for interval in service_intervals:
        ts, te = interval.start_frame / fps, interval.end_frame / fps
        services.append((interval.staff, interval.customer, ts, te))
        verification = verification_by_interval.get(interval_key(interval))
        # A left-censored interval is represented only in service_intervals.
        # Its unknown start makes a point service_end ambiguous on short clips.
        if not interval.left_censored:
            start_event: dict[str, Any] = {
                "t": round(ts, 3),
                "type": "service_start",
                "staff": interval.staff,
                "customer": interval.customer,
                "confidence": interval.confidence,
            }
            if verification is not None:
                _, decision = verification
                start_event["verification"] = decision.source
                start_event["verifier_confidence"] = round(
                    float(decision.confidence), 3
                )
            if interval.right_censored:
                start_event["right_censored"] = True
            events.append(start_event)
        if not interval.left_censored and not interval.right_censored:
            end_event: dict[str, Any] = {
                "t": round(te, 3),
                "type": "service_end",
                "staff": interval.staff,
                "customer": interval.customer,
                "confidence": interval.confidence,
            }
            if verification is not None:
                _, decision = verification
                end_event["verification"] = decision.source
                end_event["verifier_confidence"] = round(
                    float(decision.confidence), 3
                )
            events.append(end_event)

    states, customer_state_audit = _customer_states(
        customers,
        seated_at=seated_at,
        seat_idx=seat_idx,
        cust_table=cust_table,
        tables=tables,
        service_intervals=service_intervals,
    )

    events = _dedupe_exits(events, {t.gid: t for t in tracks})
    events.sort(key=lambda e: e["t"])
    interval_metadata: list[dict[str, Any]] = []
    for interval in service_intervals:
        metadata: dict[str, Any] = {
            "staff": interval.staff,
            "customer": interval.customer,
            "start": None if interval.left_censored else round(interval.start_frame / fps, 3),
            "end": None if interval.right_censored else round(interval.end_frame / fps, 3),
            "left_censored": interval.left_censored,
            "right_censored": interval.right_censored,
            "confidence": interval.confidence,
        }
        verification = verification_by_interval.get(interval_key(interval))
        if verification is not None:
            _, decision = verification
            metadata["verification"] = decision.source
            metadata["verifier_confidence"] = round(
                float(decision.confidence), 3
            )
        interval_metadata.append(metadata)

    result = {
        "events": events,
        "states": states,
        "roles": roles,
        "service_intervals": interval_metadata,
        "customer_state_audit": customer_state_audit,
        "duration_s": round(t_end, 3),
    }
    if service_verifier is not None:
        result["service_candidates"] = [
            {
                "staff": candidate.interval.staff,
                "customer": candidate.interval.customer,
                "start": (
                    None
                    if candidate.interval.left_censored
                    else round(candidate.interval.start_frame / fps, 3)
                ),
                "end": (
                    None
                    if candidate.interval.right_censored
                    else round(candidate.interval.end_frame / fps, 3)
                ),
                "observed_start": round(candidate.interval.start_frame / fps, 3),
                "observed_end": round(candidate.interval.end_frame / fps, 3),
                "left_censored": candidate.interval.left_censored,
                "right_censored": candidate.interval.right_censored,
                "rule_confidence": candidate.interval.confidence,
                "table_box": list(candidate.table_box) if candidate.table_box else None,
                "decision": {
                    "is_service": decision.is_service,
                    "staff_id": decision.staff_id,
                    "customer_id": decision.customer_id,
                    "confidence": round(float(decision.confidence), 3),
                    "source": decision.source,
                    "sampled_frames": list(decision.sampled_frames),
                    **({"model": decision.model} if decision.model else {}),
                    **({"details": decision.details} if decision.details is not None else {}),
                },
            }
            for candidate, decision in verification_audit
        ]
    return result


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
