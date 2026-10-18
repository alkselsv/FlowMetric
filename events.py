"""Service events from person trajectories (tracks.json → events.json).

Events: enter, exit, leave_kitchen, seated, service_start, service_end.
States: waiting, consuming.
Roles come from Astra when available; otherwise a motion/zone heuristic fills them in.
"""

from __future__ import annotations

import json
from collections import Counter, deque
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
EXIT_MAX_GAP_S = 0.2  # larger terminal observation gaps are not a continuous exit track
EXIT_MIN_COVERAGE = 0.5
SERVICE_DIST = 0.9  # staff-customer center distance, in staff box heights
SERVICE_RELEASE_DIST = 1.1  # expanded zone: interaction cannot re-arm before leaving it
SERVICE_MIN_S = 1.0
SERVICE_GAP_S = 0.5
SERVICE_MERGE_GAP_S = 0.5  # merge only brief jitter; a confirmed end is otherwise preserved
SERVICE_MIN_OBSERVATIONS = 3
SERVICE_MIN_COVERAGE = 0.5
SERVICE_APPROACH_WINDOW_S = 1.5
SERVICE_APPROACH_TRIGGER_DELTA = 0.10
SERVICE_APPROACH_MIN_DELTA = 0.18
SERVICE_ACQUIRE_S = 0.4
SERVICE_DEPARTURE_S = 0.3
SERVICE_DEPARTURE_MIN_DELTA = 0.05
SERVICE_STATIONARY_RATIO = 0.72
SERVICE_COOLDOWN_S = 1.5  # pair rearm timeout after disappearance, not the merge window
TABLE_GAP = 0.35  # max staff-box to customer-table gap during service, in staff box heights
TABLE_RELEASE_GAP = 0.55  # expanded table zone used by release hysteresis
TABLE_REACH = 0.3  # seated "lap point" may lie this far (customer heights) outside a table box

# SERVICE_APPROACH_MIN_DELTA and SERVICE_STATIONARY_RATIO deliberately encode the
# current benchmark operating point.  They separated the labelled positives from
# the hard geometric negatives on the development clip, but still need validation
# on held-out cameras before they should be treated as universal defaults.


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


@dataclass(frozen=True)
class ServiceInterval:
    """A confirmed service episode; censored sides have no observable point event."""

    staff: str
    customer: str
    start_frame: int
    end_frame: int
    left_censored: bool
    right_censored: bool
    confidence: float


@dataclass(frozen=True)
class _ServiceObservation:
    distance: float
    speed: float
    acquire: bool
    expanded: bool


@dataclass
class _ServiceState:
    phase: str = "idle"
    customer: str | None = None
    left_censored: bool = False
    approach_baseline: float = 0.0
    approach_start: int | None = None
    acquire_start: int | None = None
    acquire_last: int | None = None
    interaction_start: int | None = None
    interaction_frames: list[int] | None = None
    interaction_distances: list[float] | None = None
    interaction_speeds: list[float] | None = None
    last_observed: int | None = None
    departure_start: int | None = None
    departure_min_distance: float = np.inf
    outside_start: int | None = None
    pending_end: int | None = None
    cooldown_until: float = 0.0

    def clear(self) -> None:
        self.phase = "idle"
        self.customer = None
        self.left_censored = False
        self.approach_baseline = 0.0
        self.approach_start = None
        self.acquire_start = None
        self.acquire_last = None
        self.interaction_start = None
        self.interaction_frames = None
        self.interaction_distances = None
        self.interaction_speeds = None
        self.last_observed = None
        self.departure_start = None
        self.departure_min_distance = np.inf
        self.outside_start = None
        self.pending_end = None
        self.cooldown_until = 0.0


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


def rect_gap(a: np.ndarray, b: list[float]) -> float:
    """Euclidean gap between two xyxy boxes (0 when they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


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


def _finish_service_interval(
    staff: str,
    state: _ServiceState,
    end_frame: int,
    fps: float,
    *,
    right_censored: bool,
) -> ServiceInterval | None:
    """Validate an FSM episode and turn it into a stable public interval."""
    if state.customer is None or state.interaction_start is None:
        return None

    samples = [
        (f, d, v)
        for f, d, v in zip(
            state.interaction_frames or [],
            state.interaction_distances or [],
            state.interaction_speeds or [],
        )
        if state.interaction_start <= f <= end_frame
    ]
    if not samples:
        return None
    sample_frames = np.asarray([s[0] for s in samples], dtype=int)
    duration = (end_frame - state.interaction_start) / fps
    expected = end_frame - state.interaction_start + 1
    coverage = len(np.unique(sample_frames)) / max(expected, 1)
    if (
        duration < SERVICE_MIN_S
        or len(samples) < SERVICE_MIN_OBSERVATIONS
        or coverage < SERVICE_MIN_COVERAGE
    ):
        return None

    # Include the slowing-down frames captured after the approach trigger.
    # Measuring only after ``acquire_start`` makes every accepted interval look
    # almost perfectly stationary and loses the strongest signal separating a
    # deliberate stop from somebody merely lingering while passing a table.
    speeds = np.asarray([speed for _, _, speed in samples], dtype=float)
    stationary_ratio = float(np.mean(speeds < STATIONARY_SPEED))
    if stationary_ratio < SERVICE_STATIONARY_RATIO:
        return None

    first_window = state.interaction_start + max(1, int(SERVICE_GAP_S * fps))
    first_distances = [d for f, d, _ in samples if f <= first_window]
    approach_drop = state.approach_baseline - float(np.median(first_distances))
    if not state.left_censored and approach_drop < SERVICE_APPROACH_MIN_DELTA:
        return None

    duration_score = min(1.0, duration / 2.0)
    approach_score = 0.5 if state.left_censored else np.clip(
        approach_drop / (2 * SERVICE_APPROACH_MIN_DELTA), 0.0, 1.0
    )
    stationary_score = np.clip(
        (stationary_ratio - SERVICE_STATIONARY_RATIO) / (1 - SERVICE_STATIONARY_RATIO), 0.0, 1.0
    )
    boundary_score = 0.5 if right_censored or state.left_censored else 1.0
    confidence = round(float(
        0.20 + 0.25 * approach_score + 0.25 * stationary_score
        + 0.20 * duration_score + 0.10 * boundary_score
    ), 3)
    return ServiceInterval(
        staff=staff,
        customer=state.customer,
        start_frame=state.interaction_start,
        end_frame=end_frame,
        left_censored=state.left_censored,
        right_censored=right_censored,
        confidence=confidence,
    )


def _extract_service_intervals(
    staff_tracks: list[Track],
    customers: list[Track],
    cust_pos: dict[str, dict[int, np.ndarray]],
    eligible_from: dict[str, int],
    cust_table: dict[str, int | None],
    tables: list[list[float]],
    n_frames: int,
    fps: float,
) -> list[ServiceInterval]:
    """Infer service with an explicit approach → interaction → departure FSM.

    A staff FSM owns at most one customer, and an ownership map gives every
    customer to at most one staff member.  A finished pair remains in cooldown
    until it has left the expanded zone, preventing near-zone jitter from
    producing duplicate end/start pairs.
    """
    if not staff_tracks or not customers:
        return []

    # Compare elapsed frame time directly.  Rounding a duration to a frame
    # count can relax maximum gaps or shorten minimum holds at low frame rates.
    gap_frames = SERVICE_GAP_S * fps
    approach_frames = SERVICE_APPROACH_WINDOW_S * fps
    acquire_frames = SERVICE_ACQUIRE_S * fps
    departure_frames = SERVICE_DEPARTURE_S * fps
    merge_frames = SERVICE_MERGE_GAP_S * fps
    cooldown_frames = SERVICE_COOLDOWN_S * fps

    staff_index = {
        tr.gid: {int(frame): i for i, frame in enumerate(tr.frames.tolist())}
        for tr in staff_tracks
    }
    customer_index = {
        tr.gid: {int(frame): i for i, frame in enumerate(tr.frames.tolist())}
        for tr in customers
    }
    speeds = {tr.gid: tr.speed() for tr in staff_tracks}
    by_staff = {tr.gid: tr for tr in staff_tracks}
    by_customer = {tr.gid: tr for tr in customers}
    states = {tr.gid: _ServiceState() for tr in staff_tracks}
    history: dict[tuple[str, str], deque[tuple[int, float, float]]] = {}
    # A completed pair alone is disarmed until it leaves the expanded zone.
    # This must not prevent the staff serving somebody else, or another staff
    # member serving the same customer.
    blocked_pairs: dict[tuple[str, str], int] = {}
    intervals: list[ServiceInterval] = []

    def geometry(staff_id: str, customer_id: str, frame: int) -> _ServiceObservation | None:
        si = staff_index[staff_id].get(frame)
        ci = customer_index[customer_id].get(frame)
        if si is None or ci is None or frame < eligible_from.get(customer_id, n_frames):
            return None
        staff = by_staff[staff_id]
        customer = by_customer[customer_id]
        point = cust_pos[customer_id].get(frame)
        if point is None:
            return None
        height = max(float(staff.heights[si]), 1.0)
        distance = float(np.linalg.norm(point - staff.centers[si]) / height)
        acquire_table = expanded_table = True
        table_id = cust_table.get(customer_id)
        if table_id is not None:
            gap = rect_gap(staff.boxes[si], tables[table_id]) / height
            acquire_table = gap <= TABLE_GAP
            expanded_table = gap <= TABLE_RELEASE_GAP
        return _ServiceObservation(
            distance=distance,
            speed=float(speeds[staff_id][si]),
            acquire=distance <= SERVICE_DIST and acquire_table,
            expanded=distance <= SERVICE_RELEASE_DIST and expanded_table,
        )

    def start_acquire(state: _ServiceState, frame: int, obs: _ServiceObservation) -> None:
        if state.acquire_start is None or (
            state.acquire_last is not None and frame - state.acquire_last > gap_frames
        ):
            state.acquire_start = frame
            state.interaction_frames = state.interaction_frames or []
            state.interaction_distances = state.interaction_distances or []
            state.interaction_speeds = state.interaction_speeds or []
        state.acquire_last = frame
        state.interaction_frames.append(frame)
        state.interaction_distances.append(obs.distance)
        state.interaction_speeds.append(obs.speed)

    def abandon_acquire(state: _ServiceState) -> None:
        state.acquire_start = None
        state.acquire_last = None

    def finish_pending(
        staff_id: str,
        state: _ServiceState,
        frame: int,
        obs: _ServiceObservation | None,
    ) -> None:
        """Commit a departure after its grace period and release ownership."""
        customer_id = state.customer
        end_frame = state.pending_end
        if customer_id is None or end_frame is None:
            state.clear()
            return
        interval = _finish_service_interval(
            staff_id, state, end_frame, fps, right_censored=False
        )
        if interval is not None:
            intervals.append(interval)
            if obs is None or obs.expanded:
                # An occlusion exactly when the grace period expires is not
                # evidence that the pair left the expanded zone.  Preserve the
                # last real observation so a genuinely missing track is still
                # released after a bounded absence.
                blocked_pairs[(staff_id, customer_id)] = (
                    frame if state.last_observed is None else state.last_observed
                )
        state.clear()

    all_customer_ids = sorted(c.gid for c in customers if c.gid in cust_pos)
    for frame in range(n_frames):
        observations: dict[tuple[str, str], _ServiceObservation] = {}
        for staff_id in sorted(by_staff):
            for customer_id in all_customer_ids:
                obs = geometry(staff_id, customer_id, frame)
                if obs is None:
                    continue
                observations[(staff_id, customer_id)] = obs
                key = (staff_id, customer_id)
                recent = history.setdefault(key, deque())
                recent.append((frame, obs.distance, obs.speed))
                while recent and frame - recent[0][0] > approach_frames:
                    recent.popleft()

        # Rearm only the completed pair after a real expanded-zone exit.  If
        # the track disappears, a bounded absence also releases the pair.
        for pair, last_seen in list(blocked_pairs.items()):
            obs = observations.get(pair)
            if obs is not None:
                blocked_pairs[pair] = frame
                if not obs.expanded:
                    del blocked_pairs[pair]
            elif frame - last_seen > cooldown_frames:
                del blocked_pairs[pair]

        active_owners = {
            state.customer
            for state in states.values()
            if state.phase in {"interacting", "departing", "ending"}
            and state.customer is not None
        }

        # Update active and tentative states before admitting new pairs.
        for staff_id in sorted(states):
            state = states[staff_id]
            if state.phase == "idle" or state.customer is None:
                continue
            obs = observations.get((staff_id, state.customer))

            if state.phase == "ending":
                if obs is not None:
                    previous_distance = (
                        state.interaction_distances[-1]
                        if state.interaction_distances else obs.distance
                    )
                    state.last_observed = frame
                    state.interaction_frames.append(frame)
                    state.interaction_distances.append(obs.distance)
                    state.interaction_speeds.append(obs.speed)
                    if (
                        frame <= state.cooldown_until
                        and obs.acquire
                        and (
                            obs.speed < MOVE_SPEED
                            or obs.distance < previous_distance - 0.01
                        )
                    ):
                        # A brief withdrawal/occlusion belongs to the same
                        # service episode; cancel the pending end.
                        state.phase = "interacting"
                        state.departure_start = None
                        state.departure_min_distance = np.inf
                        state.outside_start = None
                        state.pending_end = None
                        state.cooldown_until = 0.0
                        continue
                if frame >= state.cooldown_until:
                    customer_id = state.customer
                    finish_pending(staff_id, state, frame, obs)
                    active_owners.discard(customer_id)
                continue

            if obs is None:
                if state.last_observed is not None and frame - state.last_observed > gap_frames:
                    if state.interaction_start is not None:
                        interval = _finish_service_interval(
                            staff_id, state, state.last_observed, fps, right_censored=True
                        )
                        if interval is not None:
                            intervals.append(interval)
                    active_owners.discard(state.customer)
                    state.clear()
                continue

            state.last_observed = frame
            if state.phase == "approaching":
                if (
                    state.approach_start is not None
                    and frame - state.approach_start > approach_frames
                ):
                    state.clear()
                    continue
                if not obs.expanded:
                    state.clear()
                    continue
                if obs.acquire and obs.speed < STATIONARY_SPEED:
                    start_acquire(state, frame, obs)
                    if state.acquire_start is not None and frame - state.acquire_start >= acquire_frames:
                        approach_drop = state.approach_baseline - obs.distance
                        if (
                            state.left_censored
                            or approach_drop >= SERVICE_APPROACH_MIN_DELTA
                        ) and state.customer not in active_owners:
                            state.phase = "interacting"
                            state.interaction_start = state.approach_start
                            active_owners.add(state.customer)
                        else:
                            state.clear()
                elif obs.acquire and obs.speed < MOVE_SPEED:
                    # Pair selection happens while the staff is slowing down;
                    # acquisition itself starts only at a full stop.
                    abandon_acquire(state)
                    state.interaction_frames.append(frame)
                    state.interaction_distances.append(obs.distance)
                    state.interaction_speeds.append(obs.speed)
                else:
                    # A tentative pair is not sticky until acquisition.  This
                    # lets the nearest customer at the actual stop win instead
                    # of locking a bystander passed during the approach.
                    state.clear()
                continue

            # Interaction observations include tentative departure frames.  The
            # validator later cuts them at the timestamped start of departure.
            state.interaction_frames.append(frame)
            state.interaction_distances.append(obs.distance)
            state.interaction_speeds.append(obs.speed)

            if state.phase == "interacting":
                if obs.speed > MOVE_SPEED or not obs.acquire:
                    state.phase = "departing"
                    state.departure_start = frame
                    state.departure_min_distance = obs.distance
                    state.outside_start = frame if not obs.expanded else None
                continue

            state.departure_min_distance = min(state.departure_min_distance, obs.distance)
            if obs.expanded:
                state.outside_start = None
            elif state.outside_start is None:
                state.outside_start = frame
            departure_age = frame - (state.departure_start or frame)
            distance_rise = obs.distance - state.departure_min_distance
            moving_departure = (
                departure_age >= departure_frames
                and (distance_rise >= SERVICE_DEPARTURE_MIN_DELTA or not obs.acquire)
            )
            expanded_exit = (
                state.outside_start is not None
                and frame - state.outside_start >= departure_frames
            )
            if moving_departure or expanded_exit:
                end_frame = state.departure_start or frame
                interval = _finish_service_interval(
                    staff_id, state, end_frame, fps, right_censored=False
                )
                if interval is not None:
                    # Keep the endpoint pending for a short grace period.  A
                    # return cancels it and joins both pieces into one episode.
                    state.phase = "ending"
                    state.pending_end = end_frame
                    state.cooldown_until = end_frame + merge_frames
                else:
                    # A rejected tentative interaction must not monopolise the
                    # staff/customer after failing the semantic gates.
                    state.clear()
            elif obs.acquire and obs.speed < MOVE_SPEED:
                # A short gap/jitter or a tentative departure that returns to
                # the tight zone is still the same interaction.
                state.phase = "interacting"
                state.departure_start = None
                state.departure_min_distance = np.inf
                state.outside_start = None

        owners = active_owners
        candidates: list[tuple[float, str, str, bool, float]] = []
        for (staff_id, customer_id), obs in observations.items():
            if (
                states[staff_id].phase != "idle"
                or customer_id in owners
                or (staff_id, customer_id) in blocked_pairs
                or not obs.acquire
                or obs.speed >= MOVE_SPEED
            ):
                continue
            recent = history[(staff_id, customer_id)]
            baseline = max(distance for _, distance, _ in recent)
            drop = baseline - obs.distance
            staff_approached = any(speed >= MOVE_SPEED for _, _, speed in recent)
            left_censored = frame == 0 and obs.acquire and obs.speed < STATIONARY_SPEED
            if left_censored or (
                staff_approached and drop >= SERVICE_APPROACH_TRIGGER_DELTA
            ):
                candidates.append((obs.distance, staff_id, customer_id, left_censored, baseline))

        for _, staff_id, customer_id, left_censored, baseline in sorted(candidates):
            if states[staff_id].phase != "idle" or customer_id in owners:
                continue
            state = states[staff_id]
            state.phase = "approaching"
            state.customer = customer_id
            state.left_censored = left_censored
            state.approach_baseline = baseline
            state.approach_start = frame
            state.interaction_frames = []
            state.interaction_distances = []
            state.interaction_speeds = []
            state.last_observed = frame
            obs = observations[(staff_id, customer_id)]
            if obs.acquire and obs.speed < STATIONARY_SPEED:
                start_acquire(state, frame, obs)
            else:
                state.interaction_frames.append(frame)
                state.interaction_distances.append(obs.distance)
                state.interaction_speeds.append(obs.speed)

    # Open interactions have an unobserved right boundary.  A confirmed
    # departure remains observable even when its grace period reaches EOF.
    for staff_id, state in states.items():
        if state.interaction_start is None or state.last_observed is None:
            continue
        if state.phase == "ending" and state.pending_end is not None:
            interval = _finish_service_interval(
                staff_id, state, state.pending_end, fps, right_censored=False
            )
            if interval is not None:
                intervals.append(interval)
            continue
        interval = _finish_service_interval(
            staff_id, state, state.last_observed, fps, right_censored=True
        )
        if interval is not None:
            intervals.append(interval)

    return sorted(intervals, key=lambda x: (x.start_frame, x.staff, x.customer))


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
    service_intervals = _extract_service_intervals(
        [t for t in tracks if t.role == "staff"],
        eligible_customers,
        cust_pos,
        eligible_from,
        cust_table,
        tables,
        n_frames,
        fps,
    )
    services: list[tuple[str, str, float, float]] = []
    for interval in service_intervals:
        ts, te = interval.start_frame / fps, interval.end_frame / fps
        services.append((interval.staff, interval.customer, ts, te))
        # A left-censored interval is deliberately represented only in
        # service_intervals.  Its unknown start makes a point service_end too
        # ambiguous on short clips (and used to create a boundary false alarm).
        if not interval.left_censored:
            start_event: dict[str, Any] = {
                "t": round(ts, 3),
                "type": "service_start",
                "staff": interval.staff,
                "customer": interval.customer,
                "confidence": interval.confidence,
            }
            if interval.right_censored:
                start_event["right_censored"] = True
            events.append(start_event)
        if not interval.left_censored and not interval.right_censored:
            events.append({
                "t": round(te, 3),
                "type": "service_end",
                "staff": interval.staff,
                "customer": interval.customer,
                "confidence": interval.confidence,
            })

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
    interval_metadata = [
        {
            "staff": interval.staff,
            "customer": interval.customer,
            "start": None if interval.left_censored else round(interval.start_frame / fps, 3),
            "end": None if interval.right_censored else round(interval.end_frame / fps, 3),
            "left_censored": interval.left_censored,
            "right_censored": interval.right_censored,
            "confidence": interval.confidence,
        }
        for interval in service_intervals
    ]
    return {
        "events": events,
        "states": states,
        "roles": roles,
        "service_intervals": interval_metadata,
        "duration_s": round(t_end, 3),
    }


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
