"""Approach → interaction → departure service finite-state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from flowmetric.domain import ServiceInterval, Track

MOVE_SPEED = 0.2
STATIONARY_SPEED = 0.12
SERVICE_DIST = 0.9
SERVICE_RELEASE_DIST = 1.1
SERVICE_MIN_S = 1.0
SERVICE_GAP_S = 0.5
SERVICE_MERGE_GAP_S = 0.5
SERVICE_MIN_OBSERVATIONS = 3
SERVICE_MIN_COVERAGE = 0.5
SERVICE_APPROACH_WINDOW_S = 1.5
SERVICE_APPROACH_TRIGGER_DELTA = 0.10
SERVICE_APPROACH_MIN_DELTA = 0.18
SERVICE_ACQUIRE_S = 0.4
SERVICE_DEPARTURE_S = 0.3
SERVICE_DEPARTURE_MIN_DELTA = 0.05
SERVICE_STATIONARY_RATIO = 0.72
SERVICE_COOLDOWN_S = 1.5
TABLE_GAP = 0.35
TABLE_RELEASE_GAP = 0.55


def rect_gap(a: np.ndarray, b: list[float]) -> float:
    """Euclidean gap between xyxy boxes, or zero when they overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


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


def extract_service_intervals(
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
