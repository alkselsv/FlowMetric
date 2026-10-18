"""Independent waiting and consuming state machines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flowmetric.domain import ServiceInterval, Track

MOVE_SPEED = 0.2
CUSTOMER_ACTIVITY_WINDOW_S = 1.0
CUSTOMER_ACTIVITY_MIN_PATH = 0.10
CUSTOMER_STATE_MIN_COVERAGE = 0.5


def customer_activity_start(tr: Track, start_index: int) -> tuple[int | None, float]:
    """Find sustained upper-body activity independently of service events."""
    if len(tr.frames) - start_index < 3:
        return None, 0.0
    window_frames = max(1, int(math.ceil(CUSTOMER_ACTIVITY_WINDOW_S * tr.fps)))
    centers = tr.centers
    feet = np.c_[(tr.boxes[:, 0] + tr.boxes[:, 2]) / 2, tr.boxes[:, 3]]
    residual = centers - feet
    speeds = tr.speed()
    best_score = 0.0
    for left in range(start_index, len(tr.frames) - 2):
        right = int(
            np.searchsorted(
                tr.frames, tr.frames[left] + window_frames, side="right"
            )
        )
        right = min(right, len(tr.frames))
        if right - left < 3:
            continue
        span = int(tr.frames[right - 1] - tr.frames[left])
        coverage = (right - left) / max(span + 1, 1)
        if coverage < CUSTOMER_STATE_MIN_COVERAGE:
            continue
        height = max(float(np.median(tr.heights[left:right])), 1.0)
        path = float(
            np.sum(np.linalg.norm(np.diff(residual[left:right], axis=0), axis=1))
            / height
        )
        stationary_ratio = float(np.mean(speeds[left:right] < MOVE_SPEED))
        score = (
            min(1.0, path / max(2 * CUSTOMER_ACTIVITY_MIN_PATH, 1e-6))
            * stationary_ratio
        )
        best_score = max(best_score, score)
        if path >= CUSTOMER_ACTIVITY_MIN_PATH and stationary_ratio >= 0.7:
            return int(tr.frames[left]), round(score, 3)
    return None, round(best_score, 3)


def customer_states(
    customers: list[Track],
    *,
    seated_at: dict[str, float],
    seat_idx: dict[str, int],
    cust_table: dict[str, int | None],
    tables: list[list[float]],
    service_intervals: list[ServiceInterval],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run waiting and consuming as explicit, independently audited states."""
    deliveries: dict[str, list[ServiceInterval]] = {}
    for interval in service_intervals:
        deliveries.setdefault(interval.customer, []).append(interval)

    states: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for customer in customers:
        if customer.gid in seat_idx:
            start_index = seat_idx[customer.gid]
            seated_start = float(seated_at[customer.gid])
            observed_seating = True
        elif customer.frames[0] == 0:
            start_index = 0
            seated_start = customer.t0
            observed_seating = False
        else:
            continue
        table_index = cust_table.get(customer.gid)

        activity_frame, activity_confidence = customer_activity_start(
            customer, start_index
        )
        delivery = min(
            deliveries.get(customer.gid, []),
            key=lambda interval: interval.start_frame,
            default=None,
        )
        delivery_time = (
            delivery.start_frame / customer.fps if delivery is not None else None
        )
        activity_time = (
            activity_frame / customer.fps if activity_frame is not None else None
        )

        wait_end_candidates = []
        if delivery_time is not None and delivery_time > seated_start:
            wait_end_candidates.append((delivery_time, "verified_delivery"))
        elif activity_time is not None and activity_time > seated_start:
            wait_end_candidates.append((activity_time, "customer_table_activity"))
        if observed_seating or delivery is not None:
            wait_end, wait_evidence = (
                min(wait_end_candidates, key=lambda item: item[0])
                if wait_end_candidates
                else (customer.t1, "track_end")
            )
            if wait_end > seated_start:
                states.append(
                    {
                        "type": "waiting",
                        "person": customer.gid,
                        "start": round(seated_start, 3),
                        "end": round(wait_end, 3),
                        "confidence": (
                            0.85 if wait_evidence == "verified_delivery" else 0.65
                        ),
                        "evidence": wait_evidence,
                    }
                )

        initial_window = max(1, int(math.ceil(customer.fps)))
        initial_stationary = float(
            np.mean(
                customer.speed()[: min(len(customer.frames), initial_window)]
                < MOVE_SPEED
            )
        )
        consuming_start: float | None = None
        consuming_evidence: str | None = None
        consuming_confidence = 0.0
        left_censored = False
        if not observed_seating and initial_stationary >= 0.7:
            consuming_start = customer.t0
            consuming_evidence = "initial_seated_stability"
            consuming_confidence = 0.55 + 0.25 * initial_stationary
            left_censored = True
        elif activity_time is not None and activity_time >= seated_start:
            consuming_start = activity_time
            consuming_evidence = "customer_table_activity"
            consuming_confidence = 0.55 + 0.35 * activity_confidence
        if consuming_start is not None and customer.t1 > consuming_start:
            states.append(
                {
                    "type": "consuming",
                    "person": customer.gid,
                    "start": round(consuming_start, 3),
                    "end": round(customer.t1, 3),
                    "confidence": round(min(consuming_confidence, 1.0), 3),
                    "evidence": consuming_evidence,
                    **({"left_censored": True} if left_censored else {}),
                }
            )

        audit.append(
            {
                "person": customer.gid,
                "table": table_index,
                "seated_start": round(seated_start, 3),
                "observed_seating": observed_seating,
                "delivery_start": (
                    round(delivery_time, 3) if delivery_time is not None else None
                ),
                "activity_start": (
                    round(activity_time, 3) if activity_time is not None else None
                ),
                "activity_confidence": activity_confidence,
                "initial_stationary_ratio": round(initial_stationary, 3),
            }
        )
    return states, audit
