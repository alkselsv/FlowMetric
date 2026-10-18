from __future__ import annotations

import unittest

import numpy as np

from events import ServiceInterval, Track, _customer_states


def _track(boxes: list[list[float]], *, fps: float = 10.0) -> Track:
    return Track(
        gid="12",
        role="customer",
        frames=np.arange(len(boxes), dtype=int),
        boxes=np.asarray(boxes, dtype=float),
        fps=fps,
    )


class CustomerStateTests(unittest.TestCase):
    def test_initial_stable_customer_is_consuming_without_service_event(self) -> None:
        customer = _track([[10, 10, 50, 90] for _ in range(21)])

        states, audit = _customer_states(
            [customer],
            seated_at={},
            seat_idx={},
            cust_table={"12": None},
            tables=[],
            service_intervals=[],
        )

        self.assertEqual([state["type"] for state in states], ["consuming"])
        self.assertTrue(states[0]["left_censored"])
        self.assertEqual(states[0]["evidence"], "initial_seated_stability")
        self.assertIsNone(audit[0]["delivery_start"])

    def test_delivery_ends_waiting_but_does_not_create_consuming(self) -> None:
        customer = _track([[10, 10, 50, 90] for _ in range(31)])
        delivery = ServiceInterval(
            staff="7",
            customer="12",
            start_frame=12,
            end_frame=20,
            left_censored=False,
            right_censored=False,
            confidence=0.8,
        )

        states, _ = _customer_states(
            [customer],
            seated_at={"12": 0.0},
            seat_idx={"12": 0},
            cust_table={"12": None},
            tables=[],
            service_intervals=[delivery],
        )

        self.assertEqual([state["type"] for state in states], ["waiting"])
        self.assertEqual(states[0]["end"], 1.2)
        self.assertEqual(states[0]["evidence"], "verified_delivery")

    def test_customer_activity_can_start_consuming_without_delivery(self) -> None:
        boxes: list[list[float]] = []
        for frame in range(31):
            top = min(25.0, frame * 2.5)
            boxes.append([10, top, 50, 100])
        customer = _track(boxes)

        states, audit = _customer_states(
            [customer],
            seated_at={"12": 0.0},
            seat_idx={"12": 0},
            cust_table={"12": None},
            tables=[],
            service_intervals=[],
        )

        consuming = [state for state in states if state["type"] == "consuming"]
        self.assertEqual(len(consuming), 1)
        self.assertEqual(consuming[0]["evidence"], "customer_table_activity")
        self.assertIsNotNone(audit[0]["activity_start"])


if __name__ == "__main__":
    unittest.main()
