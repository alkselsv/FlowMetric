import unittest

from events import extract_events


FPS = 10.0


def _boxes(frames, x, *, customer=False):
    if callable(x):
        centers = [x(frame) for frame in frames]
    else:
        centers = [x] * len(frames)
    if customer:
        # Customer head point (12% down the box) is y=100.
        return [[cx - 20, 88, cx + 20, 188] for cx in centers]
    # Staff box center is y=100.
    return [[cx - 20, 50, cx + 20, 150] for cx in centers]


def _track(frames, x, role, *, customer=False):
    frames = list(frames)
    return {
        "role": role,
        "frames": frames,
        "boxes": _boxes(frames, x, customer=customer),
    }


def _data(n_frames, tracks, *, fps=FPS):
    return {
        "fps": fps,
        "width": 640,
        "height": 480,
        "n_frames": n_frames,
        "tracks": tracks,
    }


class EventExtractionTests(unittest.TestCase):
    def test_sparse_terminal_observations_do_not_emit_exit(self):
        sparse_tail = [262, 263, 265, 269, 270, 271, 272, 273, 274, 275, 276, 284, 308, 317, 318]
        data = _data(
            361,
            {"person": _track(sparse_tail, 20, "staff")},
            fps=24.0,
        )

        result = extract_events(data, {})

        self.assertFalse(any(e["type"] == "exit" for e in result["events"]))

    def test_dense_terminal_observations_emit_exit(self):
        data = _data(
            361,
            {"person": _track(range(300, 319), 20, "staff")},
            fps=24.0,
        )

        result = extract_events(data, {})
        exits = [e for e in result["events"] if e["type"] == "exit"]

        self.assertEqual([(e["person"], e["t"]) for e in exits], [("person", 13.25)])

    def test_customer_switch_splits_service_and_does_not_bridge_between_people(self):
        frames = range(40)
        data = _data(
            40,
            {
                "staff": _track(range(1, 36), 100, "staff"),
                "a": _track(
                    frames,
                    lambda frame: 100 if frame <= 15 else 400,
                    "customer",
                    customer=True,
                ),
                "b": _track(
                    frames,
                    lambda frame: 100 if frame >= 17 else 400,
                    "customer",
                    customer=True,
                ),
            },
        )

        result = extract_events(data, {})
        service = [event for event in result["events"] if event["type"].startswith("service_")]

        self.assertEqual(
            [(e["type"], e["customer"], e["t"]) for e in service],
            [
                ("service_start", "a", 0.1),
                ("service_end", "a", 1.5),
                ("service_start", "b", 1.7),
                ("service_end", "b", 3.5),
            ],
        )

    def test_sparse_observation_gap_does_not_form_one_long_service(self):
        staff_frames = [*range(1, 7), *range(20, 26)]
        data = _data(
            30,
            {
                "staff": _track(staff_frames, 100, "staff"),
                "customer": _track(range(30), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertFalse(any(e["type"].startswith("service_") for e in result["events"]))

    def test_short_gap_is_bridged_for_the_same_customer(self):
        data = _data(
            30,
            {
                "staff": _track(range(1, 26), 100, "staff"),
                "customer": _track(
                    range(30),
                    lambda frame: 400 if frame == 12 else 100,
                    "customer",
                    customer=True,
                ),
            },
        )

        result = extract_events(data, {})
        service = [event for event in result["events"] if event["type"].startswith("service_")]

        self.assertEqual(
            [(e["type"], e["customer"], e["t"]) for e in service],
            [("service_start", "customer", 0.1), ("service_end", "customer", 2.5)],
        )

    def test_sparse_none_observation_is_not_bridged_by_list_length(self):
        data = _data(
            20,
            {
                "staff": _track([1, 6, 11], 100, "staff"),
                "customer": _track(
                    range(20),
                    lambda frame: 400 if frame == 6 else 100,
                    "customer",
                    customer=True,
                ),
            },
        )

        result = extract_events(data, {})

        self.assertFalse(any(e["type"].startswith("service_") for e in result["events"]))

    def test_mid_video_customer_is_not_served_until_a_seated_transition(self):
        data = _data(
            40,
            {
                "staff": _track(range(1, 36), 100, "staff"),
                "customer": _track(range(5, 36), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertFalse(any(e["type"].startswith("service_") for e in result["events"]))

    def test_customer_born_in_kitchen_does_not_emit_leave_kitchen(self):
        kitchen = [[50, 140], [250, 140], [250, 210], [50, 210]]
        data = _data(
            30,
            {
                "customer": _track(range(5, 21), 100, "customer", customer=True),
                "staff": _track(range(5, 21), 220, "staff"),
            },
        )

        result = extract_events(data, {"kitchen": kitchen})
        leaves = [e["person"] for e in result["events"] if e["type"] == "leave_kitchen"]

        self.assertEqual(leaves, ["staff"])

    def test_customer_without_home_table_is_not_eligible_for_service(self):
        data = _data(
            30,
            {
                "staff": _track(range(1, 26), 100, "staff"),
                "customer": _track(range(30), 100, "customer", customer=True),
            },
        )
        camera = {"tables": [[300, 300, 400, 400]]}

        result = extract_events(data, camera)

        self.assertFalse(any(e["type"].startswith("service_") for e in result["events"]))


if __name__ == "__main__":
    unittest.main()
