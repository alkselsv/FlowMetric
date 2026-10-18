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


def _approach_stop_depart(frame, *, stop_from=10, depart_from=30):
    if frame < stop_from:
        return 250 - 10 * frame
    if frame < depart_from:
        return 150
    return 150 + 10 * (frame - depart_from)


def _service_events(result):
    return [event for event in result["events"] if event["type"].startswith("service_")]


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

    def test_competing_staff_and_customers_are_mutually_exclusive(self):
        frames = range(50)
        data = _data(
            50,
            {
                "staff-a": _track(frames, _approach_stop_depart, "staff"),
                "staff-b": _track(
                    frames,
                    lambda frame: -70 + 11 * frame if frame < 10 else 40 if frame < 30 else 40 - 10 * (frame - 30),
                    "staff",
                ),
                "customer": _track(frames, 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})
        service = _service_events(result)

        self.assertEqual(
            [(e["type"], e["staff"], e["customer"]) for e in service],
            [
                ("service_start", "staff-a", "customer"),
                ("service_end", "staff-a", "customer"),
            ],
        )

        one_staff = _data(
            50,
            {
                "staff": _track(frames, _approach_stop_depart, "staff"),
                "customer-a": _track(frames, 100, "customer", customer=True),
                "customer-b": _track(frames, 100, "customer", customer=True),
            },
        )
        one_staff_result = extract_events(one_staff, {})
        self.assertEqual(
            {event["customer"] for event in _service_events(one_staff_result)},
            {"customer-a"},
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
            60,
            {
                "staff": _track(
                    range(60),
                    lambda frame: _approach_stop_depart(frame, depart_from=35),
                    "staff",
                ),
                "customer": _track(
                    range(60),
                    lambda frame: 400 if frame == 27 else 100,
                    "customer",
                    customer=True,
                ),
            },
        )

        result = extract_events(data, {})
        service = _service_events(result)

        self.assertEqual(
            [(e["type"], e["customer"], e["t"]) for e in service],
            [("service_start", "customer", 1.3), ("service_end", "customer", 3.5)],
        )

    def test_long_observation_gap_right_censors_and_splits_late_service(self):
        first_segment = range(31)
        second_segment = range(50, 111)

        def two_services(frame):
            if frame <= 30:
                return _approach_stop_depart(frame, depart_from=100)
            local = frame - 50
            return _approach_stop_depart(local, stop_from=15, depart_from=40)

        data = _data(
            115,
            {
                "staff": _track([*first_segment, *second_segment], two_services, "staff"),
                "customer": _track(range(115), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})
        intervals = result["service_intervals"]

        self.assertEqual(len(intervals), 2)
        self.assertEqual(
            (intervals[0]["start"], intervals[0]["end"], intervals[0]["right_censored"]),
            (1.3, None, True),
        )
        self.assertEqual(
            (intervals[1]["right_censored"], intervals[1]["left_censored"]),
            (False, False),
        )
        self.assertGreater(intervals[1]["start"], 5.0)
        self.assertIsNotNone(intervals[1]["end"])

    def test_table_release_hysteresis_keeps_one_service_episode(self):
        def approach_table_then_depart(frame):
            if frame < 10:
                return 260 - 10 * frame
            if frame < 35:
                return 160
            return 160 + 10 * (frame - 35)

        staff = _track(range(55), approach_table_then_depart, "staff")
        # With a constant centre and height, narrow the box for two frames so
        # its table gap is 0.36: outside TABLE_GAP (0.35), but still inside
        # TABLE_RELEASE_GAP (0.55).  The short excursion must not split service.
        for frame in (22, 23):
            staff["boxes"][frame] = [156, 50, 164, 150]

        data = _data(
            55,
            {
                "staff": staff,
                "customer": _track(range(55), 100, "customer", customer=True),
            },
        )
        camera = {"tables": [[80, 130, 120, 180]]}

        result = extract_events(data, camera)

        self.assertEqual(len(result["service_intervals"]), 1)
        self.assertEqual(
            [event["type"] for event in _service_events(result)],
            ["service_start", "service_end"],
        )
        self.assertFalse(result["service_intervals"][0]["left_censored"])
        self.assertFalse(result["service_intervals"][0]["right_censored"])

    def test_static_near_without_approach_does_not_start_service(self):
        data = _data(
            40,
            {
                "staff": _track(range(5, 36), 150, "staff"),
                "customer": _track(range(40), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(_service_events(result), [])
        self.assertEqual(result["service_intervals"], [])

    def test_customer_approaching_static_staff_is_not_service(self):
        data = _data(
            50,
            {
                "staff": _track(range(50), 150, "staff"),
                "customer": _track(
                    range(50),
                    lambda frame: 300 - 20 * frame if frame < 10 else 100,
                    "customer",
                    customer=True,
                ),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(_service_events(result), [])
        self.assertEqual(result["service_intervals"], [])

    def test_approach_stop_depart_emits_one_service_pair(self):
        data = _data(
            50,
            {
                "staff": _track(range(50), _approach_stop_depart, "staff"),
                "customer": _track(range(50), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})
        service = _service_events(result)

        self.assertEqual(
            [(event["type"], event["t"]) for event in service],
            [("service_start", 1.3), ("service_end", 3.0)],
        )
        self.assertTrue(all(0.0 <= event["confidence"] <= 1.0 for event in service))

    def test_lingering_motion_is_not_service(self):
        def approach_then_drift(frame):
            if frame < 10:
                return 250 - 10 * frame
            if frame < 16:
                return 150
            if frame < 36:
                return 150 + 1.5 * (frame - 15)
            return 180 + 10 * (frame - 36)

        data = _data(
            60,
            {
                "staff": _track(range(60), approach_then_drift, "staff"),
                "customer": _track(range(60), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(_service_events(result), [])
        self.assertEqual(result["service_intervals"], [])

    def test_cooldown_prevents_restart_until_expanded_zone_exit(self):
        def leaves_then_returns(frame):
            if frame < 10:
                return 250 - 10 * frame
            if frame < 30:
                return 150
            if frame < 35:
                return 150 + 8 * (frame - 30)
            if frame < 40:
                return 182 - 6.4 * (frame - 35)
            return 150

        data = _data(
            60,
            {
                "staff": _track(range(60), leaves_then_returns, "staff"),
                "customer": _track(range(60), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(
            [(event["type"], event.get("right_censored")) for event in _service_events(result)],
            [("service_start", True)],
        )
        self.assertEqual(len(result["service_intervals"]), 1)
        self.assertIsNone(result["service_intervals"][0]["end"])

    def test_cooldown_survives_occlusion_at_pending_end(self):
        def leaves_then_returns(frame):
            if frame < 10:
                return 250 - 10 * frame
            if frame < 30:
                return 150
            if frame < 35:
                return 150 + 8 * (frame - 30)
            if frame < 40:
                return 182 - 6.4 * (frame - 35)
            return 150

        staff_frames = [frame for frame in range(60) if frame != 35]
        data = _data(
            60,
            {
                "staff": _track(staff_frames, leaves_then_returns, "staff"),
                "customer": _track(range(60), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(
            [(event["type"], event.get("right_censored")) for event in _service_events(result)],
            [("service_start", None), ("service_end", None)],
        )
        self.assertEqual(len(result["service_intervals"]), 1)

    def test_disappearing_staff_releases_customer_for_next_staff(self):
        def second_staff(frame):
            if frame < 60:
                return 300 - 10 * (frame - 45)
            if frame < 85:
                return 150
            return 150 + 10 * (frame - 85)

        data = _data(
            115,
            {
                "staff-a": _track(range(39), _approach_stop_depart, "staff"),
                "staff-b": _track(range(45, 111), second_staff, "staff"),
                "customer": _track(range(115), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(
            [(event["type"], event["staff"]) for event in _service_events(result)],
            [
                ("service_start", "staff-a"),
                ("service_end", "staff-a"),
                ("service_start", "staff-b"),
                ("service_end", "staff-b"),
            ],
        )

    def test_tentative_lingering_staff_does_not_block_true_service(self):
        def lingering_staff(frame):
            if frame < 10:
                return 250 - 10 * frame
            if frame < 45:
                return 150 + 1.5 * (frame - 10)
            return 202.5 + 10 * (frame - 45)

        def serving_staff(frame):
            if frame < 15:
                return 300
            if frame < 25:
                return 300 - 15 * (frame - 15)
            if frame < 55:
                return 150
            return 150 + 10 * (frame - 55)

        data = _data(
            80,
            {
                "lingering": _track(range(80), lingering_staff, "staff"),
                "serving": _track(range(80), serving_staff, "staff"),
                "customer": _track(range(80), 100, "customer", customer=True),
            },
        )

        result = extract_events(data, {})

        self.assertEqual(
            {(event["staff"], event["type"]) for event in _service_events(result)},
            {("serving", "service_start"), ("serving", "service_end")},
        )

    def test_clip_boundary_services_are_censored_without_fabricated_events(self):
        def left_boundary_path(frame):
            return 150 if frame < 20 else 150 + 10 * (frame - 20)

        left = extract_events(
            _data(
                40,
                {
                    "staff": _track(range(40), left_boundary_path, "staff"),
                    "customer": _track(range(40), 100, "customer", customer=True),
                },
            ),
            {},
        )
        self.assertEqual(_service_events(left), [])
        self.assertEqual(
            (left["service_intervals"][0]["start"], left["service_intervals"][0]["left_censored"]),
            (None, True),
        )

        right = extract_events(
            _data(
                35,
                {
                    "staff": _track(
                        range(35),
                        lambda frame: 250 - 10 * frame if frame < 10 else 150,
                        "staff",
                    ),
                    "customer": _track(range(35), 100, "customer", customer=True),
                },
            ),
            {},
        )
        self.assertEqual(
            [(event["type"], event.get("right_censored")) for event in _service_events(right)],
            [("service_start", True)],
        )
        self.assertIsNone(right["service_intervals"][0]["end"])

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
