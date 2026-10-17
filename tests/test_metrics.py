import unittest

from metrics import run_benchmark, score_events, score_states


class StateMetricsTests(unittest.TestCase):
    def test_overlapping_intervals_are_counted_once(self) -> None:
        gt = {
            "states": [
                {"type": "waiting", "person": 10, "start": 0.0, "end": 7.0},
                {"type": "waiting", "person": 10, "start": 3.0, "end": 10.0},
            ]
        }
        predicted = [
            {"type": "waiting", "person": "known", "start": 0.0, "end": 7.0},
            {"type": "waiting", "person": "known", "start": 3.0, "end": 10.0},
        ]

        result = score_states(gt, predicted, {"known": 10})

        self.assertEqual(result["waiting"]["tiou"], 1.0)
        self.assertGreaterEqual(result["waiting"]["tiou"], 0.0)
        self.assertLessEqual(result["waiting"]["tiou"], 1.0)

    def test_unmapped_predicted_state_penalizes_union(self) -> None:
        gt = {
            "states": [
                {"type": "waiting", "person": 10, "start": 0.0, "end": 10.0},
            ]
        }
        predicted = [
            {"type": "waiting", "person": "known", "start": 0.0, "end": 10.0},
            {"type": "waiting", "person": "unknown", "start": 0.0, "end": 10.0},
        ]

        result = score_states(gt, predicted, {"known": 10})["waiting"]

        self.assertEqual(result["tiou"], 0.5)
        self.assertEqual(result["pred_people_mapped"], 1)
        self.assertEqual(result["pred_people_unmapped"], 1)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_non_contiguous_ids_and_unmapped_people_are_end_to_end_errors(self) -> None:
        gt = {
            "duration_s": 1.0,
            "staff": [30, 90],
            "anchors": {
                "note": "sparse labels",
                "0": {
                    "10": [10, 10],
                    "30": [110, 10],
                    "90": [210, 10],
                },
            },
            "events": [],
            "states": [],
        }
        frames = list(range(10))
        tracks = {
            "fps": 10.0,
            "tracks": {
                "track-a": {
                    "frames": frames,
                    "boxes": [[0, 0, 20, 40] for _ in frames],
                },
                "track-b": {
                    "frames": frames,
                    "boxes": [[100, 0, 120, 40] for _ in frames],
                },
            },
        }
        events = {
            "roles": {
                "track-a": "customer",
                "track-b": "staff",
                "orphan-staff-track": "staff",
            },
            "events": [],
            "states": [],
        }

        result = run_benchmark(tracks, events, gt)

        self.assertEqual(result["anchoring"]["gt_people"], 3)
        self.assertEqual(result["anchoring"]["mapped"], 2)
        self.assertEqual(result["anchoring"]["unmapped"], [90])
        self.assertEqual(result["identity"]["continuity_mean"], 0.667)
        self.assertEqual(result["identity"]["continuity_mapped"], 1.0)
        self.assertEqual(result["identity"]["continuity_coverage"], 0.667)
        self.assertEqual(result["roles"]["accuracy"], 0.667)
        self.assertEqual(result["roles"]["accuracy_conditional"], 1.0)
        self.assertEqual(result["roles"]["staff_tp"], 1)
        self.assertEqual(result["roles"]["staff_fp"], 1)
        self.assertEqual(result["roles"]["staff_fn"], 1)


class EventMetricsTests(unittest.TestCase):
    def test_score_events_regression(self) -> None:
        gt = [
            {"type": "enter", "person": 10, "t": 1.0},
            {"type": "service_start", "staff": 30, "customer": 10, "t": 5.0},
        ]
        predicted = [
            {"type": "enter", "person": "p10", "t": 1.2},
            {
                "type": "service_start",
                "staff": "p30",
                "customer": "p10",
                "t": 4.9,
            },
            {"type": "enter", "person": "p10", "t": 8.0},
        ]

        result = score_events(
            gt,
            predicted,
            {"p10": 10, "p30": 30},
            tol=0.5,
            strict=True,
        )

        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["precision"], 0.667)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
