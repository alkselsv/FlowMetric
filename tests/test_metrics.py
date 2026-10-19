import unittest

from metrics import run_benchmark, score_dense_reid, score_events, score_states


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
        self.assertFalse(result["reid"]["standard_metrics_available"])
        self.assertIsNone(result["reid"]["idf1"])
        self.assertIn("idf1", result["reid"]["proxy"])

    def test_dense_reid_reports_merges_fragmentation_switches_and_idf1(self) -> None:
        box_a = [0, 0, 20, 40]
        box_b = [100, 0, 120, 40]
        gt = {
            "staff": [2],
            "trajectories": {
                "1": {"frames": [0, 1, 2, 3], "boxes": [box_a] * 4},
                "2": {"frames": [0, 1, 2, 3], "boxes": [box_b] * 4},
            },
        }
        tracks = {
            "fps": 1.0,
            "tracks": {
                "p1": {
                    "frames": [0, 1, 2, 3],
                    "boxes": [box_a, box_a, box_b, box_b],
                },
                "p2": {"frames": [2, 3], "boxes": [box_a, box_a]},
                "p3": {"frames": [0, 1], "boxes": [box_b, box_b]},
            },
        }

        result = score_dense_reid(gt, tracks)

        self.assertTrue(result["standard_metrics_available"])
        self.assertEqual(result["idf1"], 0.5)
        self.assertGreater(result["hota"], 0.0)
        self.assertLess(result["hota"], 1.0)
        self.assertEqual(result["false_merges"]["count"], 1)
        self.assertEqual(result["false_merges"]["predicted_id_to_gt"]["p1"], [1, 2])
        self.assertEqual(result["fragmentation"]["total"], 2)
        self.assertEqual(result["id_switches"]["count"], 2)
        self.assertEqual(result["staff"]["fragments"], 1)
        self.assertEqual(result["staff"]["id_switches"], 1)


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
