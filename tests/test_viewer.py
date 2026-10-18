import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from viewer.server import _bind_server, _homography, build_session, select_best_run


class ViewerSessionTest(unittest.TestCase):
    def test_best_run_prefers_identity_and_event_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, continuity, event_f1 in (("local", 0.6, 0.3), ("hybrid", 0.91, 0.88)):
                run = root / name
                run.mkdir()
                for artifact in ("tracks.json", "events.json", "tracked.mp4"):
                    (run / artifact).write_text("{}", encoding="utf-8")
                report = {
                    "identity": {"continuity_mapped": continuity, "continuity_staff_mapped": continuity, "continuity_coverage": 1.0},
                    "events_strict": {"f1": event_f1},
                }
                (run / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
            selected = select_best_run(root)
        self.assertEqual(selected.name, "hybrid")

    def test_server_falls_forward_when_preferred_port_is_busy(self):
        handler = Mock()
        fallback = Mock(server_port=8766)
        busy = OSError(48, "Address already in use")
        with patch(
            "viewer.server.ThreadingHTTPServer", side_effect=[busy, fallback]
        ) as server_class:
            server = _bind_server("127.0.0.1", 8765, handler)
        self.assertIs(server, fallback)
        self.assertEqual(
            server_class.call_args_list[0].args[0], ("127.0.0.1", 8765)
        )
        self.assertEqual(
            server_class.call_args_list[1].args[0], ("127.0.0.1", 8766)
        )

    def test_homography_maps_calibration_points(self):
        source = [[0, 0], [100, 0], [100, 50], [0, 50]]
        target = [[0, 0], [10, 0], [10, 5], [0, 5]]
        matrix = _homography(source, target)
        for point, expected in zip(source, target):
            result = matrix @ np.asarray([*point, 1.0])
            np.testing.assert_allclose(result[:2] / result[2], expected, atol=1e-8)

    def test_build_session_uses_feet_and_preserves_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = {
                "mode": "local", "fps": 10, "width": 200, "height": 100, "n_frames": 21,
                "tracks": {"7": {"role": "customer", "frames": [0, 10], "boxes": [[10, 10, 30, 50], [30, 10, 50, 60]], "conf": [0.8, 0.9]}}
            }
            events = {
                "roles": {"7": "customer"},
                "events": [
                    {"t": 0.5, "type": "service_start", "staff": "3", "customer": "7"},
                    {"t": 1.0, "type": "service_end", "staff": "3", "customer": "7"},
                ],
                "states": [],
            }
            (root / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
            (root / "events.json").write_text(json.dumps(events), encoding="utf-8")
            session = build_session(root / "tracks.json", root / "events.json")
        self.assertEqual(session["meta"]["duration"], 2.0)
        self.assertEqual(session["tracks"][0]["points"], [[0.0, 0.1, 0.5, 0.8], [1.0, 0.2, 0.6, 0.9]])
        self.assertEqual(session["events"], events["events"])
        self.assertEqual(
            session["service_intervals"],
            [{"staff": "3", "customer": "7", "start": 0.5, "end": 1.0, "source": "point_events"}],
        )

    def test_build_session_rejects_mismatched_track_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = {"fps": 10, "width": 100, "height": 100, "n_frames": 10, "tracks": {"1": {"frames": [0], "boxes": []}}}
            (root / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
            (root / "events.json").write_text(json.dumps({"events": [], "states": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different frame and box counts"):
                build_session(root / "tracks.json", root / "events.json")

    def test_build_session_exposes_compact_quality_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = {"fps": 10, "width": 100, "height": 100, "n_frames": 2, "tracks": {}}
            events = {"events": [], "states": []}
            report = {
                "identity": {"continuity_mean": 0.86, "continuity_mapped": 0.91},
                "roles": {"accuracy": 0.9, "accuracy_mapped": 0.93},
                "events_strict": {"f1": 0.89},
                "states": {"waiting": {"tiou": 0.75}},
                "mapping": {"large": "field is deliberately not exposed"},
            }
            (root / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
            (root / "events.json").write_text(json.dumps(events), encoding="utf-8")
            (root / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
            session = build_session(root / "tracks.json", root / "events.json")
        self.assertEqual(
            session["quality"],
            {
                "identity_continuity": 0.86,
                "role_accuracy": 0.9,
                "event_f1": 0.89,
                "waiting_tiou": 0.75,
            },
        )

    def test_build_session_attaches_table_to_customer_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = {
                "fps": 10, "width": 100, "height": 100, "n_frames": 11,
                "tracks": {"7": {"role": "customer", "frames": [0, 5, 10], "boxes": [[70, 20, 90, 80], [40, 20, 60, 80], [40, 20, 60, 80]]}},
            }
            events = {
                "roles": {"7": "customer"},
                "events": [
                    {"t": 0.0, "type": "enter", "person": "7"},
                    {"t": 0.0, "type": "seated", "person": "7"},
                ],
                "states": [{"type": "consuming", "person": "7", "start": 0.1, "end": 0.5}],
            }
            tables = {"tables": [{"id": "A", "box": [45, 50, 55, 60]}, {"id": "B", "box": [75, 50, 85, 60]}]}
            for name, value in (("tracks.json", tracks), ("events.json", events), ("tables.json", tables)):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            session = build_session(
                root / "tracks.json",
                root / "events.json",
                tables_path=root / "tables.json",
                table_vacancy_grace_seconds=0.2,
            )
        self.assertNotIn("table", session["events"][0])
        self.assertEqual(session["events"][1]["table"], "A")
        self.assertEqual(session["customer_tables"], {"7": "A"})
        self.assertEqual(
            session["table_occupancy"],
            [{
                "table": "A", "start": 0.1, "end": 0.7, "customers": ["7"],
                "left_censored": False, "right_censored": False,
            }],
        )


if __name__ == "__main__":
    unittest.main()
