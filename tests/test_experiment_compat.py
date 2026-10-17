from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from experiments.astra_variants import load_production_cache
from experiments.run_experiments import _atomic_write_csv
from experiments.run_experiments import _prepare_experiment_astra
from track import LocalPass


class ExperimentCacheCompatibilityTests(unittest.TestCase):
    def test_loads_legacy_and_versioned_production_cache(self) -> None:
        batches = [{"people": [], "notes": "", "_meta": {"video_frames": [0]}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text(json.dumps(batches), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Legacy Astra cache"):
                load_production_cache(path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(
                    load_production_cache(path, allow_legacy=True), (batches, {})
                )
            self.assertEqual(len(caught), 1)

            manifest = {"astra": {"actual_models": ["model"], "sample_fps": 1.0}}
            path.write_text(
                json.dumps({"cache_version": 1, "manifest": manifest, "reids": batches}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_production_cache(path, expected_sample_fps=1.0),
                (batches, manifest),
            )

    def test_rejects_unknown_cache_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported Astra cache format"):
                load_production_cache(path)

    def test_rejects_wrong_version_and_sampling_rate(self) -> None:
        batches = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text(
                json.dumps({"cache_version": 999, "manifest": {}, "reids": batches}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "version 999"):
                load_production_cache(path)

            manifest = {"astra": {"sample_fps": 2.0}}
            path.write_text(
                json.dumps({"cache_version": 1, "manifest": manifest, "reids": batches}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample_fps"):
                load_production_cache(path, expected_sample_fps=1.0)


class ExperimentReportWritingTests(unittest.TestCase):
    def test_empty_result_does_not_replace_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.csv"
            path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no result rows"):
                _atomic_write_csv(path, [])

            self.assertEqual(path.read_text(encoding="utf-8"), "existing\n")

    def test_csv_output_uses_stable_lf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.csv"

            _atomic_write_csv(path, [{"a": 1, "b": 2}])

            self.assertEqual(path.read_bytes(), b"a,b\n1,2\n")


class ExperimentAstraProtocolTests(unittest.TestCase):
    def test_reconstructs_and_checks_sample_batches(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=50, total=21)
        artifact = {
            "meta": {"sample_fps": 1.0, "frames": 3, "calls": 1},
            "reids": [{"_meta": {"video_frames": [0, 10, 20]}}],
        }

        _prepare_experiment_astra(lp, artifact)

        self.assertEqual(lp.sample_idx, [0, 10, 20])

    def test_rejects_empty_or_incomplete_cached_batches(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=50, total=21)
        artifact = {
            "meta": {"sample_fps": 1.0, "frames": 3, "calls": 0},
            "reids": [],
        }

        with self.assertRaisesRegex(ValueError, "expected 1"):
            _prepare_experiment_astra(lp, artifact)


if __name__ == "__main__":
    unittest.main()
