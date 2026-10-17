from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from scene import SceneConfigError, load_scene_config


class SceneConfigTests(unittest.TestCase):
    def write_json(self, root: Path, name: str, value: object) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_loads_and_merges_valid_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = self.write_json(root, "camera.json", {
                "resolution": [100, 50],
                "door": [[0, 0], [10, 0], [10, 10]],
                "edge_margin": 0.1,
            })
            tables = self.write_json(root, "tables.json", {
                "source": "clip.mp4",
                "resolution": [100, 50],
                "tables": [{"box": [20, 10, 40, 30], "support": 0.8}],
            })
            config = load_scene_config(
                camera, width=100, height=50, source=Path("clip.mp4"), tables_path=tables,
            )
            self.assertEqual(config["tables"][0]["box"], [20, 10, 40, 30])

    def test_resolution_mismatch_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = self.write_json(root, "camera.json", {
                "resolution": [1920, 1080],
                "door": [[0, 0], [10, 0], [10, 10]],
            })
            with self.assertRaisesRegex(SceneConfigError, "video is 1280x720"):
                load_scene_config(camera, width=1280, height=720, source=Path("clip.mp4"))

    def test_changed_table_source_warns_but_allows_same_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = self.write_json(root, "camera.json", {"resolution": [100, 50]})
            tables = self.write_json(root, "tables.json", {
                "source": "calibration.mp4",
                "resolution": [100, 50],
                "tables": [],
            })
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_scene_config(
                    camera, width=100, height=50, source=Path("new.mp4"), tables_path=tables,
                )
            self.assertEqual(len(caught), 1)
            self.assertIn("camera view is unchanged", str(caught[0].message))

    def test_embedded_tables_are_validated_without_separate_table_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = self.write_json(root, "camera.json", {
                "resolution": [100, 50],
                "tables": [{"box": [0, 0, 999, 999]}],
            })

            with self.assertRaisesRegex(SceneConfigError, "outside"):
                load_scene_config(camera, width=100, height=50, source=Path("clip.mp4"))

    def test_zero_area_polygon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = self.write_json(root, "camera.json", {
                "resolution": [100, 50],
                "door": [[0, 0], [10, 10], [20, 20]],
            })

            with self.assertRaisesRegex(SceneConfigError, "non-zero area"):
                load_scene_config(camera, width=100, height=50, source=Path("clip.mp4"))


if __name__ == "__main__":
    unittest.main()
