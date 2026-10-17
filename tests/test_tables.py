from __future__ import annotations

import unittest
from pathlib import Path

from tables import _containment, detect_tables


class TableGeometryTests(unittest.TestCase):
    def test_containment_is_intersection_over_smaller_box(self) -> None:
        self.assertEqual(_containment([0, 0, 10, 10], [2, 2, 8, 8]), 1.0)
        self.assertEqual(_containment([0, 0, 2, 2], [3, 3, 4, 4]), 0.0)

    def test_sampling_step_must_be_positive_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "step must be positive"):
            detect_tables(Path("missing.mp4"), step=0)


if __name__ == "__main__":
    unittest.main()
