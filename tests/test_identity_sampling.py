from __future__ import annotations

import unittest

import numpy as np

from flowmetric.identity.sampling import (
    select_adaptive_reid_frames,
    select_targeted_resample_frames,
)
from flowmetric.tracking import LocalPass


class AdaptiveReIDSamplingTests(unittest.TestCase):
    def test_rare_tracklet_is_covered_with_fixed_frame_budget(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=100)
        box = np.asarray([10, 10, 40, 90], float)
        lp.dets = {
            frame: [(1, box.copy(), 0.9)]
            for frame in range(100)
        }
        lp.dets[37].append((2, box.copy(), 0.8))

        selected = select_adaptive_reid_frames(lp, count=5)

        self.assertEqual(len(selected), 5)
        self.assertIn(37, selected)
        self.assertEqual(selected, sorted(selected))

    def test_sampling_is_deterministic_and_validates_budget(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=5)
        box = np.asarray([10, 10, 40, 90], float)
        lp.dets = {frame: [(frame, box.copy(), 0.9)] for frame in range(5)}

        self.assertEqual(select_adaptive_reid_frames(lp, count=8), list(range(5)))
        with self.assertRaisesRegex(ValueError, "count"):
            select_adaptive_reid_frames(lp, count=0)

    def test_targeted_resampling_adds_only_query_frames_until_supported(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=10)
        box = np.asarray([10, 10, 50, 90], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(7, box.copy(), 0.4)],
            2: [(7, box.copy(), 0.95)],
            5: [(2, box.copy(), 0.99)],
            8: [(7, box.copy(), 0.9)],
            9: [(7, box.copy(), 0.6)],
        }

        selected = select_targeted_resample_frames(
            lp,
            query_tracklet_ids=[7],
            base_frames=[1],
            additional_count=4,
            confirmations=3,
        )

        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected) <= {2, 8, 9})
        self.assertNotIn(5, selected)
        self.assertIn(2, selected)

    def test_targeted_resampling_validates_queries_and_budget(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=2)
        lp.dets = {0: [(1, np.asarray([10, 10, 40, 90], float), 0.9)]}

        with self.assertRaisesRegex(ValueError, "additional_count"):
            select_targeted_resample_frames(
                lp,
                query_tracklet_ids=[1],
                base_frames=[],
                additional_count=-1,
            )
        with self.assertRaisesRegex(ValueError, "absent"):
            select_targeted_resample_frames(
                lp,
                query_tracklet_ids=[2],
                base_frames=[],
            )


if __name__ == "__main__":
    unittest.main()
