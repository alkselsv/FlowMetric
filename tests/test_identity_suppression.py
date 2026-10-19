from __future__ import annotations

import unittest

import numpy as np

from flowmetric.identity.suppression import (
    apply_tracklet_suppressions,
    propose_nested_tracklet_suppressions,
)
from flowmetric.tracking import LocalPass


class NestedTrackletSuppressionTests(unittest.TestCase):
    def test_suppresses_short_unverified_nested_duplicate(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=8)
        host = np.asarray([20, 5, 100, 115], float)
        nested = np.asarray([45, 15, 75, 55], float)
        lp.dets = {
            frame: [(1, host.copy(), 0.9)]
            + ([(2, nested.copy(), 0.3)] if 2 <= frame < 5 else [])
            for frame in range(8)
        }

        proposals = propose_nested_tracklet_suppressions(
            lp,
            {1: 1, 2: 2},
            {
                "1": {"status": "merged"},
                "2": {"status": "abstained"},
            },
        )

        self.assertEqual([item.tracklet_id for item in proposals], [2])
        self.assertEqual(proposals[0].host_global_id, 1)
        apply_tracklet_suppressions(lp, {2})
        self.assertFalse(any(tid == 2 for dets in lp.dets.values() for tid, _, _ in dets))

    def test_preserves_verified_long_or_side_by_side_tracklets(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=8)
        host = np.asarray([20, 5, 100, 115], float)
        nested = np.asarray([45, 15, 75, 55], float)
        separate = np.asarray([120, 5, 180, 115], float)
        lp.dets = {
            frame: [
                (1, host.copy(), 0.9),
                (2, nested.copy(), 0.3),
                (3, separate.copy(), 0.3),
                (4, nested.copy(), 0.3),
            ]
            for frame in range(8)
        }

        proposals = propose_nested_tracklet_suppressions(
            lp,
            {1: 1, 2: 2, 3: 3, 4: 4},
            {
                "1": {"status": "merged"},
                "2": {"status": "abstained"},
                "3": {"status": "abstained"},
                "4": {"status": "merged"},
            },
        )

        self.assertEqual(proposals, [])

    def test_validates_thresholds(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=1)
        with self.assertRaisesRegex(ValueError, "min_containment"):
            propose_nested_tracklet_suppressions(lp, {}, {}, min_containment=1.1)

    def test_preserves_clip_boundary_tracklets(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=10)
        host = np.asarray([20, 5, 100, 115], float)
        nested = np.asarray([45, 15, 75, 55], float)
        lp.dets = {
            frame: [(1, host.copy(), 0.9)]
            + ([(2, nested.copy(), 0.3)] if frame < 3 else [])
            for frame in range(10)
        }

        proposals = propose_nested_tracklet_suppressions(
            lp,
            {1: 1, 2: 2},
            {"1": {"status": "merged"}, "2": {"status": "abstained"}},
        )

        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
