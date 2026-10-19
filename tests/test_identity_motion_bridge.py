from __future__ import annotations

import unittest

import numpy as np

from flowmetric.identity.motion_bridge import propose_motion_bridges
from flowmetric.identity.sync import sync_ids
from flowmetric.tracking import LocalPass


def box(center_x: float, *, height: float = 40.0) -> np.ndarray:
    return np.asarray([center_x - 10, 0, center_x + 10, height], dtype=float)


def samples(vector: tuple[float, float]) -> list[tuple[int, np.ndarray, float]]:
    return [
        (0, np.asarray(vector, dtype=float), 0.9),
        (1, np.asarray(vector, dtype=float), 0.8),
    ]


class MotionBridgeTests(unittest.TestCase):
    def test_bidirectional_motion_and_appearance_propose_bridge(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=30)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            2: [(1, box(24), 0.9)],
            8: [(2, box(36), 0.9)],
            9: [(2, box(38), 0.9)],
            10: [(2, box(40), 0.9)],
        }
        lp.appearance_embeddings = {1: samples((1.0, 0.0)), 2: samples((0.99, 0.01))}

        proposals = propose_motion_bridges(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertGreater(proposals[0].appearance_similarity, 0.99)

    def test_rejects_motion_match_without_appearance_confirmation(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=30)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            8: [(2, box(36), 0.9)],
            9: [(2, box(38), 0.9)],
        }
        lp.appearance_embeddings = {1: samples((1.0, 0.0)), 2: samples((0.0, 1.0))}

        self.assertEqual(propose_motion_bridges(lp), [])

    def test_sync_shadow_records_candidate_without_merging(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=30)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            8: [(2, box(36), 0.9)],
            9: [(2, box(38), 0.9)],
        }
        lp.appearance_embeddings = {1: samples((1.0, 0.0)), 2: samples((0.99, 0.01))}

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            motion_bridge_shadow=True,
        )

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertTrue(stats["motion_bridge_shadow_enabled"])
        self.assertEqual(stats["motion_bridge_candidates"], 1)
        self.assertEqual(stats["motion_bridge_decisions"][0]["decision"], "candidate")


if __name__ == "__main__":
    unittest.main()
