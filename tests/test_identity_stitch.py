from __future__ import annotations

import unittest

import numpy as np

from flowmetric.identity.stitch import propose_tracklet_stitches
from track import LocalPass, sync_ids


def box(center_x: float, *, height: float = 40.0) -> np.ndarray:
    return np.asarray([center_x - 10, 0, center_x + 10, height], dtype=float)


class LocalTrackletStitchTests(unittest.TestCase):
    def test_proposes_mutual_unambiguous_short_gap(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=8)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            2: [(1, box(24), 0.9)],
            4: [(2, box(28), 0.9)],
            5: [(2, box(30), 0.9)],
        }

        proposals = propose_tracklet_stitches(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(
            (proposals[0].earlier_id, proposals[0].later_id), (1, 2)
        )

    def test_rejects_ambiguous_successors(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=8)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(20), 0.9)],
            3: [(2, box(21), 0.9), (3, box(19), 0.9)],
            4: [(2, box(21), 0.9), (3, box(19), 0.9)],
        }

        self.assertEqual(propose_tracklet_stitches(lp), [])

    def test_sync_can_disable_or_apply_local_stitch(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=8)
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            2: [(1, box(24), 0.9)],
            4: [(2, box(28), 0.9)],
            5: [(2, box(30), 0.9)],
        }

        separate, _, baseline_stats = sync_ids(
            lp, [], iou_thr=0.3, local_stitch=False
        )
        stitched, _, stitched_stats = sync_ids(
            lp, [], iou_thr=0.3, local_stitch=True
        )

        self.assertNotEqual(separate[1], separate[2])
        self.assertEqual(stitched[1], stitched[2])
        self.assertEqual(baseline_stats["local_stitch_merges"], 0)
        self.assertEqual(stitched_stats["local_stitch_merges"], 1)
        self.assertEqual(
            stitched_stats["track_identity_status_counts"],
            {"local_stitched": 2},
        )

    def test_existing_astra_conflict_blocks_local_stitch(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=8)
        lp.sample_idx = [1, 4]
        lp.dets = {
            0: [(1, box(20), 0.9)],
            1: [(1, box(22), 0.9)],
            2: [(1, box(24), 0.9)],
            4: [(2, box(28), 0.9)],
            5: [(2, box(30), 0.9)],
        }
        response = {
            "people": [
                {
                    "id": 10,
                    "role": "customer",
                    "detections": [{"frame_index": 0, "xyxy": box(22).tolist()}],
                },
                {
                    "id": 11,
                    "role": "customer",
                    "detections": [{"frame_index": 1, "xyxy": box(28).tolist()}],
                },
            ],
            "notes": "different identities in one batch",
            "_meta": {
                "coordinate_space": "original",
                "scale": 1.0,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "width": 200,
                "height": 100,
                "resized_width": 200,
                "resized_height": 100,
                "original_width": 200,
                "original_height": 100,
                "n_frames": 2,
                "video_frames": [1, 4],
                "requested_model": "test",
                "actual_model": "test",
            },
        }

        mapping, _, stats = sync_ids(
            lp, [response], iou_thr=0.3, min_support=1, local_stitch=True
        )

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertEqual(stats["local_stitch_merges"], 0)
        self.assertEqual(
            stats["local_stitch_decisions"][0]["reason"], "astra_conflict"
        )


if __name__ == "__main__":
    unittest.main()
