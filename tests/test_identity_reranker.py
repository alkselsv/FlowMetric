from __future__ import annotations

import unittest

import numpy as np

from flowmetric.identity.reranker import FEATURE_NAMES, run_context_reranker_shadow
from flowmetric.identity.sync import sync_ids
from flowmetric.tracking import LocalPass


class ContextRerankerTests(unittest.TestCase):
    def test_trains_on_astra_components_and_scores_only_residual_pairs(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=50)
        cluster_by: dict[int, str] = {}
        astra: dict[str, dict[int, int]] = {}
        for identity in range(10):
            cluster = f"A{identity}"
            astra[cluster] = {0: identity}
            angle = identity * 0.12
            embedding = np.asarray([np.cos(angle), np.sin(angle)])
            for offset in range(2):
                tracklet_id = identity * 2 + offset + 1
                frame = identity * 4 + offset * 2
                box = np.asarray([20 + identity, 10, 60 + identity, 110], float)
                lp.dets[frame] = [(tracklet_id, box, 0.9)]
                lp.appearance_embeddings[tracklet_id] = [(frame, embedding, 0.9)]
                cluster_by[tracklet_id] = cluster

        for tracklet_id, frame in ((100, 41), (101, 43)):
            box = np.asarray([30, 10, 70, 110], float)
            lp.dets[frame] = [(tracklet_id, box, 0.9)]
            lp.appearance_embeddings[tracklet_id] = [
                (frame, np.asarray([1.0, 0.0]), 0.9)
            ]
            cluster_by[tracklet_id] = f"T{tracklet_id}"

        result = run_context_reranker_shadow(
            lp,
            training_cluster_by_tracklet=cluster_by,
            current_cluster_by_tracklet=cluster_by,
            astra_by_cluster=astra,
            cannot_link=set(),
            role_by_tracklet={},
            scene_config={},
        )

        self.assertTrue(result["trained"])
        self.assertEqual(result["positive_pairs"], 10)
        self.assertGreaterEqual(result["negative_pairs"], 40)
        self.assertIsNotNone(result["cross_validated_auc"])
        self.assertEqual(
            set(result["standardized_coefficients"]), set(FEATURE_NAMES)
        )
        self.assertEqual(result["feature_names"], list(FEATURE_NAMES))
        self.assertGreater(len(result["candidates"]), 0)
        self.assertTrue(
            all(item["decision"].startswith("shadow_") for item in result["candidates"])
        )

    def test_sync_shadow_mode_never_changes_mapping(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=4)
        box = np.asarray([10, 10, 40, 90], float)
        lp.dets = {0: [(1, box, 0.9)], 3: [(2, box, 0.9)]}
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([1.0, 0.0]), 0.9)],
        }

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=False,
            context_reranker_shadow=True,
            scene_config={},
        )

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertTrue(stats["context_reranker_shadow_enabled"])
        self.assertFalse(stats["context_reranker"]["trained"])


if __name__ == "__main__":
    unittest.main()
