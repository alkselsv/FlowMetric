from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from flowmetric.identity.appearance import (
    AppearanceStitch,
    TrackerAppearanceEncoder,
    calibrate_appearance_threshold,
    compose_part_embeddings,
    densify_multiview_appearance,
    part_visibility,
    propose_anchored_appearance_stitches,
    propose_appearance_stitches,
    propose_ensemble_appearance_stitches,
    propose_multiview_appearance_stitches,
    propose_pose_attribute_stitches,
    propose_robust_appearance_stitches,
    propose_segmented_attribute_stitches,
    robust_temporal_prototype,
    summarize_robust_appearance,
    select_multiview_resample_tracklets,
)
from flowmetric.identity.graph import build_identity_graph_edges
from flowmetric.identity.stitch import TrackletStitch
from flowmetric.identity.gallery import normalize_gallery_response
from flowmetric.identity.sync import sync_ids
from flowmetric.identity.video_appearance import sample_tracklet_observations
from flowmetric.identity.pose_attributes import (
    encode_pose_attribute_tracklets,
    pose_guided_attribute_descriptor,
)
from flowmetric.service.local import PoseDetection
from flowmetric.identity.segmented_attributes import (
    encode_segmented_attribute_tracklets,
    masked_clothing_descriptor,
)
from flowmetric.tracking import LocalPass
from flowmetric.tracking.local import apply_tracker_appearance_bonus


def _local_pass() -> LocalPass:
    lp = LocalPass(fps=10.0, width=200, height=120, total=8)
    first = np.asarray([20, 10, 60, 110], float)
    second = np.asarray([24, 10, 64, 110], float)
    lp.dets = {
        0: [(1, first.copy(), 0.9)],
        1: [(1, first.copy(), 0.9)],
        3: [(2, second.copy(), 0.9)],
        4: [(2, second.copy(), 0.9)],
    }
    lp.appearance_embeddings = {
        1: [(0, np.asarray([1.0, 0.0, 0.0]), 0.9)],
        2: [(3, np.asarray([0.99, 0.05, 0.0]), 0.9)],
    }
    return lp


class AppearanceStitchTests(unittest.TestCase):
    def test_masked_clothing_descriptor_ignores_background_pixels(self) -> None:
        crop = np.zeros((100, 60, 3), dtype=np.uint8)
        crop[10:55, 15:45] = (0, 0, 255)
        crop[55:95, 15:45] = (255, 0, 0)
        mask = np.zeros((100, 60), dtype=bool)
        mask[10:95, 15:45] = True
        descriptor = masked_clothing_descriptor(crop, mask)
        changed = crop.copy()
        changed[~mask] = (0, 255, 0)
        changed_descriptor = masked_clothing_descriptor(changed, mask)

        self.assertEqual(descriptor.shape, (96,))
        np.testing.assert_allclose(descriptor, changed_descriptor)
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)

    def test_segmented_attribute_encoder_uses_strongest_osnet_samples(self) -> None:
        lp = LocalPass(fps=10.0, width=80, height=120, total=5)
        box = np.asarray([10, 10, 70, 115], float)
        lp.dets = {frame: [(1, box.copy(), 0.9)] for frame in range(5)}
        embedding = np.asarray([1.0, 0.0])
        lp.appearance_embeddings = {
            1: [(frame, embedding, 0.5 + frame / 10) for frame in range(5)]
        }
        frame = np.full((120, 80, 3), 100, dtype=np.uint8)
        attribute = np.ones(96, dtype=np.float32) / np.sqrt(96)
        with (
            patch(
                "flowmetric.identity.segmented_attributes.load_video_frames",
                side_effect=lambda source, indexes: {index: frame for index in indexes},
            ),
            patch(
                "flowmetric.identity.segmented_attributes.segment_person_attributes",
                return_value=(attribute, 0.8),
            ),
        ):
            stats = encode_segmented_attribute_tracklets(Path("video.mp4"), lp)

        self.assertEqual(stats["requested_observations"], 4)
        self.assertEqual(stats["accepted_observations"], 4)
        self.assertEqual(len(lp.segmented_attribute_embeddings[1]), 4)
        self.assertNotIn(0, [item[0] for item in lp.segmented_attribute_embeddings[1]])

    def test_segmented_attribute_consensus_resolves_osnet_ambiguity(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
        }
        source = np.asarray([1.0, 0.0])
        lp.appearance_embeddings = {
            1: [(0, source, 0.9), (1, source, 0.8)],
            2: [(3, np.asarray([0.80, 0.60]), 0.9), (4, np.asarray([0.80, 0.60]), 0.8)],
            3: [(3, np.asarray([0.81, np.sqrt(1 - 0.81**2)]), 0.9), (4, np.asarray([0.81, np.sqrt(1 - 0.81**2)]), 0.8)],
        }
        match, other = np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
        lp.segmented_attribute_embeddings = {
            1: [(0, match, 0.9), (1, match, 0.8)],
            2: [(3, match, 0.9), (4, match, 0.8)],
            3: [(3, other, 0.9), (4, other, 0.8)],
        }

        self.assertEqual(propose_appearance_stitches(lp), [])
        proposals = propose_segmented_attribute_stitches(lp)
        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertEqual(proposals[0].selection, "segmented_attribute_consensus")

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            segmented_attribute_appearance=True,
        )
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["segmented_attribute_consensus_merges"], 1)

    def test_pose_attribute_encoder_matches_existing_osnet_observations(self) -> None:
        lp = LocalPass(fps=10.0, width=80, height=120, total=1)
        box = np.asarray([10, 10, 70, 115], float)
        lp.dets = {0: [(1, box.copy(), 0.9)]}
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.8)]
        }
        frame = np.full((120, 80, 3), 100, dtype=np.uint8)
        keypoints = np.zeros((17, 2), dtype=float)
        keypoints[5], keypoints[6] = (25, 30), (55, 30)
        keypoints[11], keypoints[12] = (28, 65), (52, 65)
        keypoints[13], keypoints[14] = (30, 85), (50, 85)
        keypoints[15], keypoints[16] = (30, 108), (50, 108)
        pose = PoseDetection(
            box=tuple(box),
            keypoints=keypoints,
            keypoint_confidence=np.ones(17, dtype=float),
        )

        with patch(
            "flowmetric.identity.pose_attributes.load_video_frames",
            return_value={0: frame},
        ):
            stats = encode_pose_attribute_tracklets(
                Path("video.mp4"),
                lp,
                backend=lambda images: [[pose] for _ in images],
            )

        self.assertEqual(stats["matched_observations"], 1)
        self.assertEqual(stats["tracklets"], 1)
        self.assertEqual(len(lp.pose_attribute_embeddings[1]), 1)

    def test_pose_guided_attributes_use_clothing_regions_and_body_proportions(self) -> None:
        frame = np.zeros((120, 80, 3), dtype=np.uint8)
        frame[25:65, 15:65] = (0, 0, 255)
        frame[65:110, 15:65] = (255, 0, 0)
        box = np.asarray([10, 10, 70, 115], float)
        keypoints = np.zeros((17, 2), dtype=float)
        keypoints[5], keypoints[6] = (25, 30), (55, 30)
        keypoints[11], keypoints[12] = (28, 65), (52, 65)
        keypoints[13], keypoints[14] = (30, 85), (50, 85)
        keypoints[15], keypoints[16] = (30, 108), (50, 108)
        confidence = np.ones(17, dtype=float)

        descriptor, quality = pose_guided_attribute_descriptor(
            frame, box, keypoints, confidence
        )
        changed = frame.copy()
        changed[25:65, 15:65] = (0, 255, 0)
        other, _ = pose_guided_attribute_descriptor(
            changed, box, keypoints, confidence
        )

        self.assertEqual(descriptor.shape, (100,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)
        self.assertEqual(quality, 1.0)
        self.assertLess(float(np.dot(descriptor, other)), 0.8)

    def test_pose_attribute_consensus_resolves_osnet_ambiguity(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
        }
        source = np.asarray([1.0, 0.0])
        correct = np.asarray([0.80, 0.60])
        distractor = np.asarray([0.81, np.sqrt(1 - 0.81**2)])
        lp.appearance_embeddings = {
            1: [(0, source, 0.9), (1, source, 0.8)],
            2: [(3, correct, 0.9), (4, correct, 0.8)],
            3: [(3, distractor, 0.9), (4, distractor, 0.8)],
        }
        attr_match = np.asarray([1.0, 0.0])
        attr_other = np.asarray([0.0, 1.0])
        lp.pose_attribute_embeddings = {
            1: [(0, attr_match, 0.9), (1, attr_match, 0.8)],
            2: [(3, attr_match, 0.9), (4, attr_match, 0.8)],
            3: [(3, attr_other, 0.9), (4, attr_other, 0.8)],
        }

        self.assertEqual(propose_appearance_stitches(lp), [])
        proposals = propose_pose_attribute_stitches(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertEqual(proposals[0].selection, "pose_attribute_consensus")
        self.assertEqual(proposals[0].support, 2)

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            pose_attribute_appearance=True,
        )
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["pose_attribute_consensus_merges"], 1)

    def test_scene_calibration_lowers_threshold_only_for_separated_pairs(self) -> None:
        lp = LocalPass(fps=10.0, width=100, height=100, total=4)
        positive_left = np.asarray([1.0, 0.0])
        positive_right = np.asarray([0.80, 0.60])
        negative = np.asarray([0.0, 1.0])
        lp.appearance_embeddings = {
            1: [(0, positive_left, 0.9)],
            2: [(2, positive_right, 0.9)],
            3: [(0, negative, 0.9)],
        }

        calibration = calibrate_appearance_threshold(
            lp,
            cluster_by_tracklet={1: "A", 2: "A", 3: "B"},
            verified_clusters={"A", "B"},
            cannot_link={(1, 3)},
            min_positive_pairs=1,
            min_negative_pairs=1,
        )

        self.assertTrue(calibration["applied"])
        self.assertEqual(calibration["threshold"], 0.80)
        self.assertEqual(calibration["reason"], "separated_trusted_pairs")

        fallback = calibrate_appearance_threshold(
            lp,
            cluster_by_tracklet={1: "A", 2: "A", 3: "B"},
            verified_clusters={"A", "B"},
            cannot_link=set(),
        )
        self.assertFalse(fallback["applied"])
        self.assertEqual(fallback["threshold"], 0.82)

    def test_robust_temporal_prototype_rejects_low_quality_view_outlier(self) -> None:
        e1 = np.asarray([1.0, 0.0])
        e2 = np.asarray([0.0, 1.0])
        prototype, support = robust_temporal_prototype(
            [(0, e1, 0.9), (1, e1, 0.8), (2, e2, 0.1)]
        )

        np.testing.assert_allclose(prototype, e1)
        self.assertEqual(support, 2)
        lp = LocalPass(fps=10.0, width=20, height=20, total=3)
        lp.appearance_embeddings = {1: [(0, e1, 0.9), (1, e1, 0.8), (2, e2, 0.1)]}
        self.assertEqual(
            summarize_robust_appearance(lp),
            {"tracklets": 1, "samples": 3, "inliers": 2, "outliers": 1},
        )

    def test_robust_appearance_adds_only_supported_unambiguous_link(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
        }
        e1 = np.asarray([1.0, 0.0])
        outlier = np.asarray([0.0, 1.0])
        distractor = np.asarray([0.64, np.sqrt(1 - 0.64**2)])
        lp.appearance_embeddings = {
            1: [(0, e1, 0.9), (1, e1, 0.8), (2, outlier, 0.1)],
            2: [(3, e1, 0.9), (4, e1, 0.8)],
            3: [(3, distractor, 0.9), (4, distractor, 0.8)],
        }

        self.assertEqual(propose_appearance_stitches(lp), [])
        proposals = propose_robust_appearance_stitches(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertEqual(proposals[0].selection, "robust_temporal_consensus")
        self.assertEqual(proposals[0].support, 2)
        self.assertAlmostEqual(proposals[0].similarity, 1.0)

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            robust_appearance=True,
        )
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["robust_consensus_merges"], 1)

    def test_dense_multiview_resamples_only_unresolved_pair_neighbours(self) -> None:
        lp = LocalPass(fps=10.0, width=80, height=120, total=51)
        box = np.asarray([10, 10, 40, 110], float)
        lp.dets = {
            frame: [(1, box.copy(), 0.9)] for frame in range(3)
        } | {
            frame: [(2, box.copy(), 0.9)] for frame in range(4, 7)
        } | {50: [(3, box.copy(), 0.9)]}
        embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        lp.appearance_embeddings = {
            1: [(0, embedding.copy(), 0.9)],
            2: [(4, embedding.copy(), 0.9)],
            3: [(50, embedding.copy(), 0.9)],
        }

        selected = select_multiview_resample_tracklets(lp, {1})
        self.assertEqual(selected, {1, 2})

        checker = np.indices((120, 80)).sum(axis=0) % 2
        frame = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        class FakeEncoder:
            def encode(self, image, boxes):
                return np.repeat(embedding[None, :], len(boxes), axis=0)

        with (
            patch(
                "flowmetric.identity.appearance.load_video_frames",
                side_effect=lambda source, indexes: {index: frame for index in indexes},
            ),
            patch(
                "flowmetric.identity.appearance.LocalAppearanceEncoder",
                return_value=FakeEncoder(),
            ),
        ):
            stats = densify_multiview_appearance(
                Path("video.mp4"),
                lp,
                Path("osnet.pt"),
                device="cpu",
                tracklet_ids=selected,
            )

        self.assertEqual(stats["added_samples"], 4)
        self.assertEqual(stats["densified_tracklets"], [1, 2])
        self.assertEqual(len(lp.appearance_embeddings[1]), 3)
        self.assertEqual(len(lp.appearance_embeddings[2]), 3)

    def test_multiview_requires_two_independent_crop_matches(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
        }
        e1, e2 = np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
        distractor = np.asarray([0.90, np.sqrt(1 - 0.90**2)])
        lp.appearance_embeddings = {
            1: [(0, e1, 0.9), (1, e2, 0.9)],
            2: [(3, e1, 0.9), (4, e2, 0.9), (5, e2, 0.9), (6, e2, 0.9)],
            3: [(3, distractor, 0.9), (4, distractor, 0.9)],
        }

        self.assertEqual(propose_appearance_stitches(lp), [])
        proposals = propose_multiview_appearance_stitches(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertEqual(proposals[0].selection, "multiview_consensus")
        self.assertEqual(proposals[0].support, 2)
        self.assertAlmostEqual(proposals[0].similarity, 1.0)

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            multiview_appearance=True,
        )
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["multiview_consensus_merges"], 1)

    def test_full_part_ensemble_resolves_consensus_without_losing_full_links(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9), (3, box.copy(), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([0.90, np.sqrt(1 - 0.90**2)]), 0.9)],
            3: [(3, np.asarray([0.91, np.sqrt(1 - 0.91**2)]), 0.9)],
        }
        lp.part_appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([0.995, np.sqrt(1 - 0.995**2)]), 0.9)],
            3: [(3, np.asarray([0.87, np.sqrt(1 - 0.87**2)]), 0.9)],
        }

        self.assertEqual(propose_appearance_stitches(lp), [])
        proposals = propose_ensemble_appearance_stitches(lp)

        self.assertEqual(len(proposals), 1)
        self.assertEqual((proposals[0].earlier_id, proposals[0].later_id), (1, 2))
        self.assertEqual(proposals[0].selection, "full_part_consensus")
        self.assertAlmostEqual(proposals[0].primary_similarity or 0.0, 0.90)
        self.assertAlmostEqual(proposals[0].secondary_similarity or 0.0, 0.995)

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            ensemble_appearance=True,
        )
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["ensemble_consensus_merges"], 1)

    def test_identity_graph_combines_and_prioritizes_multi_source_edges(self) -> None:
        appearance = AppearanceStitch(1, 2, 0.2, 0.1, 1.0, 0.97, 0.04)
        motion = TrackletStitch(1, 2, 0.2, 0.1, 1.0, 0.03)
        appearance_only = AppearanceStitch(3, 4, 0.2, 0.1, 1.0, 0.99, 0.02)

        edges = build_identity_graph_edges(
            [appearance, appearance_only],
            [appearance],
            [motion],
        )

        self.assertEqual((edges[0].left_id, edges[0].right_id), (1, 2))
        self.assertEqual(
            set(edges[0].sources),
            {"appearance_mutual", "appearance_strict", "motion"},
        )
        self.assertEqual(len(edges), 2)

    def test_offline_graph_resolves_ambiguous_branch_with_hard_constraint(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        first = np.asarray([20, 10, 60, 110], float)
        best = np.asarray([24, 10, 64, 110], float)
        alternative = np.asarray([100, 10, 140, 110], float)
        lp.dets = {
            0: [(1, first.copy(), 0.9)],
            1: [(1, first.copy(), 0.9)],
            3: [(2, best.copy(), 0.9), (3, alternative.copy(), 0.9)],
            4: [(2, best.copy(), 0.9), (3, alternative.copy(), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([0.995, 0.099875]), 0.9)],
            3: [(3, np.asarray([0.98, 0.198997]), 0.9)],
        }

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            offline_graph=True,
        )

        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["offline_graph_merges"], 1)
        self.assertTrue(
            any(
                item.get("reason") == "temporal_overlap"
                for item in stats["offline_graph_decisions"]
            )
        )
        self.assertEqual(
            stats["track_identity_status_counts"],
            {"graph_stitched": 2, "local_only": 1},
        )

    def test_scenario_evidence_resolves_supported_interior_occlusion(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=6)
        lp.dets = {
            0: [(1, np.asarray([10, 10, 30, 50], float), 0.9)],
            1: [(1, np.asarray([12, 10, 32, 50], float), 0.9)],
            3: [
                (2, np.asarray([16, 10, 36, 50], float), 0.9),
                (3, np.asarray([60, 10, 80, 50], float), 0.9),
            ],
            4: [
                (2, np.asarray([18, 10, 38, 50], float), 0.9),
                (3, np.asarray([60, 10, 80, 50], float), 0.9),
            ],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([0.92, np.sqrt(1 - 0.92**2)]), 0.9)],
            3: [(3, np.asarray([0.93, np.sqrt(1 - 0.93**2)]), 0.9)],
        }
        scene = {"edge_margin": 0.06}

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            offline_graph=True,
            scenario_constraints=True,
            scene_config=scene,
        )

        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual(stats["scenario_supported_merges"], 1)
        merged = next(
            item for item in stats["offline_graph_decisions"]
            if item["decision"] == "merge"
        )
        self.assertEqual(
            set(merged["scenario"]["signals"]),
            {"interior_occlusion", "direction_consistent"},
        )

    def test_scenario_rejects_reassociation_after_completed_exit(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=7)
        lp.dets = {
            0: [(1, np.asarray([20, 10, 40, 50], float), 0.9)],
            1: [(1, np.asarray([0, 10, 20, 50], float), 0.9)],
            5: [(2, np.asarray([30, 10, 50, 50], float), 0.9)],
            6: [(2, np.asarray([32, 10, 52, 50], float), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(5, np.asarray([0.995, 0.099875]), 0.9)],
        }

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            local_stitch=False,
            appearance_stitch=True,
            offline_graph=True,
            scenario_constraints=True,
            scene_config={"edge_margin": 0.1},
        )

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertEqual(stats["scenario_rejected_edges"], 1)
        self.assertEqual(
            stats["offline_graph_decisions"][0]["reason"],
            "scenario_completed_exit",
        )

    def test_tracker_appearance_bonus_is_bounded_and_respects_masks(self) -> None:
        motion = np.asarray([[0.4, 0.4], [0.4, 0.4]])
        embeddings = np.asarray([[0.0, 0.1], [0.2, 0.0]])
        proximity = np.asarray([[True, True], [True, False]])

        adjusted = apply_tracker_appearance_bonus(
            motion,
            embeddings,
            proximity_mask=proximity,
            max_embedding_distance=0.15,
        )

        self.assertAlmostEqual(adjusted[0, 0], 0.39)
        self.assertGreater(adjusted[0, 1], 0.39)
        self.assertEqual(adjusted[1, 0], 0.4)
        self.assertEqual(adjusted[1, 1], 0.4)
        self.assertTrue(np.all(adjusted <= motion))

    def test_tracker_encoder_converts_xywh_to_xyxy(self) -> None:
        class StubEncoder:
            def encode(self, frame, boxes):
                self.boxes = boxes
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

        stub = StubEncoder()
        adapter = TrackerAppearanceEncoder(stub)  # type: ignore[arg-type]

        features = adapter(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.asarray([[50, 40, 20, 30, 0]], dtype=np.float32),
        )

        self.assertTrue(np.allclose(stub.boxes[0], [40, 25, 60, 55]))
        self.assertTrue(np.allclose(features[0], [1, 0]))

    def test_part_visibility_downweights_occluded_lower_body(self) -> None:
        box = np.asarray([0, 0, 100, 200], float)
        lower_occluder = np.asarray([0, 90, 100, 200], float)

        visibility = part_visibility(box, [lower_occluder])

        self.assertEqual(visibility[0], 1.0)
        self.assertGreater(visibility[1], 0.8)
        self.assertAlmostEqual(float(visibility[2]), 0.05, places=5)

    def test_part_embedding_reduces_influence_of_occluded_region(self) -> None:
        features = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        visible = compose_part_embeddings(features, np.ones((2, 3)))
        masked = compose_part_embeddings(
            features,
            np.asarray([[1.0, 1.0, 0.05], [1.0, 1.0, 0.05]]),
        )

        self.assertTrue(np.allclose(np.linalg.norm(masked, axis=1), 1.0))
        self.assertGreater(
            float(np.dot(masked[0], masked[1])),
            float(np.dot(visible[0], visible[1])),
        )

    def test_video_sampler_returns_four_temporally_spread_observations(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=10)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            frame: [(1, box.copy(), 0.9)] for frame in range(7)
        }

        selected = sample_tracklet_observations(lp)

        self.assertEqual(
            [frame for frame, _, _ in selected[1]],
            [0, 2, 4, 6],
        )

    def test_video_sampler_repeats_short_tracklets_to_fixed_length(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=2)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {0: [(1, box.copy(), 0.9)]}

        selected = sample_tracklet_observations(lp)

        self.assertEqual(len(selected[1]), 4)
        self.assertEqual({frame for frame, _, _ in selected[1]}, {0})

    def test_mutual_best_osnet_proposal_merges_unresolved_tracklets(self) -> None:
        lp = _local_pass()

        proposals = propose_appearance_stitches(lp)
        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            appearance_stitch=True,
        )

        self.assertEqual(len(proposals), 1)
        self.assertGreater(proposals[0].similarity, 0.99)
        self.assertEqual(mapping[1], mapping[2])
        self.assertEqual(stats["appearance_stitch_candidates"], 1)
        self.assertEqual(stats["appearance_stitch_merges"], 1)
        self.assertEqual(
            stats["track_identity_status_counts"],
            {"appearance_stitched": 2},
        )

    def test_dissimilar_tracklets_are_not_proposed(self) -> None:
        lp = _local_pass()
        lp.appearance_embeddings[2] = [
            (3, np.asarray([0.0, 1.0, 0.0]), 0.9)
        ]

        proposals = propose_appearance_stitches(lp)

        self.assertEqual(proposals, [])

    def test_sinet_threshold_and_audit_source_are_configurable(self) -> None:
        lp = _local_pass()
        lp.appearance_embeddings[2] = [
            (3, np.asarray([0.75, np.sqrt(1.0 - 0.75**2), 0.0]), 0.9)
        ]

        mapping, _, stats = sync_ids(
            lp,
            [],
            iou_thr=0.3,
            appearance_stitch=True,
            appearance_min_similarity=0.7018,
            appearance_source="local_sinet_stitch",
        )

        self.assertEqual(mapping[1], mapping[2])
        self.assertEqual(stats["appearance_source"], "local_sinet_stitch")
        self.assertEqual(
            stats["appearance_stitch_decisions"][0]["source"],
            "local_sinet_stitch",
        )

    def test_ambiguous_best_match_is_not_proposed(self) -> None:
        lp = _local_pass()
        third = np.asarray([28, 10, 68, 110], float)
        lp.dets[5] = [(3, third.copy(), 0.9)]
        lp.dets[6] = [(3, third.copy(), 0.9)]
        lp.appearance_embeddings[3] = [
            (5, np.asarray([0.995, 0.04, 0.0]), 0.9)
        ]

        proposals = propose_appearance_stitches(lp)

        self.assertFalse(any(item.earlier_id == 1 for item in proposals))

    def test_global_assignment_resolves_jointly_ambiguous_pairs(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=8)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            0: [(1, box.copy(), 0.9), (2, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9), (2, box.copy(), 0.9)],
            3: [(3, box.copy(), 0.9), (4, box.copy(), 0.9)],
            4: [(3, box.copy(), 0.9), (4, box.copy(), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(0, np.asarray([0.95, 0.3122499]), 0.9)],
            3: [(3, np.asarray([0.98, 0.1989975]), 0.9)],
            4: [(3, np.asarray([0.98, -0.1989975]), 0.9)],
        }

        mutual = propose_appearance_stitches(lp)
        globally_assigned = propose_appearance_stitches(
            lp, global_assignment=True
        )

        self.assertEqual(mutual, [])
        self.assertEqual(
            {(item.earlier_id, item.later_id) for item in globally_assigned},
            {(1, 4), (2, 3)},
        )
        self.assertTrue(
            all(
                item.selection == "global_supplemental_assignment"
                for item in globally_assigned
            )
        )

    def test_endpoint_rescue_uses_two_boundary_samples(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=12)
        box = np.asarray([20, 10, 60, 110], float)
        lp.dets = {
            **{frame: [(1, box.copy(), 0.9)] for frame in range(6)},
            **{frame: [(2, box.copy(), 0.9)] for frame in range(7, 12)},
        }
        angle = np.deg2rad(34.0)
        positive = np.asarray([1.0, 0.0])
        early_view = np.asarray([np.cos(angle), np.sin(angle)])
        late_view = np.asarray([np.cos(angle), -np.sin(angle)])
        lp.appearance_embeddings = {
            1: [
                (frame, early_view.copy() if frame < 4 else positive.copy(), 0.9)
                for frame in range(6)
            ],
            2: [
                (frame, positive.copy() if frame < 9 else late_view.copy(), 0.9)
                for frame in range(7, 12)
            ],
        }

        control = propose_appearance_stitches(lp)
        rescued = propose_appearance_stitches(lp, endpoint_rescue=True)

        self.assertEqual(control, [])
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0].selection, "endpoint_temporal_rescue")
        self.assertGreaterEqual(rescued[0].similarity, 0.99)
        self.assertGreaterEqual(rescued[0].global_similarity or 0.0, 0.70)

    def test_endpoint_rescue_requires_two_samples_per_side(self) -> None:
        lp = _local_pass()
        lp.appearance_embeddings[1] = [lp.appearance_embeddings[1][0]]
        lp.appearance_embeddings[2] = [lp.appearance_embeddings[2][0]]
        lp.appearance_embeddings[2][0] = (
            3,
            np.asarray([0.75, np.sqrt(1.0 - 0.75**2), 0.0]),
            0.9,
        )

        rescued = propose_appearance_stitches(lp, endpoint_rescue=True)

        self.assertEqual(rescued, [])

    def test_astra_anchor_accepts_multiple_unambiguous_fragments(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=120, total=8)
        box = np.asarray([20, 10, 60, 110], float)
        lp.sample_idx = [0, 1, 3, 4, 6, 7]
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9)],
            4: [(2, box.copy(), 0.9)],
            6: [(3, box.copy(), 0.9)],
            7: [(3, box.copy(), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([1.0, 0.0]), 0.9)],
            2: [(3, np.asarray([0.998, 0.0632]), 0.9)],
            3: [(6, np.asarray([0.997, 0.0774]), 0.9)],
        }
        response = normalize_gallery_response(
            {
                "identities": [{
                    "id": 1,
                    "role": "customer",
                    "tracklet_ids": [1],
                    "confidence": 0.98,
                }],
                "abstained_tracklet_ids": [2, 3],
            },
            tracklets=[
                {"tracklet_id": 1, "crop_count": 2},
                {"tracklet_id": 2, "crop_count": 2},
                {"tracklet_id": 3, "crop_count": 2},
            ],
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )

        mutual = propose_appearance_stitches(lp)
        anchored = propose_anchored_appearance_stitches(
            lp,
            cluster_by_tracklet={1: "verified", 2: "query-2", 3: "query-3"},
            verified_clusters={"verified"},
        )
        control, _, _ = sync_ids(
            lp,
            [response],
            iou_thr=0.3,
            appearance_stitch=True,
            anchored_appearance=False,
        )
        rescued, _, stats = sync_ids(
            lp,
            [response],
            iou_thr=0.3,
            appearance_stitch=True,
            anchored_appearance=True,
        )

        self.assertEqual(mutual, [])
        self.assertEqual(len(anchored), 2)
        self.assertNotEqual(control[1], control[2])
        self.assertNotEqual(control[1], control[3])
        self.assertEqual(rescued[1], rescued[2])
        self.assertEqual(rescued[1], rescued[3])
        self.assertEqual(stats["anchored_appearance_candidates"], 2)
        self.assertEqual(stats["anchored_appearance_merges"], 2)


if __name__ == "__main__":
    unittest.main()
