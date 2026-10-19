from __future__ import annotations

import unittest

import numpy as np

from astra import AstraResponseError
from identity_gallery import (
    GALLERY_PROTOCOL,
    build_targeted_candidate_shortlists,
    build_tracklet_gallery,
    constrain_targeted_shortlist_response,
    normalize_gallery_response,
    targeted_gallery_inventory,
    validate_targeted_shortlist_response,
    validate_tracklet_reid_response,
)
from track import LocalPass, sync_ids


class TrackletGalleryTests(unittest.TestCase):
    def test_targeted_shortlist_ranks_appearance_and_excludes_conflicts(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=12)
        near = np.asarray([10, 10, 30, 70], float)
        far = np.asarray([100, 10, 120, 70], float)
        lp.dets = {
            0: [(1, near.copy(), 0.9), (2, far.copy(), 0.9)],
            1: [(1, near.copy(), 0.9), (2, far.copy(), 0.9)],
            10: [(9, near.copy(), 0.9), (3, far.copy(), 0.9)],
            11: [(9, near.copy(), 0.9)],
        }
        lp.appearance_embeddings = {
            1: [(0, np.asarray([0.99, 0.1]), 0.9)],
            2: [(0, np.asarray([0.0, 1.0]), 0.9)],
            3: [(10, np.asarray([1.0, 0.0]), 0.9)],
            9: [(10, np.asarray([1.0, 0.0]), 0.9)],
        }

        shortlists = build_targeted_candidate_shortlists(
            lp,
            [],
            query_tracklet_ids=[9],
            reference_tracklet_ids=[1, 2, 3],
            limit=3,
        )

        self.assertEqual(shortlists, {9: [1, 2]})

    def test_targeted_shortlist_response_enforces_allowed_references(self) -> None:
        valid = {
            "identities": [
                {
                    "id": 1,
                    "role": "customer",
                    "tracklet_ids": [7, 12],
                    "confidence": 0.9,
                },
                {
                    "id": 2,
                    "role": "unknown",
                    "tracklet_ids": [13],
                    "confidence": 0.7,
                },
            ],
            "abstained_tracklet_ids": [],
        }
        self.assertIs(
            validate_targeted_shortlist_response(
                valid,
                target_tracklet_ids=[12, 13],
                candidate_shortlists={12: [7], 13: [8]},
            ),
            valid,
        )

        outside = {
            **valid,
            "identities": [{**valid["identities"][0], "tracklet_ids": [8, 12]}],
        }
        with self.assertRaisesRegex(AstraResponseError, "outside its shortlist"):
            validate_targeted_shortlist_response(
                outside,
                target_tracklet_ids=[12],
                candidate_shortlists={12: [7]},
            )

        query_merge = {
            **valid,
            "identities": [{**valid["identities"][0], "tracklet_ids": [12, 13]}],
        }
        with self.assertRaisesRegex(AstraResponseError, "multiple unresolved"):
            validate_targeted_shortlist_response(
                query_merge,
                target_tracklet_ids=[12, 13],
                candidate_shortlists={12: [7], 13: [8]},
            )

        constrained, rejected = constrain_targeted_shortlist_response(
            {
                "identities": [
                    {
                        "id": 1,
                        "role": "customer",
                        "tracklet_ids": [7, 12, 13],
                        "confidence": 0.9,
                    }
                ],
                "abstained_tracklet_ids": [8],
            },
            target_tracklet_ids=[12, 13],
            candidate_shortlists={12: [7], 13: [8]},
        )
        self.assertEqual(rejected, 2)
        self.assertEqual(constrained["identities"][0]["tracklet_ids"], [7])
        self.assertEqual(constrained["abstained_tracklet_ids"], [8, 12, 13])

    def test_targeted_inventory_keeps_unresolved_and_one_reference_per_group(self) -> None:
        reids = [
            {
                "_meta": {
                    "protocol": GALLERY_PROTOCOL,
                    "gallery_associations": [
                        {
                            "id": 1,
                            "tracklet_ids": [7, 8],
                            "confidence": 0.9,
                            "support": {"7": 1, "8": 3},
                        }
                    ],
                    "abstained_tracklet_ids": [12, 13],
                }
            },
            {
                "_meta": {
                    "protocol": GALLERY_PROTOCOL,
                    "gallery_associations": [
                        {
                            "id": 2,
                            "tracklet_ids": [12],
                            "confidence": 0.8,
                            "support": {"12": 2},
                        }
                    ],
                    "abstained_tracklet_ids": [],
                }
            },
        ]

        queries, references = targeted_gallery_inventory(reids)

        self.assertEqual(queries, [7, 13])
        self.assertEqual(references, [8, 12])

    def test_targeted_inventory_selects_temporally_diverse_references(self) -> None:
        reids = [{
            "_meta": {
                "protocol": GALLERY_PROTOCOL,
                "gallery_associations": [{
                    "id": 1,
                    "tracklet_ids": [1, 2, 3],
                    "confidence": 0.95,
                    "support": {"1": 3, "2": 3, "3": 3},
                }],
                "abstained_tracklet_ids": [9],
            }
        }]

        queries, references = targeted_gallery_inventory(
            reids,
            representatives_per_group=2,
            tracklet_midpoints={1: 0.0, 2: 5.0, 3: 100.0},
        )

        self.assertEqual(queries, [9])
        self.assertEqual(references, [1, 3])

        with self.assertRaisesRegex(ValueError, "representatives_per_group"):
            targeted_gallery_inventory(reids, representatives_per_group=0)

    def test_gallery_contains_numbered_crops_and_metadata(self) -> None:
        frames = [np.full((100, 160, 3), 40 + index * 10, np.uint8) for index in range(3)]
        detections = {
            0: [(7, np.asarray([10, 10, 50, 90], float), 0.8)],
            10: [
                (7, np.asarray([12, 10, 52, 90], float), 0.9),
                (12, np.asarray([80, 15, 120, 90], float), 0.7),
            ],
            20: [(12, np.asarray([82, 15, 122, 90], float), 0.85)],
        }

        gallery, metadata = build_tracklet_gallery(frames, [0, 10, 20], detections)

        self.assertEqual([item["tracklet_id"] for item in metadata], [7, 12])
        self.assertEqual([item["crop_count"] for item in metadata], [2, 2])
        self.assertEqual(gallery.ndim, 3)
        self.assertGreater(gallery.shape[0], 0)
        self.assertGreater(gallery.shape[1], 0)

    def test_gallery_selects_quality_within_temporal_segments(self) -> None:
        frames = [np.full((100, 160, 3), 80, np.uint8) for _ in range(6)]
        confidences = [0.2, 0.9, 0.8, 0.3, 0.4, 0.95]
        detections = {
            frame: [
                (7, np.asarray([10, 10, 50, 90], float), confidence)
            ]
            for frame, confidence in enumerate(confidences)
        }

        _, metadata = build_tracklet_gallery(
            frames, list(range(6)), detections, crops_per_tracklet=3
        )

        self.assertEqual(metadata[0]["video_frames"], [1, 2, 5])
        self.assertEqual(
            metadata[0]["selection_strategy"],
            "quality_by_temporal_segment_v1",
        )
        self.assertEqual(len(metadata[0]["crop_quality"]), 3)

    def test_response_requires_exact_partition_and_explicit_abstain(self) -> None:
        valid = {
            "identities": [
                {
                    "id": 1,
                    "role": "customer",
                    "tracklet_ids": [7],
                    "confidence": 0.9,
                }
            ],
            "abstained_tracklet_ids": [12],
        }
        self.assertIs(
            validate_tracklet_reid_response(valid, expected_tracklet_ids=[7, 12]),
            valid,
        )
        missing = {"identities": valid["identities"], "abstained_tracklet_ids": []}
        with self.assertRaisesRegex(AstraResponseError, "omitted"):
            validate_tracklet_reid_response(missing, expected_tracklet_ids=[7, 12])
        duplicated = {
            "identities": valid["identities"],
            "abstained_tracklet_ids": [7, 12],
        }
        with self.assertRaisesRegex(AstraResponseError, "more than once"):
            validate_tracklet_reid_response(duplicated, expected_tracklet_ids=[7, 12])

    def test_normalized_gallery_drives_weighted_merge_and_abstain(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=4)
        lp.sample_idx = [0, 1, 2, 3]
        left = np.asarray([10, 10, 50, 90], float)
        right = np.asarray([12, 10, 52, 90], float)
        lp.dets = {
            0: [(7, left.copy(), 0.9)],
            1: [(7, left.copy(), 0.9)],
            2: [(12, right.copy(), 0.9)],
            3: [(12, right.copy(), 0.9)],
        }
        response = {
            "identities": [
                {
                    "id": 3,
                    "role": "customer",
                    "tracklet_ids": [7, 12],
                    "confidence": 0.9,
                }
            ],
            "abstained_tracklet_ids": [],
        }
        tracklets = [
            {"tracklet_id": 7, "crop_count": 2},
            {"tracklet_id": 12, "crop_count": 2},
        ]
        normalized = normalize_gallery_response(
            response,
            tracklets=tracklets,
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )

        mapping, roles, stats = sync_ids(lp, [normalized], iou_thr=0.3)

        self.assertEqual(normalized["_meta"]["protocol"], GALLERY_PROTOCOL)
        self.assertEqual(mapping[7], mapping[12])
        self.assertEqual(roles[mapping[7]], "customer")
        self.assertTrue(all(item["decision"] == "merge" for item in stats["identity_decisions"]))

        normalized["_meta"]["gallery_associations"][0]["confidence"] = 0.4
        mapping, _, stats = sync_ids(lp, [normalized], iou_thr=0.3)
        self.assertNotEqual(mapping[7], mapping[12])
        self.assertTrue(all(item["decision"] == "abstain" for item in stats["identity_decisions"]))

    def test_sync_reports_late_revision_of_weak_identity_merge(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=4)
        box = np.asarray([10, 10, 50, 90], float)
        lp.sample_idx = [0, 1, 2, 3]
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            1: [(2, box.copy(), 0.9)],
            2: [(1, box.copy(), 0.9)],
            3: [(2, box.copy(), 0.9)],
        }
        first = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 10,
                        "role": "customer",
                        "tracklet_ids": [1, 2],
                        "confidence": 0.6,
                    }
                ],
                "abstained_tracklet_ids": [],
            },
            tracklets=[
                {"tracklet_id": 1, "crop_count": 1},
                {"tracklet_id": 2, "crop_count": 1},
            ],
            video_frames=[0, 1],
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )
        second = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 20,
                        "role": "customer",
                        "tracklet_ids": [1],
                        "confidence": 0.95,
                    },
                    {
                        "id": 21,
                        "role": "staff",
                        "tracklet_ids": [2],
                        "confidence": 0.95,
                    },
                ],
                "abstained_tracklet_ids": [],
            },
            tracklets=[
                {"tracklet_id": 1, "crop_count": 1},
                {"tracklet_id": 2, "crop_count": 1},
            ],
            video_frames=[2, 3],
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )

        mapping, roles, stats = sync_ids(
            lp, [first, second], iou_thr=0.3, min_support=1
        )

        self.assertNotEqual(mapping[1], mapping[2])
        self.assertEqual(roles[mapping[1]], "customer")
        self.assertEqual(roles[mapping[2]], "staff")
        self.assertEqual(stats["memory_strategy"], "retrospective_strongest_first_v1")
        self.assertEqual(len(stats["revision_log"]), 1)
        self.assertEqual(stats["revision_log"][0]["from"], "merge")
        self.assertEqual(stats["revision_log"][0]["to"], "reject")

    def test_targeted_gallery_can_attach_an_abstained_tracklet_to_reference(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=4)
        lp.sample_idx = [0, 1, 2, 3]
        first_box = np.asarray([10, 10, 50, 90], float)
        second_box = np.asarray([12, 10, 52, 90], float)
        lp.dets = {
            0: [(7, first_box.copy(), 0.9)],
            1: [(7, first_box.copy(), 0.9)],
            2: [(12, second_box.copy(), 0.9)],
            3: [(12, second_box.copy(), 0.9)],
        }
        base = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 1,
                        "role": "customer",
                        "tracklet_ids": [7],
                        "confidence": 0.9,
                    }
                ],
                "abstained_tracklet_ids": [12],
            },
            tracklets=[
                {"tracklet_id": 7, "crop_count": 2},
                {"tracklet_id": 12, "crop_count": 2},
            ],
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )
        targeted = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 20,
                        "role": "customer",
                        "tracklet_ids": [7, 12],
                        "confidence": 0.9,
                    }
                ],
                "abstained_tracklet_ids": [],
            },
            tracklets=[
                {"tracklet_id": 7, "crop_count": 2},
                {"tracklet_id": 12, "crop_count": 2},
            ],
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
            gallery_scope="targeted",
            targeted_query_ids=[12],
            targeted_candidate_shortlists={12: [7]},
        )

        mapping, _, stats = sync_ids(lp, [base, targeted], iou_thr=0.3)

        self.assertEqual(mapping[7], mapping[12])
        self.assertEqual(stats["targeted_batches"], 1)
        self.assertEqual(stats["targeted_query_tracklets"], 1)
        self.assertEqual(stats["targeted_shortlist_queries"], 1)
        self.assertEqual(stats["targeted_shortlist_options"], 1)

    def test_targeted_gallery_adapts_only_reinforced_single_crop_evidence(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=3)
        lp.sample_idx = [0, 1, 2]
        box = np.asarray([10, 10, 50, 90], float)
        lp.dets = {
            0: [(7, box.copy(), 0.9)],
            1: [(7, box.copy(), 0.9)],
            2: [(12, box.copy(), 0.9)],
        }
        base = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 1,
                        "role": "customer",
                        "tracklet_ids": [7, 12],
                        "confidence": 0.96,
                    }
                ],
                "abstained_tracklet_ids": [],
            },
            tracklets=[
                {"tracklet_id": 7, "crop_count": 2},
                {"tracklet_id": 12, "crop_count": 1},
            ],
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
        )
        targeted = normalize_gallery_response(
            {
                "identities": [
                    {
                        "id": 20,
                        "role": "customer",
                        "tracklet_ids": [7, 12],
                        "confidence": 0.95,
                    }
                ],
                "abstained_tracklet_ids": [],
            },
            tracklets=[
                {"tracklet_id": 7, "crop_count": 2},
                {"tracklet_id": 12, "crop_count": 1},
            ],
            video_frames=lp.sample_idx,
            detections=lp.dets,
            original_width=lp.width,
            original_height=lp.height,
            requested_model="requested",
            actual_model="actual",
            gallery_scope="targeted",
            targeted_query_ids=[12],
        )

        conservative, _, conservative_stats = sync_ids(
            lp,
            [base, targeted],
            iou_thr=0.3,
            adaptive_singletons=False,
        )
        adaptive, _, adaptive_stats = sync_ids(
            lp,
            [base, targeted],
            iou_thr=0.3,
            adaptive_singletons=True,
        )

        self.assertNotEqual(conservative[7], conservative[12])
        self.assertEqual(adaptive[7], adaptive[12])
        self.assertEqual(conservative_stats["adaptive_singleton_merges"], 0)
        self.assertEqual(adaptive_stats["adaptive_singleton_candidates"], 1)
        self.assertEqual(adaptive_stats["adaptive_singleton_merges"], 1)
        decision = next(
            item
            for item in adaptive_stats["identity_decisions"]
            if item["tracklet_id"] == 12 and item["batch"] == 1
        )
        self.assertEqual(decision["policy"], "reinforced_singleton_v1")
        self.assertEqual(decision["reference_tracklet_id"], 7)
        self.assertEqual(decision["confirming_batch"], 0)

    def test_adaptive_singleton_rejects_low_confidence_repeat(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=3)
        lp.sample_idx = [0, 1, 2]
        box = np.asarray([10, 10, 50, 90], float)
        lp.dets = {
            0: [(7, box.copy(), 0.9)],
            1: [(7, box.copy(), 0.9)],
            2: [(12, box.copy(), 0.9)],
        }
        responses = [
            normalize_gallery_response(
                {
                    "identities": [{
                        "id": identity_id,
                        "role": "customer",
                        "tracklet_ids": [7, 12],
                        "confidence": 0.89,
                    }],
                    "abstained_tracklet_ids": [],
                },
                tracklets=[
                    {"tracklet_id": 7, "crop_count": 2},
                    {"tracklet_id": 12, "crop_count": 1},
                ],
                video_frames=lp.sample_idx,
                detections=lp.dets,
                original_width=lp.width,
                original_height=lp.height,
                requested_model="requested",
                actual_model="actual",
                **(
                    {"gallery_scope": "targeted", "targeted_query_ids": [12]}
                    if identity_id == 20
                    else {}
                ),
            )
            for identity_id in (1, 20)
        ]

        mapping, _, stats = sync_ids(lp, responses, iou_thr=0.3)

        self.assertNotEqual(mapping[7], mapping[12])
        self.assertEqual(stats["adaptive_singleton_candidates"], 0)
        self.assertEqual(stats["adaptive_singleton_merges"], 0)


if __name__ == "__main__":
    unittest.main()
