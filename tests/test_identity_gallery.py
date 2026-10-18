from __future__ import annotations

import unittest

import numpy as np

from astra import AstraResponseError
from identity_gallery import (
    GALLERY_PROTOCOL,
    build_tracklet_gallery,
    normalize_gallery_response,
    validate_tracklet_reid_response,
)
from track import LocalPass, sync_ids


class TrackletGalleryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
