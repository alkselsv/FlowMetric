from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astra import (
    AstraCacheError,
    AstraResponseError,
    astra_reid_people,
    build_astra_cache_manifest,
    load_astra_cache,
    save_astra_cache,
    scale_astra_boxes_to_original,
    validate_astra_response,
    validate_reid_batches,
)
from track import (
    LocalPass,
    astra_pass,
    make_batches,
    sync_ids,
    validate_astra_options,
    validate_reids_against_local,
)


def _response(video_frames: list[int]) -> dict:
    return {
        "people": [
            {
                "id": 1,
                "role": "customer",
                "detections": [
                    {"frame_index": 0, "xyxy": [10.0, 5.0, 50.0, 40.0]},
                ],
            }
        ],
        "notes": "ok",
        "_meta": {
            "coordinate_space": "original",
            "scale": 0.5,
            "scale_x": 0.5,
            "scale_y": 0.5,
            "width": 200,
            "height": 100,
            "resized_width": 100,
            "resized_height": 50,
            "original_width": 200,
            "original_height": 100,
            "n_frames": len(video_frames),
            "video_frames": video_frames,
            "requested_model": "requested-model",
            "actual_model": "actual-model",
        },
    }


class AstraCacheTests(unittest.TestCase):
    def test_versioned_cache_roundtrip_and_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"not a real video, but stable source bytes")
            cache = root / "astra.json"
            expected = build_astra_cache_manifest(
                source,
                fps=24.0,
                width=200,
                height=100,
                frame_count=20,
                sample_frame_indexes=[0, 10],
                sample_fps=2.0,
                batch_size=16,
                requested_model="requested-model",
                effort="low",
            )
            stored = copy.deepcopy(expected)
            stored["astra"]["resolved_models"] = ["requested-model"]
            stored["astra"]["actual_model"] = "actual-model"
            stored["astra"]["actual_models"] = ["actual-model"]
            reids = [_response([0, 10])]

            save_astra_cache(cache, manifest=stored, reids=reids)

            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(payload["cache_version"], 1)
            loaded = load_astra_cache(
                cache,
                expected_manifest=expected,
                expected_batches=[[0, 10]],
                original_width=200,
                original_height=100,
            )
            self.assertEqual(loaded, reids)

            mismatched = copy.deepcopy(expected)
            mismatched["astra"]["sample_fps"] = 1.0
            with self.assertRaisesRegex(AstraCacheError, "sample_fps"):
                load_astra_cache(
                    cache,
                    expected_manifest=mismatched,
                    expected_batches=[[0, 10]],
                    original_width=200,
                    original_height=100,
                )

            shortlisted = copy.deepcopy(expected)
            shortlisted["astra"]["targeted_shortlist"] = True
            with self.assertRaisesRegex(AstraCacheError, "targeted_shortlist"):
                load_astra_cache(
                    cache,
                    expected_manifest=shortlisted,
                    expected_batches=[[0, 10]],
                    original_width=200,
                    original_height=100,
                )

            tracker_reid = copy.deepcopy(expected)
            tracker_reid["astra"]["tracker_reid"] = True
            with self.assertRaisesRegex(AstraCacheError, "tracker_reid"):
                load_astra_cache(
                    cache,
                    expected_manifest=tracker_reid,
                    expected_batches=[[0, 10]],
                    original_width=200,
                    original_height=100,
                )

    def test_legacy_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            cache = root / "astra.json"
            cache.write_text("[]", encoding="utf-8")
            manifest = build_astra_cache_manifest(
                source,
                fps=24.0,
                width=200,
                height=100,
                frame_count=20,
                sample_frame_indexes=[0, 10],
                sample_fps=2.0,
                batch_size=16,
                requested_model=None,
                effort="low",
            )

            with self.assertRaisesRegex(AstraCacheError, "legacy"):
                load_astra_cache(
                    cache,
                    expected_manifest=manifest,
                    expected_batches=[[0, 10]],
                    original_width=200,
                    original_height=100,
                )

    def test_invalid_batch_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            make_batches(5, 0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_batches(5, 2, overlap=-1)
        with self.assertRaisesRegex(ValueError, "smaller"):
            make_batches(5, 2, overlap=2)
        with self.assertRaisesRegex(ValueError, "sample_fps"):
            validate_astra_options(
                sample_fps=float("nan"), batch_size=16, workers=1, match_iou=0.3
            )
        with self.assertRaisesRegex(ValueError, "workers"):
            validate_astra_options(
                sample_fps=1.0, batch_size=16, workers=0, match_iou=0.3
            )
        with self.assertRaisesRegex(ValueError, "match_iou"):
            validate_astra_options(
                sample_fps=1.0, batch_size=16, workers=1, match_iou=1.1
            )
        with self.assertRaisesRegex(ValueError, "targeted_shortlist_size"):
            validate_astra_options(
                sample_fps=1.0,
                batch_size=16,
                workers=1,
                match_iou=0.3,
                targeted_shortlist_size=2,
            )


class AstraResponseValidationTests(unittest.TestCase):
    def test_astra_pass_runs_targeted_followup_for_unresolved_tracklet(self) -> None:
        lp = LocalPass(fps=10.0, width=160, height=100, total=5)
        lp.sample_idx = [0, 1, 2, 3]
        lp.sample_frames = [np.full((100, 160, 3), 80, np.uint8) for _ in range(4)]
        left = np.asarray([10, 10, 50, 90], float)
        right = np.asarray([12, 10, 52, 90], float)
        lp.dets = {
            0: [(7, left.copy(), 0.9)],
            1: [(7, left.copy(), 0.9)],
            2: [(12, right.copy(), 0.9)],
            3: [(12, right.copy(), 0.9)],
            4: [(12, right.copy(), 0.95)],
        }
        base_response = {
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
        targeted_response = {
            "identities": [
                {
                    "id": 20,
                    "role": "customer",
                    "tracklet_ids": [7, 12],
                    "confidence": 0.9,
                }
            ],
            "abstained_tracklet_ids": [],
        }

        with (
            patch(
                "flowmetric.infrastructure.astra.make_client",
                return_value=(SimpleNamespace(base_url="mock"), "mock-model"),
            ),
            patch(
                "flowmetric.identity.gallery.astra_reid_tracklet_gallery",
                side_effect=[
                    (base_response, "mock-model"),
                    (targeted_response, "mock-model"),
                ],
            ) as gallery_call,
            patch(
                "flowmetric.media.video.load_video_frames",
                return_value={4: np.full((100, 160, 3), 100, np.uint8)},
            ) as frame_loader,
        ):
            reids = astra_pass(
                lp,
                source=Path("video.mp4"),
                model="mock-model",
                effort="low",
                batch_size=16,
                batch_overlap=4,
                workers=1,
                targeted_pass=True,
                merge_confidence=0.55,
                min_support=2,
                targeted_resampling=True,
                targeted_resample_budget=4,
                targeted_shortlist=True,
                targeted_shortlist_size=3,
            )

        self.assertEqual(len(reids), 2)
        self.assertEqual(reids[1]["_meta"]["gallery_scope"], "targeted")
        self.assertEqual(reids[1]["_meta"]["targeted_query_ids"], [12])
        self.assertEqual(reids[1]["_meta"]["targeted_resampled_frames"], [4])
        self.assertEqual(
            reids[1]["_meta"]["targeted_candidate_shortlists"], {"12": [7]}
        )
        self.assertEqual(gallery_call.call_count, 2)
        self.assertEqual(gallery_call.call_args_list[1].kwargs["target_tracklet_ids"], [12])
        self.assertEqual(
            gallery_call.call_args_list[1].kwargs["candidate_shortlists"],
            {12: [7]},
        )
        frame_loader.assert_called_once_with(Path("video.mp4"), (4,))

    def test_valid_box_is_clipped_to_image(self) -> None:
        response = _response([0, 10])
        response["people"][0]["detections"][0]["xyxy"] = [-5, -4, 250, 90]

        validate_astra_response(response)

        self.assertEqual(
            response["people"][0]["detections"][0]["xyxy"],
            [0.0, 0.0, 200.0, 90.0],
        )

    def test_semantically_invalid_responses_are_rejected(self) -> None:
        cases = {}

        non_finite = _response([0, 10])
        non_finite["people"][0]["detections"][0]["xyxy"][0] = float("nan")
        cases["non-finite"] = non_finite

        reversed_box = _response([0, 10])
        reversed_box["people"][0]["detections"][0]["xyxy"] = [30, 5, 10, 40]
        cases["not strictly ordered"] = reversed_box

        invalid_frame = _response([0, 10])
        invalid_frame["people"][0]["detections"][0]["frame_index"] = 2
        cases["outside"] = invalid_frame

        invalid_role = _response([0, 10])
        invalid_role["people"][0]["role"] = "manager"
        cases["invalid role"] = invalid_role

        invalid_id = _response([0, 10])
        invalid_id["people"][0]["id"] = True
        cases["non-negative integer"] = invalid_id

        duplicate = _response([0, 10])
        duplicate["people"][0]["detections"].append(
            {"frame_index": 0, "xyxy": [60, 5, 80, 40]}
        )
        cases["duplicate detection"] = duplicate

        for message, response in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(AstraResponseError, message) as raised:
                    validate_astra_response(response)
                self.assertTrue(raised.exception.retry_safe)

    def test_incomplete_batch_coverage_is_rejected(self) -> None:
        with self.assertRaisesRegex(AstraResponseError, "got 1 batches, expected 2"):
            validate_reid_batches(
                [_response([0, 10])],
                expected_batches=[[0, 10], [10, 20]],
                original_width=200,
                original_height=100,
            )

    def test_targeted_followup_batch_is_allowed_but_arbitrary_extra_is_not(self) -> None:
        base = _response([0, 10])
        targeted = _response([0, 10])
        targeted["_meta"].update(
            {
                "protocol": "tracklet_gallery_v1",
                "gallery_scope": "targeted",
                "targeted_query_ids": [7],
            }
        )

        validate_reid_batches(
            [base, targeted],
            expected_batches=[[0, 10]],
            original_width=200,
            original_height=100,
        )

        with self.assertRaisesRegex(AstraResponseError, "unexpected extra"):
            validate_reid_batches(
                [base, _response([0, 10])],
                expected_batches=[[0, 10]],
                original_width=200,
                original_height=100,
            )

    def test_mixed_frame_shapes_fail_before_any_client_call(self) -> None:
        frames = [
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.zeros((21, 30, 3), dtype=np.uint8),
        ]

        with self.assertRaisesRegex(ValueError, "same shape"):
            astra_reid_people(frames, model="unused", client=object())

    def test_empty_people_cannot_claim_coverage_when_local_boxes_exist(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=1)
        lp.dets = {0: [(1, np.asarray([10, 5, 50, 40], float), 0.9)]}
        response = _response([0])
        response["people"] = []
        response["_meta"]["n_frames"] = 1

        with self.assertRaisesRegex(AstraResponseError, "no detections"):
            validate_reids_against_local(lp, [response])


class AstraScalingTests(unittest.TestCase):
    def test_scaling_to_original_is_idempotent(self) -> None:
        response = _response([0, 10])
        response["_meta"].update(
            {
                "coordinate_space": "resized",
                "width": 100,
                "height": 50,
            }
        )

        first = scale_astra_boxes_to_original(response)
        snapshot = copy.deepcopy(first)
        second = scale_astra_boxes_to_original(first)

        self.assertIs(second, first)
        self.assertEqual(second, snapshot)
        self.assertEqual(second["_meta"]["coordinate_space"], "original")
        self.assertEqual(
            second["people"][0]["detections"][0]["xyxy"],
            [20.0, 10.0, 100.0, 80.0],
        )


class LegacyExperimentCompatibilityTests(unittest.TestCase):
    def test_sync_normalizes_old_original_boxes_and_infers_sample_frames(self) -> None:
        lp = LocalPass(fps=24.0, width=200, height=100, total=20)
        lp.dets = {0: [(7, np.asarray([20.0, 10.0, 100.0, 80.0]), 0.9)]}
        legacy = {
            "people": [
                {
                    "id": 0,
                    "role": "customer",
                    "detections": [
                        {"frame_index": 0, "xyxy": [20.0, 10.0, 100.0, 80.0]}
                    ],
                }
            ],
            "notes": "historical experiment response",
            "_meta": {
                "scale": 0.5,
                "width": 100,
                "height": 50,
                "n_frames": 1,
                "video_frames": [0],
            },
        }

        tid_to_gid, gid_role, stats = sync_ids(
            lp, [legacy], iou_thr=0.3, min_support=1
        )

        self.assertEqual(tid_to_gid, {7: 1})
        self.assertEqual(gid_role, {1: "customer"})
        self.assertEqual(stats["astra_verified"], 1)
        self.assertEqual(legacy["_meta"]["coordinate_space"], "original")
        self.assertEqual(legacy["people"][0]["detections"][0]["xyxy"], [20, 10, 100, 80])


class SyncConstraintTests(unittest.TestCase):
    def test_iou_threshold_is_applied_before_hungarian_assignment(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=1)
        lp.sample_idx = [0]
        lp.dets = {
            0: [
                (1, np.asarray([0.0, 0.0, 20.0, 40.0]), 0.9),
                (2, np.asarray([30.0, 0.0, 50.0, 40.0]), 0.9),
            ]
        }
        response = _response([0])
        response["people"] = [
            {"id": 1, "role": "customer", "detections": [{"frame_index": 0, "xyxy": [0, 0, 20, 40]}]},
            {"id": 2, "role": "staff", "detections": [{"frame_index": 0, "xyxy": [30, 0, 50, 40]}]},
        ]
        response["_meta"]["n_frames"] = 1

        # Ungated maximization prefers 0.9 + 0.1; the 0.1 edge is then lost.
        # Gating first selects the two valid 0.4 off-diagonal edges instead.
        with patch(
            "flowmetric.identity.sync.iou_matrix",
            return_value=np.asarray([[0.9, 0.4], [0.4, 0.1]]),
        ):
            _, _, stats = sync_ids(lp, [response], iou_thr=0.3)

        self.assertEqual(stats["matched_boxes"], 2)

    def test_simultaneous_local_tracks_cannot_merge_into_one_person(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=11)
        lp.sample_idx = [0, 10]
        box = np.asarray([10.0, 5.0, 50.0, 40.0])
        lp.dets = {
            0: [(1, box.copy(), 0.9)],
            5: [(1, box.copy(), 0.9), (2, np.asarray([80.0, 5.0, 120.0, 40.0]), 0.9)],
            10: [(2, box.copy(), 0.9)],
        }
        response = _response([0, 10])
        response["people"][0]["detections"] = [
            {"frame_index": 0, "xyxy": box.tolist()},
            {"frame_index": 1, "xyxy": box.tolist()},
        ]

        tid_to_gid, _, stats = sync_ids(
            lp, [response], iou_thr=0.3, min_support=1
        )

        self.assertNotEqual(tid_to_gid[1], tid_to_gid[2])
        self.assertEqual(stats["rejected_merge_reasons"], {"temporal_overlap": 1})
        self.assertEqual(stats["simultaneous_local_pairs"], 1)
        self.assertEqual(stats["spatial_cannot_link_pairs"], 1)
        self.assertEqual(stats["track_identity"]["2"]["status"], "rejected")

    def test_nested_duplicate_tracks_may_merge_despite_coexistence(self) -> None:
        lp = LocalPass(fps=10.0, width=200, height=100, total=11)
        lp.sample_idx = [0, 10]
        outer = np.asarray([10.0, 5.0, 50.0, 45.0])
        inner = np.asarray([11.0, 6.0, 49.0, 44.0])
        lp.dets = {
            0: [(1, outer.copy(), 0.9)],
            5: [(1, outer.copy(), 0.9), (2, inner.copy(), 0.8)],
            10: [(2, outer.copy(), 0.9)],
        }
        response = _response([0, 10])
        response["people"][0]["detections"] = [
            {"frame_index": 0, "xyxy": outer.tolist()},
            {"frame_index": 1, "xyxy": outer.tolist()},
        ]

        tid_to_gid, _, stats = sync_ids(
            lp, [response], iou_thr=0.3, min_support=1
        )

        self.assertEqual(tid_to_gid[1], tid_to_gid[2])
        self.assertEqual(stats["rejected_merges"], 0)
        self.assertEqual(stats["simultaneous_local_pairs"], 1)
        self.assertEqual(stats["spatial_cannot_link_pairs"], 0)
        self.assertEqual(stats["track_identity"]["2"]["status"], "merged")


if __name__ == "__main__":
    unittest.main()
