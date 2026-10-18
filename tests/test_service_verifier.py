from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from astra import AstraResponseError
from events import ServiceCandidate, ServiceInterval
from service_verifier import (
    AstraServiceVerifier,
    annotate_service_crop,
    select_service_frames,
    service_crop_bounds,
    validate_service_response,
)


def _candidate(
    confidence: float,
    *,
    staff: str = "7",
    customer: str = "12",
    start: int = 2,
    end: int = 7,
) -> ServiceCandidate:
    return ServiceCandidate(
        ServiceInterval(
            staff=staff,
            customer=customer,
            start_frame=start,
            end_frame=end,
            left_censored=False,
            right_censored=False,
            confidence=confidence,
        ),
        table_box=(25.0, 30.0, 75.0, 75.0),
    )


def _tracks(count: int = 12) -> dict:
    frames = list(range(count))
    return {
        "fps": 10.0,
        "width": 100,
        "height": 100,
        "n_frames": count,
        "tracks": {
            "7": {
                "frames": frames,
                "boxes": [[10.0, 10.0, 30.0, 70.0] for _ in frames],
            },
            "12": {
                "frames": frames,
                "boxes": [[50.0, 15.0, 70.0, 75.0] for _ in frames],
            },
        },
    }


def _frames(_source: Path, indexes: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {index: np.zeros((100, 100, 3), np.uint8) for index in indexes}


class ServiceResponseTests(unittest.TestCase):
    def test_valid_response_is_normalized(self) -> None:
        response = validate_service_response(
            {
                "is_service": True,
                "staff_id": 7,
                "customer_id": 12,
                "confidence": 0.91,
            },
            expected_staff_id="7",
            expected_customer_id="12",
        )
        self.assertEqual(
            response,
            {
                "is_service": True,
                "staff_id": 7,
                "customer_id": 12,
                "confidence": 0.91,
            },
        )

    def test_extra_missing_and_semantically_invalid_fields_are_rejected(self) -> None:
        valid = {
            "is_service": True,
            "staff_id": 7,
            "customer_id": 12,
            "confidence": 0.91,
        }
        cases = []
        cases.append(({**valid, "boxes": []}, "unexpected"))
        missing = dict(valid)
        del missing["confidence"]
        cases.append((missing, "missing"))
        cases.append(({**valid, "is_service": 1}, "boolean"))
        cases.append(({**valid, "staff_id": True}, "integer"))
        cases.append(({**valid, "confidence": float("nan")}, "finite"))
        cases.append(({**valid, "confidence": 1.1}, r"\[0, 1\]"))
        cases.append(({**valid, "customer_id": 13}, "changed candidate identities"))

        for response, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AstraResponseError, message):
                    validate_service_response(
                        response,
                        expected_staff_id="7",
                        expected_customer_id="12",
                    )

    def test_production_ids_must_be_canonical_numeric_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric string"):
            validate_service_response(
                {
                    "is_service": True,
                    "staff_id": 7,
                    "customer_id": 12,
                    "confidence": 0.9,
                },
                expected_staff_id="staff-7",
                expected_customer_id="12",
            )
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_service_response(
                {
                    "is_service": True,
                    "staff_id": 7,
                    "customer_id": 12,
                    "confidence": 0.9,
                },
                expected_staff_id="07",
                expected_customer_id="12",
            )


class ServiceFrameTests(unittest.TestCase):
    def test_joint_frame_selection_is_chronological_and_bounded(self) -> None:
        candidate = _candidate(0.5, start=2, end=11)
        tracks = _tracks(14)
        selected = select_service_frames(
            candidate,
            tracks,
            context_seconds=0,
            frames_per_candidate=6,
        )

        self.assertEqual(selected, (2, 3, 5, 7, 9, 11))
        self.assertEqual(selected, tuple(sorted(selected)))
        self.assertTrue(4 <= len(selected) <= 8)

    def test_only_jointly_visible_frames_are_selected(self) -> None:
        tracks = _tracks(8)
        tracks["tracks"]["12"]["frames"] = [1, 3, 5, 7]
        tracks["tracks"]["12"]["boxes"] = [
            [50.0, 15.0, 70.0, 75.0] for _ in range(4)
        ]
        selected = select_service_frames(
            _candidate(0.5, start=0, end=7),
            tracks,
            context_seconds=0,
        )
        self.assertEqual(selected, (1, 3, 5, 7))

    def test_crop_is_union_with_context_and_contains_annotations(self) -> None:
        frame = np.zeros((100, 100, 3), np.uint8)
        crop = annotate_service_crop(
            frame,
            staff_box=(10, 20, 30, 70),
            customer_box=(40, 15, 60, 65),
            table_box=(5, 60, 70, 90),
            staff_id="7",
            customer_id="12",
            context_margin=0,
        )

        self.assertEqual(crop.shape, (75, 65, 3))
        self.assertGreater(int(crop.sum()), 0)
        self.assertEqual(int(frame.sum()), 0, "annotation must not mutate the source frame")

    def test_one_fixed_roi_is_reused_across_the_sequence(self) -> None:
        frame = np.zeros((100, 120, 3), np.uint8)
        staff = [(5, 20, 25, 70), (45, 20, 65, 70)]
        customer = [(75, 20, 95, 75), (75, 20, 95, 75)]
        bounds = service_crop_bounds(
            frame.shape,
            staff_boxes=staff,
            customer_boxes=customer,
            table_box=None,
            context_margin=0,
        )
        crops = [
            annotate_service_crop(
                frame,
                staff_box=staff[index],
                customer_box=customer[index],
                table_box=None,
                staff_id="7",
                customer_id="12",
                crop_bounds=bounds,
            )
            for index in range(2)
        ]

        self.assertEqual(bounds, (5, 20, 95, 75))
        self.assertEqual(crops[0].shape, crops[1].shape)
        self.assertEqual(crops[0].shape, (55, 90, 3))


class ServiceVerifierTests(unittest.TestCase):
    def test_triage_calls_astra_only_for_inclusive_ambiguous_band(self) -> None:
        requests: list[dict] = []
        loads: list[tuple[int, ...]] = []

        def requester(**kwargs):
            requests.append(kwargs)
            return {
                "is_service": len(requests) == 1,
                "staff_id": 7,
                "customer_id": 12,
                "confidence": 0.6,
            }, "actual-astra"

        def loader(source: Path, indexes: tuple[int, ...]):
            loads.append(indexes)
            return _frames(source, indexes)

        verifier = AstraServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            requester=requester,
            frame_loader=loader,
        )
        # No cache means the source is not fingerprinted; injected I/O keeps this unit test offline.
        decisions = verifier(
            [_candidate(0.29), _candidate(0.3), _candidate(0.8), _candidate(0.81)]
        )

        self.assertEqual(len(requests), 2)
        self.assertEqual(len(loads), 1, "the union of requested frames is decoded once")
        self.assertEqual(
            [decision.source for decision in decisions],
            ["rules_low_confidence", "astra", "astra", "rules_high_confidence"],
        )
        self.assertFalse(decisions[0].is_service)
        self.assertAlmostEqual(decisions[0].confidence, 0.71)
        self.assertTrue(decisions[3].is_service)
        self.assertEqual(requests[0]["schema"]["additionalProperties"], False)
        prompt = requests[0]["messages"][0]["content"][0]["text"]
        self.assertIn("Do not detect people", prompt)
        self.assertIn("fixed inclusive boundaries", prompt)
        self.assertIn("Return only one JSON object", prompt)

    def test_astra_boolean_is_the_decision_even_at_low_model_confidence(self) -> None:
        verifier = AstraServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            requester=lambda **_: {
                "is_service": True,
                "staff_id": 7,
                "customer_id": 12,
                "confidence": 0.01,
            },
            frame_loader=_frames,
        )

        decision = verifier([_candidate(0.5)])[0]

        self.assertTrue(decision.is_service)
        self.assertEqual(decision.confidence, 0.01)
        self.assertEqual(decision.source, "astra")

    def test_insufficient_visual_evidence_keeps_candidate_without_api(self) -> None:
        tracks = _tracks(3)
        verifier = AstraServiceVerifier(
            Path("unused.mp4"),
            tracks,
            requester=lambda **_: self.fail("requester must not be called"),
            frame_loader=lambda *_: self.fail("frame loader must not be called"),
        )

        decision = verifier([_candidate(0.5, start=0, end=2)])[0]

        self.assertTrue(decision.is_service)
        self.assertEqual(decision.confidence, 0.5)
        self.assertEqual(decision.source, "insufficient_visual_evidence")
        self.assertEqual(decision.sampled_frames, (0, 1, 2))

    def test_non_numeric_candidate_id_fails_before_io(self) -> None:
        verifier = AstraServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            requester=lambda **_: self.fail("requester must not be called"),
            frame_loader=lambda *_: self.fail("frame loader must not be called"),
        )
        with self.assertRaisesRegex(ValueError, "numeric string"):
            verifier([_candidate(0.5, staff="staff-7")])

    def test_cache_miss_without_credentials_has_clear_error(self) -> None:
        verifier = AstraServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            frame_loader=lambda *_: self.fail("must fail before decoding"),
        )
        with patch("service_verifier.has_api_key", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "after a cache miss"):
                verifier([_candidate(0.5)])

    def test_cache_roundtrip_and_configuration_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"stable source identity")
            cache = root / "service.json"
            calls: list[float] = []

            def requester(**_kwargs):
                calls.append(1.0)
                return {
                    "is_service": False,
                    "staff_id": 7,
                    "customer_id": 12,
                    "confidence": 0.87,
                }, "actual-astra"

            first = AstraServiceVerifier(
                source,
                _tracks(),
                cache_path=cache,
                model="requested-astra",
                requester=requester,
                frame_loader=_frames,
            )([_candidate(0.5)])
            self.assertFalse(first[0].is_service)
            self.assertEqual(len(calls), 1)
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest"]["astra"]["resolved_model"], "requested-astra")
            self.assertEqual(
                set(payload["results"][0]["response"]),
                {"is_service", "staff_id", "customer_id", "confidence"},
            )
            self.assertEqual(payload["results"][0]["actual_model"], "actual-astra")

            cached = AstraServiceVerifier(
                source,
                _tracks(),
                cache_path=cache,
                model="requested-astra",
                requester=lambda **_: self.fail("cache hit must not call Astra"),
                frame_loader=lambda *_: self.fail("cache hit must not decode video"),
            )([_candidate(0.5)])
            self.assertEqual(cached, first)

            refreshed = AstraServiceVerifier(
                source,
                _tracks(),
                cache_path=cache,
                model="requested-astra",
                rule_high=0.81,
                requester=requester,
                frame_loader=_frames,
            )([_candidate(0.5)])
            self.assertFalse(refreshed[0].is_service)
            self.assertEqual(len(calls), 2, "changed thresholds must invalidate the cache")

    def test_default_model_cache_is_reusable_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"stable source identity")
            cache = root / "service.json"
            with patch(
                "service_verifier.configured_default_model_id",
                return_value="openai/gpt-6-astra",
            ):
                expected = AstraServiceVerifier(
                    source,
                    _tracks(),
                    cache_path=cache,
                    requester=lambda **_: {
                        "is_service": True,
                        "staff_id": 7,
                        "customer_id": 12,
                        "confidence": 0.9,
                    },
                    frame_loader=_frames,
                )([_candidate(0.5)])

            with (
                patch("service_verifier.configured_default_model_id", return_value=None),
                patch("service_verifier.has_api_key", return_value=False),
            ):
                cached = AstraServiceVerifier(
                    source,
                    _tracks(),
                    cache_path=cache,
                    frame_loader=lambda *_: self.fail("cache hit must not decode video"),
                )([_candidate(0.5)])

            self.assertEqual(cached, expected)


if __name__ == "__main__":
    unittest.main()
