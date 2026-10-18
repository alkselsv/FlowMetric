from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from events import ServiceCandidate, ServiceInterval
from local_service_verifier import (
    LOCAL_FEATURE_NAMES,
    LocalServiceMLP,
    LocalServiceVerifier,
    ObjectDetection,
    PoseDetection,
    extract_local_service_features,
    select_local_service_frames,
)
from train_service_mlp import train_mlp

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_MODEL = ROOT / "config" / "service_mlp_bootstrap.json"


def _candidate(confidence: float, *, start: int = 2, end: int = 9) -> ServiceCandidate:
    return ServiceCandidate(
        ServiceInterval(
            staff="7",
            customer="12",
            start_frame=start,
            end_frame=end,
            left_censored=False,
            right_censored=False,
            confidence=confidence,
        ),
        table_box=(40.0, 60.0, 90.0, 90.0),
    )


def _tracks(count: int = 14) -> dict:
    frames = list(range(count))
    return {
        "fps": 10.0,
        "width": 100,
        "height": 100,
        "n_frames": count,
        "tracks": {
            "7": {
                "frames": frames,
                "boxes": [[10.0, 10.0, 30.0, 80.0] for _ in frames],
            },
            "12": {
                "frames": frames,
                "boxes": [[60.0, 10.0, 80.0, 80.0] for _ in frames],
            },
        },
    }


def _pose(*, positive: bool) -> PoseDetection:
    points = np.zeros((17, 2), dtype=float)
    confidence = np.zeros(17, dtype=float)
    if positive:
        values = {
            0: (27.0, 20.0),
            5: (22.0, 31.0),
            6: (26.0, 31.0),
            9: (44.0, 65.0),
            10: (48.0, 66.0),
            11: (18.0, 56.0),
            12: (22.0, 56.0),
        }
    else:
        values = {
            0: (14.0, 20.0),
            5: (18.0, 31.0),
            6: (22.0, 31.0),
            9: (14.0, 45.0),
            10: (17.0, 46.0),
            11: (18.0, 56.0),
            12: (22.0, 56.0),
        }
    for index, value in values.items():
        points[index] = value
        confidence[index] = 0.95
    return PoseDetection((8.0, 8.0, 50.0, 82.0), points, confidence)


def _frames(_source: Path, indexes: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {index: np.zeros((100, 100, 3), dtype=np.uint8) for index in indexes}


class LocalFeatureTests(unittest.TestCase):
    def test_dense_local_sampling_is_joint_and_bounded(self) -> None:
        tracks = _tracks(30)
        tracks["tracks"]["12"]["frames"] = list(range(0, 30, 2))
        tracks["tracks"]["12"]["boxes"] = [
            [60.0, 10.0, 80.0, 80.0] for _ in tracks["tracks"]["12"]["frames"]
        ]

        selected = select_local_service_frames(
            _candidate(0.5, start=5, end=20),
            tracks,
            sample_fps=5.0,
            context_seconds=0.2,
            max_frames=6,
        )

        self.assertEqual(selected, tuple(sorted(selected)))
        self.assertLessEqual(len(selected), 6)
        self.assertTrue(all(frame % 2 == 0 for frame in selected))
        self.assertGreaterEqual(selected[0], 3)
        self.assertLessEqual(selected[-1], 22)

    def test_pose_hands_objects_and_duration_raise_features(self) -> None:
        frames = (0, 2, 4, 6, 8)
        staff = {frame: (10.0, 10.0, 30.0, 80.0) for frame in frames}
        customer = {frame: (60.0, 10.0, 80.0, 80.0) for frame in frames}
        positive_pose = {frame: [_pose(positive=True)] for frame in frames}
        negative_pose = {frame: [_pose(positive=False)] for frame in frames}
        objects = {
            frame: [ObjectDetection((42.0, 62.0, 49.0, 69.0), "cup", 0.9)]
            for frame in frames
        }

        positive, evidence = extract_local_service_features(
            _candidate(0.5, start=0, end=8),
            frames,
            staff,
            customer,
            (40.0, 60.0, 90.0, 90.0),
            positive_pose,
            objects,
            fps=10.0,
        )
        negative, _ = extract_local_service_features(
            _candidate(0.5, start=0, end=8),
            frames,
            staff,
            customer,
            (40.0, 60.0, 90.0, 90.0),
            negative_pose,
            {frame: [] for frame in frames},
            fps=10.0,
        )

        self.assertEqual(set(positive), set(LOCAL_FEATURE_NAMES))
        self.assertEqual(positive["pose_coverage"], 1.0)
        self.assertGreater(positive["hand_table_ratio"], negative["hand_table_ratio"])
        self.assertGreater(positive["object_hand_ratio"], negative["object_hand_ratio"])
        self.assertGreater(positive["contact_duration_ratio"], 0.8)
        self.assertEqual(evidence["objects"], {"cup": 5})


class LocalMLPTests(unittest.TestCase):
    def test_bootstrap_model_is_strict_and_scores_evidence(self) -> None:
        model = LocalServiceMLP.load(BOOTSTRAP_MODEL)
        weak = {name: 0.0 for name in LOCAL_FEATURE_NAMES}
        strong = {name: 0.9 for name in LOCAL_FEATURE_NAMES}

        self.assertLess(model.predict_proba(weak), 0.5)
        self.assertGreater(model.predict_proba(strong), 0.5)

        payload = json.loads(BOOTSTRAP_MODEL.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "malformed"):
            LocalServiceMLP(payload, model_id="bad")

        payload = json.loads(BOOTSTRAP_MODEL.read_text(encoding="utf-8"))
        payload["feature_names"] = 12
        with self.assertRaisesRegex(ValueError, "feature_names"):
            LocalServiceMLP(payload, model_id="bad")

    def test_training_exports_a_runtime_compatible_model(self) -> None:
        negative = np.zeros((4, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)
        positive = np.ones((4, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)
        features = np.vstack([negative, positive])
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        payload = train_mlp(
            features,
            labels,
            hidden=4,
            epochs=150,
            learning_rate=0.03,
            seed=3,
        )
        model = LocalServiceMLP(payload, model_id="trained")
        weak = dict(zip(LOCAL_FEATURE_NAMES, negative[0].tolist()))
        strong = dict(zip(LOCAL_FEATURE_NAMES, positive[0].tolist()))
        self.assertLess(model.predict_proba(weak), model.predict_proba(strong))


class LocalVerifierTests(unittest.TestCase):
    def test_triage_runs_pose_and_objects_only_for_ambiguous_candidates(self) -> None:
        pose_calls: list[int] = []
        object_calls: list[int] = []

        def pose_backend(images: list[np.ndarray]) -> list[list[PoseDetection]]:
            pose_calls.append(len(images))
            return [[_pose(positive=True)] for _ in images]

        def object_backend(images: list[np.ndarray]) -> list[list[ObjectDetection]]:
            object_calls.append(len(images))
            return [
                [ObjectDetection((42.0, 62.0, 49.0, 69.0), "cup", 0.9)]
                for _ in images
            ]

        verifier = LocalServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            classifier_path=BOOTSTRAP_MODEL,
            pose_backend=pose_backend,
            object_backend=object_backend,
            frame_loader=_frames,
        )
        decisions = verifier([_candidate(0.29), _candidate(0.5), _candidate(0.81)])

        self.assertEqual(
            [decision.source for decision in decisions],
            ["rules_low_confidence", "local_mlp", "rules_high_confidence"],
        )
        self.assertEqual(len(pose_calls), 1)
        self.assertEqual(pose_calls, object_calls)
        self.assertGreaterEqual(pose_calls[0], 3)
        self.assertIsNotNone(decisions[1].details)
        self.assertIn("features", decisions[1].details)
        self.assertEqual(decisions[1].staff_id, "7")
        self.assertEqual(decisions[1].customer_id, "12")

    def test_missing_pose_preserves_candidate_as_insufficient_evidence(self) -> None:
        verifier = LocalServiceVerifier(
            Path("unused.mp4"),
            _tracks(),
            classifier_path=BOOTSTRAP_MODEL,
            pose_backend=lambda images: [[] for _ in images],
            object_backend=lambda images: [[] for _ in images],
            frame_loader=_frames,
        )

        decision = verifier([_candidate(0.5)])[0]

        self.assertTrue(decision.is_service)
        self.assertEqual(decision.source, "insufficient_pose_evidence")
        self.assertEqual(decision.details["evidence"]["matched_pose_frames"], 0)


if __name__ == "__main__":
    unittest.main()
