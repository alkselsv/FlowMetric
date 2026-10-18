from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import astra
import events
import identity_gallery
import local_service_verifier
import service_verifier
import track
from flowmetric.config import IdentityConfig, PipelineConfig, ServiceConfig, TrackingConfig
from flowmetric.domain import ServiceCandidate, ServiceDecision, ServiceInterval, Track
from flowmetric.events import extract_events
from flowmetric.identity.gallery import build_tracklet_gallery
from flowmetric.infrastructure.astra import AstraResponseError
from flowmetric.pipeline import run_pipeline
from flowmetric.service.astra import AstraServiceVerifier
from flowmetric.service.local import LocalServiceVerifier


class CompatibilityFacadeTests(unittest.TestCase):
    def test_repository_modules_alias_package_implementations(self) -> None:
        self.assertIs(events.extract_events, extract_events)
        self.assertIs(astra.AstraResponseError, AstraResponseError)
        self.assertIs(
            identity_gallery.build_tracklet_gallery,
            build_tracklet_gallery,
        )
        self.assertIs(service_verifier.AstraServiceVerifier, AstraServiceVerifier)
        self.assertIs(local_service_verifier.LocalServiceVerifier, LocalServiceVerifier)
        self.assertIs(track.run_pipeline, run_pipeline)

    def test_domain_contracts_are_provider_independent(self) -> None:
        interval = ServiceInterval("7", "12", 10, 20, False, False, 0.7)
        candidate = ServiceCandidate(interval, None)
        decision = ServiceDecision(True, "7", "12", 0.9, "test")
        trajectory = Track(
            "7",
            "staff",
            np.array([0]),
            np.array([[0, 0, 1, 1]]),
            1.0,
        )
        self.assertEqual(candidate.interval, interval)
        self.assertTrue(decision.is_service)
        self.assertEqual(trajectory.gid, "7")


class PipelineConfigTests(unittest.TestCase):
    def test_grouped_config_maps_to_compatibility_runner(self) -> None:
        config = PipelineConfig(
            source=Path("input.mp4"),
            destination=Path("output.mp4"),
            tracking=TrackingConfig(model_name="detector.pt", confidence=0.12),
            identity=IdentityConfig(enabled=False, minimum_support=3),
            service=ServiceConfig(verifier="local", device="cpu"),
        )
        with patch("flowmetric.pipeline.run_hybrid", return_value={"ok": True}) as runner:
            result = run_pipeline(config)

        self.assertEqual(result, {"ok": True})
        args, kwargs = runner.call_args
        self.assertEqual(args, (Path("input.mp4"), Path("output.mp4")))
        self.assertEqual(kwargs["model_name"], "detector.pt")
        self.assertEqual(kwargs["conf"], 0.12)
        self.assertFalse(kwargs["use_astra"])
        self.assertEqual(kwargs["reid_min_support"], 3)
        self.assertEqual(kwargs["service_verifier"], "local")
        self.assertEqual(kwargs["local_service_device"], "cpu")

    def test_run_pipeline_rejects_untyped_configuration(self) -> None:
        with self.assertRaisesRegex(TypeError, "PipelineConfig"):
            run_pipeline({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
