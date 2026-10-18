from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from flowmetric.device import resolve_device
from flowmetric.tracking.local import local_track


class DeviceResolutionTests(unittest.TestCase):
    @patch("flowmetric.device.torch.backends.mps.is_available", return_value=True)
    @patch("flowmetric.device.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda_over_mps(self, _cuda, _mps) -> None:
        self.assertEqual(resolve_device("auto"), "cuda:0")

    @patch("flowmetric.device.torch.backends.mps.is_available", return_value=True)
    @patch("flowmetric.device.torch.cuda.is_available", return_value=False)
    def test_auto_uses_mps_when_cuda_is_absent(self, _cuda, _mps) -> None:
        self.assertEqual(resolve_device(None), "mps")

    @patch("flowmetric.device.torch.backends.mps.is_available", return_value=False)
    @patch("flowmetric.device.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda, _mps) -> None:
        self.assertEqual(resolve_device("auto"), "cpu")

    @patch("flowmetric.device.torch.backends.mps.is_available", return_value=True)
    @patch("flowmetric.device.torch.backends.mps.is_built", return_value=True)
    def test_explicit_mps_is_validated(self, _built, _available) -> None:
        self.assertEqual(resolve_device("mps:0"), "mps")

    @patch("flowmetric.device.torch.cuda.device_count", return_value=2)
    @patch("flowmetric.device.torch.cuda.is_available", return_value=True)
    def test_cuda_aliases_are_canonical(self, _available, _count) -> None:
        self.assertEqual(resolve_device("cuda"), "cuda:0")
        self.assertEqual(resolve_device("0"), "cuda:0")
        self.assertEqual(resolve_device("cuda:1"), "cuda:1")

    @patch("flowmetric.device.torch.backends.mps.is_available", return_value=False)
    @patch("flowmetric.device.torch.backends.mps.is_built", return_value=True)
    def test_unavailable_mps_fails_instead_of_falling_back(self, _built, _available) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve_device("mps")

    @patch("flowmetric.device.torch.cuda.is_available", return_value=False)
    def test_unavailable_cuda_fails_instead_of_falling_back(self, _available) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA was requested"):
            resolve_device("cuda")

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto, cpu, mps"):
            resolve_device("gpu")


class TrackingDeviceTests(unittest.TestCase):
    @patch("flowmetric.tracking.local.cv2.VideoCapture")
    @patch("flowmetric.tracking.local.YOLO")
    def test_local_tracker_passes_resolved_device_to_ultralytics(
        self, yolo_class, capture_class
    ) -> None:
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        capture = capture_class.return_value
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 24.0,
            cv2.CAP_PROP_FRAME_WIDTH: 12,
            cv2.CAP_PROP_FRAME_HEIGHT: 8,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }[key]
        capture.read.side_effect = [(True, frame), (False, None)]
        model = yolo_class.return_value
        model.track.return_value = [SimpleNamespace(boxes=None)]

        result = local_track(
            Path("input.mp4"),
            model_name="model.pt",
            conf=0.1,
            iou=0.5,
            tracker="tracker.yaml",
            sample_fps=0,
            device="mps",
        )

        self.assertEqual(result.total, 1)
        self.assertEqual(model.track.call_args.kwargs["device"], "mps")
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
