"""Tracklet-level video appearance embeddings for experimental ReID."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from flowmetric.media.video import load_video_frames
from flowmetric.tracking import LocalPass


SINET_SEQUENCE_LENGTH = 4
SINET_MARS_MIN_SIMILARITY = 0.7018


def sample_tracklet_observations(
    local_pass: LocalPass, *, count: int = SINET_SEQUENCE_LENGTH
) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    """Select an evenly spaced, fixed-length clip for every local tracklet."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    observations: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for frame_index, detections in sorted(local_pass.dets.items()):
        for tracklet_id, box, confidence in detections:
            observations.setdefault(int(tracklet_id), []).append(
                (int(frame_index), np.asarray(box, dtype=float), float(confidence))
            )
    selected: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for tracklet_id, items in observations.items():
        indexes = np.linspace(0, len(items) - 1, count).round().astype(int)
        selected[tracklet_id] = [items[int(index)] for index in indexes]
    return selected


class SINetAppearanceEncoder:
    """Inference adapter for the official four-frame SINet-MARS checkpoint."""

    def __init__(self, weights: Path, *, device: str, batch_size: int = 8) -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise ValueError(f"SINet appearance model does not exist: {weights}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            import torch

            from flowmetric.identity._sinet import SINet
        except ImportError as exc:
            raise RuntimeError(
                "SINet appearance stitching requires torch and torchvision"
            ) from exc

        # The published checkpoint contains legacy NumPy metadata in addition
        # to its state_dict, so PyTorch's restricted weights-only loader cannot
        # read it. Only use the official or otherwise trusted checkpoint here.
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise ValueError("SINet checkpoint must contain a state_dict")
        model = SINet(
            num_classes=625,
            losses=["xent", "htri", "infonce"],
            seq_len=SINET_SEQUENCE_LENGTH,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._batch_size = batch_size
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        )[:, None, None]
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        )[:, None, None]

    def _clip(
        self,
        items: list[tuple[int, np.ndarray, float]],
        frames: dict[int, np.ndarray],
    ):
        crops = []
        for frame_index, box, _ in items:
            frame = frames[frame_index]
            x1, y1, x2, y2 = np.asarray(box).round().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"invalid SINet crop for frame {frame_index}: {(x1, y1, x2, y2)}"
                )
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_CUBIC)
            tensor = self._torch.from_numpy(
                crop.transpose(2, 0, 1).copy()
            ).float() / 255.0
            crops.append((tensor - self._mean) / self._std)
        return self._torch.stack(crops, dim=1)

    def encode(
        self,
        selected: dict[int, list[tuple[int, np.ndarray, float]]],
        frames: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        tracklet_ids = sorted(selected)
        clips = [self._clip(selected[tracklet_id], frames) for tracklet_id in tracklet_ids]
        result: dict[int, np.ndarray] = {}
        with self._torch.inference_mode():
            for offset in range(0, len(clips), self._batch_size):
                batch = self._torch.stack(
                    clips[offset : offset + self._batch_size]
                ).to(self._device)
                features = self._model(batch).mean(-1)
                features = self._torch.nn.functional.normalize(features, dim=1)
                for tracklet_id, feature in zip(
                    tracklet_ids[offset : offset + self._batch_size], features
                ):
                    result[tracklet_id] = feature.detach().cpu().numpy()
        return result


def encode_sinet_tracklets(
    source: Path,
    local_pass: LocalPass,
    weights: Path,
    *,
    device: str,
    batch_size: int = 8,
) -> dict[str, float | int | str]:
    """Populate ``local_pass.appearance_embeddings`` with SINet prototypes."""
    selected = sample_tracklet_observations(local_pass)
    frame_indexes = tuple(
        sorted(
            {
                frame_index
                for items in selected.values()
                for frame_index, _, _ in items
            }
        )
    )
    frames = load_video_frames(source, frame_indexes)
    encoder = SINetAppearanceEncoder(weights, device=device, batch_size=batch_size)
    started = time.perf_counter()
    features = encoder.encode(selected, frames)
    inference_seconds = time.perf_counter() - started
    local_pass.appearance_embeddings = {
        tracklet_id: [
            (
                selected[tracklet_id][len(selected[tracklet_id]) // 2][0],
                feature,
                1.0,
            )
        ]
        for tracklet_id, feature in features.items()
    }
    return {
        "backend": "sinet_mars",
        "sequence_length": SINET_SEQUENCE_LENGTH,
        "embedding_dimension": (
            int(next(iter(features.values())).shape[0]) if features else 0
        ),
        "embedded_tracklets": len(features),
        "decoded_frames": len(frame_indexes),
        "inference_seconds": round(inference_seconds, 4),
        "min_similarity": SINET_MARS_MIN_SIMILARITY,
    }
