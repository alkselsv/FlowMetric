"""Frame-local person detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from flowmetric.paths import weights_path


def apply_tracker_appearance_bonus(
    motion_distances: np.ndarray,
    embedding_distances: np.ndarray,
    *,
    proximity_mask: np.ndarray,
    max_embedding_distance: float,
    max_bonus: float = 0.01,
) -> np.ndarray:
    """Use appearance as a bounded tie-break without replacing motion cost."""
    result = np.asarray(motion_distances, dtype=float).copy()
    embeddings = np.asarray(embedding_distances, dtype=float)
    valid = proximity_mask & (embeddings <= max_embedding_distance)
    strength = np.clip(
        1.0 - embeddings / max(max_embedding_distance, 1e-12),
        0.0,
        1.0,
    )
    result[valid] = np.maximum(0.0, result[valid] - max_bonus * strength[valid])
    return result


@dataclass
class LocalPass:
    fps: float
    width: int
    height: int
    total: int
    dets: dict[int, list[tuple[int, np.ndarray, float]]] = field(default_factory=dict)
    sample_idx: list[int] = field(default_factory=list)
    sample_frames: list[np.ndarray] = field(default_factory=list)
    appearance_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] = field(
        default_factory=dict
    )
    part_appearance_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] = field(
        default_factory=dict
    )
    pose_attribute_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] = field(
        default_factory=dict
    )
    segmented_attribute_embeddings: dict[int, list[tuple[int, np.ndarray, float]]] = field(
        default_factory=dict
    )
    _pair_constraints: tuple[set[tuple[int, int]], set[tuple[int, int]]] | None = field(
        default=None, repr=False
    )


def local_track(
    source: Path,
    *,
    model_name: str,
    conf: float,
    iou: float,
    tracker: str,
    sample_fps: float,
    imgsz: int = 1280,
    device: str = "cpu",
    appearance_model: Path | None = None,
    appearance_interval_s: float = 0.5,
    part_aware_appearance: bool = False,
    ensemble_appearance: bool = False,
    tracker_reid_model: Path | None = None,
) -> LocalPass:
    """Run the local detector/tracker and retain optional ReID sample frames."""
    model = YOLO(weights_path(model_name))
    appearance_encoder = None
    if appearance_model is not None or tracker_reid_model is not None:
        from flowmetric.identity.appearance import LocalAppearanceEncoder

        appearance_encoder = LocalAppearanceEncoder(
            appearance_model or tracker_reid_model,
            device=device,
        )
    original_build_encoder = None
    original_get_dists = None
    if tracker_reid_model is not None:
        from flowmetric.identity.appearance import TrackerAppearanceEncoder
        from ultralytics.trackers import bot_sort

        original_build_encoder = bot_sort.build_encoder
        original_get_dists = bot_sort.BOTSORT.get_dists
        tracker_encoder = TrackerAppearanceEncoder(appearance_encoder)

        def build_encoder(with_reid, encoder_model, encoder_device=None):
            if with_reid and encoder_model == "flowmetric_osnet":
                return tracker_encoder
            return original_build_encoder(with_reid, encoder_model, encoder_device)

        bot_sort.build_encoder = build_encoder

        def get_dists(tracker, tracks, detections):
            from ultralytics.trackers.utils import matching

            motion = matching.iou_distance(tracks, detections)
            proximity = motion <= (1.0 - tracker.proximity_thresh)
            if tracker.args.fuse_score:
                motion = matching.fuse_score(motion, detections)
            if tracker.encoder is None:
                return motion
            embeddings = matching.embedding_distance(tracks, detections) / 2.0
            return apply_tracker_appearance_bonus(
                motion,
                embeddings,
                proximity_mask=proximity,
                max_embedding_distance=1.0 - tracker.appearance_thresh,
            )

        bot_sort.BOTSORT.get_dists = get_dists
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    local_pass = LocalPass(
        fps=fps,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 0

    frame_index = 0
    last_appearance_frame: dict[int, int] = {}
    appearance_interval = max(1, int(round(fps * appearance_interval_s)))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(
                frame,
                persist=True,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=[0],
                tracker=tracker,
                device=device,
                verbose=False,
            )[0]
            detections: list[tuple[int, np.ndarray, float]] = []
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id, score in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.id.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                ):
                    detections.append(
                        (int(track_id), box.astype(float), float(score))
                    )
            local_pass.dets[frame_index] = detections

            if appearance_encoder is not None and detections:
                selected = [
                    detection
                    for detection in detections
                    if frame_index - last_appearance_frame.get(
                        detection[0], -appearance_interval
                    )
                    >= appearance_interval
                ]
                if selected:
                    selected_boxes = np.stack(
                        [detection[1] for detection in selected]
                    )
                    visibility_quality = np.ones(len(selected), dtype=float)
                    part_embeddings = None
                    if part_aware_appearance or ensemble_appearance:
                        full_embeddings, part_embeddings, visibility_quality = (
                            appearance_encoder.encode_full_and_part_aware(
                                frame,
                                selected_boxes,
                                [
                                    [
                                        other_box
                                        for other_id, other_box, _ in detections
                                        if other_id != track_id
                                    ]
                                    for track_id, _, _ in selected
                                ],
                            )
                        )
                        embeddings = (
                            full_embeddings if ensemble_appearance else part_embeddings
                        )
                    else:
                        embeddings = appearance_encoder.encode(
                            frame, selected_boxes
                        )
                    secondary_embeddings = (
                        part_embeddings
                        if ensemble_appearance and part_embeddings is not None
                        else [None] * len(selected)
                    )
                    frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
                    for (
                        (track_id, box, confidence),
                        embedding,
                        secondary_embedding,
                        visibility,
                    ) in zip(
                        selected,
                        embeddings,
                        secondary_embeddings,
                        visibility_quality,
                    ):
                        box_area = max(
                            0.0,
                            float((box[2] - box[0]) * (box[3] - box[1])),
                        )
                        size_quality = min(
                            np.sqrt(box_area / frame_area) / 0.2,
                            1.0,
                        )
                        primary_visibility = 1.0 if ensemble_appearance else visibility
                        quality = float(
                            confidence
                            * (0.5 + 0.5 * size_quality)
                            * (0.5 + 0.5 * primary_visibility)
                        )
                        local_pass.appearance_embeddings.setdefault(
                            track_id, []
                        ).append((frame_index, embedding, quality))
                        if secondary_embedding is not None:
                            secondary_quality = float(
                                confidence
                                * (0.5 + 0.5 * size_quality)
                                * (0.5 + 0.5 * visibility)
                            )
                            local_pass.part_appearance_embeddings.setdefault(
                                track_id, []
                            ).append(
                                (frame_index, secondary_embedding, secondary_quality)
                            )
                        last_appearance_frame[track_id] = frame_index

            if step and frame_index % step == 0:
                local_pass.sample_idx.append(frame_index)
                local_pass.sample_frames.append(frame.copy())
            frame_index += 1
            if frame_index % 50 == 0:
                print(f"  local: {frame_index}/{local_pass.total} frames")
    finally:
        capture.release()
        if original_build_encoder is not None:
            from ultralytics.trackers import bot_sort

            bot_sort.build_encoder = original_build_encoder
            bot_sort.BOTSORT.get_dists = original_get_dists

    local_pass.total = frame_index
    return local_pass
