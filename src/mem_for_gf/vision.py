from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np

from .config import DetectionSettings


@dataclass(frozen=True)
class TrackingResults:
    hand_landmarks: list[list[Any]]
    handedness: list[list[Any]]
    face_landmarks: list[list[Any]]
    face_blendshapes: list[list[Any]]


class VisionTracker:
    """Own the two MediaPipe Tasks graphs used by the application."""

    def __init__(
        self,
        hand_model: Path,
        face_model: Path,
        settings: DetectionSettings,
    ) -> None:
        vision = mp.tasks.vision
        running_mode = vision.RunningMode.VIDEO

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=running_mode,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.min_hand_detection_confidence,
            min_hand_presence_confidence=settings.min_hand_presence_confidence,
            min_tracking_confidence=settings.min_hand_tracking_confidence,
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model)),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=settings.min_face_detection_confidence,
            min_face_presence_confidence=settings.min_face_presence_confidence,
            min_tracking_confidence=settings.min_face_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )

        self._hands = vision.HandLandmarker.create_from_options(hand_options)
        try:
            self._face = vision.FaceLandmarker.create_from_options(face_options)
        except Exception:
            self._hands.close()
            raise

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int) -> TrackingResults:
        media_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        hand_result = self._hands.detect_for_video(media_image, timestamp_ms)
        face_result = self._face.detect_for_video(media_image, timestamp_ms)
        return TrackingResults(
            hand_landmarks=list(hand_result.hand_landmarks),
            handedness=list(hand_result.handedness),
            face_landmarks=list(face_result.face_landmarks),
            face_blendshapes=list(face_result.face_blendshapes),
        )

    def close(self) -> None:
        self._hands.close()
        self._face.close()

    def __enter__(self) -> VisionTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

