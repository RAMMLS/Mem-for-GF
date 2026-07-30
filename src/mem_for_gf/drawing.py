from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .gestures import FaceExpressionResult


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

FACE_TESSELATION = tuple(
    (connection.start, connection.end)
    for connection in (
        mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    )
)
FACE_CONTOURS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
)
FACE_TESSELATION_INDEX = np.asarray(FACE_TESSELATION, dtype=np.int32)
FACE_CONTOUR_INDEX = np.asarray(FACE_CONTOURS, dtype=np.int32)


def _pixels(
    landmarks: Sequence[Any],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    return [
        (
            int(np.clip(float(item.x), 0.0, 1.0) * (width - 1)),
            int(np.clip(float(item.y), 0.0, 1.0) * (height - 1)),
        )
        for item in landmarks
    ]


def draw_hands(
    frame: np.ndarray,
    hands: Sequence[Sequence[Any]],
    handedness: Sequence[Sequence[Any]],
) -> None:
    height, width = frame.shape[:2]
    for hand_index, landmarks in enumerate(hands):
        points = _pixels(landmarks, width, height)
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (80, 230, 130),
                2,
                cv2.LINE_AA,
            )
        for point in points:
            cv2.circle(frame, point, 3, (20, 255, 255), -1, cv2.LINE_AA)

        if hand_index < len(handedness) and handedness[hand_index]:
            category = handedness[hand_index][0]
            name = str(getattr(category, "category_name", "Hand"))
            score = float(getattr(category, "score", 0.0))
            x, y = points[0]
            cv2.putText(
                frame,
                f"{name} {score:.0%}",
                (max(4, x - 30), max(22, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 255, 255),
                2,
                cv2.LINE_AA,
            )


def draw_face_mesh(frame: np.ndarray, faces: Sequence[Sequence[Any]]) -> None:
    height, width = frame.shape[:2]
    for landmarks in faces:
        points = _pixels(landmarks, width, height)
        point_array = np.asarray(points, dtype=np.int32)
        mesh_layer = frame.copy()
        mesh_segments = point_array[FACE_TESSELATION_INDEX]
        cv2.polylines(
            mesh_layer,
            mesh_segments,
            False,
            (210, 130, 60),
            1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(mesh_layer, 0.38, frame, 0.62, 0, frame)
        contour_segments = point_array[FACE_CONTOUR_INDEX]
        cv2.polylines(
            frame,
            contour_segments,
            False,
            (255, 190, 80),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float,
    detections: Iterable[str],
    active_trigger: str | None,
    labels: dict[str, str],
    face_result: FaceExpressionResult | None,
    show_landmarks: bool,
    show_memes: bool,
) -> None:
    detected = list(dict.fromkeys(detections))
    panel_width = min(frame.shape[1] - 20, 510)
    panel_height = 132 if face_result is not None else 108
    layer = frame.copy()
    cv2.rectangle(layer, (10, 10), (10 + panel_width, 10 + panel_height), (8, 12, 22), -1)
    cv2.addWeighted(layer, 0.72, frame, 0.28, 0, frame)

    active_label = labels.get(active_trigger or "", active_trigger or "none")
    detection_label = ", ".join(labels.get(item, item) for item in detected) or "none"
    lines = [
        f"FPS: {fps:5.1f} | Trigger: {active_label}",
        f"Detected: {detection_label}",
        f"L: landmarks {'ON' if show_landmarks else 'OFF'} | "
        f"M: memes {'ON' if show_memes else 'OFF'} | Q/ESC: quit",
    ]
    if face_result is not None:
        lines.append(
            "Mouth "
            f"{face_result.mouth_open_ratio:.2f} | "
            f"Jaw {face_result.jaw_open_score:.2f} | "
            f"Tongue {face_result.tongue_color_ratio:.2f}"
        )
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (22, 38 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (238, 243, 250),
            1,
            cv2.LINE_AA,
        )


def draw_no_detection_hint(frame: np.ndarray) -> None:
    text = "Show a hand or face to the camera"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = max(10, (frame.shape[1] - text_size[0]) // 2)
    y = frame.shape[0] - 28
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
