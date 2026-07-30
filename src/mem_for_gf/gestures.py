from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class HandGestureResult:
    gesture: str | None
    fingers: Mapping[str, bool]


@dataclass(frozen=True)
class FaceExpressionResult:
    expression: str | None
    mouth_open_ratio: float
    jaw_open_score: float
    tongue_color_ratio: float


def _point_array(landmark: Any) -> np.ndarray:
    return np.array(
        (
            float(getattr(landmark, "x")),
            float(getattr(landmark, "y")),
            float(getattr(landmark, "z", 0.0)),
        ),
        dtype=np.float64,
    )


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _joint_angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    first = a - vertex
    second = c - vertex
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return degrees(acos(float(np.clip(cosine, -1.0, 1.0))))


def _finger_extended(
    points: np.ndarray,
    joints: tuple[int, int, int, int],
    *,
    minimum_angle: float = 145.0,
    distance_factor: float = 1.08,
) -> bool:
    mcp, pip, dip, tip = (points[index] for index in joints)
    pip_angle = _joint_angle(mcp, pip, dip)
    dip_angle = _joint_angle(pip, dip, tip)
    wrist = points[0]
    reaches_outward = _distance(tip, wrist) > (
        _distance(pip, wrist) * distance_factor
    )
    return pip_angle >= minimum_angle and dip_angle >= minimum_angle and reaches_outward


def finger_states(landmarks: Sequence[Any]) -> dict[str, bool]:
    """Classify extended fingers using joint angles and wrist-relative reach."""
    if len(landmarks) < 21:
        raise ValueError("A hand must contain 21 MediaPipe landmarks")
    points = np.stack([_point_array(item) for item in landmarks[:21]])

    states = {
        name: _finger_extended(points, joints)
        for name, joints in FINGER_JOINTS.items()
        if name != "thumb"
    }
    states["thumb"] = _finger_extended(
        points,
        FINGER_JOINTS["thumb"],
        minimum_angle=135.0,
        distance_factor=1.04,
    )
    return {
        "thumb": states["thumb"],
        "index": states["index"],
        "middle": states["middle"],
        "ring": states["ring"],
        "pinky": states["pinky"],
    }


def detect_hand_gesture(landmarks: Sequence[Any]) -> HandGestureResult:
    states = finger_states(landmarks)
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    pinky = states["pinky"]

    if index and middle and not ring and not pinky:
        gesture = "peace"
    elif index and not middle and not ring and not pinky and not states["thumb"]:
        gesture = "one_finger"
    elif all(states.values()):
        gesture = "open_palm"
    else:
        gesture = None
    return HandGestureResult(gesture=gesture, fingers=states)


def _blendshape_score(blendshapes: Iterable[Any], name: str) -> float:
    for category in blendshapes:
        category_name = getattr(
            category,
            "category_name",
            getattr(category, "display_name", ""),
        )
        if category_name == name:
            return float(getattr(category, "score", 0.0))
    return 0.0


def _tongue_color_score(
    frame_bgr: np.ndarray,
    landmarks: Sequence[Any],
) -> float:
    """Estimate visible pink/red tongue area below the lower lip."""
    if frame_bgr.size == 0 or len(landmarks) <= 291:
        return 0.0

    height, width = frame_bgr.shape[:2]

    def pixel(index: int) -> tuple[float, float]:
        item = landmarks[index]
        return float(item.x) * width, float(item.y) * height

    left_x, _ = pixel(61)
    right_x, _ = pixel(291)
    _, lower_lip_y = pixel(17)
    _, chin_y = pixel(152)
    mouth_width = abs(right_x - left_x)
    if mouth_width < 8:
        return 0.0

    x0 = max(0, int(min(left_x, right_x) + 0.12 * mouth_width))
    x1 = min(width, int(max(left_x, right_x) - 0.12 * mouth_width))
    y0 = max(0, int(lower_lip_y + 0.02 * mouth_width))
    y1 = min(
        height,
        int(min(lower_lip_y + 0.52 * mouth_width, chin_y - 0.06 * mouth_width)),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, (0, 55, 45), (16, 255, 255))
    high_red = cv2.inRange(hsv, (160, 45, 45), (179, 255, 255))
    mask = cv2.bitwise_or(low_red, high_red)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest_component / float(mask.shape[0] * mask.shape[1])


def detect_face_expression(
    landmarks: Sequence[Any],
    blendshapes: Iterable[Any],
    frame_bgr: np.ndarray | None,
    *,
    mouth_open_threshold: float,
    jaw_open_threshold: float,
    tongue_color_threshold: float,
) -> FaceExpressionResult:
    if len(landmarks) <= 291:
        raise ValueError("Face landmarks do not contain the required mouth points")

    points = np.stack([_point_array(item) for item in landmarks])
    mouth_width = _distance(points[61], points[291])
    inner_lip_gap = _distance(points[13], points[14])
    mouth_ratio = inner_lip_gap / mouth_width if mouth_width > 1e-9 else 0.0
    jaw_score = _blendshape_score(blendshapes, "jawOpen")
    mouth_open = (
        mouth_ratio >= mouth_open_threshold or jaw_score >= jaw_open_threshold
    )

    tongue_score = (
        _tongue_color_score(frame_bgr, landmarks)
        if mouth_open and frame_bgr is not None
        else 0.0
    )
    if mouth_open and tongue_score >= tongue_color_threshold:
        expression = "tongue_out"
    elif mouth_open:
        expression = "mouth_open"
    else:
        expression = None

    return FaceExpressionResult(
        expression=expression,
        mouth_open_ratio=mouth_ratio,
        jaw_open_score=jaw_score,
        tongue_color_ratio=tongue_score,
    )

