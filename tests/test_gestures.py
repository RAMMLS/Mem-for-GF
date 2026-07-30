from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mem_for_gf.gestures import detect_face_expression, detect_hand_gesture


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Category:
    category_name: str
    score: float


def _hand(extended: set[str]) -> list[Landmark]:
    points = [Landmark(0.5, 0.9) for _ in range(21)]
    points[0] = Landmark(0.5, 0.88)

    finger_data = {
        "index": (5, 0.38),
        "middle": (9, 0.47),
        "ring": (13, 0.56),
        "pinky": (17, 0.65),
    }
    for name, (start, x) in finger_data.items():
        points[start] = Landmark(x, 0.67)
        if name in extended:
            points[start + 1] = Landmark(x, 0.52)
            points[start + 2] = Landmark(x, 0.37)
            points[start + 3] = Landmark(x, 0.22)
        else:
            points[start + 1] = Landmark(x, 0.57)
            points[start + 2] = Landmark(x + 0.07, 0.61)
            points[start + 3] = Landmark(x + 0.01, 0.66)

    points[1] = Landmark(0.42, 0.75)
    points[2] = Landmark(0.34, 0.69)
    if "thumb" in extended:
        points[3] = Landmark(0.23, 0.63)
        points[4] = Landmark(0.11, 0.57)
    else:
        points[3] = Landmark(0.38, 0.65)
        points[4] = Landmark(0.43, 0.70)
    return points


def _face(*, open_mouth: bool) -> list[Landmark]:
    points = [Landmark(0.5, 0.5) for _ in range(478)]
    points[61] = Landmark(0.35, 0.50)
    points[291] = Landmark(0.65, 0.50)
    points[13] = Landmark(0.50, 0.48 if open_mouth else 0.499)
    points[14] = Landmark(0.50, 0.56 if open_mouth else 0.501)
    points[17] = Landmark(0.50, 0.58)
    points[152] = Landmark(0.50, 0.88)
    return points


def test_detects_peace() -> None:
    result = detect_hand_gesture(_hand({"index", "middle"}))
    assert result.gesture == "peace"


def test_detects_one_finger() -> None:
    result = detect_hand_gesture(_hand({"index"}))
    assert result.gesture == "one_finger"


def test_detects_open_palm() -> None:
    result = detect_hand_gesture(
        _hand({"thumb", "index", "middle", "ring", "pinky"})
    )
    assert result.gesture == "open_palm"


def test_rejects_unmapped_hand_pose() -> None:
    result = detect_hand_gesture(_hand({"thumb", "pinky"}))
    assert result.gesture is None


def test_detects_open_mouth_from_geometry() -> None:
    result = detect_face_expression(
        _face(open_mouth=True),
        [],
        np.zeros((400, 400, 3), dtype=np.uint8),
        mouth_open_threshold=0.1,
        jaw_open_threshold=0.4,
        tongue_color_threshold=0.1,
    )
    assert result.expression == "mouth_open"
    assert result.mouth_open_ratio > 0.1


def test_detects_open_mouth_from_blendshape() -> None:
    result = detect_face_expression(
        _face(open_mouth=False),
        [Category("jawOpen", 0.8)],
        np.zeros((400, 400, 3), dtype=np.uint8),
        mouth_open_threshold=0.1,
        jaw_open_threshold=0.4,
        tongue_color_threshold=0.1,
    )
    assert result.expression == "mouth_open"
    assert result.jaw_open_score == 0.8


def test_detects_tongue_color_below_lower_lip() -> None:
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[238:282, 175:225] = (90, 90, 235)
    result = detect_face_expression(
        _face(open_mouth=True),
        [Category("jawOpen", 0.8)],
        frame,
        mouth_open_threshold=0.1,
        jaw_open_threshold=0.4,
        tongue_color_threshold=0.1,
    )
    assert result.expression == "tongue_out"
    assert result.tongue_color_ratio >= 0.1


def test_closed_mouth_has_no_expression() -> None:
    result = detect_face_expression(
        _face(open_mouth=False),
        [],
        np.zeros((400, 400, 3), dtype=np.uint8),
        mouth_open_threshold=0.1,
        jaw_open_threshold=0.4,
        tongue_color_threshold=0.1,
    )
    assert result.expression is None

