from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionSettings:
    max_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_hand_tracking_confidence: float
    min_face_detection_confidence: float
    min_face_presence_confidence: float
    min_face_tracking_confidence: float
    mouth_open_ratio: float
    jaw_open_score: float
    tongue_color_ratio: float
    trigger_confirmation_frames: int
    trigger_hold_seconds: float


@dataclass(frozen=True)
class DisplaySettings:
    capture_width: int
    capture_height: int
    capture_fps: int
    meme_width_ratio: float
    show_landmarks: bool


@dataclass(frozen=True)
class AppSettings:
    detection: DetectionSettings
    display: DisplaySettings
    meme_paths: dict[str, Path]
    trigger_labels: dict[str, str]
    trigger_priority: tuple[str, ...]


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required configuration key: {key}")
    return data[key]


def load_settings(config_path: Path, project_root: Path) -> AppSettings:
    """Load and validate the JSON application configuration."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    detection_raw = _required(raw, "detection")
    display_raw = _required(raw, "display")
    memes_raw = _required(raw, "memes")

    detection = DetectionSettings(
        max_hands=int(_required(detection_raw, "max_hands")),
        min_hand_detection_confidence=float(
            _required(detection_raw, "min_hand_detection_confidence")
        ),
        min_hand_presence_confidence=float(
            _required(detection_raw, "min_hand_presence_confidence")
        ),
        min_hand_tracking_confidence=float(
            _required(detection_raw, "min_hand_tracking_confidence")
        ),
        min_face_detection_confidence=float(
            _required(detection_raw, "min_face_detection_confidence")
        ),
        min_face_presence_confidence=float(
            _required(detection_raw, "min_face_presence_confidence")
        ),
        min_face_tracking_confidence=float(
            _required(detection_raw, "min_face_tracking_confidence")
        ),
        mouth_open_ratio=float(_required(detection_raw, "mouth_open_ratio")),
        jaw_open_score=float(_required(detection_raw, "jaw_open_score")),
        tongue_color_ratio=float(_required(detection_raw, "tongue_color_ratio")),
        trigger_confirmation_frames=int(
            _required(detection_raw, "trigger_confirmation_frames")
        ),
        trigger_hold_seconds=float(
            _required(detection_raw, "trigger_hold_seconds")
        ),
    )
    display = DisplaySettings(
        capture_width=int(_required(display_raw, "capture_width")),
        capture_height=int(_required(display_raw, "capture_height")),
        capture_fps=int(_required(display_raw, "capture_fps")),
        meme_width_ratio=float(_required(display_raw, "meme_width_ratio")),
        show_landmarks=bool(_required(display_raw, "show_landmarks")),
    )

    if detection.max_hands < 1:
        raise ValueError("detection.max_hands must be at least 1")
    if detection.trigger_confirmation_frames < 1:
        raise ValueError("detection.trigger_confirmation_frames must be at least 1")
    if not 0.1 <= display.meme_width_ratio <= 0.8:
        raise ValueError("display.meme_width_ratio must be between 0.1 and 0.8")

    meme_paths = {
        name: (project_root / str(relative_path)).resolve()
        for name, relative_path in memes_raw.items()
    }
    labels = {
        str(name): str(label)
        for name, label in _required(raw, "trigger_labels").items()
    }
    priority = tuple(str(item) for item in _required(raw, "trigger_priority"))
    unknown = set(priority) - set(meme_paths)
    if unknown:
        raise ValueError(
            "trigger_priority contains triggers without meme files: "
            + ", ".join(sorted(unknown))
        )

    return AppSettings(
        detection=detection,
        display=display,
        meme_paths=meme_paths,
        trigger_labels=labels,
        trigger_priority=priority,
    )

