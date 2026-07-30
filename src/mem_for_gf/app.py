from __future__ import annotations

import argparse
import logging
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import AppSettings, load_settings
from .drawing import (
    draw_face_mesh,
    draw_hands,
    draw_hud,
    draw_no_detection_hint,
)
from .gestures import (
    FaceExpressionResult,
    detect_face_expression,
    detect_hand_gesture,
)
from .memes import MemeLibrary
from .models import ensure_models
from .triggers import TriggerController
from .vision import VisionTracker


LOGGER = logging.getLogger(__name__)
WINDOW_TITLE = "Mem-for-GF | Real-time gestures and memes"


@dataclass(frozen=True)
class RunStatistics:
    frames: int
    hand_frames: int
    face_frames: int
    trigger_frames: dict[str, int]
    elapsed_seconds: float

    @property
    def average_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames / self.elapsed_seconds


def _open_capture(
    camera_index: int | None,
    video_path: Path | None,
    settings: AppSettings,
) -> cv2.VideoCapture:
    if video_path is not None:
        capture = cv2.VideoCapture(str(video_path))
        description = str(video_path)
    else:
        index = camera_index if camera_index is not None else 0
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        description = f"camera {index}"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.display.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.display.capture_height)
        capture.set(cv2.CAP_PROP_FPS, settings.display.capture_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        if video_path is not None:
            raise RuntimeError(f"Unable to open video file: {description}")
        raise RuntimeError(
            f"Unable to open {description}. On macOS, allow camera access for "
            "Codex/Terminal in System Settings > Privacy & Security > Camera."
        )
    return capture


def _create_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")
    return writer


def _timestamp_ms(
    *,
    frame_number: int,
    source_fps: float,
    is_video: bool,
    previous: int,
) -> int:
    if is_video:
        current = int(round((frame_number - 1) * 1000.0 / max(source_fps, 1.0)))
    else:
        current = time.monotonic_ns() // 1_000_000
    return max(current, previous + 1)


def run_application(
    *,
    project_root: Path,
    config_path: Path,
    model_dir: Path,
    camera_index: int | None,
    video_path: Path | None,
    mirror: bool,
    display: bool,
    output_path: Path | None,
    max_frames: int | None,
) -> RunStatistics:
    settings = load_settings(config_path, project_root)
    model_paths = ensure_models(model_dir)
    memes = MemeLibrary(settings.meme_paths, settings.trigger_labels)
    trigger_controller = TriggerController(
        settings.trigger_priority,
        settings.detection.trigger_confirmation_frames,
        settings.detection.trigger_hold_seconds,
    )
    capture = _open_capture(camera_index, video_path, settings)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 1.0 or source_fps > 240.0:
        source_fps = float(settings.display.capture_fps)

    writer: cv2.VideoWriter | None = None
    if display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    show_landmarks = settings.display.show_landmarks
    show_memes = True
    render_output = display or output_path is not None
    frame_count = 0
    hand_frames = 0
    face_frames = 0
    trigger_frames: Counter[str] = Counter()
    last_timestamp = -1
    smoothed_fps = 0.0
    previous_loop_time = time.perf_counter()
    started = previous_loop_time

    try:
        with VisionTracker(
            model_paths["hand_landmarker.task"],
            model_paths["face_landmarker.task"],
            settings.detection,
        ) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if video_path is not None:
                        break
                    raise RuntimeError("The camera opened but did not return a frame")

                frame_count += 1
                if mirror:
                    frame = cv2.flip(frame, 1)

                timestamp = _timestamp_ms(
                    frame_number=frame_count,
                    source_fps=source_fps,
                    is_video=video_path is not None,
                    previous=last_timestamp,
                )
                last_timestamp = timestamp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tracking = tracker.process(rgb_frame, timestamp)

                detections: list[str] = []
                if tracking.hand_landmarks:
                    hand_frames += 1
                for landmarks in tracking.hand_landmarks:
                    hand_result = detect_hand_gesture(landmarks)
                    if hand_result.gesture is not None:
                        detections.append(hand_result.gesture)

                face_result: FaceExpressionResult | None = None
                if tracking.face_landmarks:
                    face_frames += 1
                    face_landmarks = tracking.face_landmarks[0]
                    blendshapes = (
                        tracking.face_blendshapes[0]
                        if tracking.face_blendshapes
                        else []
                    )
                    face_result = detect_face_expression(
                        face_landmarks,
                        blendshapes,
                        frame,
                        mouth_open_threshold=settings.detection.mouth_open_ratio,
                        jaw_open_threshold=settings.detection.jaw_open_score,
                        tongue_color_threshold=settings.detection.tongue_color_ratio,
                    )
                    if face_result.expression is not None:
                        detections.append(face_result.expression)

                now = time.monotonic()
                active_trigger = trigger_controller.update(detections, now)
                if active_trigger is not None:
                    trigger_frames[active_trigger] += 1

                current_loop_time = time.perf_counter()
                delta = current_loop_time - previous_loop_time
                previous_loop_time = current_loop_time
                instant_fps = 1.0 / delta if delta > 0 else 0.0
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps == 0.0
                    else 0.9 * smoothed_fps + 0.1 * instant_fps
                )
                if render_output:
                    if show_landmarks:
                        draw_face_mesh(frame, tracking.face_landmarks)
                        draw_hands(
                            frame,
                            tracking.hand_landmarks,
                            tracking.handedness,
                        )
                    if show_memes and active_trigger is not None:
                        memes.overlay(
                            frame,
                            active_trigger,
                            width_ratio=settings.display.meme_width_ratio,
                        )
                    draw_hud(
                        frame,
                        fps=smoothed_fps,
                        detections=detections,
                        active_trigger=active_trigger,
                        labels=settings.trigger_labels,
                        face_result=face_result,
                        show_landmarks=show_landmarks,
                        show_memes=show_memes,
                    )
                    if not tracking.hand_landmarks and not tracking.face_landmarks:
                        draw_no_detection_hint(frame)

                if output_path is not None:
                    if writer is None:
                        writer = _create_writer(
                            output_path,
                            frame.shape[1],
                            frame.shape[0],
                            source_fps,
                        )
                    writer.write(frame)

                if display:
                    cv2.imshow(WINDOW_TITLE, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("l"):
                        show_landmarks = not show_landmarks
                    elif key == ord("m"):
                        show_memes = not show_memes
                    try:
                        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break

                if max_frames is not None and frame_count >= max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    elapsed = time.perf_counter() - started
    return RunStatistics(
        frames=frame_count,
        hand_frames=hand_frames,
        face_frames=face_frames,
        trigger_frames=dict(trigger_frames),
        elapsed_seconds=elapsed,
    )


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time MediaPipe hand/face tracking with gesture-triggered memes."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Use a video file instead")
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror frames (enabled by default for a camera)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process without opening the OpenCV window",
    )
    parser.add_argument("--output", type=Path, help="Write annotated MP4 output")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames (useful for diagnostics)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config.json",
        help="Application JSON configuration",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=project_root / "models",
        help="MediaPipe model directory",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def cli(project_root: Path | None = None) -> int:
    project_root = (project_root or Path.cwd()).resolve()
    parser = build_parser(project_root)
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    video_path = args.video.resolve() if args.video else None
    camera_index = args.camera if args.camera is not None else (None if video_path else 0)
    mirror = args.mirror if args.mirror is not None else video_path is None
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.headless and video_path is None and args.max_frames is None:
        parser.error("--headless camera mode requires --max-frames")

    LOGGER.info(
        "Starting on %s (%s), source=%s",
        platform.system(),
        platform.machine(),
        video_path if video_path else f"camera {camera_index}",
    )
    try:
        stats = run_application(
            project_root=project_root,
            config_path=args.config.resolve(),
            model_dir=args.models.resolve(),
            camera_index=camera_index,
            video_path=video_path,
            mirror=mirror,
            display=not args.headless,
            output_path=args.output.resolve() if args.output else None,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Finished: frames=%d, average_fps=%.1f, hand_frames=%d, "
        "face_frames=%d, triggers=%s",
        stats.frames,
        stats.average_fps,
        stats.hand_frames,
        stats.face_frames,
        stats.trigger_frames,
    )
    return 0
