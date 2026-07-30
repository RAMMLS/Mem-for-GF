from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

FALLBACK_COLORS = {
    "peace": (194, 80, 210),
    "one_finger": (220, 125, 35),
    "open_palm": (70, 180, 90),
    "mouth_open": (65, 80, 225),
    "tongue_out": (150, 60, 235),
}


def make_placeholder(trigger: str, label: str, size: tuple[int, int] = (720, 440)) -> np.ndarray:
    width, height = size
    base_color = np.array(FALLBACK_COLORS.get(trigger, (110, 110, 110)), dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height):
        amount = 0.62 + 0.38 * (y / max(1, height - 1))
        image[y, :] = np.clip(base_color * amount, 0, 255)

    cv2.rectangle(image, (18, 18), (width - 19, height - 19), (245, 245, 245), 5)
    cv2.putText(
        image,
        "MEME PLACEHOLDER",
        (42, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    font_scale = 2.0
    thickness = 5
    while font_scale > 0.8:
        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            thickness,
        )
        if text_size[0] <= width - 80:
            break
        font_scale -= 0.1
    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        thickness,
    )
    origin = ((width - text_size[0]) // 2, (height + text_size[1]) // 2 + 25)
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (20, 20, 25),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


class MemeLibrary:
    def __init__(self, paths: dict[str, Path], labels: dict[str, str]) -> None:
        self._images: dict[str, np.ndarray] = {}
        for trigger, path in paths.items():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                LOGGER.warning("Meme file missing or unreadable, using fallback: %s", path)
                image = make_placeholder(trigger, labels.get(trigger, trigger))
            self._images[trigger] = image

    def overlay(
        self,
        frame: np.ndarray,
        trigger: str,
        *,
        width_ratio: float,
    ) -> None:
        source = self._images.get(trigger)
        if source is None:
            return

        frame_height, frame_width = frame.shape[:2]
        target_width = max(120, int(frame_width * width_ratio))
        scale = target_width / source.shape[1]
        target_height = max(1, int(source.shape[0] * scale))
        maximum_height = int(frame_height * 0.52)
        if target_height > maximum_height:
            scale = maximum_height / source.shape[0]
            target_height = maximum_height
            target_width = max(1, int(source.shape[1] * scale))

        resized = cv2.resize(
            source,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        margin = 18
        x0 = max(0, frame_width - target_width - margin)
        y0 = max(0, frame_height - target_height - margin)
        x1 = x0 + target_width
        y1 = y0 + target_height
        destination = frame[y0:y1, x0:x1]

        if resized.ndim == 3 and resized.shape[2] == 4:
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
            color = resized[:, :, :3].astype(np.float32)
            blended = color * alpha + destination.astype(np.float32) * (1.0 - alpha)
            destination[:] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            destination[:] = resized[:, :, :3]

        cv2.rectangle(
            frame,
            (max(0, x0 - 3), max(0, y0 - 3)),
            (min(frame_width - 1, x1 + 3), min(frame_height - 1, y1 + 3)),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

