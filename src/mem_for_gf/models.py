from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str


MODEL_SPECS = (
    ModelSpec(
        filename="hand_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    ModelSpec(
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: ModelSpec, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    LOGGER.info("Downloading MediaPipe model: %s", spec.filename)
    try:
        with urllib.request.urlopen(spec.url, timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if _sha256(temporary) != spec.sha256:
            raise RuntimeError(f"Checksum mismatch for downloaded {spec.filename}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_models(model_dir: Path) -> dict[str, Path]:
    """Return verified model paths, downloading official assets when needed."""
    model_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for spec in MODEL_SPECS:
        destination = model_dir / spec.filename
        if not destination.exists() or _sha256(destination) != spec.sha256:
            _download(spec, destination)
        result[spec.filename] = destination
    return result

