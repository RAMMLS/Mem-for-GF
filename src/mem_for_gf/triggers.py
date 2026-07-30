from __future__ import annotations

from collections.abc import Iterable, Sequence


class TriggerController:
    """Debounce noisy detections and keep a triggered meme visible briefly."""

    def __init__(
        self,
        priority: Sequence[str],
        confirmation_frames: int,
        hold_seconds: float,
    ) -> None:
        self._priority = tuple(priority)
        self._confirmation_frames = confirmation_frames
        self._hold_seconds = hold_seconds
        self._candidate: str | None = None
        self._candidate_frames = 0
        self._active: str | None = None
        self._visible_until = 0.0

    def _choose(self, detections: Iterable[str]) -> str | None:
        available = set(detections)
        return next((name for name in self._priority if name in available), None)

    def update(self, detections: Iterable[str], now: float) -> str | None:
        chosen = self._choose(detections)
        if chosen is None:
            self._candidate = None
            self._candidate_frames = 0
        elif chosen == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = chosen
            self._candidate_frames = 1

        if (
            self._candidate is not None
            and self._candidate_frames >= self._confirmation_frames
        ):
            self._active = self._candidate
            self._visible_until = now + self._hold_seconds

        if self._active is not None and now > self._visible_until:
            self._active = None
        return self._active

