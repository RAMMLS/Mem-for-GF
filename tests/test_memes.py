from pathlib import Path

import numpy as np

from mem_for_gf.memes import MemeLibrary, make_placeholder


def test_placeholder_has_expected_shape() -> None:
    image = make_placeholder("peace", "PEACE")
    assert image.shape == (440, 720, 3)
    assert image.dtype == np.uint8


def test_missing_meme_uses_fallback_and_overlays(tmp_path: Path) -> None:
    library = MemeLibrary({"peace": tmp_path / "missing.png"}, {"peace": "PEACE"})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    library.overlay(frame, "peace", width_ratio=0.3)
    assert np.count_nonzero(frame) > 0

