from __future__ import annotations

import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mem_for_gf.config import load_settings  # noqa: E402
from mem_for_gf.memes import make_placeholder  # noqa: E402


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "config.json", PROJECT_ROOT)
    for trigger, path in settings.meme_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        image = make_placeholder(
            trigger,
            settings.trigger_labels.get(trigger, trigger.upper()),
        )
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Unable to write placeholder: {path}")
        print(f"Created {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

