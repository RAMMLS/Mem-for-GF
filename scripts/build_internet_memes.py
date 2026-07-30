from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "assets" / "memes"
CANVAS_SIZE = (900, 600)
IMAGE_SIZE = (900, 440)
TOP_BAR_HEIGHT = 80
BOTTOM_BAR_HEIGHT = 80
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


@dataclass(frozen=True)
class MemeSpec:
    filename: str
    source_url: str
    source_page: str
    top_text: str
    bottom_text: str


MEMES = (
    MemeSpec(
        filename="peace.png",
        source_url="https://i.imgflip.com/42et29.jpg",
        source_page=(
            "https://imgflip.com/memetemplate/245914785/"
            "guy-disappearing-peace-sign"
        ),
        top_text="Я НА МИНУТКУ",
        bottom_text="ЧЕРЕЗ ТРИ ЧАСА:",
    ),
    MemeSpec(
        filename="one_finger.png",
        source_url="https://imgflip.com/s/meme/One-Does-Not-Simply.jpg",
        source_page="https://imgflip.com/memetemplate/61579/One-Does-Not-Simply",
        top_text="ЕЩЁ ОДИН МЕМ",
        bottom_text="И ТОЧНО СПАТЬ",
    ),
    MemeSpec(
        filename="open_palm.png",
        source_url="https://i.imgflip.com/19xp8b.jpg",
        source_page=(
            "https://imgflip.com/memetemplate/77155067/"
            "stop-it-Get-some-help"
        ),
        top_text="СТОП",
        bottom_text="Я УЖЕ ОРУ",
    ),
    MemeSpec(
        filename="mouth_open.png",
        source_url="https://i.imgflip.com/2p691z.jpg",
        source_page=(
            "https://imgflip.com/memetemplate/163214423/Suprised-Pikachu"
        ),
        top_text="ВКЛЮЧИЛ КАМЕРУ",
        bottom_text="БЕЗ ПОДГОТОВКИ",
    ),
    MemeSpec(
        filename="tongue_out.png",
        source_url="https://i.imgflip.com/54nugx.png",
        source_page="https://imgflip.com/memetemplate/310161921/Cat-tongue",
        top_text="РЕЖИМ СЕРЬЁЗНОСТИ",
        bottom_text="ОТКЛЮЧЁН",
    ),
)


def _download(url: str) -> Image.Image:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mem-for-GF/1.0 (+local personal meme app)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    return Image.open(BytesIO(payload)).convert("RGB")


def _compose_image(source: Image.Image) -> Image.Image:
    background = ImageOps.fit(source, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=16))
    background = ImageEnhance.Brightness(background).enhance(0.48)

    foreground = ImageOps.contain(
        source,
        (IMAGE_SIZE[0] - 20, IMAGE_SIZE[1] - 12),
        method=Image.Resampling.LANCZOS,
    )
    x = (IMAGE_SIZE[0] - foreground.width) // 2
    y = (IMAGE_SIZE[1] - foreground.height) // 2
    background.paste(foreground, (x, y))
    return background


def _font_for(text: str, maximum_width: int, maximum_size: int) -> ImageFont.FreeTypeFont:
    size = maximum_size
    while size >= 28:
        font = ImageFont.truetype(str(FONT_PATH), size=size)
        box = font.getbbox(text, stroke_width=1)
        if box[2] - box[0] <= maximum_width:
            return font
        size -= 2
    return ImageFont.truetype(str(FONT_PATH), size=28)


def _draw_centered_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    y0: int,
    bar_height: int,
) -> None:
    font = _font_for(text, maximum_width=850, maximum_size=54)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = (CANVAS_SIZE[0] - width) // 2
    y = y0 + (bar_height - height) // 2 - box[1]
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def build(spec: MemeSpec) -> Path:
    source = _download(spec.source_url)
    canvas = Image.new("RGB", CANVAS_SIZE, color=(0, 0, 0))
    canvas.paste(_compose_image(source), (0, TOP_BAR_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    _draw_centered_caption(draw, spec.top_text, 0, TOP_BAR_HEIGHT)
    _draw_centered_caption(
        draw,
        spec.bottom_text,
        TOP_BAR_HEIGHT + IMAGE_SIZE[1],
        BOTTOM_BAR_HEIGHT,
    )

    destination = OUTPUT_DIR / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return destination


def main() -> int:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Required Cyrillic font not found: {FONT_PATH}")
    for spec in MEMES:
        destination = build(spec)
        print(f"Created {destination.relative_to(PROJECT_ROOT)} from {spec.source_page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

