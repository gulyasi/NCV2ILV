import argparse
import json
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


DEFAULT_CHARS = string.ascii_letters + string.digits + " .,!?\"'()-#$:;"
DEFAULT_FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/truetype/freefont"),
]
CANVAS_SIZE = 180
TARGET_HEIGHT = 78
SYNTHETIC_BASELINE_RATIO = 0.72
PREFERRED_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")


def discover_fonts(font_dirs: list[Path] | None = None) -> list[Path]:
    if PREFERRED_FONT.exists():
        return [PREFERRED_FONT]
    dirs = font_dirs or DEFAULT_FONT_DIRS
    fonts: list[Path] = []
    for font_dir in dirs:
        if not font_dir.exists():
            continue
        fonts.extend(sorted(font_dir.glob("*.ttf")))
        fonts.extend(sorted(font_dir.glob("*.otf")))
    return fonts[:1]


def crop_baseline_zone(image: Image.Image, baseline: int) -> Image.Image | None:
    gray = image.convert("L")
    mask = gray.point(lambda p: 255 if p < 245 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None

    x1, _, x2, _ = bbox
    if x2 - x1 < 2:
        return None

    pad_x = 8
    y1 = max(0, baseline - 72)
    y2 = min(gray.height, baseline + 38)
    crop = gray.crop((max(0, x1 - pad_x), y1, min(gray.width, x2 + pad_x), y2))

    scale = TARGET_HEIGHT / max(crop.height, 1)
    width = max(1, int(crop.width * scale))
    return crop.resize((width, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def render_variant(char: str, font_path: Path, size: int, rng: random.Random) -> Image.Image | None:
    try:
        font = ImageFont.truetype(str(font_path), size)
    except OSError:
        return None

    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(canvas)
    bbox = font.getbbox(char)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        return None

    x = (CANVAS_SIZE - width) // 2 - bbox[0] + rng.randint(-2, 2)
    baseline = 112
    y = baseline - bbox[3] + rng.randint(-2, 2)
    draw.text((x, y), char, font=font, fill=0)

    canvas = canvas.rotate(rng.uniform(-2.5, 2.5), expand=False, fillcolor=255)
    if rng.random() < 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.35)))

    glyph = crop_baseline_zone(canvas, baseline)
    if glyph is None:
        return None

    # Store glyphs in the same convention as the existing extracted masks:
    # white ink on black background. The composer normalizes this on load.
    return ImageOps.invert(glyph)


def generate_synthetic_glyphs(
    output_dir: str = "data/synthetic_glyphs",
    library_path: str = "data/synthetic_glyph_library.json",
    chars: str = DEFAULT_CHARS,
    variants: int = 12,
    seed: int = 1234,
    font_paths: list[str] | None = None,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    fonts = [Path(path) for path in font_paths] if font_paths else discover_fonts()
    if not fonts:
        raise RuntimeError("No .ttf/.otf fonts found. Pass --font one or more times.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    library: dict[str, list[str]] = {char: [] for char in chars}

    for char in chars:
        attempts = 0
        while len(library[char]) < variants and attempts < variants * 10:
            attempts += 1
            font_path = fonts[0] if len(fonts) == 1 else rng.choice(fonts)
            size = rng.randint(82, 96)
            glyph = render_variant(char, font_path, size, rng)
            if glyph is None:
                continue
            index = len(library[char])
            path = out / f"char_{ord(char)}_synthetic_{index:03d}.png"
            glyph.save(path)
            library[char].append(str(path))

    library = {char: paths for char, paths in library.items() if paths}
    Path(library_path).parent.mkdir(parents=True, exist_ok=True)
    with open(library_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=4, ensure_ascii=False)

    print(f"Synthetic glyphs written to {out}")
    print(f"Library written to {library_path}")
    print(f"Unique glyph types: {len(library)}")
    print(f"Total variants: {sum(len(paths) for paths in library.values())}")
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean synthetic character-level glyph dataset.")
    parser.add_argument("-o", "--output-dir", default="data/synthetic_glyphs")
    parser.add_argument("--library", default="data/synthetic_glyph_library.json")
    parser.add_argument("--chars", default=DEFAULT_CHARS)
    parser.add_argument("--variants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--font", action="append", dest="fonts", default=None, help="Optional .ttf/.otf font path; can be repeated")
    args = parser.parse_args()
    generate_synthetic_glyphs(
        output_dir=args.output_dir,
        library_path=args.library,
        chars=args.chars,
        variants=args.variants,
        seed=args.seed,
        font_paths=args.fonts,
    )


if __name__ == "__main__":
    main()
