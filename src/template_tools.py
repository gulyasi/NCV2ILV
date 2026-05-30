import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PAGE_SIZE = (2480, 3508)
MARGIN_X = 140
MARGIN_Y = 160
CELL_W = 210
CELL_H = 185
COLS = 10
DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\"'()-#$:;"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_label_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def template_positions(chars: str = DEFAULT_CHARS):
    for index, char in enumerate(chars):
        row = index // COLS
        col = index % COLS
        x = MARGIN_X + col * CELL_W
        y = MARGIN_Y + row * CELL_H
        yield index, char, x, y


def make_template(output_path: str = "outputs/handwriting_template.pdf", chars: str = DEFAULT_CHARS) -> None:
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_label_font(48)
    label_font = load_label_font(26)
    small_font = load_label_font(22)

    draw.text((MARGIN_X, 60), "Handwriting Template", fill=(0, 0, 0), font=title_font)
    draw.text((MARGIN_X, 115), "Write the shown character once inside each box. Keep it dark and away from the borders.", fill=(80, 80, 80), font=small_font)

    for _, char, x, y in template_positions(chars):
        draw.rectangle((x, y, x + CELL_W - 18, y + CELL_H - 18), outline=(180, 180, 180), width=2)
        draw.text((x + 10, y + 8), repr(char)[1:-1], fill=(150, 150, 150), font=label_font)
        draw.line((x + 18, y + CELL_H - 48, x + CELL_W - 36, y + CELL_H - 48), fill=(225, 225, 225), width=2)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        page.save(path, "PDF", resolution=300.0)
    else:
        page.save(path)
    print(f"Template written to {path}")


def trim_ink(img: Image.Image) -> Image.Image | None:
    gray = img.convert("L")
    # Keep only dark handwriting, not the light box/labels.
    mask = gray.point(lambda p: 255 if p < 120 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if (x2 - x1) < 3 or (y2 - y1) < 8:
        return None
    crop = gray.crop((x1, y1, x2, y2))
    return ImageOps.invert(crop.point(lambda p: 255 if p < 170 else 0))


def extract_template(image_path: str, output_dir: str = "data/template_glyphs", chars: str = DEFAULT_CHARS, prefix: str = "template") -> int:
    page = Image.open(image_path).convert("RGB").resize(PAGE_SIZE)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0

    for index, char, x, y in template_positions(chars):
        # Ignore the printed label area and border so only handwriting remains.
        cell = page.crop((x + 18, y + 50, x + CELL_W - 36, y + CELL_H - 34))
        glyph = trim_ink(cell)
        if glyph is None:
            continue
        glyph.save(out / f"char_{ord(char)}_{prefix}_{index}.png")
        saved += 1

    print(f"Extracted {saved} glyphs into {out}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and extract fixed-grid handwriting template sheets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make", help="Create a printable template sheet")
    make_parser.add_argument("-o", "--output", default="outputs/handwriting_template.pdf")
    make_parser.add_argument("--chars", default=DEFAULT_CHARS)

    extract_parser = subparsers.add_parser("extract", help="Extract glyphs from a filled template image")
    extract_parser.add_argument("image")
    extract_parser.add_argument("-o", "--output-dir", default="data/template_glyphs")
    extract_parser.add_argument("--chars", default=DEFAULT_CHARS)
    extract_parser.add_argument("--prefix", default="template")

    args = parser.parse_args()
    if args.command == "make":
        make_template(args.output, chars=args.chars)
    elif args.command == "extract":
        extract_template(args.image, output_dir=args.output_dir, chars=args.chars, prefix=args.prefix)


if __name__ == "__main__":
    main()
