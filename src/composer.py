import argparse
import json
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFont


PAGE_SIZE = (2480, 3508)
MARGIN_X = 150
MARGIN_Y = 150
MAX_X = 2300
MAX_Y = 3350
LINE_HEIGHT = 150
SPACE_WIDTH = 32
GLYPH_GAP = 4
GLYPH_BASELINE = 82
MAX_GLYPH_WIDTH = 160
MAX_GLYPH_HEIGHT = 130
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
FONT_SIZE = 72
FONT_LINE_HEIGHT = 125
FONT_WORD_SPACE = 38


@dataclass
class ComposeReport:
    output_path: str
    missing: dict[str, int] = field(default_factory=dict)
    rendered: int = 0
    lines: int = 1
    overflow: bool = False
    engine: str = "glyph"

    @property
    def coverage(self) -> float:
        total = self.rendered + sum(self.missing.values())
        return 1.0 if total == 0 else self.rendered / total


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def save_page(page: Image.Image, output_name: str) -> None:
    output_path = Path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".pdf":
        page.convert("RGB").save(output_path, "PDF", resolution=300.0)
    else:
        page.save(output_path)


def load_library(path: str = "data/glyph_library.json", writer: str | None = None) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Supports the current flat {char: [paths]} format and a future
    # nested {writer: {char: [paths]}} format.
    if raw and all(isinstance(v, dict) for v in raw.values()):
        if writer is None:
            merged: dict[str, list[str]] = {}
            for writer_lib in raw.values():
                for char, paths in writer_lib.items():
                    merged.setdefault(char, []).extend(paths)
            return merged
        return {char: list(paths) for char, paths in raw.get(writer, {}).items()}

    lib = {char: list(paths) for char, paths in raw.items()}
    if writer is None:
        return lib

    filtered = {
        char: [path for path in paths if writer in Path(path).name]
        for char, paths in lib.items()
    }
    return {char: paths for char, paths in filtered.items() if paths}


def fallback_chars(char: str) -> Iterable[str]:
    substitutions = {
        "’": ["'"],
        "‘": ["'"],
        "“": ['"'],
        "”": ['"'],
        "–": ["-"],
        "—": ["-"],
        "#": ["3"],
        "$": ["S"],
    }
    if char in substitutions:
        yield from substitutions[char]
    if char.lower() != char:
        yield char.lower()
    if char.upper() != char:
        yield char.upper()
    yield "?"


def choose_glyph_path(char: str, lib: dict[str, list[str]]) -> tuple[str | None, str | None]:
    if char in lib and lib[char]:
        return random.choice(lib[char]), char
    for candidate in fallback_chars(char):
        if candidate in lib and lib[candidate]:
            return random.choice(lib[candidate]), candidate
    return None, None


def normalized_size(width: int, height: int) -> tuple[int, int]:
    scale = min(MAX_GLYPH_WIDTH / max(width, 1), MAX_GLYPH_HEIGHT / max(height, 1), 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale))


def prepare_glyph(path: str, jitter: bool = True) -> Image.Image:
    is_synthetic = "synthetic" in Path(path).name
    glyph = Image.open(path).convert("L")

    # Extracted binary crops are often white ink on a black background.
    # Convert to black ink on white, then use the ink as paste mask.
    if glyph.resize((1, 1)).getpixel((0, 0)) < 128:
        glyph = ImageChops.invert(glyph)

    new_size = normalized_size(glyph.width, glyph.height)
    if new_size != glyph.size:
        glyph = glyph.resize(new_size, Image.Resampling.LANCZOS)

    if jitter:
        angle = random.uniform(-0.8, 0.8) if is_synthetic else random.uniform(-1.2, 1.2)
        glyph = glyph.rotate(angle, expand=True, fillcolor=255)

    if is_synthetic:
        glyph.info["baseline_offset"] = int(glyph.height * 0.72)

    return glyph


def glyph_advance(path: str) -> int:
    with Image.open(path) as glyph:
        width, _ = normalized_size(glyph.width, glyph.height)
        return width + GLYPH_GAP


def measure_glyph_token(token: str, lib: dict[str, list[str]]) -> int:
    width = 0
    for char in token:
        if char == " ":
            width += SPACE_WIDTH
            continue
        path, _ = choose_glyph_path(char, lib)
        width += glyph_advance(path) if path else SPACE_WIDTH
    return width


def iter_tokens(text: str) -> Iterable[str]:
    token = ""
    for char in text:
        if char == "\n":
            if token:
                yield token
                token = ""
            yield "\n"
        elif char == " ":
            token += char
            yield token
            token = ""
        else:
            token += char
    if token:
        yield token


def glyph_baseline_offset(glyph: Image.Image) -> int:
    if "baseline_offset" in glyph.info:
        return int(glyph.info["baseline_offset"])
    mask = glyph.point(lambda p: 255 if p < 220 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return glyph.height
    return bbox[3]


def paste_glyph(page: Image.Image, glyph: Image.Image, x: int, baseline_y: int) -> None:
    mask = glyph.point(lambda p: 255 if p < 220 else 0)
    ink = Image.new("L", glyph.size, 0)
    y = baseline_y - glyph_baseline_offset(glyph)
    page.paste(ink, (x, y), mask)


def compose_glyph(
    text: str,
    output_name: str = "handwritten_result.pdf",
    library_path: str = "data/glyph_library.json",
    writer: str | None = None,
    seed: int | None = None,
    jitter: bool = True,
) -> ComposeReport:
    if seed is not None:
        random.seed(seed)

    text = normalize_text(text)
    lib = load_library(library_path, writer=writer)
    if not lib:
        raise ValueError(f"No glyphs available in {library_path!r} for writer={writer!r}")

    page = Image.new("L", PAGE_SIZE, 255)
    x, baseline_y = MARGIN_X, MARGIN_Y + GLYPH_BASELINE
    report = ComposeReport(output_path=output_name, engine="glyph")

    for token in iter_tokens(text):
        if token == "\n":
            x = MARGIN_X
            baseline_y += LINE_HEIGHT
            report.lines += 1
            continue

        if x > MARGIN_X and x + measure_glyph_token(token, lib) > MAX_X:
            x = MARGIN_X
            baseline_y += LINE_HEIGHT
            report.lines += 1

        for char in token:
            if char == " ":
                x += SPACE_WIDTH
                continue

            path, rendered_as = choose_glyph_path(char, lib)
            if path is None:
                report.missing[char] = report.missing.get(char, 0) + 1
                x += SPACE_WIDTH
                continue

            glyph = prepare_glyph(path, jitter=jitter)
            if x + glyph.width > MAX_X:
                x = MARGIN_X
                baseline_y += LINE_HEIGHT
                report.lines += 1
            if baseline_y > MAX_Y:
                report.overflow = True
                continue

            paste_glyph(page, glyph, x, baseline_y)
            x += glyph.width + GLYPH_GAP
            report.rendered += 1
            if rendered_as != char:
                report.missing[char] = report.missing.get(char, 0) + 1

    save_page(page, output_name)
    return report


def load_font(font_path: str | None = None, font_size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    path = font_path or DEFAULT_FONT_PATH
    try:
        return ImageFont.truetype(path, font_size)
    except OSError:
        return ImageFont.load_default(size=font_size)


def text_width(font: ImageFont.ImageFont, text: str) -> int:
    if not text:
        return 0
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def measure_font_token(font: ImageFont.ImageFont, token: str) -> int:
    return sum(FONT_WORD_SPACE if char == " " else text_width(font, char) + random.randint(-2, 3) for char in token)


def draw_font_char(page: Image.Image, font: ImageFont.ImageFont, char: str, x: int, y: int, jitter: bool) -> tuple[int, int]:
    bbox = font.getbbox(char)
    width = max(1, bbox[2] - bbox[0] + 24)
    height = max(1, bbox[3] - bbox[1] + 32)
    tile = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(tile)
    draw.text((12 - bbox[0], 12 - bbox[1]), char, font=font, fill=0)

    if jitter:
        tile = tile.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=255)
        y += random.randint(-7, 7)

    mask = tile.point(lambda p: 255 if p < 230 else 0)
    page.paste(Image.new("L", tile.size, 0), (x, y), mask)
    return tile.width, tile.height


def compose_font(
    text: str,
    output_name: str = "handwritten_result.pdf",
    font_path: str | None = None,
    seed: int | None = None,
    jitter: bool = True,
) -> ComposeReport:
    if seed is not None:
        random.seed(seed)

    text = normalize_text(text)
    font = load_font(font_path)
    page = Image.new("L", PAGE_SIZE, 255)
    x, y = MARGIN_X, MARGIN_Y
    report = ComposeReport(output_path=output_name, engine="font")

    for token in iter_tokens(text):
        if token == "\n":
            x = MARGIN_X
            y += FONT_LINE_HEIGHT
            report.lines += 1
            continue

        if x > MARGIN_X and x + measure_font_token(font, token) > MAX_X:
            x = MARGIN_X
            y += FONT_LINE_HEIGHT
            report.lines += 1

        for char in token:
            if char == " ":
                x += FONT_WORD_SPACE + random.randint(-4, 6)
                continue

            width, height = draw_font_char(page, font, char, x, y, jitter=jitter)
            if x + width > MAX_X:
                x = MARGIN_X
                y += FONT_LINE_HEIGHT
                report.lines += 1
                width, height = draw_font_char(page, font, char, x, y, jitter=jitter)
            if y + height > MAX_Y:
                report.overflow = True
                continue
            x += max(18, text_width(font, char) + random.randint(-2, 5))
            report.rendered += 1

    save_page(page, output_name)
    return report


def compose(
    text: str,
    output_name: str = "handwritten_result.pdf",
    library_path: str = "data/glyph_library.json",
    writer: str | None = None,
    seed: int | None = None,
    jitter: bool = True,
    engine: str = "font",
    font_path: str | None = None,
) -> ComposeReport:
    if engine == "glyph":
        return compose_glyph(text, output_name, library_path, writer, seed, jitter)
    if engine == "font":
        return compose_font(text, output_name, font_path=font_path, seed=seed, jitter=jitter)
    raise ValueError(f"Unknown engine: {engine}")


def format_report(report: ComposeReport) -> str:
    missing = ", ".join(f"{repr(k)} x{v}" for k, v in sorted(report.missing.items()))
    if not missing:
        missing = "none"
    return (
        f"Created {report.output_path}\n"
        f"Engine: {report.engine}; coverage: {report.coverage:.1%}; rendered: {report.rendered}; "
        f"lines: {report.lines}; overflow: {report.overflow}; missing/fallback: {missing}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render text using a legible font engine or the extracted glyph library.")
    parser.add_argument("text", nargs="?", default="this is a test.")
    parser.add_argument("-o", "--output", default="handwritten_result.pdf")
    parser.add_argument("--engine", choices=["font", "glyph"], default="font")
    parser.add_argument("--library", default="data/glyph_library.json")
    parser.add_argument("--writer", default=None, help="Optional writer token to filter glyph filenames, e.g. writer1")
    parser.add_argument("--font", default=None, help="Optional .ttf/.otf path for the font engine")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-jitter", action="store_true")
    args = parser.parse_args()

    report = compose(
        args.text,
        output_name=args.output,
        library_path=args.library,
        writer=args.writer,
        seed=args.seed,
        jitter=not args.no_jitter,
        engine=args.engine,
        font_path=args.font,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
