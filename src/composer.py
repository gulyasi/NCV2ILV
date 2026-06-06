import argparse
import json
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


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
SCRIPT_LINE_HEIGHT = 150
SCRIPT_WORD_SPACE = 76
SCRIPT_LETTER_WIDTH = 64
SCRIPT_LETTER_HEIGHT = 92
SCRIPT_BASELINE = 92
SCRIPT_FONT_PATH = "assets/fonts/DancingScript.ttf"
SCRIPT_FONT_SIZE = 104
SCRIPT_WORD_GAP = 58
SCRIPT_FONT_LINE_HEIGHT = 168


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



def draw_smooth_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: int = 5) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=0, width=width, joint="curve")


def cubic_points(p0, p1, p2, p3, steps: int = 18) -> list[tuple[float, float]]:
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def script_tile(char: str, rng: random.Random) -> tuple[Image.Image, int]:
    w = SCRIPT_LETTER_WIDTH
    h = SCRIPT_LETTER_HEIGHT + 28
    tile = Image.new("L", (w + 18, h), 255)
    draw = ImageDraw.Draw(tile)
    base = SCRIPT_BASELINE
    top = 18
    mid = 54
    left = 8
    right = w - 5
    c = char.lower()
    width = rng.randint(4, 6)

    def pts(norm):
        return [(left + x * (right - left), top + y * (base - top)) for x, y in norm]

    if c in "aceos":
        # Small loop letters.
        oval = [
            (0.78, 0.55), (0.65, 0.22), (0.30, 0.22), (0.20, 0.55),
            (0.12, 0.88), (0.48, 0.98), (0.78, 0.78), (0.96, 0.62),
        ]
        draw_smooth_line(draw, pts(oval), width)
        if c == "e":
            draw_smooth_line(draw, pts([(0.20, 0.62), (0.55, 0.55), (0.85, 0.48)]), width)
        elif c == "s":
            draw_smooth_line(draw, pts([(0.78, 0.35), (0.36, 0.22), (0.24, 0.48), (0.76, 0.68), (0.42, 0.96)]), width)
        elif c == "c":
            draw_smooth_line(draw, pts([(0.78, 0.32), (0.35, 0.20), (0.16, 0.58), (0.42, 0.96), (0.86, 0.78)]), width)
    elif c in "il":
        draw_smooth_line(draw, pts([(0.35, 0.95), (0.48, 0.10), (0.62, 0.95), (0.92, 0.82)]), width)
        if c == "i":
            draw.ellipse((left + 25, top - 6, left + 35, top + 4), fill=0)
    elif c == "h":
        draw_smooth_line(draw, pts([(0.18, 0.95), (0.25, 0.02), (0.36, 0.35), (0.18, 0.98)]), width)
        draw_smooth_line(draw, pts([(0.30, 0.70), (0.52, 0.32), (0.76, 0.58), (0.82, 0.95), (0.98, 0.82)]), width)
    elif c == "t":
        draw_smooth_line(draw, pts([(0.35, 0.12), (0.48, 0.95), (0.85, 0.80)]), width)
        draw_smooth_line(draw, pts([(0.18, 0.42), (0.78, 0.36)]), max(3, width - 1))
    elif c == "a":
        draw_smooth_line(draw, pts([(0.76, 0.58), (0.55, 0.25), (0.20, 0.42), (0.18, 0.78), (0.48, 0.96), (0.76, 0.65), (0.84, 0.95), (0.98, 0.82)]), width)
    elif c == "r":
        draw_smooth_line(draw, pts([(0.25, 0.96), (0.36, 0.36), (0.48, 0.55), (0.66, 0.34), (0.88, 0.46)]), width)
    elif c in "mn":
        arches = [(0.18, 0.95), (0.28, 0.42), (0.45, 0.92), (0.58, 0.42), (0.76, 0.92), (0.96, 0.82)] if c == "m" else [(0.24, 0.95), (0.36, 0.42), (0.62, 0.92), (0.94, 0.82)]
        draw_smooth_line(draw, pts(arches), width)
    elif c in "uvwy":
        seq = [(0.15, 0.42), (0.30, 0.95), (0.52, 0.54), (0.70, 0.95), (0.96, 0.78)]
        if c == "y":
            seq = [(0.15, 0.42), (0.32, 0.95), (0.58, 0.54), (0.50, 1.18), (0.30, 1.25), (0.88, 0.80)]
        elif c == "w":
            seq = [(0.10, 0.42), (0.24, 0.95), (0.42, 0.55), (0.58, 0.95), (0.76, 0.55), (0.96, 0.80)]
        draw_smooth_line(draw, pts(seq), width)
    elif c in "bdkp":
        stem_x = 0.28
        draw_smooth_line(draw, pts([(stem_x, 0.96), (stem_x, 0.02)]), width)
        if c in "bp":
            loop = [(0.30, 0.48), (0.75, 0.28), (0.92, 0.68), (0.55, 0.96), (0.30, 0.78)]
            if c == "p":
                draw_smooth_line(draw, pts([(stem_x, 0.42), (stem_x, 1.22)]), width)
        elif c == "d":
            loop = [(0.30, 0.78), (0.58, 0.35), (0.90, 0.58), (0.72, 0.96), (0.35, 0.92)]
        else:
            loop = [(0.30, 0.68), (0.78, 0.34), (0.42, 0.66), (0.88, 0.96)]
        draw_smooth_line(draw, pts(loop), width)
    elif c in "fgjqz":
        draw_smooth_line(draw, pts([(0.70, 0.20), (0.36, 0.18), (0.30, 0.72), (0.52, 1.20), (0.24, 1.30), (0.88, 0.80)]), width)
        if c == "j":
            draw.ellipse((left + 25, top - 6, left + 35, top + 4), fill=0)
    elif c == "x":
        draw_smooth_line(draw, pts([(0.20, 0.35), (0.80, 0.96)]), width)
        draw_smooth_line(draw, pts([(0.78, 0.35), (0.22, 0.96), (0.96, 0.80)]), width)
    else:
        font = load_font(None, 74)
        draw.text((10, 16), char, font=font, fill=0)

    if char.isupper():
        tile = tile.resize((int(tile.width * 1.18), int(tile.height * 1.12)), Image.Resampling.LANCZOS)
    angle = rng.uniform(-2.0, 2.0)
    tile = tile.rotate(angle, expand=True, fillcolor=255)
    advance = max(24, int(SCRIPT_LETTER_WIDTH * rng.uniform(0.55, 0.76)))
    return tile, advance


def load_script_font(size: int = SCRIPT_FONT_SIZE) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(SCRIPT_FONT_PATH, size)
    except OSError:
        return load_font(None, size)


def render_script_word(word: str, font: ImageFont.ImageFont, rng: random.Random, jitter: bool) -> Image.Image:
    bbox = font.getbbox(word, stroke_width=1)
    width = max(1, bbox[2] - bbox[0] + 36)
    height = max(1, bbox[3] - bbox[1] + 36)
    tile = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(tile)
    draw.text((18 - bbox[0], 18 - bbox[1]), word, font=font, fill=0, stroke_width=1, stroke_fill=0)
    if jitter:
        tile = tile.rotate(rng.uniform(-0.45, 0.45), expand=True, fillcolor=255)
    return tile


def learned_style_samples(word: str, library: dict[str, list[str]], rng: random.Random, limit: int = 4) -> list[Image.Image]:
    candidates = [path for char in word for path in library.get(char, [])]
    rng.shuffle(candidates)
    samples: list[Image.Image] = []
    for path in candidates:
        try:
            glyph = Image.open(path).convert("L")
            if glyph.resize((1, 1)).getpixel((0, 0)) < 128:
                glyph = ImageChops.invert(glyph)
            ink = ImageChops.invert(glyph)
            bbox = ink.getbbox()
            if bbox:
                samples.append(ink.crop(bbox))
            if len(samples) >= limit:
                break
        except OSError:
            continue
    return samples


def roughen_ink(ink: Image.Image, rng: random.Random) -> Image.Image:
    width, height = ink.size
    warped = Image.new("L", ink.size, 0)

    # Shift narrow vertical bands independently to break the perfect font
    # silhouette while preserving connected strokes.
    band_width = max(10, width // 24)
    x = 0
    previous_shift = rng.randint(-2, 2)
    while x < width:
        next_shift = max(-4, min(4, previous_shift + rng.choice((-1, 0, 0, 1))))
        band = ink.crop((x, 0, min(width, x + band_width + 3), height))
        warped.paste(band, (x, next_shift))
        previous_shift = next_shift
        x += band_width

    # Real pen pressure changes across a word. Alternate slight dilation and
    # erosion in broad sections instead of applying one uniform stroke width.
    pressure = Image.new("L", warped.size, 0)
    section_width = max(28, width // 7)
    x = 0
    while x < width:
        section = warped.crop((x, 0, min(width, x + section_width + 4), height))
        choice = rng.random()
        if choice < 0.38:
            section = section.filter(ImageFilter.MaxFilter(3))
        elif choice < 0.58:
            section = section.filter(ImageFilter.MinFilter(3))
        pressure.paste(section, (x, 0), section)
        x += section_width

    # Remove a sparse set of edge pixels, creating dry-pen imperfections
    # without punching large holes through the writing.
    edge = ImageChops.difference(pressure, pressure.filter(ImageFilter.MinFilter(3)))
    noise = Image.frombytes("L", pressure.size, rng.randbytes(width * height))
    sparse_noise = noise.point(lambda value: 255 if value > 236 else 0)
    edge_mask = ImageChops.multiply(edge.point(lambda value: 255 if value > 12 else 0), sparse_noise)
    pressure.paste(0, mask=edge_mask)
    return pressure.filter(ImageFilter.GaussianBlur(0.28))


def apply_learned_style(tile: Image.Image, word: str, library: dict[str, list[str]], rng: random.Random) -> Image.Image:
    samples = learned_style_samples(word, library, rng)
    if not samples:
        return tile

    densities = []
    width_ratios = []
    for sample in samples:
        ink_pixels = sum(sample.histogram()[32:])
        densities.append(ink_pixels / max(1, sample.width * sample.height))
        width_ratios.append(sample.width / max(1, sample.height))

    density = sum(densities) / len(densities)
    width_ratio = sum(width_ratios) / len(width_ratios)
    x_scale = min(1.08, max(0.94, 0.98 + (width_ratio - 0.55) * 0.08))
    tile = tile.resize((max(1, int(tile.width * x_scale)), tile.height), Image.Resampling.BICUBIC)

    ink = ImageChops.invert(tile)
    if density > 0.34:
        ink = ink.filter(ImageFilter.MaxFilter(3))
    elif density < 0.20:
        ink = ink.filter(ImageFilter.MinFilter(3))

    texture = Image.new("L", ink.size, 0)
    tx = 0
    while tx < texture.width:
        sample = rng.choice(samples)
        target_h = max(12, int(texture.height * rng.uniform(0.45, 0.9)))
        target_w = max(8, int(sample.width * target_h / max(1, sample.height)))
        sample = sample.resize((target_w, target_h), Image.Resampling.BILINEAR)
        y = rng.randint(0, max(0, texture.height - target_h))
        texture.paste(sample, (tx, y), sample)
        tx += max(6, int(target_w * rng.uniform(0.6, 1.0)))
    texture = texture.filter(ImageFilter.GaussianBlur(2.2))

    # Preserve the font's connected structure while borrowing uneven ink flow
    # from real glyph crops. A small gray floor prevents artificial holes.
    modulation = texture.point(lambda value: 205 + value // 5)
    ink = ImageChops.multiply(ink, modulation)
    ink = roughen_ink(ink, rng)
    return ImageChops.invert(ink)


def script_tokens(text: str) -> Iterable[str]:
    token = ""
    for char in text:
        if char in {" ", "\n"}:
            if token:
                yield token
                token = ""
            yield char
        else:
            token += char
    if token:
        yield token


def compose_script(
    text: str,
    output_name: str = "handwritten_result.pdf",
    seed: int | None = None,
    jitter: bool = True,
    library_path: str | None = None,
    engine_name: str = "script",
) -> ComposeReport:
    rng = random.Random(seed)
    font = load_script_font()
    library = load_library(library_path) if library_path else None
    page = Image.new("L", PAGE_SIZE, 255)
    x, y = MARGIN_X, MARGIN_Y
    report = ComposeReport(output_path=output_name, engine=engine_name)

    for token in script_tokens(normalize_text(text)):
        if token == "\n":
            x = MARGIN_X
            y += SCRIPT_FONT_LINE_HEIGHT
            report.lines += 1
            continue
        if token == " ":
            x += SCRIPT_WORD_GAP + (rng.randint(-4, 6) if jitter else 0)
            continue

        tile = render_script_word(token, font, rng, jitter)
        if library:
            tile = apply_learned_style(tile, token, library, rng)
        if x > MARGIN_X and x + tile.width > MAX_X:
            x = MARGIN_X
            y += SCRIPT_FONT_LINE_HEIGHT
            report.lines += 1
        if y + tile.height > MAX_Y:
            report.overflow = True
            continue

        y_offset = rng.randint(-7, 7) if jitter and library else (rng.randint(-3, 3) if jitter else 0)
        mask = tile.point(lambda value: 255 if value < 230 else 0)
        page.paste(Image.new("L", tile.size, 0), (x, y + y_offset), mask)
        x += tile.width - 14 + (rng.randint(-5, 5) if jitter and library else 0)
        report.rendered += len(token)

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
    if engine == "script":
        return compose_script(text, output_name, seed=seed, jitter=jitter)
    if engine == "hybrid":
        return compose_script(text, output_name, seed=seed, jitter=jitter, library_path=library_path, engine_name="hybrid")
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
    parser.add_argument("--engine", choices=["font", "glyph", "script", "hybrid"], default="script")
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
