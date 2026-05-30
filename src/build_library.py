import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image


def glyph_quality(path: str, min_width: int, max_width: int, min_height: int, max_height: int, min_ink: float, max_ink: float) -> tuple[bool, str]:
    try:
        img = Image.open(path).convert("L")
    except OSError:
        return False, "unreadable"

    width, height = img.size
    if width < min_width or height < min_height:
        return False, "too_small"
    if width > max_width or height > max_height:
        return False, "too_large"

    pixels = img.getdata()
    # Existing glyph crops are usually binary masks. Count whichever side is
    # less common as ink so both black-on-white and white-on-black work.
    dark = sum(1 for value in pixels if value < 128)
    total = width * height
    ink_ratio = min(dark, total - dark) / total
    if ink_ratio < min_ink:
        return False, "mostly_blank"
    if ink_ratio > max_ink:
        return False, "mostly_filled"

    return True, "ok"


def char_from_filename(filename: str) -> str | None:
    parts = filename.split("_")
    if len(parts) < 2:
        return None
    try:
        return chr(int(parts[1]))
    except ValueError:
        return None


def build_lib(
    glyph_dir: str = "data/glyphs",
    output_path: str = "data/glyph_library.json",
    min_width: int = 3,
    max_width: int = 120,
    min_height: int = 8,
    max_height: int = 160,
    min_ink: float = 0.015,
    max_ink: float = 0.75,
) -> dict[str, list[str]]:
    lib = defaultdict(list)
    skipped = defaultdict(int)

    for filename in os.listdir(glyph_dir):
        if not filename.endswith(".png"):
            continue

        char = char_from_filename(filename)
        path = os.path.join(glyph_dir, filename)
        if char is None:
            skipped["bad_filename"] += 1
            continue

        ok, reason = glyph_quality(path, min_width, max_width, min_height, max_height, min_ink, max_ink)
        if not ok:
            skipped[reason] += 1
            continue

        lib[char].append(path)

    serializable = {char: paths for char, paths in sorted(lib.items())}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

    print(f"Library written to {output_path}")
    print(f"Unique glyph types: {len(serializable)}")
    print(f"Total variants: {sum(len(paths) for paths in serializable.values())}")
    if skipped:
        print("Skipped:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")

    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered glyph library from extracted glyph crops.")
    parser.add_argument("--glyph-dir", default="data/glyphs")
    parser.add_argument("-o", "--output", default="data/glyph_library_filtered.json")
    parser.add_argument("--min-width", type=int, default=3)
    parser.add_argument("--max-width", type=int, default=120)
    parser.add_argument("--min-height", type=int, default=8)
    parser.add_argument("--max-height", type=int, default=160)
    parser.add_argument("--min-ink", type=float, default=0.015)
    parser.add_argument("--max-ink", type=float, default=0.75)
    args = parser.parse_args()

    build_lib(
        glyph_dir=args.glyph_dir,
        output_path=args.output,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_ink=args.min_ink,
        max_ink=args.max_ink,
    )


if __name__ == "__main__":
    main()
