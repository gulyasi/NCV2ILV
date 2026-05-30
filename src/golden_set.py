import argparse
from pathlib import Path

try:
    from .composer import compose, format_report
except ImportError:
    from composer import compose, format_report


GOLDEN_QUERIES = [
    ("01_pangram", "The quick brown fox jumps over the lazy dog."),
    ("02_order", "Order #1234: 5 apples and 3 oranges, total $8.50!"),
    ("03_birthday", "Dear Mom,\nHappy Birthday!\nLove, Alex"),
    (
        "04_paragraph",
        "Personal handwriting makes typed notes feel warmer and more individual. "
        "This paragraph checks whether the renderer can wrap text across many lines "
        "while keeping spacing and baseline placement reasonably consistent over a full page.",
    ),
    ("05_rare_letters", "Zebras quiver, waxing jazz."),
]


def run_golden_set(output_dir: str = "outputs/golden_set", library_path: str = "data/glyph_library.json", writer: str | None = None, engine: str = "font", font_path: str | None = None) -> list:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (name, text) in enumerate(GOLDEN_QUERIES, start=1):
        report = compose(
            text,
            output_name=str(out / f"{name}.pdf"),
            library_path=library_path,
            writer=writer,
            seed=1000 + index,
            engine=engine,
            font_path=font_path,
        )
        reports.append(report)
        print(format_report(report))
        print()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the proposal golden-set handwriting PDFs.")
    parser.add_argument("-o", "--output-dir", default="outputs/golden_set")
    parser.add_argument("--engine", choices=["font", "glyph"], default="font")
    parser.add_argument("--library", default="data/glyph_library.json")
    parser.add_argument("--writer", default=None)
    parser.add_argument("--font", default=None)
    args = parser.parse_args()
    run_golden_set(args.output_dir, library_path=args.library, writer=args.writer, engine=args.engine, font_path=args.font)


if __name__ == "__main__":
    main()
