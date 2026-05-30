import argparse
import sys
from pathlib import Path

from src.baselines import run_baselines
from src.composer import compose, format_report
from src.demo import run_demo
from src.golden_set import run_golden_set
from src.ocr_pipeline import image_to_pdf
from src.synthetic_glyphs import generate_synthetic_glyphs
from src.template_tools import extract_template, make_template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal handwriting rendering prototype",
        epilog=(
            "Examples:\n"
            "  uv run python main.py handwrite \"The quick brown fox jumps over the lazy dog.\" -o outputs/pangram.pdf\n"
            "  uv run python main.py golden\n"
            "  uv run python main.py template make -o outputs/handwriting_template.pdf\n"
            "  uv run python main.py data synthetic\n"
            "  uv run python main.py baseline\n"
            "  uv run python main.py image-to-pdf data/raw_handwriting/handwriting_0005.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")



    demo_parser = subparsers.add_parser("demo", help="Run all MVP workflows and write a demo report")
    demo_parser.add_argument("-o", "--output-dir", default="outputs/demo")

    ocr_parser = subparsers.add_parser("image-to-pdf", help="Transcribe a handwritten image and save recognized text as PDF")
    ocr_parser.add_argument("image")
    ocr_parser.add_argument("-o", "--output", default="outputs/transcription.pdf")
    ocr_parser.add_argument("--method", choices=["auto", "metadata", "tesseract", "qwen"], default="auto")
    ocr_parser.add_argument("--metadata", default="data/metadata.csv")
    ocr_parser.add_argument("--tesseract-lang", default="eng")


    handwrite_parser = subparsers.add_parser("handwrite", help="Render text using glyph-based handwriting")
    handwrite_parser.add_argument("text")
    handwrite_parser.add_argument("-o", "--output", default="outputs/handwriting.pdf")
    handwrite_parser.add_argument("--engine", choices=["script", "glyph"], default="script")
    handwrite_parser.add_argument("--library", default="data/synthetic_glyph_library.json")
    handwrite_parser.add_argument("--variants", type=int, default=8)
    handwrite_parser.add_argument("--seed", type=int, default=7)
    handwrite_parser.add_argument("--no-jitter", action="store_true")

    render_parser = subparsers.add_parser("render", help="Render one text string to PNG or PDF")
    render_parser.add_argument("text")
    render_parser.add_argument("-o", "--output", default="outputs/handwritten_result.pdf")
    render_parser.add_argument("--engine", choices=["font", "glyph", "script"], default="font")
    render_parser.add_argument("--library", default="data/glyph_library.json")
    render_parser.add_argument("--writer", default=None)
    render_parser.add_argument("--font", default=None)
    render_parser.add_argument("--seed", type=int, default=None)
    render_parser.add_argument("--no-jitter", action="store_true")

    golden_parser = subparsers.add_parser("golden", help="Generate the five proposal golden-set PDFs")
    golden_parser.add_argument("-o", "--output-dir", default="outputs/golden_set")
    golden_parser.add_argument("--engine", choices=["font", "glyph", "script"], default="font")
    golden_parser.add_argument("--library", default="data/glyph_library.json")
    golden_parser.add_argument("--writer", default=None)
    golden_parser.add_argument("--font", default=None)

    template_parser = subparsers.add_parser("template", help="Create or extract clean handwriting template sheets")
    template_subparsers = template_parser.add_subparsers(dest="template_command")
    template_make = template_subparsers.add_parser("make", help="Create a printable handwriting template")
    template_make.add_argument("-o", "--output", default="outputs/handwriting_template.pdf")
    template_make.add_argument("--chars", default=None)

    template_extract = template_subparsers.add_parser("extract", help="Extract glyphs from a filled template image")
    template_extract.add_argument("image")
    template_extract.add_argument("-o", "--output-dir", default="data/template_glyphs")
    template_extract.add_argument("--chars", default=None)
    template_extract.add_argument("--prefix", default="template")



    baseline_parser = subparsers.add_parser("baseline", help="Generate baseline renderings and a comparison report")
    baseline_parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    baseline_parser.add_argument("-o", "--output-dir", default="outputs/baselines")
    baseline_parser.add_argument("--synthetic-library", default="data/synthetic_glyph_library.json")
    baseline_parser.add_argument("--extracted-library", default="data/glyph_library.json")
    baseline_parser.add_argument("--seed", type=int, default=1234)
    baseline_parser.add_argument("--no-extracted", action="store_true")

    data_parser = subparsers.add_parser("data", help="Generate or prepare glyph datasets")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    synthetic_parser = data_subparsers.add_parser("synthetic", help="Generate a clean synthetic glyph dataset")
    synthetic_parser.add_argument("-o", "--output-dir", default="data/synthetic_glyphs")
    synthetic_parser.add_argument("--library", default="data/synthetic_glyph_library.json")
    synthetic_parser.add_argument("--chars", default=None)
    synthetic_parser.add_argument("--variants", type=int, default=12)
    synthetic_parser.add_argument("--seed", type=int, default=1234)
    synthetic_parser.add_argument("--font", action="append", dest="fonts", default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.output_dir)
    elif args.command == "image-to-pdf":
        result = image_to_pdf(
            args.image,
            output_path=args.output,
            method=args.method,
            metadata_path=args.metadata,
            tesseract_lang=args.tesseract_lang,
        )
        print(f"Created {result.output_path}")
        print(f"Method: {result.method}")
        print(f"Text: {result.text}")

    elif args.command == "handwrite":
        library_path = Path(args.library)
        if args.engine == "glyph" and not library_path.exists():
            generate_synthetic_glyphs(library_path=str(library_path), variants=args.variants, seed=args.seed)
        report = compose(
            args.text,
            output_name=args.output,
            library_path=str(library_path),
            seed=args.seed,
            jitter=not args.no_jitter,
            engine=args.engine,
        )
        print(format_report(report))
    elif args.command == "render":
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
    elif args.command == "golden":
        run_golden_set(args.output_dir, library_path=args.library, writer=args.writer, engine=args.engine, font_path=args.font)
    elif args.command == "template":
        if args.template_command == "make":
            make_template(args.output, chars=args.chars) if args.chars else make_template(args.output)
        elif args.template_command == "extract":
            extract_template(args.image, output_dir=args.output_dir, chars=args.chars, prefix=args.prefix) if args.chars else extract_template(args.image, output_dir=args.output_dir, prefix=args.prefix)
        else:
            parser.parse_args(["template", "--help"])


    elif args.command == "baseline":
        run_baselines(
            text=args.text,
            output_dir=args.output_dir,
            synthetic_library=args.synthetic_library,
            extracted_library=args.extracted_library,
            seed=args.seed,
            include_extracted=not args.no_extracted,
        )

    elif args.command == "data":
        if args.data_command == "synthetic":
            kwargs = {
                "output_dir": args.output_dir,
                "library_path": args.library,
                "variants": args.variants,
                "seed": args.seed,
                "font_paths": args.fonts,
            }
            if args.chars is not None:
                kwargs["chars"] = args.chars
            generate_synthetic_glyphs(**kwargs)
        else:
            parser.parse_args(["data", "--help"])


if __name__ == "__main__":
    main()
