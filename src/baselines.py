import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .composer import compose
from .synthetic_glyphs import generate_synthetic_glyphs


DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def report_row(name: str, report) -> dict:
    row = asdict(report)
    row["name"] = name
    row["coverage"] = report.coverage
    row["missing_total"] = sum(report.missing.values())
    return row


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# Baseline Comparison",
        "",
        "| Baseline | Engine | Coverage | Rendered | Missing | Lines | Overflow | Output |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {engine} | {coverage:.1%} | {rendered} | {missing_total} | {lines_count} | {overflow} | {output_path} |".format(
                name=row["name"],
                engine=row["engine"],
                coverage=row["coverage"],
                rendered=row["rendered"],
                missing_total=row["missing_total"],
                lines_count=row["lines"],
                overflow=row["overflow"],
                output_path=row["output_path"],
            )
        )
    lines.extend(
        [
            "",
            "Recommended interpretation:",
            "",
            "- Font baseline: lower-bound readable rendering; not personal style.",
            "- Synthetic glyph baseline: clean character-level retrieval baseline; proves the glyph engine works with correct labels.",
            "- Extracted glyph baseline: current noisy data baseline; useful to show why clean template data is needed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baselines(
    text: str = DEFAULT_TEXT,
    output_dir: str = "outputs/baselines",
    synthetic_library: str = "data/synthetic_glyph_library.json",
    extracted_library: str = "data/glyph_library.json",
    seed: int = 1234,
    include_extracted: bool = True,
) -> list[dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    synthetic_path = Path(synthetic_library)
    if not synthetic_path.exists():
        generate_synthetic_glyphs(library_path=str(synthetic_path), seed=seed)

    rows = []
    rows.append(
        report_row(
            "font_jitter",
            compose(text, output_name=str(out / "font_jitter.pdf"), engine="font", seed=seed),
        )
    )
    rows.append(
        report_row(
            "synthetic_glyph_retrieval",
            compose(
                text,
                output_name=str(out / "synthetic_glyph_retrieval.pdf"),
                engine="glyph",
                library_path=str(synthetic_path),
                seed=seed,
            ),
        )
    )

    extracted_path = Path(extracted_library)
    if include_extracted and extracted_path.exists():
        rows.append(
            report_row(
                "extracted_glyph_retrieval",
                compose(
                    text,
                    output_name=str(out / "extracted_glyph_retrieval.pdf"),
                    engine="glyph",
                    library_path=str(extracted_path),
                    seed=seed,
                ),
            )
        )

    json_path = out / "baseline_report.json"
    md_path = out / "baseline_report.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(rows, md_path)

    print(f"Baseline outputs written to {out}")
    print(f"Report written to {md_path}")
    for row in rows:
        print(
            f"{row['name']}: coverage={row['coverage']:.1%}, "
            f"missing={row['missing_total']}, lines={row['lines']}, overflow={row['overflow']}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline renderings and coverage report.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("-o", "--output-dir", default="outputs/baselines")
    parser.add_argument("--synthetic-library", default="data/synthetic_glyph_library.json")
    parser.add_argument("--extracted-library", default="data/glyph_library.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-extracted", action="store_true")
    args = parser.parse_args()
    run_baselines(
        text=args.text,
        output_dir=args.output_dir,
        synthetic_library=args.synthetic_library,
        extracted_library=args.extracted_library,
        seed=args.seed,
        include_extracted=not args.no_extracted,
    )


if __name__ == "__main__":
    main()
