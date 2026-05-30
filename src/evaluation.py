import argparse
import difflib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .composer import compose
from .golden_set import GOLDEN_QUERIES
from .ocr_pipeline import image_to_pdf, load_metadata
from .synthetic_glyphs import generate_synthetic_glyphs

RENDER_MODELS = [
    ("baseline_font_jitter", "font", None, "Readable font baseline with jitter."),
    ("baseline_synthetic_glyph_retrieval", "glyph", "data/synthetic_glyph_library.json", "Clean synthetic glyph retrieval baseline."),
    ("baseline_extracted_glyph_retrieval", "glyph", "data/glyph_library.json", "Noisy extracted glyph retrieval baseline."),
    ("proposed_script_renderer", "script", None, "Connected pen-stroke handwriting renderer."),
]
OCR_SAMPLES = ["handwriting_0002.png", "handwriting_0005.png", "handwriting_0010.png", "handwriting_0015.png", "handwriting_0020.png"]


def render_row(model: str, description: str, query: str, report) -> dict:
    row = asdict(report)
    row.update(
        {
            "model": model,
            "description": description,
            "query": query,
            "coverage": report.coverage,
            "missing_total": sum(report.missing.values()),
        }
    )
    return row


def run_render_benchmark(out: Path, seed: int = 2025) -> list[dict]:
    generate_synthetic_glyphs(library_path="data/synthetic_glyph_library.json", variants=8, seed=7)
    rows = []
    for model, engine, library, description in RENDER_MODELS:
        model_dir = out / "render" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for idx, (query, text) in enumerate(GOLDEN_QUERIES, start=1):
            report = compose(
                text,
                output_name=str(model_dir / f"{query}.pdf"),
                engine=engine,
                library_path=library or "data/glyph_library.json",
                seed=seed + idx,
            )
            rows.append(render_row(model, description, query, report))
    return rows


def summarize_render(rows: list[dict]) -> list[dict]:
    summary = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary.append(
            {
                "model": model,
                "avg_coverage": mean(row["coverage"] for row in model_rows),
                "total_missing": sum(row["missing_total"] for row in model_rows),
                "overflow_count": sum(1 for row in model_rows if row["overflow"]),
                "avg_lines": mean(row["lines"] for row in model_rows),
                "queries": len(model_rows),
            }
        )
    return summary


def run_ocr_benchmark(out: Path, metadata_path: str = "data/metadata.csv") -> list[dict]:
    metadata = load_metadata(metadata_path)
    ocr_dir = out / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in OCR_SAMPLES:
        expected = metadata.get(name)
        image = Path("data/raw_handwriting") / name
        if not expected or not image.exists():
            continue
        result = image_to_pdf(str(image), output_path=str(ocr_dir / f"{image.stem}.pdf"), method="metadata")
        similarity = difflib.SequenceMatcher(None, expected, result.text).ratio()
        rows.append(
            {
                "image": str(image),
                "method": result.method,
                "expected": expected,
                "prediction": result.text,
                "exact_match": expected == result.text,
                "sequence_similarity": similarity,
                "output_path": result.output_path,
                "note": "Metadata method is oracle validation of PDF pipeline, not arbitrary OCR.",
            }
        )
    return rows


def write_report(out: Path, render_rows: list[dict], render_summary: list[dict], ocr_rows: list[dict]) -> Path:
    path = out / "EVALUATION_REPORT.md"
    lines = [
        "# Evaluation And Benchmark Report",
        "",
        "## Summary",
        "",
        "This report compares baseline renderers against the current proposed handwriting renderer, and validates the handwriting-image-to-text-PDF pipeline on labeled dataset samples.",
        "",
        "The OCR numbers below use metadata labels for bundled dataset images. That is an oracle pipeline validation, not proof of arbitrary OCR generalization.",
        "",
        "## Text-To-Handwriting Benchmark",
        "",
        "| Model | Avg Coverage | Total Missing | Overflow Count | Avg Lines | Queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in render_summary:
        lines.append(f"| {row['model']} | {row['avg_coverage']:.1%} | {row['total_missing']} | {row['overflow_count']} | {row['avg_lines']:.2f} | {row['queries']} |")

    lines += ["", "### Per-Query Results", "", "| Model | Query | Coverage | Missing | Lines | Overflow | Output |", "|---|---|---:|---:|---:|---:|---|"]
    for row in render_rows:
        lines.append(f"| {row['model']} | {row['query']} | {row['coverage']:.1%} | {row['missing_total']} | {row['lines']} | {row['overflow']} | `{row['output_path']}` |")

    lines += ["", "## OCR / Image-To-PDF Benchmark", "", "| Image | Method | Exact | Similarity | Expected | Prediction | Output |", "|---|---|---:|---:|---|---|---|"]
    for row in ocr_rows:
        lines.append(f"| {Path(row['image']).name} | {row['method']} | {row['exact_match']} | {row['sequence_similarity']:.1%} | {row['expected']} | {row['prediction']} | `{row['output_path']}` |")
    avg_sim = mean(row["sequence_similarity"] for row in ocr_rows) if ocr_rows else 0.0
    exact = sum(1 for row in ocr_rows if row["exact_match"])
    lines += [
        "",
        f"OCR exact matches: {exact}/{len(ocr_rows)}",
        f"OCR average sequence similarity: {avg_sim:.1%}",
        "",
        "## Interpretation",
        "",
        "- `baseline_font_jitter` is readable but font-like.",
        "- `baseline_synthetic_glyph_retrieval` shows retrieval/composition with clean labels.",
        "- `baseline_extracted_glyph_retrieval` exposes the weakness of noisy segmentation-derived glyphs.",
        "- `proposed_script_renderer` is the current model used for demo-quality text-to-handwriting because it creates connected pen-stroke output without requiring filled writer templates.",
        "- The next real improvement is collecting writer-specific template glyphs or training a conditional glyph generator.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_evaluation(output_dir: str = "outputs/evaluation") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    render_rows = run_render_benchmark(out)
    render_summary = summarize_render(render_rows)
    ocr_rows = run_ocr_benchmark(out)
    report = write_report(out, render_rows, render_summary, ocr_rows)
    payload = {"render_summary": render_summary, "render_results": render_rows, "ocr_results": ocr_rows, "report": str(report)}
    (out / "evaluation_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation report written to {report}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation and benchmark report.")
    parser.add_argument("-o", "--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    run_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
