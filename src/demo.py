import json
from pathlib import Path

from .baselines import run_baselines
from .composer import compose
from .golden_set import run_golden_set
from .ocr_pipeline import image_to_pdf
from .synthetic_glyphs import generate_synthetic_glyphs
from .template_tools import make_template


DEMO_TEXT = "Dear Mom,\nHappy Birthday!\nLove, Alex"
OCR_SAMPLE_IMAGES = [
    "data/raw_handwriting/handwriting_0002.png",
    "data/raw_handwriting/handwriting_0005.png",
    "data/raw_handwriting/handwriting_0010.png",
]


def write_demo_report(report_path: Path, rows: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_demo(output_dir: str = "outputs/demo") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Demo Report",
        "",
        "This demo runs the implemented MVP workflows end to end.",
        "",
    ]
    summary: dict = {"output_dir": str(out), "artifacts": []}

    rows.append("## 1. Handwritten Image To Text PDF")
    ocr_results = []
    for image_path in OCR_SAMPLE_IMAGES:
        path = Path(image_path)
        if not path.exists():
            continue
        output_path = out / "ocr" / f"{path.stem}.pdf"
        result = image_to_pdf(str(path), output_path=str(output_path), method="metadata")
        ocr_results.append({"image": str(path), "text": result.text, "output": result.output_path})
        summary["artifacts"].append(result.output_path)
        rows.append(f"- `{path.name}` -> `{output_path}`: {result.text}")
    rows.append("")

    rows.append("## 2. Text To Handwriting")
    script_report = compose(DEMO_TEXT, output_name=str(out / "text_to_script_handwriting.pdf"), engine="script", seed=7)
    summary["artifacts"].append(script_report.output_path)
    rows.append(f"- Script handwriting: `{script_report.output_path}` ({script_report.coverage:.1%} coverage)")

    synthetic_library = Path("data/synthetic_glyph_library.json")
    generate_synthetic_glyphs(library_path=str(synthetic_library), variants=8, seed=7)
    glyph_report = compose(
        DEMO_TEXT,
        output_name=str(out / "text_to_glyph_handwriting.pdf"),
        engine="glyph",
        library_path=str(synthetic_library),
        seed=7,
    )
    summary["artifacts"].append(glyph_report.output_path)
    rows.append(f"- Synthetic glyph handwriting: `{glyph_report.output_path}` ({glyph_report.coverage:.1%} coverage)")
    rows.append("")

    rows.append("## 3. Baselines")
    baseline_rows = run_baselines(output_dir=str(out / "baselines"), synthetic_library=str(synthetic_library), seed=7)
    summary["artifacts"].extend(str(path) for path in (out / "baselines").glob("*.pdf"))
    summary["artifacts"].append(str(out / "baselines" / "baseline_report.md"))
    for row in baseline_rows:
        rows.append(f"- {row['name']}: coverage={row['coverage']:.1%}, missing={row['missing_total']}, output=`{row['output_path']}`")
    rows.append("")

    rows.append("## 4. Golden Set")
    golden_reports = run_golden_set(str(out / "golden_set"), engine="script")
    for report in golden_reports:
        summary["artifacts"].append(report.output_path)
        rows.append(f"- `{report.output_path}`: coverage={report.coverage:.1%}, lines={report.lines}, overflow={report.overflow}")
    rows.append("")

    rows.append("## 5. Template For Personal Handwriting")
    template_path = out / "handwriting_template.pdf"
    make_template(str(template_path))
    summary["artifacts"].append(str(template_path))
    rows.append(f"- Printable template: `{template_path}`")
    rows.append("")

    rows.extend(
        [
            "## Honest Status",
            "",
            "- The project demonstrates both directions: handwriting image to text PDF, and text to handwriting-style PDF.",
            "- OCR for bundled dataset images is validated through metadata labels.",
            "- OCR for arbitrary unknown handwriting requires Tesseract or Qwen and remains the main improvement area.",
            "- Truly personal handwriting requires filled template sheets from the target writer.",
        ]
    )

    report_path = out / "DEMO_REPORT.md"
    summary_path = out / "demo_summary.json"
    write_demo_report(report_path, rows)
    summary["artifacts"].append(str(report_path))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"].append(str(summary_path))

    print(f"Demo outputs written to {out}")
    print(f"Demo report: {report_path}")
    return summary


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
