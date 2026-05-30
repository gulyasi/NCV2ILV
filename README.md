# Personal Handwriting System

Prototype for converting handwritten image input into recognized text PDFs, plus text-to-handwriting rendering experiments. The default engine is a legible handwritten-style font renderer with per-character jitter so the demo works reliably today. The extracted-glyph engine is still available for the personal-handwriting research path, but it needs cleaner template data before it looks good.

## What Works

- Transcribes handwritten images into text PDFs.
- Renders arbitrary text to a page-sized PNG or PDF.
- Uses a readable default `font` engine with light per-character jitter.
- Keeps the extracted-glyph renderer available with `--engine glyph`.
- Reports missing characters and fallback substitutions.
- Generates the proposal golden-set outputs.
- Generates a printable template sheet for collecting clean personal handwriting glyphs.
- Generates a clean synthetic glyph dataset for baseline/testing.
- Generates baseline comparison PDFs and a metrics report.
- Runs a complete demo workflow with one command.

## Quick Start



Run the full MVP demo:

```bash
uv run python main.py demo
```

This creates OCR outputs, handwriting outputs, baselines, golden-set PDFs, a handwriting template, and `outputs/demo/DEMO_REPORT.md`.

Convert a handwriting image to a text PDF:

```bash
uv run python main.py image-to-pdf data/raw_handwriting/handwriting_0005.png -o outputs/transcription.pdf
```

The default `auto` OCR mode first uses `data/metadata.csv` when the image is from the bundled labeled dataset. For unknown images, it can use local Tesseract if installed, or Qwen with `--method qwen` when model downloads are available.

Render one readable PDF with the default engine:

```bash
uv run python main.py render "The quick brown fox jumps over the lazy dog." -o outputs/pangram.pdf
```

Generate the five golden-set PDFs with the default readable engine:

```bash
uv run python main.py golden
```


Generate connected script-style handwriting directly:

```bash
uv run python main.py handwrite "hello this is a test" -o outputs/hello_handwrite.pdf
```

This uses the connected script renderer by default, which looks more handwriting-like than the font renderer.

For glyph retrieval specifically:

```bash
uv run python main.py handwrite "hello this is a test" --engine glyph -o outputs/hello_glyph.pdf
```

Run the extracted-glyph engine, which uses the current dirty glyph library:

```bash
uv run python src/build_library.py -o data/glyph_library_filtered.json
uv run python main.py render "this is a test." -o outputs/test_glyph.pdf --engine glyph --library data/glyph_library_filtered.json
```


Create a clean handwriting collection template:

```bash
uv run python main.py template make -o outputs/handwriting_template.pdf
```

After filling the template by hand and scanning/photoing it, extract clean glyphs:

```bash
uv run python main.py template extract path/to/filled_template.png -o data/template_glyphs --prefix writer1
uv run python src/build_library.py --glyph-dir data/template_glyphs -o data/writer1_glyph_library.json
uv run python main.py render "hello from my handwriting" --engine glyph --library data/writer1_glyph_library.json -o outputs/writer1_demo.pdf
```

## Useful Options

- `--engine font`: readable default renderer.
- `handwrite`: shortcut for connected script-style text-to-handwriting output.
- `--engine glyph`: use extracted glyph crops from a glyph JSON.
- `--writer writer1`: filter glyphs to filenames containing a writer token.
- `--library path/to/glyph_library.json`: use another glyph library.
- `--font path/to/font.ttf`: use a custom font for the default engine.
- `--no-jitter`: disable random rotation.
- `--seed 123`: make sampling repeatable.





## OCR / Image To PDF

The main product command is:

```bash
uv run python main.py image-to-pdf path/to/handwritten_image.png -o outputs/transcription.pdf
```

OCR methods:

- `--method auto`: use metadata for bundled dataset images, then try local Tesseract.
- `--method metadata`: only use labels from `data/metadata.csv`; useful for validating the pipeline on known data.
- `--method tesseract`: use a local Tesseract install if available.
- `--method qwen`: use the existing Qwen vision-language OCR wrapper; this may require model downloads and enough memory.

Example with a known bundled image:

```bash
uv run python main.py image-to-pdf data/raw_handwriting/handwriting_0005.png -o outputs/sample_transcription.pdf --method metadata
```

## Baseline Models

For the course report/demo, use explicit baselines rather than jumping directly to a large vision-language model. A Flamingo-style model is good for image-text understanding and few-shot multimodal reasoning, but it is not a natural baseline for generating a specific writer's handwriting glyphs. For this project, the defensible baselines are:

1. `font_jitter`: readable handwritten-style font rendering with character jitter. This is the simplest rendering baseline.
2. `synthetic_glyph_retrieval`: clean character-level glyph retrieval from generated synthetic glyphs with preserved baseline zones. This tests whether the composition engine works when labels and alignment are correct.
3. `extracted_glyph_retrieval`: retrieval from the current extracted dataset glyphs. This shows how much the noisy segmentation/labeling hurts output quality.

Run all implemented baselines:

```bash
uv run python main.py baseline
```

This writes baseline PDFs plus:

```text
outputs/baselines/baseline_report.md
outputs/baselines/baseline_report.json
```

A realistic stretch model would be a conditional glyph generator, not Flamingo: for example, a small pix2pix/diffusion-style model conditioned on `(character label, writer/style embedding)`. Keep that as a stretch goal after the retrieval baselines work.

If the synthetic glyph baseline still looks poor, regenerate it before rendering:

```bash
uv run python main.py data synthetic --variants 8 --seed 7
uv run python main.py render "The quick brown fox jumps over the lazy dog." --engine glyph --library data/synthetic_glyph_library.json -o outputs/synthetic_pangram.pdf --seed 7
```


## Evaluation Report

Run the full benchmark/report command:

```bash
uv run python main.py evaluate
```

This writes:

```text
outputs/evaluation/EVALUATION_REPORT.md
outputs/evaluation/evaluation_results.json
outputs/evaluation/render/
outputs/evaluation/ocr/
```

The report compares three baselines against the current proposed renderer:

- `baseline_font_jitter`: readable handwritten-font baseline.
- `baseline_synthetic_glyph_retrieval`: clean glyph retrieval baseline.
- `baseline_extracted_glyph_retrieval`: noisy dataset-extracted glyph baseline.
- `proposed_script_renderer`: connected pen-stroke renderer used as the current project model.

The OCR section validates image-to-text-PDF output on bundled labeled dataset samples using metadata labels. That is useful for pipeline evaluation, but it is an oracle setting rather than a claim that arbitrary handwriting OCR is solved.

## Data Strategy

The old German line-image dataset is useful for OCR experiments, but it is a poor source for glyph composition because connected words are hard to split into correctly labeled characters. Prefer these sources, in this order:

1. Filled project template sheets from consenting writers. This is the best data for personal handwriting because each cell has a known character label.
2. Synthetic clean glyphs generated from a consistent local font with preserved baseline zones. This is useful for testing, baseline training, and proving the glyph engine works with correctly labeled and aligned character crops.
3. Public isolated-character datasets such as EMNIST for classifier pretraining. These are better for recognition than for personal style generation.
4. Full line/word datasets such as IAM only after you have reliable annotation and segmentation. They should not be the MVP glyph source.

Generate a clean synthetic glyph baseline:

```bash
uv run python main.py data synthetic --variants 12
uv run python main.py render "The quick brown fox jumps over the lazy dog." --engine glyph --library data/synthetic_glyph_library.json -o outputs/synthetic_pangram.pdf
```

## Improving Output Quality

If the generated handwriting looks like fragments of words or random marks, the glyph library is polluted. Rebuild a filtered library first:

```bash
uv run python src/build_library.py -o data/glyph_library_filtered.json
uv run python main.py render "The quick brown fox jumps over the lazy dog." -o outputs/pangram_filtered.pdf --library data/glyph_library_filtered.json
```

This removes obvious bad crops, but the real fix is collecting isolated template sheets where each character is known in advance. The current dataset-derived glyphs are labeled by connected-component order, so connected handwriting and word fragments can be assigned to the wrong character.

## Current Limitations

- The default `font` engine is legible but not personal handwriting.
- The `glyph` engine can represent personal handwriting, but only after collecting clean template glyphs.
- The original dataset-derived glyph library is polluted by word fragments and mislabeled crops.
- Layout is functional but simple: it does word wrapping and explicit line breaks, not paragraph-level typography.
- OCR/transcription experiments are present, but they are not required for rendering text into handwriting.


## Project Layout

```text
.
├── main.py                 # CLI entry point
├── pyproject.toml          # Python project metadata and dependencies
├── src/                    # Application and research modules
├── data/                   # Current dataset/model artifacts
└── data_samples/           # Small sample input assets
```

## Main Files

- `main.py`: command-line entry point.
- `src/demo.py`: complete MVP demo runner.
- `src/ocr_pipeline.py`: handwritten image transcription and text-PDF output.
- `src/composer.py`: text-to-page renderer and coverage reporting.
- `src/baselines.py`: baseline rendering comparison and report generation.
- `src/golden_set.py`: golden-set PDF generation.
- `src/template_tools.py`: printable template creation and fixed-grid glyph extraction.
- `src/synthetic_glyphs.py`: synthetic clean glyph dataset generation.
- `src/build_library.py`: builds filtered glyph-library JSON files from extracted glyph crops.
- `src/download_data.py`: downloads the sample handwriting dataset and metadata.
- `src/segmentation.py`: older heuristic glyph extraction script.
