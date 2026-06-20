# Personal Handwriting Converter

A research prototype for converting handwriting images into text and rendering typed text as handwriting-style PNG or PDF documents. The project includes a local browser GUI, a command-line interface, multiple rendering engines, OCR preprocessing, evaluation utilities, and a workflow for collecting writer-specific glyphs.

## Project Status

The system supports both directions:

- **Handwriting to text:** transcribes a handwriting image and writes the recognized text into a PDF.
- **Text to handwriting:** renders typed text using font, connected-script, direct-glyph, or hybrid handwriting engines.

The current recommended rendering engine is `hybrid`. It keeps connected cursive word shapes while borrowing proportions, stroke pressure, ink texture, rough edges, and variation from extracted handwriting glyphs.

> **Important model distinction:** the production text-to-handwriting pipeline is not a trained generative neural network. It is a composition system. The only locally trained model documented in this repository is a baseline glyph classifier, described in [What Was Trained](#what-was-trained).

## Features

- Local browser GUI with mode switching between both conversion directions.
- OCR for new uploaded images through Qwen vision-language OCR.
- Optional image preprocessing before Qwen OCR.
- Text rendering to page-sized PNG or PDF files.
- Four handwriting rendering engines.
- Hybrid rendering that mixes connected cursive with learned glyph texture.
- Printable handwriting template and fixed-grid glyph extraction.
- Synthetic glyph generation for clean retrieval baselines.
- Baseline comparison, golden-set generation, and evaluation reports.

## Architecture

```text
Handwriting image
      |
      v
              optional preprocessing
                        |
                        v
                    Qwen OCR
                        |
                        v
                recognized text
                        |
                        v
                 formatted text PDF

Typed text
    |
    v
font / script / glyph / hybrid renderer
    |
    v
handwriting-style PNG or PDF
```

### Handwriting-to-Text Pipeline

All handwriting images are transcribed with **Qwen OCR** using `Qwen/Qwen3.5-0.8B`.

Qwen is loaded through Hugging Face Transformers and cached in memory for the lifetime of the process. The first run may download model files and take longer. CUDA is used when available.

Optional preprocessing modes are:

- `none`
- `grayscale`
- `otsu`
- `adaptive`
- `denoise-deskew`

### Text-to-Handwriting Engines

| Engine | Description | Trained? | Recommended use |
|---|---|---:|---|
| `font` | Legible oblique font rendered character-by-character with jitter. | No | Readable baseline |
| `script` | Whole-word connected cursive using the bundled Dancing Script font. | No | Clean connected cursive |
| `glyph` | Direct retrieval and composition of extracted or synthetic character images. | No | Writer-specific experiments and retrieval baseline |
| `hybrid` | Connected cursive structure mixed with extracted-glyph proportions, ink texture, pressure variation, local warping, and dry-pen edges. | No | Best current handwriting-style output |

The hybrid engine uses `data/glyph_library.json` as a learned-style reference library. It does not train a new neural generator; it samples statistics and textures from extracted glyph images during rendering.

## What Was Trained

### Baseline Glyph Classifier

`src/train.py` defines and trains `SimpleBaselineCNN`, a character-classification baseline.

Architecture:

- Input: grayscale glyph image resized to `64 x 64`
- Convolution: `1 -> 32` channels, ReLU, max pooling
- Convolution: `32 -> 64` channels, ReLU, max pooling
- Fully connected: `64 x 16 x 16 -> 128`
- Output: one class per character code found in `data/glyphs`

Training configuration:

- Dataset: PNG glyph crops from `data/glyphs`
- Label source: character code embedded in each filename
- Split: 80% training, 20% validation allocation
- Augmentation: resize and random affine transform
- Optimizer: Adam
- Learning rate: `0.001`
- Loss: cross-entropy
- Batch size: `16`
- Epochs: `15`
- Device: CUDA when available, otherwise CPU
- Output checkpoint: `data/baseline_model.pth`

Run training with:

```bash
uv run python src/train.py
```

The classifier checkpoint is a research baseline. It is **not currently loaded by the GUI, OCR pipeline, or handwriting renderer**.

### Pretrained Qwen OCR Model

`Qwen/Qwen3.5-0.8B` is used for arbitrary-image OCR. It is a pretrained external model loaded at inference time; it is not trained or fine-tuned by this project.

### Existing `handwriting_model.pth` Artifact

`data/handwriting_model.pth` exists in the repository, but the current application does not load it and no matching training procedure is present in the current source tree. It should therefore be treated as an unintegrated experimental artifact, not as the model behind the generated handwriting.

## Data

### Bundled Handwriting Dataset

`src/download_data.py` downloads the first 100 samples from the Hugging Face dataset `fhswf/german_handwriting` and writes:

- Images to `data/raw_handwriting/handwriting_####.png`
- Labels to `data/metadata.csv`

Current repository data summary:

| Artifact | Current size |
|---|---:|
| Metadata-labeled images | 100 |
| Raw handwriting PNGs, including an additional long sample | 101 |
| Extracted glyph PNGs | 7,492 |
| Extracted glyph library characters | 68 |
| Extracted glyph library variants | 7,492 |
| Synthetic glyph library characters | 75 |
| Synthetic glyph library variants | 600 |

Download or recreate the labeled sample data with:

```bash
uv run python src/download_data.py
```

### Extracted Glyph Library

`src/segmentation.py` extracts connected components from the labeled handwriting images and assigns characters according to label order. The resulting files use names such as:

```text
data/glyphs/char_97_12_4.png
```

Here, `97` is the Unicode code point for `a`.

This extraction method is noisy because connected handwriting can produce word fragments rather than clean isolated letters. `src/build_library.py` filters obvious failures by size and ink ratio, but it cannot correct incorrect character assignments.

Build a filtered library with:

```bash
uv run python src/build_library.py \
  --glyph-dir data/glyphs \
  -o data/glyph_library_filtered.json
```

### Synthetic Glyph Library

`src/synthetic_glyphs.py` creates clean character-level variants from a local oblique font. It adds small size, position, rotation, and blur variation while preserving a consistent baseline.

Generate it with:

```bash
uv run python main.py data synthetic --variants 12 --seed 1234
```

### Collecting Personal Handwriting

The most reliable personalization workflow uses a fixed-grid template rather than segmenting connected sentences.

1. Create the printable template:

```bash
uv run python main.py template make -o outputs/handwriting_template.pdf
```

2. Fill it by hand and scan or photograph it.

3. Extract isolated glyphs:

```bash
uv run python main.py template extract path/to/filled_template.png \
  -o data/template_glyphs \
  --prefix writer1
```

4. Build the writer library:

```bash
uv run python src/build_library.py \
  --glyph-dir data/template_glyphs \
  -o data/writer1_glyph_library.json
```

5. Render with the writer library:

```bash
uv run python main.py render "A personal handwriting sample" \
  --engine hybrid \
  --library data/writer1_glyph_library.json \
  -o outputs/writer1_sample.png
```

## Installation

### Requirements

- Python `3.12` or newer
- `uv` package manager
- CUDA-capable GPU recommended for Qwen OCR

Install the Python environment:

```bash
uv sync
```

The first Qwen OCR request may download model files from Hugging Face. Set `HF_TOKEN` if authenticated downloads or higher rate limits are required.

## Quick Start

### Browser GUI

Start the local GUI:

```bash
uv run python main.py gui
```

The command prefers `http://127.0.0.1:8000`. If that port is occupied, it selects another free local port and prints the actual URL. Press `Ctrl+C` to stop the server.

#### Text to Handwriting

1. Select **Text to Handwriting**.
2. Enter text.
3. Select an engine. `hybrid (font + learned writing)` is the recommended default.
4. Select a glyph library. Use `data/glyph_library.json` for the bundled extracted style.
5. Choose a PNG or PDF output path.
6. Press **Generate Handwriting**.

#### Handwriting to Text

1. Select **Handwriting to Text**.
2. Upload an image.
3. Optionally select preprocessing.
4. Choose an output PDF path.
5. Press **Recognize Handwriting**.

Qwen is the automatic OCR backend. A manual transcription field is also available as a fallback for producing the formatted PDF without OCR.

### Recommended CLI Commands

Render realistic hybrid handwriting:

```bash
uv run python main.py render "Food was great!" \
  --engine hybrid \
  --library data/glyph_library.json \
  --seed 7 \
  -o outputs/hybrid_handwriting.png
```

Render clean connected cursive:

```bash
uv run python main.py handwrite "This is connected cursive." \
  --engine script \
  -o outputs/script_handwriting.pdf
```

Transcribe a new image with Qwen:

```bash
uv run python main.py image-to-pdf path/to/new_image.png \
  -o outputs/transcription.pdf
```

Transcribe with preprocessing:

```bash
uv run python main.py image-to-pdf path/to/new_image.png \
  --preprocess denoise-deskew \
  -o outputs/qwen_transcription.pdf
```

## Command Reference

| Command | Purpose |
|---|---|
| `gui` | Start the local browser interface |
| `image-to-pdf` | Transcribe a handwriting image and create a text PDF |
| `render` | Render text using any handwriting engine |
| `handwrite` | Shortcut for script/glyph/hybrid handwriting output |
| `template make` | Create a printable personal-handwriting template |
| `template extract` | Extract isolated glyphs from a filled template |
| `data synthetic` | Generate a clean synthetic glyph library |
| `baseline` | Generate baseline outputs and comparison reports |
| `golden` | Render the five fixed golden-set prompts |
| `evaluate` | Generate rendering and OCR benchmark reports |
| `demo` | Run the implemented workflows end to end |

Show all CLI options:

```bash
uv run python main.py --help
```

## Evaluation

Run the full evaluation:

```bash
uv run python main.py evaluate
```

Outputs are written to:

```text
outputs/evaluation/EVALUATION_REPORT.md
outputs/evaluation/evaluation_results.json
outputs/evaluation/render/
outputs/evaluation/ocr/
```

### Rendering Evaluation

The benchmark measures:

- Character coverage
- Missing/fallback character count
- Line count
- Page overflow

The current benchmark compares:

- Font jitter baseline
- Synthetic glyph retrieval baseline
- Extracted glyph retrieval baseline
- Connected script renderer

The existing report records 100% coverage for font, synthetic glyph, and script rendering across the five golden prompts, and 92.8% average coverage for the noisy extracted-glyph baseline. The evaluation script currently predates the hybrid engine and does not yet include a perceptual realism metric.

### OCR Evaluation

The evaluation computes:

- Character error rate (CER)
- Word error rate (WER)
- Sequence similarity
- Exact match

The bundled evaluation compares Qwen predictions with the ground-truth labels in `data/metadata.csv`. Use a separately labeled test set to evaluate broader OCR generalization.

Run the Qwen sample benchmark:

```bash
uv run python src/qwen_benchmark.py
```

## Demo

Run all MVP workflows:

```bash
uv run python main.py demo
```

The demo generates OCR PDFs, script and glyph renderings, baseline comparisons, golden-set PDFs, a printable template, and:

```text
outputs/demo/DEMO_REPORT.md
outputs/demo/demo_summary.json
```

## Repository Layout

```text
.
├── assets/fonts/                 # Bundled cursive font and license
├── data/
│   ├── raw_handwriting/          # Labeled handwriting images
│   ├── glyphs/                   # Extracted character crops
│   ├── glyph_library.json        # Extracted glyph index
│   ├── synthetic_glyph_library.json
│   ├── metadata.csv              # Image-to-text labels
│   ├── baseline_model.pth        # Trained glyph-classifier baseline
│   └── handwriting_model.pth     # Unintegrated experimental artifact
├── outputs/                      # Generated reports, PDFs, and PNGs
├── src/
│   ├── composer.py               # Font, script, glyph, and hybrid renderers
│   ├── gui.py                    # Local browser GUI and API
│   ├── ocr_pipeline.py           # Qwen OCR and text-PDF generation
│   ├── ocr_preprocessing.py      # Thresholding, denoising, and deskewing
│   ├── translate.py              # Qwen OCR wrapper
│   ├── train.py                  # Baseline glyph-classifier training
│   ├── segmentation.py           # Dataset-image glyph extraction
│   ├── template_tools.py         # Personal template generation/extraction
│   ├── synthetic_glyphs.py       # Synthetic glyph generation
│   ├── evaluation.py             # Benchmark runner
│   └── demo.py                   # End-to-end demo runner
├── main.py                       # Main CLI entry point
├── pyproject.toml                # Project metadata and dependencies
└── uv.lock                       # Locked dependency versions
```

## Current Limitations

- The hybrid renderer uses extracted glyphs as style references; it is not a learned sequence-to-image generator.
- The extracted glyph library contains noisy or incorrectly labeled crops from connected handwriting.
- The bundled style library combines multiple samples and is not guaranteed to represent one consistent writer.
- Real personalization requires a clean filled template from the target writer.
- Qwen OCR accuracy depends on image quality, language, model behavior, and available compute.
- The current page renderer creates one page and reports overflow rather than automatically creating additional pages.
- Existing rendering evaluation measures coverage and layout, not human-rated realism.

## Troubleshooting

### A new image cannot be transcribed

```bash
uv run python main.py image-to-pdf path/to/image.png
```

Confirm that model downloads are allowed and sufficient memory is available.

### Glyph output contains fragments or incorrect characters

The extracted library is noisy. Build a filtered library or collect a fixed-grid writer template:

```bash
uv run python src/build_library.py -o data/glyph_library_filtered.json
```

### Hybrid output is too clean or too rough

- Use `script` for cleaner connected cursive.
- Use `hybrid` for rougher learned texture.
- Use `--no-jitter` for deterministic placement without baseline and spacing variation.
- Use `--seed` to reproduce a particular result.

### Port 8000 is occupied

The GUI automatically selects another free port. Use the URL printed in the terminal.

## Reproducibility

Use explicit seeds for repeatable rendering and data generation:

```bash
uv run python main.py render "Repeatable sample" --engine hybrid --seed 7 -o outputs/repeatable.png
uv run python main.py data synthetic --variants 12 --seed 1234
```

The hybrid engine randomly samples learned glyph references, but the same text, library, options, and seed produce repeatable output.

## License and Attribution

Project code is covered by the repository `LICENSE` file.

The bundled Dancing Script font is stored in `assets/fonts/DancingScript.ttf` and is distributed under the SIL Open Font License. Its license text is included at `assets/fonts/DancingScript-OFL.txt`.
