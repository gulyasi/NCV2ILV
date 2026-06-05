import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ocr_metrics import ocr_candidate_score
from .ocr_preprocessing import ENSEMBLE_MODES, PREPROCESSING_MODES, preprocessed_image_path


PAGE_SIZE = (2480, 3508)
MARGIN = 160
LINE_HEIGHT = 96
FONT_SIZE = 58
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass
class OCRResult:
    text: str
    method: str
    output_path: str


def load_metadata(metadata_path: str = "data/metadata.csv") -> dict[str, str]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["file_name"]: row["label"] for row in csv.DictReader(f)}


def transcribe_from_metadata(image_path: str, metadata_path: str = "data/metadata.csv") -> str | None:
    image_name = Path(image_path).name
    return load_metadata(metadata_path).get(image_name)


def transcribe_with_tesseract(image_path: str, lang: str = "eng") -> str | None:
    if shutil.which("tesseract") is None:
        return None
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "-l", lang, "--psm", "6"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = result.stdout.strip()
    return text or None


def transcribe_with_tesseract_ensemble(image_path: str, lang: str = "eng") -> tuple[str | None, str | None]:
    candidates: list[tuple[float, str, str]] = []
    for mode in ENSEMBLE_MODES:
        with preprocessed_image_path(image_path, mode=mode) as prepared:
            text = transcribe_with_tesseract(prepared, lang=lang)
        if text:
            candidates.append((ocr_candidate_score(text), text, mode))
    if not candidates:
        return None, None
    _, text, mode = max(candidates, key=lambda item: item[0])
    return text, mode


def transcribe_with_qwen(image_path: str) -> str:
    from .translate import QwenTranslator

    return QwenTranslator().transcribe(image_path)


def transcribe_image(
    image_path: str,
    method: str = "auto",
    metadata_path: str = "data/metadata.csv",
    tesseract_lang: str = "eng",
    preprocess: str = "none",
    ensemble_preprocess: bool = False,
) -> tuple[str, str]:
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    if preprocess not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode {preprocess!r}. Expected one of {PREPROCESSING_MODES}.")

    if method in {"auto", "metadata"}:
        text = transcribe_from_metadata(image_path, metadata_path)
        if text:
            return text, "metadata"
        if method == "metadata":
            raise RuntimeError(f"No metadata label found for {Path(image_path).name}")

    if method in {"auto", "tesseract"}:
        if ensemble_preprocess:
            text, selected_mode = transcribe_with_tesseract_ensemble(image_path, lang=tesseract_lang)
            used_method = f"tesseract+preprocess-ensemble:{selected_mode}" if selected_mode else "tesseract+preprocess-ensemble"
        else:
            with preprocessed_image_path(image_path, mode=preprocess) as prepared:
                text = transcribe_with_tesseract(prepared, lang=tesseract_lang)
            used_method = "tesseract" if preprocess == "none" else f"tesseract+{preprocess}"
        if text:
            return text, used_method
        if method == "tesseract":
            raise RuntimeError("Tesseract did not return text or is not installed")

    if method == "qwen":
        with preprocessed_image_path(image_path, mode=preprocess) as prepared:
            text = transcribe_with_qwen(prepared)
        if text:
            return text, "qwen" if preprocess == "none" else f"qwen+{preprocess}"
        raise RuntimeError("Qwen OCR did not return text")

    raise RuntimeError(
        "Could not transcribe image. Use a labeled dataset image, install tesseract, "
        "or run with --method qwen if model downloads are available."
    )


def load_font(size: int = FONT_SIZE) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.getlength(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def write_text_pdf(text: str, output_path: str, source_image: str | None = None, method: str | None = None) -> None:
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    title_font = load_font(52)
    body_font = load_font(FONT_SIZE)
    meta_font = load_font(34)

    y = MARGIN
    draw.text((MARGIN, y), "Transcribed Handwriting", fill=(0, 0, 0), font=title_font)
    y += 78
    if source_image:
        draw.text((MARGIN, y), f"Source: {Path(source_image).name}", fill=(90, 90, 90), font=meta_font)
        y += 46
    if method:
        draw.text((MARGIN, y), f"OCR method: {method}", fill=(90, 90, 90), font=meta_font)
        y += 70

    for line in wrap_text(text, body_font, PAGE_SIZE[0] - 2 * MARGIN):
        if y + LINE_HEIGHT > PAGE_SIZE[1] - MARGIN:
            break
        draw.text((MARGIN, y), line, fill=(0, 0, 0), font=body_font)
        y += LINE_HEIGHT

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, "PDF", resolution=300.0)


def image_to_pdf(
    image_path: str,
    output_path: str = "outputs/transcription.pdf",
    method: str = "auto",
    metadata_path: str = "data/metadata.csv",
    tesseract_lang: str = "eng",
    preprocess: str = "none",
    ensemble_preprocess: bool = False,
) -> OCRResult:
    text, used_method = transcribe_image(
        image_path,
        method=method,
        metadata_path=metadata_path,
        tesseract_lang=tesseract_lang,
        preprocess=preprocess,
        ensemble_preprocess=ensemble_preprocess,
    )
    write_text_pdf(text, output_path, source_image=image_path, method=used_method)
    return OCRResult(text=text, method=used_method, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a handwritten image and write the recognized text to PDF.")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="outputs/transcription.pdf")
    parser.add_argument("--method", choices=["auto", "metadata", "tesseract", "qwen"], default="auto")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument("--preprocess", choices=PREPROCESSING_MODES, default="none")
    parser.add_argument("--ensemble-preprocess", action="store_true")
    args = parser.parse_args()
    result = image_to_pdf(
        args.image,
        output_path=args.output,
        method=args.method,
        metadata_path=args.metadata,
        tesseract_lang=args.tesseract_lang,
        preprocess=args.preprocess,
        ensemble_preprocess=args.ensemble_preprocess,
    )
    print(f"Created {result.output_path}")
    print(f"Method: {result.method}")
    print(f"Text: {result.text}")


if __name__ == "__main__":
    main()
