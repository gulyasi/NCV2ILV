import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .ocr_preprocessing import PREPROCESSING_MODES, preprocessed_image_path


PAGE_SIZE = (2480, 3508)
_QWEN_TRANSLATOR = None
MARGIN = 160
LINE_HEIGHT = 96
FONT_SIZE = 58
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass
class OCRResult:
    text: str
    method: str
    output_path: str


def transcribe_with_qwen(image_path: str) -> str:
    global _QWEN_TRANSLATOR
    if _QWEN_TRANSLATOR is None:
        from .translate import QwenTranslator

        _QWEN_TRANSLATOR = QwenTranslator()
    return _QWEN_TRANSLATOR.transcribe(image_path)


def transcribe_image(
    image_path: str,
    preprocess: str = "none",
) -> tuple[str, str]:
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    if preprocess not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode {preprocess!r}. Expected one of {PREPROCESSING_MODES}.")

    try:
        with preprocessed_image_path(image_path, mode=preprocess) as prepared:
            text = transcribe_with_qwen(prepared)
    except Exception as exc:
        raise RuntimeError(f"Qwen OCR failed: {exc}") from exc
    if not text:
        raise RuntimeError("Qwen OCR did not return text")
    return text, "qwen" if preprocess == "none" else f"qwen+{preprocess}"


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
    preprocess: str = "none",
) -> OCRResult:
    text, used_method = transcribe_image(
        image_path,
        preprocess=preprocess,
    )
    write_text_pdf(text, output_path, source_image=image_path, method=used_method)
    return OCRResult(text=text, method=used_method, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a handwritten image and write the recognized text to PDF.")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="outputs/transcription.pdf")
    parser.add_argument("--preprocess", choices=PREPROCESSING_MODES, default="none")
    args = parser.parse_args()
    result = image_to_pdf(
        args.image,
        output_path=args.output,
        preprocess=args.preprocess,
    )
    print(f"Created {result.output_path}")
    print(f"Method: {result.method}")
    print(f"Text: {result.text}")


if __name__ == "__main__":
    main()
