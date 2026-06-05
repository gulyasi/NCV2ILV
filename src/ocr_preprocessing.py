from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


PREPROCESSING_MODES = ("none", "grayscale", "otsu", "adaptive", "denoise-deskew")
ENSEMBLE_MODES = ("none", "otsu", "adaptive", "denoise-deskew")


def _ensure_white_background(gray: np.ndarray) -> np.ndarray:
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return cv2.bitwise_not(gray) if float(np.mean(border)) < 127 else gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    ink = np.column_stack(np.where(gray < 220))
    if len(ink) < 25:
        return gray

    angle = cv2.minAreaRect(ink)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.1 or abs(angle) > 8:
        return gray

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess_array(image_path: str, mode: str = "none") -> np.ndarray:
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode {mode!r}. Expected one of {PREPROCESSING_MODES}.")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    gray = _ensure_white_background(image)
    if mode == "none":
        return gray
    if mode == "grayscale":
        return gray
    if mode == "otsu":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=18, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return _deskew(binary)


def preprocess_image(image_path: str, output_path: str, mode: str = "none") -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), preprocess_array(image_path, mode=mode))
    return str(output)


@contextmanager
def preprocessed_image_path(image_path: str, mode: str = "none") -> Iterator[str]:
    if mode == "none":
        yield image_path
        return

    with tempfile.TemporaryDirectory(prefix="handwriting_ocr_") as tmp:
        output = Path(tmp) / f"{Path(image_path).stem}_{mode}.png"
        yield preprocess_image(image_path, str(output), mode=mode)
