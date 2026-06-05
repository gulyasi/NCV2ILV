from __future__ import annotations

import re
import unicodedata


def normalize_for_ocr(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def character_error_rate(expected: str, prediction: str) -> float:
    expected = normalize_for_ocr(expected)
    prediction = normalize_for_ocr(prediction)
    if not expected:
        return 0.0 if not prediction else 1.0
    return edit_distance(expected, prediction) / len(expected)


def word_error_rate(expected: str, prediction: str) -> float:
    expected_words = normalize_for_ocr(expected).split()
    prediction_words = normalize_for_ocr(prediction).split()
    if not expected_words:
        return 0.0 if not prediction_words else 1.0
    return edit_distance(expected_words, prediction_words) / len(expected_words)


def ocr_candidate_score(text: str) -> float:
    normalized = normalize_for_ocr(text)
    if not normalized:
        return -1.0
    useful = sum(ch.isalnum() for ch in normalized)
    strange = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-\"()[]/") for ch in normalized)
    return useful - 2.5 * strange + min(len(normalized), 160) * 0.03
