from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


MEDICAL_TOKENS = {
    "abnormal",
    "albumin",
    "blood",
    "calcium",
    "culture",
    "diagnosis",
    "diabetes",
    "glucose",
    "hpf",
    "ketones",
    "lab",
    "leukocytes",
    "negative",
    "nitrite",
    "patient",
    "positive",
    "protein",
    "rbc",
    "result",
    "specimen",
    "trace",
    "urinalysis",
    "urine",
    "wbc",
}

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}(?:[/-][A-Za-zА-Яа-яЁё0-9]{1,12})?")
REPEATED_SYMBOL_RE = re.compile(r"([^A-Za-zА-Яа-яЁё0-9\s])\1{2,}")
GARBAGE_RE = re.compile(r"\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]|(?:[|_~`^*•·]{4,})")


@dataclass(frozen=True)
class TextQualityAssessment:
    score: float
    band: str
    warnings: list[str]
    metrics: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_text_quality(text: str) -> TextQualityAssessment:
    source = text or ""
    stripped = source.strip()
    length = len(stripped)
    if length == 0:
        return TextQualityAssessment(
            score=0.0,
            band="empty",
            warnings=["empty_or_near_empty_text"],
            metrics={
                "text_length": 0,
                "garbage_character_ratio": 1.0,
                "alphabetic_ratio": 0.0,
                "numeric_table_ratio": 0.0,
                "medical_token_density": 0.0,
                "repeated_symbol_ratio": 0.0,
                "latin_ratio": 0.0,
                "cyrillic_ratio": 0.0,
                "script": "unknown",
            },
        )

    visible = [char for char in stripped if not char.isspace()]
    visible_count = max(1, len(visible))
    alpha_count = sum(1 for char in visible if char.isalpha())
    digit_count = sum(1 for char in visible if char.isdigit())
    table_symbol_count = sum(1 for char in visible if char in "|:;,.+-_/%\\")
    garbage_count = sum(1 for char in visible if _is_garbage_char(char))
    repeated_symbol_chars = sum(len(match.group(0)) for match in REPEATED_SYMBOL_RE.finditer(stripped))
    latin_count = sum(1 for char in visible if "A" <= char <= "Z" or "a" <= char <= "z")
    cyrillic_count = sum(1 for char in visible if "\u0400" <= char <= "\u04ff")
    tokens = [token.lower() for token in TOKEN_RE.findall(stripped)]
    medical_hits = sum(1 for token in tokens if token in MEDICAL_TOKENS)

    garbage_ratio = min(1.0, garbage_count / visible_count)
    alphabetic_ratio = alpha_count / visible_count
    numeric_table_ratio = min(1.0, (digit_count + table_symbol_count) / visible_count)
    medical_token_density = medical_hits / max(1, len(tokens))
    repeated_symbol_ratio = min(1.0, repeated_symbol_chars / max(1, len(stripped)))
    latin_ratio = latin_count / visible_count
    cyrillic_ratio = cyrillic_count / visible_count
    script = _script_band(latin_ratio=latin_ratio, cyrillic_ratio=cyrillic_ratio)

    length_score = min(length / 800.0, 1.0)
    token_score = min(len(tokens) / 35.0, 1.0)
    medical_score = min(medical_token_density * 4.0, 1.0)
    alpha_score = min(alphabetic_ratio / 0.55, 1.0)
    table_score = 1.0 - max(0.0, numeric_table_ratio - 0.45)
    noise_penalty = (garbage_ratio * 0.7) + (repeated_symbol_ratio * 0.8)
    score = (
        (0.22 * length_score)
        + (0.22 * token_score)
        + (0.22 * alpha_score)
        + (0.18 * medical_score)
        + (0.16 * table_score)
        - noise_penalty
    )
    score = round(max(0.0, min(1.0, score)), 3)

    warnings = _warnings(
        length=length,
        garbage_ratio=garbage_ratio,
        alphabetic_ratio=alphabetic_ratio,
        numeric_table_ratio=numeric_table_ratio,
        medical_token_density=medical_token_density,
        repeated_symbol_ratio=repeated_symbol_ratio,
        token_count=len(tokens),
    )
    band = _quality_band(score=score, length=length, warnings=warnings)

    return TextQualityAssessment(
        score=score,
        band=band,
        warnings=warnings,
        metrics={
            "text_length": length,
            "garbage_character_ratio": round(garbage_ratio, 4),
            "alphabetic_ratio": round(alphabetic_ratio, 4),
            "numeric_table_ratio": round(numeric_table_ratio, 4),
            "medical_token_density": round(medical_token_density, 4),
            "repeated_symbol_ratio": round(repeated_symbol_ratio, 4),
            "latin_ratio": round(latin_ratio, 4),
            "cyrillic_ratio": round(cyrillic_ratio, 4),
            "script": script,
        },
    )


def _is_garbage_char(char: str) -> bool:
    if char == "\ufffd":
        return True
    category_ord = ord(char)
    if category_ord < 32 and char not in {"\n", "\r", "\t"}:
        return True
    return False


def _script_band(*, latin_ratio: float, cyrillic_ratio: float) -> str:
    if latin_ratio >= 0.20 and cyrillic_ratio >= 0.20:
        return "mixed"
    if cyrillic_ratio >= 0.20:
        return "ru"
    if latin_ratio >= 0.20:
        return "en"
    return "unknown"


def _warnings(
    *,
    length: int,
    garbage_ratio: float,
    alphabetic_ratio: float,
    numeric_table_ratio: float,
    medical_token_density: float,
    repeated_symbol_ratio: float,
    token_count: int,
) -> list[str]:
    warnings: list[str] = []
    if length < 10:
        warnings.append("empty_or_near_empty_text")
    elif length < 120:
        warnings.append("low_text_length")
    if garbage_ratio > 0.03:
        warnings.append("garbage_characters")
    if alphabetic_ratio < 0.35:
        warnings.append("low_alphabetic_ratio")
    if numeric_table_ratio > 0.65:
        warnings.append("table_or_numeric_heavy")
    if medical_token_density < 0.03 and token_count >= 8:
        warnings.append("low_medical_token_density")
    if repeated_symbol_ratio > 0.03:
        warnings.append("repeated_symbol_noise")
    return warnings


def _quality_band(*, score: float, length: int, warnings: list[str]) -> str:
    if length < 10:
        return "empty"
    if score >= 0.72 and "garbage_characters" not in warnings and "repeated_symbol_noise" not in warnings:
        return "good"
    if score >= 0.48 and "empty_or_near_empty_text" not in warnings:
        return "usable_with_review"
    return "poor_ocr"
