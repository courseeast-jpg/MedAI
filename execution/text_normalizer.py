"""Text normalization before local extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass


LANGUAGE_TERM_MAP: tuple[tuple[str, str], ...] = (
    (r"\bUR[O0]KULTUR(?:E|A|\u00cb)?\b", "Urine Culture"),
    (r"\bNEGAT[I1]V[EA]?\b", "Negative"),
    (r"\bVERDHE\b", "Yellow"),
)

OCR_WORD_FIXES: tuple[tuple[str, str], ...] = (
    (r"\bEXARNINATION\b", "EXAMINATION"),
    (r"\bMICR0SC0PIC\b", "MICROSCOPIC"),
    (r"\bBlLIRUBIN\b", "BILIRUBIN"),
)


@dataclass(frozen=True)
class TextNormalizationResult:
    text: str
    applied: bool
    preview: str


def normalize_text(text: str) -> TextNormalizationResult:
    original = str(text or "")
    normalized = original.replace("\ufffd", " ")
    normalized = re.sub(r"([|_=~*#])\1{2,}", " ", normalized)
    normalized = re.sub(r"-{4,}", " ", normalized)
    normalized = re.sub(r"[^\S\r\n]+", " ", normalized)

    for pattern, replacement in OCR_WORD_FIXES:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = _fix_mixed_ocr_tokens(normalized)
    normalized = _fix_rn_artifacts(normalized)

    for pattern, replacement in LANGUAGE_TERM_MAP:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return TextNormalizationResult(
        text=normalized,
        applied=normalized != original.strip(),
        preview=re.sub(r"\s+", " ", normalized)[:300],
    )


def _fix_mixed_ocr_tokens(text: str) -> str:
    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if not any(char.isalpha() for char in token) or not any(char.isdigit() for char in token):
            return token
        if re.match(r"^\d+(?:\.\d+)?(?:mg|mcg|ml|g|units?)$", token, re.IGNORECASE):
            return token
        return token.replace("0", "O").replace("1", "I")

    return re.sub(r"\b[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*\d[A-Za-z0-9]*\b|\b[A-Za-z0-9]*\d[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*\b", replace_token, text)


def _fix_rn_artifacts(text: str) -> str:
    fixes = {
        "exarnination": "examination",
        "Exarnination": "Examination",
        "EXARNINATION": "EXAMINATION",
        "microorganisrns": "microorganisms",
        "Microorganisrns": "Microorganisms",
        "MICROORGANISRNS": "MICROORGANISMS",
    }
    for source, target in fixes.items():
        text = text.replace(source, target)
    return text
