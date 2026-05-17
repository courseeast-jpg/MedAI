from __future__ import annotations

import re
from typing import Any


_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_DIGIT_RE = re.compile(r"\d")
_TABLE_SIGNAL_RE = re.compile(r"\s{2,}|[|;:\t]")


def bucket_text_length(length: int) -> str:
    if length <= 0:
        return "none"
    if length < 80:
        return "tiny"
    if length < 500:
        return "short"
    if length < 2000:
        return "medium"
    return "long"


def bucket_density(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.05:
        return "low"
    if ratio < 0.25:
        return "medium"
    return "high"


def table_like_pattern_detected(text: str | None) -> bool:
    numeric_table_lines = 0
    for line in str(text or "").splitlines():
        if _DIGIT_RE.search(line) and _TABLE_SIGNAL_RE.search(line):
            numeric_table_lines += 1
    return numeric_table_lines >= 2


def build_cyrillic_ocr_shadow_marker(
    text: str | None,
    *,
    current_ocr_skipped: bool,
    language_context: str | None = None,
) -> dict[str, Any]:
    raw = str(text or "")
    compact = re.sub(r"\s+", "", raw)
    denominator = max(1, len(compact))
    cyrillic_count = len(_CYRILLIC_RE.findall(raw))
    digit_count = len(_DIGIT_RE.findall(raw))
    text_length_bucket = bucket_text_length(len(raw.strip()))
    cyrillic_density_bucket = bucket_density(cyrillic_count / denominator)
    digit_density_bucket = bucket_density(digit_count / denominator)
    table_like = table_like_pattern_detected(raw)
    decision = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket=text_length_bucket,
        digit_density_bucket=digit_density_bucket,
        cyrillic_density_bucket=cyrillic_density_bucket,
        table_like_pattern_detected=table_like,
        current_ocr_skipped=current_ocr_skipped,
        language_context=language_context,
    )
    return {
        **decision,
        "text_length_bucket": text_length_bucket,
        "digit_density_bucket": digit_density_bucket,
        "cyrillic_density_bucket": cyrillic_density_bucket,
        "table_like_pattern_detected": table_like,
    }


def cyrillic_ocr_shadow_gate_decision(
    *,
    text_length_bucket: str,
    digit_density_bucket: str,
    cyrillic_density_bucket: str,
    table_like_pattern_detected: bool,
    current_ocr_skipped: bool,
    language_context: str | None = None,
) -> dict[str, Any]:
    has_substantial_text = text_length_bucket in {"medium", "long"}
    has_table_digits = digit_density_bucket in {"medium", "high"} and bool(table_like_pattern_detected)
    missing_cyrillic = cyrillic_density_bucket == "none"
    sparse_or_empty = text_length_bucket in {"none", "tiny", "short"}
    review_only = True
    recommended = bool(has_substantial_text and has_table_digits and missing_cyrillic and current_ocr_skipped)
    visibility = "not_applicable"
    reason = "not_recommended"
    if recommended:
        visibility = "incomplete"
        reason = "numeric_table_text_without_cyrillic"
    elif not missing_cyrillic:
        visibility = "visible"
        reason = "cyrillic_visible"
    elif sparse_or_empty:
        visibility = "not_applicable"
        reason = "insufficient_native_text_for_shadow_gate"
    elif not has_table_digits:
        visibility = "unknown"
        reason = "numeric_table_signal_not_strong"
    elif not current_ocr_skipped:
        visibility = "unknown"
        reason = "ocr_already_attempted"
    return {
        "language_text_visibility": visibility,
        "cyrillic_ocr_recommended": recommended,
        "ocr_gate_reason": reason,
        "review_only": review_only,
        "auto_accept_allowed": False,
        "ocr_fallback_executed": False,
        "language_context": language_context or "unknown",
    }
