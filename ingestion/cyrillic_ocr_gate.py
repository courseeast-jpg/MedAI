from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
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
    has_numeric_or_table_signal = digit_density_bucket in {"medium", "high"} or bool(table_like_pattern_detected)
    missing_cyrillic = cyrillic_density_bucket == "none"
    sparse_or_empty = text_length_bucket in {"none", "tiny", "short"}
    review_only = True
    recommended = bool(
        has_substantial_text
        and has_numeric_or_table_signal
        and missing_cyrillic
        and current_ocr_skipped
    )
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
    elif not has_numeric_or_table_signal:
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


def should_run_local_cyrillic_ocr_fallback(marker: dict[str, Any], *, local_only: bool = True) -> bool:
    return bool(
        local_only
        and marker.get("cyrillic_ocr_recommended") is True
        and marker.get("auto_accept_allowed") is False
    )


def choose_tesseract_cyrillic_language(languages: list[str] | tuple[str, ...]) -> str | None:
    available = {str(language).strip() for language in languages}
    if "rus" not in available:
        return None
    if "eng" in available:
        return "rus+eng"
    return "rus"


def list_tesseract_languages() -> list[str]:
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    lines = (completed.stdout or completed.stderr or "").splitlines()
    return [line.strip() for line in lines if line.strip() and "available languages" not in line.lower()]


def run_local_cyrillic_ocr_fallback(
    pdf_path: Path,
    marker: dict[str, Any],
    *,
    local_only: bool = True,
    language_lister=list_tesseract_languages,
    ocr_runner=None,
    document_type_classifier=None,
    classification_diagnostic_builder=None,
    treatment_classification_diagnostic_builder=None,
) -> dict[str, Any]:
    if not should_run_local_cyrillic_ocr_fallback(marker, local_only=local_only):
        return _fallback_metadata(
            executed=False,
            attempted=False,
            text_visibility="not_applicable",
            error_bucket="gate_not_recommended",
        )

    languages = list(language_lister())
    language = choose_tesseract_cyrillic_language(languages)
    if language is None:
        return _fallback_metadata(
            executed=False,
            attempted=False,
            text_visibility="unavailable",
            error_bucket="russian_language_unavailable",
        )

    try:
        if ocr_runner is None:
            ocr_text = _ocr_pdf_with_tesseract(pdf_path, language=language)
        else:
            ocr_text = str(ocr_runner(pdf_path, language) or "")
    except Exception:
        return _fallback_metadata(
            executed=False,
            attempted=True,
            language=language,
            text_visibility="unavailable",
            error_bucket="local_ocr_failed",
        )

    cyrillic_detected = bool(_CYRILLIC_RE.search(ocr_text))
    visibility = "recovered" if cyrillic_detected else "not_recovered"
    document_type = None
    if cyrillic_detected and document_type_classifier is not None:
        document_type = document_type_classifier(ocr_text)
    classification_diagnostic = None
    if classification_diagnostic_builder is not None:
        classification_diagnostic = classification_diagnostic_builder(ocr_text)
    treatment_classification_diagnostic = None
    if treatment_classification_diagnostic_builder is not None:
        treatment_classification_diagnostic = treatment_classification_diagnostic_builder(ocr_text)
    return _fallback_metadata(
        executed=True,
        attempted=True,
        language=language,
        cyrillic_detected=cyrillic_detected,
        text_visibility=visibility,
        document_type=document_type,
        classification_diagnostic=classification_diagnostic,
        treatment_classification_diagnostic=treatment_classification_diagnostic,
    )


def _ocr_pdf_with_tesseract(pdf_path: Path, *, language: str) -> str:
    import fitz

    pages: list[str] = []
    document = fitz.open(pdf_path)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            for index, page in enumerate(document, start=1):
                image_path = temp_dir_path / f"page_{index}.png"
                output_base = temp_dir_path / f"page_{index}_ocr"
                image_path.write_bytes(page.get_pixmap(dpi=200).tobytes("png"))
                completed = subprocess.run(
                    ["tesseract", str(image_path), str(output_base), "-l", language, "--psm", "6"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    continue
                text_path = output_base.with_suffix(".txt")
                if text_path.exists():
                    pages.append(text_path.read_text(encoding="utf-8", errors="ignore"))
    finally:
        document.close()
    return "\n".join(pages)


def _fallback_metadata(
    *,
    executed: bool,
    attempted: bool,
    language: str | None = None,
    cyrillic_detected: bool = False,
    text_visibility: str,
    error_bucket: str | None = None,
    document_type: str | None = None,
    classification_diagnostic: dict[str, Any] | None = None,
    treatment_classification_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "ocr_gate_fallback_executed": bool(executed),
        "ocr_gate_fallback_attempted": bool(attempted),
        "ocr_gate_fallback_engine": "tesseract_local" if executed else None,
        "ocr_gate_fallback_language": language,
        "ocr_gate_fallback_cyrillic_detected": bool(cyrillic_detected),
        "ocr_gate_fallback_text_visibility": text_visibility,
        "ocr_gate_fallback_review_only": True,
        "ocr_gate_fallback_auto_accept_allowed": False,
    }
    if error_bucket:
        metadata["ocr_gate_fallback_error_bucket"] = error_bucket
    if document_type:
        metadata["ocr_gate_fallback_document_type"] = document_type
    if classification_diagnostic:
        metadata["ocr_gate_fallback_classification_diagnostic"] = classification_diagnostic
    if treatment_classification_diagnostic:
        metadata["ocr_gate_fallback_treatment_classification_diagnostic"] = treatment_classification_diagnostic
    return metadata
