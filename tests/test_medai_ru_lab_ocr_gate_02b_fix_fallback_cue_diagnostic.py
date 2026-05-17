from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_lab_document_type,
    safe_fallback_ocr_classification_diagnostic,
)
from ingestion.cyrillic_ocr_gate import build_cyrillic_ocr_shadow_marker, run_local_cyrillic_ocr_fallback


NUMERIC_NATIVE_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
    ]
    * 40
)

FALLBACK_RUSSIAN_LAB_TEXT = """
\u0411\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b: \u0441\u044b\u0432\u043e\u0440\u043e\u0442\u043a\u0430
\u041d\u0430\u0438\u043c\u0435\u043d\u043e\u0432\u0430\u043d\u0438\u0435    \u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442    \u0415\u0434. \u0438\u0437\u043c.    \u0420\u0435\u0444. \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f
\u0413\u043b\u044e\u043a\u043e\u0437\u0430      5.1    \u043c\u043c\u043e\u043b\u044c/\u043b
\u041a\u0440\u0435\u0430\u0442\u0438\u043d\u0438\u043d  80     \u043c\u043a\u043c\u043e\u043b\u044c/\u043b
"""

NON_LAB_CYRILLIC_TEXT = "\u042d\u0442\u043e \u043e\u0431\u0449\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0431\u0435\u0437 \u0442\u0430\u0431\u043b\u0438\u0447\u043d\u044b\u0445 \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u044b\u0445 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432."


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_fallback_recovered_russian_lab_text_now_classifies_as_lab_result() -> None:
    assert classify_lab_document_type(FALLBACK_RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL


def test_safe_fallback_diagnostic_explains_lab_cues_without_raw_text() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(FALLBACK_RUSSIAN_LAB_TEXT)

    assert diagnostic["cyrillic_detected"] is True
    assert diagnostic["cyrillic_char_count_bucket"] in {"medium", "high"}
    assert diagnostic["matched_document_type_candidate"] == LAB_RESULT_LABEL
    assert diagnostic["classification_block_reason"] == "classified"
    assert {"specimen_or_biomaterial", "result_or_report", "table_header", "common_analyte"}.issubset(
        set(diagnostic["matched_lab_cue_keys"])
    )
    assert FALLBACK_RUSSIAN_LAB_TEXT not in str(diagnostic)
    assert "\u0413\u043b\u044e\u043a\u043e\u0437\u0430" not in str(diagnostic)


def test_fallback_metadata_carries_safe_classification_diagnostic(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: FALLBACK_RUSSIAN_LAB_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert metadata["ocr_gate_fallback_executed"] is True
    assert metadata["ocr_gate_fallback_document_type"] == LAB_RESULT_LABEL
    diagnostic = metadata["ocr_gate_fallback_classification_diagnostic"]
    assert diagnostic["matched_document_type_candidate"] == LAB_RESULT_LABEL
    assert FALLBACK_RUSSIAN_LAB_TEXT not in str(metadata)


def test_fallback_remains_review_only_and_never_auto_accepts(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: FALLBACK_RUSSIAN_LAB_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "accepted" not in str(metadata).lower()


def test_external_api_is_not_used_or_enabled(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: FALLBACK_RUSSIAN_LAB_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert metadata["ocr_gate_fallback_engine"] == "tesseract_local"
    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()


def test_unsupported_non_lab_cyrillic_text_remains_unknown() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(NON_LAB_CYRILLIC_TEXT)

    assert classify_lab_document_type(NON_LAB_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["matched_document_type_candidate"] == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["classification_block_reason"] in {
        "too_few_safe_lab_cue_keys",
        "classification_threshold_not_met",
    }
