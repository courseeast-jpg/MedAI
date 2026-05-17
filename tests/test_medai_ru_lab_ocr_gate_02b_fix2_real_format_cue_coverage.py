from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_lab_document_type,
    review_reason_for_result,
    safe_fallback_ocr_classification_diagnostic,
)
from app.main import item_status
from ingestion.cyrillic_ocr_gate import build_cyrillic_ocr_shadow_marker, run_local_cyrillic_ocr_fallback


NUMERIC_NATIVE_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
    ]
    * 40
)

# Synthetic OCR-like lab form using alternate safe headings that were not
# covered by the first fallback cue set. This is not source document text.
ALTERNATE_RUSSIAN_LAB_FORMAT = """
\u041a\u043b\u0438\u043d\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u043b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u0438\u044f
\u0417\u0430\u044f\u0432\u043a\u0430 \u043d\u0430 \u0438\u0441\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435
\u0413\u0435\u043c\u0430\u0442\u043e\u043b\u043e\u0433\u0438\u044f
\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440     \u0420\u0435\u0437-\u0442     \u0415\u0434. \u0438\u0437\u043c     \u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b
"""

NON_LAB_CYRILLIC_TEXT = """
\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435 \u0432\u0438\u0437\u0438\u0442\u0430 \u0438 \u043e\u0431\u0449\u0438\u0435 \u0437\u0430\u043c\u0435\u0442\u043a\u0438.
\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u0441\u0432\u043e\u0431\u043e\u0434\u043d\u044b\u0439 \u0442\u0435\u043a\u0441\u0442 \u0431\u0435\u0437 \u0442\u0430\u0431\u043b\u0438\u0446\u044b.
"""


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_alternate_russian_lab_format_now_has_safe_cue_keys_and_lab_result() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(ALTERNATE_RUSSIAN_LAB_FORMAT)

    assert diagnostic["cyrillic_detected"] is True
    assert diagnostic["cyrillic_char_count_bucket"] in {"medium", "high"}
    assert diagnostic["matched_lab_cue_keys"]
    assert {"generic_lab_form", "diagnostic_or_examination", "order_or_request", "table_header"}.issubset(
        set(diagnostic["matched_lab_cue_keys"])
    )
    assert diagnostic["matched_document_type_candidate"] == LAB_RESULT_LABEL
    assert diagnostic["classification_block_reason"] == "classified"
    assert classify_lab_document_type(ALTERNATE_RUSSIAN_LAB_FORMAT) == LAB_RESULT_LABEL


def test_alternate_format_fallback_metadata_is_safe_and_review_bound(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: ALTERNATE_RUSSIAN_LAB_FORMAT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": metadata["ocr_gate_fallback_document_type"],
        "confidence": 0.45,
    }

    assert metadata["ocr_gate_fallback_document_type"] == LAB_RESULT_LABEL
    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert item_status(item) == "review"
    assert "confidence is below the acceptance gate" in review_reason_for_result(
        document_type=item["document_type"],
        validation_status=item["validation_status"],
        confidence=item["confidence"],
        status=item["status"],
    )


def test_non_lab_cyrillic_text_still_remains_unknown() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(NON_LAB_CYRILLIC_TEXT)

    assert classify_lab_document_type(NON_LAB_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["matched_document_type_candidate"] == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["classification_block_reason"] in {
        "too_few_safe_lab_cue_keys",
        "classification_threshold_not_met",
    }


def test_raw_ocr_text_is_absent_from_diagnostic_and_metadata(tmp_path: Path) -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(ALTERNATE_RUSSIAN_LAB_FORMAT)
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: ALTERNATE_RUSSIAN_LAB_FORMAT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert ALTERNATE_RUSSIAN_LAB_FORMAT not in str(diagnostic)
    assert ALTERNATE_RUSSIAN_LAB_FORMAT not in str(metadata)
    assert "\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440" not in str(diagnostic)
    assert "raw_text" not in metadata
    assert "ocr_text" not in metadata


def test_external_api_confidence_and_acceptance_are_unchanged(tmp_path: Path) -> None:
    confidence = 0.45
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: ALTERNATE_RUSSIAN_LAB_FORMAT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
    assert "confidence" not in metadata
    assert confidence == 0.45
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
