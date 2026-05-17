from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    MEDICATION_PLAN_LABEL,
    TREATMENT_PLAN_LABEL,
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

# Synthetic schedule structure only. No real medication names, doses, or advice.
SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE = """
\u041f\u0420\u0415\u041f\u0410\u0420\u0410\u0422\u042b / \u0414\u0410\u0422\u0410
01.01 02.01 03.01 04.01
\u0421\u0445\u0435\u043c\u0430 \u043f\u0440\u0438\u0435\u043c\u0430
\u0424\u0418\u0417\u0418\u041e\u041f\u0420\u041e\u0426\u0415\u0414\u0423\u0420\u042b
\u0414\u0438\u0435\u0442\u0430 \u0438 \u0440\u0435\u0436\u0438\u043c
"""

SYNTHETIC_RUSSIAN_TREATMENT_SCHEDULE = """
\u041f\u043b\u0430\u043d \u043b\u0435\u0447\u0435\u043d\u0438\u044f
\u0414\u0430\u0442\u0430 01.01 02.01 03.01
\u0424\u0438\u0437\u0438\u043e\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b
\u0420\u0435\u0436\u0438\u043c \u0438 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438
"""

SYNTHETIC_RUSSIAN_LAB_TEXT = """
\u0411\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b: \u0441\u044b\u0432\u043e\u0440\u043e\u0442\u043a\u0430
\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440 \u0420\u0435\u0437-\u0442 \u0415\u0434. \u0438\u0437\u043c \u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b
\u0413\u043b\u044e\u043a\u043e\u0437\u0430
"""

NON_MEDICAL_CYRILLIC_TEXT = "\u0417\u0430\u043c\u0435\u0442\u043a\u0438 \u043e \u0432\u0441\u0442\u0440\u0435\u0447\u0435 \u0438 \u043e\u0431\u0449\u0438\u0439 \u0441\u0432\u043e\u0431\u043e\u0434\u043d\u044b\u0439 \u0442\u0435\u043a\u0441\u0442."


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_synthetic_russian_medication_schedule_is_medication_plan() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE)

    assert classify_lab_document_type(SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE) == MEDICATION_PLAN_LABEL
    assert diagnostic["matched_document_type_candidate"] == MEDICATION_PLAN_LABEL
    assert {"medication_schedule_header", "date_grid", "physiotherapy_section"}.issubset(
        set(diagnostic["matched_treatment_cue_keys"])
    )
    assert diagnostic["classification_block_reason"] == "classified"


def test_synthetic_russian_treatment_schedule_is_treatment_plan() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(SYNTHETIC_RUSSIAN_TREATMENT_SCHEDULE)

    assert classify_lab_document_type(SYNTHETIC_RUSSIAN_TREATMENT_SCHEDULE) == TREATMENT_PLAN_LABEL
    assert diagnostic["matched_document_type_candidate"] == TREATMENT_PLAN_LABEL
    assert {"date_grid", "physiotherapy_section", "diet_recommendation_section"}.issubset(
        set(diagnostic["matched_treatment_cue_keys"])
    )


def test_russian_lab_text_still_returns_lab_result() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(SYNTHETIC_RUSSIAN_LAB_TEXT)

    assert classify_lab_document_type(SYNTHETIC_RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL
    assert diagnostic["matched_document_type_candidate"] == LAB_RESULT_LABEL
    assert diagnostic["matched_lab_cue_keys"]


def test_non_medical_cyrillic_text_remains_unknown() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(NON_MEDICAL_CYRILLIC_TEXT)

    assert classify_lab_document_type(NON_MEDICAL_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["matched_document_type_candidate"] == UNKNOWN_DOCUMENT_LABEL


def test_treatment_schedule_classification_is_review_only(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["eng", "rus"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": metadata["ocr_gate_fallback_document_type"],
        "confidence": 0.45,
    }

    assert metadata["ocr_gate_fallback_document_type"] == MEDICATION_PLAN_LABEL
    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert item_status(item) == "review"
    assert "Human review is required" in review_reason_for_result(
        document_type=item["document_type"],
        validation_status=item["validation_status"],
        confidence=item["confidence"],
        status=item["status"],
    )


def test_external_api_and_acceptance_are_unchanged(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "confidence" not in metadata


def test_raw_ocr_text_is_absent_from_public_metadata(tmp_path: Path) -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE)
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE not in str(diagnostic)
    assert SYNTHETIC_RUSSIAN_MEDICATION_SCHEDULE not in str(metadata)
    assert "\u041f\u0420\u0415\u041f\u0410\u0420\u0410\u0422\u042b" not in str(diagnostic)
    assert "raw_text" not in metadata
    assert "ocr_text" not in metadata
