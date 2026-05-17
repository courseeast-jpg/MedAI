from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import (
    IMAGING_REPORT_LABEL,
    LAB_RESULT_LABEL,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_lab_document_type,
    review_reason_for_result,
    safe_fallback_ocr_classification_diagnostic,
)
from app.main import advanced_diagnostic_fields, canonical_run_result_record, item_status, operator_result_explanation
from app.test_launcher import _runtime_document_type_candidate, runtime_cyrillic_ocr_marker_for_result
from ingestion.cyrillic_ocr_gate import build_cyrillic_ocr_shadow_marker, run_local_cyrillic_ocr_fallback


NUMERIC_NATIVE_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
    ]
    * 40
)

RUSSIAN_MRI_OCR_TEXT = """
\u041c\u0420\u0422 \u0433\u043e\u043b\u043e\u0432\u043d\u043e\u0433\u043e \u043c\u043e\u0437\u0433\u0430
\u0410\u043f\u043f\u0430\u0440\u0430\u0442
\u041c\u0420-\u0442\u043e\u043c\u043e\u0433\u0440\u0430\u043c\u043c\u044b
T1 T2 FLAIR DWI
\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435
\u0417\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435
"""

RUSSIAN_LAB_TEXT = """
\u0411\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b
\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442
\u0420\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f
\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c
\u0413\u043b\u044e\u043a\u043e\u0437\u0430
"""

RUSSIAN_TREATMENT_TEXT = """
\u041f\u043b\u0430\u043d \u043b\u0435\u0447\u0435\u043d\u0438\u044f
\u0414\u0430\u0442\u0430 01.01 02.01 03.01
\u0424\u0438\u0437\u0438\u043e\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b
\u0420\u0435\u0436\u0438\u043c \u0438 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438
"""

NON_MEDICAL_CYRILLIC_TEXT = "\u0417\u0430\u043c\u0435\u0442\u043a\u0438 \u0434\u043b\u044f \u0432\u0441\u0442\u0440\u0435\u0447\u0438."


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_runtime_fallback_russian_mri_record_returns_imaging_report(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus", "eng"],
        ocr_runner=lambda _path, _language: RUSSIAN_MRI_OCR_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )
    marker = runtime_cyrillic_ocr_marker_for_result(
        {
            "document_type": UNKNOWN_DOCUMENT_LABEL,
            "language_text_visibility": "incomplete",
            "ocr_gate_reason": "numeric_table_text_without_cyrillic",
            **metadata,
        }
    )

    assert metadata["ocr_gate_fallback_document_type"] == IMAGING_REPORT_LABEL
    assert marker["document_family_classification_diagnostic"]["candidate_family"] == IMAGING_REPORT_LABEL
    assert _runtime_document_type_candidate({"document_type": UNKNOWN_DOCUMENT_LABEL}, {}, marker) == IMAGING_REPORT_LABEL


def test_run_review_main_card_and_advanced_details_use_same_document_type(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus", "eng"],
        ocr_runner=lambda _path, _language: RUSSIAN_MRI_OCR_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )
    record = canonical_run_result_record(
        {
            "status": "review",
            "validation_status": "rejected",
            "document_type": metadata["ocr_gate_fallback_document_type"],
            "document_family_classification_diagnostic": metadata["document_family_classification_diagnostic"],
            "external_api_used": False,
        }
    )
    advanced = advanced_diagnostic_fields(record)

    assert record["document_type"] == IMAGING_REPORT_LABEL
    assert advanced["document_type"] == IMAGING_REPORT_LABEL
    assert advanced["document_family_classification_diagnostic"]["candidate_family"] == IMAGING_REPORT_LABEL
    assert item_status(record) == "review"


def test_imaging_review_reason_and_operator_wording_are_review_bound() -> None:
    reason = review_reason_for_result(
        document_type=IMAGING_REPORT_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )
    explanation = operator_result_explanation(IMAGING_REPORT_LABEL)

    assert "Imaging findings and conclusions were not interpreted or accepted" in reason
    assert "Human review is required" in reason
    assert "imaging-report style document" in explanation
    assert "Imaging findings and conclusions were not interpreted or accepted" in explanation


def test_existing_lab_treatment_and_unknown_behavior_is_preserved() -> None:
    assert classify_lab_document_type(RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL
    assert classify_lab_document_type(RUSSIAN_TREATMENT_TEXT) == TREATMENT_PLAN_LABEL
    assert classify_lab_document_type(NON_MEDICAL_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL


def test_fallback_metadata_has_no_raw_ocr_text_or_external_api(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: RUSSIAN_MRI_OCR_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )

    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
    assert RUSSIAN_MRI_OCR_TEXT not in str(metadata)
    assert "raw_text" not in metadata
    assert "ocr_text" not in metadata
