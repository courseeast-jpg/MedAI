from __future__ import annotations

from pathlib import Path

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    MEDICATION_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_lab_document_type,
    safe_fallback_ocr_classification_diagnostic,
    safe_fallback_ocr_treatment_classification_diagnostic,
)
from app.test_launcher import _fallback_diagnostic_document_type, runtime_cyrillic_ocr_marker_for_result
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
SYNTHETIC_RUSSIAN_SCHEDULE_TEXT = """
\u041f\u0420\u0415\u041f\u0410\u0420\u0410\u0422\u042b / \u0414\u0410\u0422\u0410
01.01 02.01 03.01 04.01
\u0421\u0445\u0435\u043c\u0430 \u043f\u0440\u0438\u0435\u043c\u0430
\u0424\u0418\u0417\u0418\u041e\u041f\u0420\u041e\u0426\u0415\u0414\u0423\u0420\u042b
\u0414\u0438\u0435\u0442\u0430 \u0438 \u0440\u0435\u0436\u0438\u043c
"""

SYNTHETIC_RUSSIAN_LAB_TEXT = """
\u0411\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b
\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442
\u0420\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f
\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c
\u0413\u043b\u044e\u043a\u043e\u0437\u0430
"""

NON_MEDICAL_CYRILLIC_TEXT = (
    "\u041e\u0431\u0449\u0438\u0435 \u0437\u0430\u043c\u0435\u0442\u043a\u0438 "
    "\u0431\u0435\u0437 \u043c\u0435\u0434\u0438\u0446\u0438\u043d\u0441\u043a\u0438\u0445 "
    "\u0440\u0430\u0437\u0434\u0435\u043b\u043e\u0432."
)


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_fallback_runtime_metadata_includes_treatment_diagnostic(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus", "eng"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_SCHEDULE_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
        treatment_classification_diagnostic_builder=safe_fallback_ocr_treatment_classification_diagnostic,
    )

    diagnostic = metadata["ocr_gate_fallback_treatment_classification_diagnostic"]

    assert metadata["ocr_gate_fallback_document_type"] == MEDICATION_PLAN_LABEL
    assert diagnostic["matched_document_type_candidate"] == MEDICATION_PLAN_LABEL
    assert {"medication_schedule_header", "date_grid", "administration_schedule_pattern"}.issubset(
        set(diagnostic["matched_treatment_cue_keys"])
    )
    assert diagnostic["classification_block_reason"] == "classified"


def test_lab_cue_empty_schedule_still_classifies_by_treatment_cues() -> None:
    lab_diagnostic = safe_fallback_ocr_classification_diagnostic(SYNTHETIC_RUSSIAN_SCHEDULE_TEXT)
    treatment_diagnostic = safe_fallback_ocr_treatment_classification_diagnostic(SYNTHETIC_RUSSIAN_SCHEDULE_TEXT)

    assert lab_diagnostic["matched_lab_cue_keys"] == []
    assert treatment_diagnostic["matched_treatment_cue_keys"]
    assert treatment_diagnostic["matched_document_type_candidate"] == MEDICATION_PLAN_LABEL


def test_run_review_marker_copies_treatment_diagnostic() -> None:
    treatment_diagnostic = safe_fallback_ocr_treatment_classification_diagnostic(SYNTHETIC_RUSSIAN_SCHEDULE_TEXT)
    marker = runtime_cyrillic_ocr_marker_for_result(
        {
            "language_text_visibility": "incomplete",
            "ocr_gate_reason": "numeric_table_text_without_cyrillic",
            "cyrillic_ocr_recommended": True,
            "ocr_gate_fallback_executed": True,
            "ocr_gate_fallback_cyrillic_detected": True,
            "ocr_gate_fallback_text_visibility": "recovered",
            "ocr_gate_fallback_review_only": True,
            "ocr_gate_fallback_auto_accept_allowed": False,
            "ocr_gate_fallback_treatment_classification_diagnostic": treatment_diagnostic,
        }
    )

    assert marker["ocr_gate_fallback_treatment_classification_diagnostic"] == treatment_diagnostic
    assert _fallback_diagnostic_document_type(marker) == MEDICATION_PLAN_LABEL


def test_russian_lab_text_still_returns_lab_result_not_medication_plan() -> None:
    diagnostic = safe_fallback_ocr_treatment_classification_diagnostic(SYNTHETIC_RUSSIAN_LAB_TEXT)

    assert classify_lab_document_type(SYNTHETIC_RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL
    assert diagnostic["matched_document_type_candidate"] == UNKNOWN_DOCUMENT_LABEL


def test_non_medical_cyrillic_remains_unknown() -> None:
    diagnostic = safe_fallback_ocr_treatment_classification_diagnostic(NON_MEDICAL_CYRILLIC_TEXT)

    assert classify_lab_document_type(NON_MEDICAL_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["matched_document_type_candidate"] == UNKNOWN_DOCUMENT_LABEL


def test_fallback_remains_review_only_and_no_auto_accept(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_SCHEDULE_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
        treatment_classification_diagnostic_builder=safe_fallback_ocr_treatment_classification_diagnostic,
    )

    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "confidence" not in metadata


def test_external_api_and_raw_ocr_text_are_absent(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus"],
        ocr_runner=lambda _path, _language: SYNTHETIC_RUSSIAN_SCHEDULE_TEXT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
        treatment_classification_diagnostic_builder=safe_fallback_ocr_treatment_classification_diagnostic,
    )

    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
    assert SYNTHETIC_RUSSIAN_SCHEDULE_TEXT not in str(metadata)
    assert "raw_text" not in metadata
    assert "ocr_text" not in metadata
