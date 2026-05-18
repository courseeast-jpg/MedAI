from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.document_type_registry import (
    IMAGING_REPORT_LABEL,
    LAB_RESULT_LABEL,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_document_family,
    document_family_classification_diagnostic,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts import run_medai_doc_type_batch_eval as eval01
from tests.test_medai_doc_type_family_01 import RUSSIAN_MRI_REPORT
from tests.test_medai_doc_type_family_01_fix2_imaging_treatment_conflict import (
    RUSSIAN_LAB_TEXT,
    RUSSIAN_TREATMENT_SCHEDULE,
)


LATIN_LAB_TABLE = """
Test Name        Result     Units     Reference Range
Glucose          92         mg/dL     70-99
Creatinine       0.9        mg/dL     0.6-1.2
"""

LATIN_PANEL_TABLE = """
CBC
Component        Value      Unit      Ref Range
WBC              6.1        10^3/uL   4.0-10.0
RBC              4.8        10^6/uL   4.2-5.8
Hemoglobin       14.2       g/dL      13.0-17.0
"""

GENERIC_TABLE = """
Item             Count      Date
Invoice          12         01/02/2024
Service          3          01/03/2024
"""

TREATMENT_DATE_GRID = """
Treatment Plan
Date      Mon     Tue     Wed
Dose      1       1       1
Morning   yes     yes     yes
Evening   no      yes     no
"""

WEAK_AMBIGUOUS_TABLE = """
Schedule Result Date
Plan     Value  01/02
Course   Unit   01/03
"""


def test_latin_lab_table_with_value_unit_reference_structure_returns_lab_result() -> None:
    diagnostic = document_family_classification_diagnostic(LATIN_LAB_TABLE)

    assert diagnostic["candidate_family"] == LAB_RESULT_LABEL
    assert "lab_table_column_structure" in diagnostic["matched_family_cue_keys"]
    assert "analyte_value_unit_pattern" in diagnostic["matched_family_cue_keys"]
    assert "reference_range_column_pattern" in diagnostic["matched_family_cue_keys"]
    assert diagnostic["review_only"] is True
    assert diagnostic["auto_accept_allowed"] is False


def test_latin_panel_table_with_multiple_safe_structure_cues_returns_lab_result() -> None:
    diagnostic = document_family_classification_diagnostic(LATIN_PANEL_TABLE)

    assert diagnostic["candidate_family"] == LAB_RESULT_LABEL
    assert "laboratory_panel_abbreviation_latin" in diagnostic["matched_family_cue_keys"]
    assert "lab_table_column_structure" in diagnostic["matched_family_cue_keys"]
    assert classify_document_family(LATIN_PANEL_TABLE) == LAB_RESULT_LABEL


def test_generic_table_without_lab_structure_remains_unknown() -> None:
    diagnostic = document_family_classification_diagnostic(GENERIC_TABLE)

    assert diagnostic["candidate_family"] == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["classification_block_reason"] == "too_few_safe_family_cue_keys"


def test_treatment_schedule_date_grid_does_not_become_lab_result() -> None:
    diagnostic = document_family_classification_diagnostic(TREATMENT_DATE_GRID)

    assert diagnostic["candidate_family"] == TREATMENT_PLAN_LABEL
    assert diagnostic["candidate_family"] != LAB_RESULT_LABEL
    assert diagnostic["auto_accept_allowed"] is False


def test_weak_lab_treatment_table_remains_unknown_or_ambiguous_review_bound() -> None:
    diagnostic = document_family_classification_diagnostic(WEAK_AMBIGUOUS_TABLE)

    assert diagnostic["candidate_family"] == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["classification_block_reason"] in {
        "too_few_safe_family_cue_keys",
        "ambiguous_family_candidates",
    }
    assert diagnostic["review_only"] is True


def test_existing_russian_lab_treatment_and_imaging_regressions_pass() -> None:
    assert classify_document_family(RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL
    assert classify_document_family(RUSSIAN_TREATMENT_SCHEDULE) == TREATMENT_PLAN_LABEL
    assert classify_document_family(RUSSIAN_MRI_REPORT) == IMAGING_REPORT_LABEL


def test_doc_type_family_03_public_report_payload_is_privacy_clean() -> None:
    payload = {
        "report": "safe aggregate only",
        "cue_categories_added": [
            "lab_table_column_structure",
            "analyte_value_unit_pattern",
            "reference_range_column_pattern",
        ],
        "external_api_used": False,
        "auto_accept_expanded": False,
        "raw_ocr_text_in_public_reports": False,
    }

    assert check_public_report_payload(json.loads(json.dumps(payload))).passed is True


def test_batch_eval_recomputes_safe_family_diagnostic_when_existing_candidate_is_unknown(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.pdf"
    source.write_text("placeholder", encoding="utf-8")
    result = SimpleNamespace(
        outcome="queued_for_review",
        validation_status="needs_review",
        extractor_result={
            "raw_text": LATIN_LAB_TABLE,
            "document_type": UNKNOWN_DOCUMENT_LABEL,
            "document_family_classification_diagnostic": {
                "candidate_family": UNKNOWN_DOCUMENT_LABEL,
                "matched_family_cue_keys": [],
                "matched_language_cue_groups": [],
                "ambiguous_candidates": [],
                "classification_block_reason": "unknown",
                "conflict_resolution_reason": "none",
                "review_only": True,
                "auto_accept_allowed": False,
            },
            "language_text_visibility": "visible",
            "ocr_gate_reason": "not_recommended",
            "external_api_used": False,
            "auto_accept_allowed": False,
        },
        audit={"document_type": UNKNOWN_DOCUMENT_LABEL, "ocr_quality_band": "readable"},
    )

    record = eval01.build_safe_file_record(source, safe_id="file_001", result=result)

    assert record["predicted_document_type"] == LAB_RESULT_LABEL
    assert "lab_table_column_structure" in record["matched_safe_cue_keys"]
    assert record["review_status"] == "review"
    assert record["auto_accept_allowed"] is False
    assert record["external_api_used"] is False


def test_batch_eval_keeps_latin_lab_family_classification_review_bound(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.pdf"
    source.write_text("placeholder", encoding="utf-8")
    result = SimpleNamespace(
        outcome="written",
        validation_status="accepted",
        extractor_result={
            "raw_text": LATIN_LAB_TABLE,
            "document_type": UNKNOWN_DOCUMENT_LABEL,
            "language_text_visibility": "visible",
            "ocr_gate_reason": "not_recommended",
            "external_api_used": False,
            "auto_accept_allowed": False,
        },
        audit={"document_type": UNKNOWN_DOCUMENT_LABEL, "ocr_quality_band": "readable"},
    )

    record = eval01.build_safe_file_record(source, safe_id="file_001", result=result)
    report = eval01.build_report([record])

    assert record["predicted_document_type"] == LAB_RESULT_LABEL
    assert record["raw_review_status"] == "accepted"
    assert record["review_status"] == "review"
    assert record["status_mapping_action"] == "normalized_review_bound_family_runtime_accepted_to_review"
    assert report["accepted_count"] == 0
    assert report["auto_accept_allowed_count"] == 0
