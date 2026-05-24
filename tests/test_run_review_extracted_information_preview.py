"""Focused tests for the Streamlit-free UI render-plan helper.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01.
"""
from __future__ import annotations

import pytest

from app.extracted_information_preview import (
    DISCLAIMER_LINE,
    EMPTY_STATE_LINE,
    SECTION_HEADING,
    build_extracted_information_preview_plan,
)
from execution.extracted_medical_facts import (
    extract_lab_observation_entities,
    summarize_extracted_facts_for_public_report,
)


def test_empty_item_returns_empty_state_plan():
    plan = build_extracted_information_preview_plan({})
    assert plan["section_heading"] == SECTION_HEADING
    assert plan["row_count"] == 0
    assert plan["message"] == EMPTY_STATE_LINE
    assert plan["disclaimer_line"] == EMPTY_STATE_LINE
    assert plan["auto_accept_allowed"] is False
    assert plan["review_required"] is True
    assert plan["counts"]["structured_facts_extracted"] == 0
    assert plan["counts"]["written_to_mkb"] == 0
    assert plan["counts"]["needs_review"] == 0


def test_plan_renders_lab_facts_with_safe_columns():
    text = (
        "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
        "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
        "Hemoglobin: 13.5 g/dL\n"
    )
    entities = extract_lab_observation_entities(text, {"document_type": "lab_report"})
    summary = summarize_extracted_facts_for_public_report(entities)
    item = {
        "extracted_medical_facts_preview_safe": summary["extracted_medical_facts_preview_safe"],
        "extracted_medical_fact_count": summary["extracted_medical_fact_count"],
        "extraction_to_mkb_written_count": 0,
        "extraction_to_mkb_review_count": summary["extracted_medical_fact_count"],
    }
    plan = build_extracted_information_preview_plan(item)
    assert plan["row_count"] == summary["extracted_medical_fact_count"]
    assert plan["message"] == DISCLAIMER_LINE
    expected = {
        "Type",
        "Test / observation",
        "Value",
        "Unit",
        "Reference range",
        "Flag",
        "Confidence",
        "Review status",
        "MKB status",
    }
    assert expected.issubset(set(plan["columns"]))
    for row in plan["rows"]:
        # Public-safe contract: no raw line / source line / private path fields.
        for forbidden_key in ("raw_line", "source_line", "private_path", "file_path"):
            assert forbidden_key not in row
        assert row["review_status"] == "review-required"
        assert row["mkb_status"] == "pending_validation_review"
    assert plan["counts"]["needs_review"] == summary["extracted_medical_fact_count"]


def test_plan_does_not_leak_phi_or_raw_text():
    # Even if an upstream caller put PHI-style fields in the item, the plan
    # must not surface them because the helper only reads the preview list.
    item = {
        "extracted_medical_facts_preview_safe": [
            {
                "type": "test_result",
                "test_name": "Glucose",
                "value": "5.4",
                "unit": "mmol/L",
                "reference_range": "3.9-5.5",
                "flag": "normal",
                "confidence": 0.55,
                "language_hint": "en",
                "parser_name": "deterministic_lab_line_adapter",
                "requires_review": True,
                "auto_accept_allowed": False,
                "source_line_hash": "abcdef012345",
            }
        ],
        "extracted_medical_fact_count": 1,
        "extraction_to_mkb_review_count": 1,
        # Hostile keys the caller might add. They must NOT appear in the plan.
        "raw_text": "Patient name: Jane Doe DOB 1990-01-01",
        "private_path": "C:\\Users\\Bob\\corpus\\file.pdf",
    }
    plan = build_extracted_information_preview_plan(item)
    serialized = repr(plan)
    assert "Jane" not in serialized
    assert "Doe" not in serialized
    assert "Bob" not in serialized
    assert "C:" not in serialized


def test_plan_counts_distinguish_written_vs_review_bound():
    item = {
        "extracted_medical_facts_preview_safe": [
            {
                "type": "test_result",
                "test_name": "Glucose",
                "value": "5.4",
                "unit": "mmol/L",
                "confidence": 0.55,
                "requires_review": True,
                "auto_accept_allowed": False,
                "source_line_hash": "a" * 12,
            }
        ],
        "extracted_medical_fact_count": 1,
        "extraction_to_mkb_written_count": 0,
        "extraction_to_mkb_review_count": 1,
    }
    plan = build_extracted_information_preview_plan(item)
    assert plan["counts"]["written_to_mkb"] == 0
    assert plan["counts"]["needs_review"] == 1
    assert plan["counts"]["structured_facts_extracted"] == 1


def test_plan_caps_when_input_is_oversized():
    # Defensive: passing more preview entries than typical should not crash.
    item = {
        "extracted_medical_facts_preview_safe": [
            {
                "type": "test_result",
                "test_name": f"Test_{i}",
                "value": str(i),
                "confidence": 0.55,
                "requires_review": True,
                "auto_accept_allowed": False,
                "source_line_hash": "x" * 12,
            }
            for i in range(50)
        ],
        "extracted_medical_fact_count": 50,
        "extraction_to_mkb_review_count": 50,
    }
    plan = build_extracted_information_preview_plan(item)
    assert plan["row_count"] == 50
    assert plan["counts"]["structured_facts_extracted"] == 50
