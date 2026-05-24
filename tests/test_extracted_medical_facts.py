"""Focused tests for execution/extracted_medical_facts.py.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01.
"""
from __future__ import annotations

import pytest

from execution.extracted_medical_facts import (
    CONSERVATIVE_CONFIDENCE,
    MAX_PREVIEW_FACTS,
    PARSER_NAME,
    PROVENANCE_TAG,
    extract_lab_observation_entities,
    facts_for_ui,
    is_lab_style_document,
    merge_facts_into_entities,
    normalize_extracted_fact,
    summarize_extracted_facts_for_public_report,
)


ENGLISH_LAB_TEXT = (
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "Cholesterol: 4.1 mmol/L\n"
)

RUSSIAN_LAB_TEXT = (
    "Глюкоза: 5,4 ммоль/л\n"
    "Гемоглобин: 135 г/л\n"
    "Лейкоциты: 7,2 x10E9/л\n"
)

DATE_AND_ID_NOISE_TEXT = (
    "Patient ID: ABCD-12345\n"
    "DOB: 1985-03-12\n"
    "Date of collection: 2024-05-21\n"
    "Reported on: 21/05/2024\n"
    "Chart No: 99-77-12\n"
)


def test_english_lab_extraction_returns_test_result_entities():
    entities = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    assert len(entities) >= 3
    for entity in entities:
        assert entity["type"] == "test_result"
        assert entity["text"]
        structured = entity["structured"]
        assert structured["test_name"]
        assert structured["value"]
        assert structured["parser_name"] == PARSER_NAME
        assert structured["requires_human_review"] is True
        assert structured["auto_accept_allowed"] is False
        assert structured["provenance"] == PROVENANCE_TAG
        assert "source_line_hash" in structured
        assert len(structured["source_line_hash"]) == 12


def test_russian_cyrillic_lab_extraction_returns_test_result_entities():
    entities = extract_lab_observation_entities(
        RUSSIAN_LAB_TEXT, {"document_type": "lab_report"}
    )
    assert len(entities) >= 3
    names = {e["text"] for e in entities}
    assert "Глюкоза" in names
    assert "Гемоглобин" in names
    assert "Лейкоциты" in names
    for entity in entities:
        assert entity["structured"]["language_hint"] in {"ru", "mixed"}
        assert entity["structured"]["requires_human_review"] is True


def test_no_patient_identifier_or_date_only_false_positives():
    entities = extract_lab_observation_entities(
        DATE_AND_ID_NOISE_TEXT, {"document_type": "lab_report"}
    )
    assert entities == []


def test_no_extraction_when_text_is_empty():
    assert extract_lab_observation_entities("", {"document_type": "lab_report"}) == []
    assert extract_lab_observation_entities(None, None) == []  # type: ignore[arg-type]


def test_is_lab_style_document_recognizes_canonical_labels():
    assert is_lab_style_document({"document_type": "lab_report"}) is True
    assert is_lab_style_document({"document_type": "Lab result"}) is True
    assert (
        is_lab_style_document(
            {"document_family_classification_diagnostic": {"candidate_family": "Lab result"}}
        )
        is True
    )


def test_is_lab_style_document_returns_false_for_unrelated_documents():
    assert is_lab_style_document({"document_type": "unknown_medical"}) is False
    assert is_lab_style_document({"document_type": "prescription"}) is False
    assert is_lab_style_document(None) is False
    assert is_lab_style_document({}) is False


def test_merge_facts_into_entities_preserves_existing_and_adds_new():
    existing = [
        {"type": "diagnosis", "text": "Hypertension", "structured": {}},
    ]
    new_facts = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    merged = merge_facts_into_entities(existing, new_facts)
    assert len(merged) == 1 + len(new_facts)
    assert merged[0]["type"] == "diagnosis"
    for entity in merged[1:]:
        assert entity["type"] == "test_result"


def test_merge_facts_does_not_duplicate_same_test_value():
    new_facts = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    merged_once = merge_facts_into_entities([], new_facts)
    merged_twice = merge_facts_into_entities(merged_once, new_facts)
    assert len(merged_twice) == len(merged_once)


def test_normalize_extracted_fact_stamps_safety_defaults():
    payload = {
        "type": "test_result",
        "text": "Sodium",
        "structured": {"value": "140", "unit": "mmol/L"},
    }
    normalized = normalize_extracted_fact(payload)
    structured = normalized["structured"]
    assert structured["requires_human_review"] is True
    assert structured["auto_accept_allowed"] is False
    assert structured["parser_name"] == PARSER_NAME
    assert structured["provenance"] == PROVENANCE_TAG


def test_normalize_extracted_fact_strips_raw_line_keys():
    payload = {
        "type": "test_result",
        "text": "Sodium",
        "structured": {"value": "140", "raw_line": "secret raw text", "source_line": "secret"},
    }
    normalized = normalize_extracted_fact(payload)
    assert "raw_line" not in normalized["structured"]
    assert "source_line" not in normalized["structured"]


def test_summarize_for_public_report_does_not_emit_raw_lines():
    entities = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    summary = summarize_extracted_facts_for_public_report(entities)
    assert summary["extracted_medical_fact_count"] == len(entities)
    assert summary["extracted_medical_fact_types"] == ["test_result"]
    for preview in summary["extracted_medical_facts_preview_safe"]:
        assert "raw_line" not in preview
        assert "source_line" not in preview
        assert preview["auto_accept_allowed"] is False
        assert preview["requires_review"] is True
        assert preview["parser_name"] == PARSER_NAME
        assert preview["source_line_hash"]


def test_summarize_for_public_report_caps_preview():
    many = [
        {
            "type": "test_result",
            "text": f"Test_{i}",
            "structured": {"test_name": f"Test_{i}", "value": str(i)},
            "confidence": CONSERVATIVE_CONFIDENCE,
        }
        for i in range(MAX_PREVIEW_FACTS + 5)
    ]
    summary = summarize_extracted_facts_for_public_report(many)
    assert summary["extracted_medical_fact_count"] == len(many)
    assert len(summary["extracted_medical_facts_preview_safe"]) == MAX_PREVIEW_FACTS


def test_facts_for_ui_empty_state_message():
    plan = facts_for_ui({"extracted_medical_facts_preview_safe": [], "extracted_medical_fact_count": 0})
    assert plan["row_count"] == 0
    assert "No structured medical facts extracted" in plan["message"]
    assert plan["auto_accept_allowed"] is False
    assert plan["review_required"] is True


def test_facts_for_ui_renders_columns_and_rows():
    entities = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    summary = summarize_extracted_facts_for_public_report(entities)
    plan = facts_for_ui(
        {
            "extracted_medical_facts_preview_safe": summary["extracted_medical_facts_preview_safe"],
            "extracted_medical_fact_count": summary["extracted_medical_fact_count"],
            "extraction_to_mkb_written_count": 0,
            "extraction_to_mkb_review_count": summary["extracted_medical_fact_count"],
        }
    )
    assert plan["row_count"] == summary["extracted_medical_fact_count"]
    for required_col in ("Type", "Test / observation", "Value", "Unit", "Reference range", "Flag", "Confidence", "Review status", "MKB status"):
        assert required_col in plan["columns"]
    counts = plan["counts"]
    assert counts["structured_facts_extracted"] == plan["row_count"]
    assert counts["needs_review"] == plan["row_count"]


def test_facts_for_ui_review_status_marks_review_required():
    entities = extract_lab_observation_entities(
        ENGLISH_LAB_TEXT, {"document_type": "lab_report"}
    )
    summary = summarize_extracted_facts_for_public_report(entities)
    plan = facts_for_ui(
        {
            "extracted_medical_facts_preview_safe": summary["extracted_medical_facts_preview_safe"],
            "extracted_medical_fact_count": summary["extracted_medical_fact_count"],
        }
    )
    for row in plan["rows"]:
        assert row["review_status"] == "review-required"
        assert row["mkb_status"] == "pending_validation_review"


def test_conservative_confidence_is_below_typical_accept_threshold():
    # Sanity guard: conservative confidence must keep facts in review territory.
    assert 0.30 < CONSERVATIVE_CONFIDENCE < 0.80
