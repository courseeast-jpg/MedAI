"""Focused tests for MEDAI-REAL-DOC-CROSS-DOMAIN-EXTRACTION-14A."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.mkb_explorer_model import build_mkb_explorer_model
from clinical_knowledge.privacy import check_public_report_payload
from execution.extracted_medical_facts import (
    extract_card_result_entities,
    extract_cross_domain_visible_entities,
    extract_key_value_entities,
    extract_narrative_section_entities,
    extract_table_observation_entities,
)
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPO_ROOT = Path(__file__).resolve().parents[1]

GENERIC_LAB_TABLE = """Test | Result | Unit | Reference | Flag
Glucose | 5.4 | mmol/L | 3.9-5.5 |
Hemoglobin | 13.2 | g/dL | 12.0-16.0 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H
"""

URINALYSIS_TABLE = """Urinalysis
Leukocytes | positive | | negative | H
Specific Gravity | 1.020 | | 1.005-1.030 |
pH | 6.0 | | 5.0-8.0 |
"""

PORTAL_CARD_TEXT = """Vitamin D
Value: 24 ng/mL
Normal range: 30-100

Ferritin
Result: 18 ng/mL
Reference range: 20-200
"""

PATHOLOGY_TEXT = """Specimen:
Urine cytology

Cytology:
Negative for high-grade urothelial carcinoma in this visible sample.

Microscopic Description:
Benign urothelial cells are present in the submitted sample.
"""

IMAGING_TEXT = """Findings:
No focal consolidation is visible on this report text.

Impression:
No acute cardiopulmonary abnormality is stated in the source report.
"""

CONSULT_TEXT = """Plan:
Follow up with the treating clinician as written in the source note.

Recommendation:
Repeat local review if symptoms change according to the source note.
"""

KEY_VALUE_TEXT = """Specimen: urine
Source: clean catch
Collection method: midstream
Report type: culture result
"""


class NoopPiiStripper:
    last_audit = {"method": "test_noop"}

    def strip(self, text: str) -> tuple[str, str]:
        return text, "test_noop"


class EmptyLocalRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None):
        del text, specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="test_empty_then_cross_domain_adapter",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [],
                    "confidence": 0.95,
                    "latency_ms": 1,
                    "raw_text": "",
                    "notes": [],
                    "external_api_used": False,
                }
            ],
        )


class FakeSpacyExtractor:
    pass


def _build_pipeline(tmp_path: Path) -> tuple[ExecutionPipeline, SQLiteStore]:
    sql_store = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    return pipeline, sql_store


def test_generic_lab_table_synthetic_ocr_extracts_multiple_review_bound_observations() -> None:
    entities = extract_table_observation_entities(GENERIC_LAB_TABLE)

    assert len(entities) >= 3
    assert all(entity["type"] == "test_result" for entity in entities)
    assert all(entity["structured"]["requires_human_review"] is True for entity in entities)
    assert all(entity["structured"]["auto_accept_allowed"] is False for entity in entities)


def test_urinalysis_table_extracts_value_flag_units_reference_where_visible() -> None:
    entities = extract_table_observation_entities(URINALYSIS_TABLE)
    by_name = {entity["structured"]["test_name"]: entity for entity in entities}

    assert by_name["Leukocytes"]["structured"]["value"] == "positive"
    assert by_name["Leukocytes"]["structured"]["flag"] == "H"
    assert by_name["Specific Gravity"]["structured"]["reference_range"] == "1.005-1.030"


def test_portal_card_text_extracts_card_title_value_and_normal_range() -> None:
    entities = extract_card_result_entities(PORTAL_CARD_TEXT)

    assert len(entities) >= 2
    assert entities[0]["structured"]["card_title"] == "Vitamin D"
    assert entities[0]["structured"]["value"] == "24"
    assert entities[0]["structured"]["normal_range"] == "30-100"


def test_pathology_cytology_narrative_extracts_visible_sections_as_source_sections() -> None:
    entities = extract_narrative_section_entities(PATHOLOGY_TEXT, document_family="pathology_cytology")
    headings = {entity["structured"]["section_heading"] for entity in entities}

    assert {"Specimen", "Cytology", "Microscopic Description"}.issubset(headings)
    assert all(entity["type"] == "note" for entity in entities)


def test_imaging_radiology_narrative_extracts_findings_impression_not_interpretation() -> None:
    entities = extract_narrative_section_entities(IMAGING_TEXT, document_family="imaging_radiology")
    headings = {entity["structured"]["section_heading"] for entity in entities}

    assert {"Findings", "Impression"}.issubset(headings)
    assert all(entity["structured"]["not_medai_diagnosis"] is True for entity in entities)
    assert all(entity["structured"]["not_medai_recommendation"] is True for entity in entities)


def test_treatment_consult_sections_extract_plan_recommendation_as_source_sections() -> None:
    entities = extract_narrative_section_entities(CONSULT_TEXT, document_family="consult_treatment")
    headings = {entity["structured"]["section_heading"] for entity in entities}

    assert {"Plan", "Recommendation"}.issubset(headings)
    assert all(entity["type"] == "note" for entity in entities)
    assert all(entity["structured"]["not_medai_recommendation"] is True for entity in entities)


def test_key_value_report_metadata_extracts_visible_field_value_pairs_without_phi() -> None:
    entities = extract_key_value_entities(KEY_VALUE_TEXT + "MRN: ABC123\n")
    labels = {entity["structured"]["field_label"] for entity in entities}

    assert {"Specimen", "Source", "Collection method", "Report type"}.issubset(labels)
    assert "MRN" not in labels


def test_all_ocr_derived_records_are_review_bound(tmp_path: Path) -> None:
    pipeline, _sql = _build_pipeline(tmp_path)
    result = pipeline.process_text(
        GENERIC_LAB_TABLE + "\n" + PATHOLOGY_TEXT,
        specialty="urology",
        source_name="synthetic.txt",
        source_modality="image_ocr",
        document_category="Pathology",
        session_id="14a-review-bound",
    )

    assert result.queued_records
    assert all(record.tier == "quarantined" for record in result.queued_records)
    assert all(record.status == "pending_validation_review" for record in result.queued_records)
    assert all(record.requires_review is True for record in result.queued_records)


def test_accepted_count_remains_zero(tmp_path: Path) -> None:
    pipeline, _sql = _build_pipeline(tmp_path)
    result = pipeline.process_text(GENERIC_LAB_TABLE, source_modality="image_ocr", session_id="14a-zero")

    assert result.records == []
    assert result.extractor_result["extraction_to_mkb_written_count"] == 0


def test_category_and_specialty_domain_propagate(tmp_path: Path) -> None:
    pipeline, _sql = _build_pipeline(tmp_path)
    result = pipeline.process_text(
        URINALYSIS_TABLE,
        specialty="nephrology",
        document_category="Urinalysis",
        source_modality="image_ocr",
        session_id="14a-propagation",
    )

    assert result.queued_records
    assert all(record.specialty == "nephrology" for record in result.queued_records)
    assert all(record.structured.get("document_category") == "Urinalysis" for record in result.queued_records)


def test_mkb_explorer_count_model_shows_review_bound_records(tmp_path: Path) -> None:
    pipeline, sql = _build_pipeline(tmp_path)
    pipeline.process_text(GENERIC_LAB_TABLE, source_modality="image_ocr", session_id="14a-mkb")
    model = build_mkb_explorer_model(sql)

    assert model["counts"]["review_bound"] >= 3
    assert model["counts"]["active"] == 0


def test_review_queue_reader_shows_records(tmp_path: Path) -> None:
    pipeline, sql = _build_pipeline(tmp_path)
    pipeline.process_text(PORTAL_CARD_TEXT + "\n" + CONSULT_TEXT, source_modality="image_ocr", session_id="14a-review")
    model = build_mkb_explorer_model(sql, tier_filter="review_bound")

    assert model["row_count"] >= 2


def test_no_raw_ocr_text_in_public_reports() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_real_doc_cross_domain_extraction_14a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_dir = REPO_ROOT / "reports" / "medai_real_doc_cross_domain_extraction_14a"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in report_dir.iterdir())

    assert GENERIC_LAB_TABLE not in combined
    assert PATHOLOGY_TEXT not in combined


def test_external_api_used_false_and_auto_accept_false() -> None:
    entities = extract_cross_domain_visible_entities(GENERIC_LAB_TABLE + "\n" + PORTAL_CARD_TEXT)

    assert entities
    assert all(entity["structured"]["auto_accept_allowed"] is False for entity in entities)


def test_privacy_report_safe() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_real_doc_cross_domain_extraction_14a.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_dir = REPO_ROOT / "reports" / "medai_real_doc_cross_domain_extraction_14a"
    for path in report_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8")
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_12a_13c_commands_are_required_by_14a_scope() -> None:
    assert "tests/test_medai_operator_workflow_uat_12a.py"
    assert "tests/test_medai_operator_usability_polish_13c.py"
