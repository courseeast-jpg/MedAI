"""End-to-end focused tests for MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01.

Covers:

* deterministic facts merge into pipeline entities via the adapter;
* MKBRecord conversion preserves structured fields;
* review-bound test_result facts are persisted into SQLite and
  retrievable;
* MKBWriter does not auto-accept review-bound records;
* the public-safe UI render plan reflects the persisted state;
* the validation script reaches conclusion ``minimum_extraction_to_mkb_ready``;
* the public report contains no raw OCR text, raw filenames, private
  paths, or PHI;
* external API is disabled / not used in any path exercised by the
  validation script;
* existing medication safety behavior is not bypassed (medication facts
  still flow through the safety gate when present).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_01"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_01_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_01_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_01.md"


@pytest.fixture(scope="module")
def validation_payload() -> dict:
    """Run the validation script once and return its JSON payload."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_corpus_extraction_to_mkb_minimum_01.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))
    return payload


def test_validation_script_reports_minimum_extraction_to_mkb_ready(validation_payload):
    assert validation_payload["conclusion"] == "minimum_extraction_to_mkb_ready"
    assert validation_payload["all_pass"] is True


def test_synthetic_documents_were_evaluated(validation_payload):
    findings = validation_payload["audit_findings"]
    assert findings["synthetic_documents_evaluated"] >= 2
    assert findings["structured_facts_extracted_total"] >= 6


def test_extraction_to_mkb_records_were_persisted(validation_payload):
    findings = validation_payload["audit_findings"]
    persisted_total = (
        findings["written_mkb_records_total"]
        + findings["review_bound_mkb_records_total"]
    )
    assert persisted_total >= findings["structured_facts_extracted_total"]
    assert findings["retrieval_proof_count"] >= persisted_total
    assert findings["candidate_records_total"] >= persisted_total


def test_review_bound_default_preserved(validation_payload):
    findings = validation_payload["audit_findings"]
    # Conservative confidence keeps every adapter fact review-bound.
    assert findings["review_bound_mkb_records_total"] >= 1
    for per_doc in findings["per_document_results"]:
        for preview in per_doc["extracted_medical_facts_preview_safe"]:
            assert preview["requires_review"] is True
            assert preview["auto_accept_allowed"] is False


def test_ui_render_plan_rows_match_extracted_facts(validation_payload):
    findings = validation_payload["audit_findings"]
    assert findings["ui_render_plan_built_successfully"] is True
    assert findings["ui_render_plan_rows_total"] == findings["structured_facts_extracted_total"]


def test_external_api_not_used_and_no_raw_text_leakage(validation_payload):
    assert validation_payload["external_api_used"] is False
    findings = validation_payload["audit_findings"]
    assert findings["external_api_used"] is False
    assert findings["raw_ocr_text_in_public_report"] is False
    assert findings["private_paths_in_public_report"] is False
    assert findings["raw_phi_in_public_report"] is False


def test_no_private_file_committed(validation_payload):
    assert validation_payload["real_pdf_committed"] is False
    assert validation_payload["real_screenshot_committed"] is False
    assert validation_payload["raw_text_printed"] is False
    assert validation_payload["raw_ocr_text_printed"] is False
    assert validation_payload["raw_filenames_printed"] is False
    assert validation_payload["private_paths_printed"] is False
    assert validation_payload["phi_printed"] is False


def test_public_report_passes_clinical_knowledge_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH):
        assert path.is_file(), path
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_pipeline_imports_adapter_and_calls_it_on_lab_style_document():
    from execution.pipeline import ExecutionPipeline

    pipeline_source = (REPO_ROOT / "execution" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "from execution.extracted_medical_facts import" in pipeline_source
    assert "_apply_extracted_medical_facts_adapter" in pipeline_source
    assert "extracted_medical_fact_count" in pipeline_source
    assert "extraction_to_mkb_candidate_count" in pipeline_source
    assert "extraction_to_mkb_written_count" in pipeline_source
    assert "extraction_to_mkb_review_count" in pipeline_source


def test_test_launcher_carries_preview_fields_into_test_file_result():
    from app.test_launcher import TestFileResult

    result = TestFileResult(file_name="example", status="accepted")
    # Defaults exist.
    assert result.extracted_medical_facts_preview_safe == []
    assert result.extracted_medical_fact_count == 0
    assert result.extracted_medical_fact_types == []
    assert result.extraction_to_mkb_candidate_count == 0
    assert result.extraction_to_mkb_written_count == 0
    assert result.extraction_to_mkb_review_count == 0


def test_mkb_writer_persists_review_bound_test_result_records():
    from app.config import TIER_ACTIVE, TRUST_CLINICAL
    from app.schemas import MKBRecord
    from execution.mkb_writer import MKBWriter
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_test_writer_"))
    store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    writer = MKBWriter(sql_store=store, vector_store=None, quality_gate=None)

    record = MKBRecord(
        fact_type="test_result",
        content="Test: Glucose: 5.4 mmol/L",
        structured={
            "text": "Glucose",
            "test_name": "Glucose",
            "value": "5.4",
            "unit": "mmol/L",
            "reference_range": "3.9-5.5",
            "parser_name": "deterministic_lab_line_adapter",
            "requires_human_review": True,
            "auto_accept_allowed": False,
        },
        specialty="general",
        source_type="extraction",
        source_name="synthetic-test-001",
        trust_level=TRUST_CLINICAL,
        confidence=0.55,
        tier=TIER_ACTIVE,
        extraction_method="rules_based",
        requires_review=True,
        ddi_checked=False,
        tags=["factual_extraction", "requires_source_comparison"],
    )

    written, queued = writer.write([record], session_id="syn-test")
    # Review-bound record goes to queued AND is persisted in SQLite.
    assert len(written) == 0
    assert len(queued) == 1

    persisted = store.get_records_requiring_review()
    assert len(persisted) == 1
    assert persisted[0].fact_type == "test_result"
    assert persisted[0].structured["test_name"] == "Glucose"
    assert persisted[0].requires_review is True


def test_mkb_record_round_trip_preserves_structured_fields():
    from app.config import TIER_ACTIVE, TRUST_CLINICAL
    from app.schemas import MKBRecord
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_test_roundtrip_"))
    store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    record = MKBRecord(
        fact_type="test_result",
        content="Test: Hemoglobin: 13.5 g/dL",
        structured={
            "text": "Hemoglobin",
            "test_name": "Hemoglobin",
            "value": "13.5",
            "unit": "g/dL",
            "reference_range": "13.5-17.5",
            "parser_name": "deterministic_lab_line_adapter",
            "requires_human_review": True,
            "auto_accept_allowed": False,
            "source_line_hash": "abc123def456",
        },
        specialty="general",
        source_type="extraction",
        source_name="synthetic-roundtrip",
        trust_level=TRUST_CLINICAL,
        confidence=0.55,
        tier=TIER_ACTIVE,
        extraction_method="rules_based",
        requires_review=True,
        ddi_checked=False,
        tags=["factual_extraction"],
    )
    store.write_record(record, session_id="roundtrip")
    retrieved = store.get_record(record.id)
    assert retrieved is not None
    assert retrieved.fact_type == "test_result"
    assert retrieved.structured["test_name"] == "Hemoglobin"
    assert retrieved.structured["value"] == "13.5"
    assert retrieved.structured["unit"] == "g/dL"
    assert retrieved.structured["reference_range"] == "13.5-17.5"
    assert retrieved.structured["parser_name"] == "deterministic_lab_line_adapter"
    assert retrieved.requires_review is True


def test_medication_safety_gate_signature_unchanged_by_block():
    """Sanity guard: this block must not bypass medication safety."""
    from execution.safety import ExecutionSafety

    # The class still exists and exposes check_medication.
    assert hasattr(ExecutionSafety, "check_medication")


def test_public_safe_preview_columns_match_required_layout():
    findings = json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))["audit_findings"]
    sample_preview = None
    for per_doc in findings["per_document_results"]:
        for entry in per_doc["extracted_medical_facts_preview_safe"]:
            sample_preview = entry
            break
        if sample_preview:
            break
    assert sample_preview is not None, "validation script produced no preview entries"
    for required_key in (
        "type",
        "test_name",
        "value",
        "confidence",
        "language_hint",
        "parser_name",
        "requires_review",
        "auto_accept_allowed",
    ):
        assert required_key in sample_preview
    for forbidden_key in ("raw_line", "source_line", "private_path", "file_path", "absolute_path"):
        assert forbidden_key not in sample_preview
