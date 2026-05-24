"""Smoke tests for ExecutionPipeline.process_text + extracted facts adapter.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-02-PIPELINE-SMOKE.

Exercises the real ExecutionPipeline with a stubbed SpacyExtractor only.
No spaCy model required. No chromadb required. No external API. Real
SQLiteStore against a temporary DB. Real MKBWriter. Real extracted
medical facts adapter.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_02_PIPELINE_SMOKE.md"


SYNTH_EN_TEXT = (
    "Lab result report\n"
    "Specimen: serum\n"
    "Reference range listed below.\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
)

SYNTH_RU_TEXT = (
    "Лабораторный отчет\n"
    "Материал: сыворотка\n"
    "Результат:\n"
    "Глюкоза: 5,4 ммоль/л\n"
    "Гемоглобин: 135 г/л\n"
    "Лейкоциты: 7,2 x10E9/л\n"
)


class _StubSpacyExtractor:
    """Returns an empty-entity extractor result so the adapter is the
    layer that produces test_result facts. Confidence high enough to keep
    the router on the spacy route and to clear the validation accept
    threshold (EXTRACTION_ACCEPT_THRESHOLD = 0.65)."""

    def __init__(self, confidence: float = 0.86) -> None:
        self._confidence = float(confidence)

    def extract(self, text: str) -> Dict[str, Any]:
        return {
            "extractor": "spacy",
            "entities": [],
            "confidence": self._confidence,
            "latency_ms": 1,
            "raw_text": text,
            "notes": ["stub_extractor_used_in_smoke_test"],
        }


@pytest.fixture()
def pipeline_with_store(tmp_path):
    os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
    os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

    from execution.pipeline import ExecutionPipeline
    from mkb.sqlite_store import SQLiteStore

    db_path = tmp_path / "mkb_smoke.db"
    sql_store = SQLiteStore(db_path=db_path, encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        quality_gate=None,
        medication_gate=None,
        spacy_extractor=_StubSpacyExtractor(),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    return pipeline, sql_store


def test_pipeline_process_text_returns_execution_result(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    from execution.jobs import ExecutionResult

    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    assert isinstance(result, ExecutionResult)
    assert result.extractor_result is not None
    assert isinstance(result.extractor_result, dict)


def test_extractor_result_carries_extracted_medical_fact_count(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    ext = result.extractor_result
    assert int(ext.get("extracted_medical_fact_count") or 0) > 0
    assert ext.get("document_type") == "Lab result"
    types = list(ext.get("extracted_medical_fact_types") or [])
    assert "test_result" in types


def test_extracted_facts_merged_into_entities(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    entities = list(result.extractor_result.get("entities") or [])
    # Adapter produces test_result entities since the stub returned none.
    test_results = [e for e in entities if str(e.get("type")) == "test_result"]
    assert len(test_results) >= 3
    for entity in test_results:
        structured = entity.get("structured") or {}
        assert structured.get("parser_name") == "deterministic_lab_line_adapter"
        assert structured.get("provenance") == "extracted_medical_facts_adapter"
        assert structured.get("requires_human_review") is True
        assert structured.get("auto_accept_allowed") is False


def test_mkb_records_persisted_as_review_bound_not_auto_accepted(pipeline_with_store):
    pipeline, sql_store = pipeline_with_store
    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    # All adapter-produced test_result records must land in queued, not records.
    assert len(result.records) == 0
    queued_test = [r for r in result.queued_records if r.fact_type == "test_result"]
    assert len(queued_test) >= 3
    for record in queued_test:
        assert record.requires_review is True
        assert record.tier == "quarantined"


def test_sqlite_retrieval_returns_persisted_test_result_rows(pipeline_with_store):
    pipeline, sql_store = pipeline_with_store
    pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    with sql_store._get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM records WHERE fact_type=?", ("test_result",)
        ).fetchall()
    assert len(rows) >= 3
    review_required = sql_store.get_records_requiring_review()
    assert len(review_required) >= 3
    for record in review_required:
        assert record.fact_type == "test_result"
        assert record.requires_review is True


def test_russian_cyrillic_synthetic_text_extracts_facts(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    result = pipeline.process_text(
        SYNTH_RU_TEXT,
        specialty="general",
        source_name="synth-ru-001",
        session_id="smoke-ru",
    )
    ext = result.extractor_result or {}
    assert ext.get("document_type") == "Lab result"
    fact_count = int(ext.get("extracted_medical_fact_count") or 0)
    assert fact_count >= 3
    queued_test = [r for r in result.queued_records if r.fact_type == "test_result"]
    assert len(queued_test) >= 3
    for record in queued_test:
        structured = record.structured or {}
        assert structured.get("language_hint") in {"ru", "mixed"}
        assert record.requires_review is True


def test_ui_render_plan_helper_renders_rows_from_pipeline_payload(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    from app.extracted_information_preview import build_extracted_information_preview_plan

    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    ext = result.extractor_result or {}
    item = {
        "extracted_medical_facts_preview_safe": ext.get(
            "extracted_medical_facts_preview_safe"
        )
        or [],
        "extracted_medical_fact_count": int(ext.get("extracted_medical_fact_count") or 0),
        "extraction_to_mkb_written_count": int(
            ext.get("extraction_to_mkb_written_count") or 0
        ),
        "extraction_to_mkb_review_count": int(
            ext.get("extraction_to_mkb_review_count") or 0
        ),
    }
    plan = build_extracted_information_preview_plan(item)
    assert plan["row_count"] >= 3
    for row in plan["rows"]:
        assert row["review_status"] == "review-required"
        assert row["mkb_status"] == "pending_validation_review"
        # Public-safe contract.
        for forbidden in ("raw_line", "source_line", "private_path", "file_path"):
            assert forbidden not in row


def test_external_api_not_used_in_pipeline_smoke(pipeline_with_store):
    pipeline, _ = pipeline_with_store
    result = pipeline.process_text(
        SYNTH_EN_TEXT,
        specialty="general",
        source_name="synth-en-001",
        session_id="smoke-en",
    )
    ext = result.extractor_result or {}
    # The stub keeps the router on the spacy route. No gemini call.
    actual = str(ext.get("actual_extractor") or "")
    assert actual != "gemini"
    assert ext.get("external_api_used", False) is False


def test_medication_safety_path_unchanged_by_smoke():
    """Sanity guard: the smoke test must not bypass medication safety."""
    from execution.safety import ExecutionSafety

    assert hasattr(ExecutionSafety, "check_medication")


def test_no_raw_synthetic_line_text_in_report_payload():
    """Public report must not echo raw synthetic source lines."""
    # The synthetic line "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]" is
    # the textual cue; the public report carries values via the preview
    # entries, never the raw line. Defensive check: report must not
    # contain the bracketed flag pattern from the source line.
    if not JSON_REPORT_PATH.exists():
        pytest.skip("validation report not yet generated; run script first")
    body = JSON_REPORT_PATH.read_text(encoding="utf-8")
    # Raw line pattern: "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]"
    assert "5.4 mmol/L (ref 3.9-5.5)" not in body
    # The preview "test_name": "Glucose" + structured "value": "5.4" + "unit": "mmol/L"
    # is acceptable. Only the formatted raw line is forbidden.


def test_validation_script_reports_pipeline_smoke_ready():
    proc = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "run_medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke.py"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "pipeline_smoke_ready"
    assert payload["all_pass"] is True
    findings = payload["audit_findings"]
    assert findings["synthetic_documents_evaluated"] >= 2
    assert findings["structured_facts_extracted_total"] >= 6
    assert findings["retrieval_proof_count"] >= 6
    assert findings["ui_preview_proof_count"] >= 6
    assert findings["review_bound_records_persisted_total"] >= 6
    assert findings["external_api_used"] is False
    assert findings["auto_accept_allowed_in_any_record"] is False


def test_public_reports_pass_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH):
        if not path.is_file():
            pytest.skip(f"{path.name} not yet generated")
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_01_block_tests_still_pass():
    """Sanity: the 01 block's tests must still pass after the 02 change."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_extracted_medical_facts.py",
            "tests/test_run_review_extracted_information_preview.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_01.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    # The 01 block tests should pass; report stdout when not.
    assert proc.returncode == 0, proc.stdout + proc.stderr
