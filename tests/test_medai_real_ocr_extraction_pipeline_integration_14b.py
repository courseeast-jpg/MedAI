"""Focused tests for MEDAI-REAL-OCR-EXTRACTION-PIPELINE-INTEGRATION-14B."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import app.test_launcher as launcher
from app.mkb_explorer_model import build_mkb_explorer_model
from clinical_knowledge.privacy import check_public_report_payload
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LIKE_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H

Vitamin D
Value: 24 ng/mL
Normal range: 30-100

Findings:
Visible source section captured for review.
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
            decision_reason="test_ocr_pipeline_14b",
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


def _configure_launcher_dirs(root: Path) -> None:
    launcher.TEST_INPUT_DIR = root / "test_input"
    launcher.TEST_REVIEW_DIR = root / "test_review"
    launcher.TEST_ARCHIVE_DIR = root / "test_archive"
    launcher.TEST_OUTPUT_DIR = root / "test_output"
    launcher.TEST_RUN_REPORT_DIR = root / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"


def _run_image_batch(tmp_path: Path):
    _configure_launcher_dirs(tmp_path)
    pipeline, sql = _build_pipeline(tmp_path)
    launcher.TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (launcher.TEST_INPUT_DIR / "synthetic_real_like.png").write_bytes(b"not-real-image")
    launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
        available=True,
        attempted=True,
        text=REAL_LIKE_OCR_TEXT,
        engine="tesseract_local",
        language="eng",
        text_visibility="recovered",
    )
    summary = launcher.run_medai_test_batch(
        pipeline,
        specialty="urology",
        document_category="Cross-domain OCR",
    )
    return summary, sql


def test_image_ocr_recovered_text_is_dispatched_to_14a_cross_domain_entrypoint(tmp_path: Path) -> None:
    summary, _sql = _run_image_batch(tmp_path)
    result = summary.results[0]

    assert result["image_ocr_text_visibility"] == "recovered"
    assert result["extractor_dispatch_count"] >= 1


def test_extractor_dispatch_count_increments(tmp_path: Path) -> None:
    summary, _sql = _run_image_batch(tmp_path)

    assert summary.safe_ocr_pipeline_diagnostics["extractor_dispatch_count"] >= 1


def test_extraction_candidates_count_positive_for_real_like_table_ocr(tmp_path: Path) -> None:
    summary, _sql = _run_image_batch(tmp_path)

    assert summary.results[0]["extraction_candidates_count"] > 0
    assert summary.safe_ocr_pipeline_diagnostics["extraction_candidates_count"] > 0


def test_candidates_are_persisted_as_review_bound_records(tmp_path: Path) -> None:
    summary, sql = _run_image_batch(tmp_path)
    model = build_mkb_explorer_model(sql, tier_filter="review_bound")

    assert summary.results[0]["review_bound_records_written_count"] > 0
    assert model["row_count"] >= summary.results[0]["review_bound_records_written_count"]


def test_dedupe_count_is_reported_for_duplicate_candidates(tmp_path: Path) -> None:
    pipeline, _sql = _build_pipeline(tmp_path)
    result = pipeline.process_text(
        REAL_LIKE_OCR_TEXT + "\nGlucose | 5.4 | mmol/L | 3.9-5.5 |",
        source_modality="image_ocr",
        document_category="Cross-domain OCR",
        session_id="14b-dedupe",
    )

    assert result.extractor_result["cross_domain_records_deduped_count"] >= 1


def test_selected_document_category_propagates(tmp_path: Path) -> None:
    _summary, sql = _run_image_batch(tmp_path)
    records = build_mkb_explorer_model(sql, tier_filter="review_bound")["rows"]

    assert records
    assert all(sql.get_record(row["record_id_full"]).structured.get("document_category") == "Cross-domain OCR" for row in records)


def test_selected_specialty_domain_propagates(tmp_path: Path) -> None:
    _summary, sql = _run_image_batch(tmp_path)
    records = build_mkb_explorer_model(sql, tier_filter="review_bound")["rows"]

    assert records
    assert all(sql.get_record(row["record_id_full"]).specialty == "urology" for row in records)


def test_source_modality_image_ocr_preserved(tmp_path: Path) -> None:
    summary, sql = _run_image_batch(tmp_path)
    records = build_mkb_explorer_model(sql, tier_filter="review_bound")["rows"]

    assert summary.results[0]["source_modality"] == "image_ocr"
    assert all(sql.get_record(row["record_id_full"]).structured.get("source_modality") == "image_ocr" for row in records)


def test_run_review_summary_reports_structured_review_bound_observations_created(tmp_path: Path) -> None:
    summary, _sql = _run_image_batch(tmp_path)

    assert "structured review-bound observations created" in summary.results[0]["run_review_summary"]


def test_run_review_summary_reports_deduped_candidates_when_not_newly_written(tmp_path: Path) -> None:
    pipeline, _sql = _build_pipeline(tmp_path)
    result = pipeline.process_text(
        REAL_LIKE_OCR_TEXT + "\nWBC | 11.2 | x10E9/L | 4.0-10.0 | H",
        source_modality="image_ocr",
        session_id="14b-summary-dedupe",
    )
    fake_item = {
        "cross_domain_extraction_candidates_count": result.extractor_result["cross_domain_extraction_candidates_count"],
        "cross_domain_review_bound_records_written_count": 0,
        "cross_domain_records_deduped_count": result.extractor_result["cross_domain_records_deduped_count"],
    }
    summary_text = launcher._run_review_summary_from_extraction(fake_item)

    assert "equivalent extraction candidate" in summary_text


def test_no_raw_ocr_text_in_public_reports() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_real_ocr_extraction_pipeline_integration_14b.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_dir = REPO_ROOT / "reports" / "medai_real_ocr_extraction_pipeline_integration_14b"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in report_dir.iterdir())

    assert REAL_LIKE_OCR_TEXT not in combined


def test_no_private_filenames_or_paths_in_reports() -> None:
    report_dir = REPO_ROOT / "reports" / "medai_real_ocr_extraction_pipeline_integration_14b"
    subprocess.run(
        [sys.executable, "scripts/run_medai_real_ocr_extraction_pipeline_integration_14b.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "synthetic_real_like.png" not in text
        assert "C:\\" not in text
        result = check_public_report_payload(json.loads(text) if path.suffix == ".json" else text)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_external_api_used_false_and_auto_accept_false() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_real_ocr_extraction_pipeline_integration_14b.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_real_ocr_extraction_pipeline_integration_14b" / "medai_real_ocr_extraction_pipeline_integration_14b_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["external_api_used"] is False
    assert payload["auto_accept"] is False
