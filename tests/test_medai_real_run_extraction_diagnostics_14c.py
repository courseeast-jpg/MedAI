"""Focused tests for MEDAI-REAL-RUN-EXTRACTION-DIAGNOSTICS-14C."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import app.test_launcher as launcher
from app.main import advanced_diagnostic_fields
from clinical_knowledge.privacy import check_public_report_payload
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RUN_OCR_TEXT = """Patient Portal Results
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
            decision_reason="test_real_run_diagnostics_14c",
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


def _configure_launcher_dirs(root: Path) -> None:
    launcher.TEST_INPUT_DIR = root / "test_input"
    launcher.TEST_REVIEW_DIR = root / "test_review"
    launcher.TEST_ARCHIVE_DIR = root / "test_archive"
    launcher.TEST_OUTPUT_DIR = root / "test_output"
    launcher.TEST_RUN_REPORT_DIR = root / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"


def _build_pipeline(tmp_path: Path) -> ExecutionPipeline:
    return ExecutionPipeline(
        sql_store=SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key=""),
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def _run_image_batch(tmp_path: Path, text: str = REAL_RUN_OCR_TEXT):
    _configure_launcher_dirs(tmp_path)
    pipeline = _build_pipeline(tmp_path)
    launcher.TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (launcher.TEST_INPUT_DIR / "real_browser_upload_name.png").write_bytes(b"not-real-image")
    launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
        available=True,
        attempted=True,
        text=text,
        engine="tesseract_local",
        language="eng",
        text_visibility="recovered",
    )
    return launcher.run_medai_test_batch(
        pipeline,
        specialty="urology",
        document_category="Cross-domain OCR",
    )


def test_real_runtime_path_records_ocr_recovered_counter(tmp_path: Path) -> None:
    summary = _run_image_batch(tmp_path)
    item = summary.results[0]

    assert item["ocr_attempted"] is True
    assert item["ocr_available"] is True
    assert item["ocr_recovered"] is True
    assert summary.safe_real_run_extraction_diagnostics["ocr_recovered"] == 1


def test_real_runtime_path_records_text_shape_buckets_without_raw_text(tmp_path: Path) -> None:
    summary = _run_image_batch(tmp_path)
    item = summary.results[0]

    assert item["ocr_text_character_bucket"] != "0"
    assert item["ocr_line_count_bucket"] != "0"
    assert item["ocr_has_table_like_layout"] is True
    assert item["ocr_has_key_value_like_layout"] is True
    assert item["ocr_has_section_heading_like_layout"] is True
    assert REAL_RUN_OCR_TEXT not in json.dumps(item, sort_keys=True)


def test_extractor_dispatch_attempted_is_recorded(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]

    assert item["extractor_dispatch_attempted"] is True
    assert item["extractor_dispatch_family"]


def test_extraction_candidates_created_and_after_filter_are_recorded(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]

    assert item["extraction_candidates_created"] > 0
    assert item["extraction_candidates_after_filter"] > 0


def test_drop_reason_counts_are_recorded_when_candidates_are_dropped(tmp_path: Path) -> None:
    duplicate_text = REAL_RUN_OCR_TEXT + "\nGlucose | 5.4 | mmol/L | 3.9-5.5 |"
    item = _run_image_batch(tmp_path, duplicate_text).results[0]

    assert item["extraction_candidates_dropped"] >= 1
    assert item["drop_reason_counts"]["deduped_or_already_represented"] >= 1


def test_records_written_deduped_and_review_bound_written_are_recorded(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path, REAL_RUN_OCR_TEXT + "\nWBC | 11.2 | x10E9/L | 4.0-10.0 | H").results[0]

    assert item["records_written"] > 0
    assert item["records_deduped"] >= 1
    assert item["review_bound_records_written"] > 0


def test_category_specialty_and_source_modality_propagate_into_diagnostics(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]

    assert item["selected_document_category"] == "Cross-domain OCR"
    assert item["selected_specialty_domain"] == "urology"
    assert item["source_modality"] == "image_ocr"


def test_document_type_before_and_after_extraction_are_recorded(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]

    assert item["document_type_before_extraction"]
    assert item["document_type_after_extraction"] == "Lab result"


def test_advanced_diagnostics_contain_safe_14c_counters_without_raw_text(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]
    advanced = advanced_diagnostic_fields(item)
    serialized = json.dumps(advanced, sort_keys=True)

    assert advanced["ocr_text_character_bucket"]
    assert advanced["extractor_dispatch_attempted"] is True
    assert advanced["extraction_candidates_created"] > 0
    assert advanced["records_written"] > 0
    assert REAL_RUN_OCR_TEXT not in serialized
    assert "real_browser_upload_name.png" not in serialized
    assert "C:\\" not in serialized


def test_runtime_diagnostic_summary_is_compact_and_safe(tmp_path: Path) -> None:
    item = _run_image_batch(tmp_path).results[0]
    summary = item["runtime_diagnostic_summary"]

    assert "OCR recovered: yes" in summary
    assert "extractor dispatch: yes" in summary
    assert "candidates:" in summary
    assert REAL_RUN_OCR_TEXT not in summary


def test_reports_contain_no_private_filenames_or_paths() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_real_run_extraction_diagnostics_14c.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_dir = REPO_ROOT / "reports" / "medai_real_run_extraction_diagnostics_14c"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "real_browser_upload_name.png" not in text
        assert "C:\\" not in text
        result = check_public_report_payload(json.loads(text) if path.suffix == ".json" else text)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_external_api_false_and_auto_accept_false() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_real_run_extraction_diagnostics_14c.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_real_run_extraction_diagnostics_14c" / "medai_real_run_extraction_diagnostics_14c_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["external_api_used"] is False
    assert payload["auto_accept"] is False
