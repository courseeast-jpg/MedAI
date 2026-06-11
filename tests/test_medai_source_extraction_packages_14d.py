"""Focused tests for MEDAI-SOURCE-EXTRACTION-PACKAGES-14D."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import app.test_launcher as launcher
from app.mkb_explorer_model import build_mkb_explorer_model
from app.source_extraction_packages import (
    accept_package_after_source_comparison,
    build_source_extraction_packages,
    defer_package,
    reject_package,
)
from clinical_knowledge.privacy import check_public_report_payload
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore
from app.schemas import MKBRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H

Findings:
Visible source section captured for review.
"""


def _record(
    *,
    name: str,
    value: str,
    section: str,
    record_id: str = "",
    session_id: str = "session-14d",
) -> MKBRecord:
    structured = {
        "source_visible_observation": True,
        "test_name": name,
        "value": value,
        "unit": "mmol/L",
        "reference_range": "3.9-5.5",
        "flag": "H" if name == "WBC" else "",
        "section_heading": section,
        "candidate_kind": "table_observation",
        "document_category": "Cross-domain OCR",
        "document_family": "Lab result",
        "source_modality": "image_ocr",
        "parser_name": "cross_domain_visible_observation_adapter",
        "requires_human_review": True,
        "auto_accept_allowed": False,
    }
    kwargs = {"id": record_id} if record_id else {}
    return MKBRecord(
        **kwargs,
        fact_type="test_result",
        content=f"{name} {value}",
        structured=structured,
        specialty="urology",
        source_type="document",
        source_name="private_real_upload.png",
        trust_level=1,
        confidence=0.95,
        status="pending_validation_review",
        tier="quarantined",
        requires_review=True,
        extraction_method="rules_based",
        session_id=session_id,
    )


def _store_with_records(tmp_path: Path) -> SQLiteStore:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sql = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    sql.write_record(_record(name="Glucose", value="5.4", section="Patient Portal Results"), session_id="session-14d")
    sql.write_record(_record(name="WBC", value="11.2", section="Patient Portal Results"), session_id="session-14d")
    sql.write_record(_record(name="Visible source section", value="", section="Findings"), session_id="session-14d")
    return sql


def test_review_bound_atomic_records_group_into_source_extraction_packages(tmp_path: Path) -> None:
    model = build_source_extraction_packages(_store_with_records(tmp_path))

    assert model["packages_created"] == 1
    assert model["observations_grouped"] == 3
    assert model["ungrouped_records_count"] == 0


def test_package_contains_category_specialty_and_source_modality(tmp_path: Path) -> None:
    package = build_source_extraction_packages(_store_with_records(tmp_path))["packages"][0]

    assert package["selected_document_category"] == "Cross-domain OCR"
    assert package["selected_medical_specialty_domain"] == "urology"
    assert package["source_modality"] == "local OCR"


def test_package_sections_created_from_section_metadata(tmp_path: Path) -> None:
    package = build_source_extraction_packages(_store_with_records(tmp_path))["packages"][0]
    headings = {section["heading"] for section in package["sections"]}

    assert {"Patient Portal Results", "Findings"}.issubset(headings)


def test_observations_retain_label_value_reference_and_flag_fields(tmp_path: Path) -> None:
    package = build_source_extraction_packages(_store_with_records(tmp_path))["packages"][0]
    observations = [obs for section in package["sections"] for obs in section["observations"]]
    wbc = next(obs for obs in observations if obs["label"] == "WBC")

    assert wbc["value"] == "11.2"
    assert wbc["unit"] == "mmol/L"
    assert wbc["reference_interval"] == "3.9-5.5"
    assert wbc["flag"] == "H"


def test_package_remains_review_bound_and_does_not_auto_accept(tmp_path: Path) -> None:
    package = build_source_extraction_packages(_store_with_records(tmp_path))["packages"][0]

    assert package["package_status"] == "review-bound"
    assert package["review_required"] is True
    assert package["auto_accept_allowed"] is False


def test_package_accept_requires_explicit_human_action(tmp_path: Path) -> None:
    sql = _store_with_records(tmp_path)
    package = build_source_extraction_packages(sql)["packages"][0]

    assert package["actions"]["actions"][0]["label"] == "Accept package after source comparison"
    assert "source document" in package["actions"]["actions"][0]["disclaimer"]
    assert all(sql.get_record(record_id).requires_review for record_id in package["record_ids"])


def test_review_queue_package_view_shows_grouped_observations(tmp_path: Path) -> None:
    package = build_source_extraction_packages(_store_with_records(tmp_path))["packages"][0]

    assert package["record_count"] == 3
    assert sum(len(section["observations"]) for section in package["sections"]) == 3


def test_review_queue_package_actions_accept_reject_defer(tmp_path: Path) -> None:
    sql_accept = _store_with_records(tmp_path / "accept")
    package = build_source_extraction_packages(sql_accept)["packages"][0]
    accepted = accept_package_after_source_comparison(sql_accept, package["record_ids"])
    assert accepted["success"] is True
    assert all(not sql_accept.get_record(record_id).requires_review for record_id in package["record_ids"])

    sql_reject = _store_with_records(tmp_path / "reject")
    package = build_source_extraction_packages(sql_reject)["packages"][0]
    rejected = reject_package(sql_reject, package["record_ids"])
    assert rejected["success"] is True
    assert all(sql_reject.get_record(record_id).tier == "superseded" for record_id in package["record_ids"])

    sql_defer = _store_with_records(tmp_path / "defer")
    package = build_source_extraction_packages(sql_defer)["packages"][0]
    deferred = defer_package(sql_defer, package["record_ids"])
    assert deferred["success"] is True
    assert all(sql_defer.get_record(record_id).requires_review for record_id in package["record_ids"])


def test_atomic_review_fallback_remains_available(tmp_path: Path) -> None:
    sql = _store_with_records(tmp_path)
    package_model = build_source_extraction_packages(sql)
    atomic_model = build_mkb_explorer_model(sql, tier_filter="review_bound")

    assert package_model["atomic_review_fallback_preserved"] is True
    assert atomic_model["row_count"] == 3


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
            decision_reason="test_source_packages_14d",
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


def _run_launcher_batch(tmp_path: Path):
    launcher.TEST_INPUT_DIR = tmp_path / "test_input"
    launcher.TEST_REVIEW_DIR = tmp_path / "test_review"
    launcher.TEST_ARCHIVE_DIR = tmp_path / "test_archive"
    launcher.TEST_OUTPUT_DIR = tmp_path / "test_output"
    launcher.TEST_RUN_REPORT_DIR = tmp_path / "reports" / "test_runs"
    launcher.LATEST_JSON_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.json"
    launcher.LATEST_MD_REPORT = launcher.TEST_RUN_REPORT_DIR / "latest_test_run.md"
    pipeline = ExecutionPipeline(
        sql_store=SQLiteStore(db_path=tmp_path / "runtime_mkb.db", encryption_key=""),
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    launcher.TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (launcher.TEST_INPUT_DIR / "private_real_upload.png").write_bytes(b"not-real-image")
    launcher.recover_text_from_image_local = lambda path: launcher.LocalImageOcrResult(
        available=True,
        attempted=True,
        text=SYNTHETIC_OCR_TEXT,
        engine="tesseract_local",
        language="eng",
        text_visibility="recovered",
    )
    return launcher.run_medai_test_batch(
        pipeline,
        specialty="urology",
        document_category="Cross-domain OCR",
    )


def test_run_review_summary_shows_source_extraction_packages_created(tmp_path: Path) -> None:
    summary = _run_launcher_batch(tmp_path)
    item = summary.results[0]

    assert item["source_extraction_packages_created"] == 1
    assert "Source extraction packages created" in item["source_package_summary"]
    assert "human review required" in item["source_package_summary"]


def test_mkb_explorer_counts_remain_accurate(tmp_path: Path) -> None:
    sql = _store_with_records(tmp_path)
    before = build_mkb_explorer_model(sql, tier_filter="review_bound")["counts"]
    build_source_extraction_packages(sql)
    after = build_mkb_explorer_model(sql, tier_filter="review_bound")["counts"]

    assert before == after
    assert after["review_bound"] == 3


def test_reports_contain_no_raw_ocr_text_or_private_paths() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_source_extraction_packages_14d.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_dir = REPO_ROOT / "reports" / "medai_source_extraction_packages_14d"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert SYNTHETIC_OCR_TEXT not in text
        assert "private_real_upload.png" not in text
        assert "C:\\" not in text
        result = check_public_report_payload(json.loads(text) if path.suffix == ".json" else text)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_external_api_false_and_auto_accept_false() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_source_extraction_packages_14d.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_source_extraction_packages_14d" / "medai_source_extraction_packages_14d_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["external_api_used"] is False
    assert payload["auto_accept"] is False
