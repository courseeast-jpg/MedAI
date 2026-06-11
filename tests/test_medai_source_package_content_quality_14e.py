"""Focused tests for MEDAI-SOURCE-PACKAGE-CONTENT-QUALITY-14E."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import app.test_launcher as launcher
from app.schemas import MKBRecord
from app.source_extraction_packages import (
    accept_package_after_source_comparison,
    build_source_extraction_packages,
    normalize_package_observation_fields,
    package_action_plan,
)
from clinical_knowledge.privacy import check_public_report_payload
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_OCR_TEXT = """Patient Portal Results
Glucose | 5.4 | mmol/L | 3.9-5.5 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H

Vitamin D
Value: 24 ng/mL
Normal range: 30-100

Findings:
Visible source section captured for review.
"""


def _record(
    *,
    name: str,
    value: str,
    section: str = "Patient Portal Results",
    reference: str = "",
    flag: str = "",
    category: str = "Urinalysis",
    family: str = "Unknown",
    session_id: str = "session-14e",
    parser_name: str = "cross_domain_visible_observation_adapter",
) -> MKBRecord:
    return MKBRecord(
        fact_type="test_result",
        content=f"{name} {value}",
        structured={
            "test_name": name,
            "value": value,
            "unit": "mg/dL",
            "reference_range": reference,
            "flag": flag,
            "section_heading": section,
            "candidate_kind": "table_observation",
            "document_category": category,
            "document_family": family,
            "source_modality": "image_ocr",
            "parser_name": parser_name,
            "requires_human_review": True,
            "auto_accept_allowed": False,
        },
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


def _store(tmp_path: Path, records: list[MKBRecord]) -> SQLiteStore:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sql = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    for record in records:
        sql.write_record(record, session_id=record.session_id)
    return sql


def test_package_row_normalization_splits_safe_normal_range_and_normal_value_patterns() -> None:
    row = normalize_package_observation_fields(
        label="Bilirubin",
        value="Normal range: 0.2 - 1.0 mg/dL Normal value: Negative",
    )

    assert row["value"] == "Negative"
    assert row["reference_interval"] == "0.2 - 1.0 mg/dL"
    assert row["normalization_status"] == "normalized"


def test_package_row_normalization_removes_duplicated_label_text_from_value() -> None:
    row = normalize_package_observation_fields(label="Normal value", value="Normal value: Negative")

    assert row["value"] == "Negative"
    assert "Normal value:" not in row["value"]


def test_lab_table_observation_preserves_label_value_reference_flag_unit(tmp_path: Path) -> None:
    sql = _store(
        tmp_path,
        [_record(name="WBC", value="11.2", reference="4.0-10.0", flag="H")],
    )
    observation = build_source_extraction_packages(sql)["packages"][0]["sections"][0]["observations"][0]

    assert observation["label"] == "WBC"
    assert observation["value"] == "11.2"
    assert observation["reference_interval"] == "4.0-10.0"
    assert observation["flag"] == "H"
    assert observation["unit"] == "mg/dL"


def test_ambiguous_malformed_rows_remain_review_bound_notes(tmp_path: Path) -> None:
    sql = _store(
        tmp_path,
        [_record(name="Normal range", value="0.2 - 1.0 mg/dL Normal value: Negative")],
    )
    observation = build_source_extraction_packages(sql)["packages"][0]["sections"][0]["observations"][0]

    assert observation["row_kind"] == "source_visible_note"
    assert observation["review_status"] == "review-bound"
    assert observation["normalization_status"] == "malformed_note"


def test_more_eligible_review_bound_records_are_grouped_and_ungrouped_reported(tmp_path: Path) -> None:
    eligible_without_parser = _record(name="Specific gravity", value="1.020", parser_name="")
    eligible_without_parser.structured["source_visible_observation"] = False
    ungrouped = _record(name="Medication", value="example")
    ungrouped.fact_type = "medication"
    sql = _store(tmp_path, [_record(name="Glucose", value="Negative"), eligible_without_parser, ungrouped])
    model = build_source_extraction_packages(sql)

    assert model["observations_grouped"] == 2
    assert model["ungrouped_records_count"] == 1


def test_unknown_package_type_count_and_selected_category_propagate(tmp_path: Path) -> None:
    sql = _store(tmp_path, [_record(name="Glucose", value="Negative", category="General", family="Unknown")])
    model = build_source_extraction_packages(sql)
    package = model["packages"][0]

    assert model["unknown_type_package_count"] == 1
    assert package["selected_document_category"] == "General"


def test_weak_treatment_plan_cues_do_not_override_selected_urinalysis_category(tmp_path: Path) -> None:
    sql = _store(tmp_path, [_record(name="Glucose", value="Negative", category="Urinalysis", family="Treatment plan")])
    package = build_source_extraction_packages(sql)["packages"][0]

    assert package["selected_document_category"] == "Urinalysis"
    assert package["detected_document_family_type"] == "Urinalysis"


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
            decision_reason="test_source_package_quality_14e",
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
        text=RAW_OCR_TEXT,
        engine="tesseract_local",
        language="eng",
        text_visibility="recovered",
    )
    return launcher.run_medai_test_batch(pipeline, specialty="urology", document_category="Urinalysis")


def test_run_review_summary_shows_packages_observations_and_ungrouped(tmp_path: Path) -> None:
    item = _run_launcher_batch(tmp_path).results[0]

    assert "Source extraction packages created" in item["source_package_summary"]
    assert "Observations grouped:" in item["source_package_summary"]
    assert "Ungrouped records:" in item["source_package_summary"]


def test_review_queue_package_view_uses_package_first_display() -> None:
    import app.main as main

    source = inspect.getsource(main.render_review_queue_tab)
    assert "Source extraction packages" in source
    assert "Atomic record actions" in source


def test_accept_reject_defer_visual_semantics() -> None:
    actions = package_action_plan(["record-1"])["actions"]
    by_key = {action["key"]: action for action in actions}

    assert by_key["accept_package_after_source_comparison"]["visual_semantic"] == "non_red_primary"
    assert by_key["reject_package"]["visual_semantic"] == "red_destructive"
    assert by_key["defer_package"]["visual_semantic"] == "neutral_secondary"


def test_package_actions_call_existing_review_transitions_and_do_not_auto_accept(tmp_path: Path) -> None:
    sql = _store(tmp_path, [_record(name="Glucose", value="Negative")])
    package = build_source_extraction_packages(sql)["packages"][0]

    result = accept_package_after_source_comparison(sql, package["record_ids"])

    assert result["success"] is True
    assert result["auto_accept_allowed"] is False
    assert sql.get_record(package["record_ids"][0]).requires_review is False


def test_reports_contain_no_raw_ocr_text_or_private_paths() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_source_package_content_quality_14e.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_dir = REPO_ROOT / "reports" / "medai_source_package_content_quality_14e"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert RAW_OCR_TEXT not in text
        assert "private_real_upload.png" not in text
        assert "C:\\" not in text
        result = check_public_report_payload(json.loads(text) if path.suffix == ".json" else text)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_external_api_false_and_auto_accept_false() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_source_package_content_quality_14e.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_source_package_content_quality_14e" / "medai_source_package_content_quality_14e_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["external_api_used"] is False
    assert payload["auto_accept"] is False
