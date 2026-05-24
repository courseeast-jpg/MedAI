"""End-to-end focused tests for the 03 operator-review-UX block.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03-OPERATOR-REVIEW-UX.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_03_OPERATOR_REVIEW_UX.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux_audit.json"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_03_OPERATOR_REVIEW_UX_AUDIT.md"


@pytest.fixture(scope="module")
def validation_payload() -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "run_medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux.py"
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
    return json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))


def test_validation_script_reports_operator_review_ux_ready(validation_payload):
    assert validation_payload["conclusion"] == "operator_review_ux_ready"
    assert validation_payload["all_pass"] is True


def test_action_counts_cover_accept_reject_defer(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["accepted_action_count"] == 1
    assert f["rejected_action_count"] == 1
    assert f["deferred_action_count"] == 1


def test_state_transitions_match_action_table(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["review_bound_records_before_action"] >= 3
    assert f["active_records_after_action"] >= 1
    assert f["review_bound_records_after_action"] >= 1
    assert f["rejected_or_superseded_records_after_action"] >= 1


def test_ledger_records_one_event_per_action(validation_payload):
    f = validation_payload["audit_findings"]
    # Three primary actions + (missing record + already-active accept retry)
    # error paths must NOT write ledger events.
    assert f["ledger_event_count"] == 3


def test_ui_render_plan_exposes_actions_for_review_bound_rows(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["ui_action_proof_count"] >= 3


def test_external_api_not_used_and_no_auto_accept(validation_payload):
    assert validation_payload["external_api_used"] is False
    assert validation_payload["auto_accept_enabled"] is False
    f = validation_payload["audit_findings"]
    assert f["external_api_used"] is False
    assert f["auto_accept_allowed_in_any_record"] is False


def test_no_raw_source_line_in_public_report(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["raw_source_line_in_public_report"] is False


def test_error_paths_recorded_but_no_state_change(validation_payload):
    """Missing record and already-active probes must surface error codes."""
    per = validation_payload["audit_findings"]["per_record_results"]
    error_codes = {r.get("error_code") for r in per if not r.get("success")}
    assert "record_not_found" in error_codes
    assert "record_already_active" in error_codes
    # And those error paths must not have written ledger events
    # (counted by test_ledger_records_one_event_per_action above).


def test_public_reports_pass_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    paths = [p for p in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH, AUDIT_JSON_PATH, AUDIT_MD_PATH) if p.is_file()]
    assert paths, "no 03 report files generated yet"
    for path in paths:
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_pipeline_surfaces_record_ids_and_states():
    pipeline_source = (REPO_ROOT / "execution" / "pipeline.py").read_text(encoding="utf-8")
    assert "extracted_medical_fact_record_ids" in pipeline_source
    assert "extracted_medical_fact_record_states" in pipeline_source


def test_test_launcher_carries_record_ids_and_states_into_test_file_result():
    from app.test_launcher import TestFileResult

    result = TestFileResult(file_name="example", status="accepted")
    assert result.extracted_medical_fact_record_ids == []
    assert result.extracted_medical_fact_record_states == {}


def test_extracted_information_preview_includes_action_panel_plan():
    from app.extracted_information_preview import build_extracted_information_preview_plan

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
                "requires_review": True,
                "auto_accept_allowed": False,
                "source_line_hash": "a" * 12,
            }
        ],
        "extracted_medical_fact_count": 1,
        "extraction_to_mkb_review_count": 1,
        "extracted_medical_fact_record_ids": ["rec-abc-123"],
        "extracted_medical_fact_record_states": {
            "rec-abc-123": {
                "fact_type": "test_result",
                "tier": "quarantined",
                "status": "active",
                "requires_review": True,
                "ddi_status": "",
            }
        },
    }
    plan = build_extracted_information_preview_plan(item)
    row_actions = plan.get("row_actions") or []
    assert len(row_actions) == 1
    actions = {a["key"]: a for a in row_actions[0]["actions"]}
    assert actions["accept_after_source_comparison"]["enabled"] is True
    assert actions["accept_after_source_comparison"]["disclaimer"]
    assert actions["reject_extracted_fact"]["enabled"] is True
    assert actions["defer_extracted_fact"]["enabled"] is True
    assert plan.get("operator_review_disclaimers")


def test_existing_01_and_02_block_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_extracted_medical_facts.py",
            "tests/test_run_review_extracted_information_preview.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_01.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
