"""Focused tests for MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_adapter_fallback_run_review_10"
REPORT_JSON_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_report.json"
AUDIT_JSON_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_audit.json"
REPORT_MD_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_report.md"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_UI_ADAPTER_FALLBACK_RUN_REVIEW_10_AUDIT.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_ADAPTER_FALLBACK_RUN_REVIEW_10.md"

SYNTHETIC_TEXT = (
    "Lab result report\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Platelets: 230 x10E9/L (ref 150-450)\n"
)


@pytest.fixture()
def sql_store(tmp_path: Path):
    from mkb.sqlite_store import SQLiteStore

    return SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")


def test_local_adapter_fallback_persists_review_bound_records(sql_store):
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review

    result = process_adapter_fallback_run_review(sql_store, raw_text=SYNTHETIC_TEXT, session_id="test-10")
    assert result["ready"] is True
    assert result["structured_facts_extracted"] == 4
    assert result["review_bound_records_persisted"] == 4
    assert result["ui_preview_rows"] == 4
    assert result["external_api_used"] is False
    assert result["auto_accept_enabled"] is False
    item = result["run_item"]
    assert item["extraction_to_mkb_written_count"] == 0
    assert item["extraction_to_mkb_review_count"] == 4
    assert item["auto_accept_allowed"] is False
    assert len(item["extracted_medical_fact_record_ids"]) == 4
    for state in item["extracted_medical_fact_record_states"].values():
        assert state["tier"] == "quarantined"
        assert state["requires_review"] is True


def test_operator_actions_remain_review_bound_and_manual(sql_store):
    from app.local_adapter_fallback_processor import (
        process_adapter_fallback_run_review,
        run_operator_action_proof,
    )

    result = process_adapter_fallback_run_review(sql_store, raw_text=SYNTHETIC_TEXT, session_id="test-10")
    ids = list(result["run_item"]["extracted_medical_fact_record_ids"])
    assert run_operator_action_proof(sql_store, ids, session_id="test-10") == 3


def test_run_review_execution_none_branch_uses_adapter_fallback_when_sql_exists():
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "def render_adapter_fallback_panel" in main_source
    assert 'if sys_components.get("execution") is None:' in main_source
    assert 'sys_components.get("sql") is not None' in main_source
    assert "render_adapter_fallback_panel(sys_components)" in main_source


def test_validation_script_reports_ready():
    env = os.environ.copy()
    env.update(
        {
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_ui_adapter_fallback_run_review_10.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "ui_adapter_fallback_run_review_ready"
    assert payload["fallback_mode_ready"] is True
    f = payload["audit_findings"]
    assert f["structured_facts_extracted"] == 4
    assert f["review_bound_records_persisted"] == 4
    assert f["retrieval_proof_count"] == 4
    assert f["ui_preview_rows"] == 4
    assert f["operator_action_proof_count"] == 3
    assert f["run_review_works_when_execution_none"] is True
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False


def test_public_reports_pass_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    paths = [REPORT_JSON_PATH, AUDIT_JSON_PATH, REPORT_MD_PATH, AUDIT_MD_PATH, SHORT_MD_PATH]
    assert all(path.is_file() for path in paths)
    for path in paths:
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"
