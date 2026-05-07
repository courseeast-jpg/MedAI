from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "cka_term01i_term02_blueprint"
BLUEPRINT = REPORT_DIR / "CKA_TERM02_EXECUTION_BLUEPRINT.md"
RUNBOOK = REPORT_DIR / "CKA_TERM02_OPERATOR_RUNBOOK.md"
MATRIX = REPORT_DIR / "CKA_TERM02_STOP_ON_FAILURE_MATRIX.md"


def _combined_docs() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in (BLUEPRINT, RUNBOOK, MATRIX))


def test_blueprint_runbook_and_matrix_exist() -> None:
    assert BLUEPRINT.exists()
    assert RUNBOOK.exists()
    assert MATRIX.exists()


def test_term01_through_term01h_are_mentioned() -> None:
    text = _combined_docs()
    for phase in ("TERM-01", "TERM-01A", "TERM-01B", "TERM-01C", "TERM-01D", "TERM-01E", "TERM-01F", "TERM-01G", "TERM-01H"):
        assert phase in text


def test_term02_phases_are_defined() -> None:
    text = _combined_docs()
    for phrase in (
        "Preflight",
        "Inventory",
        "License confirmation",
        "Dry-run plan",
        "Capped import",
        "QA harness",
        "B07 boundary check",
        "Privacy report",
        "Commit and tag policy",
    ):
        assert phrase in text


def test_stop_on_failure_matrix_has_required_reasons() -> None:
    text = MATRIX.read_text(encoding="utf-8")
    for reason in (
        "no_supported_files_present",
        "license_ack_private_missing",
        "systems_pending_acknowledgment",
        "terminology_files_staged",
        "data_terminology_staged",
        "row_cap_exceeded",
        "malformed_rows_above_threshold",
        "ambiguous_lookup_regression",
        "unknown_code_hallucination",
        "b07_hypothesis_promotion",
        "b07_ddi_status_changed",
        "external_api_attempted",
        "privacy_report_leak",
        "terminology_db_staged",
    ):
        assert reason in text


def test_operator_commands_present() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for command in (
        "cka_terminology_prepare_intake.py",
        "cka_terminology_check_ready.py",
        "cka_terminology_import_dry_run.py",
        "cka_term02_preflight_gate.py",
        "cka_terminology_run_qa.py",
    ):
        assert command in text


def test_no_instruction_commits_terminology_files_or_private_ack() -> None:
    text = _combined_docs().lower()
    assert "never commit `terminology_data/`" in text
    assert "never commit `data/terminology/`" in text
    assert "never commit `license_ack_private.json`" in text
    assert "commit terminology_data/" not in text
    assert "commit data/terminology/" not in text
    assert "commit license_ack_private.json" not in text


def test_license_gate_not_bypassed_and_no_raw_private_paths() -> None:
    text = _combined_docs().lower()
    assert "license gate" in text
    assert "bypass" not in text
    assert "c:\\" not in text
    assert "g:\\" not in text
    assert "/users/" not in text


def test_no_clinical_advice_or_dosing_in_docs() -> None:
    text = _combined_docs().lower()
    for forbidden in ("recommended dose", "increase dose", "decrease dose", "you should take"):
        assert forbidden not in text


def test_validation_script_generates_ready_report() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term01i_term02_blueprint_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term01i_term02_blueprint_ready" in proc.stdout
    report = REPORT_DIR / "cka_term01i_term02_blueprint_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "cka_term01i_term02_blueprint_ready"
    assert payload["license_gate_preserved"] is True
    assert payload["no_real_import_performed"] is True
    assert payload["terminology_data_staged"] is False
    assert payload["data_terminology_staged"] is False
    assert payload["clinical_recommendations_generated"] is False
    assert payload["prescription_dosing_advice_generated"] is False


def test_no_forbidden_files_staged() -> None:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    staged = proc.stdout.replace("\\", "/")
    assert "terminology_data/" not in staged
    assert "data/terminology/" not in staged
    assert "LICENSE_ACK_PRIVATE.json" not in staged
