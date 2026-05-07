from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.terminology.privacy_regression import (
    SYNTHETIC_LICENSE_TEXT,
    assert_public_report_safe,
    run_privacy_regression_checks,
    sanitize_formula_cell_for_public,
)
from clinical_knowledge.terminology.safety_redteam import run_terminology_safety_redteam
from clinical_knowledge.terminology.staging_guard import check_terminology_staging


def test_raw_absolute_path_injection_blocked() -> None:
    result = run_privacy_regression_checks()
    assert result.raw_path_leak_blocked is True


def test_license_text_leak_blocked() -> None:
    result = run_privacy_regression_checks()
    assert result.license_text_leak_blocked is True
    with pytest.raises(ValueError):
        assert_public_report_safe({"leak": SYNTHETIC_LICENSE_TEXT})


def test_fake_ack_and_ack_mismatch_blocked() -> None:
    result = run_terminology_safety_redteam()
    assert result.fake_ack_blocked is True
    assert result.ack_mismatch_blocked is True


def test_staging_guard_detects_terminology_data_and_data_terminology() -> None:
    result = check_terminology_staging(staged_paths=["terminology_data/umls/MRCONSO.RRF", "data/terminology/term.db"])
    assert result.terminology_data_staged is True
    assert result.data_terminology_staged is True
    assert "terminology_data_staged" in result.blocked_reason_codes
    assert "data_terminology_staged" in result.blocked_reason_codes


def test_zip_slip_and_malformed_rows_flagged() -> None:
    result = run_terminology_safety_redteam()
    assert result.zip_slip_blocked is True
    assert result.malformed_rows_skipped is True


def test_csv_formula_injection_neutralized() -> None:
    assert sanitize_formula_cell_for_public("=cmd").startswith("'=")
    result = run_terminology_safety_redteam()
    assert result.csv_formula_injection_neutralized is True


def test_lookup_safety_and_b07_boundaries() -> None:
    result = run_terminology_safety_redteam()
    assert result.ambiguity_not_silently_resolved is True
    assert result.unknown_code_not_hallucinated is True
    assert result.b07_hypothesis_promotion_blocked is True
    assert result.b07_ddi_clear_blocked is True


def test_external_api_and_clinical_advice_absent() -> None:
    result = run_terminology_safety_redteam()
    assert result.external_api_blocked is True
    assert result.external_api_used is False
    assert result.clinical_advice_absent is True
    assert result.clinical_recommendations_generated is False
    assert result.prescription_dosing_advice_generated is False


def test_safe_public_summary_has_required_fields() -> None:
    payload = run_terminology_safety_redteam().safe_public_summary()
    assert payload["conclusion"] == "cka_term01h_safety_redteam_ready"
    assert payload["no_real_import_performed"] is True
    assert payload["real_terminology_files_committed"] is False
    assert payload["raw_phi_logged_in_public_reports"] is False
    assert payload["private_filename_path_leaks"] == 0
    assert payload["license_text_written_to_public_reports"] is False
    assert_public_report_safe(payload)


def test_privacy_guard_cli_runs() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/cka_terminology_privacy_guard.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["external_api_used"] is False
    assert payload["no_real_import_performed"] is True


def test_validation_script_generates_report() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term01h_safety_redteam_validation.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term01h_safety_redteam_ready" in proc.stdout
    report = repo / "reports" / "cka_term01h_safety_redteam" / "cka_term01h_safety_redteam_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["external_api_used"] is False
    assert payload["raw_phi_logged_in_public_reports"] is False
    assert payload["private_filename_path_leaks"] == 0
    assert "C:\\" not in json.dumps(payload)


def test_no_terminology_data_or_private_files_staged() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    staged = proc.stdout.replace("\\", "/")
    assert "terminology_data/" not in staged
    assert "data/terminology/" not in staged
    assert "LICENSE_ACK_PRIVATE.json" not in staged
