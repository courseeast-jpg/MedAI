from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology import (
    TerminologyImportCheckpoint,
    TerminologyImportLimits,
    build_import_limits,
    inventory_terminology_data_dir,
    plan_terminology_import,
    run_terminology_import_dry_run,
    simulate_checkpoint_resume,
)
from clinical_knowledge.terminology.intake_automation import compute_readiness


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_system_file(repo_root: Path, system: str, filename: str, body: str = "synthetic") -> Path:
    target = repo_root / "terminology_data" / system / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_import_limits_defaults_safe() -> None:
    limits = TerminologyImportLimits()
    summary = limits.safe_public_summary()
    assert summary["require_license_ack_for_real_import"] is True
    assert summary["allow_synthetic_test_import"] is True
    assert summary["allow_real_import"] is False
    assert summary["chunk_size"] > 0
    assert summary["checkpoint_interval_rows"] > 0


def test_real_import_disabled_by_default_even_when_requested() -> None:
    limits = build_import_limits(allow_real_import=False)
    assert limits.allow_real_import is False


def test_no_files_plan_reports_missing(tmp_path: Path) -> None:
    result = run_terminology_import_dry_run(repo_root=tmp_path)
    plan = result["plan"]
    assert plan["estimated_files"] == 0
    assert sorted(plan["systems_missing"]) == ["loinc", "rxnorm", "snomed_ct", "umls"]
    assert plan["import_allowed"] is False
    assert result["real_files_imported"] is False


def test_files_but_no_ack_are_blocked_license(tmp_path: Path) -> None:
    _write_system_file(tmp_path, "umls", "MRCONSO.RRF")
    result = run_terminology_import_dry_run(repo_root=tmp_path)
    plan = result["plan"]
    assert plan["systems_seen"] == ["umls"]
    assert plan["systems_blocked_license"] == ["umls"]
    assert plan["systems_import_ready"] == []
    assert plan["import_allowed"] is False


def test_files_with_test_ack_are_import_ready_but_dry_run_only(tmp_path: Path) -> None:
    _write_system_file(tmp_path, "loinc", "Loinc.csv", "LOINC_NUM,COMPONENT,LONG_COMMON_NAME\n1-1,A,B")
    result = run_terminology_import_dry_run(
        repo_root=tmp_path,
        license_test_mode=True,
        license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
    )
    plan = result["plan"]
    assert plan["systems_import_ready"] == ["loinc"]
    assert plan["import_allowed"] is False
    assert plan["dry_run"] is True
    assert plan["real_files_imported"] is False


def test_row_caps_and_chunking_plan_generated(tmp_path: Path) -> None:
    _write_system_file(tmp_path, "snomed_ct", "sct2_Concept_Snapshot_INT.txt")
    _write_system_file(tmp_path, "snomed_ct", "sct2_Description_Snapshot-en_INT.txt")
    limits = build_import_limits(max_rows_per_file=1_000, max_rows_per_system=1_500, chunk_size=400)
    inv = inventory_terminology_data_dir(
        repo_root=tmp_path,
        license_test_mode=True,
        license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
    )
    readiness = compute_readiness(
        repo_root=tmp_path,
        license_test_mode=True,
        license_env={"CKA_TERM01_TEST_LICENSE_ACK": "1"},
    )
    plan = plan_terminology_import(inv, readiness, limits, dry_run=True)
    summary = plan.safe_public_summary()
    assert summary["estimated_files"] == 2
    assert summary["estimated_rows_safe"] == 1_500
    assert summary["estimated_chunks"] == 4
    assert summary["row_caps_applied"]["snomed_ct"] is True


def test_checkpoint_model_safe_summary_and_resume() -> None:
    checkpoint = TerminologyImportCheckpoint(
        system="rxnorm",
        source_safe_id="source_safe_001",
        file_safe_id="file_safe_001",
        rows_seen=25,
        rows_imported=20,
        chunk_index=1,
    )
    resumed = simulate_checkpoint_resume(
        checkpoint,
        additional_rows_seen=10,
        additional_rows_imported=8,
        chunk_increment=1,
    )
    summary = resumed.safe_public_summary()
    assert summary["rows_seen"] == 35
    assert summary["rows_imported"] == 28
    assert summary["chunk_index"] == 2
    text = json.dumps(summary)
    assert ":\\" not in text
    assert "/tmp/" not in text


def test_dry_run_cli_does_not_import_or_create_index(tmp_path: Path) -> None:
    _write_system_file(tmp_path, "rxnorm", "RXNCONSO.RRF")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "cka_terminology_import_dry_run.py"),
            "--terminology-root",
            str(tmp_path / "terminology_data"),
            "--max-rows-per-file",
            "10",
            "--chunk-size",
            "5",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["real_files_imported"] is False
    assert payload["production_index_created"] is False
    assert payload["plan"]["import_allowed"] is False
    assert not (tmp_path / "data" / "terminology").exists()


def test_no_terminology_data_files_staged() -> None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0
    staged = result.stdout.splitlines()
    assert not any(line.startswith("terminology_data/") for line in staged)
    assert not any(line.startswith("data/terminology/") for line in staged)


def test_validation_report_privacy_clean() -> None:
    from scripts.run_cka_term01b_import_planner_validation import run_validation

    report = run_validation()
    assert report["all_passed"] is True
    assert report["external_api_used"] is False
    assert report["no_real_import_performed"] is True
    assert report["clinical_recommendations_generated"] is False
    assert report["prescription_dosing_advice_generated"] is False
    check = check_public_report_payload(report)
    assert check.passed


@pytest.mark.parametrize(
    "needle",
    ["clinical recommendation", "dosing advice", "source_response_raw", "replacement_map"],
)
def test_public_report_does_not_contain_forbidden_text(needle: str) -> None:
    report_path = REPO_ROOT / "reports" / "cka_term01b_import_planner" / "cka_term01b_import_planner_report.json"
    if not report_path.exists():
        from scripts.run_cka_term01b_import_planner_validation import run_validation

        run_validation()
    text = report_path.read_text(encoding="utf-8").lower()
    assert needle not in text
