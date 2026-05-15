from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.terminology_readiness_viewer import (
    build_terminology_readiness_summary,
    load_public_report,
    load_terminology_readiness_reports,
    render_readiness_text,
)


def test_viewer_imports() -> None:
    import app.terminology_readiness_viewer as viewer

    assert viewer.TERM_REPORTS


def test_missing_reports_handled(tmp_path: Path) -> None:
    reports = load_terminology_readiness_reports({"TERM-01": tmp_path / "missing.json"})
    summary = build_terminology_readiness_summary(reports)
    assert any(item.status == "missing" for item in summary.phase_statuses)
    assert summary.private_ack_file_loaded is False
    assert summary.terminology_data_files_read is False


def test_public_reports_loaded() -> None:
    reports = load_terminology_readiness_reports()
    assert "TERM-01" in reports
    assert reports["TERM-01"]["block_id"] == "CKA-TERM-01"


def test_private_files_not_read(tmp_path: Path) -> None:
    private_file = tmp_path / "LICENSE_ACK_PRIVATE.json"
    private_file.write_text('{"acknowledged": true}', encoding="utf-8")
    assert load_public_report(private_file) is None


def test_no_terminology_data_files_read(tmp_path: Path) -> None:
    reports = {
        "TERM-01": {
            "conclusion": "synthetic",
            "inventory_summary": {"sources": []},
        }
    }
    summary = build_terminology_readiness_summary(reports)
    assert summary.terminology_data_files_read is False


def test_status_summary_correct_with_synthetic_report_data() -> None:
    reports = {
        "TERM-01": {
            "conclusion": "cka_term01_local_terminology_files_required",
            "inventory_summary": {
                "sources": [
                    {"system": "umls", "file_count": 1, "status": "present", "license_confirmed": False},
                    {"system": "snomed_ct", "file_count": 0, "status": "missing", "license_confirmed": False},
                ]
            },
        },
        "TERM-01B": {
            "conclusion": "cka_term01b_import_planner_ready",
            "systems_missing": ["rxnorm"],
            "systems_blocked_license": ["umls"],
            "systems_import_ready": ["loinc"],
        },
    }
    summary = build_terminology_readiness_summary(reports)
    assert summary.systems_with_files_present == ["umls"]
    assert "snomed_ct" in summary.systems_missing
    assert "rxnorm" in summary.systems_missing
    assert summary.systems_requiring_private_license_ack == ["umls"]
    assert summary.systems_import_ready == ["loinc"]


def test_ui_render_helpers_return_non_empty_text() -> None:
    summary = build_terminology_readiness_summary({})
    text = render_readiness_text(summary)
    assert "Terminology Admin" in text
    assert "real import not run" in text.lower()


def test_existing_app_ui_imports_and_has_tab() -> None:
    import app.main as main

    assert "Terminology Admin" in main.PHASE52_OPERATOR_TABS


def test_report_privacy_clean_after_validation_script() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_cka_term01e_operator_readiness_ui_validation.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term01e_operator_readiness_ui_ready" in result.stdout
    report_path = repo / "reports" / "cka_term01e_operator_readiness_ui" / "cka_term01e_operator_readiness_ui_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["public_reports_only"] is True
    assert data["private_ack_file_loaded"] is False
    assert data["terminology_data_files_read"] is False
    assert data["raw_paths_displayed"] is False
    serialized = json.dumps(data)
    assert "LICENSE_ACK_PRIVATE" not in serialized
    assert "C:\\" not in serialized
