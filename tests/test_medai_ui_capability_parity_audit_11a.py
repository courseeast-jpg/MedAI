"""Tests for MEDAI-UI-CAPABILITY-PARITY-AUDIT-11A."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_medai_ui_capability_parity_audit_11a import (  # noqa: E402
    MATRIX_CSV_PATH,
    REPORT_JSON_PATH,
    SUMMARY_MD_PATH,
    build_report,
    classify_priorities,
    current_ui_inventory,
    feature_parity_matrix,
    historical_capabilities,
)


def test_current_ui_inventory_detects_run_review():
    names = {item["capability"] for item in current_ui_inventory()}
    assert "Run & Review" in names


def test_current_ui_inventory_detects_operator_control_panel():
    names = {item["capability"] for item in current_ui_inventory()}
    assert "Operator Control Panel" in names


def test_current_ui_inventory_detects_advanced_tools():
    names = {item["capability"] for item in current_ui_inventory()}
    assert "Advanced tools toggle" in names


def test_historical_scan_detects_specialty_domain_terms():
    caps = historical_capabilities()
    specialty = [item for item in caps if item["capability"] == "specialty/domain selector"]
    assert specialty
    assert specialty[0]["evidence_found"] is True


def test_matrix_includes_specialty_domain_selector_row():
    rows = feature_parity_matrix()
    assert any(row["expected_capability"] == "specialty/domain selector" for row in rows)


def test_missing_degraded_classification_supports_priority_levels():
    priorities = classify_priorities(feature_parity_matrix())
    assert set(priorities) == {"priority_0", "priority_1", "priority_2", "priority_3"}
    assert "specialty/domain selector" in priorities["priority_1"]


def test_report_contains_no_private_paths_raw_filenames_or_phi():
    from clinical_knowledge.privacy import check_public_report_payload

    report = build_report()
    result = check_public_report_payload(report)
    assert result.passed, result.leak_examples_redacted


def test_external_api_used_false():
    assert build_report()["external_api_used"] is False


def test_auto_accept_enabled_false():
    assert build_report()["auto_accept_enabled"] is False


def test_script_exits_with_valid_conclusion():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_ui_capability_parity_audit_11a.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
    assert payload["conclusion"] in {
        "ui_capability_parity_ready_no_critical_gaps",
        "ui_capability_parity_gaps_found_repair_required",
        "not_ready",
    }
    assert payload["conclusion"] == "ui_capability_parity_gaps_found_repair_required"
    assert payload["privacy_check_passed"] is True
    assert MATRIX_CSV_PATH.is_file()
    assert SUMMARY_MD_PATH.is_file()
