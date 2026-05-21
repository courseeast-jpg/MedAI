"""Focused tests for MEDAI-V2-FREEZE-MAINTENANCE-ONLY-01."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_freeze_maintenance_only_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_FREEZE_MAINTENANCE_ONLY_01.md",
    REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.json",
    REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.md",
)
ALLOWED_CHANGED_PATHS = {
    "reports/medai_v2_freeze_maintenance_only_01/MEDAI_V2_FREEZE_MAINTENANCE_ONLY_01.md",
    "reports/medai_v2_freeze_maintenance_only_01/medai_v2_freeze_maintenance_only_01_report.json",
    "reports/medai_v2_freeze_maintenance_only_01/medai_v2_freeze_maintenance_only_01_report.md",
    "scripts/run_medai_v2_freeze_maintenance_only_01.py",
    "tests/test_medai_v2_freeze_maintenance_only_01.py",
}


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.json").read_text(encoding="utf-8"))


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_json_required_booleans_have_expected_values():
    payload = _payload()
    for key in (
        "implementation_started",
        "new_helper_created",
        "direct_implementation_recommended",
        "runtime_behavior_changed",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
        "launcher_changed",
        "startup_config_changed",
        "extraction_changed",
        "ocr_changed",
        "db_schema_changed",
        "migration_created",
        "private_adapter_implemented",
        "runtime_wiring_added",
        "external_api_used",
        "private_data_accessed",
        "source_documents_opened",
        "raw_text_read",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "licensed_rows_read",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "tags_touched",
        "tags_created",
        "tags_modified",
        "cue_expansion_recommended",
    ):
        assert payload.get(key) is False, key
    for key in (
        "reports_only",
        "freeze_maintenance_snapshot_created",
        "freeze_maintenance_only_selected",
        "preceding_release_freeze_snapshot_present",
        "status_registry_created_previously",
        "terminology_private_adapter_blocked",
        "cue_expansion_blocked",
        "clinical_decision_logic_expansion_blocked",
        "freeze_inventory_created",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
    ):
        assert payload.get(key) is True, key


def test_prior_v2_report_folders_and_status_registry_exist():
    for name in (
        "medai_v2_architecture_spec_01",
        "medai_v2_foundation_spec_02",
        "medai_v2_runtime_contracts_01",
        "medai_v2_validation_harness_01",
        "medai_v2_ui_shell_spec_01",
        "medai_v2_data_infra_spec_01",
        "medai_v2_extraction_spec_01",
        "medai_v2_roadmap_02",
        "medai_v2_foundation_implementation_readiness_01",
        "medai_v2_packaging_spec_01",
        "medai_v2_roadmap_park_01",
        "medai_v2_foundation_default_off_implementation_plan_01",
        "medai_v2_foundation_default_off_status_registry_01",
        "medai_v2_roadmap_03",
        "medai_v2_roadmap_park_02",
        "medai_v2_release_freeze_snapshot_01",
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name
    assert (REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py").is_file()


def test_report_summarizes_chain_registry_and_release_freeze_decision():
    payload = _payload()
    text = _report_text()
    assert payload["freeze_maintenance_only_selected"] is True
    assert payload["release_freeze_decision_carried_forward"]["current_v2_state_release_frozen"] is True
    assert len(payload["freeze_maintenance_chain"]) == 17
    assert payload["status_registry_entry_count"] == 13
    assert payload["status_registry_default_enabled_count"] == 0
    assert payload["status_registry_runtime_wired_count"] == 0
    assert payload["status_registry_ui_wired_count"] == 0
    assert payload["status_registry_blocked_entry_count"] == 3
    assert "Release-Freeze Decision Carried Forward" in text
    assert "Freeze-Maintenance Decision" in text


def test_freeze_inventory_and_allowed_future_work():
    payload = _payload()
    text = _report_text()
    assert payload["freeze_inventory"]["existing_parked_frozen_anchors"]["v1_frozen_release"] == "7ef8ffd"
    assert payload["freeze_inventory"]["current_v2_chain"]["release_freeze_snapshot_01"] == "ae4b70e"
    assert payload["only_allowed_future_work"] == [
        "bugfixes_preserving_existing_frozen_behavior",
        "validation_receipt_refreshes",
        "documentation_report_corrections",
        "security_privacy_maintenance",
        "future_roadmap_audit_if_new_signal_appears",
    ]
    assert "Allowed Future Maintenance Work" in text


def test_recommended_next_sequence_and_no_direct_implementation():
    payload = _payload()
    assert payload["recommended_next_block"] == "V2-ROADMAP-04_IF_NEW_SIGNAL_ELSE_FREEZE-MAINTENANCE-ONLY"
    assert payload["recommended_next_3_blocks"] == [
        "V2-ROADMAP-04_IF_NEW_SIGNAL_ELSE_FREEZE-MAINTENANCE-ONLY",
        "V2-FOUNDATION-NEXT-DEFAULT-OFF-PLAN-01_OR_V2-VALIDATION-REGISTRY-INTEGRATION-PLAN-01_ONLY_IF_ROADMAP_APPROVES",
        "V2-ROADMAP-PARK-03_OR_NEXT_RELEASE-FREEZE_SNAPSHOT_ONLY_IF_NEW_WORK_OCCURS",
    ]
    assert payload["direct_implementation_recommended"] is False


def test_blocked_deferred_work_and_cue_expansion_preserved():
    payload = _payload()
    text = _report_text()
    for item in (
        "private_adapter_implementation",
        "real_private_store_access",
        "licensed_terminology_row_reads",
        "runtime_db_migration",
        "extraction_ocr_behavior_changes",
        "ui_implementation",
        "packaging_implementation",
        "launcher_changes",
        "clinical_decision_logic_expansion",
        "direct_v2_implementation_without_future_roadmap_approval",
    ):
        assert item in payload["blocked_work"]
    assert "CUE-EXPANSION" in payload["deferred_or_not_recommended"]
    assert "Cue expansion remains explicitly not recommended" in text


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        assert check_public_report_payload(target).passed, path.name


def test_implementation_files_limited_to_reports_script_tests():
    result = subprocess.run(
        ["git", "status", "--short", "-uall"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line[3:].replace("\\", "/") for line in result.stdout.splitlines() if line and line[:2].strip()}
    assert changed <= ALLOWED_CHANGED_PATHS


def test_runtime_files_and_tags_not_modified_by_block():
    payload = _payload()
    assert payload["tags_created"] is False
    assert payload["tags_modified"] is False
    for path in (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "startup_preflight.py",
        REPO_ROOT / "app" / "config.py",
        REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    ):
        assert "MEDAI-V2-FREEZE-MAINTENANCE-ONLY-01" not in path.read_text(encoding="utf-8", errors="replace")

