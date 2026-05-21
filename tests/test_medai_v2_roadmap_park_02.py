"""Focused tests for MEDAI-V2-ROADMAP-PARK-02."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_roadmap_park_02"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_ROADMAP_PARK_02.md",
    REPORT_DIR / "medai_v2_roadmap_park_02_report.json",
    REPORT_DIR / "medai_v2_roadmap_park_02_report.md",
)
ALLOWED_CHANGED_PATHS = {
    "reports/medai_v2_roadmap_park_02/MEDAI_V2_ROADMAP_PARK_02.md",
    "reports/medai_v2_roadmap_park_02/medai_v2_roadmap_park_02_report.json",
    "reports/medai_v2_roadmap_park_02/medai_v2_roadmap_park_02_report.md",
    "scripts/run_medai_v2_roadmap_park_02.py",
    "tests/test_medai_v2_roadmap_park_02.py",
}
PRIOR_V2_REPORT_DIRS = (
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
)


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_roadmap_park_02_report.json").read_text(encoding="utf-8"))


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_json_required_booleans_have_expected_values():
    payload = _payload()
    false_keys = (
        "implementation_started",
        "new_helper_created",
        "direct_implementation_recommended",
        "runtime_behavior_changed",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
        "launcher_changed",
        "installer_changed",
        "deployment_script_changed",
        "startup_config_changed",
        "extraction_changed",
        "ocr_changed",
        "ocr_routing_changed",
        "classifier_changed",
        "threshold_scoring_changed",
        "parser_behavior_changed",
        "fallback_behavior_changed",
        "cue_pack_changed",
        "db_schema_changed",
        "migration_created",
        "migration_executed",
        "persistence_code_changed",
        "clinical_behavior_changed",
        "ddi_behavior_changed",
        "terminology_behavior_changed",
        "private_adapter_implemented",
        "concrete_adapters_implemented",
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
        "licensed_rows_exposed",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "tags_touched",
        "tags_created",
        "tags_modified",
        "cue_expansion_recommended",
        "auto_accept_allowed_default",
    )
    true_keys = (
        "reports_only",
        "parking_snapshot_created",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "preceding_data_infra_spec_present",
        "preceding_extraction_spec_present",
        "preceding_roadmap_02_present",
        "preceding_implementation_readiness_present",
        "preceding_packaging_spec_present",
        "preceding_roadmap_park_01_present",
        "preceding_default_off_implementation_plan_present",
        "preceding_status_registry_present",
        "preceding_roadmap_03_present",
        "status_registry_created_previously",
        "status_registry_standard_library_only",
        "status_registry_import_side_effect_free",
        "terminology_private_adapter_blocked",
        "cue_expansion_blocked",
        "clinical_decision_logic_expansion_blocked",
        "current_v2_state_parked",
        "post_park_options_defined",
        "parking_inventory_created",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
    )
    for key in false_keys:
        assert payload.get(key) is False, key
    for key in true_keys:
        assert payload.get(key) is True, key


def test_prior_v2_report_folders_and_status_registry_exist():
    for name in PRIOR_V2_REPORT_DIRS:
        assert (REPO_ROOT / "reports" / name).is_dir(), name
    assert (REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py").is_file()
    assert (REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "__init__.py").is_file()


def test_report_summarizes_completed_v2_chain_and_status_registry():
    payload = _payload()
    text = _report_text()
    assert len(payload["parked_v2_chain"]) == 14
    for block in (
        "MEDAI-V2-ARCHITECTURE-SPEC-01",
        "MEDAI-V2-FOUNDATION-SPEC-02",
        "MEDAI-V2-RUNTIME-CONTRACTS-01",
        "MEDAI-V2-VALIDATION-HARNESS-01",
        "MEDAI-V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01",
        "MEDAI-V2-ROADMAP-03",
    ):
        assert block in payload["parked_v2_chain"] or block in text
    assert payload["status_registry_entry_count"] == 13
    assert payload["status_registry_default_enabled_count"] == 0
    assert payload["status_registry_runtime_wired_count"] == 0
    assert payload["status_registry_ui_wired_count"] == 0
    assert payload["status_registry_blocked_entry_count"] == 3
    assert "Status Registry Posture Summary" in text


def test_roadmap_03_decision_parking_inventory_and_post_park_options():
    payload = _payload()
    text = _report_text()
    assert payload["roadmap_03_decision_carried_forward"]["selected_next_posture"] == "V2-ROADMAP-PARK-02"
    assert payload["post_park_options"] == [
        "RELEASE-FREEZE-SNAPSHOT",
        "FREEZE-MAINTENANCE-ONLY",
        "V2-FOUNDATION-NEXT-DEFAULT-OFF-PLAN-01",
        "V2-VALIDATION-REGISTRY-INTEGRATION-PLAN-01",
    ]
    assert payload["parking_inventory"]["existing_parked_frozen_anchors"]["v1_frozen_release"] == "7ef8ffd"
    assert payload["parking_inventory"]["current_v2_chain"]["roadmap_03"] == "65e92f3"
    assert "Parking Inventory" in text
    assert "Post-Park Options" in text


def test_recommended_next_sequence_and_no_direct_implementation():
    payload = _payload()
    assert payload["recommended_next_block"] == "RELEASE-FREEZE-SNAPSHOT_OR_FREEZE-MAINTENANCE-ONLY"
    assert payload["recommended_next_3_blocks"] == [
        "RELEASE-FREEZE-SNAPSHOT_OR_FREEZE-MAINTENANCE-ONLY",
        "V2-FOUNDATION-NEXT-DEFAULT-OFF-PLAN-01_OR_V2-VALIDATION-REGISTRY-INTEGRATION-PLAN-01",
        "V2-ROADMAP-04",
    ]
    assert payload["direct_implementation_recommended"] is False
    assert payload["implementation_started"] is False


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
    ):
        assert item in payload["blocked_work"]
    assert "CUE-EXPANSION" in payload["deferred_or_not_recommended"]
    assert "Cue expansion remains explicitly not recommended" in text
    assert payload["v1_release_preserved"] is True


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, path.name


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
        assert "MEDAI-V2-ROADMAP-PARK-02" not in path.read_text(encoding="utf-8", errors="replace")

