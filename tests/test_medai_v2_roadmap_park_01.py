"""Focused tests for MEDAI-V2-ROADMAP-PARK-01."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_roadmap_park_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_ROADMAP_PARK_01.md",
    REPORT_DIR / "medai_v2_roadmap_park_01_report.json",
    REPORT_DIR / "medai_v2_roadmap_park_01_report.md",
)


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_roadmap_park_01_report.json").read_text(encoding="utf-8"))


def _report_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_prior_v2_report_folders_exist():
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
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_json_required_booleans():
    payload = _payload()
    for key in (
        "implementation_started",
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
        "default_off_helper_created",
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
    ):
        assert payload.get(key) is False, key
    for key in (
        "reports_only",
        "parking_snapshot_created",
        "v2_planning_sequence_parked",
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
        "post_park_options_defined",
        "parking_inventory_created",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
    ):
        assert payload.get(key) is True, key
    assert payload["auto_accept_allowed_default"] is False


def test_report_summarizes_all_completed_v2_planning_blocks():
    payload = _payload()
    text = _report_text()
    assert len(payload["parked_v2_planning_chain"]) == 10
    for block in payload["parked_v2_planning_chain"]:
        assert block in text


def test_readiness_candidate_inventory_and_options_carried_forward():
    payload = _payload()
    text = _report_text()
    assert payload["readiness_outcome_carried_forward"] == "conditionally_ready_after_packaging_spec"
    assert payload["safest_future_implementation_candidate_carried_forward"] == (
        "V2 foundation default-off status registry"
    )
    assert "Parking Inventory" in text
    assert payload["parking_inventory"]["existing_parked_frozen_anchors"]["v1_frozen_release"] == "7ef8ffd"
    assert payload["parking_inventory"]["current_v2_planning_chain"]["packaging_spec"] == "37d056a"
    assert payload["post_park_options"] == [
        "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01",
        "FREEZE-MAINTENANCE-ONLY",
        "V2-ROADMAP-03",
    ]


def test_no_direct_implementation_and_next_sequence():
    payload = _payload()
    text = _report_text()
    assert payload["direct_implementation_recommended"] is False
    assert payload["implementation_started"] is False
    assert payload["recommended_next_block"] == (
        "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY"
    )
    assert payload["recommended_next_3_blocks"] == [
        "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY",
        "V2-ROADMAP-03",
        "V2-ROADMAP-PARK-02_OR_RELEASE-FREEZE-SNAPSHOT",
    ]
    assert "Direct V2 implementation remains out of scope" in text


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
        "direct_v2_implementation",
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


def test_runtime_files_and_tags_not_modified_by_block():
    payload = _payload()
    assert payload["tags_created"] is False
    assert payload["tags_modified"] is False
    files = (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "startup_preflight.py",
        REPO_ROOT / "app" / "config.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    )
    for path in files:
        assert "MEDAI-V2-ROADMAP-PARK-01" not in path.read_text(encoding="utf-8", errors="replace")
