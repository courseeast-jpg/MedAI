"""Focused tests for MEDAI-V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_default_off_implementation_plan_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_FOUNDATION_DEFAULT_OFF_IMPLEMENTATION_PLAN_01.md",
    REPORT_DIR / "medai_v2_foundation_default_off_implementation_plan_01_report.json",
    REPORT_DIR / "medai_v2_foundation_default_off_implementation_plan_01_report.md",
)


def _payload() -> dict:
    return json.loads(
        (REPORT_DIR / "medai_v2_foundation_default_off_implementation_plan_01_report.json").read_text(
            encoding="utf-8"
        )
    )


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
        "medai_v2_roadmap_park_01",
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_json_required_booleans():
    payload = _payload()
    for key in (
        "implementation_started",
        "default_off_helper_created",
        "future_target_runtime_wiring_allowed",
        "future_target_ui_wiring_allowed",
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
    ):
        assert payload.get(key) is False, key
    for key in (
        "reports_only",
        "implementation_plan_created",
        "future_target_allowed",
        "future_target_default_off_required",
        "future_target_standard_library_only_required",
        "future_target_import_side_effect_free_required",
        "future_file_boundaries_defined",
        "future_registry_contract_defined",
        "future_registry_inventory_defined",
        "future_implementation_gates_defined",
        "future_rollback_stop_rules_defined",
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
        "preceding_roadmap_park_present",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
    ):
        assert payload.get(key) is True, key
    assert payload["auto_accept_allowed_default"] is False


def test_future_target_and_carried_forward_decision():
    payload = _payload()
    text = _report_text()
    assert payload["future_target"] == "V2 foundation default-off status registry"
    assert payload["readiness_outcome_carried_forward"] == "conditionally_ready_after_packaging_spec"
    assert payload["safest_future_implementation_candidate_carried_forward"] == (
        "V2 foundation default-off status registry"
    )
    assert "The default-off status registry is not created" in text


def test_future_file_boundaries_contract_inventory_gates_and_stop_rules():
    payload = _payload()
    text = _report_text()
    assert "clinical_knowledge/v2_foundation/status_registry.py" in payload["allowed_future_implementation_files"]
    assert "app/main.py" in payload["disallowed_future_implementation_file_families"]
    assert "capability_id" in payload["future_registry_entry_fields"]
    assert "foundation" in payload["future_registry_required_categories"]
    assert "planned_default_off" in payload["future_registry_required_statuses"]
    assert len(payload["future_registry_inventory"]) == 13
    assert len(payload["future_implementation_gates"]) == 15
    assert len(payload["future_rollback_stop_rules"]) == 10
    for phrase in (
        "Future File Boundaries",
        "Future Status Registry Contract",
        "Future Registry Inventory",
        "Future Implementation Gates",
        "Future Rollback / Stop Rules",
    ):
        assert phrase in text


def test_recommended_next_block_and_sequence():
    payload = _payload()
    assert payload["recommended_next_block"] == "V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01"
    assert payload["recommended_next_3_blocks"] == [
        "V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01",
        "V2-ROADMAP-03",
        "V2-ROADMAP-PARK-02_OR_RELEASE-FREEZE-SNAPSHOT",
    ]


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
        "direct_v2_implementation",
        "cue_expansion",
    ):
        assert item in payload["blocked_or_deferred_tracks"]
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
    for path in (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "startup_preflight.py",
        REPO_ROOT / "app" / "config.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    ):
        assert "MEDAI-V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01" not in path.read_text(
            encoding="utf-8", errors="replace"
        )

