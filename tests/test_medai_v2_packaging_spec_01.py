"""Focused tests for MEDAI-V2-PACKAGING-SPEC-01."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_packaging_spec_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_PACKAGING_SPEC_01.md",
    REPORT_DIR / "medai_v2_packaging_spec_01_report.json",
    REPORT_DIR / "medai_v2_packaging_spec_01_report.md",
)


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_packaging_spec_01_report.json").read_text(encoding="utf-8"))


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
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_json_required_booleans():
    payload = _payload()
    for key in (
        "launcher_changed",
        "installer_changed",
        "deployment_script_changed",
        "startup_config_changed",
        "runtime_behavior_changed",
        "implementation_started",
        "default_off_helper_created",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
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
        "cue_expansion_recommended",
    ):
        assert payload.get(key) is False, key
    for key in (
        "reports_only",
        "packaging_spec_created",
        "v1_release_preserved",
        "local_only_default",
        "review_bound_default",
        "external_api_blocked_default",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "preceding_data_infra_spec_present",
        "preceding_extraction_spec_present",
        "preceding_roadmap_02_present",
        "preceding_implementation_readiness_present",
        "local_operator_packaging_boundary_defined",
        "launcher_boundary_defined",
        "startup_preflight_boundary_defined",
        "validation_receipt_boundary_defined",
        "release_artifact_boundary_defined",
        "environment_boundary_defined",
        "operator_handoff_boundary_defined",
        "parking_freeze_boundary_defined",
        "future_packaging_implementation_gates_defined",
        "v1_validation_healthcheck_catalog_preserved",
    ):
        assert payload.get(key) is True, key
    assert payload["auto_accept_allowed_default"] is False


def test_required_packaging_sections_present():
    text = _report_text()
    for phrase in (
        "Local Operator Packaging Boundary",
        "Launcher Boundary",
        "Startup / Preflight Boundary",
        "Validation Receipt Boundary",
        "Release Artifact Boundary",
        "Environment Boundary",
        "Operator Handoff Boundary",
        "Parking / Freeze Boundary",
        "Future Packaging Implementation Gates",
    ):
        assert phrase in text


def test_no_packaging_implementation_started():
    payload = _payload()
    text = _report_text()
    assert payload["implementation_started"] is False
    assert payload["launcher_changed"] is False
    assert payload["installer_changed"] is False
    assert payload["deployment_script_changed"] is False
    assert "No packaging implementation begins" in text
    assert "Launchers remain unchanged" in text


def test_v1_healthcheck_and_readiness_carried_forward():
    payload = _payload()
    assert payload["v1_five_validation_healthcheck_set"] == [
        "final_cka_mvp_validation",
        "b07_term01_validation",
        "route_fix_validation",
        "ui_ops_validation",
        "ui_boot_validation",
    ]
    assert payload["readiness_outcome_carried_forward"] == "conditionally_ready_after_packaging_spec"
    assert payload["safest_future_implementation_candidate_carried_forward"] == (
        "V2 foundation default-off status registry"
    )
    assert payload["recommended_next_block"] == "V2-ROADMAP-PARK-01"
    assert payload["recommended_next_3_blocks"] == [
        "V2-ROADMAP-PARK-01",
        "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY",
        "V2-ROADMAP-03",
    ]


def test_defaults_v1_release_and_cue_expansion_preserved():
    payload = _payload()
    text = _report_text()
    assert payload["local_only_default"] is True
    assert payload["review_bound_default"] is True
    assert payload["external_api_blocked_default"] is True
    assert payload["auto_accept_allowed_default"] is False
    assert payload["v1_release_preserved"] is True
    assert payload["cue_expansion_recommended"] is False
    assert "Cue expansion remains explicitly not recommended" in text


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, path.name


def test_runtime_and_launcher_files_not_modified_by_block():
    files = (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "startup_preflight.py",
        REPO_ROOT / "app" / "config.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    )
    for path in files:
        assert "MEDAI-V2-PACKAGING-SPEC-01" not in path.read_text(encoding="utf-8", errors="replace")

