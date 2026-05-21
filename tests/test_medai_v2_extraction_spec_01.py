"""Focused tests for MEDAI-V2-EXTRACTION-SPEC-01."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_extraction_spec_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_EXTRACTION_SPEC_01.md",
    REPORT_DIR / "medai_v2_extraction_spec_01_report.json",
    REPORT_DIR / "medai_v2_extraction_spec_01_report.md",
)


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_extraction_spec_01_report.json").read_text(encoding="utf-8"))


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
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_json_required_booleans():
    payload = _payload()
    for key in (
        "extraction_behavior_changed",
        "ocr_behavior_changed",
        "ocr_routing_changed",
        "classifier_changed",
        "threshold_scoring_changed",
        "cue_pack_changed",
        "parser_behavior_changed",
        "fallback_behavior_changed",
        "runtime_behavior_changed",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
        "db_schema_changed",
        "migration_created",
        "persistence_code_changed",
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
        "extraction_spec_created",
        "review_bound_default",
        "local_only_default",
        "external_api_blocked_default",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "preceding_data_infra_spec_present",
        "source_intake_boundary_defined",
        "text_visibility_boundary_defined",
        "ocr_routing_boundary_defined",
        "extraction_adapter_boundary_defined",
        "confidence_fallback_isolation_defined",
        "structured_parser_boundary_defined",
        "multilingual_script_boundary_defined",
        "review_queue_boundary_defined",
        "observability_audit_boundary_defined",
        "rollback_parking_boundary_defined",
        "future_extraction_implementation_gates_defined",
        "v1_validation_healthcheck_catalog_preserved",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, key


def test_required_report_sections_present():
    text = _report_text()
    for phrase in (
        "Source Intake Boundary",
        "Text Visibility Boundary",
        "OCR Routing Boundary",
        "Extraction Adapter Boundary",
        "Confidence And Fallback Isolation",
        "Structured Parser Boundary",
        "Multilingual And Script Boundary",
        "Review Queue Boundary",
        "Observability And Audit Boundary",
        "Rollback And Parking Boundary",
        "Future Implementation Gates",
    ):
        assert phrase in text


def test_no_implementation_started_and_defaults_preserved():
    payload = _payload()
    assert payload["next_recommended_block"] == "V2-ROADMAP-02"
    assert payload["auto_accept_allowed_default"] is False
    assert payload["review_bound_default"] is True
    assert payload["local_only_default"] is True
    assert payload["external_api_blocked_default"] is True
    assert payload["v1_release_preserved"] is True
    assert payload["cue_expansion_recommended"] is False
    assert len(payload["future_implementation_gates"]) == 10
    assert payload["v1_five_validation_healthcheck_set"] == [
        "final_cka_mvp_validation",
        "b07_term01_validation",
        "route_fix_validation",
        "ui_ops_validation",
        "ui_boot_validation",
    ]


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, path.name


def test_runtime_files_not_modified_by_block():
    for path in (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    ):
        assert "MEDAI-V2-EXTRACTION-SPEC-01" not in path.read_text(encoding="utf-8", errors="replace")

