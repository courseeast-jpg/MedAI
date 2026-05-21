"""Focused tests for MEDAI-V2-FOUNDATION-IMPLEMENTATION-READINESS-01."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_implementation_readiness_01"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_FOUNDATION_IMPLEMENTATION_READINESS_01.md",
    REPORT_DIR / "medai_v2_foundation_implementation_readiness_01_report.json",
    REPORT_DIR / "medai_v2_foundation_implementation_readiness_01_report.md",
)
ALLOWED_OUTCOMES = {
    "ready_for_default_off_implementation_planning",
    "conditionally_ready_after_packaging_spec",
    "not_ready_continue_specs",
    "freeze_maintenance_only",
    "blocked",
}


def _payload() -> dict:
    return json.loads(
        (REPORT_DIR / "medai_v2_foundation_implementation_readiness_01_report.json").read_text(encoding="utf-8")
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
        "cue_expansion_recommended",
    ):
        assert payload.get(key) is False, key
    for key in (
        "reports_only",
        "implementation_readiness_audit_created",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "preceding_data_infra_spec_present",
        "preceding_extraction_spec_present",
        "preceding_roadmap_02_present",
        "readiness_gate_matrix_created",
        "readiness_scoring_model_created",
        "future_implementation_candidates_ranked",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, key


def test_report_summarizes_completed_v2_planning_blocks():
    text = _report_text()
    for phrase in (
        "Architecture direction",
        "Foundation doctrine",
        "Runtime contracts",
        "Validation harness",
        "UI shell",
        "Data infra",
        "Extraction spec",
        "ROADMAP-02",
    ):
        assert phrase in text


def test_readiness_gate_matrix_and_scoring_model_present():
    payload = _payload()
    text = _report_text()
    assert "Readiness Gate Matrix" in text
    assert "Readiness Scoring Model" in text
    for key in (
        "safety_gates",
        "privacy_gates",
        "runtime_isolation_gates",
        "extraction_ocr_gates",
        "data_persistence_gates",
        "terminology_gates",
        "validation_gates",
        "release_hygiene_gates",
    ):
        assert key in payload["readiness_gate_matrix"]
    assert payload["readiness_score_summary"]["spec_completeness"] == "ready"


def test_allowed_readiness_outcome_and_next_block():
    payload = _payload()
    assert payload["overall_readiness_status"] in ALLOWED_OUTCOMES
    assert payload["overall_readiness_status"] == "conditionally_ready_after_packaging_spec"
    assert payload["recommended_next_block"] == "V2-PACKAGING-SPEC-01"
    assert payload["recommended_next_3_blocks"] == [
        "V2-PACKAGING-SPEC-01",
        "V2-ROADMAP-PARK-01",
        "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY",
    ]
    assert payload["direct_implementation_recommended"] is False


def test_future_implementation_candidates_ranked_and_safest_selected():
    payload = _payload()
    assert len(payload["future_implementation_candidate_rankings"]) == 7
    assert payload["safest_future_implementation_candidate"] == "V2 foundation default-off status registry"
    assert payload["future_implementation_candidate_rankings"][0]["candidate"] == (
        "V2 foundation default-off status registry"
    )


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


def test_runtime_files_not_modified_by_block():
    for path in (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    ):
        assert "MEDAI-V2-FOUNDATION-IMPLEMENTATION-READINESS-01" not in path.read_text(
            encoding="utf-8", errors="replace"
        )

