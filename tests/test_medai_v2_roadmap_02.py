"""Focused tests for MEDAI-V2-ROADMAP-02."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_roadmap_02"
REPORTS = (
    REPORT_DIR / "MEDAI_V2_ROADMAP_02.md",
    REPORT_DIR / "medai_v2_roadmap_02_report.json",
    REPORT_DIR / "medai_v2_roadmap_02_report.md",
)


def _payload() -> dict:
    return json.loads((REPORT_DIR / "medai_v2_roadmap_02_report.json").read_text(encoding="utf-8"))


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
    ):
        assert (REPO_ROOT / "reports" / name).is_dir(), name


def test_json_required_booleans():
    payload = _payload()
    for key in (
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
        "roadmap_audit_created",
        "prior_v2_sequence_complete",
        "preceding_architecture_spec_present",
        "preceding_foundation_spec_present",
        "preceding_runtime_contracts_present",
        "preceding_validation_harness_present",
        "preceding_ui_shell_spec_present",
        "preceding_data_infra_spec_present",
        "preceding_extraction_spec_present",
        "remaining_workstreams_ranked",
        "next_posture_selected",
        "v1_release_preserved",
    ):
        assert payload.get(key) is True, key


def test_report_summarizes_all_seven_v2_planning_blocks():
    payload = _payload()
    text = _report_text()
    blocks = (
        "MEDAI-V2-ARCHITECTURE-SPEC-01",
        "MEDAI-V2-FOUNDATION-SPEC-02",
        "MEDAI-V2-RUNTIME-CONTRACTS-01",
        "MEDAI-V2-VALIDATION-HARNESS-01",
        "MEDAI-V2-UI-SHELL-SPEC-01",
        "MEDAI-V2-DATA-INFRA-SPEC-01",
        "MEDAI-V2-EXTRACTION-SPEC-01",
    )
    assert payload["completed_v2_planning_sequence"] == list(blocks)
    for block in blocks:
        assert block in text


def test_remaining_workstreams_ranked_and_next_posture_selected():
    payload = _payload()
    text = _report_text()
    assert len(payload["remaining_workstream_rankings"]) >= 10
    assert payload["recommended_next_block"] == "V2-FOUNDATION-IMPLEMENTATION-READINESS-01"
    assert payload["recommended_next_3_blocks"] == [
        "V2-FOUNDATION-IMPLEMENTATION-READINESS-01",
        "V2-PACKAGING-SPEC-01",
        "V2-ROADMAP-PARK-01",
    ]
    assert "direct V2 implementation" in text
    assert "not implementation authorization" in text


def test_blocked_deferred_work_remains_blocked():
    payload = _payload()
    text = _report_text()
    for item in (
        "private_adapter_implementation",
        "real_private_store_access",
        "licensed_terminology_row_reads",
        "runtime_db_migration",
        "extraction_ocr_behavior_changes",
        "ui_implementation",
    ):
        assert item in payload["blocked_work"]
    assert "CUE-EXPANSION" in payload["deferred_or_not_recommended"]
    assert "Cue expansion remains explicitly not recommended" in text


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
        assert "MEDAI-V2-ROADMAP-02" not in path.read_text(encoding="utf-8", errors="replace")

