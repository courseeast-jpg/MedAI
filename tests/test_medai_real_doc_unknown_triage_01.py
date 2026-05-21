"""Focused tests for MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_unknown_triage_01"
REPORTS = (
    REPORT_DIR / "MEDAI_REAL_DOC_UNKNOWN_TRIAGE_01.md",
    REPORT_DIR / "medai_real_doc_unknown_triage_01_report.json",
    REPORT_DIR / "medai_real_doc_unknown_triage_01_report.md",
)
ALLOWED_CHANGED_PATHS = {
    "reports/medai_real_doc_unknown_triage_01/MEDAI_REAL_DOC_UNKNOWN_TRIAGE_01.md",
    "reports/medai_real_doc_unknown_triage_01/medai_real_doc_unknown_triage_01_report.json",
    "reports/medai_real_doc_unknown_triage_01/medai_real_doc_unknown_triage_01_report.md",
    "scripts/run_medai_real_doc_unknown_triage_01.py",
    "tests/test_medai_real_doc_unknown_triage_01.py",
}


def _payload() -> dict:
    return json.loads(
        (REPORT_DIR / "medai_real_doc_unknown_triage_01_report.json").read_text(
            encoding="utf-8"
        )
    )


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_block_mode_and_triggering_signal():
    payload = _payload()
    assert payload["block_mode"] == "diagnosis_only_real_doc_unknown_triage"
    assert payload["mode"] == "diagnosis_only_real_doc_unknown_triage"
    assert payload["reports_only"] is True
    assert payload["triggered_by_real_operator_signal"] is True
    assert payload["operator_supplied_real_document_artifact"] is False
    assert payload["real_document_inspected_by_assistant"] is False


def test_no_private_data_inspected_or_committed():
    payload = _payload()
    for key in (
        "raw_text_inspected",
        "raw_ocr_text_inspected",
        "raw_filename_inspected",
        "private_paths_inspected",
        "phi_inspected",
        "real_pdf_committed",
        "raw_text_read",
        "raw_text_printed",
        "raw_ocr_text_read",
        "raw_ocr_text_printed",
        "raw_filenames_read",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "licensed_rows_read",
        "licensed_rows_exposed",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "source_documents_opened",
        "private_data_accessed",
        "external_api_used",
    ):
        assert payload.get(key) is False, key


def test_no_runtime_or_classifier_or_ocr_change():
    payload = _payload()
    for key in (
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
        "cue_expansion_recommended",
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
        "implementation_started",
        "new_helper_created",
        "direct_implementation_recommended",
        "code_fix_justified_yet",
        "smallest_next_block_committed_to",
        "tags_touched",
        "tags_created",
        "tags_modified",
        "auto_accept_allowed_default",
    ):
        assert payload.get(key) is False, key


def test_freeze_maintenance_posture_intact():
    payload = _payload()
    assert payload["freeze_maintenance_posture_intact"] is True
    assert payload["v1_release_preserved"] is True
    assert payload["preceding_release_freeze_snapshot_present"] is True
    assert payload["preceding_freeze_maintenance_only_present"] is True
    anchors = payload["freeze_maintenance_anchor"]
    assert anchors["v1_frozen_release"] == "7ef8ffd"
    assert anchors["freeze_maintenance_only_01_commit"] == "fb730f2"


def test_diagnostic_field_inventory_is_twenty():
    payload = _payload()
    fields = payload["advanced_diagnostic_fields_used_by_ui"]
    assert isinstance(fields, list)
    assert len(fields) == 20
    assert payload["advanced_diagnostic_field_count"] == 20
    for required in (
        "document_type",
        "confidence",
        "validation_status",
        "selected_extractor",
        "ocr_quality_band",
        "language_text_visibility",
        "ocr_gate_reason",
        "ocr_gate_fallback_executed",
        "document_family_classification_diagnostic",
        "operator_review_reason",
    ):
        assert required in fields, required


def test_failure_bucket_taxonomy_is_eight():
    payload = _payload()
    buckets = payload["failure_bucket_taxonomy"]
    assert isinstance(buckets, list)
    assert len(buckets) == 8
    assert payload["failure_bucket_count"] == 8
    ids = [bucket["id"] for bucket in buckets]
    assert ids == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    names = [bucket["name"] for bucket in buckets]
    assert names == [
        "file_not_queued_or_run_not_executed",
        "text_extracted_but_classifier_cues_insufficient",
        "ocr_fallback_needed_but_not_triggered",
        "document_family_unsupported",
        "image_or_table_or_layout_heavy_document",
        "language_or_script_mismatch",
        "parser_unsupported",
        "ui_run_state_confusion",
    ]


def test_decision_tree_has_five_steps():
    payload = _payload()
    tree = payload["decision_tree"]
    assert isinstance(tree, list)
    assert len(tree) == 5
    assert [step["step"] for step in tree] == [1, 2, 3, 4, 5]


def test_bucket_conditional_block_table_is_eight():
    payload = _payload()
    cond = payload["bucket_conditional_smallest_next_block"]
    assert isinstance(cond, list)
    assert len(cond) == 8
    assert payload["bucket_conditional_block_count"] == 8
    ids = [entry["bucket_id"] for entry in cond]
    assert ids == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    for entry in cond:
        assert "smallest_next_block_candidate" in entry
        assert "code_fix_justified" in entry


def test_operator_action_required_steps_are_present():
    payload = _payload()
    steps = payload["operator_action_required_steps"]
    assert isinstance(steps, list)
    assert len(steps) >= 6
    assert payload["operator_action_required"] is True


def test_recommended_next_block_is_operator_action():
    payload = _payload()
    assert (
        payload["recommended_next_block"]
        == "OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY"
    )
    next3 = payload["recommended_next_3_blocks"]
    assert isinstance(next3, list)
    assert len(next3) == 3
    assert next3[0] == payload["recommended_next_block"]


def test_long_report_documents_all_sections():
    text = _report_text()
    for phrase in (
        "Scope And Non-Scope",
        "Trigger And Operator-Observed Signal",
        "Public-Safe Diagnostic Surface",
        "Failure Bucket Taxonomy",
        "Decision Tree",
        "Bucket-Conditional Smallest Next Block",
        "Safety / Privacy Invariants",
        "Operator Action Required",
        "Recommended Next Block",
    ):
        assert phrase in text, phrase


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        assert check_public_report_payload(target).passed, path.name


def test_runtime_and_classifier_and_ocr_files_do_not_mention_this_block():
    for relative in (
        "app/main.py",
        "app/startup_preflight.py",
        "app/config.py",
        "document_classification/document_classifier.py",
        "clinical_knowledge/v2_foundation/status_registry.py",
        "clinical_knowledge/v2_contracts/runtime_contracts.py",
        "clinical_knowledge/v2_contracts/validation_harness.py",
    ):
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        assert "MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01" not in body, relative
        assert "medai_real_doc_unknown_triage_01" not in body, relative


def test_implementation_files_limited_to_reports_script_tests():
    result = subprocess.run(
        ["git", "status", "--short", "-uall"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {
        line[3:].replace("\\", "/")
        for line in result.stdout.splitlines()
        if line and line[:2].strip()
    }
    assert changed <= ALLOWED_CHANGED_PATHS, changed - ALLOWED_CHANGED_PATHS


def test_audit_script_runs_clean():
    result = subprocess.run(
        ["python3", str(REPO_ROOT / "scripts" / "run_medai_real_doc_unknown_triage_01.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout)
    assert out["all_clean"] is True
    assert (
        out["next_recommended_block"]
        == "OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY"
    )
