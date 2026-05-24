"""Focused tests for MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_pathology_report_family_spec_01"
REPORTS = (
    REPORT_DIR / "MEDAI_PATHOLOGY_REPORT_FAMILY_SPEC_01.md",
    REPORT_DIR / "medai_pathology_report_family_spec_01_report.json",
    REPORT_DIR / "medai_pathology_report_family_spec_01_report.md",
)
ALLOWED_CHANGED_PATHS = {
    "reports/medai_pathology_report_family_spec_01/MEDAI_PATHOLOGY_REPORT_FAMILY_SPEC_01.md",
    "reports/medai_pathology_report_family_spec_01/medai_pathology_report_family_spec_01_report.json",
    "reports/medai_pathology_report_family_spec_01/medai_pathology_report_family_spec_01_report.md",
    "scripts/run_medai_pathology_report_family_spec_01.py",
    "tests/test_medai_pathology_report_family_spec_01.py",
}


def _payload() -> dict:
    return json.loads(
        (
            REPORT_DIR / "medai_pathology_report_family_spec_01_report.json"
        ).read_text(encoding="utf-8")
    )


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_block_mode_and_spec_creation():
    payload = _payload()
    assert payload["block_id"] == "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01"
    assert payload["block_mode"] == "reports_only_synthetic_only_family_spec"
    assert payload["mode"] == "reports_only_synthetic_only_family_spec"
    assert payload["reports_only"] is True
    assert payload["synthetic_only"] is True
    assert payload["spec_only"] is True
    assert payload["pathology_family_spec_created"] is True
    assert payload["pathology_family_support_justified"] is True


def test_family_name_is_pathology_report_and_subtype_field_correct():
    payload = _payload()
    assert payload["family_name"] == "pathology_report"
    assert payload["subtype_field"] == "pathology_subtype"


def test_subtype_vocabulary_has_six_entries_including_required_values():
    payload = _payload()
    vocab = payload["subtype_vocabulary"]
    assert isinstance(vocab, list)
    assert len(vocab) == 6
    assert payload["subtype_vocabulary_count"] == 6
    for required in (
        "dermatopathology",
        "biopsy",
        "cytology",
        "frozen_section",
        "consultation_pathology",
        "unspecified_pathology",
    ):
        assert required in vocab, required


def test_cue_candidates_are_eleven_planning_level_only():
    payload = _payload()
    core = payload["core_cue_candidates"]
    extra = payload["additional_cue_candidates"]
    assert isinstance(core, list) and len(core) == 3
    assert isinstance(extra, list) and len(extra) == 8
    assert payload["total_cue_candidate_count"] == 11
    for required in (
        "specimen_section",
        "microscopic_description_section",
        "pathology_conclusion_section",
    ):
        assert required in core, required
    for required in (
        "gross_description",
        "clinical_diagnosis",
        "signs_and_symptoms",
        "reason_for_biopsy",
        "photomicrograph_section",
        "margin_or_edge_or_inked_edge_comment",
        "icd_code",
        "pathology_consultation_header",
    ):
        assert required in extra, required
    assert payload["cue_candidates_implemented_in_classifier"] is False
    assert payload["cue_expansion_performed"] is False
    assert payload["cue_expansion_recommended"] is False


def test_extraction_field_contract_has_eighteen_review_bound_fields():
    payload = _payload()
    fields = payload["future_extraction_field_contract"]
    assert isinstance(fields, list)
    assert len(fields) == 18
    assert payload["future_extraction_field_count"] == 18
    names = [entry["field"] for entry in fields]
    for required in (
        "document_family",
        "pathology_subtype",
        "specialty_domain",
        "specimen",
        "anatomical_site",
        "clinical_diagnosis_or_indication",
        "signs_and_symptoms",
        "gross_description",
        "microscopic_description",
        "final_diagnosis",
        "diagnosis_code_family",
        "diagnosis_code_value",
        "margin_or_edge_comment",
        "photomicrograph_present",
        "source_document_reference",
        "extraction_confidence",
        "review_required",
        "auto_accept_allowed",
    ):
        assert required in names, required
    for entry in fields:
        assert entry["review_required"] is True, entry["field"]
        assert entry["auto_accept_allowed"] is False, entry["field"]
        assert entry["source_facts_only"] is True, entry["field"]
        assert entry["ai_interpretation_allowed_in_classifier_layer"] is False, entry["field"]


def test_review_required_default_true_and_auto_accept_default_false():
    payload = _payload()
    assert payload["review_required_default"] is True
    assert payload["auto_accept_allowed_default"] is False
    assert payload["source_facts_only"] is True
    assert payload["ai_interpretation_allowed_in_classifier_layer"] is False
    assert (
        payload["ai_interpretation_tier"]
        == "hypothesis_or_comment_only_in_future_separate_agent_block"
    )


def test_dermatology_mkb_placement_doctrine_has_six_rules():
    payload = _payload()
    doctrine = payload["dermatology_mkb_placement_doctrine"]
    assert isinstance(doctrine, dict)
    assert isinstance(doctrine["rules"], list)
    assert len(doctrine["rules"]) == 6
    assert doctrine["rule_count"] == 6


def test_ai_agent_interpretation_boundary_has_seven_rules_and_not_implemented():
    payload = _payload()
    boundary = payload["ai_agent_interpretation_boundary"]
    assert isinstance(boundary, dict)
    assert isinstance(boundary["rules"], list)
    assert len(boundary["rules"]) == 7
    assert boundary["rule_count"] == 7
    assert boundary["ai_interpretation_implemented"] is False
    assert payload["ai_interpretation_implemented"] is False


def test_photomicrograph_boundary_has_four_options_and_not_implemented():
    payload = _payload()
    photo = payload["photomicrograph_boundary"]
    assert isinstance(photo, dict)
    assert isinstance(photo["options"], list)
    assert len(photo["options"]) == 4
    assert photo["option_count"] == 4
    assert photo["photomicrograph_interpretation_implemented"] is False
    assert payload["photomicrograph_interpretation_implemented"] is False


def test_future_implementation_gates_count_is_eleven():
    payload = _payload()
    gates = payload["future_implementation_gates"]
    assert isinstance(gates, list)
    assert len(gates) == 11
    assert payload["future_implementation_gate_count"] == 11


def test_no_classifier_or_cue_or_threshold_or_runtime_change():
    payload = _payload()
    for key in (
        "behavior_changed_in_this_block",
        "classifier_changed",
        "classifier_cue_added",
        "classifier_rule_changed",
        "cue_expansion_performed",
        "cue_expansion_recommended",
        "threshold_changed",
        "threshold_scoring_changed",
        "ocr_routing_changed",
        "parser_changed",
        "parser_behavior_changed",
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
        "implementation_started",
        "new_helper_created",
        "direct_implementation_recommended",
        "ai_interpretation_implemented",
        "photomicrograph_interpretation_implemented",
        "tags_touched",
        "tags_created",
        "tags_modified",
    ):
        assert payload.get(key) is False, key


def test_no_real_document_or_private_data_committed_or_inspected():
    payload = _payload()
    for key in (
        "real_pdf_committed",
        "real_screenshot_committed",
        "real_diagnosis_printed",
        "raw_text_read",
        "raw_text_printed",
        "raw_ocr_text_read",
        "raw_ocr_text_printed",
        "raw_filenames_read",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "phi_printed",
        "phi_inspected",
        "licensed_rows_read",
        "licensed_rows_exposed",
        "license_ack_read",
        "private_license_ack_read",
        "private_config_read",
        "runtime_db_accessed",
        "private_data_accessed",
        "source_documents_opened",
    ):
        assert payload.get(key) is False, key


def test_freeze_maintenance_posture_intact_and_anchors_match():
    payload = _payload()
    assert payload["freeze_maintenance_posture_intact"] is True
    assert payload["v1_release_preserved"] is True
    assert payload["preceding_release_freeze_snapshot_present"] is True
    assert payload["preceding_freeze_maintenance_only_present"] is True
    assert payload["preceding_real_doc_unknown_triage_present"] is True
    assert payload["preceding_b4_pathology_eval_present"] is True
    anchors = payload["freeze_maintenance_anchor"]
    assert anchors["v1_frozen_release"] == "7ef8ffd"
    assert anchors["freeze_maintenance_only_01_commit"] == "fb730f2"
    assert anchors["real_doc_unknown_triage_01_commit"] == "324682c"
    assert anchors["b4_pathology_eval_01_commit"] == "12fa16a"


def test_recommended_next_block_is_synthetic_coverage_or_freeze_maintenance():
    payload = _payload()
    assert (
        payload["recommended_next_block"]
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01_OR_FREEZE-MAINTENANCE-ONLY"
    )
    next3 = payload["recommended_next_3_blocks"]
    assert isinstance(next3, list)
    assert len(next3) == 3
    assert next3[0] == payload["recommended_next_block"]


def test_long_report_documents_all_sections():
    text = _report_text()
    for phrase in (
        "Scope And Non-Scope",
        "Prior B4 Evaluation Summary",
        "Controlled Family Naming",
        "Pathology Subtype Vocabulary",
        "Safe Structural Cue Groups",
        "Future Review-Bound Extraction Field Contract",
        "Dermatology MKB Placement Doctrine",
        "AI-Agent Interpretation Boundary",
        "Photomicrograph Boundary",
        "Safety / Privacy Invariants",
        "Future Implementation Gates",
        "Validation Matrix",
        "Recommended Next Block",
    ):
        assert phrase in text, phrase


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        assert check_public_report_payload(target).passed, path.name


def test_runtime_classifier_registry_files_do_not_mention_this_block():
    for relative in (
        "app/main.py",
        "app/startup_preflight.py",
        "app/config.py",
        "app/document_type_registry.py",
        "app/lab_document_metadata.py",
        "document_classification/document_classifier.py",
        "clinical_knowledge/v2_foundation/status_registry.py",
        "clinical_knowledge/v2_contracts/runtime_contracts.py",
        "clinical_knowledge/v2_contracts/validation_harness.py",
    ):
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        assert "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01" not in body, relative
        assert "medai_pathology_report_family_spec_01" not in body, relative


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
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_medai_pathology_report_family_spec_01.py"),
        ],
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
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01_OR_FREEZE-MAINTENANCE-ONLY"
    )
