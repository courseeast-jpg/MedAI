"""Focused tests for MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_b4_pathology_family_eval_01"
REPORTS = (
    REPORT_DIR / "MEDAI_REAL_DOC_B4_PATHOLOGY_FAMILY_EVAL_01.md",
    REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.json",
    REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.md",
)
ALLOWED_CHANGED_PATHS = {
    "reports/medai_real_doc_b4_pathology_family_eval_01/MEDAI_REAL_DOC_B4_PATHOLOGY_FAMILY_EVAL_01.md",
    "reports/medai_real_doc_b4_pathology_family_eval_01/medai_real_doc_b4_pathology_family_eval_01_report.json",
    "reports/medai_real_doc_b4_pathology_family_eval_01/medai_real_doc_b4_pathology_family_eval_01_report.md",
    "scripts/run_medai_real_doc_b4_pathology_family_eval_01.py",
    "tests/test_medai_real_doc_b4_pathology_family_eval_01.py",
}


def _payload() -> dict:
    return json.loads(
        (
            REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.json"
        ).read_text(encoding="utf-8")
    )


def _report_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in REPORTS)


def test_all_three_reports_exist():
    for path in REPORTS:
        assert path.is_file(), path


def test_block_mode_and_triggering_signal():
    payload = _payload()
    assert payload["block_mode"] == "evaluation_only_pathology_family_audit"
    assert payload["mode"] == "evaluation_only_pathology_family_audit"
    assert payload["reports_only"] is True
    assert payload["evaluation_only"] is True
    assert payload["synthetic_only"] is True
    assert payload["triggered_by_real_operator_signal"] is True
    assert payload["matched_unknown_triage_bucket"] == "B4"


def test_no_private_data_inspected_or_committed():
    payload = _payload()
    for key in (
        "real_pdf_committed",
        "real_screenshot_committed",
        "real_diagnosis_printed",
        "real_document_inspected_by_assistant",
        "operator_supplied_real_document_artifact",
        "raw_text_read",
        "raw_text_printed",
        "raw_ocr_text_read",
        "raw_ocr_text_printed",
        "raw_filenames_read",
        "raw_filenames_printed",
        "private_paths_printed",
        "secrets_printed",
        "phi_inspected",
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


def test_no_classifier_or_cue_or_threshold_change():
    payload = _payload()
    for key in (
        "behavior_changed_in_this_block",
        "runtime_behavior_changed",
        "classifier_changed",
        "classifier_cue_added",
        "classifier_rule_changed",
        "threshold_scoring_changed",
        "parser_behavior_changed",
        "fallback_behavior_changed",
        "cue_pack_changed",
        "cue_expansion_recommended",
        "ocr_changed",
        "ocr_routing_changed",
        "extraction_changed",
        "app_main_changed",
        "streamlit_code_changed",
        "ui_changed",
        "launcher_changed",
        "installer_changed",
        "deployment_script_changed",
        "startup_config_changed",
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
    assert payload["preceding_real_doc_unknown_triage_present"] is True
    anchors = payload["freeze_maintenance_anchor"]
    assert anchors["v1_frozen_release"] == "7ef8ffd"
    assert anchors["freeze_maintenance_only_01_commit"] == "fb730f2"
    assert anchors["real_doc_unknown_triage_01_commit"] == "324682c"


def test_core_classifier_findings_match_live_source():
    payload = _payload()
    findings = payload["core_classifier_findings"]
    assert findings["core_classifier_has_pathology_cue"] is False
    assert findings["core_classifier_has_dermatopathology_cue"] is False
    assert findings["core_classifier_has_biopsy_cue"] is False
    assert findings["core_classifier_has_microscopic_cue"] is False
    assert findings["core_classifier_has_specimen_cue_in_strong_tokens"] is False
    assert findings["core_classifier_has_specimen_cue_in_lab_indicator_only"] is True
    outcome = findings["core_classifier_outcome_for_synthetic_pathology_text"]
    assert outcome["document_type"] == "unknown_medical"
    assert outcome["review_reason"] == "unknown_document_type"
    assert outcome["confidence"] == 0.3

    from document_classification.document_classifier import (
        DOCUMENT_TYPES,
        classify_document,
    )

    assert set(DOCUMENT_TYPES) == set(findings["core_classifier_emitted_document_types"])
    result = classify_document(
        "Specimen received. Clinical diagnosis: review pending. "
        "Gross description. Microscopic description. Final diagnosis. "
        "ICD code field."
    )
    assert result.document_type == "unknown_medical"
    assert result.review_reason == "unknown_document_type"
    assert result.confidence == 0.3


def test_metadata_family_registry_findings_match_live_source():
    payload = _payload()
    findings = payload["metadata_family_registry_findings"]
    assert findings["pathology_family_label"] == "Pathology report"
    assert findings["pathology_family_label_constant"] == "PATHOLOGY_REPORT_LABEL"
    assert findings["pathology_family_threshold"] == 2
    assert findings["pathology_family_english_cue_key_count"] == 3
    assert findings["pathology_family_language_pack_count"] == 4
    assert sorted(findings["supported_language_packs"]) == [
        "albanian",
        "english",
        "polish",
        "russian",
    ]

    from app.document_type_registry import (
        DOCUMENT_FAMILY_REGISTRY,
        PATHOLOGY_REPORT_LABEL,
        SUPPORTED_LANGUAGE_PACKS,
        document_family_classification_diagnostic,
    )

    rule = DOCUMENT_FAMILY_REGISTRY[PATHOLOGY_REPORT_LABEL]
    assert rule.threshold == 2
    assert sorted(rule.cue_groups["english"].keys()) == [
        "microscopic_description_section",
        "pathology_conclusion_section",
        "specimen_section",
    ]
    assert sorted(SUPPORTED_LANGUAGE_PACKS) == [
        "albanian",
        "english",
        "polish",
        "russian",
    ]
    diag = document_family_classification_diagnostic(
        "Specimen received. Clinical diagnosis: review pending. "
        "Gross description. Microscopic description. Final diagnosis. "
        "ICD code field."
    )
    assert diag["candidate_family"] == "Pathology report"
    assert diag["review_only"] is True
    assert diag["auto_accept_allowed"] is False


def test_two_layer_split_summary_describes_unknown_vs_pathology():
    payload = _payload()
    split = payload["two_layer_split_summary"]
    assert split["core_classifier_emits"] == "unknown_medical"
    assert split["metadata_family_registry_emits"] == "Pathology report"
    assert split["ui_primary_card_field"] == "document_type"
    assert split["auto_accept_allowed"] is False
    assert split["review_required"] is True


def test_controlled_family_name_recommendation_uses_pathology_report():
    payload = _payload()
    rec = payload["controlled_family_name_recommendation"]
    assert rec["recommended_controlled_family_name"] == "pathology_report"
    assert rec["recommended_subtype_field_name"] == "pathology_subtype"
    vocab = rec["recommended_subtype_controlled_vocabulary"]
    assert isinstance(vocab, list) and len(vocab) >= 4
    for required in ("dermatopathology", "biopsy"):
        assert required in vocab, required


def test_future_extraction_field_inventory_is_nine():
    payload = _payload()
    fields = payload["future_extraction_field_inventory"]
    assert isinstance(fields, list)
    assert len(fields) == 9
    assert payload["future_extraction_field_count"] == 9
    names = [entry["field"] for entry in fields]
    for required in (
        "specimen",
        "clinical_diagnosis_or_reason_for_biopsy",
        "gross_description",
        "microscopic_description",
        "final_diagnosis",
        "icd_code",
        "margins_or_comments",
        "image_or_photomicrograph_presence",
        "signs_and_symptoms",
    ):
        assert required in names, required
    for entry in fields:
        assert entry["operator_observed"] is True
        assert entry["future_extraction_useful"] is True


def test_future_implementation_decision_is_justified_spec_only():
    payload = _payload()
    decision = payload["future_implementation_decision"]
    assert decision["future_implementation_justified"] is True
    assert (
        decision["future_implementation_block_id"]
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01"
    )
    assert (
        decision["future_implementation_block_mode"]
        == "reports_only_synthetic_only_family_spec"
    )
    blocked = decision["future_implementation_must_not_include"]
    for required in (
        "real document text",
        "real OCR text",
        "real filename",
        "real PDF",
        "real screenshot",
        "real diagnosis",
        "cue expansion in classifier code",
        "threshold change",
        "auto-accept enablement",
        "classifier rule change",
        "OCR routing change",
        "external API use",
    ):
        assert required in blocked, required


def test_recommended_next_block_is_pathology_spec_or_freeze_maintenance():
    payload = _payload()
    assert (
        payload["recommended_next_block"]
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY"
    )
    next3 = payload["recommended_next_3_blocks"]
    assert isinstance(next3, list)
    assert len(next3) == 3
    assert next3[0] == payload["recommended_next_block"]


def test_long_report_documents_all_sections():
    text = _report_text()
    for phrase in (
        "Scope And Non-Scope",
        "Trigger And Bucket Mapping",
        "Core Classifier Inspection",
        "Metadata Family Registry Inspection",
        "Two-Layer Split Summary",
        "Safe Structural Cue Inventory",
        "Controlled Family Name Recommendation",
        "Future Extraction Field Inventory",
        "Future Implementation Decision",
        "Recommended Smallest Safe Next Block",
        "Why No Behavior Is Changed In This Block",
        "Safety / Privacy Invariants",
        "Recommended Next 3-Block Sequence",
    ):
        assert phrase in text, phrase


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in REPORTS:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        assert check_public_report_payload(target).passed, path.name


def test_runtime_and_classifier_and_registry_files_do_not_mention_this_block():
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
        assert "MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01" not in body, relative
        assert "medai_real_doc_b4_pathology_family_eval_01" not in body, relative


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
            str(REPO_ROOT / "scripts" / "run_medai_real_doc_b4_pathology_family_eval_01.py"),
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
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY"
    )
