#!/usr/bin/env python3
"""MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01 — Reports-only audit.

Spec-only. Confirms:

* the three spec report files exist;
* prior dependency report dirs exist (release freeze snapshot,
  freeze-maintenance-only, real-doc unknown triage, B4 pathology eval);
* the JSON payload carries the canonical spec invariants set to the
  expected values;
* the controlled family name and subtype field are exactly as required;
* the subtype vocabulary has 6 entries;
* the cue candidate inventory has 3 core + 8 additional candidates
  (11 total);
* the future extraction field contract has 18 fields, all review-bound;
* the dermatology MKB doctrine, AI-agent interpretation boundary, and
  photomicrograph boundary are present;
* 11 future implementation gates are present;
* every report passes
  ``clinical_knowledge.privacy.check_public_report_payload``;
* no runtime / launcher / classifier / family-registry / OCR-gate file
  has been changed to mention this block.

No runtime code is modified. No private files are opened. No raw text,
raw OCR text, raw filenames, or private paths are read or printed. No
real PDF / screenshot / diagnosis is committed. No tags are touched.
No DB row is read. No classifier cue is added.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_pathology_report_family_spec_01"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_PATHOLOGY_REPORT_FAMILY_SPEC_01.md",
    REPORT_DIR / "medai_pathology_report_family_spec_01_report.json",
    REPORT_DIR / "medai_pathology_report_family_spec_01_report.md",
)

REQUIRED_PRIOR_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_release_freeze_snapshot_01",
    REPO_ROOT / "reports" / "medai_v2_freeze_maintenance_only_01",
    REPO_ROOT / "reports" / "medai_real_doc_unknown_triage_01",
    REPO_ROOT / "reports" / "medai_real_doc_b4_pathology_family_eval_01",
)

REQUIRED_FALSE_BOOLEANS = (
    "implementation_started",
    "new_helper_created",
    "direct_implementation_recommended",
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
    "private_data_accessed",
    "source_documents_opened",
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
    "ai_interpretation_implemented",
    "photomicrograph_interpretation_implemented",
    "tags_touched",
    "tags_created",
    "tags_modified",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "synthetic_only",
    "spec_only",
    "triggered_by_b4_pathology_evaluation",
    "preceding_b4_pathology_eval_present",
    "preceding_real_doc_unknown_triage_present",
    "preceding_freeze_maintenance_only_present",
    "preceding_release_freeze_snapshot_present",
    "freeze_maintenance_posture_intact",
    "pathology_family_spec_created",
    "pathology_family_support_justified",
    "review_required_default",
    "source_facts_only",
    "v1_release_preserved",
    "local_only_default",
    "review_bound_default",
    "external_api_blocked_default",
)

RUNTIME_CHECK_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
    REPO_ROOT / "app" / "document_type_registry.py",
    REPO_ROOT / "app" / "lab_document_metadata.py",
    REPO_ROOT / "Start_MedAI_UI.bat",
    REPO_ROOT / "Start_MedAI_UI_Silent.vbs",
    REPO_ROOT / "Start_MedAI_Test_UI.bat",
    REPO_ROOT / "Start_MedAI_UI_Encrypted.bat",
    REPO_ROOT / "document_classification" / "document_classifier.py",
    REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "__init__.py",
)


def _file_does_not_mention_this_block(p: Path) -> bool:
    if not p.is_file():
        return True
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return True
    return ("MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01" not in src) and (
        "medai_pathology_report_family_spec_01" not in src
    )


def _audit() -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "report_files_present": True,
        "missing_report_files": [],
        "prior_report_dirs_present": True,
        "missing_prior_report_dirs": [],
        "json_required_false_booleans_match": True,
        "json_required_false_booleans_mismatches": [],
        "json_required_true_booleans_match": True,
        "json_required_true_booleans_mismatches": [],
        "family_name_correct": False,
        "subtype_field_correct": False,
        "subtype_vocabulary_count_correct": False,
        "subtype_vocabulary_count_actual": 0,
        "core_cue_candidate_count_correct": False,
        "core_cue_candidate_count_actual": 0,
        "additional_cue_candidate_count_correct": False,
        "additional_cue_candidate_count_actual": 0,
        "total_cue_candidate_count_correct": False,
        "total_cue_candidate_count_actual": 0,
        "future_extraction_field_count_correct": False,
        "future_extraction_field_count_actual": 0,
        "all_fields_review_bound": False,
        "dermatology_mkb_doctrine_present": False,
        "ai_agent_interpretation_boundary_present": False,
        "photomicrograph_boundary_present": False,
        "future_implementation_gate_count_correct": False,
        "future_implementation_gate_count_actual": 0,
        "next_recommended_block_correct": False,
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
        "auto_accept_allowed_default_is_false": False,
        "ai_interpretation_allowed_in_classifier_layer_is_false": False,
        "public_report_privacy_passed": False,
        "public_report_privacy_failures": [],
    }

    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            findings["report_files_present"] = False
            findings["missing_report_files"].append(str(p.relative_to(REPO_ROOT)))

    for d in REQUIRED_PRIOR_REPORT_DIRS:
        if not d.is_dir():
            findings["prior_report_dirs_present"] = False
            findings["missing_prior_report_dirs"].append(
                str(d.relative_to(REPO_ROOT))
            )

    for p in RUNTIME_CHECK_FILES:
        if not _file_does_not_mention_this_block(p):
            findings["runtime_files_do_not_mention_this_block"] = False
            findings["runtime_files_with_mention"].append(
                str(p.relative_to(REPO_ROOT))
            )

    json_path = REPORT_DIR / "medai_pathology_report_family_spec_01_report.json"
    if not json_path.is_file():
        return findings

    payload: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    for key in REQUIRED_FALSE_BOOLEANS:
        if payload.get(key) is not False:
            findings["json_required_false_booleans_match"] = False
            findings["json_required_false_booleans_mismatches"].append(
                {"key": key, "value": payload.get(key)}
            )

    for key in REQUIRED_TRUE_BOOLEANS:
        if payload.get(key) is not True:
            findings["json_required_true_booleans_match"] = False
            findings["json_required_true_booleans_mismatches"].append(
                {"key": key, "value": payload.get(key)}
            )

    findings["family_name_correct"] = payload.get("family_name") == "pathology_report"
    findings["subtype_field_correct"] = (
        payload.get("subtype_field") == "pathology_subtype"
    )

    vocab = payload.get("subtype_vocabulary")
    if isinstance(vocab, list):
        findings["subtype_vocabulary_count_actual"] = len(vocab)
        findings["subtype_vocabulary_count_correct"] = len(vocab) == 6

    core = payload.get("core_cue_candidates")
    if isinstance(core, list):
        findings["core_cue_candidate_count_actual"] = len(core)
        findings["core_cue_candidate_count_correct"] = len(core) == 3

    extra = payload.get("additional_cue_candidates")
    if isinstance(extra, list):
        findings["additional_cue_candidate_count_actual"] = len(extra)
        findings["additional_cue_candidate_count_correct"] = len(extra) == 8

    findings["total_cue_candidate_count_actual"] = (
        findings["core_cue_candidate_count_actual"]
        + findings["additional_cue_candidate_count_actual"]
    )
    findings["total_cue_candidate_count_correct"] = (
        findings["total_cue_candidate_count_actual"]
        == payload.get("total_cue_candidate_count")
        == 11
    )

    fields = payload.get("future_extraction_field_contract")
    if isinstance(fields, list):
        findings["future_extraction_field_count_actual"] = len(fields)
        findings["future_extraction_field_count_correct"] = len(fields) == 18
        review_bound = all(
            isinstance(f, dict)
            and f.get("review_required") is True
            and f.get("auto_accept_allowed") is False
            and f.get("source_facts_only") is True
            and f.get("ai_interpretation_allowed_in_classifier_layer") is False
            for f in fields
        )
        findings["all_fields_review_bound"] = review_bound

    doctrine = payload.get("dermatology_mkb_placement_doctrine")
    findings["dermatology_mkb_doctrine_present"] = (
        isinstance(doctrine, dict)
        and isinstance(doctrine.get("rules"), list)
        and len(doctrine["rules"]) == doctrine.get("rule_count")
        and len(doctrine["rules"]) == 6
    )

    boundary = payload.get("ai_agent_interpretation_boundary")
    findings["ai_agent_interpretation_boundary_present"] = (
        isinstance(boundary, dict)
        and isinstance(boundary.get("rules"), list)
        and len(boundary["rules"]) == boundary.get("rule_count")
        and len(boundary["rules"]) == 7
        and boundary.get("ai_interpretation_implemented") is False
    )

    photo = payload.get("photomicrograph_boundary")
    findings["photomicrograph_boundary_present"] = (
        isinstance(photo, dict)
        and isinstance(photo.get("options"), list)
        and len(photo["options"]) == photo.get("option_count")
        and len(photo["options"]) == 4
        and photo.get("photomicrograph_interpretation_implemented") is False
    )

    gates = payload.get("future_implementation_gates")
    if isinstance(gates, list):
        findings["future_implementation_gate_count_actual"] = len(gates)
        findings["future_implementation_gate_count_correct"] = (
            len(gates) == payload.get("future_implementation_gate_count") == 11
        )

    findings["next_recommended_block_correct"] = (
        payload.get("recommended_next_block")
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01_OR_FREEZE-MAINTENANCE-ONLY"
    )

    findings["auto_accept_allowed_default_is_false"] = (
        payload.get("auto_accept_allowed_default") is False
    )
    findings["ai_interpretation_allowed_in_classifier_layer_is_false"] = (
        payload.get("ai_interpretation_allowed_in_classifier_layer") is False
    )

    from clinical_knowledge.privacy import check_public_report_payload  # noqa: WPS433

    privacy_pass = True
    privacy_failures: List[str] = []
    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            continue
        try:
            content: Any = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                content = json.loads(content)
        except Exception as exc:
            privacy_pass = False
            privacy_failures.append(f"{p.name}: read/parse failure ({exc})")
            continue
        result = check_public_report_payload(content)
        if not result.passed:
            privacy_pass = False
            privacy_failures.append(p.name)
    findings["public_report_privacy_passed"] = privacy_pass
    findings["public_report_privacy_failures"] = privacy_failures

    return findings


def _all_clean(findings: Dict[str, Any]) -> bool:
    return all(
        (
            findings["report_files_present"],
            findings["prior_report_dirs_present"],
            findings["json_required_false_booleans_match"],
            findings["json_required_true_booleans_match"],
            findings["family_name_correct"],
            findings["subtype_field_correct"],
            findings["subtype_vocabulary_count_correct"],
            findings["core_cue_candidate_count_correct"],
            findings["additional_cue_candidate_count_correct"],
            findings["total_cue_candidate_count_correct"],
            findings["future_extraction_field_count_correct"],
            findings["all_fields_review_bound"],
            findings["dermatology_mkb_doctrine_present"],
            findings["ai_agent_interpretation_boundary_present"],
            findings["photomicrograph_boundary_present"],
            findings["future_implementation_gate_count_correct"],
            findings["next_recommended_block_correct"],
            findings["runtime_files_do_not_mention_this_block"],
            findings["auto_accept_allowed_default_is_false"],
            findings["ai_interpretation_allowed_in_classifier_layer_is_false"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01",
        "mode": "reports_only_pathology_family_spec_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01_OR_FREEZE-MAINTENANCE-ONLY",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
