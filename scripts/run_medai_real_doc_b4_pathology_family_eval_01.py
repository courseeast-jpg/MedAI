#!/usr/bin/env python3
"""MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01 — Reports-only audit.

Evaluation-only. Confirms:

* the three eval report files exist;
* prior dependency report dirs exist
  (release freeze snapshot / freeze-maintenance-only / real-doc unknown
  triage);
* the JSON payload carries the canonical eval invariants set to the
  expected values;
* the report enumerates 9 future-extraction fields;
* the report enumerates the 4 supported language packs and the existing
  Pathology report metadata rule;
* every report passes
  ``clinical_knowledge.privacy.check_public_report_payload``;
* no runtime / launcher / persistence / Streamlit / V2-contract /
  classifier / OCR-gate / family-registry / lab-document-metadata file
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_b4_pathology_family_eval_01"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_REAL_DOC_B4_PATHOLOGY_FAMILY_EVAL_01.md",
    REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.json",
    REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.md",
)

REQUIRED_PRIOR_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_release_freeze_snapshot_01",
    REPO_ROOT / "reports" / "medai_v2_freeze_maintenance_only_01",
    REPO_ROOT / "reports" / "medai_real_doc_unknown_triage_01",
)

REQUIRED_FALSE_BOOLEANS = (
    "operator_supplied_real_document_artifact",
    "real_document_inspected_by_assistant",
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
    "phi_inspected",
    "implementation_started",
    "new_helper_created",
    "direct_implementation_recommended",
    "behavior_changed_in_this_block",
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
    "classifier_cue_added",
    "classifier_rule_changed",
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
    "external_api_used",
    "private_data_accessed",
    "source_documents_opened",
    "licensed_rows_read",
    "licensed_rows_exposed",
    "private_license_ack_read",
    "private_config_read",
    "runtime_db_accessed",
    "tags_touched",
    "tags_created",
    "tags_modified",
    "auto_accept_allowed_default",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "evaluation_only",
    "synthetic_only",
    "triggered_by_real_operator_signal",
    "preceding_release_freeze_snapshot_present",
    "preceding_freeze_maintenance_only_present",
    "preceding_real_doc_unknown_triage_present",
    "freeze_maintenance_posture_intact",
    "evaluation_framework_created",
    "structural_cue_inventory_created",
    "controlled_family_name_recommended",
    "extraction_field_inventory_created",
    "subtype_taxonomy_drafted",
    "future_implementation_justified",
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
    return ("MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01" not in src) and (
        "medai_real_doc_b4_pathology_family_eval_01" not in src
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
        "next_recommended_block_correct": False,
        "future_extraction_field_count_correct": False,
        "future_extraction_field_count_actual": 0,
        "language_pack_count_correct": False,
        "language_pack_count_actual": 0,
        "metadata_family_rule_present_in_source": False,
        "core_classifier_module_present": False,
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
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

    classifier_path = REPO_ROOT / "document_classification" / "document_classifier.py"
    findings["core_classifier_module_present"] = classifier_path.is_file()

    registry_path = REPO_ROOT / "app" / "document_type_registry.py"
    if registry_path.is_file():
        body = registry_path.read_text(encoding="utf-8", errors="replace")
        findings["metadata_family_rule_present_in_source"] = (
            "PATHOLOGY_REPORT_LABEL" in body
            and "pathology_conclusion_section" in body
        )

    json_path = REPORT_DIR / "medai_real_doc_b4_pathology_family_eval_01_report.json"
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

    findings["next_recommended_block_correct"] = (
        payload.get("recommended_next_block")
        == "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY"
    )

    fields = payload.get("future_extraction_field_inventory")
    if isinstance(fields, list):
        findings["future_extraction_field_count_actual"] = len(fields)
        findings["future_extraction_field_count_correct"] = len(fields) == 9

    metadata = payload.get("metadata_family_registry_findings") or {}
    langs = metadata.get("supported_language_packs")
    if isinstance(langs, list):
        findings["language_pack_count_actual"] = len(langs)
        findings["language_pack_count_correct"] = len(langs) == 4

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
            findings["next_recommended_block_correct"],
            findings["future_extraction_field_count_correct"],
            findings["language_pack_count_correct"],
            findings["metadata_family_rule_present_in_source"],
            findings["core_classifier_module_present"],
            findings["runtime_files_do_not_mention_this_block"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01",
        "mode": "reports_only_pathology_family_eval_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
