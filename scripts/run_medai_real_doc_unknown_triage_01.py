#!/usr/bin/env python3
"""MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01 — Reports-only audit.

Reports-only. Confirms:

* the three triage report files exist;
* prior freeze-maintenance dependency directories exist;
* the JSON payload carries the canonical triage invariants set to the
  expected values;
* the report enumerates 8 failure buckets and 20 public-safe diagnostic
  fields;
* every report passes
  ``clinical_knowledge.privacy.check_public_report_payload``;
* no runtime / launcher / persistence / Streamlit / V2-contract /
  classifier / OCR-gate file mentions this block.

No runtime code is modified. No private files are opened. No raw text,
raw OCR text, raw filenames, or private paths are read or printed. No
real PDF is committed. No tags are touched. No DB row is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_unknown_triage_01"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_REAL_DOC_UNKNOWN_TRIAGE_01.md",
    REPORT_DIR / "medai_real_doc_unknown_triage_01_report.json",
    REPORT_DIR / "medai_real_doc_unknown_triage_01_report.md",
)

REQUIRED_PRIOR_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_release_freeze_snapshot_01",
    REPO_ROOT / "reports" / "medai_v2_freeze_maintenance_only_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_default_off_status_registry_01",
)

REQUIRED_FALSE_BOOLEANS = (
    "operator_supplied_real_document_artifact",
    "real_document_inspected_by_assistant",
    "raw_text_inspected",
    "raw_ocr_text_inspected",
    "raw_filename_inspected",
    "private_paths_inspected",
    "phi_inspected",
    "implementation_started",
    "new_helper_created",
    "direct_implementation_recommended",
    "code_fix_justified_yet",
    "smallest_next_block_committed_to",
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
    "external_api_used",
    "private_data_accessed",
    "source_documents_opened",
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
    "tags_touched",
    "tags_created",
    "tags_modified",
    "auto_accept_allowed_default",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "triggered_by_real_operator_signal",
    "triage_framework_created",
    "diagnostic_field_inventory_created",
    "failure_bucket_taxonomy_created",
    "decision_tree_created",
    "bucket_conditional_next_block_table_created",
    "operator_action_required",
    "preceding_release_freeze_snapshot_present",
    "preceding_freeze_maintenance_only_present",
    "freeze_maintenance_posture_intact",
    "v1_release_preserved",
    "local_only_default",
    "review_bound_default",
    "external_api_blocked_default",
)

RUNTIME_CHECK_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
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
    return ("MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01" not in src) and (
        "medai_real_doc_unknown_triage_01" not in src
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
        "bucket_count_correct": False,
        "bucket_count_actual": 0,
        "diagnostic_field_count_correct": False,
        "diagnostic_field_count_actual": 0,
        "bucket_conditional_block_count_correct": False,
        "bucket_conditional_block_count_actual": 0,
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

    json_path = REPORT_DIR / "medai_real_doc_unknown_triage_01_report.json"
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
        == "OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY"
    )

    buckets = payload.get("failure_bucket_taxonomy")
    if isinstance(buckets, list):
        findings["bucket_count_actual"] = len(buckets)
        findings["bucket_count_correct"] = len(buckets) == 8

    fields = payload.get("advanced_diagnostic_fields_used_by_ui")
    if isinstance(fields, list):
        findings["diagnostic_field_count_actual"] = len(fields)
        findings["diagnostic_field_count_correct"] = len(fields) == 20

    cond_blocks = payload.get("bucket_conditional_smallest_next_block")
    if isinstance(cond_blocks, list):
        findings["bucket_conditional_block_count_actual"] = len(cond_blocks)
        findings["bucket_conditional_block_count_correct"] = len(cond_blocks) == 8

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
            findings["bucket_count_correct"],
            findings["diagnostic_field_count_correct"],
            findings["bucket_conditional_block_count_correct"],
            findings["runtime_files_do_not_mention_this_block"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01",
        "mode": "reports_only_real_doc_unknown_triage_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
