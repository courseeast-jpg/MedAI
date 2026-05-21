#!/usr/bin/env python3
"""MEDAI-V2-EXTRACTION-SPEC-01 report-only audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_extraction_spec_01"
REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_EXTRACTION_SPEC_01.md",
    REPORT_DIR / "medai_v2_extraction_spec_01_report.json",
    REPORT_DIR / "medai_v2_extraction_spec_01_report.md",
)
PRIOR_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
    REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01",
    REPO_ROOT / "reports" / "medai_v2_data_infra_spec_01",
)
FORBIDDEN_RUNTIME_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
)
FALSE_KEYS = (
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
    "launcher_changed",
    "startup_config_changed",
    "db_schema_changed",
    "migration_created",
    "migration_executed",
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
)
TRUE_KEYS = (
    "reports_only",
    "extraction_spec_created",
    "review_bound_default",
    "local_only_default",
    "external_api_blocked_default",
    "v1_release_preserved",
    "requires_preceding_spec",
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
)


def audit() -> Dict[str, Any]:
    payload = json.loads((REPORT_DIR / "medai_v2_extraction_spec_01_report.json").read_text(encoding="utf-8"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in REPORT_FILES if p.exists())
    findings: Dict[str, Any] = {
        "report_files_present": all(p.is_file() for p in REPORT_FILES),
        "prior_v2_dirs_present": all(d.is_dir() for d in PRIOR_DIRS),
        "false_keys_match": all(payload.get(k) is False for k in FALSE_KEYS),
        "true_keys_match": all(payload.get(k) is True for k in TRUE_KEYS),
        "next_recommended_block": payload.get("next_recommended_block"),
        "boundary_sections_present": all(
            phrase in text
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
            )
        ),
        "future_gate_count": len(payload.get("future_implementation_gates", [])),
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
        "privacy_passed": False,
        "privacy_failures": [],
    }
    for path in FORBIDDEN_RUNTIME_FILES:
        if path.is_file() and "MEDAI-V2-EXTRACTION-SPEC-01" in path.read_text(encoding="utf-8", errors="replace"):
            findings["runtime_files_do_not_mention_this_block"] = False
            findings["runtime_files_with_mention"].append(str(path.relative_to(REPO_ROOT)))
    from clinical_knowledge.privacy import check_public_report_payload

    failures = []
    for path in REPORT_FILES:
        content = path.read_text(encoding="utf-8")
        target: object = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        if not result.passed:
            failures.append(path.name)
    findings["privacy_passed"] = not failures
    findings["privacy_failures"] = failures
    findings["passed"] = (
        findings["report_files_present"]
        and findings["prior_v2_dirs_present"]
        and findings["false_keys_match"]
        and findings["true_keys_match"]
        and findings["next_recommended_block"] == "V2-ROADMAP-02"
        and findings["boundary_sections_present"]
        and findings["future_gate_count"] == 10
        and findings["runtime_files_do_not_mention_this_block"]
        and findings["privacy_passed"]
    )
    return findings


def main() -> int:
    findings = audit()
    print(json.dumps(findings, indent=2))
    return 0 if findings["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

