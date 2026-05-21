#!/usr/bin/env python3
"""MEDAI-V2-PACKAGING-SPEC-01 report-only audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_packaging_spec_01"
REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_PACKAGING_SPEC_01.md",
    REPORT_DIR / "medai_v2_packaging_spec_01_report.json",
    REPORT_DIR / "medai_v2_packaging_spec_01_report.md",
)
PRIOR_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
    REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01",
    REPO_ROOT / "reports" / "medai_v2_data_infra_spec_01",
    REPO_ROOT / "reports" / "medai_v2_extraction_spec_01",
    REPO_ROOT / "reports" / "medai_v2_roadmap_02",
    REPO_ROOT / "reports" / "medai_v2_foundation_implementation_readiness_01",
)
FORBIDDEN_RUNTIME_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
)
FALSE_KEYS = (
    "launcher_changed",
    "installer_changed",
    "deployment_script_changed",
    "startup_config_changed",
    "runtime_behavior_changed",
    "implementation_started",
    "default_off_helper_created",
    "app_main_changed",
    "streamlit_code_changed",
    "ui_changed",
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
)
TRUE_KEYS = (
    "reports_only",
    "packaging_spec_created",
    "v1_release_preserved",
    "local_only_default",
    "review_bound_default",
    "external_api_blocked_default",
    "requires_preceding_spec",
    "preceding_architecture_spec_present",
    "preceding_foundation_spec_present",
    "preceding_runtime_contracts_present",
    "preceding_validation_harness_present",
    "preceding_ui_shell_spec_present",
    "preceding_data_infra_spec_present",
    "preceding_extraction_spec_present",
    "preceding_roadmap_02_present",
    "preceding_implementation_readiness_present",
    "local_operator_packaging_boundary_defined",
    "launcher_boundary_defined",
    "startup_preflight_boundary_defined",
    "validation_receipt_boundary_defined",
    "release_artifact_boundary_defined",
    "environment_boundary_defined",
    "operator_handoff_boundary_defined",
    "parking_freeze_boundary_defined",
    "future_packaging_implementation_gates_defined",
    "v1_validation_healthcheck_catalog_preserved",
)


def audit() -> Dict[str, Any]:
    payload = json.loads((REPORT_DIR / "medai_v2_packaging_spec_01_report.json").read_text(encoding="utf-8"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in REPORT_FILES if p.exists())
    findings: Dict[str, Any] = {
        "report_files_present": all(p.is_file() for p in REPORT_FILES),
        "prior_v2_dirs_present": all(d.is_dir() for d in PRIOR_DIRS),
        "false_keys_match": all(payload.get(k) is False for k in FALSE_KEYS),
        "true_keys_match": all(payload.get(k) is True for k in TRUE_KEYS),
        "recommended_next_block": payload.get("recommended_next_block"),
        "recommended_next_3_blocks": payload.get("recommended_next_3_blocks"),
        "future_gate_count": len(payload.get("future_packaging_implementation_gates", [])),
        "v1_healthcheck_count": len(payload.get("v1_five_validation_healthcheck_set", [])),
        "readiness_outcome_carried_forward": payload.get("readiness_outcome_carried_forward"),
        "safest_candidate_carried_forward": payload.get("safest_future_implementation_candidate_carried_forward"),
        "required_sections_present": all(
            phrase in text
            for phrase in (
                "Scope And Non-Scope",
                "Prior V2 Dependency Chain",
                "Local Operator Packaging Boundary",
                "Launcher Boundary",
                "Startup / Preflight Boundary",
                "Validation Receipt Boundary",
                "Release Artifact Boundary",
                "Environment Boundary",
                "Operator Handoff Boundary",
                "Parking / Freeze Boundary",
                "Future Packaging Implementation Gates",
                "Safety And Privacy Invariants",
                "Validation Matrix",
                "Recommended Next Block",
            )
        ),
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
        "privacy_passed": False,
        "privacy_failures": [],
    }
    for path in FORBIDDEN_RUNTIME_FILES:
        if path.is_file() and "MEDAI-V2-PACKAGING-SPEC-01" in path.read_text(encoding="utf-8", errors="replace"):
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
        and findings["recommended_next_block"] == "V2-ROADMAP-PARK-01"
        and findings["recommended_next_3_blocks"] == [
            "V2-ROADMAP-PARK-01",
            "V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY",
            "V2-ROADMAP-03",
        ]
        and findings["future_gate_count"] == 10
        and findings["v1_healthcheck_count"] == 5
        and findings["readiness_outcome_carried_forward"] == "conditionally_ready_after_packaging_spec"
        and findings["safest_candidate_carried_forward"] == "V2 foundation default-off status registry"
        and findings["required_sections_present"]
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

