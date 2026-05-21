#!/usr/bin/env python3
"""MEDAI-V2-FREEZE-MAINTENANCE-ONLY-01 report-only audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_freeze_maintenance_only_01"
REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_FREEZE_MAINTENANCE_ONLY_01.md",
    REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.json",
    REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.md",
)
PRIOR_DIRS = (
    "medai_v2_architecture_spec_01",
    "medai_v2_foundation_spec_02",
    "medai_v2_runtime_contracts_01",
    "medai_v2_validation_harness_01",
    "medai_v2_ui_shell_spec_01",
    "medai_v2_data_infra_spec_01",
    "medai_v2_extraction_spec_01",
    "medai_v2_roadmap_02",
    "medai_v2_foundation_implementation_readiness_01",
    "medai_v2_packaging_spec_01",
    "medai_v2_roadmap_park_01",
    "medai_v2_foundation_default_off_implementation_plan_01",
    "medai_v2_foundation_default_off_status_registry_01",
    "medai_v2_roadmap_03",
    "medai_v2_roadmap_park_02",
    "medai_v2_release_freeze_snapshot_01",
)
FALSE_KEYS = (
    "implementation_started",
    "new_helper_created",
    "direct_implementation_recommended",
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
    "tags_created",
    "tags_modified",
    "cue_expansion_recommended",
    "auto_accept_allowed_default",
)
TRUE_KEYS = (
    "reports_only",
    "freeze_maintenance_snapshot_created",
    "freeze_maintenance_only_selected",
    "preceding_architecture_spec_present",
    "preceding_foundation_spec_present",
    "preceding_runtime_contracts_present",
    "preceding_validation_harness_present",
    "preceding_ui_shell_spec_present",
    "preceding_data_infra_spec_present",
    "preceding_extraction_spec_present",
    "preceding_roadmap_02_present",
    "preceding_implementation_readiness_present",
    "preceding_packaging_spec_present",
    "preceding_roadmap_park_01_present",
    "preceding_default_off_implementation_plan_present",
    "preceding_status_registry_present",
    "preceding_roadmap_03_present",
    "preceding_roadmap_park_02_present",
    "preceding_release_freeze_snapshot_present",
    "status_registry_created_previously",
    "status_registry_standard_library_only",
    "status_registry_import_side_effect_free",
    "terminology_private_adapter_blocked",
    "cue_expansion_blocked",
    "clinical_decision_logic_expansion_blocked",
    "freeze_inventory_created",
    "v1_release_preserved",
    "local_only_default",
    "review_bound_default",
    "external_api_blocked_default",
)


def audit() -> Dict[str, Any]:
    payload = json.loads((REPORT_DIR / "medai_v2_freeze_maintenance_only_01_report.json").read_text(encoding="utf-8"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in REPORT_FILES if path.exists())
    from clinical_knowledge.v2_foundation.status_registry import summarize_v2_capability_registry

    summary = summarize_v2_capability_registry()
    findings: Dict[str, Any] = {
        "report_files_present": all(path.is_file() for path in REPORT_FILES),
        "prior_v2_dirs_present": all((REPO_ROOT / "reports" / name).is_dir() for name in PRIOR_DIRS),
        "required_modules_present": all(
            path.is_file()
            for path in (
                REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py",
                REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "__init__.py",
                REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
                REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
            )
        ),
        "false_keys_match": all(payload.get(key) is False for key in FALSE_KEYS),
        "true_keys_match": all(payload.get(key) is True for key in TRUE_KEYS),
        "recommended_next_block": payload.get("recommended_next_block"),
        "recommended_next_3_blocks": payload.get("recommended_next_3_blocks"),
        "chain_count": len(payload.get("freeze_maintenance_chain", [])),
        "allowed_future_work_count": len(payload.get("only_allowed_future_work", [])),
        "freeze_inventory_created": bool(payload.get("freeze_inventory")),
        "registry_summary_matches": (
            summary["total_count"] == payload.get("status_registry_entry_count")
            and summary["default_enabled_count"] == payload.get("status_registry_default_enabled_count")
            and summary["runtime_wired_count"] == payload.get("status_registry_runtime_wired_count")
            and summary["ui_wired_count"] == payload.get("status_registry_ui_wired_count")
            and summary["blocked_count"] == payload.get("status_registry_blocked_entry_count")
        ),
        "required_sections_present": all(
            phrase in text
            for phrase in (
                "Scope And Non-Scope",
                "Completed V2 Chain Summary",
                "Status Registry Posture Summary",
                "Release-Freeze Decision Carried Forward",
                "Freeze-Maintenance Decision",
                "Freeze Inventory",
                "Allowed Future Maintenance Work",
                "Blocked And Deferred Track Status",
                "Safety And Privacy Invariant Summary",
                "Validation Matrix",
                "Recommended Next 3-Block Sequence",
                "Final Recommendation",
            )
        ),
        "privacy_passed": False,
        "privacy_failures": [],
    }
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
        and findings["required_modules_present"]
        and findings["false_keys_match"]
        and findings["true_keys_match"]
        and findings["recommended_next_block"] == "V2-ROADMAP-04_IF_NEW_SIGNAL_ELSE_FREEZE-MAINTENANCE-ONLY"
        and findings["chain_count"] == 17
        and findings["allowed_future_work_count"] == 5
        and findings["freeze_inventory_created"]
        and findings["registry_summary_matches"]
        and findings["required_sections_present"]
        and findings["privacy_passed"]
    )
    return findings


def main() -> int:
    findings = audit()
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0 if findings["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

