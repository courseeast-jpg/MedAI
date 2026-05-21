#!/usr/bin/env python3
"""MEDAI-V2-ROADMAP-02 report-only audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_roadmap_02"
REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_ROADMAP_02.md",
    REPORT_DIR / "medai_v2_roadmap_02_report.json",
    REPORT_DIR / "medai_v2_roadmap_02_report.md",
)
PRIOR_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
    REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01",
    REPO_ROOT / "reports" / "medai_v2_data_infra_spec_01",
    REPO_ROOT / "reports" / "medai_v2_extraction_spec_01",
)
FORBIDDEN_RUNTIME_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
)
FALSE_KEYS = (
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
)
TRUE_KEYS = (
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
)


def audit() -> Dict[str, Any]:
    payload = json.loads((REPORT_DIR / "medai_v2_roadmap_02_report.json").read_text(encoding="utf-8"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in REPORT_FILES if p.exists())
    findings: Dict[str, Any] = {
        "report_files_present": all(p.is_file() for p in REPORT_FILES),
        "prior_v2_dirs_present": all(d.is_dir() for d in PRIOR_DIRS),
        "false_keys_match": all(payload.get(k) is False for k in FALSE_KEYS),
        "true_keys_match": all(payload.get(k) is True for k in TRUE_KEYS),
        "recommended_next_block": payload.get("recommended_next_block"),
        "recommended_next_3_blocks": payload.get("recommended_next_3_blocks"),
        "completed_v2_block_count": len(payload.get("completed_v2_planning_sequence", [])),
        "ranked_workstream_count": len(payload.get("remaining_workstream_rankings", [])),
        "blocked_work_present": bool(payload.get("blocked_work")),
        "deferred_work_present": bool(payload.get("deferred_or_not_recommended")),
        "required_sections_present": all(
            phrase in text
            for phrase in (
                "Scope And Non-Scope",
                "Prior V2 Planning Sequence Summary",
                "Validation Health Summary",
                "Remaining Workstream Ranking",
                "Risk Matrix",
                "Blocked And Deferred Work Status",
                "Recommended Next Posture",
                "Recommended Next 3-Block Sequence",
                "Safety And Privacy Invariant Summary",
                "Freeze And Parking Strategy",
                "Validation Matrix",
                "Final Recommendation",
            )
        ),
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
        "privacy_passed": False,
        "privacy_failures": [],
    }
    for path in FORBIDDEN_RUNTIME_FILES:
        if path.is_file() and "MEDAI-V2-ROADMAP-02" in path.read_text(encoding="utf-8", errors="replace"):
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
        and findings["recommended_next_block"] == "V2-FOUNDATION-IMPLEMENTATION-READINESS-01"
        and findings["recommended_next_3_blocks"] == [
            "V2-FOUNDATION-IMPLEMENTATION-READINESS-01",
            "V2-PACKAGING-SPEC-01",
            "V2-ROADMAP-PARK-01",
        ]
        and findings["completed_v2_block_count"] == 7
        and findings["ranked_workstream_count"] >= 10
        and findings["blocked_work_present"]
        and findings["deferred_work_present"]
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

