#!/usr/bin/env python3
"""MEDAI-V2-FOUNDATION-SPEC-02 — Reports-only foundation doctrine
re-affirmation script.

Pure read-only audit. Confirms that the three V2 foundation SPEC
report files exist, that the JSON report carries the required
invariant booleans set to the expected values, that the JSON report
references each of the 13 parked / frozen / blocked track anchors,
and that the public-report privacy checker accepts every report file.

No runtime code is modified. No private files are opened. No
licensed terminology rows are read. No tags are touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_spec_02"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_FOUNDATION_SPEC_02.md",
    REPORT_DIR / "medai_v2_foundation_spec_02_report.json",
    REPORT_DIR / "medai_v2_foundation_spec_02_report.md",
)

REQUIRED_FALSE_BOOLEANS = (
    "v2_implementation_started",
    "runtime_behavior_changed",
    "app_main_changed",
    "ui_changed",
    "launcher_changed",
    "startup_config_changed",
    "extraction_changed",
    "ocr_changed",
    "classifier_changed",
    "threshold_scoring_changed",
    "cue_pack_changed",
    "clinical_behavior_changed",
    "ddi_behavior_changed",
    "terminology_behavior_changed",
    "private_adapter_implemented",
    "external_api_used",
    "private_data_accessed",
    "licensed_rows_read",
    "private_license_ack_read",
    "private_config_read",
    "runtime_db_accessed",
    "source_documents_opened",
    "raw_text_printed",
    "raw_filenames_printed",
    "private_paths_printed",
    "secrets_printed",
    "tags_touched",
    "cue_expansion_recommended",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "v1_release_preserved",
    "foundation_doctrine_created",
    "stop_on_failure_rules_defined",
    "future_block_taxonomy_defined",
)

REQUIRED_PARKED_FROZEN_COMMIT_ANCHORS = (
    "7ef8ffd",
    "3e46461",
    "9f9e22d",
    "f4d3cc6",
    "748c32a",
    "1b14ffe",
    "6b31678",
    "91b9eba",
    "e398a75",
    "b9b19ad",
    "60f1114",
    "cedbbd3",
    "376ca4e",
)


def _audit() -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "report_files_present": True,
        "missing_report_files": [],
        "json_required_false_booleans_match": True,
        "json_required_false_booleans_mismatches": [],
        "json_required_true_booleans_match": True,
        "json_required_true_booleans_mismatches": [],
        "next_recommended_block_correct": False,
        "all_parked_frozen_anchors_referenced": True,
        "missing_anchor_references": [],
        "public_report_privacy_passed": False,
        "public_report_privacy_failures": [],
    }

    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            findings["report_files_present"] = False
            findings["missing_report_files"].append(str(p.relative_to(REPO_ROOT)))

    json_path = REPORT_DIR / "medai_v2_foundation_spec_02_report.json"
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
        payload.get("next_recommended_block") == "MEDAI-V2-RUNTIME-CONTRACTS-01"
    )

    # Anchor cross-reference: every required short SHA must appear
    # somewhere in the JSON payload (rendered as text). This makes the
    # parked/frozen track table tamper-evident.
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False)
    for anchor in REQUIRED_PARKED_FROZEN_COMMIT_ANCHORS:
        if anchor not in payload_text:
            findings["all_parked_frozen_anchors_referenced"] = False
            findings["missing_anchor_references"].append(anchor)

    # Privacy check — invoked inline so the audit script doubles as a
    # privacy gate.
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
        except Exception as exc:  # pragma: no cover (defensive)
            privacy_failures.append(f"{p.name}: read/parse failure ({exc})")
            privacy_pass = False
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
            findings["json_required_false_booleans_match"],
            findings["json_required_true_booleans_match"],
            findings["next_recommended_block_correct"],
            findings["all_parked_frozen_anchors_referenced"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-V2-FOUNDATION-SPEC-02",
        "mode": "v2_foundation_reports_only_spec_audit",
        "reports_only": True,
        "v2_implementation_started": False,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "MEDAI-V2-RUNTIME-CONTRACTS-01",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
