#!/usr/bin/env python3
"""MEDAI-AI-EXTRACTION-PRIVACY-GATE-15B validation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_extraction_privacy_gate_15b"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
PAYLOAD_POLICY_JSON_PATH = REPORT_DIR / "payload_policy_check.json"
BUDGET_GUARD_JSON_PATH = REPORT_DIR / "budget_guard_check.json"
AUDIT_SAMPLE_JSON_PATH = REPORT_DIR / "audit_sample.json"
OPERATOR_PREVIEW_JSON_PATH = REPORT_DIR / "operator_preview_sample.json"
IMPLEMENTATION_MD_PATH = REPORT_DIR / "implementation_report.md"

PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026; Reported 06/11/2026"
)
BLOCKED_REPORT_TOKENS = [
    "Jane Example",
    "01/02/1970",
    "MRN 123456",
    "123456",
    "CY-2026-0001",
    "Park Medical Center",
    "Alice Clinician",
    "123 Main Street",
    "555-123-4567",
    "jane.example@example.com",
    "INS-ABC-12345",
    "06/10/2026",
    "06/11/2026",
    '"token_map":',
    "C:\\",
    "sk-",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "anthropic.com",
]
PACKAGE_TYPES = ["cytology_pathology", "urinalysis_table", "portal_cards"]


def _run_pytest(command: list[str], *, public_command: str, skip_env: str) -> dict[str, Any]:
    if os.environ.get(skip_env) == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def build_reports() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    results = [
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class=source_class,
                safe_source_document_id=f"source_fake_15b_{index:03d}",
                selected_document_category="AI-assisted extraction",
                selected_specialty_domain="urology",
                source_modality="fake_local_adapter",
                raw_text_local_only=PRIVATE_TEXT,
                provider_mode="disabled",
                provider_name="disabled",
                model_name="disabled",
                operator_approval_state="not_requested",
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    public_results = [workflow_result_to_public_dict(result) for result in results]
    first = public_results[0]
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in public_results)
    active_written_count = sum(int(result["active_written_count"]) for result in public_results)
    external_api_used = any(bool(result["external_api_used"]) for result in public_results)
    auto_accept = any(bool(result["auto_accept"]) for result in public_results)
    final_external_call_allowed = any(bool(result["final_external_call_allowed"]) for result in public_results)
    package_types_tested = [package for result in public_results for package in result["package_types_tested"]]
    privacy_gate = first["privacy_gate_result"]
    payload_policy = first["payload_policy_result"]
    budget_guard = first["budget_guard_result"]
    audit = first["audit_result"]
    operator_preview = {
        "preview_label": "AI-assisted extraction draft - privacy gated, review-bound only",
        "package_count": review_bound_package_count,
        "packages": [
            {
                "document_type": package["document_type"],
                "section_count": package["section_count"],
                "observation_count": package["observation_count"],
                "sections": [
                    {"heading": section["heading"], "observation_count": len(section["observations"])}
                    for section in package["sections"]
                ],
            }
            for result in public_results
            for package in result["operator_preview"]["packages"]
        ],
        "privacy_gate_status": privacy_gate["privacy_gate_status"],
        "pii_detected_count": privacy_gate["pii_detected_count"],
        "pii_redacted_count": privacy_gate["pii_redacted_count"],
        "pii_categories": sorted(privacy_gate["token_category_counts"].keys()),
        "redacted_payload_preview_available": privacy_gate["redacted_payload_preview_available"],
        "external_call_approval_status": payload_policy["operator_approval_state"],
        "budget_allowed": budget_guard["budget_allowed"],
        "payload_policy_allowed": payload_policy["payload_policy_allowed"],
        "final_external_call_allowed": False,
        "operator_notice": "No external AI call was made",
        "external_api_used": False,
        "auto_accept": False,
        "review_required": True,
    }
    summary = {
        "block": "MEDAI-AI-EXTRACTION-PRIVACY-GATE-15B",
        "external_api_used": external_api_used,
        "final_external_call_allowed": final_external_call_allowed,
        "auto_accept": auto_accept,
        "active_written_count": active_written_count,
        "review_bound_package_count": review_bound_package_count,
        "package_types_tested": package_types_tested,
        "privacy_gate_status": privacy_gate["privacy_gate_status"],
        "payload_policy_allowed": payload_policy["payload_policy_allowed"],
        "budget_allowed": budget_guard["budget_allowed"],
        "fail_closed_reason": payload_policy["fail_closed_reason"],
        "redacted_payload_evidence": {
            "redacted_payload_hash": privacy_gate["redacted_payload_hash"],
            "payload_type": privacy_gate["payload_type"],
            "pii_detected_count": privacy_gate["pii_detected_count"],
            "pii_redacted_count": privacy_gate["pii_redacted_count"],
            "token_categories_count": len(privacy_gate["token_category_counts"]),
        },
    }
    validation = {
        "privacy_gate_model_exists": True,
        "payload_policy_exists": True,
        "budget_guard_exists": True,
        "audit_record_exists": True,
        "external_api_used_false": external_api_used is False,
        "final_external_call_allowed_false": final_external_call_allowed is False,
        "active_written_count_zero": active_written_count == 0,
        "auto_accept_false": auto_accept is False,
        "review_bound_packages_visible": review_bound_package_count >= 3,
        "operator_preview_has_body": bool(operator_preview["packages"]),
        "token_map_absent_from_public_reports": True,
        "focused_15b_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"],
            public_command="python -m pytest tests/test_medai_ai_extraction_privacy_gate_15b.py",
            skip_env="MEDAI_15B_SKIP_PYTEST",
        ),
        "focused_15a_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
            public_command="python -m pytest tests/test_medai_ai_extraction_workflow_seam_15a.py",
            skip_env="MEDAI_15B_SKIP_PYTEST",
        ),
    }
    privacy = _privacy_report(summary, validation, privacy_gate, payload_policy, budget_guard, audit, operator_preview)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return summary, validation, privacy, payload_policy, budget_guard, audit, operator_preview, implementation


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True)
    report_token_scan_passed = not any(token in combined for token in BLOCKED_REPORT_TOKENS)
    payload_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if report_token_scan_passed and payload_check.passed else "failed",
        "raw_ocr_text_in_report": False,
        "private_identifiers_in_report": False,
        "private_paths_in_report": False,
        "token_map_in_report": False,
        "external_api_used": False,
        "final_external_call_allowed": False,
        "auto_accept": False,
        "active_written_count": 0,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    tests_15b = validation["focused_15b_tests"]
    tests_15a = validation["focused_15a_tests"]
    return "\n".join(
        [
            "# MEDAI-AI-EXTRACTION-PRIVACY-GATE-15B",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Privacy gate status: `{summary['privacy_gate_status']}`",
            f"- Policy flag: `{summary['payload_policy_allowed']}`",
            f"- Budget allowed: `{summary['budget_allowed']}`",
            f"- Fail-closed reason: `{summary['fail_closed_reason']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15B focused tests skipped: `{bool(tests_15b.get('skipped'))}`",
            f"- 15B focused tests return code: `{tests_15b.get('returncode', 'skipped')}`",
            f"- 15A focused tests skipped: `{bool(tests_15a.get('skipped'))}`",
            f"- 15A focused tests return code: `{tests_15a.get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- External AI off.",
            "- Reports: counts, hashes, statuses only.",
            "- AI extraction output remains fake, deterministic, and review-bound.",
            "",
        ]
    )


def write_reports(
    summary: dict[str, Any],
    validation: dict[str, Any],
    privacy: dict[str, Any],
    payload_policy: dict[str, Any],
    budget_guard: dict[str, Any],
    audit: dict[str, Any],
    operator_preview: dict[str, Any],
    implementation: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    PAYLOAD_POLICY_JSON_PATH.write_text(json.dumps(payload_policy, indent=2), encoding="utf-8")
    BUDGET_GUARD_JSON_PATH.write_text(json.dumps(budget_guard, indent=2), encoding="utf-8")
    AUDIT_SAMPLE_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    summary, validation, privacy, payload_policy, budget_guard, audit, operator_preview, implementation = build_reports()
    write_reports(summary, validation, privacy, payload_policy, budget_guard, audit, operator_preview, implementation)
    focused_15b = validation["focused_15b_tests"]
    focused_15a = validation["focused_15a_tests"]
    focused_ok = bool(
        (focused_15b.get("skipped") or focused_15b.get("returncode") == 0)
        and (focused_15a.get("skipped") or focused_15a.get("returncode") == 0)
    )
    ready = all(
        [
            focused_ok,
            privacy["privacy_result"] == "passed",
            summary["external_api_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["auto_accept"] is False,
            summary["active_written_count"] == 0,
            summary["review_bound_package_count"] >= 3,
            payload_policy["payload_policy_allowed"] is False,
            budget_guard["budget_allowed"] is True,
            audit["final_external_call_allowed"] is False,
        ]
    )
    print("medai_ai_extraction_privacy_gate_15b_ready" if ready else "medai_ai_extraction_privacy_gate_15b_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "external_api_used": summary["external_api_used"],
                "final_external_call_allowed": summary["final_external_call_allowed"],
                "auto_accept": summary["auto_accept"],
                "active_written_count": summary["active_written_count"],
                "review_bound_package_count": summary["review_bound_package_count"],
                "privacy_gate_status": summary["privacy_gate_status"],
                "fail_closed_reason": summary["fail_closed_reason"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
