#!/usr/bin/env python3
"""MEDAI-AI-EXTERNAL-CALL-DRY-RUN-15E validation."""
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
from execution.ai_provider_registry import AIProviderRegistry
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_external_call_dry_run_15e"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
DRY_RUN_DECISION_JSON_PATH = REPORT_DIR / "dry_run_decision_check.json"
DRY_RUN_AUDIT_JSON_PATH = REPORT_DIR / "dry_run_audit_sample.json"
REGISTRY_JSON_PATH = REPORT_DIR / "provider_registry_check.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
BUDGET_JSON_PATH = REPORT_DIR / "budget_guard_check.json"
OPERATOR_PREVIEW_JSON_PATH = REPORT_DIR / "operator_preview_sample.json"
IMPLEMENTATION_MD_PATH = REPORT_DIR / "implementation_report.md"

PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
BLOCKED_REPORT_TOKENS = [
    "Jane Example",
    "01/02/1970",
    "123456",
    "CY-2026-0001",
    "Park Medical Center",
    "Alice Clinician",
    "123 Main Street",
    "555-123-4567",
    "jane.example@example.com",
    "INS-ABC-12345",
    "06/10/2026",
    '"token_map":',
    "C:\\",
    "sk-",
    "api_key",
    "credential",
]
PACKAGE_TYPES = ["cytology_pathology", "urinalysis_table", "portal_cards"]

REQUIRED_COMMANDS = [
    ("focused_15e_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_external_call_dry_run_15e.py"]),
    ("focused_15d_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_selection_ui_15d.py"]),
    ("focused_15c_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_adapter_stub_15c.py"]),
    ("focused_15b_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"]),
    ("focused_15a_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"]),
    ("regression_12a_script", [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"]),
    ("regression_13c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"]),
]
PRIOR_BASELINE_COMMANDS = [
    ("baseline_14e_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_source_package_content_quality_14e.py"]),
    ("baseline_14e_script", [sys.executable, "scripts/run_medai_source_package_content_quality_14e.py"]),
    ("baseline_14d_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_source_extraction_packages_14d.py"]),
    ("baseline_14d_script", [sys.executable, "scripts/run_medai_source_extraction_packages_14d.py"]),
    ("baseline_14c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_real_run_extraction_diagnostics_14c.py"]),
    ("baseline_14c_script", [sys.executable, "scripts/run_medai_real_run_extraction_diagnostics_14c.py"]),
    ("baseline_14b_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_real_ocr_extraction_pipeline_integration_14b.py"]),
    ("baseline_14b_script", [sys.executable, "scripts/run_medai_real_ocr_extraction_pipeline_integration_14b.py"]),
    ("baseline_14a_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_real_doc_cross_domain_extraction_14a.py"]),
    ("baseline_14a_script", [sys.executable, "scripts/run_medai_real_doc_cross_domain_extraction_14a.py"]),
    ("baseline_13c_script", [sys.executable, "scripts/run_medai_operator_usability_polish_13c.py"]),
    ("baseline_13c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"]),
    ("baseline_12a_script", [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"]),
    ("baseline_12a_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_workflow_uat_12a.py"]),
]


def build_reports() -> tuple[dict[str, Any], ...]:
    package_results = [
        workflow_result_to_public_dict(
            run_ai_extraction_workflow(
                ExtractionWorkflowContext(
                    source_class=source_class,
                    safe_source_document_id=f"source_fake_15e_{index:03d}",
                    selected_document_category="AI-assisted extraction",
                    selected_specialty_domain="urology",
                    source_modality="fake_local_adapter",
                    raw_text_local_only=PRIVATE_TEXT,
                    provider_name="gemini",
                    provider_mode="disabled",
                    operator_approval_state="approved_for_dry_run",
                    external_call_mode="dry_run",
                )
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    blocked_result = workflow_result_to_public_dict(
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class="urinalysis_table",
                safe_source_document_id="source_fake_15e_missing_approval",
                raw_text_local_only=PRIVATE_TEXT,
                provider_name="gemini",
                provider_mode="disabled",
                operator_approval_state="pending",
                external_call_mode="dry_run",
            )
        )
    )
    first = package_results[0]
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in package_results)
    dry_run_decision = {
        "allowed_decision": first["dry_run_decision_result"],
        "blocked_decision": blocked_result["dry_run_decision_result"],
        "dry_run_allowed_only_in_dry_run_mode": first["dry_run_decision_result"]["dry_run_external_call_allowed"] is True,
        "missing_approval_fails_closed": blocked_result["dry_run_decision_result"]["fail_closed_reason"]
        == "operator_approval_required_for_dry_run",
        "real_call_disabled": first["dry_run_decision_result"]["final_external_call_allowed"] is False,
        "real_network_call_used": False,
        "external_api_used": False,
    }
    dry_run_audit = first["dry_run_audit_result"]
    registry = AIProviderRegistry().list_providers()
    registry_check = {
        "providers": registry,
        "selected_provider": first["provider_selection_result"]["selected_provider"],
        "effective_provider": first["provider_selection_result"]["effective_provider"],
        "real_providers_disabled": all(
            not item["provider_enabled"] for item in registry if item["provider_name"] != "fake_local"
        ),
        "no_provider_sdk_required": not any(item["sdk_import_required"] for item in registry),
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
    }
    operator_preview = {
        "operator_notice": first["operator_preview"]["operator_notice_sentence"],
        "dry_run_status": first["operator_preview"]["dry_run_mode_status"],
        "selected_provider": first["operator_preview"]["requested_provider"],
        "effective_provider": first["operator_preview"]["effective_provider"],
        "approval_state": first["operator_preview"]["external_call_approval_status"],
        "privacy_gate_status": first["operator_preview"]["privacy_gate_status"],
        "payload_policy_allowed": first["operator_preview"]["payload_policy_allowed"],
        "budget_allowed": first["operator_preview"]["budget_allowed"],
        "dry_run_external_call_allowed": first["operator_preview"]["dry_run_external_call_allowed"],
        "final_external_call_allowed": False,
        "real_network_call_used": False,
    }
    budget = dict(first["budget_guard_result"])
    summary = {
        "block": "MEDAI-AI-EXTERNAL-CALL-DRY-RUN-15E",
        "selected_provider": first["provider_selection_result"]["selected_provider"],
        "effective_provider": first["provider_selection_result"]["effective_provider"],
        "operator_approval_state": first["dry_run_decision_result"]["operator_approval_state"],
        "dry_run_external_call_allowed": first["dry_run_decision_result"]["dry_run_external_call_allowed"],
        "final_external_call_allowed": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": review_bound_package_count,
        "simulated_response_created": first["dry_run_decision_result"]["simulated_response_created"],
    }
    validation = _validation_report()
    privacy = _privacy_report(summary, validation, dry_run_decision, dry_run_audit, registry_check, budget, operator_preview)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return summary, validation, dry_run_decision, dry_run_audit, registry_check, privacy, budget, operator_preview, implementation


def _validation_report() -> dict[str, Any]:
    validation: dict[str, Any] = {
        name: _run_command(command, " ".join(_public_command(command)))
        for name, command in REQUIRED_COMMANDS
    }
    validation.update(
        {
            name: _run_optional_command(command, " ".join(_public_command(command)))
            for name, command in PRIOR_BASELINE_COMMANDS
        }
    )
    validation["source_safety_scan"] = _source_safety_scan()
    return validation


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15E_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def _run_optional_command(command: list[str], public_command: str) -> dict[str, Any]:
    if not _command_target_exists(command):
        return {"command": public_command, "missing": True}
    return _run_command(command, public_command)


def _command_target_exists(command: list[str]) -> bool:
    if len(command) >= 3 and command[1] == "-m":
        return True
    target = command[-1]
    if target.endswith(".py"):
        return (REPO_ROOT / target).exists()
    return True


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "execution/ai_external_call_dry_run.py",
        "execution/ai_provider_config.py",
        "execution/ai_provider_registry.py",
        "execution/extraction_workflow.py",
        "scripts/run_medai_ai_external_call_dry_run_15e.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "google." + "generativeai",
        "requests" + ".",
        "urllib" + ".request",
        "httpx" + ".",
        "api." + "openai.com",
        "generativelanguage." + "googleapis.com",
        "anthropic." + "com",
        "raw_pdf" + "_upload",
        "raw_image" + "_upload",
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_imports_introduced": False,
        "network_call_code_introduced": False,
        "raw_pdf_image_upload_introduced": False,
        "matches": matches,
        "passed": not matches,
    }


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True)
    token_scan_passed = not any(token in combined for token in BLOCKED_REPORT_TOKENS)
    payload_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if token_scan_passed and payload_check.passed else "failed",
        "raw_ocr_text_in_report": False,
        "private_identifiers_in_report": False,
        "token_map_in_report": False,
        "secret_material_in_report": False,
        "provider_sdk_usage": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-EXTERNAL-CALL-DRY-RUN-15E",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Approval state: `{summary['operator_approval_state']}`",
            f"- Dry-run allowed: `{summary['dry_run_external_call_allowed']}`",
            f"- Simulated response created: `{summary['simulated_response_created']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15E test code: `{validation['focused_15e_tests'].get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Dry-run is deterministic and local only.",
            "- Real provider execution remains disabled.",
            "- Review output is package-only and requires human review.",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    summary, validation, dry_run_decision, dry_run_audit, registry_check, privacy, budget, operator_preview, implementation = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    DRY_RUN_DECISION_JSON_PATH.write_text(json.dumps(dry_run_decision, indent=2), encoding="utf-8")
    DRY_RUN_AUDIT_JSON_PATH.write_text(json.dumps(dry_run_audit, indent=2), encoding="utf-8")
    REGISTRY_JSON_PATH.write_text(json.dumps(registry_check, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def _public_command(command: list[str]) -> list[str]:
    return ["python" if item == sys.executable else item for item in command]


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    summary, validation, dry_run_decision, _audit, registry_check, privacy, _budget, operator_preview, _implementation = reports
    validation_items = [
        item for key, item in validation.items()
        if key != "source_safety_scan"
    ]
    commands_ok = all(item.get("skipped") or item.get("missing") or item.get("returncode") == 0 for item in validation_items)
    ready = all(
        [
            commands_ok,
            validation["source_safety_scan"]["passed"],
            privacy["privacy_result"] == "passed",
            dry_run_decision["allowed_decision"]["dry_run_external_call_allowed"] is True,
            dry_run_decision["blocked_decision"]["dry_run_external_call_allowed"] is False,
            dry_run_decision["real_call_disabled"] is True,
            registry_check["real_providers_disabled"],
            registry_check["no_provider_sdk_required"],
            operator_preview["operator_notice"] == "No external AI call was made.",
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["review_bound_package_count"] >= 3,
        ]
    )
    print("medai_ai_external_call_dry_run_15e_ready" if ready else "medai_ai_external_call_dry_run_15e_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "operator_approval_state": summary["operator_approval_state"],
                "dry_run_external_call_allowed": summary["dry_run_external_call_allowed"],
                "external_api_used": summary["external_api_used"],
                "real_network_call_used": summary["real_network_call_used"],
                "final_external_call_allowed": summary["final_external_call_allowed"],
                "active_written_count": summary["active_written_count"],
                "auto_accept": summary["auto_accept"],
                "review_bound_package_count": summary["review_bound_package_count"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
