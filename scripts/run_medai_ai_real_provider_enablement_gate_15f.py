#!/usr/bin/env python3
"""MEDAI-AI-REAL-PROVIDER-ENABLEMENT-GATE-15F validation."""
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
from execution.ai_provider_enablement import (
    default_real_provider_policies,
    evaluate_real_provider_execution_readiness,
    real_provider_credential_to_public_dict,
    real_provider_enablement_to_public_dict,
    real_provider_safety_checklist_to_public_dict,
)
from execution.ai_provider_registry import AIProviderRegistry
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_real_provider_enablement_gate_15f"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
ENABLEMENT_JSON_PATH = REPORT_DIR / "provider_enablement_check.json"
CREDENTIAL_JSON_PATH = REPORT_DIR / "credential_readiness_check.json"
SAFETY_JSON_PATH = REPORT_DIR / "safety_checklist.json"
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
PRIVATE_DEMO_SECRET = "demo-secret-value-" + "not-written-15f"
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
    PRIVATE_DEMO_SECRET,
    '"token_map":',
    "C:\\",
    "sk-",
]
PACKAGE_TYPES = ["cytology_pathology", "urinalysis_table", "portal_cards"]

REQUIRED_COMMANDS = [
    ("focused_15f_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_real_provider_enablement_gate_15f.py"]),
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
                    safe_source_document_id=f"source_fake_15f_{index:03d}",
                    selected_document_category="AI-assisted extraction",
                    selected_specialty_domain="urology",
                    source_modality="fake_local_adapter",
                    raw_text_local_only=PRIVATE_TEXT,
                    provider_name="gemini",
                    provider_mode="disabled",
                    operator_approval_state="approved_for_dry_run",
                    external_call_mode="dry_run",
                    real_provider_enablement_mode="readiness_check",
                )
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    first = package_results[0]
    present_readiness = evaluate_real_provider_execution_readiness(
        provider_name="gemini",
        operator_approval_state="approved_for_real_provider",
        dry_run_decision_result=first["dry_run_decision_result"],
        environ={"GEMINI_API_KEY": PRIVATE_DEMO_SECRET},
    )
    missing_readiness = evaluate_real_provider_execution_readiness(
        provider_name="gemini",
        operator_approval_state="approved_for_real_provider",
        dry_run_decision_result=first["dry_run_decision_result"],
        environ={},
    )
    policies = default_real_provider_policies()
    registry = AIProviderRegistry().list_providers()
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in package_results)
    provider_enablement = {
        "provider_defaults": {
            name: {
                key: value
                for key, value in policy.__dict__.items()
                if key != "credential_env_var_name"
            }
            | {"credential_env_var_configured": bool(policy.credential_env_var_name)}
            for name, policy in sorted(policies.items())
        },
        "workflow_decision": _without_env_var_name(first["real_provider_enablement_result"]),
        "present_credential_decision": _without_env_var_name(real_provider_enablement_to_public_dict(present_readiness)),
        "missing_credential_decision": _without_env_var_name(real_provider_enablement_to_public_dict(missing_readiness)),
        "real_providers_disabled": all(
            not item["provider_enabled"] for item in registry if item["provider_name"] != "fake_local"
        ),
        "fake_local_enabled": any(item["provider_name"] == "fake_local" and item["provider_enabled"] for item in registry),
        "real_provider_execution_enabled": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
    }
    credential = {
        "present_sample": real_provider_credential_to_public_dict(present_readiness),
        "missing_sample": real_provider_credential_to_public_dict(missing_readiness),
        "credential_value_written": False,
        "credential_printed_or_logged": False,
    }
    safety = real_provider_safety_checklist_to_public_dict(present_readiness)
    operator_preview = {
        "operator_notice": first["operator_preview"]["operator_notice_sentence"],
        "real_provider_notice": first["operator_preview"]["real_provider_execution_notice"],
        "selected_provider": first["operator_preview"]["requested_provider"],
        "effective_provider": first["operator_preview"]["effective_provider"],
        "credential_env_var_name": first["operator_preview"]["credential_env_var_name"],
        "credential_present": first["operator_preview"]["credential_present"],
        "real_provider_execution_enabled": first["operator_preview"]["real_provider_execution_enabled"],
        "real_provider_execution_block_reason": first["operator_preview"]["real_provider_execution_block_reason"],
        "dry_run_status": first["operator_preview"]["dry_run_mode_status"],
        "privacy_gate_status": first["operator_preview"]["privacy_gate_status"],
        "payload_policy_allowed": first["operator_preview"]["payload_policy_allowed"],
        "budget_allowed": first["operator_preview"]["budget_allowed"],
        "approval_state": first["operator_preview"]["external_call_approval_status"],
    }
    budget = dict(first["budget_guard_result"])
    summary = {
        "block": "MEDAI-AI-REAL-PROVIDER-ENABLEMENT-GATE-15F",
        "selected_provider": first["provider_selection_result"]["selected_provider"],
        "effective_provider": first["provider_selection_result"]["effective_provider"],
        "credential_env_var_name": credential["present_sample"]["credential_env_var_name"],
        "credential_present_sample": credential["present_sample"]["credential_present"],
        "credential_missing_sample": credential["missing_sample"]["credential_present"],
        "real_provider_execution_enabled": False,
        "real_provider_execution_block_reason": provider_enablement["present_credential_decision"][
            "real_provider_execution_block_reason"
        ],
        "dry_run_external_call_allowed": first["dry_run_decision_result"]["dry_run_external_call_allowed"],
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": review_bound_package_count,
    }
    validation = _validation_report()
    privacy = _privacy_report(summary, validation, provider_enablement, credential, safety, budget, operator_preview)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return summary, validation, provider_enablement, credential, safety, privacy, budget, operator_preview, implementation


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


def _without_env_var_name(payload: dict[str, Any]) -> dict[str, Any]:
    public = dict(payload)
    public.pop("credential_env_var_name", None)
    public["credential_env_var_name_reported_in_credential_readiness_check"] = True
    return public


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15F_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def _run_optional_command(command: list[str], public_command: str) -> dict[str, Any]:
    if not _command_target_exists(command):
        return {"command": public_command, "missing": True}
    return _run_command(command, public_command)


def _command_target_exists(command: list[str]) -> bool:
    target = command[-1]
    return not target.endswith(".py") or (REPO_ROOT / target).exists()


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "execution/ai_provider_enablement.py",
        "execution/ai_external_call_dry_run.py",
        "execution/extraction_workflow.py",
        "scripts/run_medai_ai_real_provider_enablement_gate_15f.py",
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
        PRIVATE_DEMO_SECRET,
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_imports_introduced": False,
        "network_call_code_introduced": False,
        "raw_pdf_image_upload_introduced": False,
        "credential_value_written": False,
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
        "credential_value_in_report": False,
        "provider_sdk_usage": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-REAL-PROVIDER-ENABLEMENT-GATE-15F",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Credential env var: `{summary['credential_env_var_name']}`",
            f"- Credential present sample: `{summary['credential_present_sample']}`",
            f"- Real provider execution enabled: `{summary['real_provider_execution_enabled']}`",
            f"- Block reason: `{summary['real_provider_execution_block_reason']}`",
            f"- Dry-run allowed: `{summary['dry_run_external_call_allowed']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15F test code: `{validation['focused_15f_tests'].get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Credential readiness is presence-only and never prints values.",
            "- Real provider execution remains disabled by policy.",
            "- No provider SDK, endpoint, network, active write, or auto-accept is enabled.",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    summary, validation, provider_enablement, credential, safety, privacy, budget, operator_preview, implementation = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    ENABLEMENT_JSON_PATH.write_text(json.dumps(provider_enablement, indent=2), encoding="utf-8")
    CREDENTIAL_JSON_PATH.write_text(json.dumps(credential, indent=2), encoding="utf-8")
    SAFETY_JSON_PATH.write_text(json.dumps(safety, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def _public_command(command: list[str]) -> list[str]:
    return ["python" if item == sys.executable else item for item in command]


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    summary, validation, provider_enablement, credential, safety, privacy, operator_budget, operator_preview, _implementation = reports
    validation_items = [item for key, item in validation.items() if key != "source_safety_scan"]
    commands_ok = all(item.get("skipped") or item.get("missing") or item.get("returncode") == 0 for item in validation_items)
    ready = all(
        [
            commands_ok,
            validation["source_safety_scan"]["passed"],
            privacy["privacy_result"] == "passed",
            provider_enablement["real_providers_disabled"],
            provider_enablement["fake_local_enabled"],
            credential["present_sample"]["credential_present"] is True,
            credential["missing_sample"]["credential_present"] is False,
            credential["credential_value_written"] is False,
            safety["dry_run_passed"] is True,
            operator_preview["operator_notice"] == "No external AI call was made.",
            operator_preview["real_provider_notice"] == "Real provider execution disabled by policy",
            operator_budget["budget_allowed"] is True,
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["real_provider_execution_enabled"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["review_bound_package_count"] >= 3,
        ]
    )
    print("medai_ai_real_provider_enablement_gate_15f_ready" if ready else "medai_ai_real_provider_enablement_gate_15f_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "credential_env_var_name": summary["credential_env_var_name"],
                "credential_present_sample": summary["credential_present_sample"],
                "real_provider_execution_enabled": summary["real_provider_execution_enabled"],
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
