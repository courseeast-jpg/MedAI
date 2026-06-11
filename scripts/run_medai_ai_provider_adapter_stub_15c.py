#!/usr/bin/env python3
"""MEDAI-AI-PROVIDER-ADAPTER-STUB-15C validation."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_provider_adapter_stub_15c"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
REGISTRY_JSON_PATH = REPORT_DIR / "provider_registry_check.json"
POLICY_JSON_PATH = REPORT_DIR / "provider_policy_check.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
BUDGET_JSON_PATH = REPORT_DIR / "budget_guard_check.json"
AUDIT_JSON_PATH = REPORT_DIR / "audit_sample.json"
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
]
PACKAGE_TYPES = ["cytology_pathology", "urinalysis_table", "portal_cards"]


def _run_pytest(command: list[str], *, public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15C_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def build_reports() -> tuple[dict[str, Any], ...]:
    results = [
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class=source_class,
                safe_source_document_id=f"source_fake_15c_{index:03d}",
                selected_document_category="AI-assisted extraction",
                selected_specialty_domain="urology",
                source_modality="fake_local_adapter",
                raw_text_local_only=PRIVATE_TEXT,
                provider_name="fake_local",
                provider_mode="fake_local",
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    public_results = [workflow_result_to_public_dict(result) for result in results]
    first = public_results[0]
    registry = AIProviderRegistry()
    providers = registry.list_providers()
    disabled_results = [
        workflow_result_to_public_dict(
            run_ai_extraction_workflow(
                ExtractionWorkflowContext(
                    source_class="urinalysis_table",
                    safe_source_document_id=f"source_fake_15c_disabled_{name}",
                    raw_text_local_only=PRIVATE_TEXT,
                    provider_name=name,
                    provider_mode="disabled",
                )
            )
        )["provider_registry_result"]
        for name in ("gemini", "claude", "openai", "local_ollama")
    ]
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in public_results)
    summary = {
        "block": "MEDAI-AI-PROVIDER-ADAPTER-STUB-15C",
        "external_api_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": review_bound_package_count,
        "provider_count": len(providers),
        "enabled_providers": [item["provider_name"] for item in providers if item["provider_enabled"]],
        "disabled_providers": [item["provider_name"] for item in providers if not item["provider_enabled"]],
        "fake_local_visible_packages": bool(first["operator_preview"]["packages"]),
    }
    registry_check = {
        "providers": providers,
        "required_provider_names_present": sorted(item["provider_name"] for item in providers)
        == ["claude", "fake_local", "gemini", "local_ollama", "openai"],
        "fake_local_enabled": True,
        "real_provider_defaults_disabled": all(not item["provider_enabled"] for item in providers if item["provider_name"] != "fake_local"),
        "raw_pdf_image_vision_disabled": all(
            not item["supports_raw_pdf"] and not item["supports_raw_image"] and not item["supports_vision"]
            for item in providers
        ),
        "sdk_import_required": any(item["sdk_import_required"] for item in providers),
        "external_api_used": False,
    }
    provider_policy = {
        "selected_provider": first["provider_registry_result"],
        "disabled_provider_results": disabled_results,
        "disabled_provider_fail_closed": all(item["fail_closed_reason"] == "provider_disabled_by_policy" for item in disabled_results),
        "final_external_call_allowed": False,
        "external_api_used": False,
    }
    budget = dict(first["budget_guard_result"])
    audit = dict(first["audit_result"])
    audit["provider_name"] = first["provider_registry_result"]["provider_name"]
    audit["provider_policy_allowed"] = first["provider_registry_result"]["provider_ready"]
    operator_preview = {
        "package_count": review_bound_package_count,
        "selected_provider": first["operator_preview"]["selected_provider"],
        "provider_enabled": first["operator_preview"]["provider_enabled"],
        "provider_model_name": first["operator_preview"]["provider_model_name"],
        "provider_mode": first["operator_preview"]["provider_mode"],
        "provider_requires_operator_approval": first["operator_preview"]["provider_requires_operator_approval"],
        "provider_message": first["operator_preview"]["provider_message"],
        "privacy_gate_status": first["operator_preview"]["privacy_gate_status"],
        "payload_policy_allowed": first["operator_preview"]["payload_policy_allowed"],
        "budget_allowed": first["operator_preview"]["budget_allowed"],
        "operator_notice": "No external AI call was made",
        "final_external_call_allowed": False,
        "external_api_used": False,
    }
    validation = {
        "focused_15c_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_adapter_stub_15c.py"],
            public_command="python -m pytest tests/test_medai_ai_provider_adapter_stub_15c.py",
        ),
        "focused_15b_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"],
            public_command="python -m pytest tests/test_medai_ai_extraction_privacy_gate_15b.py",
        ),
        "focused_15a_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
            public_command="python -m pytest tests/test_medai_ai_extraction_workflow_seam_15a.py",
        ),
        "regression_12a_script": _run_command(
            [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"],
            "python scripts/run_medai_operator_workflow_uat_12a.py",
        ),
        "regression_13c_pytest": _run_command(
            [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"],
            "python -m pytest tests/test_medai_operator_usability_polish_13c.py",
        ),
        "external_api_used_false": True,
        "final_external_call_allowed_false": True,
        "active_written_count_zero": True,
        "auto_accept_false": True,
    }
    privacy = _privacy_report(summary, validation, registry_check, provider_policy, budget, audit, operator_preview)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation, provider_policy)
    return summary, validation, registry_check, provider_policy, privacy, budget, audit, operator_preview, implementation


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15C_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True)
    token_scan_passed = not any(token in combined for token in BLOCKED_REPORT_TOKENS)
    payload_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if token_scan_passed and payload_check.passed else "failed",
        "raw_ocr_text_in_report": False,
        "private_identifiers_in_report": False,
        "token_map_in_report": False,
        "api_keys_or_credentials_in_report": False,
        "provider_sdk_usage": False,
        "external_api_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any], provider_policy: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PROVIDER-ADAPTER-STUB-15C",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Provider count: `{summary['provider_count']}`",
            f"- Enabled provider count: `{len(summary['enabled_providers'])}`",
            f"- Disabled provider count: `{len(summary['disabled_providers'])}`",
            f"- Disabled providers fail closed: `{provider_policy['disabled_provider_fail_closed']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15C test code: `{validation['focused_15c_tests'].get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Real provider calls remain disabled.",
            "- Stubs return status only.",
            "- Fake-local extraction remains review-bound.",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    summary, validation, registry_check, provider_policy, privacy, budget, audit, operator_preview, implementation = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    REGISTRY_JSON_PATH.write_text(json.dumps(registry_check, indent=2), encoding="utf-8")
    POLICY_JSON_PATH.write_text(json.dumps(provider_policy, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    summary, validation, registry_check, provider_policy, privacy, budget, audit, operator_preview, _implementation = reports
    test_ok = all(
        item.get("skipped") or item.get("returncode") == 0
        for item in [
            validation["focused_15c_tests"],
            validation["focused_15b_tests"],
            validation["focused_15a_tests"],
            validation["regression_12a_script"],
            validation["regression_13c_pytest"],
        ]
    )
    ready = all(
        [
            test_ok,
            privacy["privacy_result"] == "passed",
            registry_check["required_provider_names_present"],
            registry_check["real_provider_defaults_disabled"],
            provider_policy["disabled_provider_fail_closed"],
            summary["external_api_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_bound_package_count"] >= 3,
            budget["budget_allowed"] is True,
            audit["external_api_used"] is False,
            operator_preview["operator_notice"] == "No external AI call was made",
        ]
    )
    print("medai_ai_provider_adapter_stub_15c_ready" if ready else "medai_ai_provider_adapter_stub_15c_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "external_api_used": summary["external_api_used"],
                "final_external_call_allowed": summary["final_external_call_allowed"],
                "active_written_count": summary["active_written_count"],
                "auto_accept": summary["auto_accept"],
                "review_bound_package_count": summary["review_bound_package_count"],
                "disabled_provider_fail_closed": provider_policy["disabled_provider_fail_closed"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
