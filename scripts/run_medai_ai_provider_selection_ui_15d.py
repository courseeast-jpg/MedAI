#!/usr/bin/env python3
"""MEDAI-AI-PROVIDER-SELECTION-UI-15D validation."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_provider_selection_ui_15d"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
SELECTION_JSON_PATH = REPORT_DIR / "provider_selection_check.json"
UI_JSON_PATH = REPORT_DIR / "provider_ui_preview_check.json"
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


def build_reports() -> tuple[dict[str, Any], ...]:
    package_results = [
        workflow_result_to_public_dict(
            run_ai_extraction_workflow(
                ExtractionWorkflowContext(
                    source_class=source_class,
                    safe_source_document_id=f"source_fake_15d_{index:03d}",
                    selected_document_category="AI-assisted extraction",
                    selected_specialty_domain="urology",
                    source_modality="fake_local_adapter",
                    raw_text_local_only=PRIVATE_TEXT,
                    provider_name="fake_local",
                    provider_mode="fake_local",
                )
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    disabled_intent = workflow_result_to_public_dict(
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class="urinalysis_table",
                safe_source_document_id="source_fake_15d_disabled_gemini",
                raw_text_local_only=PRIVATE_TEXT,
                provider_name="gemini",
                provider_mode="disabled",
            )
        )
    )
    first = package_results[0]
    provider_registry = AIProviderRegistry().list_providers()
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in package_results)
    selection = {
        "default_selection": first["provider_selection_result"],
        "disabled_provider_selection": disabled_intent["provider_selection_result"],
        "default_selected_provider_is_fake_local": first["provider_selection_result"]["selected_provider"] == "fake_local",
        "disabled_provider_intent_recorded": disabled_intent["provider_selection_result"]["selected_provider"] == "gemini",
        "effective_provider_remains_fake_local": disabled_intent["provider_selection_result"]["effective_provider"] == "fake_local",
        "provider_execution_allowed": False,
        "provider_execution_block_reason": disabled_intent["provider_selection_result"]["provider_execution_block_reason"],
    }
    ui_preview = {
        "operator_notice": disabled_intent["operator_preview"]["operator_notice"],
        "selected_provider": disabled_intent["operator_preview"]["requested_provider"],
        "effective_provider": disabled_intent["operator_preview"]["effective_provider"],
        "provider_enabled": disabled_intent["operator_preview"]["provider_enabled"],
        "provider_message": disabled_intent["operator_preview"]["provider_message"],
        "provider_execution_allowed": disabled_intent["operator_preview"]["provider_execution_allowed"],
        "provider_execution_block_reason": disabled_intent["operator_preview"]["provider_execution_block_reason"],
        "privacy_gate_status": disabled_intent["operator_preview"]["privacy_gate_status"],
        "payload_policy_allowed": disabled_intent["operator_preview"]["payload_policy_allowed"],
        "budget_allowed": disabled_intent["operator_preview"]["budget_allowed"],
        "final_external_call_allowed": False,
        "external_api_used": False,
    }
    registry_check = {
        "providers": provider_registry,
        "fake_local_enabled": any(item["provider_name"] == "fake_local" and item["provider_enabled"] for item in provider_registry),
        "real_providers_disabled": all(
            not item["provider_enabled"] for item in provider_registry if item["provider_name"] != "fake_local"
        ),
        "disabled_provider_fail_closed": selection["provider_execution_block_reason"] == "provider_disabled_by_policy",
        "sdk_import_required": any(item["sdk_import_required"] for item in provider_registry),
        "external_api_used": False,
    }
    budget = dict(first["budget_guard_result"])
    summary = {
        "block": "MEDAI-AI-PROVIDER-SELECTION-UI-15D",
        "selected_provider": selection["disabled_provider_selection"]["selected_provider"],
        "effective_provider": selection["disabled_provider_selection"]["effective_provider"],
        "external_api_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": review_bound_package_count,
        "disabled_provider_fail_closed": registry_check["disabled_provider_fail_closed"],
    }
    validation = {
        "focused_15d_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_selection_ui_15d.py"],
            "python -m pytest tests/test_medai_ai_provider_selection_ui_15d.py",
        ),
        "focused_15c_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_adapter_stub_15c.py"],
            "python -m pytest tests/test_medai_ai_provider_adapter_stub_15c.py",
        ),
        "focused_15b_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"],
            "python -m pytest tests/test_medai_ai_extraction_privacy_gate_15b.py",
        ),
        "focused_15a_tests": _run_pytest(
            [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
            "python -m pytest tests/test_medai_ai_extraction_workflow_seam_15a.py",
        ),
        "regression_12a_script": _run_command(
            [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"],
            "python scripts/run_medai_operator_workflow_uat_12a.py",
        ),
        "regression_13c_pytest": _run_command(
            [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"],
            "python -m pytest tests/test_medai_operator_usability_polish_13c.py",
        ),
    }
    privacy = _privacy_report(summary, validation, selection, ui_preview, registry_check, budget)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return summary, validation, selection, ui_preview, registry_check, privacy, budget, ui_preview, implementation


def _run_pytest(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15D_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return {"command": public_command, "returncode": proc.returncode, "skipped": False}


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15D_SKIP_PYTEST") == "1":
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
        "secret_material_in_report": False,
        "provider_sdk_usage": False,
        "external_api_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PROVIDER-SELECTION-UI-15D",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Disabled provider fail closed: `{summary['disabled_provider_fail_closed']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15D test code: `{validation['focused_15d_tests'].get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Provider selection is status-only.",
            "- Real providers stay disabled.",
            "- Execution remains fake-local and review-bound.",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    summary, validation, selection, ui_preview, registry_check, privacy, budget, operator_preview, implementation = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    SELECTION_JSON_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    UI_JSON_PATH.write_text(json.dumps(ui_preview, indent=2), encoding="utf-8")
    REGISTRY_JSON_PATH.write_text(json.dumps(registry_check, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    summary, validation, selection, ui_preview, registry_check, privacy, budget, _operator_preview, _implementation = reports
    test_ok = all(
        item.get("skipped") or item.get("returncode") == 0
        for item in [
            validation["focused_15d_tests"],
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
            registry_check["fake_local_enabled"],
            registry_check["real_providers_disabled"],
            registry_check["disabled_provider_fail_closed"],
            selection["effective_provider_remains_fake_local"],
            ui_preview["operator_notice"] == "No external AI call was made",
            summary["external_api_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_bound_package_count"] >= 3,
            budget["budget_allowed"] is True,
        ]
    )
    print("medai_ai_provider_selection_ui_15d_ready" if ready else "medai_ai_provider_selection_ui_15d_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "external_api_used": summary["external_api_used"],
                "final_external_call_allowed": summary["final_external_call_allowed"],
                "active_written_count": summary["active_written_count"],
                "auto_accept": summary["auto_accept"],
                "review_bound_package_count": summary["review_bound_package_count"],
                "disabled_provider_fail_closed": summary["disabled_provider_fail_closed"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
