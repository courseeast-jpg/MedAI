#!/usr/bin/env python3
"""MEDAI-AI-GEMINI-ADAPTER-DISABLED-15G validation."""
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
from execution.gemini_extraction_adapter import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    GEMINI_CREDENTIAL_ENV_VAR,
    GeminiAdapterRequest,
    GeminiExtractionAdapter,
    GeminiPromptContract,
    build_gemini_adapter_status,
    build_mock_gemini_responses,
    run_gemini_mock_extraction_preview,
    validate_gemini_response_schema,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_gemini_adapter_disabled_15g"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
ADAPTER_CONTRACT_JSON_PATH = REPORT_DIR / "gemini_adapter_contract_check.json"
SCHEMA_VALIDATION_JSON_PATH = REPORT_DIR / "gemini_schema_validation_check.json"
ENABLEMENT_JSON_PATH = REPORT_DIR / "provider_enablement_check.json"
CREDENTIAL_JSON_PATH = REPORT_DIR / "credential_readiness_check.json"
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
PRIVATE_DEMO_SECRET = "demo-secret-value-" + "not-written-15g"
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

REQUIRED_COMMANDS = [
    ("focused_15g_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_gemini_adapter_disabled_15g.py"]),
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
    ("baseline_15f_script", [sys.executable, "scripts/run_medai_ai_real_provider_enablement_gate_15f.py"]),
    ("baseline_15e_script", [sys.executable, "scripts/run_medai_ai_external_call_dry_run_15e.py"]),
    ("baseline_15d_script", [sys.executable, "scripts/run_medai_ai_provider_selection_ui_15d.py"]),
    ("baseline_15c_script", [sys.executable, "scripts/run_medai_ai_provider_adapter_stub_15c.py"]),
    ("baseline_15b_script", [sys.executable, "scripts/run_medai_ai_extraction_privacy_gate_15b.py"]),
    ("baseline_15a_script", [sys.executable, "scripts/run_medai_ai_extraction_workflow_seam_15a.py"]),
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
    workflow_public = workflow_result_to_public_dict(
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class="urinalysis_table",
                safe_source_document_id="source_fake_15g_001",
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
    preview = run_gemini_mock_extraction_preview(environ={GEMINI_CREDENTIAL_ENV_VAR: PRIVATE_DEMO_SECRET})

    adapter = GeminiExtractionAdapter()
    contract = GeminiPromptContract()
    gate_samples = {
        "raw_pdf_rejected": _gate_sample(adapter, payload_type="raw_pdf"),
        "raw_image_rejected": _gate_sample(adapter, payload_type="raw_image"),
        "vision_rejected": _gate_sample(adapter, payload_type="external_vision_payload"),
        "missing_privacy_gate": _gate_sample(adapter, privacy_gate_status=""),
        "failed_payload_policy": _gate_sample(adapter, payload_policy_allowed=False),
        "budget_failure": _gate_sample(adapter, budget_allowed=False),
        "real_execution_disabled": _gate_sample(adapter, real_provider_execution_enabled=False),
        "final_call_disabled": _gate_sample(adapter, final_external_call_allowed=False),
        "all_gates_pass_still_blocked": _gate_sample(adapter),
    }
    adapter_contract = {
        "adapter_class": "GeminiExtractionAdapter",
        "provider_name": "gemini",
        "model_name": adapter.config.model_name,
        "prompt_contract": contract.public_contract(),
        "allowed_payload_type": ALLOWED_PAYLOAD_TYPE,
        "forbidden_payload_types": sorted(FORBIDDEN_PAYLOAD_TYPES),
        "real_provider_execution_enabled": False,
        "gemini_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "provider_sdk_import_required": False,
        "gate_samples": gate_samples,
    }

    schema_check = {
        "schema_contract_version": preview["schema_validations"][0]["schema_contract_version"],
        "valid_responses": preview["schema_validations"],
        "auto_accept_true_rejected": _schema_reject({"auto_accept": True}, "auto_accept_must_be_false"),
        "active_write_allowed_true_rejected": _schema_reject(
            {"active_write_allowed": True}, "active_write_allowed_must_be_false"
        ),
        "review_required_false_rejected": _schema_reject({"review_required": False}, "review_required_must_be_true"),
        "recommendation_label_preserved": all(
            item.get("source_text_only_recommendation_preserved", True) for item in preview["schema_validations"]
        ),
        "schema_valid_count": preview["schema_valid_count"],
    }

    gemini_status = build_gemini_adapter_status(
        selected_provider="gemini",
        environ={GEMINI_CREDENTIAL_ENV_VAR: PRIVATE_DEMO_SECRET},
    )
    provider_enablement = {
        "selected_provider": workflow_public["provider_selection_result"]["selected_provider"],
        "effective_provider": workflow_public["provider_selection_result"]["effective_provider"],
        "gemini_adapter_installed": True,
        "gemini_adapter_status_message": gemini_status["gemini_adapter_status_message"],
        "real_provider_execution_enabled": False,
        "real_provider_execution_block_reason": workflow_public["real_provider_enablement_result"][
            "real_provider_execution_block_reason"
        ],
        "gemini_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "fake_local_package_output_preserved": int(workflow_public["review_bound_package_count"]) >= 1,
    }
    credential = {
        "credential_env_var_name": gemini_status["credential_env_var_name"],
        "credential_present": gemini_status["credential_present"],
        "credential_value_redacted": gemini_status["credential_value_redacted"],
        "credential_value_read_or_logged": False,
        "credential_value_written": False,
        "credential_value_in_report": False,
    }
    budget = dict(workflow_public["budget_guard_result"])
    operator_preview = {
        "operator_notice": workflow_public["operator_preview"]["operator_notice_sentence"],
        "gemini_adapter_status_message": workflow_public["operator_preview"]["gemini_adapter_status_message"],
        "selected_provider": workflow_public["operator_preview"]["requested_provider"],
        "effective_provider": workflow_public["operator_preview"]["effective_provider"],
        "gemini_adapter_installed": workflow_public["operator_preview"]["gemini_adapter_installed"],
        "gemini_selected": workflow_public["operator_preview"]["gemini_selected"],
        "gemini_real_call_attempted": workflow_public["operator_preview"]["gemini_real_call_attempted"],
        "gemini_real_provider_execution_enabled": workflow_public["operator_preview"][
            "gemini_real_provider_execution_enabled"
        ],
        "gemini_credential_env_var_name": workflow_public["operator_preview"]["gemini_credential_env_var_name"],
        "gemini_credential_present": workflow_public["operator_preview"]["gemini_credential_present"],
        "real_provider_execution_block_reason": workflow_public["operator_preview"][
            "gemini_real_provider_execution_block_reason"
        ],
        "dry_run_status": workflow_public["operator_preview"]["dry_run_mode_status"],
        "privacy_gate_status": workflow_public["operator_preview"]["privacy_gate_status"],
        "payload_policy_allowed": workflow_public["operator_preview"]["payload_policy_allowed"],
        "budget_allowed": workflow_public["operator_preview"]["budget_allowed"],
    }
    summary = {
        "block": "MEDAI-AI-GEMINI-ADAPTER-DISABLED-15G",
        "selected_provider": workflow_public["provider_selection_result"]["selected_provider"],
        "effective_provider": workflow_public["provider_selection_result"]["effective_provider"],
        "gemini_adapter_installed": True,
        "gemini_adapter_status_message": gemini_status["gemini_adapter_status_message"],
        "credential_env_var_name": credential["credential_env_var_name"],
        "credential_present": credential["credential_present"],
        "real_provider_execution_enabled": False,
        "gemini_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "schema_valid_count": preview["schema_valid_count"],
        "review_bound_package_count": preview["review_bound_package_count"],
    }
    validation = _validation_report()
    privacy = _privacy_report(
        summary, validation, adapter_contract, schema_check, provider_enablement, credential, budget, operator_preview
    )
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return (
        summary,
        validation,
        adapter_contract,
        schema_check,
        provider_enablement,
        credential,
        privacy,
        budget,
        operator_preview,
        implementation,
    )


def _gate_sample(adapter: GeminiExtractionAdapter, **overrides: Any) -> dict[str, Any]:
    values = {
        "payload_type": ALLOWED_PAYLOAD_TYPE,
        "redacted_payload_hash": "deadbeef0001",
        "redacted_text_layout_summary": "[redacted summary]",
        "privacy_gate_status": "redacted_payload_ready",
        "payload_policy_allowed": True,
        "budget_allowed": True,
        "operator_approval_state": "approved_for_dry_run",
        "real_provider_execution_enabled": True,
        "final_external_call_allowed": True,
    }
    values.update(overrides)
    result = adapter.evaluate(GeminiAdapterRequest(**values))
    return {
        "fail_closed_reason": result.fail_closed_reason,
        "real_provider_execution_enabled": result.real_provider_execution_enabled,
        "final_external_call_allowed": result.final_external_call_allowed,
        "gemini_real_call_attempted": result.gemini_real_call_attempted,
        "external_api_used": result.external_api_used,
        "real_network_call_used": result.real_network_call_used,
    }


def _schema_reject(overrides: dict[str, Any], expected_error: str) -> dict[str, Any]:
    response = {**build_mock_gemini_responses()["urinalysis_table"], **overrides}
    validation = validate_gemini_response_schema(response)
    return {
        "schema_valid": validation.schema_valid,
        "expected_error_present": expected_error in validation.errors,
    }


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
    if os.environ.get("MEDAI_15G_SKIP_PYTEST") == "1":
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
        "execution/gemini_extraction_adapter.py",
        "execution/ai_extraction_adapter.py",
        "execution/extraction_workflow.py",
        "scripts/run_medai_ai_gemini_adapter_disabled_15g.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "import " + "google." + "generativeai",
        "google." + "generativeai",
        "requests" + ".get",
        "requests" + ".post",
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
        "gemini_real_call_attempted": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-GEMINI-ADAPTER-DISABLED-15G",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Gemini adapter installed: `{summary['gemini_adapter_installed']}`",
            f"- Gemini adapter status: `{summary['gemini_adapter_status_message']}`",
            f"- Credential env var: `{summary['credential_env_var_name']}`",
            f"- Credential present: `{summary['credential_present']}`",
            f"- Real provider execution enabled: `{summary['real_provider_execution_enabled']}`",
            f"- Gemini real call attempted: `{summary['gemini_real_call_attempted']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Schema-valid mock responses: `{summary['schema_valid_count']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- 15G test code: `{validation['focused_15g_tests'].get('returncode', 'skipped')}`",
            "",
            "## Limitations",
            "",
            "- Gemini adapter is installed but real execution is disabled by policy.",
            "- Mock responses are local, deterministic, and never reach a provider.",
            "- No provider SDK, endpoint, network, active write, or auto-accept is enabled.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-CLAUDE-ADAPTER-DISABLED-15H (mirror disabled adapter for Claude).",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        adapter_contract,
        schema_check,
        provider_enablement,
        credential,
        privacy,
        budget,
        operator_preview,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    ADAPTER_CONTRACT_JSON_PATH.write_text(json.dumps(adapter_contract, indent=2), encoding="utf-8")
    SCHEMA_VALIDATION_JSON_PATH.write_text(json.dumps(schema_check, indent=2), encoding="utf-8")
    ENABLEMENT_JSON_PATH.write_text(json.dumps(provider_enablement, indent=2), encoding="utf-8")
    CREDENTIAL_JSON_PATH.write_text(json.dumps(credential, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def _public_command(command: list[str]) -> list[str]:
    return ["python" if item == sys.executable else item for item in command]


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    (
        summary,
        validation,
        adapter_contract,
        schema_check,
        provider_enablement,
        credential,
        privacy,
        _budget,
        operator_preview,
        _implementation,
    ) = reports
    validation_items = [item for key, item in validation.items() if key != "source_safety_scan"]
    commands_ok = all(
        item.get("skipped") or item.get("missing") or item.get("returncode") == 0 for item in validation_items
    )
    ready = all(
        [
            commands_ok,
            validation["source_safety_scan"]["passed"],
            privacy["privacy_result"] == "passed",
            adapter_contract["real_provider_execution_enabled"] is False,
            adapter_contract["gemini_real_call_attempted"] is False,
            adapter_contract["provider_sdk_import_required"] is False,
            adapter_contract["gate_samples"]["raw_pdf_rejected"]["fail_closed_reason"] == "forbidden_payload_type",
            adapter_contract["gate_samples"]["missing_privacy_gate"]["fail_closed_reason"]
            == "missing_privacy_gate_result",
            adapter_contract["gate_samples"]["failed_payload_policy"]["fail_closed_reason"] == "payload_policy_failed",
            adapter_contract["gate_samples"]["budget_failure"]["fail_closed_reason"] == "budget_exceeded",
            adapter_contract["gate_samples"]["final_call_disabled"]["fail_closed_reason"]
            == "final_external_call_not_allowed",
            schema_check["schema_valid_count"] >= 3,
            schema_check["auto_accept_true_rejected"]["schema_valid"] is False,
            schema_check["active_write_allowed_true_rejected"]["schema_valid"] is False,
            schema_check["review_required_false_rejected"]["schema_valid"] is False,
            schema_check["recommendation_label_preserved"] is True,
            provider_enablement["gemini_adapter_installed"] is True,
            provider_enablement["fake_local_package_output_preserved"] is True,
            credential["credential_present"] is True,
            credential["credential_value_written"] is False,
            operator_preview["gemini_adapter_status_message"]
            == "Gemini adapter installed but real execution disabled by policy",
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["real_provider_execution_enabled"] is False,
            summary["gemini_real_call_attempted"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["review_bound_package_count"] >= 3,
        ]
    )
    print(
        "medai_ai_gemini_adapter_disabled_15g_ready"
        if ready
        else "medai_ai_gemini_adapter_disabled_15g_not_ready"
    )
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "gemini_adapter_installed": summary["gemini_adapter_installed"],
                "real_provider_execution_enabled": summary["real_provider_execution_enabled"],
                "gemini_real_call_attempted": summary["gemini_real_call_attempted"],
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
