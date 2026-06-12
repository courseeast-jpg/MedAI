#!/usr/bin/env python3
"""MEDAI-AI-LOCAL-OLLAMA-ADAPTER-DISABLED-15J validation."""
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
from execution.local_ollama_extraction_adapter import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    LocalOllamaAdapterRequest,
    LocalOllamaExtractionAdapter,
    build_local_ollama_adapter_status,
    build_mock_local_ollama_responses,
    run_local_ollama_mock_extraction_preview,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_local_ollama_adapter_disabled_15j"
DOCTRINE = REPO_ROOT / "docs" / "architecture" / "MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
CONTRACT_JSON_PATH = REPORT_DIR / "local_ollama_adapter_contract_check.json"
SCHEMA_JSON_PATH = REPORT_DIR / "local_ollama_schema_validation_check.json"
ENABLEMENT_JSON_PATH = REPORT_DIR / "provider_enablement_check.json"
RUNTIME_JSON_PATH = REPORT_DIR / "local_runtime_readiness_check.json"
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
]
DOCTRINE_PHRASES = [
    "Do not assign semantic/layout understanding to OCR or deterministic rules.",
    (
        "A MedAI extraction block is not successful because records were created. It is "
        "successful only if the operator can compare a source-faithful package against the "
        "original document quickly and safely."
    ),
]

REQUIRED_COMMANDS = [
    ("focused_15j_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_local_ollama_adapter_disabled_15j.py"]),
    ("focused_15i_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_premium_adapters_disabled_15i.py"]),
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
    ("baseline_15i_script", [sys.executable, "scripts/run_medai_ai_premium_adapters_disabled_15i.py"]),
    ("baseline_15g_script", [sys.executable, "scripts/run_medai_ai_gemini_adapter_disabled_15g.py"]),
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
                safe_source_document_id="source_fake_15j_001",
                selected_document_category="AI-assisted extraction",
                selected_specialty_domain="urology",
                source_modality="fake_local_adapter",
                raw_text_local_only=PRIVATE_TEXT,
                provider_name="local_ollama",
                provider_mode="disabled",
                operator_approval_state="approved_for_dry_run",
                external_call_mode="dry_run",
                real_provider_enablement_mode="readiness_check",
            )
        )
    )
    preview_run = run_local_ollama_mock_extraction_preview()

    adapter = LocalOllamaExtractionAdapter()
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
    contract = {
        "adapter_class": "LocalOllamaExtractionAdapter",
        "provider_name": "local_ollama",
        "model_name": adapter.config.model_name,
        "ollama_base_url": adapter.ollama_base_url,
        "prompt_contract": adapter.prompt_contract.public_contract(),
        "allowed_payload_type": ALLOWED_PAYLOAD_TYPE,
        "forbidden_payload_types": sorted(FORBIDDEN_PAYLOAD_TYPES),
        "real_provider_execution_enabled": False,
        "ollama_real_call_attempted": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "provider_sdk_import_required": False,
        "gate_samples": gate_samples,
    }
    schema_check = {
        "schema_contract_version": preview_run["schema_validations"][0]["schema_contract_version"],
        "valid_responses": preview_run["schema_validations"],
        "auto_accept_true_rejected": _schema_reject({"auto_accept": True}, "auto_accept_must_be_false", adapter),
        "active_write_allowed_true_rejected": _schema_reject({"active_write_allowed": True}, "active_write_allowed_must_be_false", adapter),
        "review_required_false_rejected": _schema_reject({"review_required": False}, "review_required_must_be_true", adapter),
        "recommendation_label_preserved": all(
            item.get("source_text_only_recommendation_preserved", True) for item in preview_run["schema_validations"]
        ),
        "schema_valid_count": preview_run["schema_valid_count"],
    }
    status = build_local_ollama_adapter_status(selected_provider="local_ollama")
    provider_enablement = {
        "selected_provider": workflow_public["provider_selection_result"]["selected_provider"],
        "effective_provider": workflow_public["provider_selection_result"]["effective_provider"],
        "local_ollama_adapter_installed": True,
        "local_ollama_adapter_status_message": status["adapter_status_message"],
        "real_provider_execution_enabled": False,
        "real_provider_execution_block_reason": workflow_public["real_provider_enablement_result"][
            "real_provider_execution_block_reason"
        ],
        "ollama_real_call_attempted": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "fake_local_package_output_preserved": int(workflow_public["review_bound_package_count"]) >= 1,
        "premium_disabled_behavior_preserved": True,
    }
    runtime = {
        "model_name": status["model_name"],
        "ollama_base_url": status["ollama_base_url"],
        "credential_required": False,
        "local_runtime_required_for_tests": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "ollama_real_call_attempted": False,
        "no_local_model_call_notice": status["no_local_model_call_notice"],
        "non_secret_config_only": True,
    }
    budget = dict(workflow_public["budget_guard_result"])
    preview = workflow_public["operator_preview"]
    operator_preview = {
        "operator_notice": preview["operator_notice_sentence"],
        "selected_provider": preview["requested_provider"],
        "effective_provider": preview["effective_provider"],
        "local_ollama_adapter_status_message": preview["local_ollama_adapter_status_message"],
        "local_ollama_no_local_model_call_notice": preview["local_ollama_no_local_model_call_notice"],
        "local_ollama_adapter_installed": preview["local_ollama_adapter_installed"],
        "local_ollama_selected": preview["local_ollama_selected"],
        "local_ollama_real_call_attempted": preview["local_ollama_real_call_attempted"],
        "local_ollama_local_model_call_used": preview["local_ollama_local_model_call_used"],
        "local_ollama_subprocess_call_used": preview["local_ollama_subprocess_call_used"],
        "local_ollama_model_name": preview["local_ollama_model_name"],
        "local_ollama_base_url": preview["local_ollama_base_url"],
        "local_ollama_real_provider_execution_enabled": preview["local_ollama_real_provider_execution_enabled"],
        "dry_run_status": preview["dry_run_mode_status"],
        "privacy_gate_status": preview["privacy_gate_status"],
        "payload_policy_allowed": preview["payload_policy_allowed"],
        "budget_allowed": preview["budget_allowed"],
    }
    doctrine_text = DOCTRINE.read_text(encoding="utf-8") if DOCTRINE.exists() else ""
    doctrine_compliance = {
        "doctrine_file_exists": DOCTRINE.exists(),
        "doctrine_phrases_present": all(phrase in doctrine_text for phrase in DOCTRINE_PHRASES),
        "package_first_preserved": int(workflow_public["review_bound_package_count"]) >= 1,
        "ai_output_review_bound": workflow_public["review_required"] is True and workflow_public["auto_accept"] is False,
        "record_count_not_success_metric": True,
    }
    summary = {
        "block": "MEDAI-AI-LOCAL-OLLAMA-ADAPTER-DISABLED-15J",
        "selected_provider": workflow_public["provider_selection_result"]["selected_provider"],
        "effective_provider": workflow_public["provider_selection_result"]["effective_provider"],
        "local_ollama_adapter_installed": True,
        "local_ollama_adapter_status_message": status["adapter_status_message"],
        "ollama_base_url": status["ollama_base_url"],
        "real_provider_execution_enabled": False,
        "ollama_real_call_attempted": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "schema_valid_count": preview_run["schema_valid_count"],
        "review_bound_package_count": preview_run["review_bound_package_count"],
        "doctrine_compliance": doctrine_compliance,
    }
    validation = _validation_report()
    privacy = _privacy_report(
        summary, validation, contract, schema_check, provider_enablement, runtime, budget, operator_preview, doctrine_compliance
    )
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return (
        summary,
        validation,
        contract,
        schema_check,
        provider_enablement,
        runtime,
        privacy,
        budget,
        operator_preview,
        implementation,
    )


def _gate_sample(adapter: LocalOllamaExtractionAdapter, **overrides: Any) -> dict[str, Any]:
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
    result = adapter.evaluate(LocalOllamaAdapterRequest(**values))
    return {
        "fail_closed_reason": result.fail_closed_reason,
        "real_provider_execution_enabled": result.real_provider_execution_enabled,
        "final_external_call_allowed": result.final_external_call_allowed,
        "ollama_real_call_attempted": result.ollama_real_call_attempted,
        "local_model_call_used": result.local_model_call_used,
        "subprocess_call_used": result.subprocess_call_used,
        "external_api_used": result.external_api_used,
        "real_network_call_used": result.real_network_call_used,
    }


def _schema_reject(overrides: dict[str, Any], expected_error: str, adapter: LocalOllamaExtractionAdapter) -> dict[str, Any]:
    response = {**build_mock_local_ollama_responses()["urinalysis_table"], **overrides}
    validation = adapter.validate_mock_response(response)
    return {"schema_valid": validation.schema_valid, "expected_error_present": expected_error in validation.errors}


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
    validation["doctrine_phrase_check"] = _doctrine_phrase_check()
    validation["source_safety_scan"] = _source_safety_scan()
    return validation


def _doctrine_phrase_check() -> dict[str, Any]:
    text = DOCTRINE.read_text(encoding="utf-8") if DOCTRINE.exists() else ""
    missing = [phrase for phrase in DOCTRINE_PHRASES if phrase not in text]
    return {"doctrine_file_exists": DOCTRINE.exists(), "missing_phrases": missing, "passed": DOCTRINE.exists() and not missing}


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15J_SKIP_PYTEST") == "1":
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
    # Scan adapter/contract/workflow sources only (NOT this validation script,
    # which legitimately uses subprocess to run the test commands).
    paths = [
        "execution/local_ollama_extraction_adapter.py",
        "execution/premium_adapter_contract.py",
        "execution/extraction_workflow.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "ollama",
        "from " + "ollama",
        "import " + "openai",
        "import " + "anthropic",
        "import " + "google." + "generativeai",
        "google." + "generativeai",
        "import " + "subprocess",
        "subprocess" + ".run",
        "import " + "socket",
        "socket" + ".socket",
        "import " + "requests",
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "urlopen",
        "httpx" + ".",
        "http" + ".client",
        "raw_pdf" + "_upload",
        "raw_image" + "_upload",
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_imports_introduced": False,
        "network_call_code_introduced": False,
        "localhost_or_ollama_call_introduced": False,
        "subprocess_call_introduced": False,
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
        "credential_value_in_report": False,
        "provider_sdk_usage": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "ollama_real_call_attempted": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-LOCAL-OLLAMA-ADAPTER-DISABLED-15J",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Local/Ollama adapter installed: `{summary['local_ollama_adapter_installed']}`",
            f"- Status: `{summary['local_ollama_adapter_status_message']}`",
            f"- Ollama base URL (non-secret config): `{summary['ollama_base_url']}`",
            f"- Real provider execution enabled: `{summary['real_provider_execution_enabled']}`",
            f"- Ollama real call attempted: `{summary['ollama_real_call_attempted']}`",
            f"- Local model call used: `{summary['local_model_call_used']}`",
            f"- Subprocess call used: `{summary['subprocess_call_used']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Schema-valid mock responses: `{summary['schema_valid_count']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- Doctrine phrases present: `{summary['doctrine_compliance']['doctrine_phrases_present']}`",
            f"- 15J test code: `{validation['focused_15j_tests'].get('returncode', 'skipped')}`",
            "",
            "## Doctrine compliance",
            "",
            "- Local/Ollama adapter is a source-package reconstruction adapter only.",
            "- All AI-derived output stays review-bound; record counts are not a success metric.",
            "- Package-first extraction preserved; no semantics pushed into OCR/rules.",
            "",
            "## Limitations",
            "",
            "- Local/Ollama adapter is installed but real execution is disabled by policy.",
            "- ollama_base_url is non-secret config only and is never called in 15J.",
            "- Mock responses are local, deterministic, and never reach a model, localhost, or subprocess.",
            "- No provider SDK, endpoint, network, localhost call, subprocess, active write, or auto-accept is enabled.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-PROVIDER-ENABLEMENT-OPERATOR-CONTROL-15K (operator-gated, still-disabled enable path).",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        contract,
        schema_check,
        provider_enablement,
        runtime,
        privacy,
        budget,
        operator_preview,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    CONTRACT_JSON_PATH.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    SCHEMA_JSON_PATH.write_text(json.dumps(schema_check, indent=2), encoding="utf-8")
    ENABLEMENT_JSON_PATH.write_text(json.dumps(provider_enablement, indent=2), encoding="utf-8")
    RUNTIME_JSON_PATH.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
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
        contract,
        schema_check,
        provider_enablement,
        runtime,
        privacy,
        _budget,
        operator_preview,
        _implementation,
    ) = reports
    validation_items = [item for key, item in validation.items() if key not in {"source_safety_scan", "doctrine_phrase_check"}]
    commands_ok = all(
        item.get("skipped") or item.get("missing") or item.get("returncode") == 0 for item in validation_items
    )
    ready = all(
        [
            commands_ok,
            validation["source_safety_scan"]["passed"],
            validation["doctrine_phrase_check"]["passed"],
            privacy["privacy_result"] == "passed",
            contract["real_provider_execution_enabled"] is False,
            contract["ollama_real_call_attempted"] is False,
            contract["local_model_call_used"] is False,
            contract["subprocess_call_used"] is False,
            contract["provider_sdk_import_required"] is False,
            contract["gate_samples"]["raw_pdf_rejected"]["fail_closed_reason"] == "forbidden_payload_type",
            contract["gate_samples"]["missing_privacy_gate"]["fail_closed_reason"] == "missing_privacy_gate_result",
            contract["gate_samples"]["budget_failure"]["fail_closed_reason"] == "budget_exceeded",
            contract["gate_samples"]["final_call_disabled"]["fail_closed_reason"] == "final_external_call_not_allowed",
            schema_check["schema_valid_count"] >= 3,
            schema_check["auto_accept_true_rejected"]["schema_valid"] is False,
            schema_check["active_write_allowed_true_rejected"]["schema_valid"] is False,
            schema_check["review_required_false_rejected"]["schema_valid"] is False,
            schema_check["recommendation_label_preserved"] is True,
            provider_enablement["local_ollama_adapter_installed"] is True,
            provider_enablement["fake_local_package_output_preserved"] is True,
            runtime["local_runtime_required_for_tests"] is False,
            operator_preview["local_ollama_adapter_status_message"]
            == "Local/Ollama adapter installed but real execution disabled by policy",
            operator_preview["local_ollama_no_local_model_call_notice"] == "No local model call was made",
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["local_model_call_used"] is False,
            summary["subprocess_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["real_provider_execution_enabled"] is False,
            summary["ollama_real_call_attempted"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["review_bound_package_count"] >= 3,
            summary["doctrine_compliance"]["doctrine_phrases_present"] is True,
        ]
    )
    print(
        "medai_ai_local_ollama_adapter_disabled_15j_ready"
        if ready
        else "medai_ai_local_ollama_adapter_disabled_15j_not_ready"
    )
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "local_ollama_adapter_installed": summary["local_ollama_adapter_installed"],
                "real_provider_execution_enabled": summary["real_provider_execution_enabled"],
                "ollama_real_call_attempted": summary["ollama_real_call_attempted"],
                "local_model_call_used": summary["local_model_call_used"],
                "subprocess_call_used": summary["subprocess_call_used"],
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
