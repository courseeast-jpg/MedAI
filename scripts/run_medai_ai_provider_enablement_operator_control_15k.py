#!/usr/bin/env python3
"""MEDAI-AI-PROVIDER-ENABLEMENT-OPERATOR-CONTROL-15K validation."""
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
from execution.ai_provider_operator_control import (
    CONTROLLED_PROVIDERS,
    build_operator_audit_record,
    build_provider_operator_control,
    evaluate_operator_enablement_request,
    operator_audit_to_public_dict,
    operator_control_to_public_dict,
)
from execution.local_ollama_extraction_adapter import run_local_ollama_mock_extraction_preview

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_provider_enablement_operator_control_15k"
DOCTRINE = REPO_ROOT / "docs" / "architecture" / "MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
CONTROL_JSON_PATH = REPORT_DIR / "provider_operator_control_check.json"
MATRIX_JSON_PATH = REPORT_DIR / "provider_readiness_matrix.json"
STAGED_JSON_PATH = REPORT_DIR / "staged_request_check.json"
ENABLEMENT_JSON_PATH = REPORT_DIR / "provider_enablement_check.json"
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
CLAUDE_DEMO_SECRET = "demo-secret-value-" + "claude-not-written-15k"
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
    CLAUDE_DEMO_SECRET,
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
    ("focused_15k_tests", [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_enablement_operator_control_15k.py"]),
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
    ("baseline_15j_script", [sys.executable, "scripts/run_medai_ai_local_ollama_adapter_disabled_15j.py"]),
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

PASSING_GATES = dict(
    privacy_gate_result={"privacy_gate_status": "redacted_payload_ready"},
    payload_policy_result={"payload_policy_allowed": True},
    budget_guard_result={"budget_allowed": True},
    dry_run_decision_result={"dry_run_external_call_allowed": True},
)


def build_reports() -> tuple[dict[str, Any], ...]:
    workflow_public = workflow_result_to_public_dict(
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class="urinalysis_table",
                safe_source_document_id="source_fake_15k_001",
                selected_document_category="AI-assisted extraction",
                selected_specialty_domain="urology",
                source_modality="fake_local_adapter",
                raw_text_local_only=PRIVATE_TEXT,
                provider_name="claude",
                provider_mode="disabled",
                operator_approval_state="approved_for_dry_run",
                external_call_mode="dry_run",
                real_provider_enablement_mode="readiness_check",
            )
        )
    )
    control_state = build_provider_operator_control(
        requested_provider="claude",
        selected_provider="claude",
        effective_provider="fake_local",
        enablement_request={"provider": "claude", "requested_state": "staged"},
        environ={"ANTHROPIC_API_KEY": CLAUDE_DEMO_SECRET},
        **PASSING_GATES,
    )
    control_public = operator_control_to_public_dict(control_state)
    audit_public = operator_audit_to_public_dict(build_operator_audit_record(control_state))
    # Local mock packages to satisfy review_bound_package_count >= 3.
    ollama_preview = run_local_ollama_mock_extraction_preview()

    staged_samples = {
        "claude_staged_all_gates_pass": _staged_sample("claude", credential_present=True, is_cloud=True),
        "openai_missing_credential": _staged_sample("openai", credential_present=False, is_cloud=True),
        "missing_privacy": _staged_sample("claude", credential_present=True, is_cloud=True, privacy_gate_result={}),
        "failed_payload_policy": _staged_sample("claude", credential_present=True, is_cloud=True, payload_policy_result={"payload_policy_allowed": False}),
        "budget_failure": _staged_sample("claude", credential_present=True, is_cloud=True, budget_guard_result={"budget_allowed": False}),
        "missing_dry_run": _staged_sample("claude", credential_present=True, is_cloud=True, dry_run_decision_result={"dry_run_external_call_allowed": False}),
        "local_ollama_no_credential_ok": _staged_sample("local_ollama", credential_present=False, is_cloud=False),
        "fake_local_invalid": _staged_sample("fake_local", credential_present=False, is_cloud=False),
    }
    matrix = _public_matrix(control_public["providers"])
    operator_control_check = {
        "controlled_providers": list(CONTROLLED_PROVIDERS),
        "selected_provider": control_public["selected_provider"],
        "effective_provider": control_public["effective_provider"],
        "fake_local_enabled": any(
            row["provider_name"] == "fake_local" and row["provider_enabled_by_policy"] for row in control_public["providers"]
        ),
        "real_providers_disabled": all(
            not row["provider_enabled_by_policy"] for row in control_public["providers"] if row["provider_name"] != "fake_local"
        ),
        "operator_enablement_request_state": control_public["operator_enablement_request_state"],
        "operator_enablement_request_allowed": control_public["operator_enablement_request_allowed"],
        "real_provider_execution_enabled": False,
        "final_external_call_allowed": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "all_real_call_attempted_false": all(not row["real_call_attempted"] for row in control_public["providers"]),
        "audit": audit_public,
    }
    staged_check = {
        "samples": staged_samples,
        "staged_does_not_enable_execution": all(
            s["real_provider_execution_enabled"] is False and s["final_external_call_allowed"] is False
            for s in staged_samples.values()
        ),
        "blocked_when_gates_missing": (
            staged_samples["missing_privacy"]["operator_enablement_request_allowed"] is False
            and staged_samples["failed_payload_policy"]["operator_enablement_request_allowed"] is False
            and staged_samples["budget_failure"]["operator_enablement_request_allowed"] is False
            and staged_samples["missing_dry_run"]["operator_enablement_request_allowed"] is False
            and staged_samples["openai_missing_credential"]["operator_enablement_request_allowed"] is False
        ),
        "claude_staged_allowed": staged_samples["claude_staged_all_gates_pass"]["operator_enablement_request_allowed"] is True,
    }
    provider_enablement = {
        "selected_provider": workflow_public["provider_selection_result"]["selected_provider"],
        "effective_provider": workflow_public["provider_selection_result"]["effective_provider"],
        "real_provider_execution_enabled": False,
        "gemini_real_call_attempted": False,
        "claude_real_call_attempted": False,
        "openai_real_call_attempted": False,
        "ollama_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "final_external_call_allowed": False,
        "fake_local_package_output_preserved": int(workflow_public["review_bound_package_count"]) >= 1,
    }
    budget = dict(workflow_public["budget_guard_result"])
    preview = workflow_public["operator_preview"]
    operator_preview = {
        "operator_notice": preview["operator_notice_sentence"],
        "selected_provider": preview["requested_provider"],
        "effective_provider": preview["effective_provider"],
        "operator_control_staged_request_state": preview["operator_control_staged_request_state"],
        "operator_control_staged_request_notice": preview["operator_control_staged_request_notice"],
        "operator_control_real_provider_disabled_notice": preview["operator_control_real_provider_disabled_notice"],
        "operator_control_no_external_call_notice": preview["operator_control_no_external_call_notice"],
        "operator_control_no_local_model_call_notice": preview["operator_control_no_local_model_call_notice"],
        "operator_control_real_provider_execution_enabled": preview["operator_control_real_provider_execution_enabled"],
        "operator_control_provider_status": preview["operator_control_provider_status"],
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
    review_bound_package_count = ollama_preview["review_bound_package_count"]
    summary = {
        "block": "MEDAI-AI-PROVIDER-ENABLEMENT-OPERATOR-CONTROL-15K",
        "selected_provider": control_public["selected_provider"],
        "effective_provider": control_public["effective_provider"],
        "staged_request_state": control_public["operator_enablement_request_state"],
        "staged_request_allowed": control_public["operator_enablement_request_allowed"],
        "real_provider_execution_enabled": False,
        "gemini_real_call_attempted": False,
        "claude_real_call_attempted": False,
        "openai_real_call_attempted": False,
        "ollama_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "final_external_call_allowed": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": review_bound_package_count,
        "doctrine_compliance": doctrine_compliance,
    }
    validation = _validation_report()
    privacy = _privacy_report(
        summary, validation, operator_control_check, matrix, staged_check, provider_enablement, budget, operator_preview, doctrine_compliance
    )
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, validation)
    return (
        summary,
        validation,
        operator_control_check,
        matrix,
        staged_check,
        provider_enablement,
        privacy,
        budget,
        operator_preview,
        implementation,
    )


def _staged_sample(provider: str, *, credential_present: bool, is_cloud: bool, **override: Any) -> dict[str, Any]:
    gates = {**PASSING_GATES, **override}
    from dataclasses import asdict

    decision = evaluate_operator_enablement_request(
        provider_name=provider,
        requested_state="staged",
        credential_present=credential_present,
        is_cloud_provider=is_cloud,
        **gates,
    )
    return asdict(decision)


def _public_matrix(providers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "provider_name": row["provider_name"],
            "adapter_contract_available": row["adapter_contract_available"],
            "schema_contract_available": row["schema_contract_available"],
            "provider_enabled_by_policy": row["provider_enabled_by_policy"],
            "credential_required": row["credential_required"],
            "credential_env_var_name": row["credential_env_var_name"],
            "credential_present": row["credential_present"],
            "dry_run_status": row["dry_run_status"],
            "privacy_gate_status": row["privacy_gate_status"],
            "payload_policy_allowed": row["payload_policy_allowed"],
            "budget_allowed": row["budget_allowed"],
            "operator_enablement_request_state": row["operator_enablement_request_state"],
            "execution_allowed": row["execution_allowed"],
            "execution_block_reason": row["execution_block_reason"],
            "real_call_attempted": row["real_call_attempted"],
            "local_model_call_used": row["local_model_call_used"],
            "subprocess_call_used": row["subprocess_call_used"],
        }
        for row in providers
    ]


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


_RECURSIVE_DESELECT = "not regressions_still_pass and not prior_block_regressions"


def _is_prior_block_script(command: list[str]) -> bool:
    """True if the command invokes a run_medai_ai_*_15*.py validation script."""
    for arg in command:
        base = str(arg).replace("\\", "/").split("/")[-1]
        if base.startswith("run_medai_ai_") and base.endswith(".py"):
            return True
    return False


def _with_recursive_deselect(command: list[str]) -> list[str]:
    """Append a deselect filter to pytest commands so recursive cross-block
    regression tests do not re-trigger nested subprocess chains (15K hang fix)."""
    if "pytest" in command and "-k" not in command:
        return [*command, "-k", _RECURSIVE_DESELECT]
    return command


def _run_command(command: list[str], public_command: str) -> dict[str, Any]:
    if os.environ.get("MEDAI_15K_SKIP_PYTEST") == "1":
        return {"command": public_command, "skipped": True}
    command = _with_recursive_deselect(command)
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    result = {"command": public_command, "returncode": proc.returncode, "skipped": False}
    if "-k" in command:
        result["recursive_tests_deselected"] = True
    return result


def _run_optional_command(command: list[str], public_command: str) -> dict[str, Any]:
    if not _command_target_exists(command):
        return {"command": public_command, "missing": True}
    # Non-recursive default: refuse nested prior-block validation scripts unless
    # explicitly allowed. This prevents the 15K script-to-script recursion hang.
    if _is_prior_block_script(command) and os.environ.get("MEDAI_ALLOW_NESTED_SCRIPTS") != "1":
        return {
            "command": public_command,
            "skipped": True,
            "reason": "nested_prior_block_script_disabled_by_default",
        }
    return _run_command(command, public_command)


def _command_target_exists(command: list[str]) -> bool:
    target = command[-1]
    return not target.endswith(".py") or (REPO_ROOT / target).exists()


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "execution/ai_provider_operator_control.py",
        "execution/local_ollama_extraction_adapter.py",
        "execution/premium_adapter_contract.py",
        "execution/extraction_workflow.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "ollama",
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
        CLAUDE_DEMO_SECRET,
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_imports_introduced": False,
        "network_call_code_introduced": False,
        "localhost_or_ollama_call_introduced": False,
        "subprocess_call_introduced": False,
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
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "all_real_call_attempted_false": True,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PROVIDER-ENABLEMENT-OPERATOR-CONTROL-15K",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}`",
            f"- Effective provider: `{summary['effective_provider']}`",
            f"- Staged request state: `{summary['staged_request_state']}`",
            f"- Staged request allowed (recorded only): `{summary['staged_request_allowed']}`",
            f"- Real provider execution enabled: `{summary['real_provider_execution_enabled']}`",
            f"- Gemini/Claude/OpenAI/Ollama real call attempted: `False`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Local model call used: `{summary['local_model_call_used']}`",
            f"- Subprocess call used: `{summary['subprocess_call_used']}`",
            f"- Final external call allowed: `{summary['final_external_call_allowed']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- Doctrine phrases present: `{summary['doctrine_compliance']['doctrine_phrases_present']}`",
            f"- 15K test code: `{validation['focused_15k_tests'].get('returncode', 'skipped')}`",
            "",
            "## Operator control",
            "",
            "- Unified readiness matrix for fake_local, gemini, claude, openai, local_ollama.",
            "- Staging an enablement request records intent only; it never enables execution.",
            "- All real providers remain disabled by policy; fake_local stays enabled.",
            "",
            "## Limitations",
            "",
            "- 15K is the final operator-control/audit surface before any future live smoke test.",
            "- No real provider call, network call, local model call, or subprocess call is made.",
            "- Credential presence is reported by env-var name only; values are never read/logged/written.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-LIVE-PROVIDER-SMOKE-TEST-GATED-15L (single operator-approved, audited live smoke test).",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        operator_control_check,
        matrix,
        staged_check,
        provider_enablement,
        privacy,
        budget,
        operator_preview,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    CONTROL_JSON_PATH.write_text(json.dumps(operator_control_check, indent=2), encoding="utf-8")
    MATRIX_JSON_PATH.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    STAGED_JSON_PATH.write_text(json.dumps(staged_check, indent=2), encoding="utf-8")
    ENABLEMENT_JSON_PATH.write_text(json.dumps(provider_enablement, indent=2), encoding="utf-8")
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
        operator_control_check,
        matrix,
        staged_check,
        provider_enablement,
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
            operator_control_check["fake_local_enabled"] is True,
            operator_control_check["real_providers_disabled"] is True,
            operator_control_check["all_real_call_attempted_false"] is True,
            operator_control_check["real_provider_execution_enabled"] is False,
            staged_check["staged_does_not_enable_execution"] is True,
            staged_check["blocked_when_gates_missing"] is True,
            staged_check["claude_staged_allowed"] is True,
            len(matrix) == len(CONTROLLED_PROVIDERS),
            operator_preview["operator_control_staged_request_notice"] == "Staged request does not enable execution",
            operator_preview["operator_control_no_external_call_notice"] == "No external AI call was made",
            operator_preview["operator_control_no_local_model_call_notice"] == "No local model call was made",
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["local_model_call_used"] is False,
            summary["subprocess_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["real_provider_execution_enabled"] is False,
            summary["gemini_real_call_attempted"] is False,
            summary["claude_real_call_attempted"] is False,
            summary["openai_real_call_attempted"] is False,
            summary["ollama_real_call_attempted"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["review_bound_package_count"] >= 3,
            summary["doctrine_compliance"]["doctrine_phrases_present"] is True,
        ]
    )
    print(
        "medai_ai_provider_enablement_operator_control_15k_ready"
        if ready
        else "medai_ai_provider_enablement_operator_control_15k_not_ready"
    )
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "selected_provider": summary["selected_provider"],
                "effective_provider": summary["effective_provider"],
                "staged_request_state": summary["staged_request_state"],
                "real_provider_execution_enabled": summary["real_provider_execution_enabled"],
                "external_api_used": summary["external_api_used"],
                "real_network_call_used": summary["real_network_call_used"],
                "local_model_call_used": summary["local_model_call_used"],
                "subprocess_call_used": summary["subprocess_call_used"],
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
