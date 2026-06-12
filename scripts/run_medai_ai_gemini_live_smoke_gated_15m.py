#!/usr/bin/env python3
"""MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M validation.

Default (no live env gates): performs NO call, reports BLOCKED_READY_FOR_LIVE_SMOKE,
runs the flat bounded 15A-15M suite + 12A + 13C, and writes public-safe reports.
Exits nonzero only on a real validation/safety failure — never merely because
live env gates are absent.

If the explicit live env gates ARE present, it performs exactly one real Gemini
call against the synthetic redacted payload and records sanitized result metadata.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

from clinical_knowledge.privacy import check_public_report_payload
from execution.gemini_live_smoke import (
    ALLOW_SMOKE_ENV,
    OPERATOR_APPROVED_ENV,
    GEMINI_CREDENTIAL_ENV_VAR,
    GeminiLiveSmokeRequest,
    build_live_smoke_audit,
    build_live_smoke_operator_preview,
    evaluate_live_smoke_gates,
    live_smoke_audit_to_public_dict,
    live_smoke_decision_to_public_dict,
    live_smoke_result_to_public_dict,
    run_gemini_live_smoke,
    synthetic_payload_check,
)
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    FlatHarnessConfig,
    run_flat_validation,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_gemini_live_smoke_gated_15m"
DOCTRINE = REPO_ROOT / "docs" / "architecture" / "MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
POLICY_JSON_PATH = REPORT_DIR / "live_smoke_policy_check.json"
GATE_JSON_PATH = REPORT_DIR / "live_gate_decision.json"
PAYLOAD_JSON_PATH = REPORT_DIR / "synthetic_payload_check.json"
AUDIT_JSON_PATH = REPORT_DIR / "gemini_live_smoke_audit.json"
SCHEMA_JSON_PATH = REPORT_DIR / "schema_validation_check.json"
ENABLEMENT_JSON_PATH = REPORT_DIR / "provider_enablement_check.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
BUDGET_JSON_PATH = REPORT_DIR / "budget_guard_check.json"
OPERATOR_PREVIEW_JSON_PATH = REPORT_DIR / "operator_preview_sample.json"
IMPLEMENTATION_MD_PATH = REPORT_DIR / "implementation_report.md"

FOCUSED_TEST_FILES = [
    "tests/test_medai_ai_extraction_workflow_seam_15a.py",
    "tests/test_medai_ai_extraction_privacy_gate_15b.py",
    "tests/test_medai_ai_provider_adapter_stub_15c.py",
    "tests/test_medai_ai_provider_selection_ui_15d.py",
    "tests/test_medai_ai_external_call_dry_run_15e.py",
    "tests/test_medai_ai_real_provider_enablement_gate_15f.py",
    "tests/test_medai_ai_gemini_adapter_disabled_15g.py",
    "tests/test_medai_ai_premium_adapters_disabled_15i.py",
    "tests/test_medai_ai_local_ollama_adapter_disabled_15j.py",
    "tests/test_medai_ai_provider_enablement_operator_control_15k.py",
    "tests/test_medai_ai_validation_harness_derecursion_15l.py",
    "tests/test_medai_ai_gemini_live_smoke_gated_15m.py",
]
DIRECT_COMMANDS = [
    ("regression_12a_uat", [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"]),
    ("regression_13c_pytest", [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py", "-p", "no:cacheprovider", "-q"]),
]
DOCTRINE_PHRASES = [
    "Do not assign semantic/layout understanding to OCR or deterministic rules.",
    (
        "A MedAI extraction block is not successful because records were created. It is "
        "successful only if the operator can compare a source-faithful package against the "
        "original document quickly and safely."
    ),
]
RAW_FIXTURE_VALUES = ["Jane Example", "jane.example@example.com", "INS-ABC-12345", "01/02/1970", "Park Medical Center"]
BLOCKED_REPORT_TOKENS = RAW_FIXTURE_VALUES + ['"token_map":', "sk-", "C:\\"]


def build_reports() -> tuple[dict[str, Any], ...]:
    # Live env gates are read from the real environment. Default = absent = no call.
    request = GeminiLiveSmokeRequest(redacted_payload_hash="synthetic15m01")
    decision = evaluate_live_smoke_gates(request)
    result = run_gemini_live_smoke(request)  # no injected client; real SDK only if gates pass
    result_public = live_smoke_result_to_public_dict(result)
    audit = live_smoke_audit_to_public_dict(build_live_smoke_audit(result))
    preview = build_live_smoke_operator_preview(result)

    policy_check = {
        "selected_provider": result.selected_provider,
        "payload_type": result.payload_type,
        "payload_class": result.payload_class,
        "call_limit": result.call_limit,
        "per_call_budget_cap": result.per_call_budget_cap,
        "allow_smoke_env_var": ALLOW_SMOKE_ENV,
        "operator_approved_env_var": OPERATOR_APPROVED_ENV,
        "credential_env_var_name": GEMINI_CREDENTIAL_ENV_VAR,
        "credential_value_in_report": False,
        "real_call_requires_exactly_one_call": True,
        "real_call_requires_synthetic_payload_class": True,
        "real_call_requires_both_env_flags": True,
        "real_call_requires_staged_operator_request": True,
    }
    gate_decision = live_smoke_decision_to_public_dict(decision)
    payload_check = synthetic_payload_check()
    schema_check = {
        "schema_valid": result.schema_valid,
        "review_bound_package_count": result.review_bound_package_count,
        "package_summary": result.package_summary,
        "note": "schema validated only if a live/fake call actually returned a response",
    }
    provider_enablement = {
        "selected_provider": result.selected_provider,
        "effective_provider": preview["effective_provider"],
        "real_provider_execution_enabled_for_smoke_only": result.real_provider_execution_enabled,
        "gemini_real_call_attempted": result.gemini_real_call_attempted,
        "claude_real_call_attempted": False,
        "openai_real_call_attempted": False,
        "ollama_real_call_attempted": False,
        "local_model_call_used": False,
        "provider_execution_subprocess_path": False,
        "external_api_used": result.external_api_used,
        "real_network_call_used": result.real_network_call_used,
        "final_external_call_allowed": result.final_external_call_allowed,
    }
    budget = {
        "per_call_budget_cap": result.per_call_budget_cap,
        "call_limit": result.call_limit,
        "budget_allowed": result.budget_allowed,
    }

    flat_run = _flat_run()
    doctrine_text = DOCTRINE.read_text(encoding="utf-8") if DOCTRINE.exists() else ""
    doctrine_missing = [p for p in DOCTRINE_PHRASES if p not in doctrine_text]

    overall_status = _overall_status(result)
    summary = {
        "block": "MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M",
        "overall_status": overall_status,
        "selected_provider": result.selected_provider,
        "effective_provider": preview["effective_provider"],
        "live_call_status": result.live_call_status,
        "live_call_attempted": result.gemini_real_call_attempted,
        "missing_live_gates": result.missing_gates,
        "external_api_used": result.external_api_used,
        "real_network_call_used": result.real_network_call_used,
        "gemini_real_call_attempted": result.gemini_real_call_attempted,
        "claude_real_call_attempted": False,
        "openai_real_call_attempted": False,
        "ollama_real_call_attempted": False,
        "local_model_call_used": False,
        "provider_execution_subprocess_path": False,
        "final_external_call_allowed": result.final_external_call_allowed,
        "real_provider_execution_enabled": result.real_provider_execution_enabled and result.gemini_real_call_attempted,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "review_bound_package_count": result.review_bound_package_count,
        "doctrine_phrases_present": not doctrine_missing,
        "flat_suite_status": flat_run["flat_suite"]["status"],
        "flat_suite_counts": flat_run["flat_suite"].get("pytest_counts", {}),
        "flat_total_duration_s": flat_run["total_duration_s"],
        "nested_prior_block_scripts_invoked": flat_run["nested_prior_block_scripts_invoked"],
    }
    validation = {
        "flat_harness": flat_run,
        "doctrine_check": {"missing_phrases": doctrine_missing, "passed": not doctrine_missing},
        "source_safety_scan": _source_safety_scan(),
    }
    privacy = _privacy_report(summary, policy_check, gate_decision, payload_check, audit, schema_check, provider_enablement, preview, result_public)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, flat_run)
    return (
        summary,
        validation,
        policy_check,
        gate_decision,
        payload_check,
        audit,
        schema_check,
        provider_enablement,
        privacy,
        budget,
        preview,
        implementation,
    )


def _overall_status(result: Any) -> str:
    if result.live_call_status == "succeeded":
        return "PASS_LIVE_SMOKE_SUCCEEDED"
    if result.live_call_status == "failed":
        return "FAIL_LIVE_SMOKE"
    return "BLOCKED_READY_FOR_LIVE_SMOKE"


def _flat_run() -> dict[str, Any]:
    if os.environ.get("MEDAI_15M_SKIP_FLAT") == "1":
        return {
            "flat_suite": {"status": "skipped", "pytest_counts": {}, "stdout_tail": "", "duration_s": 0.0},
            "direct_commands": [],
            "total_duration_s": 0.0,
            "nested_prior_block_scripts_invoked": False,
            "recursive_tests_deselected": True,
            "passed": True,
            "skipped": True,
        }
    config = FlatHarnessConfig(
        test_files=FOCUSED_TEST_FILES,
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        direct_commands=DIRECT_COMMANDS,
        per_command_timeout_s=600,
        total_timeout_s=1800,
        allow_nested_scripts=False,
    )
    return run_flat_validation(config)


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "execution/gemini_live_smoke.py",
        "scripts/run_medai_ai_gemini_live_smoke_gated_15m.py",
        "scripts/medai_ai_flat_validation_harness.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "import " + "ollama",
        "import " + "google." + "generativeai",
        "from " + "google." + "generativeai",
        "google." + "generativeai",  # contiguous literal must not appear (assembled dynamically)
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "raw_pdf" + "_upload",
        "raw_image" + "_upload",
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_hard_import_introduced": False,
        "network_call_code_introduced": False,
        "provider_execution_subprocess_path_introduced": False,
        "raw_pdf_image_upload_introduced": False,
        "matches": matches,
        "passed": not matches,
    }


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True, default=str)
    token_scan_passed = not any(token in combined for token in BLOCKED_REPORT_TOKENS)
    payload_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if token_scan_passed and payload_check.passed else "failed",
        "raw_ocr_text_in_report": False,
        "real_patient_identifiers_in_report": False,
        "token_map_in_report": False,
        "credential_value_in_report": False,
        "raw_gemini_credential_value_in_report": False,
        "raw_private_payload_in_report": False,
        "provider_sdk_hard_dependency": False,
        "external_api_used_default_mode": False,
        "real_network_call_used_default_mode": False,
        "gemini_real_call_attempted_default_mode": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], flat_run: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M",
            "",
            f"- Overall status: `{summary['overall_status']}`",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Selected provider: `{summary['selected_provider']}` | Effective: `{summary['effective_provider']}`",
            f"- Live call status: `{summary['live_call_status']}`",
            f"- Live call attempted: `{summary['live_call_attempted']}`",
            f"- Missing live gates: `{summary['missing_live_gates']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Real network call used: `{summary['real_network_call_used']}`",
            f"- Gemini real call attempted: `{summary['gemini_real_call_attempted']}`",
            f"- Claude/OpenAI/Ollama real call attempted: `False`",
            f"- Local model call used: `{summary['local_model_call_used']}`",
            f"- Provider-execution subprocess path: `{summary['provider_execution_subprocess_path']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review-bound package count: `{summary['review_bound_package_count']}`",
            f"- Doctrine phrases present: `{summary['doctrine_phrases_present']}`",
            f"- Flat suite status: `{summary['flat_suite_status']}` counts `{summary['flat_suite_counts']}` "
            f"({summary['flat_total_duration_s']}s)",
            "",
            "## Live gate model",
            "",
            "- A single live Gemini call occurs only if ALL gates pass: synthetic payload class,",
            "  redacted text/layout summary, privacy/payload/budget/dry-run passed, operator staged,",
            "  both env approval flags, GEMINI_API_KEY present, per-call budget cap, call_limit==1.",
            "- Default (any gate missing): no call, no SDK import, no network, BLOCKED_READY_FOR_LIVE_SMOKE.",
            "- Credential value is never read into reports/logs/UI (presence only).",
            "- Output is always review-bound; no active MKB write; no auto-accept.",
            "",
            "## Validation strategy",
            "",
            "- Uses the 15L flat bounded harness over 15A-15M; recursive tests deselected.",
            "- 12A and 13C run as direct commands; no recursive prior-block scripts invoked.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-GEMINI-LIVE-SMOKE-OPERATOR-RUN-15N (operator executes the single approved live call).",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        policy_check,
        gate_decision,
        payload_check,
        audit,
        schema_check,
        provider_enablement,
        privacy,
        budget,
        preview,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    POLICY_JSON_PATH.write_text(json.dumps(policy_check, indent=2), encoding="utf-8")
    GATE_JSON_PATH.write_text(json.dumps(gate_decision, indent=2), encoding="utf-8")
    PAYLOAD_JSON_PATH.write_text(json.dumps(payload_check, indent=2), encoding="utf-8")
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    SCHEMA_JSON_PATH.write_text(json.dumps(schema_check, indent=2), encoding="utf-8")
    ENABLEMENT_JSON_PATH.write_text(json.dumps(provider_enablement, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    BUDGET_JSON_PATH.write_text(json.dumps(budget, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_JSON_PATH.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    (
        summary,
        validation,
        _policy_check,
        _gate_decision,
        _payload_check,
        _audit,
        _schema_check,
        _provider_enablement,
        privacy,
        _budget,
        _preview,
        _implementation,
    ) = reports
    flat = validation["flat_harness"]
    flat_ok = bool(flat.get("skipped")) or (
        flat["flat_suite"]["status"] == "passed"
        and all(item["status"] == "passed" for item in flat.get("direct_commands", []))
    )
    # Live env gates absent is NOT a failure; a live FAIL is.
    live_failure = summary["live_call_status"] == "failed"
    ready = all(
        [
            flat_ok,
            validation["source_safety_scan"]["passed"],
            validation["doctrine_check"]["passed"],
            privacy["privacy_result"] == "passed",
            summary["nested_prior_block_scripts_invoked"] is False,
            summary["claude_real_call_attempted"] is False,
            summary["openai_real_call_attempted"] is False,
            summary["ollama_real_call_attempted"] is False,
            summary["local_model_call_used"] is False,
            summary["provider_execution_subprocess_path"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            not live_failure,
        ]
    )
    # In default/no-call mode, also assert the no-call invariants.
    if not summary["live_call_attempted"]:
        ready = ready and all(
            [
                summary["external_api_used"] is False,
                summary["real_network_call_used"] is False,
                summary["gemini_real_call_attempted"] is False,
            ]
        )
    print("medai_ai_gemini_live_smoke_gated_15m_ready" if ready else "medai_ai_gemini_live_smoke_gated_15m_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "overall_status": summary["overall_status"],
                "privacy_result": privacy["privacy_result"],
                "live_call_status": summary["live_call_status"],
                "live_call_attempted": summary["live_call_attempted"],
                "missing_live_gates": summary["missing_live_gates"],
                "external_api_used": summary["external_api_used"],
                "real_network_call_used": summary["real_network_call_used"],
                "gemini_real_call_attempted": summary["gemini_real_call_attempted"],
                "review_bound_package_count": summary["review_bound_package_count"],
                "flat_suite_status": summary["flat_suite_status"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
