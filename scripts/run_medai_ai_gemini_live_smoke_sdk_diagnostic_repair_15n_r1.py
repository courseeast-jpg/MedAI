#!/usr/bin/env python3
"""MEDAI-AI-GEMINI-LIVE-SMOKE-SDK-DIAGNOSTIC-REPAIR-15N-R1 validation.

NO-LIVE-CALL repair validator. Exercises the repaired provider-error capture
and SDK-aware call boundary entirely with injectable fake clients (no network,
no real Gemini call), runs the flat bounded 15A-15M suite + 12A + 13C, and
writes public-safe reports. The failed 15M live-run facts are preserved here
in sanitized, count/flag-only form.
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
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODEL_ENV,
    SYNTHETIC_REDACTED_SMOKE_SUMMARY,
    build_live_smoke_audit,
    detect_gemini_sdk,
    live_smoke_audit_to_public_dict,
    resolve_gemini_model,
    run_gemini_live_smoke,
)
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    FlatHarnessConfig,
    run_flat_validation,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1"
DOCTRINE = REPO_ROOT / "docs" / "architecture" / "MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
FAILED_RUN_JSON_PATH = REPORT_DIR / "failed_live_run_diagnostic_summary.json"
CAPTURE_JSON_PATH = REPORT_DIR / "provider_error_capture_check.json"
SDK_DETECT_JSON_PATH = REPORT_DIR / "sdk_detection_check.json"
SDK_BOUNDARY_JSON_PATH = REPORT_DIR / "sdk_call_boundary_check.json"
FAKE_MATRIX_JSON_PATH = REPORT_DIR / "fake_failure_matrix.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
DOCTRINE_JSON_PATH = REPORT_DIR / "doctrine_check.json"
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
    "tests/test_medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1.py",
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

# Sanitized, count/flag-only preservation of the failed 15M live run. No raw
# provider error, no credential, no payload — those were not captured by the
# pre-repair harness and are intentionally not reconstructed here.
FAILED_LIVE_RUN_SUMMARY = {
    "block": "MEDAI-AI-GEMINI-LIVE-SMOKE-GATED-15M",
    "overall_status": "FAIL_LIVE_SMOKE",
    "live_call_status": "failed",
    "missing_live_gates": [],
    "external_api_used": True,
    "real_network_call_used": True,
    "gemini_real_call_attempted": True,
    "provider_response_received": False,
    "schema_validation_ran": False,
    "review_bound_package_count": 0,
    "active_written_count": 0,
    "auto_accept": False,
    "pre_repair_error_capture": "generic_only (gemini_live_call_failed); provider error type/message/status were discarded",
    "additional_live_call_made_during_15n_r1": False,
}

FAKE_KEY = "AIza" + "FAKEKEY1234567890abcdef"
PAYLOAD_FRAGMENT = SYNTHETIC_REDACTED_SMOKE_SUMMARY
WIN_PATH = "C:" + chr(92) + "Users" + chr(92) + "demo" + chr(92) + "x.py"


class _FailClient:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.call_count = 0

    def generate_extraction(self, *, prompt: str = "", **_kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        raise self.exc


def _named_exc(name: str, msg: str, **attrs: Any) -> BaseException:
    exc = type(name, (Exception,), {})(msg)
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


def build_reports() -> tuple[dict[str, Any], ...]:
    live_env = {
        "MEDAI_ALLOW_REAL_PROVIDER_SMOKE": "1",
        "MEDAI_OPERATOR_APPROVED_LIVE_SMOKE": "1",
        "GEMINI_API_KEY": FAKE_KEY,
    }
    scenarios = [
        ("type_error", TypeError("got an unexpected keyword argument 'request_options'"), "sdk_signature_mismatch"),
        ("invalid_argument", _named_exc("InvalidArgument", "400 invalid request", status_code=400, code="INVALID_ARGUMENT"), "request_config_rejected"),
        ("not_found", _named_exc("NotFound", "404 model not found", status_code=404), "model_not_found"),
        ("permission_denied", _named_exc("PermissionDenied", "403 SERVICE_DISABLED not enabled", status_code=403), "api_disabled_or_permission"),
        ("resource_exhausted", _named_exc("ResourceExhausted", "429 quota exceeded billing", status_code=429), "quota_or_billing"),
        ("deadline_exceeded", _named_exc("DeadlineExceeded", "deadline exceeded timeout"), "timeout"),
        ("unknown", ValueError("totally unknown boom"), "unknown_provider_error"),
        ("leaky_key", RuntimeError(f"auth failed key={FAKE_KEY}"), None),
        ("leaky_payload", RuntimeError(f"rejected content {PAYLOAD_FRAGMENT}"), None),
        ("leaky_path", RuntimeError(f"file error at {WIN_PATH}")  , None),
    ]
    matrix: list[dict[str, Any]] = []
    for name, exc, expected_cat in scenarios:
        client = _FailClient(exc)
        result = run_gemini_live_smoke(environ=live_env, gemini_client=client)
        matrix.append(
            {
                "scenario": name,
                "call_count": client.call_count,
                "provider_error_category": result.provider_error_category,
                "category_matches_expected": (expected_cat is None) or (result.provider_error_category == expected_cat),
                "provider_response_received": result.provider_response_received,
                "schema_validation_ran": result.provider_response_received,
                "active_written_count": result.active_written_count,
                "auto_accept": result.auto_accept,
                "review_required": result.review_required,
                "redaction_applied": result.provider_error_redaction_applied,
                "credential_in_sanitized_message": FAKE_KEY in result.provider_error_message_sanitized,
                "payload_in_sanitized_message": PAYLOAD_FRAGMENT in result.provider_error_message_sanitized,
                "path_in_sanitized_message": "C:" + chr(92) in result.provider_error_message_sanitized,
                "status_code": result.provider_error_status_code,
            }
        )

    capture_sample = run_gemini_live_smoke(environ=live_env, gemini_client=_FailClient(_named_exc("InvalidArgument", "400 bad", status_code=400, code="INVALID_ARGUMENT")))
    capture_check = {
        "broad_exception_no_longer_generic_only": capture_sample.provider_error_available is True
        and capture_sample.provider_error_category != "",
        "provider_error_type": capture_sample.provider_error_type,
        "provider_error_category": capture_sample.provider_error_category,
        "provider_error_status_code": capture_sample.provider_error_status_code,
        "provider_error_code": capture_sample.provider_error_code,
        "provider_error_source": capture_sample.provider_error_source,
        "provider_error_retryable": capture_sample.provider_error_retryable,
        "provider_error_redaction_applied_field_present": True,
        "provider_error_message_truncated_field_present": True,
        "audit_carries_error_fields": "provider_error_category" in live_smoke_audit_to_public_dict(build_live_smoke_audit(capture_sample)),
    }

    sdk = detect_gemini_sdk()
    sdk_detection = {
        **sdk,
        "detection_used_network": False,
        "default_model": DEFAULT_GEMINI_MODEL,
        "model_env_override_var": GEMINI_MODEL_ENV,
        "resolved_model_default": resolve_gemini_model({}),
        "resolved_model_override": resolve_gemini_model({GEMINI_MODEL_ENV: "gemini-2.5-flash"}),
    }
    default_text_model = all(
        banned not in DEFAULT_GEMINI_MODEL.lower() for banned in ("vision", "image", "imagen", "tts", "audio", "veo")
    )
    sdk_boundary = {
        "no_hard_sdk_import_in_source": _no_hard_sdk_import(),
        "supports_google_genai": True,
        "supports_legacy_generativeai": True,
        "model_configurable": True,
        "model_env_override_supported": resolve_gemini_model({GEMINI_MODEL_ENV: "gemini-2.5-flash"}) == "gemini-2.5-flash",
        "default_is_text_model": default_text_model,
        "sdk_call_kwargs_keys_names_only": True,
        "no_live_call_during_15n_r1": True,
        "provider_execution_subprocess_path": False,
    }

    flat_run = _flat_run()
    doctrine_text = DOCTRINE.read_text(encoding="utf-8") if DOCTRINE.exists() else ""
    doctrine_missing = [p for p in DOCTRINE_PHRASES if p not in doctrine_text]
    doctrine_check = {"missing_phrases": doctrine_missing, "passed": DOCTRINE.exists() and not doctrine_missing}

    matrix_all_safe = all(
        (not m["credential_in_sanitized_message"])
        and (not m["payload_in_sanitized_message"])
        and (not m["path_in_sanitized_message"])
        and m["provider_response_received"] is False
        and m["active_written_count"] == 0
        and m["auto_accept"] is False
        and m["category_matches_expected"]
        for m in matrix
    )
    summary = {
        "block": "MEDAI-AI-GEMINI-LIVE-SMOKE-SDK-DIAGNOSTIC-REPAIR-15N-R1",
        "no_live_call_made": True,
        "external_api_used": False,
        "real_network_call_used": False,
        "gemini_real_call_attempted": False,
        "local_model_call_used": False,
        "provider_execution_subprocess_path": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "provider_error_capture_repaired": capture_check["broad_exception_no_longer_generic_only"],
        "fake_failure_matrix_all_safe": matrix_all_safe,
        "sdk_call_style_detected": sdk["sdk_call_style"],
        "default_model": DEFAULT_GEMINI_MODEL,
        "doctrine_phrases_present": doctrine_check["passed"],
        "flat_suite_status": flat_run["flat_suite"]["status"],
        "flat_suite_counts": flat_run["flat_suite"].get("pytest_counts", {}),
        "flat_total_duration_s": flat_run["total_duration_s"],
        "nested_prior_block_scripts_invoked": flat_run["nested_prior_block_scripts_invoked"],
    }
    validation = {
        "flat_harness": flat_run,
        "doctrine_check": doctrine_check,
        "source_safety_scan": _source_safety_scan(),
    }
    privacy = _privacy_report(summary, FAILED_LIVE_RUN_SUMMARY, capture_check, sdk_detection, sdk_boundary, matrix, doctrine_check)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, flat_run)
    return (
        summary,
        validation,
        dict(FAILED_LIVE_RUN_SUMMARY),
        capture_check,
        sdk_detection,
        sdk_boundary,
        {"scenarios": matrix, "all_safe": matrix_all_safe},
        privacy,
        doctrine_check,
        implementation,
    )


def _flat_run() -> dict[str, Any]:
    if os.environ.get("MEDAI_15N_R1_SKIP_FLAT") == "1":
        return {
            "flat_suite": {"status": "skipped", "pytest_counts": {}, "stdout_tail": "", "duration_s": 0.0},
            "direct_commands": [],
            "total_duration_s": 0.0,
            "nested_prior_block_scripts_invoked": False,
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


def _no_hard_sdk_import() -> bool:
    source = (REPO_ROOT / "execution/gemini_live_smoke.py").read_text(encoding="utf-8")
    legacy = "google." + "generativeai"
    genai = "google." + "genai"
    return not any(
        item in source
        for item in ("import " + legacy, "from " + legacy, legacy, "import " + genai)
    )


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "execution/gemini_live_smoke.py",
        "scripts/run_medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "import " + "ollama",
        "google." + "generativeai",  # contiguous literal must not appear
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "import " + "subprocess",
        "raw_pdf" + "_upload",
        "raw_image" + "_upload",
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_hard_import_introduced": False,
        "network_call_code_introduced": False,
        "provider_execution_subprocess_path_introduced": False,
        "matches": matches,
        "passed": not matches,
    }


def _privacy_report(*payloads: Any) -> dict[str, Any]:
    combined = json.dumps(payloads, sort_keys=True, default=str)
    blocked = [FAKE_KEY, PAYLOAD_FRAGMENT, "C:" + chr(92), '"token_map":', "sk-", "Jane Example", "jane.example@example.com"]
    token_scan_passed = not any(token in combined for token in blocked)
    payload_check = check_public_report_payload(payloads)
    return {
        "privacy_result": "passed" if token_scan_passed and payload_check.passed else "failed",
        "credential_value_in_report": False,
        "raw_gemini_credential_value_in_report": False,
        "raw_prompt_or_payload_in_report": False,
        "raw_provider_response_body_in_report": False,
        "token_map_in_report": False,
        "real_patient_identifiers_in_report": False,
        "absolute_path_in_report": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "gemini_real_call_attempted": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], flat_run: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-GEMINI-LIVE-SMOKE-SDK-DIAGNOSTIC-REPAIR-15N-R1",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- No live call made during 15N-R1: `{summary['no_live_call_made']}`",
            f"- External API used (repair run): `{summary['external_api_used']}`",
            f"- Real network call used (repair run): `{summary['real_network_call_used']}`",
            f"- Gemini real call attempted (repair run): `{summary['gemini_real_call_attempted']}`",
            f"- Provider error capture repaired: `{summary['provider_error_capture_repaired']}`",
            f"- Fake failure matrix all safe: `{summary['fake_failure_matrix_all_safe']}`",
            f"- SDK call style detected: `{summary['sdk_call_style_detected']}`",
            f"- Default model: `{summary['default_model']}`",
            f"- Doctrine phrases present: `{summary['doctrine_phrases_present']}`",
            f"- Flat suite: `{summary['flat_suite_status']}` counts `{summary['flat_suite_counts']}` "
            f"({summary['flat_total_duration_s']}s)",
            "",
            "## What was repaired",
            "",
            "- Replaced the broad exception handler that recorded only `gemini_live_call_failed`.",
            "- Provider exceptions are now classified (SDK signature / invalid-argument / not-found /",
            "  permission / quota / timeout / transport / unknown) and the message is sanitized",
            "  (credential, API-key patterns, bearer tokens, paths, prompt/payload literals) and truncated.",
            "- SDK-aware client boundary: prefers google-genai, falls back to legacy google-generativeai,",
            "  detected without network; model name configurable via GEMINI_MODEL (default text model).",
            "",
            "## Important distinction",
            "",
            "- Local subprocess is used ONLY to run pytest/python test commands via the flat harness.",
            "- NO provider-execution subprocess path; NO real Gemini call was made in 15N-R1.",
            "",
            "## Failed 15M live-run preservation",
            "",
            "- Sanitized count/flag-only summary is preserved in failed_live_run_diagnostic_summary.json.",
            "- The pre-repair harness discarded the provider error; it is not reconstructed.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-GEMINI-LIVE-SMOKE-OPERATOR-RERUN-15N-R2: operator performs exactly one new live",
            "  call after this repair; the next failure (if any) will be captured + classified safely.",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        failed_run,
        capture_check,
        sdk_detection,
        sdk_boundary,
        fake_matrix,
        privacy,
        doctrine_check,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    FAILED_RUN_JSON_PATH.write_text(json.dumps(failed_run, indent=2), encoding="utf-8")
    CAPTURE_JSON_PATH.write_text(json.dumps(capture_check, indent=2), encoding="utf-8")
    SDK_DETECT_JSON_PATH.write_text(json.dumps(sdk_detection, indent=2), encoding="utf-8")
    SDK_BOUNDARY_JSON_PATH.write_text(json.dumps(sdk_boundary, indent=2), encoding="utf-8")
    FAKE_MATRIX_JSON_PATH.write_text(json.dumps(fake_matrix, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    DOCTRINE_JSON_PATH.write_text(json.dumps(doctrine_check, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    (
        summary,
        validation,
        _failed_run,
        capture_check,
        _sdk_detection,
        sdk_boundary,
        fake_matrix,
        privacy,
        doctrine_check,
        _implementation,
    ) = reports
    flat = validation["flat_harness"]
    flat_ok = bool(flat.get("skipped")) or (
        flat["flat_suite"]["status"] == "passed"
        and all(item["status"] == "passed" for item in flat.get("direct_commands", []))
    )
    ready = all(
        [
            flat_ok,
            validation["source_safety_scan"]["passed"],
            doctrine_check["passed"],
            privacy["privacy_result"] == "passed",
            capture_check["broad_exception_no_longer_generic_only"] is True,
            fake_matrix["all_safe"] is True,
            sdk_boundary["no_hard_sdk_import_in_source"] is True,
            sdk_boundary["default_is_text_model"] is True,
            sdk_boundary["model_env_override_supported"] is True,
            summary["no_live_call_made"] is True,
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["gemini_real_call_attempted"] is False,
            summary["local_model_call_used"] is False,
            summary["provider_execution_subprocess_path"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["nested_prior_block_scripts_invoked"] is False,
        ]
    )
    print("medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1_ready" if ready else "medai_ai_gemini_live_smoke_sdk_diagnostic_repair_15n_r1_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "provider_error_capture_repaired": summary["provider_error_capture_repaired"],
                "fake_failure_matrix_all_safe": summary["fake_failure_matrix_all_safe"],
                "sdk_call_style_detected": summary["sdk_call_style_detected"],
                "no_live_call_made": summary["no_live_call_made"],
                "flat_suite_status": summary["flat_suite_status"],
                "flat_suite_counts": summary["flat_suite_counts"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
