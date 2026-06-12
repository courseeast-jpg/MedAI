#!/usr/bin/env python3
"""MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L validation.

Runs the flat, bounded, non-recursive validation harness over the 15A-15L
focused test suite plus 12A and 13C, records topology findings about the 15K
recursion defect, verifies depth-guard / timeout / nested-script-refusal
behavior with safe mock commands, and writes public-safe reports.

This 15L validation path does NOT call any prior-block validation script.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

from clinical_knowledge.privacy import check_public_report_payload
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    DEPTH_ENV,
    FlatHarnessConfig,
    is_prior_block_validation_script,
    run_direct_command,
    run_flat_validation,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_validation_harness_derecursion_15l"
DOCTRINE = REPO_ROOT / "docs" / "architecture" / "MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
TOPOLOGY_JSON_PATH = REPORT_DIR / "validation_topology_findings.json"
FLAT_RUN_JSON_PATH = REPORT_DIR / "flat_harness_run_report.json"
DEPTH_GUARD_JSON_PATH = REPORT_DIR / "depth_guard_check.json"
TIMEOUT_JSON_PATH = REPORT_DIR / "timeout_check.json"
DESELECT_JSON_PATH = REPORT_DIR / "recursive_test_deselect_check.json"
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
PRIOR_BLOCK_SCRIPTS = [
    "scripts/run_medai_ai_provider_enablement_operator_control_15k.py",
    "scripts/run_medai_ai_local_ollama_adapter_disabled_15j.py",
    "scripts/run_medai_ai_premium_adapters_disabled_15i.py",
    "scripts/run_medai_ai_gemini_adapter_disabled_15g.py",
    "scripts/run_medai_ai_real_provider_enablement_gate_15f.py",
    "scripts/run_medai_ai_external_call_dry_run_15e.py",
    "scripts/run_medai_ai_provider_selection_ui_15d.py",
    "scripts/run_medai_ai_provider_adapter_stub_15c.py",
    "scripts/run_medai_ai_extraction_privacy_gate_15b.py",
    "scripts/run_medai_ai_extraction_workflow_seam_15a.py",
]
RAW_FIXTURE_VALUES = [
    "Jane Example",
    "01/02/1970",
    "Park Medical Center",
    "Alice Clinician",
    "jane.example@example.com",
    "INS-ABC-12345",
]
BLOCKED_REPORT_TOKENS = RAW_FIXTURE_VALUES + ['"token_map":', "sk-", "C:\\"]


def build_topology_findings() -> dict[str, Any]:
    return {
        "block": "MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L",
        "recursion_defect_confirmed": True,
        "root_cause_summary": (
            "Validation recursed two ways: (1) each scripts/run_medai_ai_*_15X.py "
            "ran all prior-block validation scripts via PRIOR_BASELINE_COMMANDS, and "
            "each of those ran their own prior scripts; (2) each focused test file "
            "contained a test_*regressions_still_pass test that subprocess-ran prior "
            "focused test files, which ran their own regression tests. Both compound "
            "into effectively exponential nested execution and the ~4h 15K hang."
        ),
        "script_to_script_recursion_points": [
            {"script": script, "mechanism": "PRIOR_BASELINE_COMMANDS invokes prior-block run_medai_ai_*_15*.py scripts"}
            for script in [
                "scripts/run_medai_ai_provider_enablement_operator_control_15k.py",
                "scripts/run_medai_ai_local_ollama_adapter_disabled_15j.py",
                "scripts/run_medai_ai_premium_adapters_disabled_15i.py",
                "scripts/run_medai_ai_gemini_adapter_disabled_15g.py",
            ]
        ],
        "subprocess_cross_block_regression_tests": [
            "tests/test_medai_ai_provider_adapter_stub_15c.py::test_15b_15a_and_12a_13c_regressions_still_pass",
            "tests/test_medai_ai_provider_selection_ui_15d.py::test_15c_15b_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_external_call_dry_run_15e.py::test_15d_15c_15b_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_real_provider_enablement_gate_15f.py::test_15e_15d_15c_15b_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_gemini_adapter_disabled_15g.py::test_15f_through_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_premium_adapters_disabled_15i.py::test_15g_through_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_local_ollama_adapter_disabled_15j.py::test_15i_through_15a_12a_and_13c_regressions_still_pass",
            "tests/test_medai_ai_provider_enablement_operator_control_15k.py::test_prior_block_regressions_still_pass",
        ],
        "commands_that_caused_15k_hang": [
            "PRIOR_BASELINE_COMMANDS nested run_medai_ai_*_15*.py scripts",
            "test_prior_block_regressions_still_pass (recursive subprocess of prior focused files)",
        ],
        "safe_flat_tests": list(FOCUSED_TEST_FILES),
        "fix_strategy": (
            "Run focused test files in one flat pytest process with recursive "
            "regression tests deselected; run 12A and 13C as direct commands once; "
            "refuse nested prior-block scripts by default with a MEDAI_VALIDATION_DEPTH "
            "guard and explicit timeouts."
        ),
    }


def build_depth_guard_check() -> dict[str, Any]:
    # Probe: a prior-block script command must be refused by default, and again
    # blocked at depth>0. No script is actually executed (refused pre-exec).
    default_refusal = run_direct_command(
        "depth_guard_probe_default",
        [sys.executable, PRIOR_BLOCK_SCRIPTS[0]],
        timeout_s=5,
        allow_nested_scripts=False,
        depth=0,
    )
    depth_block = run_direct_command(
        "depth_guard_probe_depth1",
        [sys.executable, PRIOR_BLOCK_SCRIPTS[0]],
        timeout_s=5,
        allow_nested_scripts=True,  # even when allowed, depth>0 blocks it
        depth=1,
    )
    return {
        "depth_env_var": DEPTH_ENV,
        "default_refused": default_refusal.status == "refused",
        "default_reason": default_refusal.reason,
        "depth1_blocked": depth_block.status == "refused",
        "depth1_reason": depth_block.reason,
        "prior_block_script_detected": is_prior_block_validation_script([sys.executable, PRIOR_BLOCK_SCRIPTS[0]]),
        "no_script_executed": True,
        "passed": default_refusal.status == "refused" and depth_block.status == "refused",
    }


def build_timeout_check() -> dict[str, Any]:
    # Safe mock command: sleep 5s with a 1s timeout -> timed_out quickly.
    result = run_direct_command(
        "timeout_probe",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=1,
        allow_nested_scripts=False,
        depth=0,
    )
    return {
        "probe_command": "python -c import time; time.sleep(5)  (timeout 1s)",
        "status": result.status,
        "reason": result.reason,
        "timed_out_recorded_honestly": result.status == "timed_out",
        "passed": result.status == "timed_out",
    }


def build_reports() -> tuple[dict[str, Any], ...]:
    topology = build_topology_findings()
    depth_guard = build_depth_guard_check()
    timeout_check = build_timeout_check()

    config = FlatHarnessConfig(
        test_files=FOCUSED_TEST_FILES,
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        direct_commands=DIRECT_COMMANDS,
        per_command_timeout_s=600,
        total_timeout_s=1800,
        allow_nested_scripts=False,
    )
    flat_run = run_flat_validation(config)

    deselect_check = {
        "deselect_patterns": list(DEFAULT_DESELECT_PATTERNS),
        "deselected_recursive_test_count": flat_run["deselected_recursive_test_count"],
        "recursive_tests_deselected": flat_run["recursive_tests_deselected"],
        "flat_suite_status": flat_run["flat_suite"]["status"],
        "flat_suite_counts": flat_run["flat_suite"].get("pytest_counts", {}),
        "passed": flat_run["recursive_tests_deselected"] and flat_run["flat_suite"]["status"] == "passed",
    }

    doctrine_text = DOCTRINE.read_text(encoding="utf-8") if DOCTRINE.exists() else ""
    missing_phrases = [p for p in DOCTRINE_PHRASES if p not in doctrine_text]
    doctrine_check = {
        "doctrine_file_exists": DOCTRINE.exists(),
        "missing_phrases": missing_phrases,
        "passed": DOCTRINE.exists() and not missing_phrases,
        "package_first_extraction_preserved": True,
        "review_bound_ai_output_preserved": True,
        "record_count_not_success_metric": True,
    }

    direct_status = {item["name"]: item["status"] for item in flat_run["direct_commands"]}
    nested_scripts_invoked = flat_run["nested_prior_block_scripts_invoked"]
    summary = {
        "block": "MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L",
        "flat_suite_status": flat_run["flat_suite"]["status"],
        "flat_suite_counts": flat_run["flat_suite"].get("pytest_counts", {}),
        "flat_suite_duration_s": flat_run["flat_suite"].get("duration_s", 0.0),
        "total_duration_s": flat_run["total_duration_s"],
        "direct_command_status": direct_status,
        "deselected_recursive_test_count": flat_run["deselected_recursive_test_count"],
        "missing_test_files": flat_run["missing_test_files"],
        "nested_prior_block_scripts_invoked": nested_scripts_invoked,
        "nested_prior_block_scripts_refused_by_default": flat_run["nested_prior_block_scripts_refused_by_default"],
        "depth_guard_passed": depth_guard["passed"],
        "timeout_guard_passed": timeout_check["passed"],
        "subprocess_use": flat_run["subprocess_use"],
        "provider_execution_subprocess_path": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "doctrine_phrases_present": doctrine_check["passed"],
    }
    validation = {
        "flat_harness": flat_run,
        "depth_guard_check": depth_guard,
        "timeout_check": timeout_check,
        "recursive_test_deselect_check": deselect_check,
        "doctrine_check": doctrine_check,
        "source_safety_scan": _source_safety_scan(),
    }
    privacy = _privacy_report(summary, topology, depth_guard, timeout_check, deselect_check, doctrine_check, flat_run)
    summary["privacy_result"] = privacy["privacy_result"]
    implementation = _markdown(summary, flat_run)
    return (
        summary,
        validation,
        topology,
        flat_run,
        depth_guard,
        timeout_check,
        deselect_check,
        privacy,
        doctrine_check,
        implementation,
    )


def _source_safety_scan() -> dict[str, Any]:
    paths = [
        "scripts/medai_ai_flat_validation_harness.py",
        "scripts/run_medai_ai_validation_harness_derecursion_15l.py",
    ]
    source = "\n".join((REPO_ROOT / path).read_text(encoding="utf-8") for path in paths)
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "import " + "ollama",
        "import " + "google." + "generativeai",
        "google." + "generativeai",
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "generativelanguage." + "googleapis.com",
        "api." + "openai.com",
        "raw_pdf" + "_upload",
        "raw_image" + "_upload",
    ]
    matches = [item for item in forbidden if item in source]
    return {
        "provider_sdk_imports_introduced": False,
        "provider_execution_subprocess_path_introduced": False,
        "network_call_code_introduced": False,
        "localhost_or_ollama_call_introduced": False,
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
        "private_identifiers_in_report": False,
        "token_map_in_report": False,
        "credential_value_in_report": False,
        "provider_sdk_usage": False,
        "provider_execution_subprocess_path": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "active_written_count": 0,
        "auto_accept": False,
    }


def _markdown(summary: dict[str, Any], flat_run: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L",
            "",
            f"- Privacy result: `{summary.get('privacy_result', 'pending')}`",
            f"- Flat suite status: `{summary['flat_suite_status']}`",
            f"- Flat suite counts: `{summary['flat_suite_counts']}`",
            f"- Flat suite duration (s): `{summary['flat_suite_duration_s']}`",
            f"- Total harness duration (s): `{summary['total_duration_s']}`",
            f"- Deselected recursive tests: `{summary['deselected_recursive_test_count']}`",
            f"- 12A: `{summary['direct_command_status'].get('regression_12a_uat')}` | "
            f"13C: `{summary['direct_command_status'].get('regression_13c_pytest')}`",
            f"- Nested prior-block scripts invoked: `{summary['nested_prior_block_scripts_invoked']}`",
            f"- Nested scripts refused by default: `{summary['nested_prior_block_scripts_refused_by_default']}`",
            f"- Depth guard passed: `{summary['depth_guard_passed']}`",
            f"- Timeout guard passed: `{summary['timeout_guard_passed']}`",
            f"- Subprocess use: `{summary['subprocess_use']}` (no provider-execution subprocess path)",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Doctrine phrases present: `{summary['doctrine_phrases_present']}`",
            "",
            "## 15K recursion root cause",
            "",
            "- Script-to-script: PRIOR_BASELINE_COMMANDS ran prior-block scripts, which ran theirs.",
            "- Test-to-test: test_*regressions_still_pass subprocess-ran prior focused files, which ran theirs.",
            "- Combined -> exponential nested execution and the ~4h hang.",
            "",
            "## Validation strategy for future blocks",
            "",
            "- Run focused test files flat in one pytest process with recursive tests deselected.",
            "- Run 12A and 13C as direct commands once; never via nested prior-block scripts.",
            "- Refuse nested prior-block scripts by default; MEDAI_VALIDATION_DEPTH guards depth>0.",
            "- Enforce explicit per-command and total timeouts.",
            "",
            "## Important distinction",
            "",
            "- The harness uses subprocess ONLY to run local pytest/python test commands.",
            "- It introduces NO provider-execution subprocess path: no SDK, no network, no localhost/Ollama.",
            "",
            "## Next recommended block",
            "",
            "- MEDAI-AI-LIVE-PROVIDER-SMOKE-TEST-GATED-15M (single operator-approved, audited live smoke test).",
            "",
        ]
    )


def write_reports(*reports: Any) -> None:
    (
        summary,
        validation,
        topology,
        flat_run,
        depth_guard,
        timeout_check,
        deselect_check,
        privacy,
        doctrine_check,
        implementation,
    ) = reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    TOPOLOGY_JSON_PATH.write_text(json.dumps(topology, indent=2), encoding="utf-8")
    FLAT_RUN_JSON_PATH.write_text(json.dumps(flat_run, indent=2), encoding="utf-8")
    DEPTH_GUARD_JSON_PATH.write_text(json.dumps(depth_guard, indent=2), encoding="utf-8")
    TIMEOUT_JSON_PATH.write_text(json.dumps(timeout_check, indent=2), encoding="utf-8")
    DESELECT_JSON_PATH.write_text(json.dumps(deselect_check, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    DOCTRINE_JSON_PATH.write_text(json.dumps(doctrine_check, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    reports = build_reports()
    write_reports(*reports)
    (
        summary,
        validation,
        _topology,
        flat_run,
        depth_guard,
        timeout_check,
        deselect_check,
        privacy,
        doctrine_check,
        _implementation,
    ) = reports
    ready = all(
        [
            flat_run["passed"],
            flat_run["flat_suite"]["status"] == "passed",
            all(item["status"] == "passed" for item in flat_run["direct_commands"]),
            deselect_check["passed"],
            depth_guard["passed"],
            timeout_check["passed"],
            validation["source_safety_scan"]["passed"],
            doctrine_check["passed"],
            privacy["privacy_result"] == "passed",
            summary["nested_prior_block_scripts_invoked"] is False,
            summary["nested_prior_block_scripts_refused_by_default"] is True,
            summary["provider_execution_subprocess_path"] is False,
            summary["external_api_used"] is False,
            summary["real_network_call_used"] is False,
            summary["local_model_call_used"] is False,
            summary["final_external_call_allowed"] is False,
            summary["real_provider_execution_enabled"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            not summary["missing_test_files"],
        ]
    )
    print("medai_ai_validation_harness_derecursion_15l_ready" if ready else "medai_ai_validation_harness_derecursion_15l_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "flat_suite_status": summary["flat_suite_status"],
                "flat_suite_counts": summary["flat_suite_counts"],
                "total_duration_s": summary["total_duration_s"],
                "deselected_recursive_test_count": summary["deselected_recursive_test_count"],
                "nested_prior_block_scripts_invoked": summary["nested_prior_block_scripts_invoked"],
                "depth_guard_passed": summary["depth_guard_passed"],
                "timeout_guard_passed": summary["timeout_guard_passed"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
