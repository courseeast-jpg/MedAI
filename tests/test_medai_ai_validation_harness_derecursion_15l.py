"""Focused tests for MEDAI-AI-VALIDATION-HARNESS-DERECURSION-15L."""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy import check_public_report_payload
from scripts.medai_ai_flat_validation_harness import (
    DEFAULT_DESELECT_PATTERNS,
    DEPTH_ENV,
    CommandResult,
    FlatHarnessConfig,
    deselect_expression,
    is_prior_block_validation_script,
    run_direct_command,
    run_flat_pytest,
    run_flat_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"

# Prior focused files only (NOT 15L itself, to avoid self-recursion in tests).
PRIOR_FOCUSED_FILES = [
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
PRIOR_BLOCK_SCRIPT = "scripts/run_medai_ai_provider_enablement_operator_control_15k.py"


@functools.lru_cache(maxsize=1)
def _flat_run_prior() -> dict:
    """One bounded flat run over the PRIOR focused suite (cached, shared)."""
    config = FlatHarnessConfig(
        test_files=PRIOR_FOCUSED_FILES,
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        direct_commands=DIRECT_COMMANDS,
        per_command_timeout_s=600,
        total_timeout_s=1800,
        allow_nested_scripts=False,
    )
    return run_flat_validation(config)


# 1. Flat harness module exists.
def test_flat_harness_module_exists() -> None:
    assert callable(run_flat_validation)
    assert callable(run_flat_pytest)
    assert callable(run_direct_command)


# 2. Flat harness refuses nested prior-block script execution by default.
def test_refuses_nested_prior_block_script_by_default() -> None:
    result = run_direct_command(
        "probe",
        [sys.executable, PRIOR_BLOCK_SCRIPT],
        timeout_s=5,
        allow_nested_scripts=False,
        depth=0,
    )
    assert result.status == "refused"
    assert "nested_prior_block" in result.reason


# 3. Flat harness accepts focused pytest files.
def test_accepts_focused_pytest_files() -> None:
    result, missing = run_flat_pytest(
        ["tests/test_medai_ai_extraction_workflow_seam_15a.py"],
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        timeout_s=300,
        depth=0,
    )
    assert result.status == "passed"
    assert not missing


# 4. Supports deselecting recursive regression tests by pattern.
def test_deselect_expression_covers_recursive_tests() -> None:
    expr = deselect_expression(DEFAULT_DESELECT_PATTERNS)
    assert "not regressions_still_pass" in expr
    assert "not prior_block_regressions" in expr


# 5. Records deselected/skipped honestly.
def test_records_deselected_honestly() -> None:
    run = _flat_run_prior()
    assert run["recursive_tests_deselected"] is True
    assert run["deselected_recursive_test_count"] > 0


# 6. Records missing commands honestly.
def test_records_missing_commands_honestly() -> None:
    result, missing = run_flat_pytest(
        ["tests/test_medai_ai_DOES_NOT_EXIST_15zz.py"],
        deselect_patterns=DEFAULT_DESELECT_PATTERNS,
        timeout_s=30,
        depth=0,
    )
    assert result.status == "missing"
    assert "tests/test_medai_ai_DOES_NOT_EXIST_15zz.py" in missing


# 7. Records timeouts honestly.
def test_records_timeouts_honestly() -> None:
    result = run_direct_command(
        "timeout_probe",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=1,
        allow_nested_scripts=False,
        depth=0,
    )
    assert result.status == "timed_out"


# 8. Enforces depth guard.
def test_depth_guard_blocks_nested_script_even_when_allowed() -> None:
    result = run_direct_command(
        "depth_probe",
        [sys.executable, PRIOR_BLOCK_SCRIPT],
        timeout_s=5,
        allow_nested_scripts=True,  # explicitly allowed, but depth>0 blocks
        depth=1,
    )
    assert result.status == "refused"
    assert "depth" in result.reason


# 9-10. 12A and 13C run as direct commands, not via prior-block scripts.
def test_direct_regression_commands_not_prior_block_scripts() -> None:
    run = _flat_run_prior()
    direct = {item["name"]: item for item in run["direct_commands"]}
    assert direct["regression_12a_uat"]["status"] == "passed"
    assert direct["regression_13c_pytest"]["status"] == "passed"
    for _, command in DIRECT_COMMANDS:
        assert is_prior_block_validation_script(command) is False


# 11. 15L validation script exists.
def test_15l_validation_script_exists() -> None:
    assert (REPO_ROOT / "scripts/run_medai_ai_validation_harness_derecursion_15l.py").exists()


# 12. 15L validation script does not call prior block validation scripts recursively.
def test_15l_script_does_not_invoke_prior_block_scripts() -> None:
    source = (REPO_ROOT / "scripts/run_medai_ai_validation_harness_derecursion_15l.py").read_text(encoding="utf-8")
    # The script must not DEFINE a recursive PRIOR_BASELINE_COMMANDS list (the
    # phrase may appear in descriptive root-cause text, which is harmless).
    assert "PRIOR_BASELINE_COMMANDS = [" not in source
    run = _flat_run_prior()
    assert run["nested_prior_block_scripts_invoked"] is False


# 13. 15K validation path no longer defaults to recursive prior-block script execution.
def test_15k_validation_path_non_recursive_by_default() -> None:
    import os
    import subprocess

    env = dict(os.environ)
    env["MEDAI_15K_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_provider_enablement_operator_control_15k.py"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0
    validation = json.loads((REPO_ROOT / "reports/medai_ai_provider_enablement_operator_control_15k/validation.json").read_text(encoding="utf-8"))
    nested = [v for k, v in validation.items() if k.startswith("baseline_15") and k.endswith("_script")]
    assert nested, "expected prior-block script baseline entries"
    for entry in nested:
        assert entry.get("skipped") is True
        assert entry.get("reason") == "nested_prior_block_script_disabled_by_default"


# 14. Topology findings identify the 15K recursion defect.
def test_topology_findings_identify_recursion_defect() -> None:
    from scripts.run_medai_ai_validation_harness_derecursion_15l import build_topology_findings

    findings = build_topology_findings()
    assert findings["recursion_defect_confirmed"] is True
    assert findings["script_to_script_recursion_points"]
    assert findings["subprocess_cross_block_regression_tests"]


# 15-19. Report content: durations, tails, privacy, doctrine, guard status.
def test_report_includes_durations_and_tails() -> None:
    run = _flat_run_prior()
    assert "duration_s" in run["flat_suite"]
    assert run["total_duration_s"] >= 0.0
    assert "stdout_tail" in run["flat_suite"]


def test_guard_status_and_subprocess_distinction_present() -> None:
    run = _flat_run_prior()
    assert run["subprocess_use"] == "local_test_runner_only"
    assert run["provider_execution_subprocess_path"] is False


# 20-28. Invariants.
def test_harness_invariants() -> None:
    run = _flat_run_prior()
    assert run["external_api_used"] is False
    assert run["real_network_call_used"] is False
    assert run["local_model_call_used"] is False
    assert run["final_external_call_allowed"] is False
    assert run["real_provider_execution_enabled"] is False
    assert run["provider_execution_subprocess_path"] is False


# 29-31. Privacy: no token map / PII / credentials in harness output.
def test_harness_output_is_public_safe() -> None:
    run = _flat_run_prior()
    serialized = json.dumps(run, sort_keys=True, default=str)
    assert '"token_map":' not in serialized
    assert "sk-" not in serialized
    for raw_value in ("Jane Example", "jane.example@example.com", "INS-ABC-12345"):
        assert raw_value not in serialized
    assert check_public_report_payload(run).passed


# 32. Doctrine exact phrases remain present.
def test_capability_boundary_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 33-42. Each prior block's focused tests pass under the flat harness.
@pytest.mark.parametrize("test_file", PRIOR_FOCUSED_FILES)
def test_prior_block_focused_passes_under_flat_harness(test_file) -> None:
    run = _flat_run_prior()
    assert run["flat_suite"]["status"] == "passed", run["flat_suite"].get("stdout_tail", "")
    assert int(run["flat_suite"]["pytest_counts"].get("failed", 0)) == 0
    assert test_file not in run["missing_test_files"]


# 43-44. 12A and 13C pass.
def test_12a_and_13c_pass_under_flat_harness() -> None:
    run = _flat_run_prior()
    direct = {item["name"]: item["status"] for item in run["direct_commands"]}
    assert direct["regression_12a_uat"] == "passed"
    assert direct["regression_13c_pytest"] == "passed"
