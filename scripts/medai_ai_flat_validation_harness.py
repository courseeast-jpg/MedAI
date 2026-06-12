#!/usr/bin/env python3
"""Reusable flat, bounded, non-recursive validation harness for MedAI AI blocks.

Why this exists
---------------
The 15A-15K validation chain recursed two ways and could hang for hours:

1. Script-to-script: each ``scripts/run_medai_ai_*_15X.py`` ran all PRIOR block
   validation scripts, and each of those ran *their* prior scripts -> exponential.
2. Test-to-test: each ``tests/test_medai_ai_*_15X.py`` contained a
   ``test_*regressions_still_pass`` test that subprocess-ran the prior focused
   test files, which ran *their* regression tests -> exponential.

This harness replaces that pattern with a single flat pytest process over the
focused test files (recursive cross-block regression tests deselected by name),
plus a small set of explicit direct commands (e.g. 12A, 13C) run once.

Safety / scope
--------------
This harness uses ``subprocess`` ONLY to run local pytest/python *test* commands.
It introduces NO provider-execution subprocess path: it never calls a provider
SDK, a network endpoint, localhost, or an Ollama runtime. By default it REFUSES
to run prior-block validation scripts (``run_medai_ai_*_15*.py``), and a depth
guard (``MEDAI_VALIDATION_DEPTH``) fails such nested scripts closed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEPTH_ENV = "MEDAI_VALIDATION_DEPTH"
ALLOW_NESTED_ENV = "MEDAI_ALLOW_NESTED_SCRIPTS"

# Default deselect patterns: the recursive cross-block subprocess regression
# tests. Matched against pytest -k (substring of the test node name).
DEFAULT_DESELECT_PATTERNS = ("regressions_still_pass", "prior_block_regressions")

# A "prior-block validation script" is any AI-block runner script. Running one
# of these from inside the harness re-introduces script-to-script recursion.
PRIOR_BLOCK_SCRIPT_RE = re.compile(r"run_medai_ai_.*\.py$|run_medai_ai_.*\.py\"$")

DEFAULT_PER_COMMAND_TIMEOUT_S = 600        # 10 minutes
DEFAULT_TOTAL_TIMEOUT_S = 1800             # 30 minutes
TAIL_CHARS = 2000


@dataclass
class CommandResult:
    name: str
    command: str
    status: str                 # passed | failed | skipped | missing | timed_out | refused
    returncode: int | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    pytest_counts: dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass
class FlatHarnessConfig:
    test_files: list[str] = field(default_factory=list)
    deselect_patterns: tuple[str, ...] = DEFAULT_DESELECT_PATTERNS
    direct_commands: list[tuple[str, list[str]]] = field(default_factory=list)
    per_command_timeout_s: int = DEFAULT_PER_COMMAND_TIMEOUT_S
    total_timeout_s: int = DEFAULT_TOTAL_TIMEOUT_S
    allow_nested_scripts: bool = False


def is_prior_block_validation_script(command: list[str]) -> bool:
    """True if the command invokes a run_medai_ai_*_15*.py validation script."""
    for arg in command:
        base = str(arg).replace("\\", "/").split("/")[-1]
        if base.startswith("run_medai_ai_") and base.endswith(".py"):
            return True
    return False


def deselect_expression(patterns: tuple[str, ...]) -> str:
    return " and ".join(f"not {pattern}" for pattern in patterns)


_ABS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s'\"]*|/(?:home|Users)/[^\s'\"]*")
_BACKSLASH_PATH_RE = re.compile(r"[^\s'\"]*\\[^\s'\"]+")
_FILENAME_RE = re.compile(r"[\w./\\-]+\.(?:json|py|md|txt|log|ini|csv|db|sqlite)\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?\b")
_HEADER_LINE_RE = re.compile(r"^(rootdir|cachedir|platform|plugins|configfile):.*$", re.MULTILINE)


def _sanitize_tail(text: str) -> str:
    """Redact paths, filenames, and timestamps so command tails are public-safe.

    Captured stdout/stderr can include local paths (pytest ``rootdir:`` header),
    report filenames (e.g. ``medai_*_report.json``), and log timestamps. None of
    these are needed for debugging status, and the conservative public-report
    privacy scanner flags filenames/dates, so they are redacted here.
    """
    text = text or ""
    text = text.replace(str(REPO_ROOT), "<repo>")
    text = _HEADER_LINE_RE.sub("<redacted-header>", text)
    text = _ABS_PATH_RE.sub("<path>", text)
    text = _BACKSLASH_PATH_RE.sub("<path>", text)
    text = _FILENAME_RE.sub("<file>", text)
    text = _DATE_RE.sub("<date>", text)
    return text


def _tail(text: str) -> str:
    return _sanitize_tail(text or "")[-TAIL_CHARS:]


def _parse_pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in ("passed", "failed", "deselected", "skipped", "error", "errors", "xfailed", "warnings"):
        match = re.search(rf"(\d+)\s+{label}\b", output)
        if match:
            key = "errors" if label == "error" else label
            counts[key] = int(match.group(1))
    return counts


def _child_env(depth: int) -> dict[str, str]:
    env = dict(os.environ)
    env[DEPTH_ENV] = str(depth + 1)
    return env


def run_direct_command(
    name: str,
    command: list[str],
    *,
    timeout_s: int,
    allow_nested_scripts: bool,
    depth: int,
) -> CommandResult:
    public_command = " ".join("python" if item == sys.executable else item for item in command)
    # Depth guard + nested-script refusal.
    if is_prior_block_validation_script(command) and not allow_nested_scripts:
        return CommandResult(
            name=name,
            command=public_command,
            status="refused",
            reason="nested_prior_block_validation_script_refused_by_default",
        )
    if is_prior_block_validation_script(command) and depth > 0:
        return CommandResult(
            name=name,
            command=public_command,
            status="refused",
            reason=f"nested_prior_block_validation_script_blocked_at_depth_{depth}",
        )
    # Missing target file (e.g. a .py that does not exist).
    target = command[-1]
    if str(target).endswith(".py") and not (REPO_ROOT / target).exists() and "pytest" not in command:
        return CommandResult(name=name, command=public_command, status="missing", reason="command_target_missing")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_child_env(depth),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=public_command,
            status="timed_out",
            duration_s=round(time.monotonic() - start, 3),
            stdout_tail=_tail(exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
            stderr_tail=_tail(exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")),
            reason=f"exceeded_timeout_{timeout_s}s",
        )
    duration = round(time.monotonic() - start, 3)
    counts = _parse_pytest_counts((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return CommandResult(
        name=name,
        command=public_command,
        status="passed" if proc.returncode == 0 else "failed",
        returncode=proc.returncode,
        duration_s=duration,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
        pytest_counts=counts,
    )


def run_flat_pytest(
    test_files: list[str],
    *,
    deselect_patterns: tuple[str, ...],
    timeout_s: int,
    depth: int,
) -> tuple[CommandResult, list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for test_file in test_files:
        (present if (REPO_ROOT / test_file).exists() else missing).append(test_file)
    if not present:
        return (
            CommandResult(
                name="flat_focused_suite",
                command="python -m pytest <none>",
                status="missing",
                reason="no_present_test_files",
            ),
            missing,
        )
    command = [sys.executable, "-m", "pytest", *present, "-p", "no:cacheprovider", "-q"]
    if deselect_patterns:
        command += ["-k", deselect_expression(deselect_patterns)]
    result = run_direct_command(
        "flat_focused_suite",
        command,
        timeout_s=timeout_s,
        allow_nested_scripts=False,
        depth=depth,
    )
    return result, missing


def run_flat_validation(config: FlatHarnessConfig) -> dict[str, Any]:
    depth = int(os.environ.get(DEPTH_ENV, "0") or 0)
    start = time.monotonic()
    flat_result, missing_files = run_flat_pytest(
        config.test_files,
        deselect_patterns=config.deselect_patterns,
        timeout_s=min(config.per_command_timeout_s, config.total_timeout_s),
        depth=depth,
    )
    direct_results: list[CommandResult] = []
    for name, command in config.direct_commands:
        if time.monotonic() - start > config.total_timeout_s:
            direct_results.append(
                CommandResult(
                    name=name,
                    command=" ".join(command),
                    status="timed_out",
                    reason="total_harness_timeout_exceeded",
                )
            )
            continue
        direct_results.append(
            run_direct_command(
                name,
                command,
                timeout_s=config.per_command_timeout_s,
                allow_nested_scripts=config.allow_nested_scripts,
                depth=depth,
            )
        )
    total_duration = round(time.monotonic() - start, 3)
    all_results = [flat_result, *direct_results]
    real_failures = [r for r in all_results if r.status in {"failed", "timed_out"}]
    refused = [r for r in all_results if r.status == "refused"]
    nested_scripts_refused = bool(refused) or not config.allow_nested_scripts
    deselected_total = int(flat_result.pytest_counts.get("deselected", 0))
    return {
        "depth": depth,
        "flat_suite": asdict(flat_result),
        "direct_commands": [asdict(r) for r in direct_results],
        "missing_test_files": missing_files,
        "deselect_patterns": list(config.deselect_patterns),
        "deselected_recursive_test_count": deselected_total,
        "recursive_tests_deselected": deselected_total > 0,
        "nested_prior_block_scripts_invoked": any(
            is_prior_block_validation_script(cmd) and r.status not in {"refused"}
            for (_, cmd), r in zip(config.direct_commands, direct_results)
        ),
        "nested_prior_block_scripts_refused_by_default": nested_scripts_refused,
        "per_command_timeout_s": config.per_command_timeout_s,
        "total_timeout_s": config.total_timeout_s,
        "total_duration_s": total_duration,
        "subprocess_use": "local_test_runner_only",
        "provider_execution_subprocess_path": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "passed": not real_failures and flat_result.status == "passed",
        "real_failures": [r.name for r in real_failures],
    }


__all__ = [
    "DEPTH_ENV",
    "ALLOW_NESTED_ENV",
    "DEFAULT_DESELECT_PATTERNS",
    "DEFAULT_PER_COMMAND_TIMEOUT_S",
    "DEFAULT_TOTAL_TIMEOUT_S",
    "CommandResult",
    "FlatHarnessConfig",
    "is_prior_block_validation_script",
    "deselect_expression",
    "run_direct_command",
    "run_flat_pytest",
    "run_flat_validation",
]


if __name__ == "__main__":
    # Smoke entry: validate the harness module loads and reports its guards.
    print(
        json.dumps(
            {
                "module": "medai_ai_flat_validation_harness",
                "default_deselect_patterns": list(DEFAULT_DESELECT_PATTERNS),
                "depth_env": DEPTH_ENV,
                "nested_prior_block_scripts_refused_by_default": True,
                "provider_execution_subprocess_path": False,
            },
            indent=2,
        )
    )
