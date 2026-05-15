"""Operator-only Streamlit control panel for fixed validation commands.

The panel intentionally exposes only allowlisted commands. It does not provide
free-form shell execution and does not modify clinical runtime behavior.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_BUNDLE = Path("backups/medai_final_release_2026-05-14.bundle")
DEFAULT_TIMEOUT_SECONDS = 120
FULL_TEST_TIMEOUT_SECONDS = int(os.getenv("MEDAI_UI_OPS_FULL_TEST_TIMEOUT", "3600"))


@dataclass(frozen=True)
class OperatorCommand:
    command_id: str
    label: str
    group: str
    argv: tuple[str, ...]
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    report_paths: tuple[str, ...] = ()
    requires_confirmation: bool = False
    description: str = ""


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    label: str
    status: str
    exit_code: int | None
    duration_seconds: float
    output_summary: str
    stderr_summary: str
    report_paths: tuple[str, ...]


COMMAND_HELP_TEXT = {
    "quick_health_check": "Fast check that MedAI is ready.",
    "final_mvp_validation": "Confirms the core release still passes.",
    "git_safety_check": "Confirms no unsafe files are staged.",
}


COMMAND_ALLOWLIST: dict[str, OperatorCommand] = {
    "quick_health_check": OperatorCommand(
        command_id="quick_health_check",
        label="Quick health check",
        group="Main checks",
        argv=(sys.executable, "scripts/run_cka_final_mvp_release_validation.py"),
        report_paths=("reports/cka_final_mvp_release/cka_final_mvp_release_report.json",),
        description="Runs the fast final MVP validation.",
    ),
    "final_mvp_validation": OperatorCommand(
        command_id="final_mvp_validation",
        label="Final MVP validation",
        group="Main checks",
        argv=(sys.executable, "scripts/run_cka_final_mvp_release_validation.py"),
        report_paths=("reports/cka_final_mvp_release/cka_final_mvp_release_report.json",),
    ),
    "full_test_suite": OperatorCommand(
        command_id="full_test_suite",
        label="Full test suite",
        group="Advanced: full test suite",
        argv=(sys.executable, "-m", "pytest", "tests"),
        timeout_seconds=FULL_TEST_TIMEOUT_SECONDS,
        requires_confirmation=True,
        description="Long-running full pytest suite; confirmation required.",
    ),
    "terminology_source_preflight": OperatorCommand(
        command_id="terminology_source_preflight",
        label="Terminology preflight",
        group="Advanced: terminology checks",
        argv=(sys.executable, "scripts/run_medai_terminology_sources_preflight.py"),
        report_paths=("reports/terminology_sources_preflight/terminology_sources_preflight_report.json",),
    ),
    "terminology_inventory": OperatorCommand(
        command_id="terminology_inventory",
        label="Terminology inventory",
        group="Advanced: terminology checks",
        argv=(sys.executable, "scripts/run_medai_terminology_inventory.py", "--terminology-root", "terminology_data"),
        report_paths=("reports/terminology_data_inventory/terminology_data_inventory_report.json",),
    ),
    "b07_term_validation": OperatorCommand(
        command_id="b07_term_validation",
        label="B07-TERM validation",
        group="Advanced: terminology checks",
        argv=(sys.executable, "scripts/run_b07_term01_opt_in_integration_validation.py"),
        report_paths=("reports/b07_term01_opt_in_integration/b07_term01_opt_in_integration_report.json",),
    ),
    "route_fix_validation": OperatorCommand(
        command_id="route_fix_validation",
        label="ROUTE-FIX validation",
        group="Advanced: routing and extraction checks",
        argv=(sys.executable, "scripts/run_medai_route_fix01_validation.py"),
        report_paths=("reports/medai_route_fix_01/medai_route_fix_01_report.json",),
    ),
    "focused_routing_tests": OperatorCommand(
        command_id="focused_routing_tests",
        label="Focused routing tests",
        group="Advanced: routing and extraction checks",
        argv=(
            sys.executable,
            "-m",
            "pytest",
            "tests/test_phase23_routing_efficiency.py",
            "tests/test_connector_orchestration.py",
            "tests/test_phase10_hardening.py",
            "-q",
        ),
        timeout_seconds=300,
    ),
    "git_safety_check": OperatorCommand(
        command_id="git_safety_check",
        label="Git safety check",
        group="Main checks",
        argv=("__internal_git_safety_check__",),
        description="Shows staged files and short status without staging anything.",
    ),
    "show_last_validation_reports": OperatorCommand(
        command_id="show_last_validation_reports",
        label="Last validation reports",
        group="Advanced: reports and recovery",
        argv=("__internal_show_reports__",),
        report_paths=(
            "reports/medai_release_validate_01/medai_release_validate_01_report.json",
            "reports/medai_final_release_package/medai_final_release_report.json",
            "reports/cka_final_mvp_release/cka_final_mvp_release_report.json",
            "reports/terminology_sources_preflight/terminology_sources_preflight_report.json",
            "reports/terminology_data_inventory/terminology_data_inventory_report.json",
        ),
    ),
    "show_release_tags": OperatorCommand(
        command_id="show_release_tags",
        label="Release tags",
        group="Advanced: reports and recovery",
        argv=("git", "tag", "--list", "medai-*2026-05-14"),
    ),
    "verify_final_bundle": OperatorCommand(
        command_id="verify_final_bundle",
        label="Verify release bundle",
        group="Advanced: reports and recovery",
        argv=("git", "bundle", "verify", FINAL_BUNDLE.as_posix()),
    ),
}


PANEL_GROUP_ORDER = (
    "Main checks",
    "Advanced: terminology checks",
    "Advanced: routing and extraction checks",
    "Advanced: full test suite",
    "Advanced: reports and recovery",
)


PRIVATE_MARKERS = (
    "LICENSE_ACK_PRIVATE",
    "terminology_data",
    "data/terminology",
    ".RRF",
    ".rrf",
    ".nlm",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".zip",
)


def get_command(command_id: str) -> OperatorCommand:
    try:
        return COMMAND_ALLOWLIST[command_id]
    except KeyError as exc:
        raise ValueError(f"Unknown operator command id: {command_id}") from exc


def command_groups() -> dict[str, list[OperatorCommand]]:
    groups: dict[str, list[OperatorCommand]] = {}
    for command in COMMAND_ALLOWLIST.values():
        groups.setdefault(command.group, []).append(command)
    return groups


def operator_status_label(status: str | None) -> str:
    labels = {
        None: "Not run yet",
        "not_run": "Not run yet",
        "running": "Running...",
        "passed": "Passed",
        "success": "Passed",
        "failed": "Needs attention",
        "error": "Needs attention",
    }
    return labels.get(status, "Needs attention")


def operator_panel_summary(results: dict[str, CommandResult]) -> dict[str, str]:
    if not results:
        return {
            "last_check": "Not run",
            "system_status": "Waiting for check",
            "safety": "Local-only checks",
        }
    latest = next(reversed(results.values()))
    return {
        "last_check": latest.label,
        "system_status": operator_status_label(latest.status),
        "safety": "Local-only checks",
    }


def redact_output(text: str, *, max_chars: int = 4000) -> str:
    redacted = text
    redacted = re.sub(r"[A-Za-z]:\\[^\r\n\t ]+", "[LOCAL_PATH]", redacted)
    redacted = re.sub(r"(?:/home/|/tmp/|/var/|/usr/|/opt/)[^\r\n\t ]+", "[LOCAL_PATH]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", redacted)
    redacted = redacted.replace("LICENSE_ACK_PRIVATE.json", "[PRIVATE_ACK_FILE]")
    return redacted[:max_chars]


def summarize_output(text: str, *, max_lines: int = 80) -> str:
    lines = [line.rstrip() for line in redact_output(text).splitlines()]
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) <= max_lines:
        return "\n".join(non_empty)
    return "\n".join(non_empty[:max_lines] + [f"... truncated {len(non_empty) - max_lines} lines ..."])


def _script_missing(command: OperatorCommand, repo_root: Path) -> str | None:
    if len(command.argv) >= 2 and command.argv[1].startswith("scripts/"):
        script = repo_root / command.argv[1]
        if not script.exists():
            return command.argv[1]
    return None


def _run_subprocess(command: OperatorCommand, repo_root: Path) -> CommandResult:
    missing = _script_missing(command, repo_root)
    if missing:
        return CommandResult(
            command_id=command.command_id,
            label=command.label,
            status="failed",
            exit_code=None,
            duration_seconds=0.0,
            output_summary=f"Script missing: {missing}",
            stderr_summary="",
            report_paths=command.report_paths,
        )

    start = time.monotonic()
    try:
        completed = subprocess.run(
            list(command.argv),
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=command.timeout_seconds,
            shell=False,
            check=False,
        )
        duration = time.monotonic() - start
        return CommandResult(
            command_id=command.command_id,
            label=command.label,
            status="passed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
            duration_seconds=duration,
            output_summary=summarize_output(completed.stdout),
            stderr_summary=summarize_output(completed.stderr),
            report_paths=command.report_paths,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        return CommandResult(
            command_id=command.command_id,
            label=command.label,
            status="failed",
            exit_code=None,
            duration_seconds=duration,
            output_summary=summarize_output(exc.stdout or ""),
            stderr_summary=f"Timed out after {command.timeout_seconds} seconds.",
            report_paths=command.report_paths,
        )


def _run_git_safety_check(command: OperatorCommand, repo_root: Path) -> CommandResult:
    start = time.monotonic()
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=command.timeout_seconds,
        shell=False,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=command.timeout_seconds,
        shell=False,
        check=False,
    )
    duration = time.monotonic() - start
    staged_files = [line.strip() for line in staged.stdout.splitlines() if line.strip()]
    unsafe = [path for path in staged_files if any(marker.lower() in path.lower() for marker in PRIVATE_MARKERS)]
    summary = {
        "staged_file_count": len(staged_files),
        "staged_files": staged_files[:80],
        "unsafe_staged_file_count": len(unsafe),
        "dirty_status_count": len([line for line in status.stdout.splitlines() if line.strip()]),
    }
    exit_code = 0 if staged.returncode == 0 and status.returncode == 0 and not unsafe else 1
    return CommandResult(
        command_id=command.command_id,
        label=command.label,
        status="passed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        duration_seconds=duration,
        output_summary=json.dumps(summary, indent=2),
        stderr_summary=summarize_output(staged.stderr + "\n" + status.stderr),
        report_paths=command.report_paths,
    )


def _show_reports(command: OperatorCommand, repo_root: Path) -> CommandResult:
    existing = []
    missing = []
    for rel_path in command.report_paths:
        path = repo_root / rel_path
        if path.exists():
            existing.append(rel_path)
        else:
            missing.append(rel_path)
    return CommandResult(
        command_id=command.command_id,
        label=command.label,
        status="passed",
        exit_code=0,
        duration_seconds=0.0,
        output_summary=json.dumps({"existing_reports": existing, "missing_reports": missing}, indent=2),
        stderr_summary="",
        report_paths=command.report_paths,
    )


def run_operator_command(command_id: str, *, repo_root: Path = REPO_ROOT) -> CommandResult:
    command = get_command(command_id)
    if command.argv == ("__internal_git_safety_check__",):
        return _run_git_safety_check(command, repo_root)
    if command.argv == ("__internal_show_reports__",):
        return _show_reports(command, repo_root)
    if command.command_id == "verify_final_bundle" and not (repo_root / FINAL_BUNDLE).exists():
        return CommandResult(
            command_id=command.command_id,
            label=command.label,
            status="failed",
            exit_code=None,
            duration_seconds=0.0,
            output_summary="bundle not found",
            stderr_summary="",
            report_paths=command.report_paths,
        )
    return _run_subprocess(command, repo_root)


def command_summary_for_report() -> list[dict[str, object]]:
    def public_argv(argv: tuple[str, ...]) -> list[str]:
        values = list(argv)
        if values and Path(values[0]).name.lower().startswith("python"):
            values[0] = "python"
        return values

    return [
        {
            "command_id": command.command_id,
            "label": command.label,
            "group": command.group,
            "argv": public_argv(command.argv),
            "timeout_seconds": command.timeout_seconds,
            "requires_confirmation": command.requires_confirmation,
        }
        for command in COMMAND_ALLOWLIST.values()
    ]


def render_operator_control_panel() -> None:
    import streamlit as st

    st.header("Operator Control Panel")
    st.caption("Run safe local checks and maintenance actions.")
    st.caption("Only approved local checks are available.")

    if "operator_command_results" not in st.session_state:
        st.session_state.operator_command_results = {}

    summary = operator_panel_summary(st.session_state.operator_command_results)
    status_cols = st.columns(3)
    status_cols[0].metric("Last check", summary["last_check"])
    status_cols[1].metric("System status", summary["system_status"])
    status_cols[2].metric("Safety", summary["safety"])

    groups = command_groups()

    def render_commands(commands: list[OperatorCommand]) -> None:
        for command in commands:
            help_text = COMMAND_HELP_TEXT.get(command.command_id)
            if help_text:
                st.caption(help_text)
            disabled = False
            if command.requires_confirmation:
                disabled = not st.checkbox(
                    "Confirm long-running full test suite",
                    key=f"confirm_{command.command_id}",
                    help="The full test suite can take a long time.",
                )
            if st.button(command.label, key=f"run_{command.command_id}", disabled=disabled):
                with st.spinner(f"Running {command.label}..."):
                    st.session_state.operator_command_results[command.command_id] = run_operator_command(command.command_id)

            result = st.session_state.operator_command_results.get(command.command_id)
            if result is None:
                st.caption("Not run yet")
                continue
            st.caption(operator_status_label(result.status))
            st.write(
                {
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "duration_seconds": round(result.duration_seconds, 2),
                    "reports": list(result.report_paths),
                }
            )
            if result.output_summary:
                st.code(result.output_summary, language="text")
            if result.stderr_summary:
                st.warning(result.stderr_summary)

    st.subheader("Main checks")
    render_commands(groups.get("Main checks", []))

    for group in PANEL_GROUP_ORDER[1:]:
        with st.expander(group, expanded=False):
            render_commands(groups.get(group, []))
