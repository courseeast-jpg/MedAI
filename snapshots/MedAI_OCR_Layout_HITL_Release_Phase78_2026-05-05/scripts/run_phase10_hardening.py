from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_command(command: list[str], *, cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "phase10"))
    parser.add_argument("--pytest-target", nargs="*", default=["tests"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    junit_path = output_dir / "phase10_pytest_junit.xml"
    pytest_cmd = [sys.executable, "-m", "pytest", *args.pytest_target, f"--junitxml={junit_path}"]
    pytest_result = run_command(pytest_cmd, cwd=ROOT)
    commands.append(pytest_result)

    (output_dir / "phase10_pytest_stdout.txt").write_text(pytest_result["stdout"], encoding="utf-8")
    (output_dir / "phase10_pytest_stderr.txt").write_text(pytest_result["stderr"], encoding="utf-8")

    audit_cmd = [sys.executable, str(ROOT / "scripts" / "generate_phase10_audit_report.py"), "--output-dir", str(output_dir)]
    audit_result = run_command(audit_cmd, cwd=ROOT)
    commands.append(audit_result)

    (output_dir / "phase10_audit_stdout.txt").write_text(audit_result["stdout"], encoding="utf-8")
    (output_dir / "phase10_audit_stderr.txt").write_text(audit_result["stderr"], encoding="utf-8")

    audit_report = {}
    audit_report_path = output_dir / "phase10_audit_report.json"
    if audit_report_path.exists():
        audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))

    overall_passed = pytest_result["returncode"] == 0 and audit_result["returncode"] == 0 and bool(audit_report.get("overall_passed", False))
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 10 Hardening Protocol",
        "overall_passed": overall_passed,
        "pytest_passed": pytest_result["returncode"] == 0,
        "audit_report_passed": audit_result["returncode"] == 0 and bool(audit_report.get("overall_passed", False)),
        "commands": [
            {"command": item["command"], "returncode": item["returncode"]}
            for item in commands
        ],
        "artifacts": {
            "pytest_junit_xml": str(junit_path),
            "pytest_stdout": str(output_dir / "phase10_pytest_stdout.txt"),
            "pytest_stderr": str(output_dir / "phase10_pytest_stderr.txt"),
            "audit_report_json": str(audit_report_path),
            "audit_report_md": str(output_dir / "phase10_audit_report.md"),
        },
    }

    summary_json_path = output_dir / "phase10_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = [
        "# Phase 10 Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Overall passed: {summary['overall_passed']}",
        f"- Pytest passed: {summary['pytest_passed']}",
        f"- Audit report passed: {summary['audit_report_passed']}",
        "",
        "## Commands",
        "",
    ]
    for item in summary["commands"]:
        summary_md.append(f"- `{' '.join(item['command'])}` -> exit {item['returncode']}")
    (output_dir / "phase10_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
