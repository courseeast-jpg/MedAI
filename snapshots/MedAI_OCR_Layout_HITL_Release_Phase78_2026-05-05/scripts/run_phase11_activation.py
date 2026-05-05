from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "artifacts" / "phase11_activation"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests\\test_phase10_hardening.py",
        "tests\\test_phase11_governance.py",
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 11.1 controlled governance activation",
        "flags_enabled": {
            "ENABLE_HYPOTHESIS_TIER": True,
            "ENABLE_TRUTH_RESOLUTION": True,
            "ENABLE_DECISION_SCORING": True,
        },
        "command": command,
        "returncode": completed.returncode,
        "phase10_tests_still_pass": completed.returncode == 0,
        "activation_passed": completed.returncode == 0,
        "stdout_path": str(OUTPUT_DIR / "phase11_activation_stdout.txt"),
        "stderr_path": str(OUTPUT_DIR / "phase11_activation_stderr.txt"),
    }

    (OUTPUT_DIR / "phase11_activation_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (OUTPUT_DIR / "phase11_activation_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (OUTPUT_DIR / "phase11_activation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Phase 11.1 Activation Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Activation passed: {summary['activation_passed']}",
        f"- Phase 10 tests still pass: {summary['phase10_tests_still_pass']}",
        "- Flags enabled:",
        "  - ENABLE_HYPOTHESIS_TIER=true",
        "  - ENABLE_TRUTH_RESOLUTION=true",
        "  - ENABLE_DECISION_SCORING=true",
        f"- Command: `{' '.join(command)}`",
        f"- Return code: {completed.returncode}",
    ]
    (OUTPUT_DIR / "phase11_activation_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
