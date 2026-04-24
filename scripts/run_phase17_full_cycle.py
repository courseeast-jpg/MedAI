from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SCRIPT = ROOT / "scripts" / "run_phase12_real_world_validation.py"
DASHBOARD_SCRIPT = ROOT / "scripts" / "run_phase17_dashboard.py"


def main(argv: list[str]) -> int:
    validation_command = [sys.executable, str(VALIDATION_SCRIPT), *argv]
    validation_result = subprocess.run(validation_command, cwd=ROOT)
    if validation_result.returncode != 0:
        return validation_result.returncode

    dashboard_result = subprocess.run(
        [sys.executable, str(DASHBOARD_SCRIPT), "--latest"],
        cwd=ROOT,
    )
    return dashboard_result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
