from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests\\test_phase11_governance.py"],
        cwd=str(ROOT),
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
