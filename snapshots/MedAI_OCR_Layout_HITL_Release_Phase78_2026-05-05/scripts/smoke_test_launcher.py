from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_DIRS = [
    ROOT / "test_input",
    ROOT / "test_output",
    ROOT / "test_review",
    ROOT / "test_archive",
    ROOT / "reports" / "test_runs",
]
LAUNCHER_PATH = ROOT / "Start_MedAI_Test_UI.bat"


def main() -> int:
    from app.test_launcher import ensure_test_launcher_dirs

    ensure_test_launcher_dirs()

    missing_dirs = [str(path) for path in REQUIRED_DIRS if not path.is_dir()]
    if missing_dirs:
        print(f"Missing required directories: {missing_dirs}")
        return 1

    if not LAUNCHER_PATH.is_file():
        print(f"Missing launcher: {LAUNCHER_PATH}")
        return 1

    launcher_text = LAUNCHER_PATH.read_text(encoding="utf-8")
    expected_command = "python -m streamlit run app/main.py"
    if expected_command not in launcher_text.replace("\\", "/"):
        print(f"Launcher does not contain expected command: {expected_command}")
        return 1

    for module_name in ("app.test_launcher", "app.main"):
        importlib.import_module(module_name)

    print("MedAI launcher smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

