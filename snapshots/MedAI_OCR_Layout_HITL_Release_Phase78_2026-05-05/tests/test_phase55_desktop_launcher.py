from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "Start_MedAI_UI.bat"
VBS = ROOT / "Start_MedAI_UI_Silent.vbs"
README = ROOT / "START_MEDAI_HERE.txt"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_phase55_start_bat_exists_and_runs_streamlit_app():
    source = read(BAT)

    assert BAT.exists()
    assert "app\\main.py" in source
    assert "python -m streamlit run" in source


def test_phase55_start_bat_sets_local_only_and_privacy_defaults():
    source = read(BAT)

    assert "set MEDAI_LOCAL_ONLY=1" in source
    assert "set MEDAI_ALLOW_EXTERNAL_API=0" in source
    assert "set MEDAI_REQUIRE_PII_SCRUB=1" in source
    assert "set MEDAI_PRIVACY_AUDIT=1" in source


def test_phase55_start_bat_uses_expected_port_and_browser_url():
    source = read(BAT)

    assert "--server.port 8501" in source
    assert "http://localhost:8501" in source
    assert "Browser opening at localhost:8501" in source


def test_phase55_user_start_file_exists_and_has_basic_instructions():
    source = read(README)

    assert README.exists()
    assert "Double-click Start_MedAI_UI.bat" in source
    assert "http://localhost:8501" in source
    assert "Ctrl+C" in source
    assert "Local-only mode is ON by default" in source


def test_phase55_optional_vbs_keeps_console_visible():
    source = read(VBS)

    assert VBS.exists()
    assert "Start_MedAI_UI.bat" in source
    assert ", 1, False" in source


def test_phase55_launcher_files_do_not_contain_api_keys_or_phi_paths():
    combined = "\n".join(read(path) for path in [BAT, VBS, README])
    forbidden = [
        "GEMINI_API_KEY=",
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        "MRN",
        "DOB",
        "patient",
        "real_validation_input\\",
        "test_review\\",
        "test_archive\\",
    ]

    for token in forbidden:
        assert token not in combined


def test_phase55_launcher_files_do_not_enable_external_apis():
    combined = "\n".join(read(path).lower() for path in [BAT, VBS, README])

    assert "medai_allow_external_api=1" not in combined
    assert "medai_allow_external_api=true" not in combined
    assert "external apis disabled" in combined
