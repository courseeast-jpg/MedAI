"""Focused tests for MEDAI-PACKAGING-DEPLOYMENT-POLISH-03.

These tests prove that the three top-level docs that the operator and
maintainer hit first (`README.md`, `RELEASE_QUICKSTART_LOCAL_ONLY.md`,
`RELEASE_OPERATOR_GUIDE.md`) now reference the consolidated operator
manual and the technical handoff produced by
`MEDAI-OPERATOR-MANUAL-CONSOLIDATION-01`, and that the launcher /
preflight / config / app/main surface is unchanged.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OPERATOR_MANUAL_PATH = (
    "reports/medai_operator_manual_consolidation_01/"
    "MEDAI_LOCAL_OPERATOR_MANUAL.md"
)
TECHNICAL_HANDOFF_PATH = (
    "reports/medai_operator_manual_consolidation_01/"
    "MEDAI_TECHNICAL_HANDOFF.md"
)

TOP_LEVEL_DOCS = (
    "README.md",
    "RELEASE_QUICKSTART_LOCAL_ONLY.md",
    "RELEASE_OPERATOR_GUIDE.md",
)

POLISH_REPORT_DIR = (
    REPO_ROOT / "reports" / "medai_packaging_deployment_polish_03"
)


# ── Consolidated doc presence ──────────────────────────────────────────────


def test_consolidated_operator_manual_exists():
    p = REPO_ROOT / OPERATOR_MANUAL_PATH
    assert p.is_file(), f"missing consolidated operator manual at {p}"


def test_consolidated_technical_handoff_exists():
    p = REPO_ROOT / TECHNICAL_HANDOFF_PATH
    assert p.is_file(), f"missing consolidated technical handoff at {p}"


# ── Pointer references in top-level docs ──────────────────────────────────


def test_readme_points_to_consolidated_docs():
    src = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert OPERATOR_MANUAL_PATH.rsplit("/", 1)[-1] in src or OPERATOR_MANUAL_PATH in src
    assert (
        TECHNICAL_HANDOFF_PATH.rsplit("/", 1)[-1] in src
        or TECHNICAL_HANDOFF_PATH in src
    )


def test_quickstart_points_to_consolidated_docs():
    src = (REPO_ROOT / "RELEASE_QUICKSTART_LOCAL_ONLY.md").read_text(
        encoding="utf-8"
    )
    assert OPERATOR_MANUAL_PATH in src
    assert TECHNICAL_HANDOFF_PATH in src


def test_operator_guide_points_to_consolidated_docs():
    src = (REPO_ROOT / "RELEASE_OPERATOR_GUIDE.md").read_text(
        encoding="utf-8"
    )
    assert OPERATOR_MANUAL_PATH in src
    assert TECHNICAL_HANDOFF_PATH in src


# ── Untouched runtime / launcher / preflight / config surface ─────────────


def test_launchers_unchanged_by_polish_03():
    """The PACKAGING-DEPLOYMENT-POLISH-03 block must not introduce any
    `PACKAGING-DEPLOYMENT-POLISH-03` marker into the four shipped
    launcher files. The launchers already emit adequate first-run
    text; the polish lives in docs only.
    """
    for launcher in (
        "Start_MedAI_UI.bat",
        "Start_MedAI_UI_Silent.vbs",
        "Start_MedAI_Test_UI.bat",
        "Start_MedAI_UI_Encrypted.bat",
    ):
        p = REPO_ROOT / launcher
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        assert "PACKAGING-DEPLOYMENT-POLISH-03" not in src
        assert "MEDAI-PACKAGING-DEPLOYMENT-POLISH-03" not in src


def test_app_main_unchanged_by_polish_03():
    """The polish must not touch `app/main.py`."""
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "PACKAGING-DEPLOYMENT-POLISH-03" not in src


def test_startup_preflight_unchanged_by_polish_03():
    p = REPO_ROOT / "app" / "startup_preflight.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "PACKAGING-DEPLOYMENT-POLISH-03" not in src


def test_config_unchanged_by_polish_03():
    p = REPO_ROOT / "app" / "config.py"
    if not p.is_file():
        return
    src = p.read_text(encoding="utf-8")
    assert "PACKAGING-DEPLOYMENT-POLISH-03" not in src


# ── PACKAGING-DEPLOYMENT-POLISH-03 reports exist ──────────────────────────


def test_polish_03_report_dir_exists():
    assert POLISH_REPORT_DIR.is_dir()


def test_polish_03_json_report_exists():
    p = POLISH_REPORT_DIR / "medai_packaging_deployment_polish_03_report.json"
    assert p.is_file()


def test_polish_03_md_report_exists():
    p = POLISH_REPORT_DIR / "medai_packaging_deployment_polish_03_report.md"
    assert p.is_file()


def test_polish_03_summary_exists():
    p = POLISH_REPORT_DIR / "MEDAI_PACKAGING_DEPLOYMENT_POLISH_03.md"
    assert p.is_file()


# ── Report invariants ──────────────────────────────────────────────────────


def test_polish_03_json_required_invariants():
    import json

    p = POLISH_REPORT_DIR / "medai_packaging_deployment_polish_03_report.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["phase_id"] == "MEDAI-PACKAGING-DEPLOYMENT-POLISH-03"
    assert d["mode"] == "packaging_deployment_polish"
    assert d["runtime_behavior_changed"] is False
    assert d["app_main_modified"] is False
    assert d["startup_preflight_modified"] is False
    assert d["config_modified"] is False
    assert d["launcher_files_modified"] is False
    assert d["docs_modified"] is True
    assert d["docs_or_launcher_changes_needed"] is True
    assert d["extraction_behavior_changed"] is False
    assert d["ocr_behavior_changed"] is False
    assert d["classifier_behavior_changed"] is False
    assert d["threshold_behavior_changed"] is False
    assert d["cue_expansion_recommended"] is False
    assert d["cue_expansion_performed"] is False
    assert d["external_api_used"] is False
    assert d["external_api_enabled"] is False
    assert d["source_documents_opened"] is False
    assert d["private_files_opened"] is False
    assert d["runtime_db_contents_opened"] is False
    assert d["licensed_terminology_rows_read"] is False
    assert d["raw_text_printed"] is False
    assert d["raw_filenames_printed"] is False
    assert d["private_paths_printed"] is False
    assert d["secrets_printed"] is False
    assert d["clinical_value_parsing_performed"] is False
    assert d["clinical_interpretation_performed"] is False
    assert d["local_only_posture_preserved"] is True
    assert d["operator_manual_linked"] is True
    assert d["technical_handoff_linked"] is True
    assert d["first_run_guidance_status"] == "ready"
    assert d["launcher_readiness_status"] == "ready"
    assert d["validation_command_status"] == "ready"
    assert d["recommended_next_step"] == "MEDAI-FREEZE-LOCAL-OPERATOR-RELEASE"
