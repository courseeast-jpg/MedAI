"""Focused tests for MEDAI-UI-STARTUP-RESILIENCE-09."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_streamlit() -> None:
    if "streamlit" in sys.modules:
        return

    class _Stub:
        class _NS:
            def __call__(self, *a, **k):
                return None

            def __getattr__(self, name):
                return _Stub._NS()

        def __getattr__(self, name):
            return _Stub._NS()

        def cache_resource(self, fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

    sys.modules["streamlit"] = _Stub()


_stub_streamlit()


REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_startup_resilience_09"
JSON_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_09_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_09_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_STARTUP_RESILIENCE_09.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_ui_startup_resilience_09_audit.json"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_UI_STARTUP_RESILIENCE_09_AUDIT.md"

REPAIR_COMMAND = "python scripts/run_medai_local_self_healing_validation_07.py"


from app.main import build_system_components  # noqa: E402
from app.startup_preflight import (  # noqa: E402
    build_startup_diagnostics,
    initialize_startup_state,
)


class _OkVectorStore:
    def __init__(self, *_a, **_kw):
        self.ok = True


class _OkPipeline:
    def __init__(self, *_a, **_kw):
        self.ok = True


class _FailingVectorStore:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated chromadb ConfigValidationError")


class _FailingPipeline:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated ExecutionPipeline init failure")


class _FailingSQLite:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated sqlite store init failure")


@pytest.fixture()
def tmp_paths(tmp_path: Path):
    return tmp_path / "mkb.db", tmp_path / "chroma"


def _state_for(
    *,
    db_path: Path,
    chroma_path: Path,
    sqlite_store_factory=None,
    vector_store_factory=None,
    execution_pipeline_factory=None,
):
    def factory():
        return build_system_components(
            db_path=db_path,
            chroma_path=chroma_path,
            db_encryption_key="default_dev_key",
            sqlite_store_factory=sqlite_store_factory,
            vector_store_factory=vector_store_factory,
            execution_pipeline_factory=execution_pipeline_factory,
        )

    return initialize_startup_state(factory)


def test_sqlite_ok_pipeline_failure_keeps_state_ok(tmp_paths):
    db, chroma = tmp_paths
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        vector_store_factory=_OkVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    assert state.ok is True
    assert state.components is not None
    assert state.components.get("sql") is not None
    assert state.components.get("execution") is None
    assert state.diagnostics.app_startup_status == "app_startup_ok_with_degraded_pipeline"


def test_sqlite_ok_pipeline_failure_does_not_say_mkb_initialization_failed(tmp_paths):
    db, chroma = tmp_paths
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        vector_store_factory=_OkVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    summary = state.diagnostics.safe_public_summary()
    for line in summary["safe_operator_guidance"]:
        assert "MKB initialization failed." not in line


def test_sqlite_ok_pipeline_failure_guidance_includes_repair_command(tmp_paths):
    db, chroma = tmp_paths
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        vector_store_factory=_OkVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    guidance = state.diagnostics.safe_public_summary()["safe_operator_guidance"]
    assert any(REPAIR_COMMAND in line for line in guidance)


def test_sqlite_ok_both_optional_failed_is_sqlite_only(tmp_paths):
    db, chroma = tmp_paths
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    assert state.ok is True
    assert state.diagnostics.app_startup_status == "app_startup_ok_sqlite_only"
    guidance = state.diagnostics.safe_public_summary()["safe_operator_guidance"]
    assert any(REPAIR_COMMAND in line for line in guidance)
    for line in guidance:
        assert "MKB initialization failed." not in line


def test_sqlite_failure_still_returns_diagnostics_only(tmp_paths):
    db, chroma = tmp_paths
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        sqlite_store_factory=_FailingSQLite,
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    assert state.ok is False
    assert state.components is None
    assert state.diagnostics.app_startup_status == "app_startup_failed"
    guidance = state.diagnostics.safe_public_summary()["safe_operator_guidance"]
    assert any("MKB initialization failed." in line for line in guidance)


def test_render_run_review_unavailable_panel_exists_in_main():
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "def render_run_review_unavailable_panel" in main_source
    assert "Document processing is unavailable in this startup mode" in main_source
    assert "python scripts/run_medai_local_self_healing_validation_07.py" in main_source


def test_run_review_tabs_guard_execution_none():
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    # Guard pattern in both Upload and Current Run tabs.
    occurrences = main_source.count(
        'if sys_components.get("execution") is None:'
    )
    assert occurrences >= 2, occurrences


def test_degraded_pipeline_banner_contains_exact_repair_command():
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "render_degraded_pipeline_banner" in main_source
    assert "python scripts/run_medai_local_self_healing_validation_07.py" in main_source
    assert "Processing pipeline unavailable" in main_source


def test_initialize_startup_state_ok_floor_is_sqlite_only(tmp_paths):
    """SQLite alone is the floor. Vector OK or Pipeline OK should not be required."""
    db, chroma = tmp_paths
    # SQLite OK + Vector failed + Pipeline OK -> ok=True, degraded vector
    state = _state_for(
        db_path=db,
        chroma_path=chroma,
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_OkPipeline,
    )
    assert state.ok is True
    assert state.diagnostics.app_startup_status == "app_startup_ok_with_degraded_vector"


def test_validation_script_reports_ui_startup_resilience_09_ready():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_medai_ui_startup_resilience_09.py"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "ui_startup_resilience_09_ready"
    assert payload["all_pass"] is True
    f = payload["audit_findings"]
    assert f["case_all_ok_status"] == "app_startup_ok"
    assert f["case_pipeline_failed_status"] == "app_startup_ok_with_degraded_pipeline"
    assert f["case_pipeline_failed_state_ok"] is True
    assert f["case_pipeline_failed_sql_remains_available"] is True
    assert f["case_pipeline_failed_execution_is_none"] is True
    assert f["case_pipeline_failed_no_mkb_init_failed_text"] is True
    assert f["case_both_optional_failed_status"] == "app_startup_ok_sqlite_only"
    assert f["case_sqlite_failed_status"] == "app_startup_failed"
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False


def test_public_reports_pass_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    paths = [
        p
        for p in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH, AUDIT_JSON_PATH, AUDIT_MD_PATH)
        if p.is_file()
    ]
    assert paths, "no 09 report files yet"
    for path in paths:
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_08_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_medai_ui_startup_resilience_08.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_existing_05_06_07_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_medai_local_one_doc_operator_validation_05.py",
            "tests/test_medai_local_one_click_operator_validation_06.py",
            "tests/test_medai_local_self_healing_validation_07.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
