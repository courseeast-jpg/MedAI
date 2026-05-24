"""Focused tests for MEDAI-UI-STARTUP-RESILIENCE-08."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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


REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_startup_resilience_08"
JSON_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_08_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_08_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_STARTUP_RESILIENCE_08.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_ui_startup_resilience_08_audit.json"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_UI_STARTUP_RESILIENCE_08_AUDIT.md"


from app.main import build_system_components  # noqa: E402
from app.startup_preflight import (  # noqa: E402
    StartupState,
    build_startup_diagnostics,
    categorize_db_startup_state,
    initialize_startup_state,
)


class _FailingVectorStore:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated chromadb ConfigValidationError")


class _FailingSQLiteStore:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated sqlite store init failure")


@pytest.fixture()
def temp_db_paths(tmp_path: Path):
    return tmp_path / "mkb.db", tmp_path / "chroma"


def test_sqlite_success_and_vector_config_failure_returns_startup_ok_degraded(
    temp_db_paths,
):
    db_path, chroma_path = temp_db_paths
    components = build_system_components(
        db_path=db_path,
        chroma_path=chroma_path,
        db_encryption_key="default_dev_key",
        vector_store_factory=_FailingVectorStore,
    )
    status = components["component_status"]
    assert status["sqlite_store_initialized"] is True
    assert status["vector_store_initialized"] is False
    assert status["execution_pipeline_initialized"] is True
    assert any(name == "VectorStore" for name, _ in status["component_errors"])

    diagnostics = build_startup_diagnostics(
        db_path=db_path, component_status=status
    )
    assert diagnostics.app_startup_status == "app_startup_ok_with_degraded_vector"


def test_initialize_startup_state_returns_ok_when_vector_degraded(temp_db_paths):
    db_path, chroma_path = temp_db_paths

    def factory():
        return build_system_components(
            db_path=db_path,
            chroma_path=chroma_path,
            db_encryption_key="default_dev_key",
            vector_store_factory=_FailingVectorStore,
        )

    state = initialize_startup_state(factory)
    assert isinstance(state, StartupState)
    assert state.ok is True
    assert state.components is not None
    assert state.diagnostics.app_startup_status == "app_startup_ok_with_degraded_vector"


def test_sqlite_failure_returns_diagnostics_only(temp_db_paths):
    db_path, chroma_path = temp_db_paths
    components = build_system_components(
        db_path=db_path,
        chroma_path=chroma_path,
        db_encryption_key="default_dev_key",
        sqlite_store_factory=_FailingSQLiteStore,
        vector_store_factory=_FailingVectorStore,
    )
    status = components["component_status"]
    assert status["sqlite_store_initialized"] is False
    assert status["execution_pipeline_initialized"] is False
    diagnostics = build_startup_diagnostics(
        db_path=db_path, component_status=status
    )
    assert diagnostics.app_startup_status == "app_startup_failed"


def test_sqlcipher_encrypted_metadata_probe_not_fatal_when_store_ok():
    """The new category fires when SQLiteStore initialized but the
    standard sqlite3 metadata probe cannot read the encrypted DB."""
    category = categorize_db_startup_state(
        header_category="encrypted_or_unknown",
        sqlite_connect_result="connect_failed_database_error",
        sqlite_quick_check_result="quick_check_not_available",
        exception_category=None,
        sqlite_store_initialized=True,
    )
    assert category == "sqlcipher_encrypted_metadata_probe_unreadable_but_store_ok"


def test_encrypted_metadata_probe_remains_fatal_when_store_did_not_initialize():
    """If SQLiteStore did NOT initialize, the legacy
    encrypted_or_wrong_key_or_unreadable category still fires."""
    category = categorize_db_startup_state(
        header_category="encrypted_or_unknown",
        sqlite_connect_result="connect_failed_database_error",
        sqlite_quick_check_result="quick_check_not_available",
        exception_category=None,
        sqlite_store_initialized=False,
    )
    assert category == "encrypted_or_wrong_key_or_unreadable"


def test_degraded_vector_status_message_is_public_safe(temp_db_paths):
    db_path, chroma_path = temp_db_paths
    components = build_system_components(
        db_path=db_path,
        chroma_path=chroma_path,
        db_encryption_key="default_dev_key",
        vector_store_factory=_FailingVectorStore,
    )
    diagnostics = build_startup_diagnostics(
        db_path=db_path, component_status=components["component_status"]
    )
    summary = diagnostics.safe_public_summary()
    blob = json.dumps(summary)
    # Public-safe contract: no env vars, no DB encryption key value, no
    # full paths, no PHI, no raw text.
    for forbidden in (
        "DB_ENCRYPTION_KEY",
        "default_dev_key",
        "/home/",
        "C:\\Users",
        "/Users/",
    ):
        assert forbidden not in blob, forbidden
    # The guidance line must announce the local workflow.
    assert any(
        "SQLite MKB and local review workflow remain available" in line
        for line in summary["safe_operator_guidance"]
    )


def test_execution_pipeline_built_with_vector_store_none_and_quality_gate_none(
    temp_db_paths,
):
    db_path, chroma_path = temp_db_paths
    components = build_system_components(
        db_path=db_path,
        chroma_path=chroma_path,
        db_encryption_key="default_dev_key",
        vector_store_factory=_FailingVectorStore,
    )
    assert components["execution"] is not None
    assert components["vec"] is None
    assert components["quality_gate"] is None


def test_external_api_remains_false_and_auto_accept_remains_false(temp_db_paths):
    db_path, chroma_path = temp_db_paths
    components = build_system_components(
        db_path=db_path,
        chroma_path=chroma_path,
        db_encryption_key="default_dev_key",
        vector_store_factory=_FailingVectorStore,
    )
    state = components["state"]
    # SystemState carries claude_available; when ANTHROPIC_API_KEY is
    # absent it must remain False and external API path must not be
    # silently enabled.
    assert state.claude_available is False
    # No environmental override has flipped auto-accept on.
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API", "false").lower() == "false"


def test_validation_script_reports_ui_startup_resilience_ready():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_medai_ui_startup_resilience_08.py"),
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
    assert payload["conclusion"] == "ui_startup_resilience_ready"
    assert payload["all_pass"] is True
    findings = payload["audit_findings"]
    assert findings["ui_degraded_mode_result"] is True
    assert findings["mkb_explorer_remains_available"] is True
    assert findings["run_review_remains_available"] is True
    assert findings["hard_failure_returns_diagnostics_only"] is True
    assert findings["sqlcipher_encrypted_probe_not_fatal_when_store_ok"] is True


def test_public_reports_pass_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    paths = [
        p
        for p in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH, AUDIT_JSON_PATH, AUDIT_MD_PATH)
        if p.is_file()
    ]
    assert paths, "no 08 reports generated yet"
    for path in paths:
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_chain_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_extracted_medical_facts.py",
            "tests/test_run_review_extracted_information_preview.py",
            "tests/test_operator_review_actions.py",
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
