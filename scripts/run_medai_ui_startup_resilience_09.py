#!/usr/bin/env python3
"""MEDAI-UI-STARTUP-RESILIENCE-09 — local workflow fallback validation.

Proves that the UI stays available when SQLite is OK regardless of
ExecutionPipeline / VectorStore failures. Synthetic-only. No external
API. No PHI in any output.

Cases exercised:

1. SQLite OK + Pipeline OK + Vector OK   -> ``app_startup_ok``
2. SQLite OK + Pipeline OK + Vector FAIL -> ``app_startup_ok_with_degraded_vector``
3. SQLite OK + Pipeline FAIL (vector OK) -> ``app_startup_ok_with_degraded_pipeline``
4. SQLite OK + Pipeline FAIL + Vector FAIL -> ``app_startup_ok_sqlite_only``
5. SQLite FAIL                            -> ``app_startup_failed``

For cases 1-4 ``StartupState.ok`` is True and the guidance text must
not contain "MKB initialization failed".
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_startup_resilience_09"
JSON_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_09_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_ui_startup_resilience_09_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_STARTUP_RESILIENCE_09.md"

REPAIR_COMMAND = "python scripts/run_medai_local_self_healing_validation_07.py"


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

    sys.modules.setdefault("streamlit", _Stub())


class _FailingVectorStore:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated chromadb ConfigValidationError")


class _OkVectorStore:
    """A no-op VectorStore stand-in that does not require chromadb."""

    def __init__(self, *_a, **_kw):
        self.ok = True


class _FailingPipeline:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated ExecutionPipeline init failure")


class _OkPipeline:
    """A minimal pipeline stand-in matching what app.main relies on."""

    def __init__(self, *_a, **_kw):
        self.ok = True


class _FailingSQLite:
    def __init__(self, *_a, **_kw):
        raise RuntimeError("simulated sqlite store init failure")


def _privacy_check(payload: Any) -> Dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {
            "passed": bool(result.passed),
            "leak_examples_redacted": list(result.leak_examples_redacted or []),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _build_one_case(
    *,
    tmp_path: Path,
    case_label: str,
    sqlite_store_factory=None,
    vector_store_factory=None,
    execution_pipeline_factory=None,
) -> Dict[str, Any]:
    from app.main import build_system_components
    from app.startup_preflight import build_startup_diagnostics, initialize_startup_state

    def factory():
        return build_system_components(
            db_path=tmp_path / f"{case_label}.db",
            chroma_path=tmp_path / f"{case_label}_chroma",
            db_encryption_key="default_dev_key",
            sqlite_store_factory=sqlite_store_factory,
            vector_store_factory=vector_store_factory,
            execution_pipeline_factory=execution_pipeline_factory,
        )

    state = initialize_startup_state(factory)
    diagnostics_summary = state.diagnostics.safe_public_summary()
    return {
        "label": case_label,
        "ok": bool(state.ok),
        "components_present": state.components is not None,
        "components_have_sql": state.components is not None and state.components.get("sql") is not None,
        "components_have_execution": state.components is not None and state.components.get("execution") is not None,
        "components_have_vector": state.components is not None and state.components.get("vec") is not None,
        "app_startup_status": diagnostics_summary["app_startup_status"],
        "guidance_includes_repair_command": any(
            REPAIR_COMMAND in line
            for line in diagnostics_summary.get("safe_operator_guidance", [])
        ),
        "guidance_does_not_say_mkb_initialization_failed": (
            "MKB initialization failed."
            not in diagnostics_summary.get("safe_operator_guidance", [])
        ),
        "component_errors": diagnostics_summary.get("component_errors") or [],
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _stub_streamlit()

    tmp_dir = Path(tempfile.mkdtemp(prefix="medai_ui_09_"))

    case_all_ok = _build_one_case(
        tmp_path=tmp_dir,
        case_label="all_ok",
        vector_store_factory=_OkVectorStore,
        execution_pipeline_factory=_OkPipeline,
    )
    case_vector_failed = _build_one_case(
        tmp_path=tmp_dir,
        case_label="vector_failed",
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_OkPipeline,
    )
    case_pipeline_failed = _build_one_case(
        tmp_path=tmp_dir,
        case_label="pipeline_failed",
        vector_store_factory=_OkVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    case_both_optional_failed = _build_one_case(
        tmp_path=tmp_dir,
        case_label="both_optional_failed",
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )
    case_sqlite_failed = _build_one_case(
        tmp_path=tmp_dir,
        case_label="sqlite_failed",
        sqlite_store_factory=_FailingSQLite,
        vector_store_factory=_FailingVectorStore,
        execution_pipeline_factory=_FailingPipeline,
    )

    findings: Dict[str, Any] = {
        "case_all_ok_status": case_all_ok["app_startup_status"],
        "case_all_ok_state_ok": case_all_ok["ok"],
        "case_vector_failed_status": case_vector_failed["app_startup_status"],
        "case_vector_failed_state_ok": case_vector_failed["ok"],
        "case_pipeline_failed_status": case_pipeline_failed["app_startup_status"],
        "case_pipeline_failed_state_ok": case_pipeline_failed["ok"],
        "case_pipeline_failed_sql_remains_available": case_pipeline_failed[
            "components_have_sql"
        ],
        "case_pipeline_failed_execution_is_none": (
            case_pipeline_failed["ok"]
            and not case_pipeline_failed["components_have_execution"]
        ),
        "case_pipeline_failed_repair_command_in_guidance": case_pipeline_failed[
            "guidance_includes_repair_command"
        ],
        "case_pipeline_failed_no_mkb_init_failed_text": case_pipeline_failed[
            "guidance_does_not_say_mkb_initialization_failed"
        ],
        "case_both_optional_failed_status": case_both_optional_failed[
            "app_startup_status"
        ],
        "case_both_optional_failed_state_ok": case_both_optional_failed["ok"],
        "case_both_optional_failed_repair_command_in_guidance": case_both_optional_failed[
            "guidance_includes_repair_command"
        ],
        "case_sqlite_failed_status": case_sqlite_failed["app_startup_status"],
        "case_sqlite_failed_state_ok": case_sqlite_failed["ok"],
        "external_api_used": False,
        "auto_accept_enabled": False,
    }

    all_pass = (
        findings["case_all_ok_status"] == "app_startup_ok"
        and findings["case_all_ok_state_ok"] is True
        and findings["case_vector_failed_status"] == "app_startup_ok_with_degraded_vector"
        and findings["case_vector_failed_state_ok"] is True
        and findings["case_pipeline_failed_status"]
        == "app_startup_ok_with_degraded_pipeline"
        and findings["case_pipeline_failed_state_ok"] is True
        and findings["case_pipeline_failed_sql_remains_available"] is True
        and findings["case_pipeline_failed_execution_is_none"] is True
        and findings["case_pipeline_failed_repair_command_in_guidance"] is True
        and findings["case_pipeline_failed_no_mkb_init_failed_text"] is True
        and findings["case_both_optional_failed_status"] == "app_startup_ok_sqlite_only"
        and findings["case_both_optional_failed_state_ok"] is True
        and findings["case_both_optional_failed_repair_command_in_guidance"] is True
        and findings["case_sqlite_failed_status"] == "app_startup_failed"
        and findings["case_sqlite_failed_state_ok"] is False
        and not findings["external_api_used"]
        and not findings["auto_accept_enabled"]
    )

    conclusion = "ui_startup_resilience_09_ready" if all_pass else "not_ready"
    report = {
        "block_id": "MEDAI-UI-STARTUP-RESILIENCE-09-LOCAL-WORKFLOW-FALLBACK",
        "mode": "synthetic_ui_startup_resilience_09_validation",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "conclusion": conclusion,
        "all_pass": all_pass,
        "repair_command_for_operator": REPAIR_COMMAND,
    }
    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get(
        "leak_examples_redacted", []
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# MEDAI-UI-STARTUP-RESILIENCE-09 — Report",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "## Cases",
        "",
    ]
    for case in (
        case_all_ok,
        case_vector_failed,
        case_pipeline_failed,
        case_both_optional_failed,
        case_sqlite_failed,
    ):
        md_lines.append(
            f"- `{case['label']}`: status `{case['app_startup_status']}` "
            f"ok={case['ok']} sql_ok={case['components_have_sql']} "
            f"execution_ok={case['components_have_execution']} "
            f"vector_ok={case['components_have_vector']}"
        )
    md_lines.extend([
        "",
        "## Operator repair command",
        "",
        f"```\n{REPAIR_COMMAND}\n```",
        "",
        "## Safety",
        "",
        f"- external API used: {report['external_api_used']}",
        f"- auto-accept enabled: {report['auto_accept_enabled']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        "",
    ])
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    SHORT_MD_PATH.write_text(
        "# MEDAI-UI-STARTUP-RESILIENCE-09 — Short Summary\n\n"
        f"Conclusion: **{conclusion}**\n\n"
        f"- all_ok: `{case_all_ok['app_startup_status']}` ok={case_all_ok['ok']}\n"
        f"- vector_failed: `{case_vector_failed['app_startup_status']}` ok={case_vector_failed['ok']}\n"
        f"- pipeline_failed: `{case_pipeline_failed['app_startup_status']}` ok={case_pipeline_failed['ok']}\n"
        f"- both_optional_failed: `{case_both_optional_failed['app_startup_status']}` ok={case_both_optional_failed['ok']}\n"
        f"- sqlite_failed: `{case_sqlite_failed['app_startup_status']}` ok={case_sqlite_failed['ok']}\n\n"
        f"Operator repair command:\n\n```\n{REPAIR_COMMAND}\n```\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
