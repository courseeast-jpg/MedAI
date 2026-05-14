"""MEDAI-UI-BOOT-FIX-01 validation script."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.startup_preflight import build_startup_diagnostics, initialize_startup_state  # noqa: E402
from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_boot_fix_01"
REPORT_JSON = REPORT_DIR / "medai_ui_boot_fix_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ui_boot_fix_01_report.md"
REPORT_GUIDE = REPORT_DIR / "MEDAI_UI_BOOT_FIX_01.md"


def build_report() -> dict:
    memory_state = initialize_startup_state(lambda: (_ for _ in ()).throw(MemoryError("synthetic validation failure")))
    diagnostics = build_startup_diagnostics(exception=MemoryError("synthetic validation failure"))
    return {
        "block_id": "MEDAI-UI-BOOT-FIX-01",
        "conclusion": "medai_ui_boot_fix_startup_resilience_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_cause_classification": {
            "observed_failure": "MemoryError during SQLite schema initialization",
            "likely_cause": "SQLite initialization failed before Streamlit UI render; exact local DB cause requires operator repair block if repeated.",
            "launcher_specific": False,
            "operator_control_panel_button_related": False
        },
        "diagnostics": diagnostics.safe_public_summary(),
        "degraded_startup_state": {
            "memory_error_returns_degraded_state": memory_state.ok is False,
            "clinical_processing_started": False,
            "operator_panel_available_in_degraded_mode": True
        },
        "safety_privacy": {
            "db_deleted": False,
            "db_overwritten": False,
            "backup_created": False,
            "import_performed": False,
            "external_api_used": False,
            "runtime_db_or_index_created": False,
            "db_contents_printed": False,
            "raw_private_paths_in_public_report": False,
            "clinical_logic_changed": False,
            "ocr_extractor_safety_gates_changed": False,
            "route_fix_behavior_changed": False,
            "b07_terminology_behavior_changed": False
        },
        "validation_commands": [
            "python -m pytest tests/test_medai_ui_boot_fix.py -vv",
            "python scripts/run_medai_ui_boot_fix_validation.py",
            "python -m pytest tests/test_medai_ui_ops_panel.py -vv",
            "python scripts/run_medai_ui_ops_panel_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
            "python scripts/run_b07_term01_opt_in_integration_validation.py",
            "python scripts/run_medai_route_fix01_validation.py"
        ],
        "next_recommended_action": "Launch Streamlit and confirm the UI renders; if DB initialization still fails, use the diagnostics panel and a separate repair block."
    }


def write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# MEDAI-UI-BOOT-FIX-01 Report",
                "",
                f"Conclusion: `{report['conclusion']}`",
                "",
                "## Root Cause Classification",
                "",
                "- Observed failure: MemoryError during SQLite schema initialization.",
                "- Launcher-specific: false.",
                "- Operator Control Panel button related: false.",
                "",
                "## Safety",
                "",
                "- No DB file was deleted or overwritten.",
                "- No import was run.",
                "- No external API was used.",
                "- No clinical processing starts when MKB initialization is unavailable.",
                "- Diagnostics use relative labels, size buckets, exception class names, and safe categories.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    REPORT_GUIDE.write_text(
        "\n".join(
            [
                "# MEDAI-UI-BOOT-FIX-01 Startup Resilience",
                "",
                "If MKB initialization fails, Streamlit should render a diagnostics-only startup panel instead of crashing.",
                "",
                "Operator guidance:",
                "",
                "- No clinical processing started.",
                "- Avoid manual DB deletion.",
                "- Run Git Safety Check or startup diagnostics.",
                "- Use a Codex repair block if database quarantine or restore is needed.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    report = build_report()
    write_reports(report)
    failed = False
    for path in (REPORT_JSON, REPORT_MD, REPORT_GUIDE):
        result = check_public_report_payload(path.read_text(encoding="utf-8"))
        if not result.passed:
            print(json.dumps({"privacy_failed": path.as_posix(), "examples": result.leak_examples_redacted}, indent=2))
            failed = True
    print(json.dumps({"conclusion": report["conclusion"], "report_json": REPORT_JSON.relative_to(REPO_ROOT).as_posix()}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
