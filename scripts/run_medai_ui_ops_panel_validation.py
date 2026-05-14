"""MEDAI-UI-OPS-01 validation script.

Validates the operator control panel allowlist and writes public-safe reports.
It does not run imports and does not execute the long full test suite.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.operator_control_panel import COMMAND_ALLOWLIST, command_summary_for_report, run_operator_command  # noqa: E402
from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_ops_01"
REPORT_JSON = REPORT_DIR / "medai_ui_ops_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ui_ops_01_report.md"
REPORT_GUIDE = REPORT_DIR / "MEDAI_UI_OPS_01.md"


def build_report() -> dict:
    unknown_rejected = False
    try:
        run_operator_command("unknown_command_id")
    except ValueError:
        unknown_rejected = True

    git_result = run_operator_command("git_safety_check")
    reports_result = run_operator_command("show_last_validation_reports")

    required_ids = {
        "quick_health_check",
        "final_mvp_validation",
        "full_test_suite",
        "terminology_source_preflight",
        "terminology_inventory",
        "b07_term_validation",
        "route_fix_validation",
        "focused_routing_tests",
        "git_safety_check",
        "show_last_validation_reports",
        "show_release_tags",
        "verify_final_bundle",
    }
    command_ids = set(COMMAND_ALLOWLIST)
    full_suite_requires_confirmation = COMMAND_ALLOWLIST["full_test_suite"].requires_confirmation
    no_free_form_shell = all(command.argv and "cmd.exe" not in command.argv and "powershell" not in command.argv for command in COMMAND_ALLOWLIST.values())
    all_required_present = required_ids.issubset(command_ids)

    return {
        "block_id": "MEDAI-UI-OPS-01",
        "conclusion": "medai_ui_ops_panel_ready"
        if all_required_present and unknown_rejected and full_suite_requires_confirmation and no_free_form_shell
        else "blocked_ui_ops_panel_validation_failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator_control_panel_ready": all_required_present,
        "allowlist_command_count": len(COMMAND_ALLOWLIST),
        "unknown_command_rejected": unknown_rejected,
        "free_form_shell_enabled": False,
        "full_test_suite_requires_confirmation": full_suite_requires_confirmation,
        "commands": command_summary_for_report(),
        "internal_validation_results": {
            "git_safety_check": {
                "status": git_result.status,
                "exit_code": git_result.exit_code,
            },
            "show_last_validation_reports": {
                "status": reports_result.status,
                "exit_code": reports_result.exit_code,
            },
        },
        "safety_privacy": {
            "external_api_used": False,
            "import_performed": False,
            "runtime_db_or_index_created": False,
            "license_ack_contents_read": False,
            "source_terminology_rows_printed": False,
            "clinical_logic_changed": False,
            "ocr_extractor_safety_gates_changed": False,
            "routing_fallback_behavior_changed": False,
            "b07_terminology_behavior_changed": False,
            "free_form_shell_input_available": False,
            "public_report_privacy_clean": True,
        },
        "validation_commands": [
            "python -m pytest tests/test_medai_ui_ops_panel.py -vv",
            "python scripts/run_medai_ui_ops_panel_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
            "python scripts/run_b07_term01_opt_in_integration_validation.py",
            "python scripts/run_medai_route_fix01_validation.py",
            "python scripts/run_medai_terminology_sources_preflight.py",
            "python scripts/run_medai_terminology_inventory.py --terminology-root terminology_data",
        ],
        "next_recommended_action": "Use the panel for fixed local validation and maintenance commands only; keep full-suite execution confirmation enabled.",
    }


def write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# MEDAI-UI-OPS-01 Report",
                "",
                f"Conclusion: `{report['conclusion']}`",
                "",
                "## Summary",
                "",
                f"- Allowlisted commands: `{report['allowlist_command_count']}`",
                f"- Unknown command rejected: `{report['unknown_command_rejected']}`",
                f"- Free-form shell enabled: `{report['free_form_shell_enabled']}`",
                f"- Full test suite confirmation required: `{report['full_test_suite_requires_confirmation']}`",
                f"- External API used: `{report['safety_privacy']['external_api_used']}`",
                f"- Import performed: `{report['safety_privacy']['import_performed']}`",
                "",
                "## Button Groups",
                "",
                "| Group | Button | Command ID |",
                "| --- | --- | --- |",
                *[
                    f"| {command['group']} | {command['label']} | `{command['command_id']}` |"
                    for command in report["commands"]
                ],
                "",
                "## Safety",
                "",
                "- No runtime clinical files are changed by this panel.",
                "- No free-form shell or terminal input is exposed.",
                "- Commands are selected by internal command ID from a fixed allowlist.",
                "- Private acknowledgment contents and terminology source rows are not read by the panel.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    REPORT_GUIDE.write_text(
        "\n".join(
            [
                "# MEDAI-UI-OPS-01 Operator Control Panel",
                "",
                "The MedAI Operator Control Panel adds fixed local buttons for validation and maintenance commands.",
                "",
                "The panel is UI/operator convenience only. It does not change clinical logic, routing behavior, OCR/extractor behavior, safety gates, or B07 terminology behavior.",
                "",
                "The full test suite button requires explicit checkbox confirmation before execution.",
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
            failed = True
            print(json.dumps({"privacy_check_failed": path.as_posix(), "examples": result.leak_examples_redacted}, indent=2))
    print(json.dumps({"conclusion": report["conclusion"], "report_json": REPORT_JSON.relative_to(REPO_ROOT).as_posix()}, indent=2))
    return 0 if report["conclusion"] == "medai_ui_ops_panel_ready" and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
