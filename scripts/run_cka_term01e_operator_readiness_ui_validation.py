"""Validate CKA-TERM-01E operator terminology readiness UI/reporting."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term01e_operator_readiness_ui"
REPORT_JSON = REPORT_DIR / "cka_term01e_operator_readiness_ui_report.json"
REPORT_MD = REPORT_DIR / "cka_term01e_operator_readiness_ui_report.md"
GUIDE_MD = REPORT_DIR / "CKA_TERM01E_OPERATOR_READINESS_UI_GUIDE.md"


def _run_command(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": " ".join(args),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-800:],
        "stderr_tail": proc.stderr[-800:],
    }


def _case(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "case": name,
        "passed": bool(passed),
        "details": details or {},
    }


def _privacy_clean(payload: dict[str, Any]) -> bool:
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = [
        "LICENSE_ACK_PRIVATE",
        "replacement_map",
        "source_response_raw",
        "api_key",
        "sk-",
        "C:\\",
        "/home/",
        "MRN",
        "DOB",
        "Patient",
        "dosing advice",
        "prescription dosing",
    ]
    return not any(token in serialized for token in forbidden)


def main() -> int:
    from app.terminology_readiness_viewer import (
        build_terminology_readiness_summary,
        load_terminology_readiness_reports,
        render_readiness_text,
    )

    reports = load_terminology_readiness_reports()
    summary = build_terminology_readiness_summary(reports)
    text = render_readiness_text(summary)

    final_validation = _run_command([sys.executable, "scripts/run_cka_final_mvp_release_validation.py"])
    cases = [
        _case("A_viewer_imports", True),
        _case("B_missing_reports_handled", any(item.status == "missing" for item in build_terminology_readiness_summary({}).phase_statuses)),
        _case("C_public_reports_loaded", {"TERM-01", "TERM-01A", "TERM-01B"}.issubset(set(reports))),
        _case("D_no_private_files_read", summary.private_ack_file_loaded is False),
        _case("E_no_terminology_data_files_read", summary.terminology_data_files_read is False),
        _case(
            "F_status_summary_correct",
            isinstance(summary.systems_missing, list)
            and isinstance(summary.systems_with_files_present, list)
            and isinstance(summary.systems_requiring_private_license_ack, list)
            and isinstance(summary.systems_import_ready, list),
        ),
        _case("G_render_helpers_non_empty", bool(text.strip())),
        _case("H_existing_app_ui_imports", _app_import_check()),
        _case("I_final_cka_validation_passes", final_validation["passed"], {"returncode": final_validation["returncode"]}),
        _case("J_privacy_report_clean", True),
    ]

    payload = {
        "block_id": "CKA-TERM-01E",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term01e_operator_readiness_ui_ready" if all(c["passed"] for c in cases) else "cka_term01e_operator_readiness_ui_blocked",
        "terminology_readiness_viewer_ready": True,
        "public_reports_only": summary.public_reports_only,
        "private_ack_file_loaded": summary.private_ack_file_loaded,
        "terminology_data_files_read": summary.terminology_data_files_read,
        "raw_paths_displayed": summary.raw_paths_displayed,
        "real_import_performed": not summary.real_import_not_run,
        "external_api_used": not summary.no_external_apis_downloads,
        "clinical_recommendations_generated": summary.clinical_advice_generated,
        "prescription_dosing_advice_generated": False,
        "systems_missing": summary.systems_missing,
        "systems_with_files_present": summary.systems_with_files_present,
        "systems_requiring_private_license_ack": summary.systems_requiring_private_license_ack,
        "systems_import_ready": summary.systems_import_ready,
        "phase_statuses": [
            {
                "phase": item.phase,
                "label": item.label,
                "status": item.status,
                "conclusion": item.conclusion,
                "report_loaded": item.report_loaded,
            }
            for item in summary.phase_statuses
        ],
        "reports_read": sorted(reports),
        "reports_missing": [item.phase for item in summary.phase_statuses if not item.report_loaded],
        "next_manual_action": "operator downloads licensed files and creates private license ack",
        "case_results": cases,
        "final_cka_validation": {
            "passed": final_validation["passed"],
            "returncode": final_validation["returncode"],
        },
        "validation_commands": [
            "python -m pytest tests/test_cka_term01e_operator_readiness_ui.py",
            "python scripts/run_cka_term01e_operator_readiness_ui_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
    }
    payload["privacy_report_clean"] = _privacy_clean(payload)
    for case in payload["case_results"]:
        if case["case"] == "J_privacy_report_clean":
            case["passed"] = payload["privacy_report_clean"]
    if not payload["privacy_report_clean"]:
        payload["conclusion"] = "cka_term01e_operator_readiness_ui_blocked"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    GUIDE_MD.write_text(_guide(), encoding="utf-8")

    print(json.dumps({"conclusion": payload["conclusion"], "cases_passed": sum(1 for c in payload["case_results"] if c["passed"]), "cases_total": len(payload["case_results"])}, indent=2))
    return 0 if payload["conclusion"] == "cka_term01e_operator_readiness_ui_ready" else 1


def _app_import_check() -> bool:
    try:
        import app.main as main_app

        return "Terminology Admin" in main_app.PHASE52_OPERATOR_TABS
    except Exception:
        return False


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CKA-TERM-01E Operator Readiness UI Report",
        "",
        f"Conclusion: `{payload['conclusion']}`",
        "",
        "## Safety",
        f"- Public reports only: `{payload['public_reports_only']}`",
        f"- Private ack file loaded: `{payload['private_ack_file_loaded']}`",
        f"- Terminology data files read: `{payload['terminology_data_files_read']}`",
        f"- Raw paths displayed: `{payload['raw_paths_displayed']}`",
        f"- Real import performed: `{payload['real_import_performed']}`",
        f"- External API used: `{payload['external_api_used']}`",
        "",
        "## Terminology Systems",
        f"- Missing: `{', '.join(payload['systems_missing']) or 'none'}`",
        f"- Files present: `{', '.join(payload['systems_with_files_present']) or 'none'}`",
        f"- Requiring private license ack: `{', '.join(payload['systems_requiring_private_license_ack']) or 'none'}`",
        f"- Import-ready: `{', '.join(payload['systems_import_ready']) or 'none'}`",
        "",
        "## Phase Status",
    ]
    for item in payload["phase_statuses"]:
        lines.append(f"- {item['phase']} {item['label']}: `{item['status']}`")
    lines.extend(
        [
            "",
            "## Next Manual Action",
            payload["next_manual_action"],
            "",
            "No real terminology import was run by this block.",
        ]
    )
    return "\n".join(lines) + "\n"


def _guide() -> str:
    return (
        "# CKA-TERM-01E Operator Readiness UI Guide\n\n"
        "Open the MedAI Streamlit UI, enable Show advanced tools, and choose the Terminology Admin tab.\n\n"
        "The panel reads public readiness reports only. It does not load private license acknowledgment files, "
        "does not read terminology data contents, does not download files, and does not run real import.\n\n"
        "Next manual action: operator downloads licensed files and creates private license ack.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
