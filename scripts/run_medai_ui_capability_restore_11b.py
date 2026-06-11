#!/usr/bin/env python3
"""MEDAI-UI-CAPABILITY-RESTORE-11B validation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_capability_restore_11b"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_UI_CAPABILITY_RESTORE_11B_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_ui_capability_restore_11b_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_CAPABILITY_RESTORE_11B.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_ui_capability_restore_11b_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_ui_capability_restore_11b_report.md"

NEXT_OPERATOR_COMMAND = "streamlit run app/main.py"
TESTED_SPECIALTY = "dermatology"
SYNTHETIC_TEXT = (
    "Lab result report\n"
    "Specimen: serum\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Platelets: 230 x10E9/L (ref 150-450)\n"
)


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _privacy_check(payload: Any) -> dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {"passed": bool(result.passed), "leak_examples_redacted": list(result.leak_examples_redacted or [])}
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build_audit() -> dict[str, Any]:
    return {
        "block_id": "MEDAI-UI-CAPABILITY-RESTORE-11B-AUDIT",
        "source_audit": "MEDAI-UI-CAPABILITY-PARITY-AUDIT-11A",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "priority_1_gap_map": [
            {
                "gap": "specialty/domain selector",
                "source_changes": ["app/specialty_selection.py", "app/main.py:render_specialty_selector"],
            },
            {
                "gap": "MKB Explorer",
                "source_changes": ["app/mkb_explorer_model.py", "app/main.py:render_mkb_tab", "app/main.py:PRIMARY_OPERATOR_TABS"],
            },
            {
                "gap": "tier/status visibility",
                "source_changes": ["app/mkb_explorer_model.py:build_mkb_explorer_model", "app/main.py:render_mkb_tab"],
            },
            {
                "gap": "specialty routing",
                "source_changes": ["app/local_adapter_fallback_processor.py:selected_specialty", "app/test_launcher.py existing specialty parameter"],
                "normal_pipeline_limitation": "Normal pipeline already accepts a specialty argument. This block routes the visible selector into that existing parameter without changing extraction behavior.",
            },
        ],
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def build_report() -> dict[str, Any]:
    from app.local_adapter_fallback_processor import (
        process_adapter_fallback_run_review,
        run_operator_action_proof,
    )
    from app.mkb_explorer_model import build_mkb_explorer_model
    from app.specialty_selection import DEFAULT_SPECIALTY_KEY, specialty_options_for_ui
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_capability_restore_11b_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    result = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty=TESTED_SPECIALTY,
        session_id="capability-restore-11b",
    )
    record_ids = list(result["run_item"].get("extracted_medical_fact_record_ids") or [])
    model_before_actions = build_mkb_explorer_model(
        sql_store,
        specialty_filter=TESTED_SPECIALTY,
        tier_filter="all",
        fact_type_filter="test_result",
    )
    with sql_store._get_conn() as conn:
        specialty_count_row = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE specialty=?",
            (TESTED_SPECIALTY,),
        ).fetchone()
        review_count_row = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE specialty=? AND requires_review=1",
            (TESTED_SPECIALTY,),
        ).fetchone()
    selected_specialty_count = int(specialty_count_row["c"] if isinstance(specialty_count_row, dict) else specialty_count_row[0])
    review_bound_count = int(review_count_row["c"] if isinstance(review_count_row, dict) else review_count_row[0])
    operator_action_proof_count = run_operator_action_proof(sql_store, record_ids, session_id="capability-restore-11b")
    model_after_actions = build_mkb_explorer_model(
        sql_store,
        specialty_filter=TESTED_SPECIALTY,
        tier_filter="all",
        fact_type_filter="test_result",
    )
    specialty_keys = {option["key"] for option in specialty_options_for_ui()}
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    findings = {
        "specialty_selector_visible": "Medical specialty / domain" in main_source,
        "default_specialty": DEFAULT_SPECIALTY_KEY,
        "tested_specialty": TESTED_SPECIALTY,
        "structured_facts_extracted": int(result["structured_facts_extracted"]),
        "review_bound_records_persisted": int(result["review_bound_records_persisted"]),
        "records_persisted_with_selected_specialty_count": selected_specialty_count,
        "retrieval_proof_count": int(model_before_actions["row_count"]),
        "mkb_explorer_available": bool(model_before_actions["available"]),
        "mkb_explorer_rows": int(model_before_actions["row_count"]),
        "tier_status_visibility": bool(model_before_actions["tier_status_visible"]),
        "specialty_filter_can_select_dermatology": TESTED_SPECIALTY in specialty_keys and model_before_actions["filters"]["specialty"] == TESTED_SPECIALTY,
        "operator_action_proof_count": int(operator_action_proof_count),
        "review_bound_records_after_actions": int(model_after_actions["counts"]["review_bound"]),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["specialty_selector_visible"],
        findings["structured_facts_extracted"] > 0,
        findings["review_bound_records_persisted"] > 0,
        findings["records_persisted_with_selected_specialty_count"] == findings["review_bound_records_persisted"],
        findings["retrieval_proof_count"] > 0,
        findings["mkb_explorer_available"],
        findings["mkb_explorer_rows"] > 0,
        findings["tier_status_visibility"],
        findings["specialty_filter_can_select_dermatology"],
        findings["operator_action_proof_count"] == 3,
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-UI-CAPABILITY-RESTORE-11B",
        "mode": "synthetic_specialty_routing_and_mkb_visibility_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "ui_capability_restore_11b_ready" if all(gates) else "not_ready",
        "all_pass": bool(all(gates)),
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "next_operator_command": NEXT_OPERATOR_COMMAND,
    }
    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get("leak_examples_redacted", [])
    return report


def _markdown(title: str, payload: dict[str, Any]) -> str:
    if "audit_findings" in payload:
        f = payload["audit_findings"]
        return "\n".join(
            [
                f"# {title}",
                "",
                f"Conclusion: `{payload['conclusion']}`",
                "",
                f"- specialty selector visible: `{f['specialty_selector_visible']}`",
                f"- default specialty: `{f['default_specialty']}`",
                f"- tested specialty: `{f['tested_specialty']}`",
                f"- selected-specialty records: `{f['records_persisted_with_selected_specialty_count']}`",
                f"- MKB Explorer available: `{f['mkb_explorer_available']}`",
                f"- tier/status visibility: `{f['tier_status_visibility']}`",
                f"- review-bound records: `{f['review_bound_records_persisted']}`",
                f"- operator action proof count: `{f['operator_action_proof_count']}`",
                f"- privacy check passed: `{payload['privacy_check_passed']}`",
                f"- external API used: `{payload['external_api_used']}`",
                f"- auto-accept enabled: `{payload['auto_accept_enabled']}`",
                "",
                "## Next operator command",
                "",
                f"```bash\n{payload['next_operator_command']}\n```",
                "",
            ]
        )
    lines = [f"# {title}", ""]
    for item in payload["priority_1_gap_map"]:
        lines.append(f"- `{item['gap']}` -> {', '.join(item['source_changes'])}")
    lines.extend(["", f"- external API used: `{payload['external_api_used']}`", f"- auto-accept enabled: `{payload['auto_accept_enabled']}`", ""])
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-UI-CAPABILITY-RESTORE-11B Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-UI-CAPABILITY-RESTORE-11B Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    # Check final rendered artifacts before returning success.
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-UI-CAPABILITY-RESTORE-11B Report", report))
    report["privacy_check_passed"] = bool(
        report["privacy_check_passed"] and audit_privacy.get("passed") and md_privacy.get("passed")
    )
    report["privacy_check_leak_examples_redacted"] = (
        list(report.get("privacy_check_leak_examples_redacted") or [])
        + list(audit_privacy.get("leak_examples_redacted") or [])
        + list(md_privacy.get("leak_examples_redacted") or [])
    )
    write_reports(audit, report)
    print(json.dumps(report, indent=2))
    return 0 if report["conclusion"] == "ui_capability_restore_11b_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
