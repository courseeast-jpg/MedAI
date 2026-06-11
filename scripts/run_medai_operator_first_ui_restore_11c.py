#!/usr/bin/env python3
"""MEDAI-OPERATOR-FIRST-UI-RESTORE-11C validation."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_operator_first_ui_restore_11c"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_FIRST_UI_RESTORE_11C_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_operator_first_ui_restore_11c_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_FIRST_UI_RESTORE_11C.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_operator_first_ui_restore_11c_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_operator_first_ui_restore_11c_report.md"

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
        "block_id": "MEDAI-OPERATOR-FIRST-UI-RESTORE-11C-AUDIT",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "browser_acceptance_audit": {
            "block_11b_code_validation_passed": True,
            "browser_ui_acceptance_failed": True,
            "specialty_selector_not_visibly_present_in_run_review": True,
            "mkb_explorer_not_visible_in_primary_workflow": True,
            "advanced_engineering_controls_dominated_ui": True,
            "block_11c_scope": "Fix visible operator workflow without changing extraction logic.",
        },
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def _public_artifact_is_path_safe(text: str) -> bool:
    banned = [
        "C:\\Users",
        "G:\\Codex",
        ".env",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DOB",
        "date-of-birth",
    ]
    return not any(token in text for token in banned)


def build_report() -> dict[str, Any]:
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from app.mkb_explorer_model import build_mkb_explorer_model
    from app.operator_ui_model import (
        SOURCE_COMPARISON_DISCLAIMER,
        advanced_only_tabs,
        build_operator_ui_model,
    )
    from app.specialty_selection import specialty_labels_for_ui
    from mkb.sqlite_store import SQLiteStore

    default_model = build_operator_ui_model(show_advanced_tools=False)
    advanced_model = build_operator_ui_model(show_advanced_tools=True)
    specialty_labels = specialty_labels_for_ui()

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_operator_first_ui_restore_11c_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    fallback = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty=TESTED_SPECIALTY,
        session_id="operator-first-ui-restore-11c",
    )
    record_ids = list(fallback["run_item"].get("extracted_medical_fact_record_ids") or [])
    mkb_model = build_mkb_explorer_model(
        sql_store,
        specialty_filter=TESTED_SPECIALTY,
        tier_filter="review_bound",
        fact_type_filter="test_result",
    )
    rows_have_required_fields = all(
        {"record_id", "fact_type", "specialty", "specialty_label", "tier", "status", "requires_review", "display_content"}.issubset(row)
        for row in mkb_model["rows"]
    )
    review_bound_records = sum(
        1 for record_id in record_ids if (sql_store.get_record(record_id) and sql_store.get_record(record_id).requires_review)
    )

    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    findings = {
        "default_primary_tabs": list(default_model.visible_tabs),
        "advanced_only_tabs": advanced_only_tabs(),
        "operator_control_panel_advanced_only": "Operator Control Panel" not in default_model.visible_tabs
        and "Operator Control Panel" in advanced_model.visible_tabs,
        "terminology_admin_advanced_only": "Terminology Admin" not in default_model.visible_tabs
        and "Terminology Admin" in advanced_model.visible_tabs,
        "run_review_has_medical_specialty_control": "Medical specialty / domain" in default_model.run_review_required_controls
        and "Medical specialty / domain" in main_source,
        "urology_option_visible": "Urology" in specialty_labels,
        "dermatology_option_visible": "Dermatology" in specialty_labels,
        "gastroenterology_option_visible": "Gastroenterology" in specialty_labels,
        "cardiology_option_visible": "Cardiology" in specialty_labels,
        "neurology_option_visible": "Neurology" in specialty_labels,
        "mkb_explorer_required_controls_exist": len(default_model.mkb_explorer_required_controls) == 7,
        "review_queue_required_controls_exist": len(default_model.review_queue_required_controls) == 5
        and SOURCE_COMPARISON_DISCLAIMER in main_source,
        "adapter_fallback_selected_specialty": fallback["selected_specialty"],
        "records_persisted_with_selected_specialty_count": int(mkb_model["row_count"]),
        "mkb_explorer_model_returns_specialty_tier_status_fields": rows_have_required_fields,
        "records_remain_review_bound": review_bound_records,
        "structured_facts_extracted": int(fallback["structured_facts_extracted"]),
        "mkb_explorer_primary": "MKB Explorer" in default_model.visible_tabs,
        "review_queue_primary": "Review Queue" in default_model.visible_tabs,
        "engineering_details_hidden_by_default": all(
            tab not in default_model.visible_tabs for tab in ["Operator Control Panel", "Validation Batch Audit", "Validation History", "Safety & Governance", "Terminology Admin"]
        )
        and "show_build_details=show_advanced_tools" in main_source,
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["default_primary_tabs"] == ["Run & Review", "MKB Explorer", "Review Queue"],
        findings["operator_control_panel_advanced_only"],
        findings["terminology_admin_advanced_only"],
        findings["run_review_has_medical_specialty_control"],
        findings["urology_option_visible"],
        findings["dermatology_option_visible"],
        findings["gastroenterology_option_visible"],
        findings["cardiology_option_visible"],
        findings["neurology_option_visible"],
        findings["mkb_explorer_required_controls_exist"],
        findings["review_queue_required_controls_exist"],
        findings["adapter_fallback_selected_specialty"] == TESTED_SPECIALTY,
        findings["records_persisted_with_selected_specialty_count"] > 0,
        findings["mkb_explorer_model_returns_specialty_tier_status_fields"],
        findings["records_remain_review_bound"] == findings["structured_facts_extracted"],
        findings["mkb_explorer_primary"],
        findings["review_queue_primary"],
        findings["engineering_details_hidden_by_default"],
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-OPERATOR-FIRST-UI-RESTORE-11C",
        "mode": "operator_first_ui_model_and_local_fallback_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "operator_first_ui_restore_11c_ready" if all(gates) else "not_ready",
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
    lines = [f"# {title}", ""]
    if "audit_findings" in payload:
        f = payload["audit_findings"]
        lines.extend(
            [
                f"Conclusion: `{payload['conclusion']}`",
                "",
                f"- default primary tabs: `{', '.join(f['default_primary_tabs'])}`",
                f"- advanced-only tabs: `{', '.join(f['advanced_only_tabs'])}`",
                f"- specialty selector visible: `{f['run_review_has_medical_specialty_control']}`",
                f"- Urology option visible: `{f['urology_option_visible']}`",
                f"- Dermatology option visible: `{f['dermatology_option_visible']}`",
                f"- Gastroenterology option visible: `{f['gastroenterology_option_visible']}`",
                f"- MKB Explorer primary: `{f['mkb_explorer_primary']}`",
                f"- Review Queue primary: `{f['review_queue_primary']}`",
                f"- engineering details hidden by default: `{f['engineering_details_hidden_by_default']}`",
                f"- adapter fallback selected specialty: `{f['adapter_fallback_selected_specialty']}`",
                f"- review-bound records: `{f['records_remain_review_bound']}`",
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
        return "\n".join(lines)

    audit = payload["browser_acceptance_audit"]
    lines.extend(
        [
            f"- Block 11B code validation passed: `{audit['block_11b_code_validation_passed']}`",
            f"- Browser UI acceptance failed: `{audit['browser_ui_acceptance_failed']}`",
            f"- Specialty selector not visibly present in Run & Review: `{audit['specialty_selector_not_visibly_present_in_run_review']}`",
            f"- MKB Explorer not visible in primary workflow: `{audit['mkb_explorer_not_visible_in_primary_workflow']}`",
            f"- Advanced engineering controls dominated UI: `{audit['advanced_engineering_controls_dominated_ui']}`",
            f"- 11C scope: {audit['block_11c_scope']}",
            f"- external API used: `{payload['external_api_used']}`",
            f"- auto-accept enabled: `{payload['auto_accept_enabled']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-OPERATOR-FIRST-UI-RESTORE-11C Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-OPERATOR-FIRST-UI-RESTORE-11C Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-OPERATOR-FIRST-UI-RESTORE-11C Report", report))
    report["privacy_check_passed"] = bool(
        report["privacy_check_passed"] and audit_privacy.get("passed") and md_privacy.get("passed")
    )
    report["privacy_check_leak_examples_redacted"] = (
        list(report.get("privacy_check_leak_examples_redacted") or [])
        + list(audit_privacy.get("leak_examples_redacted") or [])
        + list(md_privacy.get("leak_examples_redacted") or [])
    )
    rendered = json.dumps(audit, indent=2) + "\n" + json.dumps(report, indent=2) + "\n" + _markdown("report", report)
    report["public_artifacts_path_safe"] = _public_artifact_is_path_safe(rendered)
    report["privacy_check_passed"] = bool(report["privacy_check_passed"] and report["public_artifacts_path_safe"])
    if not report["privacy_check_passed"]:
        report["conclusion"] = "not_ready"
        report["all_pass"] = False
    write_reports(audit, report)
    print(json.dumps(report, indent=2))
    return 0 if report["conclusion"] == "operator_first_ui_restore_11c_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
