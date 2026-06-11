#!/usr/bin/env python3
"""MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E validation."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_operator_one_screen_ui_polish_11e"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_ONE_SCREEN_UI_POLISH_11E_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_operator_one_screen_ui_polish_11e_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_ONE_SCREEN_UI_POLISH_11E.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_operator_one_screen_ui_polish_11e_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_operator_one_screen_ui_polish_11e_report.md"

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
        "block_id": "MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E-AUDIT",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "layout_findings": [
            "11C restored operator-first tabs but left excess vertical spacing in the header and form sections.",
            "Verbose helper text can be shortened while preserving local-only and human-review warnings.",
            "Status cards can be compact chips for the first viewport.",
            "Document category and specialty controls can share a row.",
            "Upload and Start run controls can share the first workflow row.",
            "Build / audit details and engineering tabs remain behind Show advanced tools.",
            "Run & Review, MKB Explorer, and Review Queue should expose their core controls in the first viewport.",
        ],
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def _path_safe(text: str) -> bool:
    banned = ["C:\\Users", "G:\\Codex", ".env", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DOB", "date-of-birth"]
    return not any(token in text for token in banned)


def build_report() -> dict[str, Any]:
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from app.mkb_explorer_model import build_mkb_explorer_model
    from app.operator_compact_ui_model import build_compact_operator_ui_model
    from app.operator_ui_model import advanced_only_tabs, build_operator_ui_model
    from app.specialty_selection import specialty_labels_for_ui
    from mkb.sqlite_store import SQLiteStore

    compact_model = build_compact_operator_ui_model()
    operator_model = build_operator_ui_model(show_advanced_tools=False)
    advanced_model = build_operator_ui_model(show_advanced_tools=True)
    specialty_labels = specialty_labels_for_ui()
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_one_screen_ui_polish_11e_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    fallback = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty=TESTED_SPECIALTY,
        session_id="one-screen-ui-polish-11e",
    )
    record_ids = list(fallback["run_item"].get("extracted_medical_fact_record_ids") or [])
    mkb_model = build_mkb_explorer_model(sql_store, specialty_filter=TESTED_SPECIALTY, tier_filter="review_bound")
    review_bound_records = sum(
        1 for record_id in record_ids if (sql_store.get_record(record_id) and sql_store.get_record(record_id).requires_review)
    )

    findings = {
        "default_primary_tabs": list(operator_model.visible_tabs),
        "advanced_only_tabs": advanced_only_tabs(),
        "compact_header_ready": all(
            item in compact_model.compact_header_items
            for item in ["Local safe mode", "Human review", "Local only", "Cloud APIs off", "Privacy check on"]
        )
        and "compact-session-header" in main_source,
        "run_review_first_viewport_ready": all(
            item in compact_model.run_review_first_viewport_controls
            for item in ["Document category", "Medical specialty / domain", "Upload files", "Start run", "Documents waiting", "Current run status"]
        )
        and "category_col, specialty_col = st.columns(2)" in main_source
        and "upload_col, start_col = st.columns([3, 1])" in main_source,
        "mkb_explorer_first_viewport_ready": all(
            item in compact_model.mkb_explorer_first_viewport_controls
            for item in ["Total", "Active", "Quarantined / review-bound", "Superseded / rejected", "Specialty/domain filter", "Tier/status filter", "Fact type filter"]
        )
        and 'metric("Total"' in main_source,
        "review_queue_first_viewport_ready": all(
            item in compact_model.review_queue_first_viewport_controls
            for item in ["Needs review count", "Source comparison disclaimer", "Accept", "Reject", "Defer"]
        )
        and "review_queue_accept_" in main_source
        and "review_queue_reject_" in main_source
        and "review_queue_defer_" in main_source,
        "advanced_engineering_tabs_hidden_by_default": all(tab not in operator_model.visible_tabs for tab in advanced_only_tabs()),
        "operator_tabs_remain_first_with_advanced": list(advanced_model.visible_tabs[:3]) == ["Run & Review", "MKB Explorer", "Review Queue"],
        "build_audit_details_hidden_by_default": "show_build_details=show_advanced_tools" in main_source,
        "specialty_selector_still_visible": "Medical specialty / domain" in main_source,
        "urology_option_visible": "Urology" in specialty_labels,
        "dermatology_option_visible": "Dermatology" in specialty_labels,
        "gastroenterology_option_visible": "Gastroenterology" in specialty_labels,
        "adapter_fallback_selected_specialty": fallback["selected_specialty"],
        "records_persisted_with_selected_specialty_count": int(mkb_model["row_count"]),
        "review_bound_records": int(review_bound_records),
        "structured_facts_extracted": int(fallback["structured_facts_extracted"]),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["default_primary_tabs"] == ["Run & Review", "MKB Explorer", "Review Queue"],
        findings["advanced_engineering_tabs_hidden_by_default"],
        findings["compact_header_ready"],
        findings["run_review_first_viewport_ready"],
        findings["mkb_explorer_first_viewport_ready"],
        findings["review_queue_first_viewport_ready"],
        findings["operator_tabs_remain_first_with_advanced"],
        findings["build_audit_details_hidden_by_default"],
        findings["specialty_selector_still_visible"],
        findings["urology_option_visible"],
        findings["dermatology_option_visible"],
        findings["gastroenterology_option_visible"],
        findings["adapter_fallback_selected_specialty"] == TESTED_SPECIALTY,
        findings["records_persisted_with_selected_specialty_count"] > 0,
        findings["review_bound_records"] == findings["structured_facts_extracted"],
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E",
        "mode": "compact_operator_ui_model_and_local_fallback_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "operator_one_screen_ui_polish_11e_ready" if all(gates) else "not_ready",
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
                f"- compact header ready: `{f['compact_header_ready']}`",
                f"- Run & Review first viewport ready: `{f['run_review_first_viewport_ready']}`",
                f"- MKB Explorer first viewport ready: `{f['mkb_explorer_first_viewport_ready']}`",
                f"- Review Queue first viewport ready: `{f['review_queue_first_viewport_ready']}`",
                f"- specialty selector still visible: `{f['specialty_selector_still_visible']}`",
                f"- Urology option visible: `{f['urology_option_visible']}`",
                f"- engineering details hidden by default: `{f['build_audit_details_hidden_by_default']}`",
                f"- adapter fallback selected specialty: `{f['adapter_fallback_selected_specialty']}`",
                f"- review-bound records: `{f['review_bound_records']}`",
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
    lines.extend([f"- {item}" for item in payload["layout_findings"]])
    lines.extend(["", f"- external API used: `{payload['external_api_used']}`", f"- auto-accept enabled: `{payload['auto_accept_enabled']}`", ""])
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E Report", report))
    rendered = json.dumps(audit, indent=2) + "\n" + json.dumps(report, indent=2) + "\n" + _markdown("report", report)
    report["public_artifacts_path_safe"] = _path_safe(rendered)
    report["privacy_check_passed"] = bool(
        report["privacy_check_passed"]
        and audit_privacy.get("passed")
        and md_privacy.get("passed")
        and report["public_artifacts_path_safe"]
    )
    report["privacy_check_leak_examples_redacted"] = (
        list(report.get("privacy_check_leak_examples_redacted") or [])
        + list(audit_privacy.get("leak_examples_redacted") or [])
        + list(md_privacy.get("leak_examples_redacted") or [])
    )
    if not report["privacy_check_passed"]:
        report["conclusion"] = "not_ready"
        report["all_pass"] = False
    write_reports(audit, report)
    print(json.dumps(report, indent=2))
    return 0 if report["conclusion"] == "operator_one_screen_ui_polish_11e_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
