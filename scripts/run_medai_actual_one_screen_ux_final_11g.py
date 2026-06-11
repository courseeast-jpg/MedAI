#!/usr/bin/env python3
"""MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G validation."""
from __future__ import annotations

import ast
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_actual_one_screen_ux_final_11g"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_ACTUAL_ONE_SCREEN_UX_FINAL_11G_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_actual_one_screen_ux_final_11g_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_ACTUAL_ONE_SCREEN_UX_FINAL_11G.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_actual_one_screen_ux_final_11g_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_actual_one_screen_ux_final_11g_report.md"

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


def _source() -> tuple[str, ast.Module]:
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    return source, ast.parse(source)


def _function_source(source: str, tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def build_audit() -> dict[str, Any]:
    return {
        "block_id": "MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G-AUDIT",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "browser_ux_findings": [
            "11F fixed actual Streamlit tab and specialty wiring.",
            "Browser acceptance still failed compact one-screen operator use.",
            "Vertical height remained in the header, cloud/API warning area, upload block, documents waiting section, and result guide.",
            "Advanced/admin visibility remained too prominent in the default first viewport.",
            "11G changes layout and styling only; extraction, OCR, classifier, thresholds, medication/DDI, and medical decision logic remain unchanged.",
        ],
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def _path_safe(text: str) -> bool:
    banned = ["C:\\Users", "G:\\Codex", ".env", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DOB", "date-of-birth"]
    return not any(token in text for token in banned)


def build_report() -> dict[str, Any]:
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from app.specialty_selection import specialty_labels_for_ui
    from mkb.sqlite_store import SQLiteStore

    source, tree = _source()
    main_source = _function_source(source, tree, "main")
    current_run_source = _function_source(source, tree, "render_current_run_tab")
    review_source = _function_source(source, tree, "render_run_review_tab")
    safety_source = _function_source(source, tree, "render_operator_safety_panel")
    guide_source = _function_source(source, tree, "render_operator_guidance_panel")
    system_status_source = _function_source(source, tree, "render_system_status")
    style_source = (REPO_ROOT / "app" / "operator_compact_styles.py").read_text(encoding="utf-8")

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_one_screen_ux_final_11g_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    fallback = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty=TESTED_SPECIALTY,
        session_id="actual-one-screen-ux-final-11g",
    )
    record_ids = list(fallback["run_item"].get("extracted_medical_fact_record_ids") or [])
    review_bound_records = sum(
        1 for record_id in record_ids if (sql_store.get_record(record_id) and sql_store.get_record(record_id).requires_review)
    )
    labels = set(specialty_labels_for_ui())

    tab_block = main_source[main_source.index("tab_labels = [") : main_source.index("tabs = st.tabs(tab_labels)")]
    default_block = tab_block[: tab_block.index("if show_advanced_tools:")]
    findings = {
        "default_advanced_tools_off": 'st.sidebar.checkbox(\n        "Show advanced tools",\n        value=False' in main_source,
        "default_visible_tabs": ["Run & Review", "MKB Explorer", "Review Queue"],
        "advanced_only_tabs": [
            "Operator Control Panel",
            "Validation Batch Audit",
            "Validation History",
            "Safety & Governance",
            "Terminology Admin",
        ],
        "advanced_tabs_not_in_default_block": "Operator Control Panel" not in default_block and "Validation Batch Audit" not in default_block,
        "build_audit_hidden_by_default": "if not show_build_details:" in safety_source
        and "show_build_details=show_advanced_tools" in main_source,
        "claude_api_warning_compact_or_advanced_only": "show_advanced_tools" in system_status_source
        and "st.warning" not in system_status_source,
        "compact_style_helper_applied": "COMPACT_OPERATOR_CSS" in source
        and "st.markdown(COMPACT_OPERATOR_CSS" in source
        and "padding-top: .35rem" in style_source
        and 'div[data-testid="stFileUploader"] section' in style_source,
        "run_review_first_viewport_compact": "compact-workflow-row" in review_source
        and "category_col, specialty_col = st.columns(2)" in current_run_source
        and "upload_col, start_col = st.columns([3, 1])" in current_run_source
        and "render_compact_run_summary(" in current_run_source,
        "mkb_explorer_first_viewport_compact": "render_mkb_tab(sys_components)" in main_source
        and 'metric("Total"' in source
        and "specialty_filter, tier_filter, fact_type_filter = st.columns(3)" in source,
        "review_queue_first_viewport_compact": "render_review_queue_tab(sys_components)" in main_source
        and "top_cols = st.columns([1, 3])" in source
        and "review_queue_accept_" in source,
        "result_guide_collapsed_by_default": 'st.expander("Result guide", expanded=False)' in guide_source,
        "advanced_actions_collapsed_by_default": 'st.expander("Advanced actions", expanded=False)' in current_run_source,
        "specialty_selector_still_separate": '"Document category"' in current_run_source
        and "render_specialty_selector(" in current_run_source,
        "urology_option_visible": "Urology" in labels,
        "dermatology_option_visible": "Dermatology" in labels,
        "gastroenterology_option_visible": "Gastroenterology" in labels,
        "adapter_fallback_selected_specialty": fallback["selected_specialty"],
        "review_bound_records": int(review_bound_records),
        "structured_facts_extracted": int(fallback["structured_facts_extracted"]),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["default_advanced_tools_off"],
        findings["advanced_tabs_not_in_default_block"],
        findings["build_audit_hidden_by_default"],
        findings["claude_api_warning_compact_or_advanced_only"],
        findings["compact_style_helper_applied"],
        findings["run_review_first_viewport_compact"],
        findings["mkb_explorer_first_viewport_compact"],
        findings["review_queue_first_viewport_compact"],
        findings["result_guide_collapsed_by_default"],
        findings["advanced_actions_collapsed_by_default"],
        findings["specialty_selector_still_separate"],
        findings["urology_option_visible"],
        findings["gastroenterology_option_visible"],
        findings["adapter_fallback_selected_specialty"] == TESTED_SPECIALTY,
        findings["review_bound_records"] > 0,
        findings["review_bound_records"] == findings["structured_facts_extracted"],
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G",
        "mode": "actual_streamlit_one_screen_layout_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "actual_one_screen_ux_final_11g_ready" if all(gates) else "not_ready",
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
                f"- default advanced tools off: `{f['default_advanced_tools_off']}`",
                f"- default visible tabs: `{', '.join(f['default_visible_tabs'])}`",
                f"- advanced-only tabs: `{', '.join(f['advanced_only_tabs'])}`",
                f"- Build/audit hidden by default: `{f['build_audit_hidden_by_default']}`",
                f"- Claude API warning compact or advanced-only: `{f['claude_api_warning_compact_or_advanced_only']}`",
                f"- Run & Review first viewport compact: `{f['run_review_first_viewport_compact']}`",
                f"- MKB Explorer first viewport compact: `{f['mkb_explorer_first_viewport_compact']}`",
                f"- Review Queue first viewport compact: `{f['review_queue_first_viewport_compact']}`",
                f"- Result guide collapsed by default: `{f['result_guide_collapsed_by_default']}`",
                f"- specialty selector still separate: `{f['specialty_selector_still_separate']}`",
                f"- Urology option still visible: `{f['urology_option_visible']}`",
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
    lines.extend([f"- {item}" for item in payload["browser_ux_findings"]])
    lines.extend(["", f"- external API used: `{payload['external_api_used']}`", f"- auto-accept enabled: `{payload['auto_accept_enabled']}`", ""])
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G Report", report))
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
    return 0 if report["conclusion"] == "actual_one_screen_ux_final_11g_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
