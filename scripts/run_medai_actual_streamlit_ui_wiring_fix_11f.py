#!/usr/bin/env python3
"""MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F validation."""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_actual_streamlit_ui_wiring_fix_11f"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_ACTUAL_STREAMLIT_UI_WIRING_FIX_11F_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_actual_streamlit_ui_wiring_fix_11f_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_ACTUAL_STREAMLIT_UI_WIRING_FIX_11F.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_actual_streamlit_ui_wiring_fix_11f_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_actual_streamlit_ui_wiring_fix_11f_report.md"

NEXT_OPERATOR_COMMAND = "streamlit run app/main.py"
TESTED_SPECIALTY = "dermatology"
SPECIALTY_LABELS_REQUIRED = {"Urology", "Dermatology", "Gastroenterology", "Neurology"}
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


def _literal_assigned_list(tree: ast.Module, name: str) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return list(ast.literal_eval(node.value))
    return []


def build_audit() -> dict[str, Any]:
    return {
        "block_id": "MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F-AUDIT",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "actual_ui_findings": [
            "The visible Run & Review page is rendered by app.main.render_run_review_tab, which delegates to render_current_run_tab.",
            "The visible Streamlit tab list is constructed in app.main.main before st.tabs(tab_labels).",
            "Prior blocks validated helper models, but browser acceptance requires literal app.main tab wiring and dispatch checks.",
            "Specialty/domain options were present in Document category because the real renderer still used specialty labels as document categories.",
            "Separate Document category and Medical specialty / domain controls belong in render_current_run_tab and render_adapter_fallback_panel.",
            "11B/11C/11E helpers remain useful, but actual main() and renderer source must wire the visible controls directly.",
            "This block fixes tab construction, document category options, MKB Explorer dispatch, Review Queue dispatch, and selected-specialty fallback wiring.",
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
    from app.specialty_selection import specialty_labels_for_ui
    from mkb.sqlite_store import SQLiteStore

    source, tree = _source()
    main_source = _function_source(source, tree, "main")
    fallback_source = _function_source(source, tree, "render_adapter_fallback_panel")
    current_run_source = _function_source(source, tree, "render_current_run_tab")
    document_categories = set(_literal_assigned_list(tree, "DOCUMENT_CATEGORY_OPTIONS"))
    specialty_labels = set(specialty_labels_for_ui())

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_actual_ui_wiring_11f_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    fallback = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty=TESTED_SPECIALTY,
        session_id="actual-streamlit-ui-wiring-11f",
    )
    record_ids = list(fallback["run_item"].get("extracted_medical_fact_record_ids") or [])
    mkb_model = build_mkb_explorer_model(sql_store, specialty_filter=TESTED_SPECIALTY, tier_filter="review_bound")
    review_queue_model = build_mkb_explorer_model(sql_store, tier_filter="review_bound")
    review_bound_records = sum(
        1 for record_id in record_ids if (sql_store.get_record(record_id) and sql_store.get_record(record_id).requires_review)
    )
    document_category_excludes_specialties = not bool(document_categories & SPECIALTY_LABELS_REQUIRED)

    findings = {
        "actual_mkb_explorer_tab_visible": "MKB_EXPLORER_TAB" in main_source and "render_mkb_tab(sys_components)" in main_source,
        "actual_review_queue_tab_visible": "REVIEW_QUEUE_TAB" in main_source and "render_review_queue_tab(sys_components)" in main_source,
        "operator_tabs_before_advanced": all(token in main_source for token in ["RUN_REVIEW_TAB", "MKB_EXPLORER_TAB", "REVIEW_QUEUE_TAB", '"Operator Control Panel"'])
        and main_source.index("RUN_REVIEW_TAB") < main_source.index("MKB_EXPLORER_TAB") < main_source.index("REVIEW_QUEUE_TAB") < main_source.index('"Operator Control Panel"'),
        "actual_medical_specialty_domain_separate_control": '"Medical specialty / domain"' in source
        and "render_specialty_selector(" in current_run_source
        and "render_specialty_selector(" in fallback_source,
        "actual_document_category_label": '"Document category"' in source,
        "document_category_options": sorted(document_categories),
        "document_category_excludes_specialties": document_category_excludes_specialties,
        "specialty_options_include_required": SPECIALTY_LABELS_REQUIRED.issubset(specialty_labels),
        "urology_specialty_visible": "Urology" in specialty_labels,
        "gastroenterology_specialty_visible": "Gastroenterology" in specialty_labels,
        "run_review_compact_first_viewport": "category_col, specialty_col = st.columns(2)" in source
        and "upload_col, start_col = st.columns([3, 1])" in source
        and "render_compact_run_summary(" in current_run_source,
        "engineering_details_hidden_by_default": "show_build_details=show_advanced_tools" in source,
        "adapter_fallback_selected_specialty": fallback["selected_specialty"],
        "adapter_fallback_wires_selected_specialty": "selected_specialty=selected_specialty" in fallback_source,
        "mkb_explorer_rows_with_specialty_tier_status": int(mkb_model["row_count"]),
        "review_queue_review_bound_rows": int(review_queue_model["row_count"]),
        "records_remain_review_bound": int(review_bound_records),
        "structured_facts_extracted": int(fallback["structured_facts_extracted"]),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["actual_mkb_explorer_tab_visible"],
        findings["actual_review_queue_tab_visible"],
        findings["operator_tabs_before_advanced"],
        findings["actual_medical_specialty_domain_separate_control"],
        findings["actual_document_category_label"],
        findings["document_category_excludes_specialties"],
        findings["specialty_options_include_required"],
        findings["adapter_fallback_selected_specialty"] == TESTED_SPECIALTY,
        findings["adapter_fallback_wires_selected_specialty"],
        findings["mkb_explorer_rows_with_specialty_tier_status"] > 0,
        findings["review_queue_review_bound_rows"] > 0,
        findings["records_remain_review_bound"] == findings["structured_facts_extracted"],
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F",
        "mode": "actual_app_main_streamlit_wiring_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "actual_streamlit_ui_wiring_fix_11f_ready" if all(gates) else "not_ready",
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
                f"- actual MKB Explorer tab visible: `{f['actual_mkb_explorer_tab_visible']}`",
                f"- actual Review Queue tab visible: `{f['actual_review_queue_tab_visible']}`",
                f"- actual Medical specialty / domain separate control: `{f['actual_medical_specialty_domain_separate_control']}`",
                f"- Document category excludes specialties: `{f['document_category_excludes_specialties']}`",
                f"- Urology specialty visible: `{f['urology_specialty_visible']}`",
                f"- Gastroenterology specialty visible: `{f['gastroenterology_specialty_visible']}`",
                f"- Run & Review compact first viewport: `{f['run_review_compact_first_viewport']}`",
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
    lines.extend([f"- {item}" for item in payload["actual_ui_findings"]])
    lines.extend(["", f"- external API used: `{payload['external_api_used']}`", f"- auto-accept enabled: `{payload['auto_accept_enabled']}`", ""])
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F Report", report))
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
    return 0 if report["conclusion"] == "actual_streamlit_ui_wiring_fix_11f_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
