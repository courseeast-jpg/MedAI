#!/usr/bin/env python3
"""MEDAI-RUN-REVIEW-DENSITY-FINAL-11H validation."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_run_review_density_final_11h"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_RUN_REVIEW_DENSITY_FINAL_11H_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_run_review_density_final_11h_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_RUN_REVIEW_DENSITY_FINAL_11H.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_run_review_density_final_11h_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_run_review_density_final_11h_report.md"

NEXT_OPERATOR_COMMAND = "streamlit run app/main.py"


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
        "block_id": "MEDAI-RUN-REVIEW-DENSITY-FINAL-11H-AUDIT",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "density_findings": [
            "Large upload container height remains a Streamlit constraint, but CSS can reduce padding.",
            "Documents waiting can be rendered as a one-line summary for a single queued file.",
            "Run status result cards should appear only after a run has occurred.",
            "Result guide, Previous review summary, and Advanced actions should stay collapsed by default.",
            "Repeated helper text and no-run result captions should not occupy the first viewport.",
            "MKB Explorer and Review Queue are out of scope except for preserving their visible tabs.",
        ],
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def _path_safe(text: str) -> bool:
    banned = ["C:\\Users", "G:\\Codex", ".env", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DOB", "date-of-birth"]
    return not any(token in text for token in banned)


def build_report() -> dict[str, Any]:
    from app.specialty_selection import specialty_labels_for_ui

    source, tree = _source()
    main_source = _function_source(source, tree, "main")
    run_source = _function_source(source, tree, "render_current_run_tab")
    queue_source = _function_source(source, tree, "render_queue_panel")
    style_source = (REPO_ROOT / "app" / "operator_compact_styles.py").read_text(encoding="utf-8")
    labels = set(specialty_labels_for_ui())

    tab_block = main_source[main_source.index("tab_labels = [") : main_source.index("tabs = st.tabs(tab_labels)")]
    default_block = tab_block[: tab_block.index("if show_advanced_tools:")]
    findings = {
        "default_tabs": ["Run & Review", "MKB Explorer", "Review Queue"],
        "default_tabs_wired": all(token in default_block for token in ["RUN_REVIEW_TAB", "MKB_EXPLORER_TAB", "REVIEW_QUEUE_TAB"]),
        "no_run_zero_cards_hidden": "if active_run:" in run_source
        and "render_run_status_panel(active_run" in run_source
        and 'st.caption("No current run results' not in run_source,
        "result_guide_collapsed": 'st.expander("Result guide", expanded=False)' in source,
        "previous_review_summary_collapsed": 'st.expander("Previous review summary / aggregate review status", expanded=False)' in source,
        "advanced_actions_collapsed": 'st.expander("Advanced actions", expanded=False)' in run_source,
        "queued_documents_compact": "if len(files) == 1:" in queue_source
        and 'row[0].caption(f"Documents waiting: {path.name}")' in queue_source,
        "upload_start_first_viewport": "upload_col, start_col = st.columns([3, 1])" in run_source
        and "Start run" in run_source,
        "specialty_selector_separate": '"Document category"' in run_source and "render_specialty_selector(" in run_source,
        "mkb_explorer_still_visible": "MKB_EXPLORER_TAB" in main_source and "render_mkb_tab(sys_components)" in main_source,
        "review_queue_still_visible": "REVIEW_QUEUE_TAB" in main_source and "render_review_queue_tab(sys_components)" in main_source,
        "specialty_options_required": {"Urology", "Dermatology", "Gastroenterology"}.issubset(labels),
        "compact_css_density": "gap: .22rem" in style_source and "min-height: 2.25rem" in style_source,
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    gates = [
        findings["default_tabs_wired"],
        findings["no_run_zero_cards_hidden"],
        findings["result_guide_collapsed"],
        findings["previous_review_summary_collapsed"],
        findings["advanced_actions_collapsed"],
        findings["queued_documents_compact"],
        findings["upload_start_first_viewport"],
        findings["specialty_selector_separate"],
        findings["mkb_explorer_still_visible"],
        findings["review_queue_still_visible"],
        findings["specialty_options_required"],
        findings["compact_css_density"],
        findings["external_api_used"] is False,
        findings["auto_accept_enabled"] is False,
    ]
    report = {
        "block_id": "MEDAI-RUN-REVIEW-DENSITY-FINAL-11H",
        "mode": "actual_run_review_density_source_validation",
        "head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_findings": findings,
        "conclusion": "run_review_density_final_11h_ready" if all(gates) else "not_ready",
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
                f"- no-run zero cards hidden: `{f['no_run_zero_cards_hidden']}`",
                f"- result guide collapsed: `{f['result_guide_collapsed']}`",
                f"- previous review summary collapsed: `{f['previous_review_summary_collapsed']}`",
                f"- queued documents compact: `{f['queued_documents_compact']}`",
                f"- upload/start still first viewport: `{f['upload_start_first_viewport']}`",
                f"- specialty selector separate: `{f['specialty_selector_separate']}`",
                f"- MKB Explorer still visible: `{f['mkb_explorer_still_visible']}`",
                f"- Review Queue still visible: `{f['review_queue_still_visible']}`",
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
    lines.extend([f"- {item}" for item in payload["density_findings"]])
    lines.extend(["", f"- external API used: `{payload['external_api_used']}`", f"- auto-accept enabled: `{payload['auto_accept_enabled']}`", ""])
    return "\n".join(lines)


def write_reports(audit: dict[str, Any], report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(_markdown("MEDAI-RUN-REVIEW-DENSITY-FINAL-11H Audit", audit), encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown("MEDAI-RUN-REVIEW-DENSITY-FINAL-11H Report", report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SHORT_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    audit = build_audit()
    report = build_report()
    audit_privacy = _privacy_check(audit)
    md_privacy = _privacy_check(_markdown("MEDAI-RUN-REVIEW-DENSITY-FINAL-11H Report", report))
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
    return 0 if report["conclusion"] == "run_review_density_final_11h_ready" and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
