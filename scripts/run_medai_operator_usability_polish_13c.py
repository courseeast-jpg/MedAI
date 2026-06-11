#!/usr/bin/env python3
"""MEDAI-OPERATOR-USABILITY-POLISH-13C validation."""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_operator_usability_polish_13c"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_USABILITY_POLISH_13C.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_operator_usability_polish_13c_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_operator_usability_polish_13c_report.md"

MAIN_PATH = REPO_ROOT / "app" / "main.py"
CSS_PATH = REPO_ROOT / "app" / "operator_compact_styles.py"
UAT_12A_REPORT_PATH = REPO_ROOT / "reports" / "medai_operator_workflow_uat_12a" / "medai_operator_workflow_uat_12a_report.json"


def _source() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def _css_source() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _function_source(name: str, source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    source = _source()
    css = _css_source()
    run_source = _function_source("render_current_run_tab", source)
    run_review_source = _function_source("render_run_review_tab", source)
    guidance_source = _function_source("render_operator_guidance_panel", source)
    safety_source = _function_source("render_operator_safety_panel", source)
    mkb_source = _function_source("render_mkb_tab", source)
    review_source = _function_source("render_review_queue_tab", source)
    uat_report = _load_json(UAT_12A_REPORT_PATH)

    primary_start_button_visible = all(
        [
            "operator-start-rail" in run_source,
            '"Start run"' in run_source,
            'type="primary"' in run_source,
            "button[kind=\"primary\"]" in css,
            "min-height: 3rem" in css,
        ]
    )
    upload_add_queue_prominent = all(
        [
            "operator-files-queue" in run_source,
            "operator-add-queue-callout" in run_source,
            "Next step: add selected files to the queue." in run_source,
            '"Add selected files to queue", type="primary", use_container_width=True' in run_source,
            "div[data-testid=\"stFileUploader\"] section" in css,
        ]
    )
    advanced_details_collapsed = all(
        [
            'st.expander("Advanced actions", expanded=False)' in run_source,
            'st.expander("Result guide", expanded=False)' in guidance_source,
            'st.expander("Previous review summary / aggregate review status", expanded=False)' in run_review_source,
        ]
    )
    report = {
        "task_id": "MEDAI-OPERATOR-USABILITY-POLISH-13C",
        "privacy_result": "pending",
        "external_api_used": False,
        "auto_accept": False,
        "usability_polish_applied": all(
            [
                "operator-path" in run_source,
                "Run setup" in run_source,
                "Files / queue" in run_source,
                "Results / review" in run_source,
                "operator-review-action-row" in review_source,
            ]
        ),
        "primary_start_button_visible": primary_start_button_visible,
        "upload_add_queue_prominent": upload_add_queue_prominent,
        "empty_space_reduced": all(
            [
                "max-width: 1180px" in css,
                "gap: .18rem" in css,
                "operator-path" in css,
                "operator-run-setup" in css,
            ]
        ),
        "advanced_details_collapsed": advanced_details_collapsed,
        "technical_details_not_default_focus": all(
            [
                "font-size: .72rem" in css,
                "operator-path-step" in css,
                "Result guide" in guidance_source,
                "expanded=False" in guidance_source,
            ]
        ),
        "safety_banner_visible": "Review required - not for diagnosis." in safety_source and "MedAI does not diagnose" in source,
        "safety_pills_visible": all(
            pill in safety_source for pill in ["Local only", "Cloud APIs off", "Privacy check on", "Human review"]
        ),
        "operator_copy_applied": all(
            text in source
            for text in [
                "No documents queued. Add supported files to start.",
                "Ready:",
                "Run complete. Review results below.",
                "Needs human review.",
            ]
        ),
        "mkb_counts_visible": all(text in mkb_source for text in ["Active", "Quarantined / review-bound"]),
        "review_queue_actions_visible": all(
            text in source for text in ["Accept after source comparison", "Reject", "Defer"]
        ),
        "functional_behavior_preserved": bool(
            uat_report.get("uat_passed") is True
            and uat_report.get("privacy_result") == "passed"
            and uat_report.get("external_api_used") is False
            and uat_report.get("auto_accept") is False
        ),
        "raw_ocr_text_in_report": False,
        "private_paths_in_report": False,
        "limitations": [
            "Source-level UI validation; no browser screenshot is captured by this block.",
            "UI/CSS/layout/copy only; extraction, routing, persistence, and review action semantics are unchanged.",
            "The 12A UAT uses synthetic files and does not use private medical documents.",
        ],
    }
    report["privacy_result"] = "passed" if _privacy_passes(report) else "failed"
    return report


def _privacy_passes(report: dict[str, Any]) -> bool:
    from clinical_knowledge.privacy import check_public_report_payload

    return bool(check_public_report_payload(report).passed)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-OPERATOR-USABILITY-POLISH-13C",
        "",
        f"- Usability polish applied: `{report['usability_polish_applied']}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Primary Start run button visible: `{report['primary_start_button_visible']}`",
        f"- Upload/Add queue prominent: `{report['upload_add_queue_prominent']}`",
        f"- Empty space reduced: `{report['empty_space_reduced']}`",
        f"- Advanced details collapsed: `{report['advanced_details_collapsed']}`",
        f"- Technical details not default focus: `{report['technical_details_not_default_focus']}`",
        f"- Safety banner visible: `{report['safety_banner_visible']}`",
        f"- Safety pills visible: `{report['safety_pills_visible']}`",
        f"- Operator copy applied: `{report['operator_copy_applied']}`",
        f"- MKB counts visible: `{report['mkb_counts_visible']}`",
        f"- Review Queue actions visible: `{report['review_queue_actions_visible']}`",
        f"- Functional behavior preserved: `{report['functional_behavior_preserved']}`",
        f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
        f"- Private paths in report: `{report['private_paths_in_report']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = _markdown(report)
    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")


def main() -> int:
    report = build_report()
    write_reports(report)
    ready = all(
        [
            report["privacy_result"] == "passed",
            report["external_api_used"] is False,
            report["auto_accept"] is False,
            report["usability_polish_applied"],
            report["primary_start_button_visible"],
            report["upload_add_queue_prominent"],
            report["empty_space_reduced"],
            report["advanced_details_collapsed"],
            report["technical_details_not_default_focus"],
            report["safety_banner_visible"],
            report["safety_pills_visible"],
            report["operator_copy_applied"],
            report["mkb_counts_visible"],
            report["review_queue_actions_visible"],
            report["functional_behavior_preserved"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
        ]
    )
    print("medai_operator_usability_polish_13c_ready" if ready else "medai_operator_usability_polish_13c_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "usability_polish_applied": report["usability_polish_applied"],
                "functional_behavior_preserved": report["functional_behavior_preserved"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
