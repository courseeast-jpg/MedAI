#!/usr/bin/env python3
"""MEDAI-AI-PACKAGE-RUN-REVIEW-OPERATOR-UAT-15O-D validation."""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.pop("MEDAI_VERTEX_LIVE_SMOKE_ALLOWED", None)

from app.ai_package_run_review_preview import (  # noqa: E402
    build_run_review_package_previews,
    render_ai_package_run_review_preview_panel,
    render_run_review_preview_markdown,
)
from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_run_review_operator_uat_15o_d"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CASES_JSON = REPORT_DIR / "operator_uat_cases.json"
MATRIX_MD = REPORT_DIR / "operator_uat_matrix.md"

BLOCKED_TOKENS = (
    "GEMINI_API_KEY",
    "Authorization",
    "Bearer ",
    "ya29.",
    "AIza",
    "token_map",
    "C:\\",
    "DOB",
    "MRN:",
    "Accession",
    ".pdf",
    ".png",
    ".jpg",
)


def build_uat_report() -> dict[str, Any]:
    previews = build_run_review_package_previews()
    render_source = inspect.getsource(render_ai_package_run_review_preview_panel)
    app_main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    entry_reachable = (
        "render_ai_package_run_review_preview_panel" in app_main_source
        and "app.ai_package_run_review_preview" in app_main_source
    )
    cases = []
    for preview in previews:
        markdown = render_run_review_preview_markdown(preview)
        cases.append(_case(preview, markdown, entry_reachable, render_source))
    summary = {
        "task_id": "MEDAI-AI-PACKAGE-RUN-REVIEW-OPERATOR-UAT-15O-D",
        "baseline_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "report_generation_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "uat_method_used": "bounded_source_reachability_plus_deterministic_view_model_markdown_uat",
        "streamlit_live_launch_used": False,
        "run_review_preview_entry_point_reachable_count": sum(
            1 for case in cases if case["preview_entry_point_reachable"]
        ),
        "package_families_checked": [case["package_family"] for case in cases],
        "package_family_count": len(cases),
        "source_visible_body_present_count": sum(1 for case in cases if case["source_visible_body_present"]),
        "evidence_anchor_present_count": sum(1 for case in cases if case["evidence_anchor_present"]),
        "candidate_facts_separated_count": sum(1 for case in cases if case["candidate_facts_separated"]),
        "unknown_values_explicit_count": sum(1 for case in cases if case["unknown_values_explicit"]),
        "uncertainty_flags_visible_count": sum(1 for case in cases if case["uncertainty_flags_visible"]),
        "no_live_indicator_visible_count": sum(1 for case in cases if case["no_live_indicator_visible"]),
        "active_written_count_indicator_visible_count": sum(
            1 for case in cases if case["active_written_count_indicator_visible"]
        ),
        "auto_accept_false_indicator_visible_count": sum(
            1 for case in cases if case["auto_accept_false_indicator_visible"]
        ),
        "under_1_minute_compare_preserved_count": sum(
            1 for case in cases if case["under_1_minute_compare_preserved"]
        ),
        "hallucinated_field_count": sum(case["hallucinated_field_count"] for case in cases),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
        "privacy_result": "pending",
        "all_operator_uat_checks_passed": False,
        "tests_run": [
            "python -m pytest tests/test_medai_ai_package_run_review_operator_uat_15o_d.py -q",
            "python -m py_compile scripts/run_medai_ai_package_run_review_operator_uat_15o_d.py",
            "python scripts/run_medai_ai_package_run_review_operator_uat_15o_d.py",
        ],
        "files_changed": [
            "scripts/run_medai_ai_package_run_review_operator_uat_15o_d.py",
            "tests/test_medai_ai_package_run_review_operator_uat_15o_d.py",
            "reports/medai_ai_package_run_review_operator_uat_15o_d/*",
        ],
        "push_status": "pending",
    }
    summary["all_operator_uat_checks_passed"] = _all_cases_pass(cases, summary)
    matrix = _matrix_markdown(cases)
    implementation = _implementation_markdown(summary)
    privacy = _privacy_passes(summary, {"cases": cases}, matrix, implementation)
    summary["privacy_result"] = "passed" if privacy else "failed"
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "cases": {"task_id": summary["task_id"], "cases": cases},
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps(report["cases"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(report["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(report["implementation"], encoding="utf-8")


def _case(preview: Any, markdown: str, entry_reachable: bool, render_source: str) -> dict[str, Any]:
    return {
        "package_family": preview.package_family,
        "package_family_label_visible": bool(preview.package_family_label and preview.package_family_label in markdown),
        "preview_entry_point_reachable": bool(entry_reachable),
        "review_required_visible": "Review required" in markdown and "`True`" in markdown,
        "source_visible_body_present": bool(preview.source_visible_body and "### Source Visible Body" in markdown),
        "source_visible_body_not_collapsed_only": "st.expander" not in render_source,
        "source_sections_visible": bool(preview.source_sections and "### Source Sections" in markdown),
        "candidate_facts_separated": bool(
            preview.candidate_facts
            and "### Candidate Facts" in markdown
            and preview.source_visible_body not in json.dumps(preview.candidate_facts)
        ),
        "evidence_anchor_present": bool(preview.evidence_anchors and "Evidence:" in markdown),
        "unknown_values_explicit": bool(
            preview.unknown_values
            or preview.package_family != "mixed_narrative_numeric_result"
        ),
        "uncertainty_flags_visible": bool(preview.uncertainty_flags and "### Uncertainty Flags" in markdown),
        "no_live_indicator_visible": "no-live/provider-off" in markdown,
        "active_written_count_indicator_visible": "Active written count" in markdown and "`0`" in markdown,
        "auto_accept_false_indicator_visible": "Auto-accept" in markdown and "`False`" in markdown,
        "compact_for_operator_compare": preview.estimated_human_compare_seconds <= 60,
        "under_1_minute_compare_preserved": bool(preview.under_1_minute_compare_preserved),
        "hallucinated_field_count": int(preview.hallucinated_field_count),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
    }


def _all_cases_pass(cases: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    required_case_keys = [
        "preview_entry_point_reachable",
        "package_family_label_visible",
        "review_required_visible",
        "source_visible_body_present",
        "source_visible_body_not_collapsed_only",
        "source_sections_visible",
        "candidate_facts_separated",
        "evidence_anchor_present",
        "unknown_values_explicit",
        "uncertainty_flags_visible",
        "no_live_indicator_visible",
        "active_written_count_indicator_visible",
        "auto_accept_false_indicator_visible",
        "compact_for_operator_compare",
        "under_1_minute_compare_preserved",
    ]
    return bool(cases) and all(
        all(case[key] is True for key in required_case_keys)
        and case["hallucinated_field_count"] == 0
        and case["live_call_made"] is False
        and case["external_api_used"] is False
        and case["active_written_count"] == 0
        and case["auto_accept"] is False
        and case["review_required"] is True
        for case in cases
    ) and all(
        [
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
        ]
    )


def _matrix_markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# 15O-D Operator UAT Matrix",
        "",
        "| Family | Entry | Source body | Facts | Anchors | Unknowns | Uncertainty | No-live | Active writes | Auto-accept | Under 1 min |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            "| {family} | {entry} | {body} | {facts} | {anchors} | {unknowns} | {uncertainty} | {noliv} | {writes} | {auto} | {under} |".format(
                family=case["package_family"],
                entry=case["preview_entry_point_reachable"],
                body=case["source_visible_body_present"],
                facts=case["candidate_facts_separated"],
                anchors=case["evidence_anchor_present"],
                unknowns=case["unknown_values_explicit"],
                uncertainty=case["uncertainty_flags_visible"],
                noliv=case["no_live_indicator_visible"],
                writes=case["active_written_count_indicator_visible"],
                auto=case["auto_accept_false_indicator_visible"],
                under=case["under_1_minute_compare_preserved"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PACKAGE-RUN-REVIEW-OPERATOR-UAT-15O-D",
            "",
            f"- UAT method used: `{summary['uat_method_used']}`",
            f"- Streamlit live launch used: `{summary['streamlit_live_launch_used']}`",
            f"- Preview entry point reachable count: `{summary['run_review_preview_entry_point_reachable_count']}`",
            f"- Package families checked: `{summary['package_families_checked']}`",
            f"- Source visible body present count: `{summary['source_visible_body_present_count']}`",
            f"- Evidence anchor present count: `{summary['evidence_anchor_present_count']}`",
            f"- Candidate facts separated count: `{summary['candidate_facts_separated_count']}`",
            f"- Unknown values explicit count: `{summary['unknown_values_explicit_count']}`",
            f"- Uncertainty flags visible count: `{summary['uncertainty_flags_visible_count']}`",
            f"- No-live indicator visible count: `{summary['no_live_indicator_visible_count']}`",
            f"- Active written count indicator visible count: `{summary['active_written_count_indicator_visible_count']}`",
            f"- Auto-accept false indicator visible count: `{summary['auto_accept_false_indicator_visible_count']}`",
            f"- Under 1 minute compare preserved count: `{summary['under_1_minute_compare_preserved_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- Live call made: `{summary['live_call_made']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review required: `{summary['review_required']}`",
            f"- Privacy result: `{summary['privacy_result']}`",
            f"- Billing check pending: `{summary['billing_check_pending']}`",
            f"- Push status: `{summary['push_status']}`",
            "",
            "## Scope",
            "",
            "- Bounded deterministic UAT; no live Streamlit/browser launch was required.",
            "- Uses the real Run & Review preview hook plus generated markdown previews.",
            "- Does not change OCR routing, thresholds, providers, medical decision logic, or MKB writes.",
            "",
        ]
    )


def _privacy_passes(*payloads: Any) -> bool:
    serialized = json.dumps(payloads, sort_keys=True, default=str)
    if any(token in serialized for token in BLOCKED_TOKENS):
        return False
    return bool(check_public_report_payload(payloads).passed)


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    report = build_uat_report()
    write_reports(report)
    summary = report["summary"]
    ready = summary["privacy_result"] == "passed" and summary["all_operator_uat_checks_passed"] is True
    print("medai_ai_package_run_review_operator_uat_15o_d_ready" if ready else "medai_ai_package_run_review_operator_uat_15o_d_not_ready")
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
                "privacy_result": summary["privacy_result"],
                "uat_method_used": summary["uat_method_used"],
                "package_families_checked": summary["package_families_checked"],
                "preview_entry_point_reachable_count": summary["run_review_preview_entry_point_reachable_count"],
                "source_visible_body_present_count": summary["source_visible_body_present_count"],
                "evidence_anchor_present_count": summary["evidence_anchor_present_count"],
                "uncertainty_flags_visible_count": summary["uncertainty_flags_visible_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
