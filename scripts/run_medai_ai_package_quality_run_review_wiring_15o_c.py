#!/usr/bin/env python3
"""MEDAI-AI-PACKAGE-QUALITY-RUN-REVIEW-WIRING-15O-C validation."""
from __future__ import annotations

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

from app.ai_package_run_review_preview import (
    build_run_review_preview_report,
    render_run_review_preview_markdown,
)
from clinical_knowledge.privacy import check_public_report_payload

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_quality_run_review_wiring_15o_c"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CASES_JSON = REPORT_DIR / "run_review_surface_cases.json"
MATRIX_MD = REPORT_DIR / "run_review_surface_matrix.md"

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


def build_reports() -> dict[str, Any]:
    report = build_run_review_preview_report()
    summary = {
        "task_id": "MEDAI-AI-PACKAGE-QUALITY-RUN-REVIEW-WIRING-15O-C",
        "baseline_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "report_generation_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        **report["summary"],
        "privacy_result": "pending",
        "tests_run": [
            "python -m pytest tests/test_medai_ai_package_quality_run_review_wiring_15o_c.py -q",
            "python -m py_compile scripts/run_medai_ai_package_quality_run_review_wiring_15o_c.py",
            "python scripts/run_medai_ai_package_quality_run_review_wiring_15o_c.py",
        ],
        "files_changed": [
            "app/ai_package_run_review_preview.py",
            "app/main.py",
            "app/operator_compact_styles.py",
            "scripts/run_medai_ai_package_quality_run_review_wiring_15o_c.py",
            "tests/test_medai_ai_package_quality_run_review_wiring_15o_c.py",
            "reports/medai_ai_package_quality_run_review_wiring_15o_c/*",
        ],
        "push_status": "pending",
    }
    cases = {
        "task_id": summary["task_id"],
        "cases": report["cases"],
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
    }
    matrix = _matrix_markdown(report["cases"])
    implementation = _implementation_markdown(summary)
    privacy = _privacy_passes(summary, cases, matrix, implementation)
    summary["privacy_result"] = "passed" if privacy else "failed"
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "cases": cases,
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps(reports["cases"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    serialized = json.dumps(payloads, sort_keys=True, default=str)
    if any(token in serialized for token in BLOCKED_TOKENS):
        return False
    return bool(check_public_report_payload(payloads).passed)


def _matrix_markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# 15O-C Run & Review Surface Matrix",
        "",
        "| Family | Source body | Anchors | Candidate facts | Unknowns | Uncertainty flags | Compare preserved | Hallucinated fields |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for case in cases:
        lines.append(
            "| {family} | {body} | {anchors} | {facts} | {unknowns} | {uncertainty} | {compare} | {hallucinated} |".format(
                family=case["package_family_label"],
                body=bool(case["source_visible_body"]),
                anchors=len(case["evidence_anchors"]),
                facts=len(case["candidate_facts"]),
                unknowns=len(case["unknown_values"]),
                uncertainty=len(case["uncertainty_flags"]),
                compare=case["under_1_minute_compare_preserved"],
                hallucinated=case["hallucinated_field_count"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PACKAGE-QUALITY-RUN-REVIEW-WIRING-15O-C",
            "",
            f"- Baseline commit: `{summary['baseline_commit_short']}`",
            f"- Report generation commit: `{summary['report_generation_commit_short']}`",
            f"- Privacy result: `{summary['privacy_result']}`",
            f"- Run & Review package preview entry point added: `{summary['run_review_package_preview_entry_point_added']}`",
            f"- Package families previewed: `{summary['package_families_previewed']}`",
            f"- Source visible body present count: `{summary['source_visible_body_present_count']}`",
            f"- Evidence anchor present count: `{summary['evidence_anchor_present_count']}`",
            f"- Candidate facts separated count: `{summary['candidate_facts_separated_count']}`",
            f"- Unknown values explicit count: `{summary['unknown_values_explicit_count']}`",
            f"- Uncertainty flags visible count: `{summary['uncertainty_flags_visible_count']}`",
            f"- Under 1 minute compare preserved count: `{summary['under_1_minute_compare_preserved_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- Live call made: `{summary['live_call_made']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review required: `{summary['review_required']}`",
            f"- Billing check pending: `{summary['billing_check_pending']}`",
            f"- Push status: `{summary['push_status']}`",
            "",
            "## Scope",
            "",
            "- Adds deterministic fake/local AI package review preview wiring to the Run & Review tab.",
            "- Does not change OCR routing, thresholds, provider execution, medical decision logic, or MKB write logic.",
            "- Does not require or read MEDAI_VERTEX_LIVE_SMOKE_ALLOWED.",
            "",
        ]
    )


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    ready = all(
        [
            summary["privacy_result"] == "passed",
            summary["run_review_package_preview_entry_point_added"] is True,
            summary["all_run_review_preview_invariants_passed"] is True,
            summary["source_visible_body_present_count"] == summary["package_family_count"],
            summary["evidence_anchor_present_count"] == summary["package_family_count"],
            summary["candidate_facts_separated_count"] == summary["package_family_count"],
            summary["unknown_values_explicit_count"] == summary["package_family_count"],
            summary["uncertainty_flags_visible_count"] == summary["package_family_count"],
            summary["under_1_minute_compare_preserved_count"] == summary["package_family_count"],
            summary["hallucinated_field_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
        ]
    )
    print("medai_ai_package_quality_run_review_wiring_15o_c_ready" if ready else "medai_ai_package_quality_run_review_wiring_15o_c_not_ready")
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
                "privacy_result": summary["privacy_result"],
                "package_families_previewed": summary["package_families_previewed"],
                "run_review_package_preview_entry_point_added": summary["run_review_package_preview_entry_point_added"],
                "source_visible_body_present_count": summary["source_visible_body_present_count"],
                "evidence_anchor_present_count": summary["evidence_anchor_present_count"],
                "candidate_facts_separated_count": summary["candidate_facts_separated_count"],
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
