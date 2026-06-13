#!/usr/bin/env python3
"""MEDAI-AI-PACKAGE-QUALITY-EVAL-FAKE-LOCAL-15O-A validation."""
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

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_package_quality_eval import evaluate_all_package_quality

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_quality_eval_fake_local_15o_a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CASES_JSON = REPORT_DIR / "package_quality_cases.json"
MATRIX_MD = REPORT_DIR / "package_quality_matrix.md"

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
    result = evaluate_all_package_quality()
    summary = {
        "task_id": "MEDAI-AI-PACKAGE-QUALITY-EVAL-FAKE-LOCAL-15O-A",
        "baseline_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "report_generation_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "package_families_evaluated": result["summary"]["package_families_evaluated"],
        "under_1_minute_compare_pass_count": result["summary"]["under_1_minute_compare_pass_count"],
        "hallucinated_field_count": result["summary"]["hallucinated_field_count"],
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
        "all_cases_under_1_minute": result["summary"]["all_cases_under_1_minute"],
        "all_safety_invariants_passed": result["summary"]["all_safety_invariants_passed"],
        "privacy_result": "pending",
        "tests_run": [
            "python -m pytest tests/test_medai_ai_package_quality_eval_fake_local_15o_a.py -q",
            "python -m py_compile scripts/run_medai_ai_package_quality_eval_fake_local_15o_a.py",
            "python scripts/run_medai_ai_package_quality_eval_fake_local_15o_a.py",
        ],
        "files_changed": [
            "execution/ai_package_quality_eval.py",
            "scripts/run_medai_ai_package_quality_eval_fake_local_15o_a.py",
            "tests/test_medai_ai_package_quality_eval_fake_local_15o_a.py",
            "reports/medai_ai_package_quality_eval_fake_local_15o_a/*",
        ],
        "push_status": "pending",
    }
    cases = {
        "task_id": summary["task_id"],
        "cases": result["cases"],
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
    }
    matrix = _matrix_markdown(result["cases"])
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
        "# 15O-A Package Quality Matrix",
        "",
        "| Family | Sections | Observations | Anchors | Compare seconds | Under 1 minute | Hallucinated fields |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for case in cases:
        metrics = case["metrics"]
        lines.append(
            "| {display} | {sections} | {observations} | {anchors} | {seconds} | {under} | {hallucinated} |".format(
                display=metrics["display_name"],
                sections=metrics["section_count"],
                observations=metrics["observation_count"],
                anchors=metrics["evidence_anchor_count"],
                seconds=metrics["estimated_human_compare_seconds"],
                under=metrics["under_1_minute_compare_pass"],
                hallucinated=metrics["hallucinated_field_count"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-AI-PACKAGE-QUALITY-EVAL-FAKE-LOCAL-15O-A",
            "",
            f"- Baseline commit: `{summary['baseline_commit_short']}`",
            f"- Report generation commit: `{summary['report_generation_commit_short']}`",
            f"- Privacy result: `{summary['privacy_result']}`",
            f"- Package families evaluated: `{summary['package_families_evaluated']}`",
            f"- Under 1 minute compare pass count: `{summary['under_1_minute_compare_pass_count']}`",
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
            "- Deterministic fake/local fixtures only.",
            "- Scores operator review-package usability, not clinical correctness.",
            "- Does not call Gemini, Vertex, AI Studio, Claude, OpenAI, Ollama, or local LLM providers.",
            "- Does not mutate runtime databases or write active MKB records.",
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
            summary["all_cases_under_1_minute"] is True,
            summary["all_safety_invariants_passed"] is True,
            summary["hallucinated_field_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
        ]
    )
    print("medai_ai_package_quality_eval_fake_local_15o_a_ready" if ready else "medai_ai_package_quality_eval_fake_local_15o_a_not_ready")
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
                "privacy_result": summary["privacy_result"],
                "package_families_evaluated": summary["package_families_evaluated"],
                "under_1_minute_compare_pass_count": summary["under_1_minute_compare_pass_count"],
                "hallucinated_field_count": summary["hallucinated_field_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
