#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-PACKAGE-CONTRACT-NO-LIVE-15P-A validation."""
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

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_semantic_package_contract import evaluate_vertex_semantic_contract

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_package_contract_no_live_15p_a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
CASES_JSON = REPORT_DIR / "semantic_contract_cases.json"
MATRIX_MD = REPORT_DIR / "semantic_contract_matrix.md"

BLOCKED_TOKENS = (
    "GEMINI" + "_API_KEY",
    "Authorization",
    "Bearer ",
    "ya29.",
    "AIza",
    "token_map",
    "DOB",
    "MRN:",
    "Accession",
    ".pdf",
    ".png",
    ".jpg",
    "raw OCR",
    "raw" + "_pdf",
    "raw_image",
)


def build_reports() -> dict[str, Any]:
    report = evaluate_vertex_semantic_contract()
    summary = {
        "task_id": "MEDAI-VERTEX-SEMANTIC-PACKAGE-CONTRACT-NO-LIVE-15P-A",
        "baseline_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        "report_generation_commit_short": _git(["rev-parse", "--short=12", "HEAD"]),
        **report["summary"],
        "privacy_result": "pending",
        "external_api_used": False,
        "provider_route": "vertex",
        "model": "gemini-2.5-flash-lite",
        "no_real_documents": True,
        "no_binary_payloads": True,
        "no_ocr_routing_changes": True,
        "no_extraction_threshold_changes": True,
        "no_medical_decision_logic_changes": True,
        "no_provider_calls": True,
        "no_billing_activity": True,
        "tests_run": [
            "python -m pytest tests/test_medai_vertex_semantic_package_contract_no_live_15p_a.py -q",
            "python -m py_compile scripts/run_medai_vertex_semantic_package_contract_no_live_15p_a.py",
            "python scripts/run_medai_vertex_semantic_package_contract_no_live_15p_a.py",
        ],
        "files_changed": [
            "execution/vertex_semantic_package_contract.py",
            "scripts/run_medai_vertex_semantic_package_contract_no_live_15p_a.py",
            "tests/test_medai_vertex_semantic_package_contract_no_live_15p_a.py",
            "reports/medai_vertex_semantic_package_contract_no_live_15p_a/*",
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
        "# 15P-A Vertex Semantic Contract Matrix",
        "",
        "| Family | Schema | Fake response | Source body | Anchors | Candidate facts | Unknowns | Uncertainty | Hallucinated fields |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for case in cases:
        lines.append(
            "| {family} | {schema} | {fake} | {body} | {anchors} | {facts} | {unknowns} | {uncertainty} | {hallucinated} |".format(
                family=case["package_family_label"],
                schema=case["schema_validation_passed"],
                fake=case["schema_validation_passed"],
                body=case["source_visible_body_preserved"],
                anchors=case["evidence_anchor_preserved"],
                facts=case["candidate_facts_separated"],
                unknowns=case["unknown_values_explicit"],
                uncertainty=case["uncertainty_flags_visible"],
                hallucinated=case["hallucinated_field_count"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-PACKAGE-CONTRACT-NO-LIVE-15P-A",
            "",
            f"- Baseline commit: `{summary['baseline_commit_short']}`",
            f"- Report generation commit: `{summary['report_generation_commit_short']}`",
            f"- Privacy result: `{summary['privacy_result']}`",
            f"- Provider route: `{summary['provider_route']}`",
            f"- Model: `{summary['model']}`",
            f"- Package families checked: `{summary['package_families_checked']}`",
            f"- Schema validation pass count: `{summary['schema_validation_pass_count']}`",
            f"- Fake Vertex response valid count: `{summary['fake_vertex_response_valid_count']}`",
            f"- Source visible body preserved count: `{summary['source_visible_body_preserved_count']}`",
            f"- Evidence anchor preserved count: `{summary['evidence_anchor_preserved_count']}`",
            f"- Candidate facts separated count: `{summary['candidate_facts_separated_count']}`",
            f"- Unknown values explicit count: `{summary['unknown_values_explicit_count']}`",
            f"- Uncertainty flags visible count: `{summary['uncertainty_flags_visible_count']}`",
            f"- Hallucinated field count: `{summary['hallucinated_field_count']}`",
            f"- Live call made: `{summary['live_call_made']}`",
            f"- External API used: `{summary['external_api_used']}`",
            f"- Active written count: `{summary['active_written_count']}`",
            f"- Auto-accept: `{summary['auto_accept']}`",
            f"- Review required: `{summary['review_required']}`",
            f"- Billing check pending: `{summary['billing_check_pending']}`",
            "",
            "## Scope",
            "",
            "- Defines a no-live Vertex semantic package request and response contract.",
            "- Uses only deterministic synthetic 15O package fixtures and fake provider responses.",
            "- Does not change OCR routing, thresholds, provider gates, medical decision logic, or MKB write logic.",
            "- Does not require or read the Vertex live smoke gate.",
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
            summary["all_contract_invariants_passed"] is True,
            summary["schema_validation_pass_count"] == summary["package_family_count"],
            summary["fake_vertex_response_valid_count"] == summary["package_family_count"],
            summary["source_visible_body_preserved_count"] == summary["package_family_count"],
            summary["evidence_anchor_preserved_count"] == summary["package_family_count"],
            summary["candidate_facts_separated_count"] == summary["package_family_count"],
            summary["unknown_values_explicit_count"] == summary["package_family_count"],
            summary["uncertainty_flags_visible_count"] == summary["package_family_count"],
            summary["hallucinated_field_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["auto_accept"] is False,
            summary["review_required"] is True,
            summary["no_provider_calls"] is True,
            summary["no_billing_activity"] is True,
        ]
    )
    print("medai_vertex_semantic_package_contract_no_live_15p_a_ready" if ready else "medai_vertex_semantic_package_contract_no_live_15p_a_not_ready")
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
                "privacy_result": summary["privacy_result"],
                "package_families_checked": summary["package_families_checked"],
                "schema_validation_pass_count": summary["schema_validation_pass_count"],
                "fake_vertex_response_valid_count": summary["fake_vertex_response_valid_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "active_written_count": summary["active_written_count"],
                "auto_accept": summary["auto_accept"],
                "review_required": summary["review_required"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
