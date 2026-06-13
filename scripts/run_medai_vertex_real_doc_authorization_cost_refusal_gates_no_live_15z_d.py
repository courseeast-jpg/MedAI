#!/usr/bin/env python3
"""MEDAI-VERTEX-REAL-DOC-AUTHORIZATION-COST-REFUSAL-GATES-NO-LIVE-15Z-D."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_authorization_cost_refusal_gates import (  # noqa: E402
    evaluate_all_authorization_cost_refusal_cases,
)
from execution.vertex_real_doc_readiness_gates import sanitize_readiness_report_payload  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "authorization_cost_cases.json"
REFUSALS_JSON = REPORT_DIR / "refusal_records.json"
FUTURE_PACKAGE_JSON = REPORT_DIR / "future_authorization_package_preview.json"
MATRIX_MD = REPORT_DIR / "authorization_cost_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_PUBLISHED_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/")


def build_reports() -> dict[str, Any]:
    report = evaluate_all_authorization_cost_refusal_cases()
    summary = {
        "block": "MEDAI-VERTEX-REAL-DOC-AUTHORIZATION-COST-REFUSAL-GATES-NO-LIVE-15Z-D",
        **report["summary"],
    }
    cases = {"cases": report["cases"]}
    refusals = {"refusal_records": report["refusal_records"]}
    future = {"future_authorization_packages": report["future_packages"]}
    matrix = _matrix_markdown(report["cases"], summary)
    implementation = _implementation_markdown(summary)
    privacy = _privacy_passes(summary, cases, refusals, future, matrix, implementation)
    summary["privacy_result"] = "passed" if privacy else "failed"
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "cases": cases,
        "refusals": refusals,
        "future": future,
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CASES_JSON.write_text(
        json.dumps(sanitize_readiness_report_payload(reports["cases"]), indent=2),
        encoding="utf-8",
    )
    REFUSALS_JSON.write_text(
        json.dumps(sanitize_readiness_report_payload(reports["refusals"]), indent=2),
        encoding="utf-8",
    )
    FUTURE_PACKAGE_JSON.write_text(
        json.dumps(sanitize_readiness_report_payload(reports["future"]), indent=2),
        encoding="utf-8",
    )
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str) for p in payloads)
    if any(token in published for token in FORBIDDEN_PUBLISHED_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _matrix_markdown(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = [
        "| {case_id} | {auth} | {bill} | {handoff} | {future} | {status} | {blocked} |".format(
            case_id=c["case_id"],
            auth=c["authorization_intent_present"],
            bill=c["billing_ack_present"],
            handoff=c["review_handoff_present"],
            future=c["future_package_created"],
            status=c["readiness_status"],
            blocked=c["blocked"],
        )
        for c in cases
    ]
    return "\n".join(
        [
            "# 15Z-D authorization cost refusal matrix",
            "",
            "| Case | Auth intent | Billing ack | Handoff | Future package | Status | Blocked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| authorization_cases_total | `{summary['authorization_cases_total']}` |",
            f"| blocked_case_count | `{summary['blocked_case_count']}` |",
            f"| refusal_records_created_count | `{summary['refusal_records_created_count']}` |",
            f"| future_authorization_package_created_count | `{summary['future_authorization_package_created_count']}` |",
            f"| real_doc_live_allowed_count | `{summary['real_doc_live_allowed_count']}` |",
            f"| no_billing_api_used | `{summary['no_billing_api_used']}` |",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "authorization_cost_refusal_gates_created",
        "authorization_cases_total",
        "authorization_cases_passed",
        "blocked_case_count",
        "refusal_records_created_count",
        "human_authorization_required_count",
        "billing_ack_required_count",
        "authorization_intent_created_count",
        "billing_ack_created_count",
        "future_authorization_package_created_count",
        "future_authorization_only_count",
        "real_doc_live_allowed_count",
        "explicit_live_call_request_blocked",
        "active_write_blocked",
        "auto_accept_blocked",
        "medication_safety_non_bypass_enforced",
        "no_billing_api_used",
        "estimated_cost_ceiling_usd_max",
        "token_budget_ceiling_max",
        "raw_pii_in_report_count",
        "token_map_in_report_count",
        "live_call_made",
        "external_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ]
    lines = ["# MEDAI-VERTEX-REAL-DOC-AUTHORIZATION-COST-REFUSAL-GATES-NO-LIVE-15Z-D", ""]
    lines.extend(f"- {k}: `{summary[k]}`" for k in keys)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Models no-live human authorization intent and billing/cost-cap acknowledgement.",
            "- Creates sanitized refusal records for every blocked path.",
            "- Creates a future authorization package preview only; no live call is authorized.",
            "- No provider call, no billing API call, no active MKB write, no production queue mutation, no auto-accept.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["authorization_cost_refusal_gates_created"] is True,
            summary["authorization_cases_total"] == 17,
            summary["authorization_cases_passed"] == 17,
            summary["blocked_case_count"] == 16,
            summary["refusal_records_created_count"] == 16,
            summary["human_authorization_required_count"] == 4,
            summary["billing_ack_required_count"] == 4,
            summary["future_authorization_package_created_count"] == 1,
            summary["future_authorization_only_count"] == 1,
            summary["real_doc_live_allowed_count"] == 0,
            summary["explicit_live_call_request_blocked"] is True,
            summary["active_write_blocked"] is True,
            summary["auto_accept_blocked"] is True,
            summary["medication_safety_non_bypass_enforced"] is True,
            summary["no_billing_api_used"] is True,
            summary["estimated_cost_ceiling_usd_max"] <= 0.01,
            summary["token_budget_ceiling_max"] <= 512,
            summary["raw_pii_in_report_count"] == 0,
            summary["token_map_in_report_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["privacy_result"] == "passed",
            summary["billing_check_pending"] is True,
        ]
    )


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    ready = _ready(summary)
    print(
        "medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d_ready"
        if ready
        else "medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d_not_ready"
    )
    print(
        json.dumps(
            {
                "authorization_cases_total": summary["authorization_cases_total"],
                "authorization_cases_passed": summary["authorization_cases_passed"],
                "blocked_case_count": summary["blocked_case_count"],
                "refusal_records_created_count": summary["refusal_records_created_count"],
                "future_authorization_package_created_count": summary["future_authorization_package_created_count"],
                "real_doc_live_allowed_count": summary["real_doc_live_allowed_count"],
                "no_billing_api_used": summary["no_billing_api_used"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
