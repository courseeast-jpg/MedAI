#!/usr/bin/env python3
"""MEDAI-VERTEX-REAL-DOC-READINESS-INTEGRATED-HARNESS-NO-LIVE-15Z-F."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_readiness_gates import sanitize_readiness_report_payload  # noqa: E402
from execution.vertex_real_doc_readiness_integrated_harness import (  # noqa: E402
    evaluate_all_integrated_readiness_cases,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "integrated_readiness_cases.json"
MATRIX_MD = REPORT_DIR / "integrated_gate_trace_matrix.md"
REFUSALS_JSON = REPORT_DIR / "integrated_refusal_records.json"
FUTURE_JSON = REPORT_DIR / "future_operator_review_package_preview.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_PUBLISHED_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/")
FORBIDDEN_ADVICE_TEXT = ("should take", "recommended dose", "contraindicated because", "interaction severity", "diagnosis is", "treatment plan")


def build_reports() -> dict[str, Any]:
    report = evaluate_all_integrated_readiness_cases()
    summary = {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-INTEGRATED-HARNESS-NO-LIVE-15Z-F",
        **report["summary"],
    }
    cases = {"cases": report["cases"]}
    refusals = {"integrated_refusal_records": report["refusal_records"]}
    future = {"future_operator_review_packages": report["future_operator_review_packages"]}
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
    CASES_JSON.write_text(json.dumps(sanitize_readiness_report_payload(reports["cases"]), indent=2), encoding="utf-8")
    REFUSALS_JSON.write_text(json.dumps(sanitize_readiness_report_payload(reports["refusals"]), indent=2), encoding="utf-8")
    FUTURE_JSON.write_text(json.dumps(sanitize_readiness_report_payload(reports["future"]), indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str) for p in payloads)
    if any(token in published for token in FORBIDDEN_PUBLISHED_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    if any(text in published.lower() for text in FORBIDDEN_ADVICE_TEXT):
        return False
    return bool(check_public_report_payload(payloads).passed)


def _matrix_markdown(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = [
        "| {case_id} | {status} | {package} | {pii} | {vault} | {shape} | {handoff} | {auth} | {billing} | {med} | {blocked} |".format(
            case_id=c["case_id"],
            status=c["integrated_status"],
            package=c["package_created"],
            pii=c["pii_redaction_passed"],
            vault=c["vault_isolated"],
            shape=c["request_shape_valid"],
            handoff=c["review_handoff_created"],
            auth=c["authorization_intent_present"],
            billing=c["billing_ack_present"],
            med=c["medication_safety_proof_present"],
            blocked=c["blocked"],
        )
        for c in cases
    ]
    return "\n".join(
        [
            "# 15Z-F integrated gate trace matrix",
            "",
            "| Case | Status | Package | PII | Vault | Request shape | Handoff | Auth | Billing | Med proof | Blocked |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| integrated_cases_total | `{summary['integrated_cases_total']}` |",
            f"| future_operator_review_package_created_count | `{summary['future_operator_review_package_created_count']}` |",
            f"| blocked_case_count | `{summary['blocked_case_count']}` |",
            f"| real_doc_live_allowed_count | `{summary['real_doc_live_allowed_count']}` |",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "integrated_readiness_harness_created",
        "integrated_cases_total",
        "integrated_cases_passed",
        "future_operator_review_package_created_count",
        "future_operator_review_only_count",
        "blocked_case_count",
        "refusal_records_created_count",
        "pii_redaction_pass_count",
        "vault_isolation_pass_count",
        "request_shape_valid_count",
        "review_handoff_created_count",
        "authorization_intent_present_count",
        "billing_ack_present_count",
        "medication_safety_proof_present_count",
        "explicit_live_call_request_blocked",
        "active_write_blocked",
        "auto_accept_blocked",
        "medication_safety_non_bypass_enforced",
        "medical_decision_blocked",
        "real_doc_live_allowed_count",
        "raw_pii_in_package_count",
        "raw_pii_in_report_count",
        "token_map_in_package_count",
        "token_map_in_report_count",
        "medical_decision_made_count",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ]
    lines = ["# MEDAI-VERTEX-REAL-DOC-READINESS-INTEGRATED-HARNESS-NO-LIVE-15Z-F", ""]
    lines.extend(f"- {k}: `{summary[k]}`" for k in keys)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Composes 15Z-A through 15Z-E no-live gates into one integrated harness.",
            "- Produces future operator review packages only; no live routing is authorized.",
            "- Reports contain gate states, refusal reasons, and fingerprints only.",
            "- No provider call, billing API call, active write, production queue mutation, auto-accept, or medical decision logic.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["integrated_readiness_harness_created"] is True,
            summary["integrated_cases_total"] == 17,
            summary["integrated_cases_passed"] == 17,
            summary["future_operator_review_package_created_count"] == 3,
            summary["future_operator_review_only_count"] == 3,
            summary["blocked_case_count"] == 14,
            summary["refusal_records_created_count"] == 14,
            summary["real_doc_live_allowed_count"] == 0,
            summary["raw_pii_in_package_count"] == 0,
            summary["raw_pii_in_report_count"] == 0,
            summary["token_map_in_package_count"] == 0,
            summary["token_map_in_report_count"] == 0,
            summary["medical_decision_made_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
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
        "medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f_ready"
        if ready
        else "medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f_not_ready"
    )
    print(
        json.dumps(
            {
                "integrated_cases_total": summary["integrated_cases_total"],
                "integrated_cases_passed": summary["integrated_cases_passed"],
                "future_operator_review_package_created_count": summary["future_operator_review_package_created_count"],
                "blocked_case_count": summary["blocked_case_count"],
                "real_doc_live_allowed_count": summary["real_doc_live_allowed_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "billing_api_used": summary["billing_api_used"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
