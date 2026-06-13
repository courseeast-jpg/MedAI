#!/usr/bin/env python3
"""MEDAI-VERTEX-REAL-DOC-MEDICATION-SAFETY-NON-BYPASS-NO-LIVE-15Z-E."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_medication_safety_non_bypass import (  # noqa: E402
    evaluate_all_medication_safety_cases,
)
from execution.vertex_real_doc_readiness_gates import sanitize_readiness_report_payload  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "medication_safety_cases.json"
REFUSALS_JSON = REPORT_DIR / "medication_refusal_records.json"
MATRIX_MD = REPORT_DIR / "medication_safety_matrix.md"
FUTURE_JSON = REPORT_DIR / "future_review_package_preview.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_PUBLISHED_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/")
FORBIDDEN_ADVICE_TEXT = (
    "should take",
    "recommended dose",
    "contraindicated because",
    "interaction severity",
    "diagnosis is",
    "treatment plan",
)


def build_reports() -> dict[str, Any]:
    report = evaluate_all_medication_safety_cases()
    summary = {
        "block": "MEDAI-VERTEX-REAL-DOC-MEDICATION-SAFETY-NON-BYPASS-NO-LIVE-15Z-E",
        **report["summary"],
    }
    cases = {"cases": report["cases"]}
    refusals = {"medication_refusal_records": report["refusal_records"]}
    future = {"future_review_packages": report["future_review_packages"]}
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
        "| {case_id} | {facts} | {proof} | {forbidden} | {status} | {future} | {blocked} |".format(
            case_id=c["case_id"],
            facts=c["medication_candidate_facts_count"],
            proof=c["medication_safety_proof_present"],
            forbidden=c["forbidden_medical_decision_detected"],
            status=c["readiness_status"],
            future=c["future_review_package_created"],
            blocked=c["blocked"],
        )
        for c in cases
    ]
    return "\n".join(
        [
            "# 15Z-E medication safety matrix",
            "",
            "| Case | Candidate facts | Safety proof | Forbidden decision requested | Status | Future review package | Blocked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| medication_cases_total | `{summary['medication_cases_total']}` |",
            f"| medication_candidate_facts_total | `{summary['medication_candidate_facts_total']}` |",
            f"| forbidden_medical_decision_block_count | `{summary['forbidden_medical_decision_block_count']}` |",
            f"| future_review_package_created_count | `{summary['future_review_package_created_count']}` |",
            f"| real_doc_live_allowed_count | `{summary['real_doc_live_allowed_count']}` |",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "medication_safety_non_bypass_created",
        "medication_cases_total",
        "medication_cases_passed",
        "medication_facts_case_count",
        "medication_candidate_facts_total",
        "medication_safety_proof_created_count",
        "medication_safety_proof_required_block_count",
        "forbidden_medical_decision_block_count",
        "ddi_decision_blocked",
        "contraindication_decision_blocked",
        "dosage_advice_blocked",
        "treatment_advice_blocked",
        "diagnosis_output_blocked",
        "future_review_package_created_count",
        "future_review_only_count",
        "real_doc_live_allowed_count",
        "explicit_live_call_request_blocked",
        "active_write_blocked",
        "auto_accept_blocked",
        "ddi_decision_made_count",
        "contraindication_decision_made_count",
        "dosage_advice_made_count",
        "treatment_advice_made_count",
        "diagnosis_made_count",
        "raw_pii_in_report_count",
        "token_map_in_report_count",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ]
    lines = ["# MEDAI-VERTEX-REAL-DOC-MEDICATION-SAFETY-NON-BYPASS-NO-LIVE-15Z-E", ""]
    lines.extend(f"- {k}: `{summary[k]}`" for k in keys)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Medication mentions are candidate facts only and remain review-bound.",
            "- Forbidden medication decision output requests are blocked by boundary checks only.",
            "- Future packages are review-package-only; no real-doc live authorization is created.",
            "- No provider call, billing API call, active MKB write, production queue mutation, or auto-accept.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["medication_safety_non_bypass_created"] is True,
            summary["medication_cases_total"] == 17,
            summary["medication_cases_passed"] == 17,
            summary["medication_facts_case_count"] == 16,
            summary["medication_candidate_facts_total"] == 16,
            summary["medication_safety_proof_required_block_count"] == 2,
            summary["forbidden_medical_decision_block_count"] == 5,
            summary["ddi_decision_blocked"] is True,
            summary["contraindication_decision_blocked"] is True,
            summary["dosage_advice_blocked"] is True,
            summary["treatment_advice_blocked"] is True,
            summary["diagnosis_output_blocked"] is True,
            summary["future_review_package_created_count"] == 6,
            summary["future_review_only_count"] == 6,
            summary["real_doc_live_allowed_count"] == 0,
            summary["explicit_live_call_request_blocked"] is True,
            summary["active_write_blocked"] is True,
            summary["auto_accept_blocked"] is True,
            summary["ddi_decision_made_count"] == 0,
            summary["contraindication_decision_made_count"] == 0,
            summary["dosage_advice_made_count"] == 0,
            summary["treatment_advice_made_count"] == 0,
            summary["diagnosis_made_count"] == 0,
            summary["raw_pii_in_report_count"] == 0,
            summary["token_map_in_report_count"] == 0,
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
        "medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e_ready"
        if ready
        else "medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e_not_ready"
    )
    print(
        json.dumps(
            {
                "medication_cases_total": summary["medication_cases_total"],
                "medication_cases_passed": summary["medication_cases_passed"],
                "medication_candidate_facts_total": summary["medication_candidate_facts_total"],
                "forbidden_medical_decision_block_count": summary["forbidden_medical_decision_block_count"],
                "future_review_package_created_count": summary["future_review_package_created_count"],
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
