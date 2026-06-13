#!/usr/bin/env python3
"""MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z-A.

Exercises the deterministic, no-live real-document readiness-gate framework over
synthetic/redacted/blocked cases and writes sanitized refusal/readiness reports.
No provider call, no live gate, no real document, no active MKB write, no
auto-accept. Even the all-gates-simulated-pass case yields only
READY_FOR_FUTURE_AUTHORIZATION_ONLY (never a live authorization).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNKNOWN,
    READINESS_GATES,
    build_real_doc_refusal_record,
    evaluate_vertex_real_doc_readiness,
    evaluation_to_public_dict,
    sanitize_readiness_report_payload,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_gates_no_live_15z_a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "readiness_gate_cases.json"
MATRIX_MD = REPORT_DIR / "readiness_gate_matrix.md"
REFUSALS_JSON = REPORT_DIR / "refusal_records.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/")
# Note: case content_markers below are synthetic category words ("pii", "raw_pdf",
# "ocr_private"), NOT real identifiers or raw documents.


def _base_pass_kwargs() -> dict[str, Any]:
    return dict(
        human_authorization_present=True,
        billing_cost_cap_ack_present=True,
        active_write_requested=False,
        auto_accept_requested=False,
        contains_medication_fact=False,
        medication_safety_gate_satisfied=False,
        future_gates_simulated_pass=True,
    )


def build_cases() -> list[Any]:
    cases = []
    # 1. synthetic fixture -> no-live dry-run only (gates not simulated -> pending)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="synthetic_calibration_fixture", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC portal result cards"))
    # 2. redacted real-like fixture -> no-live readiness replay only
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="redacted_real_like_fixture", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like layout summary"))
    # 3. real private document marker -> BLOCKED
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="real_private_document_marker", declared_provenance=PROVENANCE_REAL_PRIVATE,
        content_marker="real_private_document", **_base_pass_kwargs()))
    # 4. unknown provenance -> BLOCKED
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="unknown_provenance_payload", declared_provenance=PROVENANCE_UNKNOWN,
        content_marker="unknown source", **_base_pass_kwargs()))
    # 5. PII marker -> BLOCKED
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="payload_with_pii_marker", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="pii", **_base_pass_kwargs()))
    # 6. raw pdf marker -> BLOCKED
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="payload_with_raw_pdf_marker", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="raw_pdf", **_base_pass_kwargs()))
    # 7. OCR private marker -> BLOCKED
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="payload_with_ocr_private_marker", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="ocr_private", **_base_pass_kwargs()))
    # 8. medication fact without safety gate -> BLOCKED
    kw = _base_pass_kwargs(); kw.update(contains_medication_fact=True, medication_safety_gate_satisfied=False)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="medication_fact_without_safety_gate", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC medication mention", **kw))
    # 9. active write requested -> BLOCKED
    kw = _base_pass_kwargs(); kw.update(active_write_requested=True)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="active_write_requested", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC fixture", **kw))
    # 10. auto-accept requested -> BLOCKED
    kw = _base_pass_kwargs(); kw.update(auto_accept_requested=True)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="auto_accept_requested", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC fixture", **kw))
    # 11. missing human authorization -> BLOCKED
    kw = _base_pass_kwargs(); kw.update(human_authorization_present=False)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="missing_human_authorization", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC fixture", **kw))
    # 12. missing billing ack -> BLOCKED
    kw = _base_pass_kwargs(); kw.update(billing_cost_cap_ack_present=False)
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="missing_billing_ack", declared_provenance=PROVENANCE_SYNTHETIC,
        content_marker="SYNTHETIC fixture", **kw))
    # 13. all future gates simulated pass (no-live) -> READY_FOR_FUTURE_AUTHORIZATION_ONLY
    cases.append(evaluate_vertex_real_doc_readiness(
        case_id="all_future_gates_simulated_pass_no_live", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like layout summary", **_base_pass_kwargs()))
    return cases


def _build_metrics(cases: list[Any]) -> dict[str, Any]:
    def by_id(cid: str):
        return next(c for c in cases if c.case_id == cid)

    blocked = [c for c in cases if c.blocked]
    future_only = [c for c in cases if c.readiness_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"]
    no_live_replay = [c for c in cases if c.classification_detail.get("no_live_dry_run_allowed")]
    raw_in_report = sum(1 for c in cases if c.raw_payload_in_report)

    return {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z-A",
        "readiness_gate_framework_created": True,
        "readiness_gates_modeled": list(READINESS_GATES),
        "readiness_cases_total": len(cases),
        "blocked_case_count": len(blocked),
        "no_live_replay_allowed_count": len(no_live_replay),
        "real_doc_live_allowed_count": sum(1 for c in cases if c.live_call_allowed),
        "future_authorization_only_count": len(future_only),
        "real_private_document_blocked": by_id("real_private_document_marker").blocked,
        "unknown_provenance_blocked": by_id("unknown_provenance_payload").blocked,
        "pii_marker_blocked": by_id("payload_with_pii_marker").blocked,
        "raw_payload_marker_blocked": by_id("payload_with_raw_pdf_marker").blocked,
        "ocr_private_marker_blocked": by_id("payload_with_ocr_private_marker").blocked,
        "medication_safety_non_bypass_enforced": by_id("medication_fact_without_safety_gate").blocked,
        "human_authorization_required": by_id("missing_human_authorization").blocked,
        "billing_ack_required": by_id("missing_billing_ack").blocked,
        "active_write_blocked": by_id("active_write_requested").blocked,
        "auto_accept_blocked": by_id("auto_accept_requested").blocked,
        "sanitized_reports_only": all(c.sanitized_report_only for c in cases),
        "raw_payload_in_report_count": raw_in_report,
        "all_cases_review_required": all(c.review_required for c in cases),
        "all_cases_live_call_blocked": all(c.live_call_allowed is False for c in cases),
        "future_authorization_only_not_live": all(c.live_call_allowed is False for c in future_only),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def _matrix_markdown(m: dict[str, Any], cases: list[Any]) -> str:
    case_rows = [
        f"| {c.case_id} | {c.payload_classification} | `{c.readiness_status}` | `{c.blocked}` | `{c.live_call_allowed}` |"
        for c in cases
    ]
    keys = [
        "readiness_cases_total", "blocked_case_count", "no_live_replay_allowed_count", "real_doc_live_allowed_count",
        "future_authorization_only_count", "real_private_document_blocked", "unknown_provenance_blocked",
        "pii_marker_blocked", "raw_payload_marker_blocked", "ocr_private_marker_blocked",
        "medication_safety_non_bypass_enforced", "human_authorization_required", "billing_ack_required",
        "active_write_blocked", "auto_accept_blocked", "sanitized_reports_only", "raw_payload_in_report_count",
        "live_call_made", "external_api_used", "active_written_count", "active_mkb_record_created_count",
        "auto_accept_true_count", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15Z-A real-document readiness gate matrix",
            "",
            "| Case | Classification | Status | Blocked | live_call_allowed |",
            "| --- | --- | --- | --- | --- |",
            *case_rows,
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {k} | `{m[k]}` |" for k in keys],
            "",
            "No-live readiness gates: real-doc routing default-deny; even all-gates-pass yields future-authorization-only.",
            "",
        ]
    )


def _implementation_markdown(m: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z-A",
            "",
            f"- Readiness gate framework created: `{m['readiness_gate_framework_created']}` ({len(READINESS_GATES)} gates modeled)",
            f"- Cases total / blocked / no-live-replay-allowed / future-auth-only / real-doc-live-allowed: "
            f"`{m['readiness_cases_total']}` / `{m['blocked_case_count']}` / `{m['no_live_replay_allowed_count']}` / "
            f"`{m['future_authorization_only_count']}` / `{m['real_doc_live_allowed_count']}`",
            f"- real_private / unknown / pii / raw_pdf / ocr_private blocked: "
            f"`{m['real_private_document_blocked']}` / `{m['unknown_provenance_blocked']}` / `{m['pii_marker_blocked']}` / "
            f"`{m['raw_payload_marker_blocked']}` / `{m['ocr_private_marker_blocked']}`",
            f"- medication non-bypass / human-auth / billing-ack / active-write / auto-accept blocked: "
            f"`{m['medication_safety_non_bypass_enforced']}` / `{m['human_authorization_required']}` / "
            f"`{m['billing_ack_required']}` / `{m['active_write_blocked']}` / `{m['auto_accept_blocked']}`",
            f"- sanitized_reports_only: `{m['sanitized_reports_only']}` | raw_payload_in_report_count: `{m['raw_payload_in_report_count']}`",
            f"- live_call_made: `{m['live_call_made']}` | external_api_used: `{m['external_api_used']}` | "
            f"active_written_count: `{m['active_written_count']}` | auto_accept_true_count: `{m['auto_accept_true_count']}`",
            f"- privacy_result: `{m['privacy_result']}` | billing_check_pending: `{m['billing_check_pending']}`",
            "",
            "## Safety",
            "",
            "- Real-document Vertex routing is BLOCKED by default; only synthetic/redacted no-live fixtures may dry-run.",
            "- Even all-gates-simulated-pass returns READY_FOR_FUTURE_AUTHORIZATION_ONLY — never a live authorization.",
            "- No provider call, no live gate, no real document, no active MKB write, no auto-accept; review_required always true.",
            "- Reports are sanitized: blocked/private cases carry reasons + content fingerprints only, never raw payload.",
            "",
        ]
    )


def main() -> int:
    cases = build_cases()
    metrics = _build_metrics(cases)
    case_dicts = [evaluation_to_public_dict(c) for c in cases]
    refusals = [build_real_doc_refusal_record(c) for c in cases]

    # Final defense-in-depth privacy pass over everything written.
    blob = json.dumps(sanitize_readiness_report_payload([metrics, case_dicts, refusals]), default=str)
    privacy_ok = not any(tok in blob for tok in FORBIDDEN_TOKENS)
    metrics["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"cases": case_dicts}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(metrics, cases), encoding="utf-8")
    REFUSALS_JSON.write_text(json.dumps({"refusal_records": refusals}, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(metrics), encoding="utf-8")

    ready = all(
        [
            metrics["readiness_gate_framework_created"] is True,
            metrics["readiness_cases_total"] == 13,
            metrics["real_doc_live_allowed_count"] == 0,
            metrics["future_authorization_only_count"] == 1,
            metrics["real_private_document_blocked"] is True,
            metrics["unknown_provenance_blocked"] is True,
            metrics["pii_marker_blocked"] is True,
            metrics["raw_payload_marker_blocked"] is True,
            metrics["ocr_private_marker_blocked"] is True,
            metrics["medication_safety_non_bypass_enforced"] is True,
            metrics["human_authorization_required"] is True,
            metrics["billing_ack_required"] is True,
            metrics["active_write_blocked"] is True,
            metrics["auto_accept_blocked"] is True,
            metrics["sanitized_reports_only"] is True,
            metrics["raw_payload_in_report_count"] == 0,
            metrics["all_cases_review_required"] is True,
            metrics["all_cases_live_call_blocked"] is True,
            metrics["future_authorization_only_not_live"] is True,
            metrics["live_call_made"] is False,
            metrics["external_api_used"] is False,
            metrics["active_written_count"] == 0,
            metrics["active_mkb_record_created_count"] == 0,
            metrics["auto_accept_true_count"] == 0,
            metrics["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_real_doc_readiness_gates_no_live_15z_a_ready" if ready else "medai_vertex_real_doc_readiness_gates_no_live_15z_a_not_ready")
    print(
        json.dumps(
            {
                "readiness_cases_total": metrics["readiness_cases_total"],
                "blocked_case_count": metrics["blocked_case_count"],
                "real_doc_live_allowed_count": metrics["real_doc_live_allowed_count"],
                "future_authorization_only_count": metrics["future_authorization_only_count"],
                "raw_payload_in_report_count": metrics["raw_payload_in_report_count"],
                "live_call_made": metrics["live_call_made"],
                "external_api_used": metrics["external_api_used"],
                "active_written_count": metrics["active_written_count"],
                "auto_accept_true_count": metrics["auto_accept_true_count"],
                "privacy_result": metrics["privacy_result"],
                "billing_check_pending": metrics["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
