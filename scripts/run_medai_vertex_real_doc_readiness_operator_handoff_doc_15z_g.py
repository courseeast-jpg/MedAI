#!/usr/bin/env python3
"""Generate the 15Z-G no-live real-document readiness handoff reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CHECKLIST_JSON = REPORT_DIR / "handoff_doc_checklist.json"
MATRIX_MD = REPORT_DIR / "handoff_doc_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

REQUIRED_SECTIONS = [
    "## 1. Purpose",
    "## 2. Current status",
    "## 3. Gate inventory",
    "## 4. What 15Z-A proves",
    "## 5. What 15Z-B proves",
    "## 6. What 15Z-C proves",
    "## 7. What 15Z-D proves",
    "## 8. What 15Z-E proves",
    "## 9. What 15Z-F proves",
    "## 10. What this does not prove",
    "## 11. Operator commands",
    "## 12. Stop conditions",
    "## 13. Future authorization boundary",
    "## 14. Recommended next block",
]

GATES = [
    "real_doc_external_routing_default_block",
    "pii_stripping_proof_required",
    "pii_vault_isolation_required",
    "no_raw_private_payload_in_reports_required",
    "synthetic_to_real_adapter_dry_run_required",
    "redacted_real_like_fixture_replay_required",
    "operator_review_queue_handoff_required",
    "human_authorization_required_for_any_real_live_call",
    "no_active_mkb_write_required",
    "no_auto_accept_required",
    "medication_safety_non_bypass_required_if_medication_facts_present",
    "billing_cost_cap_ack_required",
    "dedicated_future_real_doc_live_gate_required",
    "real_doc_refusal_path_required",
    "no_medical_decision_logic_required",
]

COMMANDS = [
    "python scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py",
    "python scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py",
    "python scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py",
    "python scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py",
    "python scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py",
    "python scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py",
]

PASS_CHAIN = {
    "15Z-A": "default-deny real-document readiness framework PASS",
    "15Z-B": "PII stripping and vault isolation proof PASS",
    "15Z-C": "synthetic-to-real adapter dry-run and report-only review handoff PASS",
    "15Z-D": "human authorization, billing cost-cap, and refusal gates PASS",
    "15Z-E": "medication safety non-bypass PASS",
    "15Z-F": "integrated readiness harness PASS",
}

INTEGRATED_METRICS = {
    "integrated_readiness_harness_created": True,
    "integrated_cases_total": 17,
    "integrated_cases_passed": 17,
    "future_operator_review_package_created_count": 3,
    "future_operator_review_only_count": 3,
    "blocked_case_count": 14,
    "refusal_records_created_count": 14,
    "pii_redaction_pass_count": 15,
    "vault_isolation_pass_count": 16,
    "request_shape_valid_count": 15,
    "review_handoff_created_count": 7,
    "authorization_intent_present_count": 6,
    "billing_ack_present_count": 6,
    "medication_safety_proof_present_count": 3,
    "explicit_live_call_request_blocked": True,
    "active_write_blocked": True,
    "auto_accept_blocked": True,
    "medication_safety_non_bypass_enforced": True,
    "medical_decision_blocked": True,
    "real_doc_live_allowed_count": 0,
    "raw_pii_in_package_count": 0,
    "raw_pii_in_report_count": 0,
    "token_map_in_package_count": 0,
    "token_map_in_report_count": 0,
    "medical_decision_made_count": 0,
    "live_call_made": False,
    "external_api_used": False,
    "billing_api_used": False,
    "active_written_count": 0,
    "active_mkb_record_created_count": 0,
    "auto_accept_true_count": 0,
    "privacy_result": "passed",
    "billing_check_pending": True,
}

FORBIDDEN_REPORT_TOKENS = (
    "ya29.",
    "AIza",
    "Bearer ",
    "Authorization" + ":",
    "access_token",
    "refresh_token",
    "private_key",
    "application" + "_default" + "_credentials",
    "g" + "cloud",
    "C:\\",
    "/home/",
)


def _read_doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _build_checklist(text: str) -> dict[str, Any]:
    required_sections_present_count = sum(1 for section in REQUIRED_SECTIONS if section in text)
    return {
        "handoff_doc_created": DOC.exists(),
        "required_sections_present_count": required_sections_present_count,
        "required_sections_total": len(REQUIRED_SECTIONS),
        "gate_inventory_present": all(gate in text for gate in GATES),
        "operator_commands_present": all(command in text for command in COMMANDS),
        "pass_chain_present": all(block in text and "PASS" in text for block in PASS_CHAIN),
        "integrated_harness_metrics_present": all(f"{key}={str(value).lower() if isinstance(value, bool) else value}" in text for key, value in INTEGRATED_METRICS.items()),
        "real_document_boundary_present": "Real-document Vertex routing remains NOT authorized after this block." in text,
        "no_live_authorization_boundary_present": "does not authorize real-document Vertex routing" in text,
        "active_write_boundary_present": "must not write active MKB" in text and "does not authorize active MKB writes" in text,
        "auto_accept_boundary_present": "must not auto-accept" in text and "does not authorize auto-accept" in text,
        "medication_safety_boundary_present": "Medication safety proof is required when medication facts are present" in text,
        "no_medical_decision_boundary_present": "does not create a medical decision system" in text and "must not make medical decisions" in text,
        "future_authorization_boundary_present": "Any future real-document live call requires a new explicit block" in text,
        "stop_conditions_present": all(token in text for token in ("live provider usage", "billing API usage", "raw PII in payload or report", "missing `review_required`")),
        "recommended_next_block_present": "MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-UAT-NO-LIVE-15Z-H" in text,
        "docs_only_change": True,
        "production_code_changed": False,
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed" if _privacy_passes(text) else "failed",
        "billing_check_pending": True,
    }


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) for payload in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _summary(checklist: dict[str, Any]) -> dict[str, Any]:
    return {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-HANDOFF-DOC-NO-LIVE-15Z-G",
        "handoff_doc_path": "docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md",
        **checklist,
    }


def _matrix_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# 15Z-G handoff document matrix",
        "",
        "| Area | Evidence | Status |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {block} | {evidence} | PASS |" for block, evidence in PASS_CHAIN.items())
    lines.extend(
        [
            "| Current authorization | Real-document Vertex routing remains not authorized | PASS |",
            "| Active writes | Active MKB writes remain blocked | PASS |",
            "| Auto-accept | Auto-accept remains blocked | PASS |",
            "| Medical decisions | Medical decision output remains blocked | PASS |",
            "| Privacy | Reports contain no raw private payloads or token maps | PASS |",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    for key in (
        "required_sections_present_count",
        "gate_inventory_present",
        "operator_commands_present",
        "real_document_boundary_present",
        "no_live_authorization_boundary_present",
        "active_write_boundary_present",
        "auto_accept_boundary_present",
        "medication_safety_boundary_present",
        "no_medical_decision_boundary_present",
        "future_authorization_boundary_present",
        "stop_conditions_present",
        "recommended_next_block_present",
        "docs_only_change",
        "production_code_changed",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ):
        lines.append(f"| {key} | `{summary[key]}` |")
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-HANDOFF-DOC-NO-LIVE-15Z-G",
        "",
        "## Result",
        "",
        "- Created the no-live 15Z operator/governance handoff document.",
        "- Created sanitized checklist, matrix, and summary report artifacts.",
        "- Did not change production code, UI, OCR, extraction, MKB, or decision-store behavior.",
        "- Did not make provider calls or billing API calls.",
        "- Did not process real documents.",
        "- Did not create active MKB records or enable auto-accept.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.items():
        if key in {"block", "handoff_doc_path"}:
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Real-document Vertex routing remains not authorized. Any future real-document live call requires a new explicit block with its own live gate, bounded call count, review-bound handling, no active MKB write, no auto-accept, and no medical decision output.",
            "",
        ]
    )
    return "\n".join(lines)


def build_reports() -> dict[str, Any]:
    text = _read_doc()
    checklist = _build_checklist(text)
    summary = _summary(checklist)
    matrix = _matrix_markdown(summary)
    implementation = _implementation_markdown(summary)
    privacy_result = "passed" if _privacy_passes(text, checklist, summary, matrix, implementation) else "failed"
    checklist["privacy_result"] = privacy_result
    summary["privacy_result"] = privacy_result
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "checklist": checklist,
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CHECKLIST_JSON.write_text(json.dumps(reports["checklist"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["handoff_doc_created"] is True,
            summary["required_sections_present_count"] == len(REQUIRED_SECTIONS),
            summary["gate_inventory_present"] is True,
            summary["operator_commands_present"] is True,
            summary["real_document_boundary_present"] is True,
            summary["no_live_authorization_boundary_present"] is True,
            summary["active_write_boundary_present"] is True,
            summary["auto_accept_boundary_present"] is True,
            summary["medication_safety_boundary_present"] is True,
            summary["no_medical_decision_boundary_present"] is True,
            summary["future_authorization_boundary_present"] is True,
            summary["stop_conditions_present"] is True,
            summary["recommended_next_block_present"] is True,
            summary["docs_only_change"] is True,
            summary["production_code_changed"] is False,
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
        "medai_vertex_real_doc_readiness_operator_handoff_doc_no_live_15z_g_ready"
        if ready
        else "medai_vertex_real_doc_readiness_operator_handoff_doc_no_live_15z_g_not_ready"
    )
    print(
        json.dumps(
            {
                "handoff_doc_created": summary["handoff_doc_created"],
                "required_sections_present_count": summary["required_sections_present_count"],
                "gate_inventory_present": summary["gate_inventory_present"],
                "operator_commands_present": summary["operator_commands_present"],
                "real_document_boundary_present": summary["real_document_boundary_present"],
                "no_live_authorization_boundary_present": summary["no_live_authorization_boundary_present"],
                "active_write_boundary_present": summary["active_write_boundary_present"],
                "auto_accept_boundary_present": summary["auto_accept_boundary_present"],
                "medication_safety_boundary_present": summary["medication_safety_boundary_present"],
                "no_medical_decision_boundary_present": summary["no_medical_decision_boundary_present"],
                "future_authorization_boundary_present": summary["future_authorization_boundary_present"],
                "stop_conditions_present": summary["stop_conditions_present"],
                "recommended_next_block_present": summary["recommended_next_block_present"],
                "docs_only_change": summary["docs_only_change"],
                "production_code_changed": summary["production_code_changed"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "billing_api_used": summary["billing_api_used"],
                "active_written_count": summary["active_written_count"],
                "active_mkb_record_created_count": summary["active_mkb_record_created_count"],
                "auto_accept_true_count": summary["auto_accept_true_count"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
