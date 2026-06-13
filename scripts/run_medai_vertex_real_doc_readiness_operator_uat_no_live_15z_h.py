#!/usr/bin/env python3
"""Generate the 15Z-H no-live operator UAT reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

HANDOFF_DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h"
SUMMARY_JSON = REPORT_DIR / "summary.json"
STEPS_JSON = REPORT_DIR / "operator_uat_steps.json"
MATRIX_MD = REPORT_DIR / "operator_uat_matrix.md"
TRANSCRIPT_MD = REPORT_DIR / "operator_uat_transcript.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

COMMANDS = [
    "python scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py",
    "python scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py",
    "python scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py",
    "python scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py",
    "python scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py",
    "python scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py",
]

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

EXPECTED_METRICS = {
    "real_doc_live_allowed_count": 0,
    "live_call_made": False,
    "external_api_used": False,
    "billing_api_used": False,
    "active_written_count": 0,
    "active_mkb_record_created_count": 0,
    "auto_accept_true_count": 0,
    "medical_decision_made_count": 0,
    "privacy_result": "passed",
    "billing_check_pending": True,
}

STOP_CONDITIONS = [
    "live provider usage",
    "billing API usage",
    "real-document live authorization",
    "active MKB write",
    "production review queue mutation",
    "`auto_accept=true`",
    "raw PII in payload or report",
    "token map leakage",
    "unknown provenance allowed",
    "real/private marker allowed",
    "forbidden request metadata accepted",
    "invalid `generationConfig` accepted",
    "medication decision output",
    "medical advice",
    "missing `review_required`",
]

FUTURE_BOUNDARY = [
    "Any future real-document live call requires a new explicit block",
    "its own live gate",
    "redacted/tokenized payloads",
    "bounded call count",
    "stop on first failure",
    "review-bound",
    "must not write active MKB",
    "must not auto-accept",
    "must not make medical decisions",
]

FORBIDDEN_REPORT_TOKENS = (
    "ya29.",
    "AIza",
    "Bearer ",
    "Authorization" + ":",
    "access" + "_token",
    "refresh" + "_token",
    "private" + "_key",
    "application" + "_default" + "_credentials",
    "g" + "cloud",
    "C:\\",
    "/home/",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_summary(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _base_step(step_id: int, step_name: str, source: str, expected: str, observed: str, status: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "step_name": step_name,
        "source_artifact_checked": source,
        "expected_result": expected,
        "observed_result": observed,
        "status": status,
        "operator_action_required": "review_local_artifact_only",
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_write_allowed": False,
        "auto_accept_allowed": False,
        "real_doc_live_allowed": False,
        "review_required": True,
    }


def _pass(status: bool) -> str:
    return "pass" if status else "fail"


def _build_steps() -> list[dict[str, Any]]:
    handoff = _read_text(HANDOFF_DOC)
    summary_f = _read_summary("reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json")
    summary_g = _read_summary("reports/medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g/summary.json")
    doc_source = "docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"
    f_source = "reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json"
    g_source = "reports/medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g/summary.json"
    steps = [
        _base_step(1, "operator_can_find_handoff_doc", doc_source, "handoff doc exists", f"exists={HANDOFF_DOC.exists()}", _pass(HANDOFF_DOC.exists())),
        _base_step(2, "operator_can_identify_real_doc_boundary", doc_source, "real-document route remains blocked", "boundary text present", _pass("Real-document Vertex routing remains NOT authorized after this block." in handoff)),
        _base_step(3, "operator_can_identify_gate_inventory", doc_source, "all readiness gates are listed", f"gates_found={sum(g in handoff for g in GATES)}", _pass(all(g in handoff for g in GATES))),
    ]
    for index, command in enumerate(COMMANDS, start=4):
        block = ["15z_a", "15z_b", "15z_c", "15z_d", "15z_e", "15z_f"][index - 4]
        steps.append(
            _base_step(
                index,
                f"operator_can_run_{block}_command_no_live",
                doc_source,
                "command is listed for local no-live validation",
                command,
                _pass(command in handoff),
            )
        )
    metric_pass = all(summary_f.get(key) == value for key, value in EXPECTED_METRICS.items())
    steps.extend(
        [
            _base_step(10, "operator_can_verify_integrated_harness_pass", f_source, "integrated harness pass metrics present", f"cases={summary_f.get('integrated_cases_passed')}/{summary_f.get('integrated_cases_total')}", _pass(summary_f.get("integrated_cases_total") == 17 and summary_f.get("integrated_cases_passed") == 17 and metric_pass)),
            _base_step(11, "operator_can_verify_future_review_package_only", f_source, "future packages are review-only", f"future_review_only_count={summary_f.get('future_operator_review_only_count')}", _pass(summary_f.get("future_operator_review_only_count") == 3 and summary_f.get("future_operator_review_package_created_count") == 3)),
            _base_step(12, "operator_can_verify_blocked_failure_injections", f_source, "unsafe cases are blocked", f"blocked_case_count={summary_f.get('blocked_case_count')}", _pass(summary_f.get("blocked_case_count") == 14 and summary_f.get("refusal_records_created_count") == 14)),
            _base_step(13, "operator_can_verify_no_provider_call", f_source, "no provider call", f"live_call_made={summary_f.get('live_call_made')}", _pass(summary_f.get("live_call_made") is False and summary_f.get("external_api_used") is False)),
            _base_step(14, "operator_can_verify_no_billing_api", f_source, "no billing API", f"billing_api_used={summary_f.get('billing_api_used')}", _pass(summary_f.get("billing_api_used") is False)),
            _base_step(15, "operator_can_verify_no_active_write", f_source, "no active write", f"active_written_count={summary_f.get('active_written_count')}", _pass(summary_f.get("active_written_count") == 0 and summary_f.get("active_mkb_record_created_count") == 0)),
            _base_step(16, "operator_can_verify_no_auto_accept", f_source, "no auto-accept", f"auto_accept_true_count={summary_f.get('auto_accept_true_count')}", _pass(summary_f.get("auto_accept_true_count") == 0)),
            _base_step(17, "operator_can_verify_no_medical_decision", f_source, "no medical decision", f"medical_decision_made_count={summary_f.get('medical_decision_made_count')}", _pass(summary_f.get("medical_decision_made_count") == 0 and summary_f.get("medical_decision_blocked") is True)),
            _base_step(18, "operator_can_verify_no_raw_pii_or_token_map_reports", f_source, "no raw private values or token maps in reports", f"raw_pii={summary_f.get('raw_pii_in_report_count')}; token_map={summary_f.get('token_map_in_report_count')}", _pass(summary_f.get("raw_pii_in_report_count") == 0 and summary_f.get("token_map_in_report_count") == 0 and summary_f.get("privacy_result") == "passed")),
            _base_step(19, "operator_can_identify_stop_conditions", doc_source, "stop conditions are visible", f"stop_conditions_found={sum(c in handoff for c in STOP_CONDITIONS)}", _pass(all(c in handoff for c in STOP_CONDITIONS))),
            _base_step(20, "operator_can_identify_next_block_boundary", g_source, "future boundary and next block are visible", "future block boundary present; 15Z-H referenced", _pass(all(token in handoff for token in FUTURE_BOUNDARY) and summary_g.get("recommended_next_block_present") is True)),
        ]
    )
    return steps


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) for payload in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for step in steps if step["status"] == "pass")
    summary_f = _read_summary("reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json")
    summary = {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-UAT-NO-LIVE-15Z-H",
        "operator_uat_created": True,
        "operator_uat_steps_total": len(steps),
        "operator_uat_steps_passed": passed,
        "handoff_doc_found": any(step["step_name"] == "operator_can_find_handoff_doc" and step["status"] == "pass" for step in steps),
        "gate_inventory_verified": any(step["step_name"] == "operator_can_identify_gate_inventory" and step["status"] == "pass" for step in steps),
        "operator_commands_verified": all(step["status"] == "pass" for step in steps if "_command_no_live" in step["step_name"]),
        "integrated_harness_verified": any(step["step_name"] == "operator_can_verify_integrated_harness_pass" and step["status"] == "pass" for step in steps),
        "future_review_package_only_verified": any(step["step_name"] == "operator_can_verify_future_review_package_only" and step["status"] == "pass" for step in steps),
        "blocked_failure_injections_verified": any(step["step_name"] == "operator_can_verify_blocked_failure_injections" and step["status"] == "pass" for step in steps),
        "stop_conditions_verified": any(step["step_name"] == "operator_can_identify_stop_conditions" and step["status"] == "pass" for step in steps),
        "future_authorization_boundary_verified": any(step["step_name"] == "operator_can_identify_next_block_boundary" and step["status"] == "pass" for step in steps),
        "real_doc_live_allowed_count": summary_f.get("real_doc_live_allowed_count"),
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": summary_f.get("active_written_count"),
        "active_mkb_record_created_count": summary_f.get("active_mkb_record_created_count"),
        "auto_accept_true_count": summary_f.get("auto_accept_true_count"),
        "medical_decision_made_count": summary_f.get("medical_decision_made_count"),
        "raw_pii_in_report_count": summary_f.get("raw_pii_in_report_count"),
        "token_map_in_report_count": summary_f.get("token_map_in_report_count"),
        "privacy_result": "pending",
        "billing_check_pending": True,
    }
    return summary


def _matrix_markdown(steps: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# 15Z-H operator UAT matrix",
        "",
        "| Step | UAT case | Source | Expected | Observed | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for step in steps:
        lines.append(
            f"| {step['step_id']} | {step['step_name']} | {step['source_artifact_checked']} | {step['expected_result']} | {step['observed_result']} | {step['status']} |"
        )
    lines.extend(["", "| Metric | Value |", "| --- | --- |"])
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _transcript_markdown(steps: list[dict[str, Any]]) -> str:
    lines = [
        "# 15Z-H operator UAT transcript",
        "",
        "This no-live transcript records local artifact checks only. No provider call, billing API call, real-document processing, active write, auto-accept, or medical decision output occurred.",
        "",
    ]
    for step in steps:
        lines.extend(
            [
                f"## Step {step['step_id']}: {step['step_name']}",
                "",
                f"- Source artifact checked: `{step['source_artifact_checked']}`",
                f"- Expected result: {step['expected_result']}",
                f"- Observed result: {step['observed_result']}",
                f"- Status: `{step['status']}`",
                f"- Operator action required: `{step['operator_action_required']}`",
                "- Safety: no live call, no external API, no billing API, no active write, no auto-accept, no real-doc live authorization, review required.",
                "",
            ]
        )
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-UAT-NO-LIVE-15Z-H",
        "",
        "## Result",
        "",
        "- Generated a no-live operator UAT over the 15Z handoff and integrated harness.",
        "- Verified the handoff doc, gate inventory, local no-live commands, integrated status, review-only packages, blocked unsafe cases, stop conditions, and future authorization boundary.",
        "- Did not change production code, UI, OCR, extraction, MKB, or decision-store behavior.",
        "- Did not call any provider or billing API.",
        "- Did not process real documents, write active MKB, auto-accept, or produce medical decision output.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_reports() -> dict[str, Any]:
    steps = _build_steps()
    summary = _summary(steps)
    matrix = _matrix_markdown(steps, summary)
    transcript = _transcript_markdown(steps)
    implementation = _implementation_markdown(summary)
    privacy_result = "passed" if _privacy_passes(steps, summary, matrix, transcript, implementation) else "failed"
    summary["privacy_result"] = privacy_result
    matrix = _matrix_markdown(steps, summary)
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "steps": {"operator_uat_steps": steps},
        "matrix": matrix,
        "transcript": transcript,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    STEPS_JSON.write_text(json.dumps(reports["steps"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    TRANSCRIPT_MD.write_text(reports["transcript"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["operator_uat_created"] is True,
            summary["operator_uat_steps_total"] == 20,
            summary["operator_uat_steps_passed"] == 20,
            summary["handoff_doc_found"] is True,
            summary["gate_inventory_verified"] is True,
            summary["operator_commands_verified"] is True,
            summary["integrated_harness_verified"] is True,
            summary["future_review_package_only_verified"] is True,
            summary["blocked_failure_injections_verified"] is True,
            summary["stop_conditions_verified"] is True,
            summary["future_authorization_boundary_verified"] is True,
            summary["real_doc_live_allowed_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["medical_decision_made_count"] == 0,
            summary["raw_pii_in_report_count"] == 0,
            summary["token_map_in_report_count"] == 0,
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
        "medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h_ready"
        if ready
        else "medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h_not_ready"
    )
    print(
        json.dumps(
            {
                "operator_uat_created": summary["operator_uat_created"],
                "operator_uat_steps_total": summary["operator_uat_steps_total"],
                "operator_uat_steps_passed": summary["operator_uat_steps_passed"],
                "handoff_doc_found": summary["handoff_doc_found"],
                "gate_inventory_verified": summary["gate_inventory_verified"],
                "operator_commands_verified": summary["operator_commands_verified"],
                "integrated_harness_verified": summary["integrated_harness_verified"],
                "future_review_package_only_verified": summary["future_review_package_only_verified"],
                "blocked_failure_injections_verified": summary["blocked_failure_injections_verified"],
                "stop_conditions_verified": summary["stop_conditions_verified"],
                "future_authorization_boundary_verified": summary["future_authorization_boundary_verified"],
                "real_doc_live_allowed_count": summary["real_doc_live_allowed_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "billing_api_used": summary["billing_api_used"],
                "active_written_count": summary["active_written_count"],
                "active_mkb_record_created_count": summary["active_mkb_record_created_count"],
                "auto_accept_true_count": summary["auto_accept_true_count"],
                "medical_decision_made_count": summary["medical_decision_made_count"],
                "raw_pii_in_report_count": summary["raw_pii_in_report_count"],
                "token_map_in_report_count": summary["token_map_in_report_count"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
