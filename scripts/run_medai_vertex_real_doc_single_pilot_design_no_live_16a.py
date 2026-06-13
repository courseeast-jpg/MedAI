#!/usr/bin/env python3
"""Generate the 16A no-live single pilot design package."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16A"
DESIGN_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_DESIGN_16A.md"
AUTH_TEMPLATE_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_AUTHORIZATION_TEMPLATE_16A.md"
GO_NO_GO_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_GO_NO_GO_CHECKLIST_16A.md"
TEST_PLAN_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_TEST_PLAN_16A.md"
NEXT_DECISION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_NEXT_DECISION_16A.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_design_no_live_16a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CHECKLIST_JSON = REPORT_DIR / "design_checklist.json"
DESIGN_MATRIX_MD = REPORT_DIR / "design_matrix.md"
GO_NO_GO_MATRIX_MD = REPORT_DIR / "go_no_go_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FUTURE_GATE_PLACEHOLDER = "MEDAI_FUTURE_SINGLE_DOC_VERTEX_LIVE_GATE_PLACEHOLDER"

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


def _docs() -> dict[str, str]:
    design = f"""# MEDAI Vertex Real Document Single Pilot Design 16A

## Purpose

This is a no-live design package for a future single redacted-real-like or real-document Vertex pilot. It defines the criteria for a future block but does not execute that pilot.

## Pilot Scope

- Explicit no-live status: this block is design-only.
- Real-document live execution is not authorized.
- Required future live gate name placeholder: `{FUTURE_GATE_PLACEHOLDER}`.
- One-document maximum.
- One-call limit.
- Redacted/tokenized payload only.
- No raw PII.
- No token map outbound; the token map remains local only.
- No active MKB write.
- No auto-accept.
- Review-required for every output.
- No medical decision output.
- Medication safety non-bypass is required if medication facts appear.
- Stop-on-first-failure rule.
- Bounded cost/token ceiling: future design must specify a hard token ceiling and cost cap before any call.
- Evidence anchoring requirements: every extracted fact must cite a source span from the redacted/tokenized payload.
- Declared label alias policy: only declared aliases may map source labels to normalized labels.
- Request-shape requirements: request body contains only `contents` and `generationConfig`; forbidden provider/MedAI metadata remains local.
- Report sanitization requirements: no raw private values, raw OCR/private payloads, raw PDF/image payloads, token maps, credentials, auth headers, or local secret paths.
- Operator approval requirements: named operator, timestamp, provenance declaration, billing/cost acknowledgement, and review-bound acknowledgement.

## Refusal Conditions

Refuse the future pilot if any required approval is missing, provenance is unknown, raw PII remains, token map would leave local custody, request shape is invalid, cost cap is missing, medication safety proof is missing when required, active write is requested, auto-accept is requested, medical advice is requested, or a previous failure has occurred.
"""
    auth = """# MEDAI Vertex Real Document Single Pilot Authorization Template 16A

This template does not itself authorize live execution.

- Operator identity placeholder: `[OPERATOR_ID]`
- Date/time placeholder: `[AUTHORIZED_AT]`
- Selected document provenance declaration: `[PROVENANCE_DECLARATION]`
- Confirmation document is permitted for testing: `[CONFIRM_PERMITTED_FOR_TESTING]`
- Confirmation PII stripping proof passed: `[CONFIRM_PII_STRIPPING_PASS]`
- Confirmation vault isolation passed: `[CONFIRM_VAULT_ISOLATION_PASS]`
- Confirmation no raw PII in outbound payload: `[CONFIRM_NO_RAW_PII_OUTBOUND]`
- Confirmation token map remains local: `[CONFIRM_TOKEN_MAP_LOCAL_ONLY]`
- Confirmation billing/cost cap accepted: `[CONFIRM_COST_CAP_ACCEPTED]`
- Confirmation no active write: `[CONFIRM_NO_ACTIVE_WRITE]`
- Confirmation no auto-accept: `[CONFIRM_NO_AUTO_ACCEPT]`
- Confirmation no medical decision output: `[CONFIRM_NO_MEDICAL_DECISION_OUTPUT]`
- Confirmation review-bound only: `[CONFIRM_REVIEW_BOUND_ONLY]`
- Confirmation stop-on-first-failure: `[CONFIRM_STOP_ON_FIRST_FAILURE]`
"""
    go_no_go = f"""# MEDAI Vertex Real Document Single Pilot Go/No-Go Checklist 16A

## GO Requirements

- Human authorization required.
- Cost cap acknowledgement required.
- Dedicated future live gate required: `{FUTURE_GATE_PLACEHOLDER}`.
- One-call limit required.
- One-document limit required.
- Redacted/tokenized only.
- No raw reports.
- No active writes.
- No auto-accept.
- No medical decision logic.
- Medication safety proof required if medication facts appear.
- Review-required output handling.

## NO-GO Blockers

- Missing human authorization.
- Missing billing/cost acknowledgement.
- Missing dedicated future live gate.
- More than one document.
- More than one call.
- Raw PII or raw private content remains.
- Token map outbound would occur.
- Active write requested.
- Auto-accept requested.
- Medical decision logic requested.
- Medication safety proof missing when medication facts appear.
- Request shape contains forbidden metadata.
- Report sanitization fails.
"""
    test_plan = """# MEDAI Vertex Real Document Single Pilot Test Plan 16A

## Pre-Live No-Live Validations

- Run the 15Z release snapshot validation.
- Run PII stripping proof.
- Run vault isolation proof.
- Run adapter dry-run request shape validation.
- Run authorization and cost refusal checks.
- Run medication safety non-bypass checks.
- Run integrated readiness harness.
- Run operator UAT.

## Future Live Pilot Execution Outline

This document intentionally does not include commands that set a live gate. A separate future block must define any live execution procedure.

## Expected Pass Criteria

- Exactly one document selected.
- Exactly one call planned.
- Redacted/tokenized payload only.
- Evidence anchors present for every candidate fact.
- Declared label aliases only.
- Review-required output.
- No active write, no auto-accept, and no medical decision output.

## Expected Fail Criteria

- Any raw private value remains.
- Token map would be sent outbound.
- Request shape includes forbidden metadata.
- Provider or billing error occurs.
- Evidence anchoring fails.
- Medication safety proof is missing when required.
- Any active write, auto-accept, or medical decision output is requested.

## Stop Conditions

Stop on first failure, including privacy, request-shape, cost, provider, billing, evidence-anchor, medication-safety, review-boundary, active-write, auto-accept, or medical-decision failures.

## Rollback / Cleanup Expectations

No active MKB records or production review queue mutations should exist. Generated future reports must be sanitized and review-bound.

## Report Requirements

Reports must include no-live validation provenance, one-call and one-document proof, sanitized request fingerprint, evidence-anchor status, review-required status, cost cap status, and privacy result.
"""
    next_decision = """# MEDAI Vertex Real Document Single Pilot Next Decision 16A

## Option A: Remain No-Live

Run more synthetic/redacted stress and failure-injection coverage.

## Option B: Prepare Future Explicit Single-Document Live Pilot Block

Prepare a separate future block with its own approval, future live gate, bounded one-call execution plan, and stop-on-first-failure handling. This 16A package does not run it.

## Option C: Pause/Freeze

Pause and keep the 15Z/16A governance state frozen.

## Recommended Next

Recommended next: user/operator decision, not automatic live execution.
"""
    return {
        "design": design,
        "authorization": auth,
        "go_no_go": go_no_go,
        "test_plan": test_plan,
        "next_decision": next_decision,
    }


def write_design_docs(docs: dict[str, str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_MD.write_text(docs["design"], encoding="utf-8")
    AUTH_TEMPLATE_MD.write_text(docs["authorization"], encoding="utf-8")
    GO_NO_GO_MD.write_text(docs["go_no_go"], encoding="utf-8")
    TEST_PLAN_MD.write_text(docs["test_plan"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) for payload in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _checklist(docs: dict[str, str]) -> dict[str, Any]:
    all_docs = "\n".join(docs.values())
    return {
        "pilot_design_created": DESIGN_MD.exists(),
        "authorization_template_created": AUTH_TEMPLATE_MD.exists(),
        "go_no_go_checklist_created": GO_NO_GO_MD.exists(),
        "test_plan_created": TEST_PLAN_MD.exists(),
        "next_decision_memo_created": NEXT_DECISION_MD.exists(),
        "no_live_status_present": "Explicit no-live status" in all_docs and "design-only" in all_docs,
        "one_document_limit_present": "One-document maximum" in all_docs and "One-document limit required" in all_docs,
        "one_call_limit_present": "One-call limit" in all_docs and "One-call limit required" in all_docs,
        "redacted_tokenized_only_present": "Redacted/tokenized payload only" in all_docs,
        "no_raw_pii_boundary_present": "No raw PII" in all_docs,
        "no_token_map_outbound_boundary_present": "No token map outbound" in all_docs and "token map remains local" in all_docs,
        "request_shape_boundary_present": "contents` and `generationConfig" in all_docs and "forbidden metadata" in all_docs,
        "evidence_anchor_boundary_present": "Evidence anchoring requirements" in all_docs and "source span" in all_docs,
        "label_alias_boundary_present": "Declared label alias policy" in all_docs,
        "medication_safety_boundary_present": "Medication safety non-bypass" in all_docs,
        "no_active_write_boundary_present": "No active MKB write" in all_docs and "No active writes" in all_docs,
        "no_auto_accept_boundary_present": "No auto-accept" in all_docs,
        "no_medical_decision_boundary_present": "No medical decision output" in all_docs and "No medical decision logic" in all_docs,
        "review_required_boundary_present": "Review-required" in all_docs or "review-required" in all_docs,
        "future_live_gate_required": FUTURE_GATE_PLACEHOLDER in all_docs and "Dedicated future live gate required" in all_docs,
        "cost_cap_boundary_present": "cost cap" in all_docs.lower() and "token ceiling" in all_docs,
        "stop_on_first_failure_present": "Stop-on-first-failure" in all_docs or "stop-on-first-failure" in all_docs,
        "operator_authorization_required": "Operator approval requirements" in all_docs and "Operator identity placeholder" in all_docs,
        "refusal_conditions_present": "Refusal Conditions" in all_docs and "NO-GO Blockers" in all_docs,
        "next_decision_required": "user/operator decision, not automatic live execution" in all_docs,
        "real_doc_live_allowed_count": 0,
    }


def _summary(checklist: dict[str, Any], privacy_result: str) -> dict[str, Any]:
    return {
        "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-DESIGN-NO-LIVE-16A",
        **checklist,
        "required_design_checks_passed": sum(1 for key, value in checklist.items() if key != "real_doc_live_allowed_count" and value is True),
        "required_design_checks_total": len([k for k in checklist if k != "real_doc_live_allowed_count"]),
        "docs_only_change": True,
        "production_code_changed": False,
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "medical_decision_made_count": 0,
        "privacy_result": privacy_result,
        "billing_check_pending": True,
    }


def _design_matrix(summary: dict[str, Any]) -> str:
    lines = ["# 16A design matrix", "", "| Check | Value |", "| --- | --- |"]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _go_no_go_matrix(checklist: dict[str, Any]) -> str:
    rows = [
        ("human_authorization_required", checklist["operator_authorization_required"]),
        ("cost_cap_acknowledgement_required", checklist["cost_cap_boundary_present"]),
        ("dedicated_future_live_gate_required", checklist["future_live_gate_required"]),
        ("one_call_limit_required", checklist["one_call_limit_present"]),
        ("one_document_limit_required", checklist["one_document_limit_present"]),
        ("redacted_tokenized_only", checklist["redacted_tokenized_only_present"]),
        ("no_raw_reports", checklist["no_raw_pii_boundary_present"]),
        ("no_active_writes", checklist["no_active_write_boundary_present"]),
        ("no_auto_accept", checklist["no_auto_accept_boundary_present"]),
        ("no_medical_decision_logic", checklist["no_medical_decision_boundary_present"]),
        ("medication_safety_proof_if_needed", checklist["medication_safety_boundary_present"]),
    ]
    lines = ["# 16A go/no-go matrix", "", "| Requirement | Status |", "| --- | --- |"]
    lines.extend(f"| {name} | `{status}` |" for name, status in rows)
    lines.append("")
    return "\n".join(lines)


def _implementation(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-DESIGN-NO-LIVE-16A",
        "",
        "## Result",
        "",
        "- Created a no-live design package for a future single-document Vertex pilot.",
        "- Did not call any provider or billing API.",
        "- Did not process real documents, set live gates, write active MKB, mutate production review queue, auto-accept, or produce medical decision output.",
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
    docs = _docs()
    write_design_docs(docs)
    checklist = _checklist(docs)
    privacy_result = "passed" if _privacy_passes(docs, checklist) else "failed"
    summary = _summary(checklist, privacy_result)
    design_matrix = _design_matrix(summary)
    go_no_go_matrix = _go_no_go_matrix(checklist)
    implementation = _implementation(summary)
    privacy_result = "passed" if _privacy_passes(docs, checklist, summary, design_matrix, go_no_go_matrix, implementation) else "failed"
    summary["privacy_result"] = privacy_result
    design_matrix = _design_matrix(summary)
    implementation = _implementation(summary)
    return {
        "summary": summary,
        "checklist": checklist,
        "design_matrix": design_matrix,
        "go_no_go_matrix": go_no_go_matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CHECKLIST_JSON.write_text(json.dumps(reports["checklist"], indent=2), encoding="utf-8")
    DESIGN_MATRIX_MD.write_text(reports["design_matrix"], encoding="utf-8")
    GO_NO_GO_MATRIX_MD.write_text(reports["go_no_go_matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["required_design_checks_passed"] == summary["required_design_checks_total"],
            summary["real_doc_live_allowed_count"] == 0,
            summary["docs_only_change"] is True,
            summary["production_code_changed"] is False,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["medical_decision_made_count"] == 0,
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
        "medai_vertex_real_doc_single_pilot_design_no_live_16a_ready"
        if ready
        else "medai_vertex_real_doc_single_pilot_design_no_live_16a_not_ready"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "block"}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
