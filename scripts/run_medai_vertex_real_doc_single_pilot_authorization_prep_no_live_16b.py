#!/usr/bin/env python3
"""Generate the 16B no-live single-pilot authorization-prep package.

16B prepares the *future* single-real-document Vertex pilot's governance: its own
approval record, the dedicated future live gate specification, a bounded one-call
execution plan, and stop-on-first-failure handling. It is design/prep-only and runs
nothing: no provider call, no Vertex/Gemini/Claude/OpenAI live execution, no billing
API call, no real/private document processing, no corpus read, no active MKB write,
no auto-accept, no medical decision, no production queue mutation.

The dedicated future live gate is *named* but explicitly NOT set here; the validator
also confirms the gate is absent from the current environment.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16B"
DESIGN_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_AUTHORIZATION_PREP_16B.md"
APPROVAL_RECORD_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_APPROVAL_RECORD_16B.md"
FUTURE_GATE_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_FUTURE_LIVE_GATE_16B.md"
ONE_CALL_PLAN_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_BOUNDED_ONE_CALL_PLAN_16B.md"
STOP_FIRST_FAILURE_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_STOP_ON_FIRST_FAILURE_16B.md"
NEXT_DECISION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_NEXT_DECISION_16B.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_authorization_prep_no_live_16b"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CHECKLIST_JSON = REPORT_DIR / "authorization_prep_checklist.json"
DESIGN_MATRIX_MD = REPORT_DIR / "authorization_prep_matrix.md"
APPROVAL_MATRIX_MD = REPORT_DIR / "approval_gate_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

# The dedicated future live gate is NAMED but never set in this block.
FUTURE_LIVE_GATE_NAME = "MEDAI_FUTURE_SINGLE_DOC_VERTEX_LIVE_PILOT_GATE_PLACEHOLDER"

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
    design = f"""# MEDAI Vertex Real Document Single Pilot Authorization Prep 16B

## Purpose

This is a no-live authorization-prep package for a future single-document Vertex
pilot. It prepares the pilot's own approval, the dedicated future live gate, a
bounded one-call execution plan, and stop-on-first-failure handling. It does not run
the pilot.

## Status

- Explicit no-live status: this block is design-only.
- Real-document live execution is not authorized.
- This block builds on the frozen 15Z/16A governance state and does not modify it.
- The GCP synthetic sandbox remains separate environment evidence only and is not
  treated as MedAI real-doc readiness validation.

## Own Approval Requirement

This future pilot requires its own approval, separate from 16A. Human authorization required before any future live pilot. A named operator must complete the 16B approval record. There is no auto-accept of approval and no default authorization.

## Dedicated Future Live Gate

Required future live gate name placeholder: `{FUTURE_LIVE_GATE_NAME}`. This block does
not set that gate and contains no command that sets a live gate. The gate must be
defined, separately authorized, and explicitly set only in a later, separately
approved execution block.

## Bounded One-Call Execution Plan

- One-document maximum.
- One-call limit.
- Redacted/tokenized payload only.
- No raw PII.
- No token map outbound; the token map remains local only.
- Bounded token ceiling and a hard cost cap are required before any call.
- Request shape: the request body contains only `contents` and `generationConfig`;
  forbidden provider/MedAI metadata remains local.
- Review-required for every output; no active MKB write; no auto-accept; no medical
  decision output.
- Medication safety non-bypass is required if medication facts appear.

## Stop-On-First-Failure Handling

Stop-on-first-failure rule applies. On the first privacy, request-shape, cost,
provider, billing, evidence-anchor, medication-safety, review-boundary, active-write,
auto-accept, or medical-decision failure, halt immediately, preserve sanitized
evidence, and require fresh re-authorization.

## Refusal Conditions

Refuse the future pilot if any required approval is missing, provenance is unknown,
raw PII remains, token map would leave local custody, request shape is invalid, cost
cap is missing, medication safety proof is missing when required, active write is
requested, auto-accept is requested, medical advice is requested, or a previous
failure has occurred.
"""
    approval = f"""# MEDAI Vertex Real Document Single Pilot Approval Record 16B

This record does not itself authorize live execution. Completing it prepares, but does
not grant, a future single-document live pilot. Human authorization required.

- Operator identity placeholder: `[OPERATOR_ID]`
- Approver identity placeholder: `[APPROVER_ID]`
- Date/time placeholder: `[AUTHORIZED_AT]`
- Selected single document provenance declaration: `[PROVENANCE_DECLARATION]`
- Confirmation document is permitted for testing: `[CONFIRM_PERMITTED_FOR_TESTING]`
- Confirmation PII stripping proof passed: `[CONFIRM_PII_STRIPPING_PASS]`
- Confirmation vault isolation passed: `[CONFIRM_VAULT_ISOLATION_PASS]`
- Confirmation no raw PII in outbound payload: `[CONFIRM_NO_RAW_PII_OUTBOUND]`
- Confirmation token map remains local: `[CONFIRM_TOKEN_MAP_LOCAL_ONLY]`
- Confirmation bounded token ceiling accepted: `[CONFIRM_TOKEN_CEILING_ACCEPTED]`
- Confirmation billing/cost cap accepted: `[CONFIRM_COST_CAP_ACCEPTED]`
- Confirmation dedicated future live gate name acknowledged:
  `[CONFIRM_FUTURE_LIVE_GATE_ACK]` (gate: `{FUTURE_LIVE_GATE_NAME}`)
- Confirmation one-document limit: `[CONFIRM_ONE_DOCUMENT]`
- Confirmation one-call limit: `[CONFIRM_ONE_CALL]`
- Confirmation no active write: `[CONFIRM_NO_ACTIVE_WRITE]`
- Confirmation no auto-accept: `[CONFIRM_NO_AUTO_ACCEPT]`
- Confirmation no medical decision output: `[CONFIRM_NO_MEDICAL_DECISION_OUTPUT]`
- Confirmation review-bound only: `[CONFIRM_REVIEW_BOUND_ONLY]`
- Confirmation stop-on-first-failure: `[CONFIRM_STOP_ON_FIRST_FAILURE]`
- Explicit approval decision: `[APPROVED / NOT_APPROVED]`
"""
    future_gate = f"""# MEDAI Vertex Real Document Single Pilot Future Live Gate 16B

## Gate Name

Dedicated future live gate required: `{FUTURE_LIVE_GATE_NAME}`.

## Not Set In This Block

This block does not set the live gate. This document intentionally contains no command
that sets a live gate. Setting the gate is reserved for a separate, explicitly approved
future execution block.

## Preconditions Before The Gate May Ever Be Set

All of the following must be satisfied first:

1. Completed 16B approval record with explicit human authorization.
2. Passed redaction/tokenization preflight (no raw PII; token map remains local).
3. Bounded token ceiling and hard cost cap declared and acknowledged.
4. Valid request shape (`contents` and `generationConfig` only).
5. One-document and one-call limits confirmed.
6. Stop-on-first-failure handling acknowledged.
7. Confirmation that the GCP synthetic sandbox is separate evidence only.

## Hard Boundaries

Setting the gate authorizes no active MKB write, no auto-accept, no medical decision,
and no production queue mutation; any future output is review-required only.
"""
    one_call_plan = """# MEDAI Vertex Real Document Single Pilot Bounded One-Call Plan 16B

## Bounded Execution Outline (Future, Not Run Here)

- Exactly one document selected.
- Exactly one call planned (one-call limit).
- Redacted/tokenized payload only; no raw PII; token map remains local only.
- Request shape contains only `contents` and `generationConfig`; no forbidden metadata.
- Bounded token ceiling and hard cost cap enforced before the call.
- Evidence anchoring: every candidate fact must cite a source span from the
  redacted/tokenized payload.
- Declared label alias policy: only declared aliases may map source labels.
- Review-required output; no active MKB write; no auto-accept; no medical decision.

## Explicitly Not Included

This document intentionally does not include any command that sets a live gate or makes
a provider call. A separate future block must define any live execution procedure.

## Expected Pass Criteria

- One document, one call, redacted/tokenized payload only, evidence anchors present,
  declared label aliases only, review-required output, no active write, no auto-accept,
  and no medical decision output.

## Expected Fail Criteria

- Any raw private value remains, token map would be sent outbound, request shape
  includes forbidden metadata, provider/billing error occurs, evidence anchoring fails,
  or medication safety proof is missing when required.
"""
    stop_first_failure = """# MEDAI Vertex Real Document Single Pilot Stop-On-First-Failure 16B

## Rule

Stop-on-first-failure: a future pilot halts immediately on the first failure and makes
no further call.

## Trigger Conditions

Stop on first failure, including privacy, request-shape, cost, provider, billing,
evidence-anchor, medication-safety, review-boundary, active-write, auto-accept, or
medical-decision failures, or any uncertainty about payload privacy.

## On Trigger

- Stop immediately; retries are not permitted.
- Preserve sanitized evidence only (fingerprints, counts, token classes); never raw
  PII or token map.
- No active MKB write, no auto-accept, no medical decision, no production queue
  mutation.
- Require fresh explicit human authorization and a fresh approval before any retry.

## Rollback / Cleanup Expectations

No active MKB records or production review queue mutations should exist. Generated
future reports must be sanitized and review-bound.
"""
    next_decision = """# MEDAI Vertex Real Document Single Pilot Next Decision 16B

## Option A: Remain No-Live

Continue synthetic/redacted stress and failure-injection coverage; keep preparing
without executing.

## Option B: Execute Future Explicit Single-Document Live Pilot

A separate, explicitly approved future execution block would set the dedicated future
live gate and run exactly one bounded call. This 16B package does not run it.

## Option C: Pause/Freeze

Pause and keep the 15Z/16A/16B governance state frozen.

## Recommended Next

Recommended next: user/operator decision, not automatic live execution.
"""
    return {
        "design": design,
        "approval": approval,
        "future_gate": future_gate,
        "one_call_plan": one_call_plan,
        "stop_first_failure": stop_first_failure,
        "next_decision": next_decision,
    }


def write_design_docs(docs: dict[str, str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_MD.write_text(docs["design"], encoding="utf-8")
    APPROVAL_RECORD_MD.write_text(docs["approval"], encoding="utf-8")
    FUTURE_GATE_MD.write_text(docs["future_gate"], encoding="utf-8")
    ONE_CALL_PLAN_MD.write_text(docs["one_call_plan"], encoding="utf-8")
    STOP_FIRST_FAILURE_MD.write_text(docs["stop_first_failure"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str) for p in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _future_live_gate_currently_set() -> bool:
    # Defense-in-depth: confirm the named future gate (and common live-gate names)
    # are NOT present in the environment. This block never sets them.
    candidate_gates = (
        FUTURE_LIVE_GATE_NAME,
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED",
    )
    return any(os.environ.get(name) not in (None, "", "0", "false", "False") for name in candidate_gates)


def _checklist(docs: dict[str, str]) -> dict[str, Any]:
    # Normalize whitespace so prose line-wrapping never breaks phrase checks.
    all_docs = re.sub(r"\s+", " ", "\n".join(docs.values()))
    return {
        "authorization_prep_design_created": DESIGN_MD.exists(),
        "approval_record_template_created": APPROVAL_RECORD_MD.exists(),
        "future_live_gate_spec_created": FUTURE_GATE_MD.exists(),
        "bounded_one_call_plan_created": ONE_CALL_PLAN_MD.exists(),
        "stop_on_first_failure_plan_created": STOP_FIRST_FAILURE_MD.exists(),
        "next_decision_memo_created": NEXT_DECISION_MD.exists(),
        "no_live_status_present": "Explicit no-live status" in all_docs and "design-only" in all_docs,
        "own_approval_required_present": "requires its own approval" in all_docs and "Human authorization required" in all_docs,
        "future_live_gate_named_present": FUTURE_LIVE_GATE_NAME in all_docs and "Dedicated future live gate required" in all_docs,
        "future_live_gate_not_set_present": "This block does not set the live gate" in all_docs and "no command that sets a live gate" in all_docs,
        "one_document_limit_present": "One-document maximum" in all_docs and "one-document limit" in all_docs.lower(),
        "one_call_limit_present": "One-call limit" in all_docs and "one-call limit" in all_docs.lower(),
        "redacted_tokenized_only_present": "Redacted/tokenized payload only" in all_docs,
        "no_raw_pii_boundary_present": "No raw PII" in all_docs,
        "no_token_map_outbound_boundary_present": "No token map outbound" in all_docs and "token map remains local" in all_docs,
        "bounded_token_ceiling_present": "Bounded token ceiling" in all_docs or "bounded token ceiling" in all_docs,
        "cost_cap_boundary_present": "cost cap" in all_docs.lower(),
        "request_shape_boundary_present": "contents` and `generationConfig" in all_docs and "forbidden" in all_docs and "metadata" in all_docs,
        "evidence_anchor_boundary_present": "Evidence anchoring" in all_docs and "source span" in all_docs,
        "label_alias_boundary_present": "Declared label alias policy" in all_docs,
        "medication_safety_boundary_present": "Medication safety non-bypass" in all_docs,
        "no_active_write_boundary_present": "no active mkb write" in all_docs.lower(),
        "no_auto_accept_boundary_present": "no auto-accept" in all_docs.lower(),
        "no_medical_decision_boundary_present": "no medical decision" in all_docs.lower(),
        "review_required_boundary_present": "Review-required" in all_docs or "review-required" in all_docs,
        "stop_on_first_failure_present": "Stop-on-first-failure" in all_docs or "stop-on-first-failure" in all_docs,
        "human_authorization_required_present": "Human authorization required" in all_docs,
        "refusal_conditions_present": "Refusal Conditions" in all_docs,
        "next_decision_required": "user/operator decision, not automatic live execution" in all_docs,
        "gcp_sandbox_separate_present": "separate environment evidence only" in all_docs,
        "future_live_gate_currently_set": _future_live_gate_currently_set(),
        "real_doc_live_allowed_count": 0,
    }


def _bool_checks(checklist: dict[str, Any]) -> list[str]:
    # Boolean design checks that must all be True (excludes counts and the
    # "currently_set" guard, which must be False).
    return [
        k for k, v in checklist.items()
        if isinstance(v, bool) and k not in ("future_live_gate_currently_set",)
    ]


def _summary(checklist: dict[str, Any], privacy_result: str) -> dict[str, Any]:
    bool_keys = _bool_checks(checklist)
    return {
        "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-AUTHORIZATION-PREP-NO-LIVE-16B",
        **checklist,
        "required_design_checks_passed": sum(1 for k in bool_keys if checklist[k] is True),
        "required_design_checks_total": len(bool_keys),
        "docs_only_change": True,
        "production_code_changed": False,
        "future_live_gate_set_in_this_block": False,
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "real_private_document_processed": False,
        "whole_corpus_processed": False,
        "private_corpus_read": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "medical_decision_made_count": 0,
        "production_queue_mutated": False,
        "sandbox_treated_as_medai_validation": False,
        "privacy_result": privacy_result,
        "billing_check_pending": True,
    }


def _design_matrix(summary: dict[str, Any]) -> str:
    lines = ["# 16B authorization-prep matrix", "", "| Check | Value |", "| --- | --- |"]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _approval_matrix(checklist: dict[str, Any]) -> str:
    rows = [
        ("own_approval_required", checklist["own_approval_required_present"]),
        ("human_authorization_required", checklist["human_authorization_required_present"]),
        ("dedicated_future_live_gate_named", checklist["future_live_gate_named_present"]),
        ("future_live_gate_not_set", checklist["future_live_gate_not_set_present"]),
        ("future_live_gate_currently_set", checklist["future_live_gate_currently_set"]),
        ("cost_cap_acknowledgement", checklist["cost_cap_boundary_present"]),
        ("bounded_token_ceiling", checklist["bounded_token_ceiling_present"]),
        ("one_document_limit", checklist["one_document_limit_present"]),
        ("one_call_limit", checklist["one_call_limit_present"]),
        ("stop_on_first_failure", checklist["stop_on_first_failure_present"]),
        ("no_active_write", checklist["no_active_write_boundary_present"]),
        ("no_auto_accept", checklist["no_auto_accept_boundary_present"]),
        ("no_medical_decision", checklist["no_medical_decision_boundary_present"]),
    ]
    lines = ["# 16B approval gate matrix", "", "| Requirement | Status |", "| --- | --- |"]
    lines.extend(f"| {name} | `{status}` |" for name, status in rows)
    lines.append("")
    return "\n".join(lines)


def _implementation(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-AUTHORIZATION-PREP-NO-LIVE-16B",
        "",
        "## Result",
        "",
        "- Prepared a no-live authorization package for a future single-document Vertex pilot:",
        "  its own approval record, a dedicated future live gate spec, a bounded one-call",
        "  execution plan, and stop-on-first-failure handling.",
        "- Did not set any live gate (confirmed the gate is absent from the environment).",
        "- Did not call any provider or billing API, process real documents, read a corpus,",
        "  write active MKB, mutate the production review queue, auto-accept, or produce a",
        "  medical decision.",
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
    approval_matrix = _approval_matrix(checklist)
    implementation = _implementation(summary)
    privacy_result = "passed" if _privacy_passes(docs, checklist, summary, design_matrix, approval_matrix, implementation) else "failed"
    summary["privacy_result"] = privacy_result
    design_matrix = _design_matrix(summary)
    implementation = _implementation(summary)
    return {
        "summary": summary,
        "checklist": checklist,
        "design_matrix": design_matrix,
        "approval_matrix": approval_matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CHECKLIST_JSON.write_text(json.dumps(reports["checklist"], indent=2), encoding="utf-8")
    DESIGN_MATRIX_MD.write_text(reports["design_matrix"], encoding="utf-8")
    APPROVAL_MATRIX_MD.write_text(reports["approval_matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["required_design_checks_passed"] == summary["required_design_checks_total"],
            summary["real_doc_live_allowed_count"] == 0,
            summary["future_live_gate_currently_set"] is False,
            summary["future_live_gate_set_in_this_block"] is False,
            summary["docs_only_change"] is True,
            summary["production_code_changed"] is False,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
            summary["real_private_document_processed"] is False,
            summary["whole_corpus_processed"] is False,
            summary["private_corpus_read"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["medical_decision_made_count"] == 0,
            summary["production_queue_mutated"] is False,
            summary["sandbox_treated_as_medai_validation"] is False,
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
        "medai_vertex_real_doc_single_pilot_authorization_prep_no_live_16b_ready"
        if ready
        else "medai_vertex_real_doc_single_pilot_authorization_prep_no_live_16b_not_ready"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "block"}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
