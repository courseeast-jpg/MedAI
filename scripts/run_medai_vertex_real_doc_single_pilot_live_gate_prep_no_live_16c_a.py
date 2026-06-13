#!/usr/bin/env python3
"""Generate the 16C-A no-live live-gate-preparation package.

16C-A defines (text-only) the dedicated future live gate, the operator approval
packet, the gate environment template, and a pre-16D readiness matrix for a future
one-document Vertex pilot. It proves the gate is NOT active and runs nothing.

Hard boundaries (never relaxed): no provider call; no Vertex/Gemini/Claude/OpenAI
live execution; no billing API call; no real/private document processing; no private
corpus read; no whole-corpus processing; no PDF/image/OCR processing; no active MKB
write; no MKB DB open; no auto-accept; no medical decision; no production queue
mutation. This block does NOT set the dedicated future live gate.

Stdlib only. Imports no provider/billing/network/PDF-OCR/DB/MKB modules. Inspects only
the 16C-A docs and repo metadata.
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

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16C_A"
LIVE_GATE_PREP_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_GATE_PREP_NO_LIVE_16C_A.md"
DEDICATED_GATE_SPEC_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_DEDICATED_GATE_SPEC_16C_A.md"
OPERATOR_APPROVAL_PACKET_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_OPERATOR_APPROVAL_PACKET_16C_A.md"
GATE_ENVIRONMENT_TEMPLATE_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_GATE_ENVIRONMENT_TEMPLATE_16C_A.md"
PRE_16D_READINESS_MATRIX_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_PRE_16D_READINESS_MATRIX_16C_A.md"
NEXT_DECISION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_NEXT_DECISION_16C_A.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
LIVE_GATE_MATRIX_MD = REPORT_DIR / "live_gate_matrix.md"

# The dedicated future live gate is NAMED only. This block never sets it.
DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

# Mandatory no-live phrases that must appear across the 16C-A docs (normalized).
REQUIRED_PHRASES = (
    "no-live",
    "no provider call",
    "no Vertex live execution",
    "no Gemini live execution",
    "no Claude/OpenAI live execution",
    "no real/private document processing",
    "no private corpus read",
    "no corpus processing",
    "no PDF/image/OCR processing",
    "no active MKB write",
    "no auto-accept",
    "no medical decision",
    "no billing API call",
    "no production queue mutation",
    "explicit operator approval",
    "explicit user authorization",
    "separate 16D authorization",
    "cost cap",
    "redaction/tokenization preflight",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "evidence capture",
    "rollback",
    "16D is not started",
)

# Phrases that, if present, indicate a forbidden permission/claim.
FORBIDDEN_PHRASES = (
    "16C-A authorizes live execution",
    "whole-corpus processing is allowed",
    "active MKB write is allowed",
    "auto-accept is allowed",
    "sandbox proves MedAI real-document readiness",
)

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
    live_gate_prep = f"""# MEDAI Vertex Real Document Single Pilot Live Gate Prep 16C-A

## Status

- This block is no-live.
- This block does not authorize live execution.
- This block does not set the live gate.
- The dedicated future live gate is only named/speced here, never set.
- This block builds on the frozen 15Z, 16A, and 16B governance state and does not
  modify it.

## Hard Boundaries

This block enforces, and the future pilot inherits: no provider call; no Vertex live
execution; no Gemini live execution; no Claude/OpenAI live execution; no billing API
call; no real/private document processing; no private corpus read; no corpus
processing; no PDF/image/OCR processing; no active MKB write; no MKB DB open; no
auto-accept; no medical decision; no production queue mutation.

## Authorization Requirements Before Any Future Live Call

- Explicit user authorization is required before any later live call.
- Separate 16D authorization is required before any one-call pilot.
- Explicit operator approval record is required before any future live run.
- Cost cap confirmation is required before any future live run.
- Redaction/tokenization preflight verification is required before any future outbound
  payload is assembled.
- One-document limit: the future pilot processes one document only.
- One-call limit: the future pilot makes exactly one call.
- Stop-on-first-failure is required in the future pilot.
- Evidence capture is required before and after any future live run.
- Rollback and failure handling are required before any future live run.

## Boundaries That Stay In Force

- No corpus processing; no whole-corpus processing.
- No active MKB write and no auto-accept during the future pilot unless separately
  authorized by a later block.
- No medical decision output.

## GCP Synthetic Sandbox Separation

The GCP synthetic sandbox evidence proves only environment, authentication, and the
request path. It does not prove MedAI real-document readiness and is separate
environment evidence only.

## 16D Status

16D is not started. No one-call pilot is initiated by this block.
"""
    dedicated_gate_spec = f"""# MEDAI Vertex Real Document Single Pilot Dedicated Gate Spec 16C-A

## Gate Name (Text Only)

Dedicated future live gate name: `{DEDICATED_GATE_NAME}`.

## Current Required Value

- Current required value: unset or false.
- This block must not set it.
- Any value of 1, true, or enabled before 16D is a NO-GO.

## The Gate Alone Is Not Enough

Setting the gate is necessary but not sufficient. Before any future live call, all of
the following must also pass: explicit operator approval, cost cap confirmation,
redaction/tokenization preflight proof, one-document limit, and one-call limit. A
failure of any one is a stop-on-first-failure NO-GO.

## No-Live Confirmation

This spec is no-live: no provider call and no Vertex live execution occur from this
document. The gate is named/speced only.
"""
    operator_approval_packet = f"""# MEDAI Vertex Real Document Single Pilot Operator Approval Packet 16C-A

This packet does not itself authorize live execution. Completing it prepares, but does
not grant, a future single-document live pilot. Explicit operator approval is required
before any future live run, and explicit user authorization plus separate 16D
authorization are required before any later live call.

- Operator identity placeholder: `[OPERATOR_ID]`
- Approver identity placeholder: `[APPROVER_ID]`
- Date/time placeholder: `[AUTHORIZED_AT]`
- Selected single document provenance declaration: `[PROVENANCE_DECLARATION]`
- Confirmation redaction/tokenization preflight passed: `[CONFIRM_REDACTION_PREFLIGHT]`
- Confirmation no raw PII in outbound payload: `[CONFIRM_NO_RAW_PII_OUTBOUND]`
- Confirmation token map remains local: `[CONFIRM_TOKEN_MAP_LOCAL_ONLY]`
- Confirmation cost cap accepted: `[CONFIRM_COST_CAP_ACCEPTED]`
- Confirmation one-document limit: `[CONFIRM_ONE_DOCUMENT]`
- Confirmation one-call limit: `[CONFIRM_ONE_CALL]`
- Confirmation stop-on-first-failure acknowledged: `[CONFIRM_STOP_ON_FIRST_FAILURE]`
- Confirmation evidence capture plan ready: `[CONFIRM_EVIDENCE_CAPTURE]`
- Confirmation rollback/failure handling ready: `[CONFIRM_ROLLBACK_READY]`
- Confirmation dedicated future live gate name acknowledged:
  `[CONFIRM_FUTURE_LIVE_GATE_ACK]` (gate: `{DEDICATED_GATE_NAME}`)
- Confirmation no active MKB write and no auto-accept during pilot: `[CONFIRM_NO_WRITE_NO_AUTO_ACCEPT]`
- Explicit approval decision: `[APPROVED / NOT_APPROVED]`
"""
    gate_environment_template = f"""# MEDAI Vertex Real Document Single Pilot Gate Environment Template 16C-A

## Purpose

Text-only template describing how the dedicated future live gate environment variable
would be referenced. This block does not set it and contains no command that sets it.

## Gate Variable

- Name: `{DEDICATED_GATE_NAME}`
- Current required value: unset or false.
- Forbidden values before 16D: 1, true, TRUE, yes, enabled, ENABLED.

## This Block Does Not Set The Gate

This block must not set the live gate. This template intentionally contains no command
that sets a live gate. Setting the gate is reserved for a separate, explicitly approved
16D execution block, and only after explicit operator approval, cost cap, redaction
preflight, one-document limit, and one-call limit all pass.

## No-Live Confirmation

No provider call, no Vertex live execution, and no billing API call result from this
template.
"""
    pre_16d_readiness_matrix = f"""# MEDAI Vertex Real Document Single Pilot Pre-16D Readiness Matrix 16C-A

| Pre-16D requirement | Status (this block) | Required before any future live call |
| --- | --- | --- |
| Dedicated future live gate named/speced | done (named only) | yes |
| Dedicated future live gate set | not set (must not be set here) | only in 16D, separately authorized |
| Explicit user authorization | pending | yes |
| Separate 16D authorization | pending; 16D is not started | yes |
| Explicit operator approval record | template only | yes |
| Cost cap confirmation | pending | yes |
| Redaction/tokenization preflight proof | pending | yes |
| One-document limit | required | yes |
| One-call limit | required | yes |
| Stop-on-first-failure | required | yes |
| Evidence capture before and after | required | yes |
| Rollback/failure handling | required | yes |
| No corpus processing | enforced | yes |
| No active MKB write / no auto-accept | enforced unless separately authorized | yes |
| GCP synthetic sandbox = environment/auth/request path only | acknowledged | yes |

All rows must be satisfied, in addition to the gate, before any future one-call pilot.
The gate alone is not enough.
"""
    next_decision = """# MEDAI Vertex Real Document Single Pilot Next Decision 16C-A

## Option A: Remain No-Live

Continue no-live preparation (redaction preflight proofs, approval packets) without
executing anything.

## Option B: Proceed To Redaction Preflight Prep (16C-B)

Prepare the redaction/tokenization preflight proof package next. This remains no-live
and does not set the live gate.

## Option C: Pause/Freeze

Pause and keep the 15Z/16A/16B/16C-A governance state frozen.

## Recommended Next

Recommended next: user/operator decision, not automatic live execution. 16D is not
started.
"""
    return {
        "live_gate_prep": live_gate_prep,
        "dedicated_gate_spec": dedicated_gate_spec,
        "operator_approval_packet": operator_approval_packet,
        "gate_environment_template": gate_environment_template,
        "pre_16d_readiness_matrix": pre_16d_readiness_matrix,
        "next_decision": next_decision,
    }


def write_docs(docs: dict[str, str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_GATE_PREP_MD.write_text(docs["live_gate_prep"], encoding="utf-8")
    DEDICATED_GATE_SPEC_MD.write_text(docs["dedicated_gate_spec"], encoding="utf-8")
    OPERATOR_APPROVAL_PACKET_MD.write_text(docs["operator_approval_packet"], encoding="utf-8")
    GATE_ENVIRONMENT_TEMPLATE_MD.write_text(docs["gate_environment_template"], encoding="utf-8")
    PRE_16D_READINESS_MATRIX_MD.write_text(docs["pre_16d_readiness_matrix"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str) for p in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _gate_environment_active() -> bool:
    value = os.environ.get(DEDICATED_GATE_NAME)
    if value is None:
        return False
    return value.strip() in ACTIVE_VALUES


def evaluate() -> tuple[dict[str, Any], list[str], dict[str, str]]:
    docs = _docs()
    write_docs(docs)
    failures: list[str] = []

    required_paths = {
        "live_gate_prep": LIVE_GATE_PREP_MD,
        "dedicated_gate_spec": DEDICATED_GATE_SPEC_MD,
        "operator_approval_packet": OPERATOR_APPROVAL_PACKET_MD,
        "gate_environment_template": GATE_ENVIRONMENT_TEMPLATE_MD,
        "pre_16d_readiness_matrix": PRE_16D_READINESS_MATRIX_MD,
        "next_decision": NEXT_DECISION_MD,
    }
    for name, path in required_paths.items():
        if not path.exists():
            failures.append(f"required_doc_missing:{name}")

    # Normalize whitespace so prose line-wrapping never breaks phrase checks.
    all_docs = re.sub(r"\s+", " ", "\n".join(docs.values()))
    for phrase in REQUIRED_PHRASES:
        if re.sub(r"\s+", " ", phrase) not in all_docs:
            failures.append(f"required_phrase_absent:{phrase}")

    for phrase in FORBIDDEN_PHRASES:
        if phrase in all_docs:
            failures.append(f"forbidden_phrase_present:{phrase}")

    if DEDICATED_GATE_NAME not in all_docs:
        failures.append("dedicated_gate_name_absent")
    if "This block must not set it" not in all_docs:
        failures.append("docs_do_not_state_gate_not_set")
    if "1, true, or enabled before 16D is a NO-GO" not in all_docs:
        failures.append("docs_do_not_state_active_gate_is_no_go")

    gate_env_active = _gate_environment_active()
    if gate_env_active:
        failures.append("future_live_gate_environment_active")

    report = {
        "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-LIVE-GATE-PREP-NO-LIVE-16C-A",
        "no_live": True,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_live_execution": False,
        "claude_live_execution": False,
        "openai_live_execution": False,
        "billing_api_call_made": False,
        "real_private_document_processed": False,
        "private_corpus_read": False,
        "whole_corpus_processed": False,
        "pdf_or_image_processed": False,
        "ocr_routing_executed": False,
        "active_mkb_write": False,
        "mkb_db_opened": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "future_live_gate_named": DEDICATED_GATE_NAME in all_docs,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": gate_env_active,
        "operator_approval_required": True,
        "cost_cap_required": True,
        "redaction_preflight_required": True,
        "one_document_limit_required": True,
        "one_call_limit_required": True,
        "stop_on_first_failure_required": True,
        "future_16d_not_started": True,
        "sandbox_treated_as_medai_validation": False,
        "safety_result": "passed",
    }

    if not _privacy_passes(docs, report):
        failures.append("privacy_check_failed")

    if failures:
        report["safety_result"] = "failed"
    return report, failures, docs


def _matrix(report: dict[str, Any]) -> str:
    keys = [
        "no_live", "provider_call_made", "vertex_live_execution", "gemini_live_execution",
        "claude_live_execution", "openai_live_execution", "billing_api_call_made",
        "real_private_document_processed", "private_corpus_read", "whole_corpus_processed",
        "pdf_or_image_processed", "ocr_routing_executed", "active_mkb_write", "mkb_db_opened",
        "auto_accept_enabled", "medical_decision_made", "production_queue_mutated",
        "future_live_gate_named", "future_live_gate_set", "future_live_gate_environment_active",
        "operator_approval_required", "cost_cap_required", "redaction_preflight_required",
        "one_document_limit_required", "one_call_limit_required", "stop_on_first_failure_required",
        "future_16d_not_started", "sandbox_treated_as_medai_validation", "safety_result",
    ]
    return "\n".join(
        [
            "# 16C-A live gate matrix",
            "",
            f"Dedicated future live gate (named only): `{DEDICATED_GATE_NAME}` — current required value: unset or false.",
            "",
            "| Field | Value |",
            "| --- | --- |",
            *[f"| {k} | `{report[k]}` |" for k in keys],
            "",
            "The gate is named/speced only and is NOT set in this block. The gate alone is not",
            "enough: operator approval, cost cap, redaction preflight, one-document limit, and",
            "one-call limit must also pass before any future one-call pilot. 16D is not started.",
            "",
        ]
    )


def _implementation(report: dict[str, Any], failures: list[str]) -> str:
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-LIVE-GATE-PREP-NO-LIVE-16C-A",
        "",
        f"## Status: **{status}**",
        "",
        "## Result",
        "",
        "- Prepared a no-live live-gate package: dedicated future live gate spec (named only),",
        "  operator approval packet, gate environment template, and pre-16D readiness matrix.",
        "- Did not set the dedicated future live gate; confirmed it is not active in the",
        "  environment.",
        "- Did not call any provider or billing API, process real/private documents, read a",
        "  corpus, process PDF/image/OCR, open the MKB DB, write active MKB, mutate the",
        "  production queue, auto-accept, or produce a medical decision.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in report.items():
        if key == "block":
            continue
        lines.append(f"- {key}: `{value}`")
    if failures:
        lines += ["", "## Failures", ""] + [f"- `{f}`" for f in failures]
    lines += [
        "",
        "## Recommended next (NOT started)",
        "",
        "- MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-REDACTION-PREFLIGHT-NO-LIVE-16C-B — not started;",
        "  requires explicit user authorization. 16D is not started.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], failures: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(report, failures), encoding="utf-8")
    LIVE_GATE_MATRIX_MD.write_text(_matrix(report), encoding="utf-8")


def main() -> int:
    report, failures, _docs_unused = evaluate()
    write_reports(report, failures)
    ok = not failures and report["safety_result"] == "passed"
    print(
        "medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a_pass"
        if ok
        else "medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a_fail"
    )
    print(json.dumps({**report, "failures": failures}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
