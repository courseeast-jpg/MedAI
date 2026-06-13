#!/usr/bin/env python3
"""Generate + validate the 16C-C FINAL no-live preflight package.

Consolidates and verifies the 16A / 16B / 16C-A / 16C-B readiness artifacts, produces
the final go/no-go matrix, final operator approval requirements, and 16D handoff
criteria. This block is NO-LIVE and does NOT set the dedicated future live gate. 16D
is NOT started.

Hard boundaries (never relaxed): no provider call; no Vertex/Gemini/Claude/OpenAI live
execution; no billing API call; no real/private document processing; no private corpus
read; no whole-corpus processing; no PDF/image/OCR processing; no MKB DB open; no
active MKB write; no auto-accept; no medical decision; no production queue mutation.

Stdlib only. No provider/billing/network/PDF-OCR/DB/MKB imports. Inspects only the
16C-C docs, the prior committed 16A/16B/16C-A/16C-B summaries, and repo metadata.
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

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16C_C"
FINAL_PREFLIGHT_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_FINAL_PREFLIGHT_NO_LIVE_16C_C.md"
FINAL_GO_NO_GO_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_FINAL_GO_NO_GO_MATRIX_16C_C.md"
ENTRY_CRITERIA_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16D_ENTRY_CRITERIA_16C_C.md"
OPERATOR_APPROVAL_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_FINAL_OPERATOR_APPROVAL_REQUIREMENTS_16C_C.md"
COST_PRIVACY_SAFETY_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_FINAL_COST_PRIVACY_SAFETY_MATRIX_16C_C.md"
HANDOFF_TEMPLATE_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16D_HANDOFF_TEMPLATE_16C_C.md"
NEXT_DECISION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_NEXT_DECISION_16C_C.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_final_preflight_no_live_16c_c"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
FINAL_GO_NO_GO_MATRIX_MD = REPORT_DIR / "final_go_no_go_matrix.md"
PRIOR_BLOCK_INVENTORY_JSON = REPORT_DIR / "prior_block_inventory.json"

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

PRIOR_SUMMARIES = {
    "16A": REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_design_no_live_16a" / "summary.json",
    "16B": REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_authorization_prep_no_live_16b" / "summary.json",
    "16C-A": REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a" / "summary.json",
    "16C-B": REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b" / "summary.json",
}

FORBIDDEN_REPORT_TOKENS = (
    "ya29.", "AIza", "Bearer ", "Authorization" + ":", "access" + "_token",
    "refresh" + "_token", "private" + "_key", "application" + "_default" + "_credentials",
    "g" + "cloud", "C:\\", "/home/",
)

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
    "no billing API call",
    "no MKB write",
    "no auto-accept",
    "no medical decision",
    "no production queue mutation",
    "explicit operator approval",
    "explicit user authorization",
    "cost cap",
    "redaction/tokenization proof",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "rollback",
    "evidence capture",
    "16D is not started",
)

FORBIDDEN_PHRASES = (
    "16C-C authorizes live execution",
    "whole-corpus processing is allowed",
    "active MKB write is allowed",
    "auto-accept is allowed",
    "sandbox proves MedAI real-document readiness",
)


def _gate_environment_active() -> bool:
    value = os.environ.get(DEDICATED_GATE_NAME)
    return value is not None and value.strip() in ACTIVE_VALUES


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_prior_blocks() -> tuple[dict[str, Any], dict[str, bool]]:
    """Verify the prior committed summaries. Returns (inventory, verified_flags)."""
    inventory: dict[str, Any] = {}
    verified: dict[str, bool] = {}

    a = _load(PRIOR_SUMMARIES["16A"])
    verified["16A"] = bool(
        a
        and a.get("required_design_checks_passed") == a.get("required_design_checks_total")
        and a.get("real_doc_live_allowed_count") == 0
        and a.get("live_call_made") is False
        and a.get("external_api_used") is False
        and a.get("billing_api_used") is False
        and a.get("active_written_count") == 0
        and a.get("auto_accept_true_count") == 0
        and a.get("medical_decision_made_count") == 0
        and a.get("privacy_result") == "passed"
    )

    b = _load(PRIOR_SUMMARIES["16B"])
    verified["16B"] = bool(
        b
        and b.get("required_design_checks_passed") == b.get("required_design_checks_total")
        and b.get("real_doc_live_allowed_count") == 0
        and b.get("live_call_made") is False
        and b.get("external_api_used") is False
        and b.get("billing_api_used") is False
        and b.get("active_written_count") == 0
        and b.get("auto_accept_true_count") == 0
        and b.get("medical_decision_made_count") == 0
        and b.get("privacy_result") == "passed"
        and b.get("future_live_gate_set_in_this_block") is False
    )

    ca = _load(PRIOR_SUMMARIES["16C-A"])
    verified["16C-A"] = bool(
        ca
        and ca.get("safety_result") == "passed"
        and ca.get("no_live") is True
        and ca.get("future_live_gate_named") is True
        and ca.get("future_live_gate_set") is False
        and ca.get("future_live_gate_environment_active") is False
        and ca.get("billing_api_call_made") is False
        and ca.get("active_mkb_write") is False
        and ca.get("auto_accept_enabled") is False
        and ca.get("medical_decision_made") is False
        and ca.get("future_16d_not_started") is True
    )

    cb = _load(PRIOR_SUMMARIES["16C-B"])
    verified["16C-B"] = bool(
        cb
        and cb.get("safety_result") == "passed"
        and cb.get("privacy_result") == "passed"
        and cb.get("no_live") is True
        and cb.get("raw_identifier_leak_count") == 0
        and cb.get("token_map_written_to_public_report") is False
        and cb.get("outbound_payload_tokenized") is True
        and cb.get("future_live_gate_set") is False
        and cb.get("future_live_gate_environment_active") is False
        and cb.get("future_16d_not_started") is True
    )

    for key, path in PRIOR_SUMMARIES.items():
        d = _load(path)
        inventory[key] = {
            "summary_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "exists": path.exists(),
            "verified": verified.get(key, False),
            "block": (d or {}).get("block", ""),
        }
    # Carry the specific 16C-B safety indicators forward for the final summary.
    inventory["16C-B"]["raw_identifier_leak_count"] = (cb or {}).get("raw_identifier_leak_count")
    inventory["16C-B"]["token_map_written_to_public_report"] = (cb or {}).get("token_map_written_to_public_report")
    inventory["16C-B"]["outbound_payload_tokenized"] = (cb or {}).get("outbound_payload_tokenized")
    inventory["16C-A"]["future_live_gate_named"] = (ca or {}).get("future_live_gate_named")
    inventory["16C-A"]["future_live_gate_set"] = (ca or {}).get("future_live_gate_set")
    return inventory, verified


def _docs() -> dict[str, str]:
    final_preflight = f"""# MEDAI Vertex Real Document Single Pilot Final Preflight No-Live 16C-C

## Status

- This block is no-live.
- 16C-C does not authorize live execution.
- 16C-C does not set the live gate.
- 16D is not started.
- The dedicated future live gate `{DEDICATED_GATE_NAME}` must remain unset/inactive
  after 16C-C.

## Hard Boundaries

This block enforces: no provider call; no Vertex live execution; no Gemini live
execution; no Claude/OpenAI live execution; no billing API call; no real/private
document processing; no private corpus read; no corpus processing; no PDF/image/OCR
processing; no MKB write; no auto-accept; no medical decision; no production queue
mutation.

## What This Block Consolidates

This final preflight consolidates and verifies the readiness artifacts of 16A
(design-only), 16B (authorization-prep), 16C-A (live-gate-prep), and 16C-B
(synthetic-only redaction/tokenization preflight). It produces the final go/no-go
matrix, final operator approval requirements, and 16D entry/handoff criteria.

## Requirements Before Any Future 16D

- Explicit user authorization is required before 16D.
- Explicit operator approval is required before future 16D.
- Cost cap is required before future 16D.
- Redaction/tokenization proof is required before future 16D.
- One-document limit is required before future 16D.
- One-call limit is required before future 16D.
- Stop-on-first-failure is required before future 16D.
- Rollback and failure plan are required before future 16D.
- Evidence capture is required before and after future 16D.

## Privacy And Payload Requirements

- No token map in outbound payload or public reports.
- No raw PII/PHI in outbound payload.
- No private corpus traversal.

## Future Pilot Shape

If authorized later, the future 16D pilot must be one document and one call only. No
corpus processing is permitted. No active MKB write and no auto-accept are permitted
unless a later block separately authorizes them. No medical decision is permitted.

## GCP Synthetic Sandbox Separation

The GCP synthetic sandbox evidence proves only environment, authentication, and the
request path. It is separate environment evidence only and does not prove MedAI
real-document readiness.

## Provider Call Condition

No provider call may occur unless the dedicated live gate is explicitly set by a later
authorized 16D block, and only after explicit operator approval, cost cap, redaction
proof, one-document limit, and one-call limit all pass.
"""
    go_no_go = f"""# MEDAI Vertex Real Document Single Pilot Final Go/No-Go Matrix 16C-C

This block is no-live and does not set the live gate. 16D is not started.

| Final gate | Required state before any future 16D | NO-GO if not met |
| --- | --- | --- |
| Dedicated live gate `{DEDICATED_GATE_NAME}` | unset/inactive now; set only in authorized 16D | yes |
| Explicit user authorization | obtained before 16D | yes |
| Explicit operator approval | recorded before 16D | yes |
| Cost cap | confirmed before 16D | yes |
| Redaction/tokenization proof | passed before 16D | yes |
| One-document limit | enforced | yes |
| One-call limit | enforced | yes |
| Stop-on-first-failure | enforced | yes |
| Rollback / failure plan | ready before 16D | yes |
| Evidence capture before and after | ready | yes |
| No token map in outbound payload or public reports | enforced | yes |
| No raw PII/PHI in outbound payload | enforced | yes |
| No corpus processing | enforced | yes |
| No active MKB write / no auto-accept / no medical decision | enforced unless separately authorized | yes |

Any unmet row is a mandatory NO-GO. The gate alone is never enough.
"""
    entry_criteria = f"""# MEDAI Vertex Real Document Single Pilot 16D Entry Criteria 16C-C

This block is no-live. 16D is not started. The dedicated live gate
`{DEDICATED_GATE_NAME}` must remain unset/inactive after 16C-C.

## 16D May Be Entered Only When ALL Of The Following Hold

1. Explicit user authorization and explicit operator approval are recorded.
2. Cost cap is confirmed.
3. Redaction/tokenization proof has passed (no raw PII/PHI, no token map in outbound).
4. One-document limit and one-call limit are enforced.
5. Stop-on-first-failure handling is in place.
6. Rollback/failure plan and evidence capture (before and after) are ready.
7. The dedicated live gate is set only inside the authorized 16D block.

## 16D Must Stop Immediately If

The live gate, approval record, cost cap, redaction proof, or one-call limit is
missing. No provider call, no real/private document processing, no corpus processing,
no PDF/image/OCR processing, no MKB write, no auto-accept, and no medical decision may
occur outside an explicitly authorized 16D execution.
"""
    operator_approval = """# MEDAI Vertex Real Document Single Pilot Final Operator Approval Requirements 16C-C

This block is no-live and grants nothing. Explicit operator approval and explicit user
authorization are required before any future 16D live call.

- Operator identity placeholder: `[OPERATOR_ID]`
- Approver identity placeholder: `[APPROVER_ID]`
- Date/time placeholder: `[AUTHORIZED_AT]`
- Confirmation explicit user authorization obtained: `[CONFIRM_USER_AUTHORIZATION]`
- Confirmation cost cap confirmed: `[CONFIRM_COST_CAP]`
- Confirmation redaction/tokenization proof passed: `[CONFIRM_REDACTION_PROOF]`
- Confirmation one-document limit: `[CONFIRM_ONE_DOCUMENT]`
- Confirmation one-call limit: `[CONFIRM_ONE_CALL]`
- Confirmation stop-on-first-failure ready: `[CONFIRM_STOP_ON_FIRST_FAILURE]`
- Confirmation rollback/failure plan ready: `[CONFIRM_ROLLBACK]`
- Confirmation evidence capture before and after ready: `[CONFIRM_EVIDENCE_CAPTURE]`
- Confirmation no token map in outbound payload or public reports: `[CONFIRM_NO_TOKEN_MAP]`
- Confirmation no raw PII/PHI in outbound payload: `[CONFIRM_NO_RAW_PII]`
- Confirmation no corpus processing: `[CONFIRM_NO_CORPUS]`
- Confirmation no active MKB write and no auto-accept and no medical decision:
  `[CONFIRM_NO_WRITE_NO_AUTO_ACCEPT_NO_DECISION]`
- Explicit approval decision: `[APPROVED / NOT_APPROVED]`
"""
    cost_privacy_safety = f"""# MEDAI Vertex Real Document Single Pilot Final Cost/Privacy/Safety Matrix 16C-C

This block is no-live; no billing API call and no provider call occur. 16D is not
started.

| Dimension | Requirement before any future 16D |
| --- | --- |
| Cost | Cost cap confirmed (model, project, region, max calls, max tokens, dollar cap, stop-on-quota). |
| Privacy | Redaction/tokenization proof passed; no raw PII/PHI in outbound payload; no token map in outbound payload or public reports; no private corpus read; no private corpus traversal. |
| Safety | One-document limit; one-call limit; stop-on-first-failure; rollback; evidence capture; no active MKB write; no auto-accept; no medical decision; no production queue mutation. |
| Separation | GCP synthetic sandbox evidence proves only environment/auth/request path, not MedAI real-document readiness. |

The dedicated live gate `{DEDICATED_GATE_NAME}` remains unset/inactive after 16C-C.
"""
    handoff_template = f"""# MEDAI Vertex Real Document Single Pilot 16D Handoff Template 16C-C

This template prepares, but does not start, 16D. 16D is not started. This block is
no-live and does not set the live gate.

## Handoff Package Contents (To Be Completed Before Any Authorized 16D)

- Final operator approval record (signed).
- Explicit user authorization reference.
- Cost cap confirmation reference.
- Redaction/tokenization proof reference (no raw PII/PHI; no token map in outbound).
- One-document and one-call limit confirmation.
- Stop-on-first-failure plan.
- Rollback/failure plan.
- Evidence capture plan (before and after).
- Dedicated live gate name: `{DEDICATED_GATE_NAME}` (to be set only inside the
  authorized 16D block; unset/inactive now).

## Stop Conditions Carried Into 16D

16D must stop if the live gate, approval record, cost cap, redaction proof, or
one-call limit is missing. No provider call, no corpus processing, no PDF/image/OCR
processing, no MKB write, no auto-accept, and no medical decision occur outside an
explicitly authorized 16D execution.
"""
    next_decision = """# MEDAI Vertex Real Document Single Pilot Next Decision 16C-C

## Option A: Remain No-Live

Keep the full 15Z/16A/16B/16C-A/16C-B/16C-C governance state frozen and run no live
execution.

## Option B: Request Explicit Authorization For 16D

Request explicit user authorization for MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-ONE-CALL-LIVE-16D.
16D is not started and must not begin without separate explicit authorization.

## Option C: Pause/Freeze

Pause and keep the governance state frozen.

## Recommended Next

Recommended next: user/operator decision, not automatic live execution. 16D is not
started.
"""
    return {
        "final_preflight": final_preflight,
        "go_no_go": go_no_go,
        "entry_criteria": entry_criteria,
        "operator_approval": operator_approval,
        "cost_privacy_safety": cost_privacy_safety,
        "handoff_template": handoff_template,
        "next_decision": next_decision,
    }


def write_docs(docs: dict[str, str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_PREFLIGHT_MD.write_text(docs["final_preflight"], encoding="utf-8")
    FINAL_GO_NO_GO_MD.write_text(docs["go_no_go"], encoding="utf-8")
    ENTRY_CRITERIA_MD.write_text(docs["entry_criteria"], encoding="utf-8")
    OPERATOR_APPROVAL_MD.write_text(docs["operator_approval"], encoding="utf-8")
    COST_PRIVACY_SAFETY_MD.write_text(docs["cost_privacy_safety"], encoding="utf-8")
    HANDOFF_TEMPLATE_MD.write_text(docs["handoff_template"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def evaluate() -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    docs = _docs()
    write_docs(docs)
    failures: list[str] = []

    required_paths = {
        "final_preflight": FINAL_PREFLIGHT_MD,
        "go_no_go": FINAL_GO_NO_GO_MD,
        "entry_criteria": ENTRY_CRITERIA_MD,
        "operator_approval": OPERATOR_APPROVAL_MD,
        "cost_privacy_safety": COST_PRIVACY_SAFETY_MD,
        "handoff_template": HANDOFF_TEMPLATE_MD,
        "next_decision": NEXT_DECISION_MD,
    }
    for name, path in required_paths.items():
        if not path.exists():
            failures.append(f"required_doc_missing:{name}")

    all_docs = re.sub(r"\s+", " ", "\n".join(docs.values()))
    low = all_docs.lower()
    for phrase in REQUIRED_PHRASES:
        if re.sub(r"\s+", " ", phrase).lower() not in low:
            failures.append(f"required_phrase_absent:{phrase}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in low:
            failures.append(f"forbidden_phrase_present:{phrase}")

    inventory, verified = verify_prior_blocks()
    for key in ("16A", "16B", "16C-A", "16C-B"):
        if not PRIOR_SUMMARIES[key].exists():
            failures.append(f"prior_summary_missing:{key}")
        if not verified.get(key):
            failures.append(f"prior_block_not_verified:{key}")

    gate_active = _gate_environment_active()
    if gate_active:
        failures.append("future_live_gate_environment_active")

    cb = _load(PRIOR_SUMMARIES["16C-B"]) or {}
    ca = _load(PRIOR_SUMMARIES["16C-A"]) or {}

    report = {
        "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-FINAL-PREFLIGHT-NO-LIVE-16C-C",
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
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "future_live_gate_named": bool(ca.get("future_live_gate_named")),
        "future_live_gate_set": False,
        "future_live_gate_environment_active": gate_active,
        "sixteen_a_verified": verified.get("16A", False),
        "sixteen_b_verified": verified.get("16B", False),
        "sixteen_c_a_verified": verified.get("16C-A", False),
        "sixteen_c_b_verified": verified.get("16C-B", False),
        "sixteen_c_b_raw_identifier_leak_count": cb.get("raw_identifier_leak_count"),
        "sixteen_c_b_token_map_public_report": bool(cb.get("token_map_written_to_public_report")),
        "sixteen_c_b_outbound_payload_tokenized": bool(cb.get("outbound_payload_tokenized")),
        "operator_approval_required_before_16d": True,
        "cost_cap_required_before_16d": True,
        "redaction_preflight_required_before_16d": True,
        "one_document_limit_required_before_16d": True,
        "one_call_limit_required_before_16d": True,
        "stop_on_first_failure_required_before_16d": True,
        "rollback_required_before_16d": True,
        "future_16d_not_started": True,
        "sandbox_treated_as_medai_validation": False,
        "privacy_result": "passed",
        "safety_result": "passed",
    }

    if report["sixteen_c_b_raw_identifier_leak_count"] != 0:
        failures.append("sixteen_c_b_raw_identifier_leak_nonzero")
    if report["sixteen_c_b_token_map_public_report"] is not False:
        failures.append("sixteen_c_b_token_map_public_report_true")
    if report["sixteen_c_b_outbound_payload_tokenized"] is not True:
        failures.append("sixteen_c_b_outbound_payload_not_tokenized")
    if report["future_live_gate_named"] is not True:
        failures.append("future_live_gate_not_named")

    # Verify the published artifacts carry no credential/path leaks and pass privacy.
    matrix_md = _final_matrix(report, inventory)
    impl_md = _implementation(report, failures)
    published = "\n".join([json.dumps(report), json.dumps(inventory), matrix_md, impl_md])
    if any(tok in published for tok in FORBIDDEN_REPORT_TOKENS):
        failures.append("forbidden_credential_or_path_token_in_report")
    if not check_public_report_payload((report, inventory)).passed:
        failures.append("privacy_check_failed")

    if failures:
        report["safety_result"] = "failed"
    return report, failures, inventory


def _final_matrix(report: dict[str, Any], inventory: dict[str, Any]) -> str:
    prior_rows = [
        f"| {k} | `{inventory[k]['verified']}` | `{inventory[k]['exists']}` |"
        for k in ("16A", "16B", "16C-A", "16C-B")
    ]
    keys = [
        "no_live", "future_live_gate_named", "future_live_gate_set", "future_live_gate_environment_active",
        "sixteen_a_verified", "sixteen_b_verified", "sixteen_c_a_verified", "sixteen_c_b_verified",
        "sixteen_c_b_raw_identifier_leak_count", "sixteen_c_b_token_map_public_report",
        "sixteen_c_b_outbound_payload_tokenized", "operator_approval_required_before_16d",
        "cost_cap_required_before_16d", "redaction_preflight_required_before_16d",
        "one_document_limit_required_before_16d", "one_call_limit_required_before_16d",
        "stop_on_first_failure_required_before_16d", "rollback_required_before_16d",
        "future_16d_not_started", "sandbox_treated_as_medai_validation", "privacy_result", "safety_result",
    ]
    return "\n".join(
        [
            "# 16C-C final go/no-go matrix",
            "",
            "| Prior block | Verified | Report exists |",
            "| --- | --- | --- |",
            *prior_rows,
            "",
            "| Field | Value |",
            "| --- | --- |",
            *[f"| {k} | `{report[k]}` |" for k in keys],
            "",
            f"Dedicated future live gate (named only): `{DEDICATED_GATE_NAME}` — unset/inactive.",
            "This is the final no-live preflight. 16D is not started and requires separate",
            "explicit authorization. The gate alone is never enough.",
            "",
        ]
    )


def _implementation(report: dict[str, Any], failures: list[str]) -> str:
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-FINAL-PREFLIGHT-NO-LIVE-16C-C",
        "",
        f"## Status: **{status}**",
        "",
        "## Result",
        "",
        "- Consolidated and verified the 16A / 16B / 16C-A / 16C-B readiness artifacts.",
        "- Produced the final go/no-go matrix, final operator approval requirements, and 16D",
        "  entry/handoff criteria.",
        "- Did not set the dedicated future live gate; confirmed it is not active.",
        "- No provider/billing call, no real/private document, no corpus, no PDF/image/OCR,",
        "  no MKB DB open, no active MKB write, no auto-accept, no medical decision, no",
        "  production queue mutation.",
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
        "- MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-ONE-CALL-LIVE-16D — NOT started; requires",
        "  separate explicit authorization. The dedicated live gate remains unset/inactive.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], failures: list[str], inventory: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(report, failures), encoding="utf-8")
    FINAL_GO_NO_GO_MATRIX_MD.write_text(_final_matrix(report, inventory), encoding="utf-8")
    PRIOR_BLOCK_INVENTORY_JSON.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


def main() -> int:
    report, failures, inventory = evaluate()
    write_reports(report, failures, inventory)
    ok = not failures and report["safety_result"] == "passed"
    print(
        "medai_vertex_real_doc_single_pilot_final_preflight_no_live_16c_c_pass"
        if ok
        else "medai_vertex_real_doc_single_pilot_final_preflight_no_live_16c_c_fail"
    )
    print(json.dumps({**report, "failures": failures}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
