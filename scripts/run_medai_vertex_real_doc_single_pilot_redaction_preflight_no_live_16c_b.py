#!/usr/bin/env python3
"""Generate + validate the 16C-B no-live redaction/tokenization preflight package.

SYNTHETIC-ONLY. This block prepares the redaction/tokenization preflight for a future
one-document Vertex pilot: it generates synthetic fixtures, runs deterministic
local rule-based detection, tokenizes detected synthetic identifiers, and proves the
outbound payload preview contains tokens only (no raw synthetic identifiers) while the
token map stays private (never written to any public report).

Hard boundaries (never relaxed): no provider call; no Vertex/Gemini/Claude/OpenAI live
execution; no billing API call; no real/private document processing; no private corpus
read; no whole-corpus processing; no PDF/image/OCR processing; no MKB DB open; no
active MKB write; no auto-accept; no medical decision; no production queue mutation.
This block does NOT set the dedicated future live gate.

Stdlib only. No provider/billing/network/PDF-OCR/DB/MKB imports. No document-path CLI
argument. Synthetic fixture text is embedded below.
"""
from __future__ import annotations

import hashlib
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

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_16C_B"
REDACTION_PREFLIGHT_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_REDACTION_PREFLIGHT_NO_LIVE_16C_B.md"
TOKENIZATION_SPEC_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_TOKENIZATION_SPEC_16C_B.md"
SYNTHETIC_FIXTURE_SPEC_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_SYNTHETIC_FIXTURE_SPEC_16C_B.md"
OUTBOUND_SAFETY_RULES_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_OUTBOUND_PAYLOAD_SAFETY_RULES_16C_B.md"
TOKEN_VAULT_ISOLATION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_TOKEN_VAULT_ISOLATION_16C_B.md"
PRE_16D_PRIVACY_MATRIX_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_PRE_16D_PRIVACY_MATRIX_16C_B.md"
NEXT_DECISION_MD = DOC_DIR / "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_NEXT_DECISION_16C_B.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
REDACTION_MATRIX_MD = REPORT_DIR / "redaction_matrix.md"
SYNTHETIC_PAYLOAD_PREVIEW_JSON = REPORT_DIR / "synthetic_payload_preview.json"

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

# Deterministic, label-anchored detectors over SYNTHETIC fixture text only.
_DETECTORS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (cls, re.compile(rx, re.IGNORECASE | re.MULTILINE))
    for cls, rx in (
        ("PATIENT_NAME", r"^(?:patient name|patient|name)\s*:\s*(?P<v>.+?)\s*$"),
        ("DOB", r"^(?:dob|date of birth)\s*:\s*(?P<v>.+?)\s*$"),
        ("DATE", r"^(?:service date|collected|date)\s*:\s*(?P<v>.+?)\s*$"),
        ("ADDRESS", r"^(?:address|addr)\s*:\s*(?P<v>.+?)\s*$"),
        ("PHONE", r"^(?:phone|tel|telephone)\s*:\s*(?P<v>.+?)\s*$"),
        ("EMAIL", r"^(?:email|e-mail)\s*:\s*(?P<v>.+?)\s*$"),
        ("MRN", r"^(?:mrn|medical record number)\s*:\s*(?P<v>.+?)\s*$"),
        ("INSURANCE_ID", r"^(?:insurance id|insurance)\s*:\s*(?P<v>.+?)\s*$"),
        ("ACCOUNT_ID", r"^(?:account id|account number|account)\s*:\s*(?P<v>.+?)\s*$"),
        ("PROVIDER", r"^(?:ordering provider|provider|physician)\s*:\s*(?P<v>.+?)\s*$"),
        ("FACILITY", r"^(?:facility|clinic|hospital)\s*:\s*(?P<v>.+?)\s*$"),
        ("ACCESSION", r"^(?:accession id|accession|specimen id|specimen)\s*:\s*(?P<v>.+?)\s*$"),
        ("FILENAME", r"^(?:filename|file)\s*:\s*(?P<v>.+?)\s*$"),
        ("PATH", r"^(?:local path|path)\s*:\s*(?P<v>.+?)\s*$"),
        ("METADATA", r"^(?:embedded metadata|metadata|author)\s*:\s*(?P<v>.+?)\s*$"),
        ("OCR_ARTIFACT", r"^(?:ocr artifact|ocr)\s*:\s*(?P<v>.+?)\s*$"),
        ("FREETEXT_ID", r"^(?:free-text id|known as|nickname)\s*:\s*(?P<v>.+?)\s*$"),
        ("RARE_COMBO", r"^(?:rare combination|rare-combo)\s*:\s*(?P<v>.+?)\s*$"),
    )
)
TOKEN_CLASSES = tuple(c for c, _ in _DETECTORS)

FORBIDDEN_REPORT_TOKENS = (
    "ya29.", "AIza", "Bearer ", "Authorization" + ":", "access" + "_token",
    "refresh" + "_token", "private" + "_key", "application" + "_default" + "_credentials",
    "g" + "cloud", "C:\\", "/home/",
)

REQUIRED_PHRASES = (
    "no-live",
    "synthetic only",
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
    "token maps must never be included in outbound payload",
    "token maps must never appear in public reports",
    "redaction/tokenization preflight",
    "explicit operator approval",
    "cost cap",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "rollback",
    "16D is not started",
)

FORBIDDEN_PHRASES = (
    "16C-B authorizes live execution",
    "whole-corpus processing is allowed",
    "active MKB write is allowed",
    "auto-accept is allowed",
    "sandbox proves MedAI real-document readiness",
)


def _fp(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Synthetic fixtures (SYNTHETIC ONLY — obviously fake, never real patient data).
# ----------------------------------------------------------------------------
def synthetic_fixtures() -> list[dict[str, str]]:
    return [
        {
            "case_id": "synthetic_case_1_demographics",
            "text": (
                "Patient: Synthpat Alphaname\n"
                "DOB: 1990-01-01\n"
                "Address: 100 Synthetic Ave, Townsville, ST 00000\n"
                "Phone: +1-555-0100\n"
                "Email: synth.user@example-synth.test\n"
                "Note: Synthetic demographics block, all values fabricated."
            ),
        },
        {
            "case_id": "synthetic_case_2_identifiers",
            "text": (
                "MRN: MRN-SYN-0001\n"
                "Insurance ID: INS-SYN-0001\n"
                "Account ID: ACCT-SYN-0001\n"
                "Accession ID: ACC-SYN-0001\n"
                "Note: Synthetic identifier block."
            ),
        },
        {
            "case_id": "synthetic_case_3_care_team",
            "text": (
                "Provider: Dr Synthprovider Beta\n"
                "Facility: Synthville Test Clinic\n"
                "Service date: 2020-02-02\n"
                "Note: Synthetic care-team block."
            ),
        },
        {
            "case_id": "synthetic_case_4_file_metadata",
            "text": (
                "Filename: synthetic_note_04.txt\n"
                "Local path: fixtures/synthetic/note_04.txt\n"
                "Embedded metadata: synth-metadata-tag-04\n"
                "OCR artifact: synthetic-ocr-noise-04\n"
                "Note: Synthetic file/metadata block."
            ),
        },
        {
            "case_id": "synthetic_case_5_freetext_and_rare",
            "text": (
                "Known as: SynthNickFive\n"
                "Rare combination: 1990-01-01 plus ZIP 00000 synthetic combo\n"
                "Note: Synthetic free-text and rare-combination block."
            ),
        },
        {
            "case_id": "synthetic_case_6_mixed",
            "text": (
                "Patient: Testpat Gamma\n"
                "DOB: 1985-05-05\n"
                "MRN: MRN-SYN-0006\n"
                "Provider: Dr Synthprovider Delta\n"
                "Facility: Synthville Test Clinic\n"
                "Accession ID: ACC-SYN-0006\n"
                "Note: Synthetic mixed block, all values fabricated."
            ),
        },
    ]


def redact(text: str) -> tuple[str, list[dict[str, Any]], dict[str, str], list[str]]:
    """Deterministically tokenize labelled synthetic identifiers.

    Returns (tokenized_text, findings, token_map_private, raw_values).
    """
    matches: list[tuple[int, int, str, str]] = []
    for cls, rx in _DETECTORS:
        for m in rx.finditer(text):
            value = (m.group("v") or "").strip()
            if value:
                matches.append((m.start("v"), m.end("v"), cls, value))

    value_to_token: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    findings: list[dict[str, Any]] = []
    for _s, _e, cls, value in sorted(matches, key=lambda x: x[0]):
        key = (cls, value)
        if key not in value_to_token:
            counters[cls] = counters.get(cls, 0) + 1
            value_to_token[key] = f"[{cls}_{counters[cls]}]"
            findings.append({"pii_class": cls, "token": value_to_token[key], "value_fingerprint": _fp(value)})

    tokenized = text
    for s, e, cls, value in sorted(matches, key=lambda x: x[0], reverse=True):
        tokenized = tokenized[:s] + value_to_token[(cls, value)] + tokenized[e:]

    token_map_private = {tok: val for (cls, val), tok in value_to_token.items()}  # NEVER written
    raw_values = sorted({v for _s, _e, _c, v in matches})
    return tokenized, findings, token_map_private, raw_values


def _docs() -> dict[str, str]:
    preflight = """# MEDAI Vertex Real Document Single Pilot Redaction Preflight No-Live 16C-B

## Status

- This block is no-live.
- 16C-B does not authorize live execution.
- 16C-B does not set the live gate.
- 16D is not started.
- All fixtures are synthetic only; no real/private document may be used.

## Hard Boundaries

This block enforces: no provider call; no Vertex live execution; no Gemini live
execution; no Claude/OpenAI live execution; no billing API call; no real/private
document processing; no private corpus read; no corpus processing; no PDF/image/OCR
processing; no MKB write; no auto-accept; no medical decision; no production queue
mutation.

## Redaction/Tokenization Preflight Requirement

A redaction/tokenization preflight must run before any future outbound payload is
assembled. The preflight must detect and block or tokenize at least: patient names,
DOB, dates, addresses, phone numbers, email addresses, MRN, insurance IDs, account
IDs, provider names, facility names, specimen/accession IDs, filenames and local
paths, embedded metadata, OCR artifacts, token maps, free-text identifiers, and rare
combinations that could re-identify a person.

## Pre-16D Requirements

- Explicit operator approval is required before any later 16D.
- Redaction/tokenization proof is required before any later 16D.
- Cost cap proof is required before any later 16D.
- One-document limit and one-call limit are required before any later 16D.
- Stop-on-first-failure is required before any later 16D.
- Rollback and failure plan are required before any later 16D.

## GCP Synthetic Sandbox Separation

The GCP synthetic sandbox evidence proves only environment, authentication, and the
request path. It is separate environment evidence only and does not prove MedAI
real-document readiness.
"""
    tokenization_spec = """# MEDAI Vertex Real Document Single Pilot Tokenization Spec 16C-B

## Status

This spec is no-live and synthetic only. No provider call, no Vertex live execution,
and no billing API call occur. 16D is not started.

## Token Scheme

- Each detected synthetic identifier is replaced with a stable token of the form
  `[CLASS_n]` (for example `[PATIENT_NAME_1]`).
- Repeated identical values map deterministically to the same token.
- Tokens are reversible only via the local token map, which is never sent and never
  reported.

## Covered Classes

Patient names, DOB, dates, addresses, phone numbers, email addresses, MRN, insurance
IDs, account IDs, provider names, facility names, specimen/accession IDs, filenames,
local paths, embedded metadata, OCR artifacts, free-text identifiers, and rare
combinations.

## Boundaries

No real/private document processing; no private corpus read; no corpus processing; no
PDF/image/OCR processing; no MKB write; no auto-accept; no medical decision; no
production queue mutation.
"""
    fixture_spec = """# MEDAI Vertex Real Document Single Pilot Synthetic Fixture Spec 16C-B

## Status

All fixtures are synthetic only. No real/private document may be used. This block is
no-live; no provider call and no billing API call occur. 16D is not started.

## Fixture Rules

- Every value is fabricated and obviously synthetic.
- No real names, real DOBs, real addresses, real provider names, real facility names,
  real MRNs, real insurance IDs, real file paths, or real patient data are used.
- Fixtures exist only to exercise the redaction/tokenization preflight locally.
- No PDF/image/OCR processing; fixtures are plain synthetic text.

## Boundaries

No real/private document processing; no private corpus read; no corpus processing; no
MKB write; no auto-accept; no medical decision; no production queue mutation.
"""
    outbound_rules = """# MEDAI Vertex Real Document Single Pilot Outbound Payload Safety Rules 16C-B

## Status

No-live, synthetic only. No provider call, no Vertex live execution, no billing API
call. 16D is not started.

## Outbound Payload Rules

- The outbound payload must contain tokens only; no raw identifier may appear.
- The outbound payload must pass the redaction/tokenization preflight before assembly.
- Token maps must never be included in outbound payload.
- Any raw identifier in the outbound payload is a hard NO-GO and triggers
  stop-on-first-failure.
- Future request shape is limited to tokenized content; no provider/MedAI metadata is
  attached.

## Boundaries

No real/private document processing; no private corpus read; no corpus processing; no
PDF/image/OCR processing; no MKB write; no auto-accept; no medical decision; no
production queue mutation.
"""
    vault_isolation = """# MEDAI Vertex Real Document Single Pilot Token Vault Isolation 16C-B

## Status

No-live, synthetic only. 16D is not started. No provider call and no billing API call
occur.

## Token Vault Rules

- Token maps must never be included in outbound payload.
- Token maps must never appear in public reports.
- Token maps must be isolated from any provider-facing payload.
- Any token-map leak is a hard NO-GO.
- Restoration of tokens is local-only and is not part of this block.

## Reporting

Public reports contain only safe summary counts, token labels, and fingerprints —
never raw synthetic identifiers and never token map values.

## Boundaries

No MKB write; no auto-accept; no medical decision; no production queue mutation.
"""
    privacy_matrix = f"""# MEDAI Vertex Real Document Single Pilot Pre-16D Privacy Matrix 16C-B

| Privacy requirement | Status (this block) | Required before any future live call |
| --- | --- | --- |
| Redaction/tokenization preflight | demonstrated on synthetic fixtures | yes |
| Outbound payload contains tokens only | demonstrated | yes |
| Token maps excluded from outbound payload | enforced | yes |
| Token maps excluded from public reports | enforced | yes |
| Explicit operator approval | required before 16D | yes |
| Cost cap | required before 16D | yes |
| One-document limit | required before 16D | yes |
| One-call limit | required before 16D | yes |
| Stop-on-first-failure | required before 16D | yes |
| Rollback / failure plan | required before 16D | yes |
| Dedicated future live gate `{DEDICATED_GATE_NAME}` | not set / not active | only in 16D, separately authorized |
| GCP synthetic sandbox = environment/auth/request path only | acknowledged | yes |

This block is no-live, synthetic only, and does not set the live gate. 16D is not
started.
"""
    next_decision = """# MEDAI Vertex Real Document Single Pilot Next Decision 16C-B

## Option A: Remain No-Live

Continue no-live synthetic preparation without executing anything.

## Option B: Proceed To Final Preflight (16C-C)

Prepare the final no-live preflight package next. This remains no-live and does not
set the live gate.

## Option C: Pause/Freeze

Pause and keep the 15Z/16A/16B/16C-A/16C-B governance state frozen.

## Recommended Next

Recommended next: user/operator decision, not automatic live execution. 16D is not
started.
"""
    return {
        "preflight": preflight,
        "tokenization_spec": tokenization_spec,
        "fixture_spec": fixture_spec,
        "outbound_rules": outbound_rules,
        "vault_isolation": vault_isolation,
        "privacy_matrix": privacy_matrix,
        "next_decision": next_decision,
    }


def write_docs(docs: dict[str, str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    REDACTION_PREFLIGHT_MD.write_text(docs["preflight"], encoding="utf-8")
    TOKENIZATION_SPEC_MD.write_text(docs["tokenization_spec"], encoding="utf-8")
    SYNTHETIC_FIXTURE_SPEC_MD.write_text(docs["fixture_spec"], encoding="utf-8")
    OUTBOUND_SAFETY_RULES_MD.write_text(docs["outbound_rules"], encoding="utf-8")
    TOKEN_VAULT_ISOLATION_MD.write_text(docs["vault_isolation"], encoding="utf-8")
    PRE_16D_PRIVACY_MATRIX_MD.write_text(docs["privacy_matrix"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def _gate_environment_active() -> bool:
    value = os.environ.get(DEDICATED_GATE_NAME)
    return value is not None and value.strip() in ACTIVE_VALUES


def run_preflight() -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Run redaction/tokenization over synthetic fixtures.

    Returns (public_preview_cases, all_raw_values, aggregate_token_map_private).
    The token map is returned only so the caller can verify it is NOT in any report.
    """
    preview_cases: list[dict[str, Any]] = []
    all_raw: list[str] = []
    token_map_private: dict[str, str] = {}
    for fx in synthetic_fixtures():
        tokenized, findings, tmap, raw_values = redact(fx["text"])
        all_raw.extend(raw_values)
        token_map_private.update(tmap)
        preview_cases.append(
            {
                "case_id": fx["case_id"],
                "tokenized_text": tokenized,  # tokens only; raw values replaced
                "token_classes": sorted({f["pii_class"] for f in findings}),
                "token_count": len({f["token"] for f in findings}),
                "finding_count": len(findings),
                "outbound_fingerprint": _fp(tokenized),
            }
        )
    return preview_cases, sorted(set(all_raw)), token_map_private


def evaluate() -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    docs = _docs()
    write_docs(docs)
    failures: list[str] = []

    required_paths = {
        "preflight": REDACTION_PREFLIGHT_MD,
        "tokenization_spec": TOKENIZATION_SPEC_MD,
        "fixture_spec": SYNTHETIC_FIXTURE_SPEC_MD,
        "outbound_rules": OUTBOUND_SAFETY_RULES_MD,
        "vault_isolation": TOKEN_VAULT_ISOLATION_MD,
        "privacy_matrix": PRE_16D_PRIVACY_MATRIX_MD,
        "next_decision": NEXT_DECISION_MD,
    }
    for name, path in required_paths.items():
        if not path.exists():
            failures.append(f"required_doc_missing:{name}")

    all_docs = re.sub(r"\s+", " ", "\n".join(docs.values()))
    all_docs_lower = all_docs.lower()
    for phrase in REQUIRED_PHRASES:
        if re.sub(r"\s+", " ", phrase).lower() not in all_docs_lower:
            failures.append(f"required_phrase_absent:{phrase}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in all_docs_lower:
            failures.append(f"forbidden_phrase_present:{phrase}")

    gate_active = _gate_environment_active()
    if gate_active:
        failures.append("future_live_gate_environment_active")

    preview_cases, raw_values, token_map_private = run_preflight()

    # The published preview is the per-case dict WITHOUT any token map.
    published_preview = {"cases": preview_cases}
    preview_blob = json.dumps(published_preview, ensure_ascii=False)

    # Raw-identifier leak: any raw synthetic value present in the outbound preview.
    raw_in_preview = sum(1 for v in raw_values if v and v in preview_blob)
    outbound_contains_raw = raw_in_preview > 0
    outbound_tokenized = all("[" in c["tokenized_text"] and "]" in c["tokenized_text"] for c in preview_cases)

    # Token-map leak: any token-map raw value or serialized map in any public report.
    serialized_map = json.dumps(token_map_private, sort_keys=True, ensure_ascii=False)

    report = {
        "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-REDACTION-PREFLIGHT-NO-LIVE-16C-B",
        "no_live": True,
        "synthetic_only": True,
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
        "future_live_gate_set": False,
        "future_live_gate_environment_active": gate_active,
        "redaction_preflight_executed": True,
        "tokenization_preflight_executed": True,
        "synthetic_fixture_count": len(preview_cases),
        "raw_identifier_leak_count": raw_in_preview,
        "token_map_written_to_public_report": False,  # finalized after report assembly
        "outbound_payload_contains_raw_identifier": outbound_contains_raw,
        "outbound_payload_tokenized": outbound_tokenized,
        "operator_approval_required_before_16d": True,
        "cost_cap_required_before_16d": True,
        "one_document_limit_required_before_16d": True,
        "one_call_limit_required_before_16d": True,
        "stop_on_first_failure_required_before_16d": True,
        "future_16d_not_started": True,
        "sandbox_treated_as_medai_validation": False,
        "privacy_result": "passed",
        "safety_result": "passed",
    }

    if report["synthetic_fixture_count"] != 6:
        failures.append(f"unexpected_fixture_count:{report['synthetic_fixture_count']}")
    if outbound_contains_raw:
        failures.append("outbound_payload_contains_raw_identifier")
    if not outbound_tokenized:
        failures.append("outbound_payload_not_tokenized")

    # Assemble the full set of public report blobs and verify NO raw value / token map leak.
    matrix_md = _redaction_matrix(report, preview_cases)
    impl_md = _implementation(report, failures, preview_cases)
    public_blobs = [
        json.dumps(report, ensure_ascii=False),
        preview_blob,
        matrix_md,
        impl_md,
    ]
    published = "\n".join(public_blobs)
    raw_in_reports = any(v and v in published for v in raw_values)
    token_map_in_reports = (serialized_map and serialized_map in published) or any(
        val and val in published for val in token_map_private.values()
    )
    report["token_map_written_to_public_report"] = bool(token_map_in_reports)
    if raw_in_reports:
        failures.append("raw_identifier_in_public_report")
    if token_map_in_reports:
        failures.append("token_map_in_public_report")
    if any(tok in published for tok in FORBIDDEN_REPORT_TOKENS):
        failures.append("forbidden_credential_or_path_token_in_report")
    if not check_public_report_payload((published_preview, report)).passed:
        failures.append("privacy_check_failed")

    if failures:
        report["privacy_result"] = "failed" if any(
            f.startswith(("raw_identifier", "token_map", "privacy", "outbound_payload_contains"))
            for f in failures
        ) else report["privacy_result"]
        report["safety_result"] = "failed"

    return report, failures, preview_cases


def _redaction_matrix(report: dict[str, Any], preview_cases: list[dict[str, Any]]) -> str:
    case_rows = [
        f"| {c['case_id']} | {c['finding_count']} | {c['token_count']} | {', '.join(c['token_classes'])} |"
        for c in preview_cases
    ]
    keys = [
        "no_live", "synthetic_only", "redaction_preflight_executed", "tokenization_preflight_executed",
        "synthetic_fixture_count", "raw_identifier_leak_count", "token_map_written_to_public_report",
        "outbound_payload_contains_raw_identifier", "outbound_payload_tokenized",
        "future_live_gate_set", "future_live_gate_environment_active", "future_16d_not_started",
        "privacy_result", "safety_result",
    ]
    return "\n".join(
        [
            "# 16C-B redaction matrix",
            "",
            "| Synthetic case | Findings | Tokens | Classes |",
            "| --- | --- | --- | --- |",
            *case_rows,
            "",
            "| Field | Value |",
            "| --- | --- |",
            *[f"| {k} | `{report[k]}` |" for k in keys],
            "",
            f"Token classes covered: {', '.join(TOKEN_CLASSES)}.",
            "Tokens only ever appear as `[CLASS_n]`. Raw synthetic identifiers and the token",
            "map never appear in the outbound payload preview or in these reports.",
            "",
        ]
    )


def _implementation(report: dict[str, Any], failures: list[str], preview_cases: list[dict[str, Any]]) -> str:
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-REDACTION-PREFLIGHT-NO-LIVE-16C-B",
        "",
        f"## Status: **{status}**",
        "",
        "## Result",
        "",
        "- Ran a synthetic-only redaction/tokenization preflight over 6 synthetic fixtures.",
        "- Outbound payload preview contains tokens only; no raw synthetic identifier present.",
        "- Token map kept private; never written to any public report.",
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
        "- MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-FINAL-PREFLIGHT-NO-LIVE-16C-C — not started;",
        "  requires explicit user authorization. 16D is not started.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], failures: list[str], preview_cases: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(report, failures, preview_cases), encoding="utf-8")
    REDACTION_MATRIX_MD.write_text(_redaction_matrix(report, preview_cases), encoding="utf-8")
    SYNTHETIC_PAYLOAD_PREVIEW_JSON.write_text(
        json.dumps({"cases": preview_cases}, indent=2), encoding="utf-8"
    )


def main() -> int:
    report, failures, preview_cases = evaluate()
    write_reports(report, failures, preview_cases)
    ok = not failures and report["safety_result"] == "passed"
    print(
        "medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b_pass"
        if ok
        else "medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b_fail"
    )
    print(json.dumps({**report, "failures": failures}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
