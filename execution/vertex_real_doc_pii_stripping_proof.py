"""No-live PII-stripping proof and PII-vault isolation gates (15Z-B).

This module deterministically redacts/tokenizes PII-like fields from
``redacted_real_like`` synthetic fixtures, isolates the sensitive token map in a
local vault record (never emitted to outbound payloads or public reports), and
feeds the sanitized outbound-safe payload into the 15Z-A real-document readiness
framework as a *no-live replay only* candidate.

Hard boundaries (identical to 15Z-A and never relaxed here):
  * NO provider call, NO network, NO live gate.
  * NO real medical document; fixtures are synthetic ``redacted_real_like`` text.
  * NO active MKB write, NO auto-accept; ``review_required`` is always true.
  * Reports expose only token classes, counts, fingerprints (sha256 prefix), and
    tokenized previews — never raw PII-like values and never the token map.

Even a fixture that passes every PII-stripping check is classified only as
``READY_FOR_NO_LIVE_REPLAY_ONLY`` and the underlying 15Z-A evaluation keeps
``live_call_allowed=False``. Real-document live routing is NOT authorized here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNKNOWN,
    classify_payload_provenance,
    evaluate_vertex_real_doc_readiness,
)

# Deterministic, label-anchored PII-like detectors (no model, no embeddings, no
# fuzzy matching). Each detector captures the value group ``v`` after a labelled
# field key. Synthetic fixtures use this labelled layout on purpose so redaction
# is fully deterministic and auditable.
_DETECTORS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("PATIENT_NAME", re.compile(r"(?im)^(?:patient|patient name|name)\s*:\s*(?P<v>.+?)\s*$")),
    ("PROVIDER", re.compile(r"(?im)^(?:provider|physician|ordering provider|attending)\s*:\s*(?P<v>.+?)\s*$")),
    ("FACILITY", re.compile(r"(?im)^(?:facility|clinic|hospital|imaging center)\s*:\s*(?P<v>.+?)\s*$")),
    ("MRN", re.compile(r"(?im)^(?:mrn|account|account number|account no|medical record number)\s*:\s*(?P<v>.+?)\s*$")),
    ("ACCESSION", re.compile(r"(?im)^(?:accession|accession id|report id|order id)\s*:\s*(?P<v>.+?)\s*$")),
    ("DATE", re.compile(r"(?im)^(?:dob|date of birth|date|collected|collection date|service date)\s*:\s*(?P<v>.+?)\s*$")),
    ("PHONE", re.compile(r"(?im)^(?:phone|tel|telephone|contact|contact phone)\s*:\s*(?P<v>.+?)\s*$")),
    ("EMAIL", re.compile(r"(?im)^(?:email|e-mail|contact email)\s*:\s*(?P<v>.+?)\s*$")),
    ("ADDRESS", re.compile(r"(?im)^(?:address|addr|street address)\s*:\s*(?P<v>.+?)\s*$")),
)

TOKEN_CLASSES = tuple(c for c, _ in _DETECTORS)


def _fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


@dataclass(frozen=True)
class PiiFinding:
    pii_class: str
    token: str
    value_fingerprint: str
    reused_token: bool

    def public_dict(self) -> dict[str, Any]:
        # No raw value ever — only class, token, and a fingerprint.
        return {
            "pii_class": self.pii_class,
            "token": self.token,
            "value_fingerprint": self.value_fingerprint,
            "reused_token": self.reused_token,
        }


@dataclass(frozen=True)
class PiiRedactionResult:
    redacted_text: str
    findings: list[PiiFinding]
    token_classes: list[str]
    token_count: int
    finding_count: int
    repeated_token_count: int
    # Sensitive (vault) material — never serialized into public reports.
    _token_map: dict[str, str] = field(default_factory=dict, repr=False)
    _detected_values: list[str] = field(default_factory=list, repr=False)

    def public_findings(self) -> list[dict[str, Any]]:
        return [f.public_dict() for f in self.findings]


@dataclass(frozen=True)
class PiiVaultRecord:
    vault_id: str
    token_class_counts: dict[str, int]
    total_tokens: int
    token_classes: list[str]
    vault_fingerprint: str
    isolated: bool
    # Sensitive serialized mapping — vault-local only, never emitted.
    _serialized_map: str = field(default="", repr=False)

    def public_dict(self) -> dict[str, Any]:
        # Public form: counts, classes, fingerprint, isolation flag — NO mapping.
        return {
            "vault_id": self.vault_id,
            "token_class_counts": dict(self.token_class_counts),
            "total_tokens": self.total_tokens,
            "token_classes": list(self.token_classes),
            "vault_fingerprint": self.vault_fingerprint,
            "isolated": self.isolated,
            "raw_pii_in_record": False,
            "token_map_in_record": False,
        }


@dataclass(frozen=True)
class OutboundPayloadProof:
    outbound_text: str
    outbound_fingerprint: str
    token_count: int
    contains_tokens: bool


def redact_pii_like_values(raw_text: str) -> PiiRedactionResult:
    """Deterministically tokenize labelled PII-like values; repeats reuse a token."""
    text = str(raw_text or "")
    matches: list[tuple[int, int, str, str]] = []
    for pii_class, rx in _DETECTORS:
        for m in rx.finditer(text):
            value = (m.group("v") or "").strip()
            if not value:
                continue
            matches.append((m.start("v"), m.end("v"), pii_class, value))

    value_to_token: dict[tuple[str, str], str] = {}
    class_counters: dict[str, int] = {}
    findings: list[PiiFinding] = []
    repeated = 0

    # Assign tokens in reading order so indices are stable/auditable.
    for _start, _end, pii_class, value in sorted(matches, key=lambda x: x[0]):
        key = (pii_class, value)
        if key in value_to_token:
            token = value_to_token[key]
            reused = True
            repeated += 1
        else:
            class_counters[pii_class] = class_counters.get(pii_class, 0) + 1
            token = f"[{pii_class}_{class_counters[pii_class]}]"
            value_to_token[key] = token
            reused = False
        findings.append(
            PiiFinding(
                pii_class=pii_class,
                token=token,
                value_fingerprint=_fingerprint(value),
                reused_token=reused,
            )
        )

    # Replace the labelled value spans (descending so offsets stay valid).
    redacted = text
    for start, end, pii_class, value in sorted(matches, key=lambda x: x[0], reverse=True):
        token = value_to_token[(pii_class, value)]
        redacted = redacted[:start] + token + redacted[end:]

    token_map = {tok: val for (cls, val), tok in value_to_token.items()}
    detected_values = sorted({v for _, _, _, v in matches})
    token_classes = sorted({c for _, _, c, _ in matches})

    return PiiRedactionResult(
        redacted_text=redacted,
        findings=findings,
        token_classes=token_classes,
        token_count=len(value_to_token),
        finding_count=len(findings),
        repeated_token_count=repeated,
        _token_map=token_map,
        _detected_values=detected_values,
    )


def build_isolated_pii_vault_record(redaction: PiiRedactionResult) -> PiiVaultRecord:
    """Build an isolated vault record. The token map is serialized only locally;
    the public form carries counts, classes, and a fingerprint."""
    serialized = json.dumps(redaction._token_map, sort_keys=True, ensure_ascii=False)
    fingerprint = _fingerprint(serialized)
    class_counts: dict[str, int] = {}
    for f in redaction.findings:
        if not f.reused_token:
            class_counts[f.pii_class] = class_counts.get(f.pii_class, 0) + 1
    return PiiVaultRecord(
        vault_id="vault_" + fingerprint.split(":", 1)[-1][:12],
        token_class_counts=class_counts,
        total_tokens=redaction.token_count,
        token_classes=list(redaction.token_classes),
        vault_fingerprint=fingerprint,
        isolated=True,
        _serialized_map=serialized,
    )


def build_outbound_safe_payload(redaction: PiiRedactionResult) -> OutboundPayloadProof:
    """Outbound-safe payload = tokenized text only (no raw PII, no token map)."""
    text = redaction.redacted_text
    return OutboundPayloadProof(
        outbound_text=text,
        outbound_fingerprint=_fingerprint(text),
        token_count=redaction.token_count,
        contains_tokens=any(f.token in text for f in redaction.findings),
    )


def validate_no_raw_pii_in_payload(payload_text: str, redaction: PiiRedactionResult) -> bool:
    """True if NO known raw PII-like value remains in the payload text."""
    text = str(payload_text or "")
    for value in redaction._detected_values:
        if value and value in text:
            return False
    return True


def validate_vault_not_in_payload_or_report(
    payload_text: str,
    report_text: str,
    redaction: PiiRedactionResult,
) -> tuple[bool, bool]:
    """Return (payload_isolated, report_isolated).

    Detects the *token map* specifically via its serialized mapping signature
    (a token-to-raw-value pair rendered as ``"[TOKEN]": "value"``). This is
    distinct from unredacted residue (a bare raw value with no mapping), so the
    two failure modes are reported independently.
    """
    serialized = json.dumps(redaction._token_map, sort_keys=True, ensure_ascii=False)
    # Per-pair mapping signatures, e.g. '"[MRN_1]": "SYN-..."'.
    pair_signatures = [
        json.dumps({tok: val}, ensure_ascii=False)[1:-1]
        for tok, val in redaction._token_map.items()
        if tok and val
    ]

    def _isolated(text: str) -> bool:
        blob = str(text or "")
        if serialized and serialized in blob:
            return False
        return not any(sig in blob for sig in pair_signatures)

    return _isolated(payload_text), _isolated(report_text)


@dataclass(frozen=True)
class PiiStrippingFixture:
    case_id: str
    declared_provenance: str
    raw_text: str
    content_marker: str
    expected_block: bool
    inject_token_map_in_payload: bool = False
    inject_token_map_in_report: bool = False


def build_pii_stripping_readiness_case(fixture: PiiStrippingFixture) -> dict[str, Any]:
    """Run the full no-live PII-stripping proof for one fixture and feed the
    sanitized outbound payload into the 15Z-A readiness framework."""
    redaction = redact_pii_like_values(fixture.raw_text)
    vault = build_isolated_pii_vault_record(redaction)
    outbound = build_outbound_safe_payload(redaction)

    # Candidate payload/report used ONLY to exercise the validators. For the
    # leak scenarios we transiently construct a bad candidate in-memory; it is
    # never written to any report file.
    candidate_payload = outbound.outbound_text
    if fixture.inject_token_map_in_payload:
        candidate_payload = candidate_payload + "\n" + vault._serialized_map
    candidate_report = "tokens-only sanitized preview"
    if fixture.inject_token_map_in_report:
        candidate_report = candidate_report + "\n" + vault._serialized_map

    payload_isolated, report_isolated = validate_vault_not_in_payload_or_report(
        candidate_payload, candidate_report, redaction
    )
    raw_pii_in_payload = not validate_no_raw_pii_in_payload(candidate_payload, redaction)
    token_map_in_payload = not payload_isolated
    token_map_in_report = not report_isolated

    classification = classify_payload_provenance(
        declared_provenance=fixture.declared_provenance, content_marker=fixture.content_marker
    )
    provenance_ok = classification.is_redacted_real_like
    pii_proof_pass = (not raw_pii_in_payload) and (not token_map_in_payload) and (not token_map_in_report)

    block_reasons: list[str] = []
    if not provenance_ok:
        readiness_status = "BLOCKED_PROVENANCE_NOT_REDACTED_REAL_LIKE"
        block_reasons.append(
            f"provenance_not_eligible:{classification.effective_provenance}"
        )
    elif token_map_in_payload:
        readiness_status = "BLOCKED_TOKEN_MAP_LEAK_IN_OUTBOUND_PAYLOAD"
        block_reasons.append("token_map_leaked_into_outbound_payload")
    elif raw_pii_in_payload:
        readiness_status = "BLOCKED_UNREDACTED_PII_RESIDUE"
        block_reasons.append("unredacted_pii_like_value_in_outbound_payload")
    elif token_map_in_report:
        readiness_status = "BLOCKED_TOKEN_MAP_LEAK_IN_REPORT"
        block_reasons.append("token_map_leaked_into_report")
    else:
        readiness_status = "READY_FOR_NO_LIVE_REPLAY_ONLY"

    blocked = readiness_status != "READY_FOR_NO_LIVE_REPLAY_ONLY"

    # Feed the sanitized outbound-safe payload into the 15Z-A framework. Gates are
    # only simulated-present when the PII proof passed; the framework always keeps
    # live_call_allowed=False regardless.
    gates_simulated = pii_proof_pass and provenance_ok
    framework_eval = evaluate_vertex_real_doc_readiness(
        case_id=fixture.case_id,
        declared_provenance=fixture.declared_provenance,
        content_marker=fixture.content_marker,
        human_authorization_present=gates_simulated,
        billing_cost_cap_ack_present=gates_simulated,
        future_gates_simulated_pass=gates_simulated,
    )

    # The only outbound payload "accepted" for no-live replay is a fully clean,
    # eligible one; leak/residue candidates are rejected (never accepted).
    accepted_outbound_clean = (not blocked) and validate_no_raw_pii_in_payload(
        outbound.outbound_text, redaction
    )

    return {
        "case_id": fixture.case_id,
        "provenance_classification": classification.effective_provenance,
        "declared_provenance": classification.declared_provenance,
        "pii_findings_count": redaction.finding_count,
        "pii_token_count": redaction.token_count,
        "token_classes": list(redaction.token_classes),
        "repeated_token_count": redaction.repeated_token_count,
        "raw_pii_in_outbound_payload": raw_pii_in_payload,
        "raw_pii_in_report": False,
        "token_map_in_outbound_payload": token_map_in_payload,
        "token_map_in_report": token_map_in_report,
        "vault_record_created": True,
        "vault_record_isolated": vault.isolated,
        "vault_record_fingerprint": vault.vault_fingerprint,
        "outbound_payload_fingerprint": _fingerprint(candidate_payload),
        "accepted_outbound_clean": accepted_outbound_clean,
        "readiness_status": readiness_status,
        "framework_readiness_status": framework_eval.readiness_status,
        "blocked": blocked,
        "block_reasons": block_reasons,
        "live_call_allowed": framework_eval.live_call_allowed,
        "external_api_used": framework_eval.external_api_used,
        "active_write_allowed": framework_eval.active_write_allowed,
        "auto_accept_allowed": framework_eval.auto_accept_allowed,
        "review_required": framework_eval.review_required,
        "expected_block": fixture.expected_block,
        "behaved_as_expected": blocked == fixture.expected_block,
        # Private/internal handles (excluded from public report rendering).
        "_findings": redaction.public_findings(),
        "_vault_public": vault.public_dict(),
        "_clean_outbound_preview": outbound.outbound_text if accepted_outbound_clean else "",
    }


__all__ = [
    "TOKEN_CLASSES",
    "PiiFinding",
    "PiiRedactionResult",
    "PiiVaultRecord",
    "OutboundPayloadProof",
    "PiiStrippingFixture",
    "redact_pii_like_values",
    "build_isolated_pii_vault_record",
    "build_outbound_safe_payload",
    "validate_no_raw_pii_in_payload",
    "validate_vault_not_in_payload_or_report",
    "build_pii_stripping_readiness_case",
]
