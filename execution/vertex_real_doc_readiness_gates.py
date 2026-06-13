"""No-live real-document readiness-gate framework for Vertex routing (15Z-A).

Real-document Vertex routing is BLOCKED by default. This module deterministically
evaluates whether a payload *could ever* be considered for a future, separately
authorized real-document live call — and records exactly why routing is blocked.

It performs NO provider call, sets NO live gate, sends NO real document anywhere,
makes NO active MKB write, and never auto-accepts. Even when every modeled future
gate is simulated as passing, the status is ``READY_FOR_FUTURE_AUTHORIZATION_ONLY``
— never live authorization.

Reports are sanitized: blocked/private cases never include raw payload text, only
a reason and a content fingerprint (sha256 prefix).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


# Payload provenance classes.
PROVENANCE_SYNTHETIC = "synthetic_calibration_fixture"
PROVENANCE_REDACTED_REAL_LIKE = "redacted_real_like_fixture"
PROVENANCE_REAL_PRIVATE = "real_private_document"
PROVENANCE_UNKNOWN = "unknown_provenance_payload"

# The 14 modeled readiness gates (all must be future-satisfied before any real
# live call could even be proposed; this block never authorizes one).
READINESS_GATES = (
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
)

# Markers that force a BLOCK regardless of declared provenance.
_PII_MARKERS = ("pii", "DOB", "MRN", "SSN", "patient_name", "date_of_birth")
_RAW_PDF_MARKERS = ("raw_pdf", ".pdf", "raw_image", ".png", ".jpg")
_OCR_PRIVATE_MARKERS = ("ocr_private", "raw OCR", "private_ocr_payload")

_SANITIZE_PATTERNS = [
    (re.compile(r"ya29\.[0-9A-Za-z_\-\.]+"), "<redacted-credential>"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{6,}"), "<redacted-credential>"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"), "<redacted-credential>"),
    (re.compile(r"[A-Za-z]:\\[^\s'\"]*"), "<redacted-path>"),
    (re.compile(r"/(?:home|Users)/[^\s'\"]*"), "<redacted-path>"),
]


@dataclass(frozen=True)
class ReadinessPayloadClassification:
    declared_provenance: str
    effective_provenance: str
    is_synthetic: bool
    is_redacted_real_like: bool
    is_real_private: bool
    is_unknown_provenance: bool
    forced_block_marker: str
    content_fingerprint: str


@dataclass(frozen=True)
class ReadinessGateResult:
    gate: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class ReadinessEvaluation:
    case_id: str
    payload_classification: str
    readiness_status: str
    blocked: bool
    block_reasons: list[str]
    gates_passed: list[str]
    gates_failed: list[str]
    live_call_allowed: bool
    external_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    sanitized_report_only: bool
    raw_payload_in_report: bool
    classification_detail: dict[str, Any] = field(default_factory=dict)
    gate_results: list[dict[str, Any]] = field(default_factory=list)


def _fingerprint(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def sanitize_readiness_report_payload(value: Any) -> Any:
    """Recursively strip any credential/path tokens from report payloads.

    Note: this module never copies a payload's raw text into reports for blocked
    or private cases — callers pass only reasons, fingerprints, and synthetic
    snippets. This sanitizer is a defense-in-depth final pass.
    """
    if isinstance(value, str):
        out = value
        for pattern, repl in _SANITIZE_PATTERNS:
            out = pattern.sub(repl, out)
        return out
    if isinstance(value, Mapping):
        return {k: sanitize_readiness_report_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_readiness_report_payload(v) for v in value]
    return value


def classify_payload_provenance(
    *,
    declared_provenance: str,
    content_marker: str = "",
) -> ReadinessPayloadClassification:
    declared = str(declared_provenance or PROVENANCE_UNKNOWN).strip()
    marker_text = f"{declared} {content_marker}".lower()
    raw_marker_text = f"{declared} {content_marker}"

    forced = ""
    if any(m.lower() in marker_text for m in _PII_MARKERS):
        forced = "pii_marker_present"
    elif any(m.lower() in marker_text for m in _RAW_PDF_MARKERS):
        forced = "raw_pdf_or_image_marker_present"
    elif any(m.lower() in marker_text for m in _OCR_PRIVATE_MARKERS):
        forced = "ocr_private_marker_present"

    known = {PROVENANCE_SYNTHETIC, PROVENANCE_REDACTED_REAL_LIKE, PROVENANCE_REAL_PRIVATE, PROVENANCE_UNKNOWN}
    effective = declared if declared in known else PROVENANCE_UNKNOWN
    # A forced marker downgrades effective provenance to real_private for blocking.
    if forced and effective in {PROVENANCE_SYNTHETIC, PROVENANCE_REDACTED_REAL_LIKE}:
        effective = PROVENANCE_REAL_PRIVATE

    return ReadinessPayloadClassification(
        declared_provenance=declared,
        effective_provenance=effective,
        is_synthetic=effective == PROVENANCE_SYNTHETIC,
        is_redacted_real_like=effective == PROVENANCE_REDACTED_REAL_LIKE,
        is_real_private=effective == PROVENANCE_REAL_PRIVATE,
        is_unknown_provenance=effective == PROVENANCE_UNKNOWN,
        forced_block_marker=forced,
        # Fingerprint only — never the raw marker text — for private/unknown cases.
        content_fingerprint=_fingerprint(raw_marker_text),
    )


def evaluate_vertex_real_doc_readiness(
    *,
    case_id: str,
    declared_provenance: str,
    content_marker: str = "",
    human_authorization_present: bool = False,
    billing_cost_cap_ack_present: bool = False,
    active_write_requested: bool = False,
    auto_accept_requested: bool = False,
    contains_medication_fact: bool = False,
    medication_safety_gate_satisfied: bool = False,
    future_gates_simulated_pass: bool = False,
) -> ReadinessEvaluation:
    """Deterministically evaluate real-doc readiness. Always non-live, never authorizes a live call."""
    classification = classify_payload_provenance(
        declared_provenance=declared_provenance, content_marker=content_marker
    )

    block_reasons: list[str] = []
    gate_results: list[ReadinessGateResult] = []

    def gate(name: str, passed: bool, reason: str) -> None:
        gate_results.append(ReadinessGateResult(gate=name, passed=passed, reason=reason))
        if not passed:
            block_reasons.append(f"{name}: {reason}")

    # Gate 1: external routing is blocked by default for anything that is not a
    # synthetic or redacted-real-like NO-LIVE fixture.
    routing_ok = classification.is_synthetic or classification.is_redacted_real_like
    gate(
        "real_doc_external_routing_default_block",
        routing_ok,
        "non_synthetic_non_redacted_payload_blocked_by_default" if not routing_ok else "no_live_dry_run_fixture_only",
    )

    # Forced-marker gates (PII / raw pdf-image / OCR private).
    gate("pii_stripping_proof_required", classification.forced_block_marker != "pii_marker_present",
         "pii_marker_present_requires_stripping_proof" if classification.forced_block_marker == "pii_marker_present" else "no_pii_marker")
    gate("no_raw_private_payload_in_reports_required", classification.forced_block_marker != "raw_pdf_or_image_marker_present",
         "raw_pdf_or_image_payload_blocked" if classification.forced_block_marker == "raw_pdf_or_image_marker_present" else "no_raw_pdf_image_marker")
    gate("pii_vault_isolation_required", classification.forced_block_marker != "ocr_private_marker_present",
         "ocr_private_payload_blocked" if classification.forced_block_marker == "ocr_private_marker_present" else "no_ocr_private_marker")

    # Provenance-derived future gates (modeled; satisfied only by simulation flag
    # AND only for synthetic/redacted no-live fixtures).
    sim = future_gates_simulated_pass and routing_ok
    gate("synthetic_to_real_adapter_dry_run_required", sim, "dry_run_simulated_pass" if sim else "dry_run_not_proven")
    gate("redacted_real_like_fixture_replay_required", sim, "replay_simulated_pass" if sim else "replay_not_proven")
    gate("operator_review_queue_handoff_required", sim, "handoff_simulated_pass" if sim else "handoff_not_proven")
    gate("real_doc_refusal_path_required", sim, "refusal_path_simulated_pass" if sim else "refusal_path_not_proven")
    gate("dedicated_future_real_doc_live_gate_required", sim, "future_gate_simulated_present" if sim else "dedicated_future_gate_absent")

    # Hard human/operator/billing gates.
    gate("human_authorization_required_for_any_real_live_call", bool(human_authorization_present),
         "human_authorization_present" if human_authorization_present else "human_authorization_absent")
    gate("billing_cost_cap_ack_required", bool(billing_cost_cap_ack_present),
         "billing_cost_cap_acknowledged" if billing_cost_cap_ack_present else "billing_cost_cap_ack_absent")

    # Hard safety invariants (must NEVER be requested-true).
    gate("no_active_mkb_write_required", not active_write_requested,
         "active_write_requested_blocked" if active_write_requested else "no_active_write_requested")
    gate("no_auto_accept_required", not auto_accept_requested,
         "auto_accept_requested_blocked" if auto_accept_requested else "no_auto_accept_requested")

    # Medication safety non-bypass.
    med_ok = (not contains_medication_fact) or medication_safety_gate_satisfied
    gate("medication_safety_non_bypass_required_if_medication_facts_present", med_ok,
         "medication_safety_gate_satisfied_or_no_medication" if med_ok else "medication_fact_without_safety_gate_blocked")

    gates_passed = [g.gate for g in gate_results if g.passed]
    gates_failed = [g.gate for g in gate_results if not g.passed]
    all_gates_pass = not gates_failed

    if all_gates_pass:
        # Even with all modeled gates simulated-passing, this block NEVER authorizes a live call.
        readiness_status = "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
        blocked = True  # still blocked for live routing in this block
        block_reasons.append("live_real_doc_call_not_authorized_in_this_block")
    elif classification.is_synthetic or classification.is_redacted_real_like:
        readiness_status = "BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES"
        blocked = True
    else:
        readiness_status = "BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY"
        blocked = True

    # No-live dry-run allowance flag (synthetic/redacted fixtures may be replayed
    # offline) — distinct from any live authorization.
    no_live_dry_run_allowed = classification.is_synthetic or classification.is_redacted_real_like

    classification_detail = asdict(classification)
    return ReadinessEvaluation(
        case_id=case_id,
        payload_classification=classification.effective_provenance,
        readiness_status=readiness_status,
        blocked=blocked,
        block_reasons=block_reasons,
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        live_call_allowed=False,
        external_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        sanitized_report_only=True,
        raw_payload_in_report=False,
        classification_detail={
            **classification_detail,
            "no_live_dry_run_allowed": no_live_dry_run_allowed,
        },
        gate_results=[asdict(g) for g in gate_results],
    )


def build_real_doc_refusal_record(evaluation: ReadinessEvaluation) -> dict[str, Any]:
    """Sanitized refusal record — reasons + fingerprint only, never raw payload."""
    return {
        "case_id": evaluation.case_id,
        "payload_classification": evaluation.payload_classification,
        "content_fingerprint": evaluation.classification_detail.get("content_fingerprint", ""),
        "readiness_status": evaluation.readiness_status,
        "blocked": evaluation.blocked,
        "block_reasons": list(evaluation.block_reasons),
        "live_call_allowed": False,
        "external_api_used": False,
        "active_write_allowed": False,
        "auto_accept_allowed": False,
        "review_required": True,
        "raw_payload_in_report": False,
        "sanitized_report_only": True,
    }


def evaluation_to_public_dict(evaluation: ReadinessEvaluation) -> dict[str, Any]:
    return sanitize_readiness_report_payload(asdict(evaluation))


__all__ = [
    "PROVENANCE_SYNTHETIC",
    "PROVENANCE_REDACTED_REAL_LIKE",
    "PROVENANCE_REAL_PRIVATE",
    "PROVENANCE_UNKNOWN",
    "READINESS_GATES",
    "ReadinessPayloadClassification",
    "ReadinessGateResult",
    "ReadinessEvaluation",
    "classify_payload_provenance",
    "evaluate_vertex_real_doc_readiness",
    "build_real_doc_refusal_record",
    "sanitize_readiness_report_payload",
    "evaluation_to_public_dict",
]
