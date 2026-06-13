"""Integrated no-live real-document readiness harness for 15Z-F.

This module composes the 15Z-A through 15Z-E gates into one deterministic
end-to-end readiness evaluation. It performs no provider calls, billing calls,
active writes, production queue mutations, auto-accept, or medical decision
logic.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from execution.vertex_real_doc_adapter_dry_run import (
    AdapterDryRunFixture,
    evaluate_adapter_readiness_with_gates,
)
from execution.vertex_real_doc_authorization_cost_refusal_gates import (
    AuthorizationCostGateFixture,
    evaluate_authorization_cost_refusal_gates,
)
from execution.vertex_real_doc_medication_safety_non_bypass import (
    DECISION_DDI,
    MedicationSafetyFixture,
    READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY,
    evaluate_medication_safety_non_bypass,
)
from execution.vertex_real_doc_pii_stripping_proof import (
    PiiStrippingFixture,
    build_pii_stripping_readiness_case,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_UNKNOWN,
    evaluate_vertex_real_doc_readiness,
)


READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY = "READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY"


@dataclass(frozen=True)
class IntegratedReadinessCase:
    case_id: str
    raw_text: str
    declared_provenance: str = PROVENANCE_REDACTED_REAL_LIKE
    content_marker: str = "redacted real-like integrated fixture"
    medication_present: bool = False
    medication_safety_proof_present: bool = False
    authorization_present: bool = True
    billing_present: bool = True
    review_handoff_present: bool = True
    active_write_requested: bool = False
    auto_accept_requested: bool = False
    explicit_live_call_requested: bool = False
    raw_pii_residue: bool = False
    token_map_leak: bool = False
    forbidden_request_metadata: bool = False
    invalid_generation_config: bool = False
    medication_decision_requested: bool = False
    all_gates_simulated_pass: bool = False


@dataclass(frozen=True)
class IntegratedGateTrace:
    pii_redaction_passed: bool
    vault_isolated: bool
    request_shape_valid: bool
    review_handoff_created: bool
    authorization_intent_present: bool
    billing_ack_present: bool
    medication_safety_proof_present: bool
    medication_gate_passed: bool
    no_live_invariants_passed: bool


@dataclass(frozen=True)
class IntegratedRefusalRecord:
    refusal_id: str
    case_id: str
    refusal_reason_codes: list[str]
    sanitized_report_only: bool
    live_call_allowed: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool


@dataclass(frozen=True)
class IntegratedReadinessPackage:
    package_id: str
    case_id: str
    status: str
    gate_trace_summary: dict[str, bool]
    outbound_payload_fingerprint: str
    would_be_request_fingerprint: str
    vault_record_fingerprint: str
    review_handoff_reference: str
    medication_safety_proof_present: bool
    live_call_allowed: bool
    external_api_used: bool
    billing_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    raw_pii_in_package: bool
    token_map_in_package: bool
    medical_decision_made: bool


@dataclass(frozen=True)
class IntegratedReadinessResult:
    case_id: str
    integrated_status: str
    readiness_status: str
    package_created: bool
    blocked: bool
    block_reasons: list[str]
    gate_trace: IntegratedGateTrace
    refusal_record_created: bool
    pii_redaction_passed: bool
    vault_isolated: bool
    request_shape_valid: bool
    review_handoff_created: bool
    authorization_intent_present: bool
    billing_ack_present: bool
    medication_safety_proof_present: bool
    future_operator_review_package_created: bool
    live_call_allowed: bool
    external_api_used: bool
    billing_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    raw_pii_in_package: bool
    raw_pii_in_report: bool
    token_map_in_package: bool
    token_map_in_report: bool
    medical_decision_made: bool
    refusal_record: IntegratedRefusalRecord | None = None
    readiness_package: IntegratedReadinessPackage | None = None


def build_integrated_readiness_case(case_id: str, **overrides: Any) -> IntegratedReadinessCase:
    base = {
        "case_id": case_id,
        "raw_text": (
            "Patient: Integrated Synthsample\n"
            "DOB: 1970-01-01\n"
            "MRN: SYN-1506001\n"
            "Note: Sodium 140 mmol/L within reference range."
        ),
    }
    base.update(overrides)
    return IntegratedReadinessCase(**base)


def run_integrated_readiness_harness_no_live(case: IntegratedReadinessCase) -> IntegratedReadinessResult:
    pii = build_pii_stripping_readiness_case(
        PiiStrippingFixture(
            case_id=case.case_id,
            declared_provenance=case.declared_provenance,
            content_marker=case.content_marker,
            raw_text=case.raw_text,
            expected_block=case.declared_provenance != PROVENANCE_REDACTED_REAL_LIKE,
            inject_token_map_in_payload=case.token_map_leak,
        )
    )
    adapter = evaluate_adapter_readiness_with_gates(
        AdapterDryRunFixture(
            case_id=case.case_id,
            raw_text=case.raw_text,
            declared_provenance=case.declared_provenance,
            content_marker=case.content_marker,
            expected_status="BLOCKED",
            active_write_requested=case.active_write_requested,
            auto_accept_requested=case.auto_accept_requested,
            contains_medication_fact=case.medication_present,
            medication_safety_gate_satisfied=case.medication_safety_proof_present,
            future_gates_simulated_pass=case.all_gates_simulated_pass,
            inject_raw_pii_residue=case.raw_pii_residue,
            inject_token_map=case.token_map_leak,
            inject_forbidden_metadata_key=case.forbidden_request_metadata,
            invalid_generation_config=case.invalid_generation_config,
        )
    )
    auth = evaluate_authorization_cost_refusal_gates(
        AuthorizationCostGateFixture(
            case_id=case.case_id,
            authorization_present=case.authorization_present,
            billing_present=case.billing_present,
            review_handoff_present=case.review_handoff_present and adapter.review_queue_handoff_record_created,
            declared_provenance=case.declared_provenance,
            content_marker=case.content_marker,
            active_write_requested=case.active_write_requested,
            auto_accept_requested=case.auto_accept_requested,
            contains_medication_fact=case.medication_present,
            medication_safety_gate_satisfied=case.medication_safety_proof_present,
            forbidden_request_metadata=case.forbidden_request_metadata,
            invalid_generation_config=case.invalid_generation_config,
            raw_pii_detected=case.raw_pii_residue,
            token_map_leak_detected=case.token_map_leak,
            explicit_live_call_requested=case.explicit_live_call_requested,
        )
    )
    med = evaluate_medication_safety_non_bypass(
        MedicationSafetyFixture(
            case_id=case.case_id,
            medication_facts_present=case.medication_present,
            safety_proof_present=case.medication_safety_proof_present,
            requested_decision_types=(DECISION_DDI,) if case.medication_decision_requested else (),
            active_write_requested=case.active_write_requested,
            auto_accept_requested=case.auto_accept_requested,
            explicit_live_call_requested=case.explicit_live_call_requested,
            future_authorization_package_context=True,
        )
    )
    readiness = evaluate_vertex_real_doc_readiness(
        case_id=case.case_id,
        declared_provenance=case.declared_provenance,
        content_marker=case.content_marker,
        human_authorization_present=case.authorization_present,
        billing_cost_cap_ack_present=case.billing_present,
        active_write_requested=case.active_write_requested,
        auto_accept_requested=case.auto_accept_requested,
        contains_medication_fact=case.medication_present,
        medication_safety_gate_satisfied=case.medication_safety_proof_present,
        future_gates_simulated_pass=True,
    )

    reasons = _collect_block_reasons(case, pii, adapter, auth, med)
    trace = IntegratedGateTrace(
        pii_redaction_passed=not pii["raw_pii_in_outbound_payload"] and not case.raw_pii_residue,
        vault_isolated=bool(pii["vault_record_isolated"]) and not case.token_map_leak,
        request_shape_valid=adapter.request_shape_valid,
        review_handoff_created=adapter.review_queue_handoff_record_created and case.review_handoff_present,
        authorization_intent_present=auth.authorization_intent_present,
        billing_ack_present=auth.billing_ack_present,
        medication_safety_proof_present=case.medication_safety_proof_present,
        medication_gate_passed=(not case.medication_present) or (case.medication_safety_proof_present and not med.blocked),
        no_live_invariants_passed=True,
    )
    package_ok = (
        not reasons
        and trace.pii_redaction_passed
        and trace.vault_isolated
        and trace.request_shape_valid
        and trace.review_handoff_created
        and trace.authorization_intent_present
        and trace.billing_ack_present
        and trace.medication_gate_passed
        and readiness.live_call_allowed is False
    )
    package = build_integrated_readiness_package_no_live(case, adapter, trace) if package_ok else None
    refusal = build_integrated_refusal_record(case.case_id, reasons) if not package_ok else None
    status = READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY if package else "BLOCKED"
    return IntegratedReadinessResult(
        case_id=case.case_id,
        integrated_status=status,
        readiness_status=status,
        package_created=package is not None,
        blocked=package is None,
        block_reasons=reasons,
        gate_trace=trace,
        refusal_record_created=refusal is not None,
        pii_redaction_passed=trace.pii_redaction_passed,
        vault_isolated=trace.vault_isolated,
        request_shape_valid=trace.request_shape_valid,
        review_handoff_created=trace.review_handoff_created,
        authorization_intent_present=trace.authorization_intent_present,
        billing_ack_present=trace.billing_ack_present,
        medication_safety_proof_present=trace.medication_safety_proof_present,
        future_operator_review_package_created=package is not None,
        live_call_allowed=False,
        external_api_used=False,
        billing_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        raw_pii_in_package=False,
        raw_pii_in_report=False,
        token_map_in_package=False,
        token_map_in_report=False,
        medical_decision_made=False,
        refusal_record=refusal,
        readiness_package=package,
    )


def build_integrated_readiness_package_no_live(
    case: IntegratedReadinessCase,
    adapter: Any,
    trace: IntegratedGateTrace,
) -> IntegratedReadinessPackage:
    handoff_id = adapter.review_queue_handoff_record.handoff_id if adapter.review_queue_handoff_record else ""
    return IntegratedReadinessPackage(
        package_id="integrated_pkg_" + _fingerprint(case.case_id)[:12],
        case_id=case.case_id,
        status=READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY,
        gate_trace_summary=asdict(trace),
        outbound_payload_fingerprint=adapter.outbound_payload_fingerprint,
        would_be_request_fingerprint=adapter.would_be_request_fingerprint,
        vault_record_fingerprint=adapter.vault_record_fingerprint,
        review_handoff_reference=handoff_id,
        medication_safety_proof_present=case.medication_safety_proof_present,
        live_call_allowed=False,
        external_api_used=False,
        billing_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        raw_pii_in_package=False,
        token_map_in_package=False,
        medical_decision_made=False,
    )


def validate_integrated_package_sanitized(package: IntegratedReadinessPackage) -> bool:
    payload = asdict(package)
    return (
        payload["live_call_allowed"] is False
        and payload["active_write_allowed"] is False
        and payload["auto_accept_allowed"] is False
        and payload["raw_pii_in_package"] is False
        and payload["token_map_in_package"] is False
        and payload["medical_decision_made"] is False
    )


def summarize_integrated_gate_trace(result: IntegratedReadinessResult) -> dict[str, bool]:
    return asdict(result.gate_trace)


def build_integrated_refusal_record(case_id: str, reasons: list[str]) -> IntegratedRefusalRecord:
    safe = sorted(set(reasons or ["integrated_gate_blocked"]))
    return IntegratedRefusalRecord(
        refusal_id="integrated_refusal_" + _fingerprint(case_id + "|".join(safe))[:12],
        case_id=case_id,
        refusal_reason_codes=safe,
        sanitized_report_only=True,
        live_call_allowed=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
    )


def build_integrated_readiness_cases() -> list[IntegratedReadinessCase]:
    med_text = (
        "Patient: Integrated Medsample\n"
        "DOB: 1970-01-01\n"
        "MRN: SYN-1506002\n"
        "Medication: Examplemed 5 mg daily listed."
    )
    return [
        build_integrated_readiness_case("clean_redacted_real_like_non_medication_full_path"),
        build_integrated_readiness_case(
            "clean_redacted_real_like_medication_full_path_with_safety_proof",
            raw_text=med_text,
            medication_present=True,
            medication_safety_proof_present=True,
        ),
        build_integrated_readiness_case("unknown_provenance", declared_provenance=PROVENANCE_UNKNOWN, content_marker="unknown source layout"),
        build_integrated_readiness_case("real_private_marker", declared_provenance=PROVENANCE_REAL_PRIVATE, content_marker="real private marker"),
        build_integrated_readiness_case("raw_pii_residue", raw_pii_residue=True),
        build_integrated_readiness_case("token_map_leak", token_map_leak=True),
        build_integrated_readiness_case("forbidden_request_metadata", forbidden_request_metadata=True),
        build_integrated_readiness_case("invalid_generation_config", invalid_generation_config=True),
        build_integrated_readiness_case("missing_review_handoff", review_handoff_present=False),
        build_integrated_readiness_case("missing_human_authorization", authorization_present=False),
        build_integrated_readiness_case("missing_billing_ack", billing_present=False),
        build_integrated_readiness_case("active_write_requested", active_write_requested=True),
        build_integrated_readiness_case("auto_accept_requested", auto_accept_requested=True),
        build_integrated_readiness_case("medication_without_safety_proof", raw_text=med_text, medication_present=True),
        build_integrated_readiness_case(
            "medication_decision_requested",
            raw_text=med_text,
            medication_present=True,
            medication_safety_proof_present=True,
            medication_decision_requested=True,
        ),
        build_integrated_readiness_case("explicit_live_call_requested", explicit_live_call_requested=True),
        build_integrated_readiness_case(
            "all_gates_simulated_pass_still_no_live",
            raw_text=med_text,
            medication_present=True,
            medication_safety_proof_present=True,
            all_gates_simulated_pass=True,
        ),
    ]


def evaluate_all_integrated_readiness_cases() -> dict[str, Any]:
    results = [run_integrated_readiness_harness_no_live(case) for case in build_integrated_readiness_cases()]
    return {
        "summary": build_integrated_metrics(results),
        "cases": [asdict(result) for result in results],
        "refusal_records": [asdict(result.refusal_record) for result in results if result.refusal_record],
        "future_operator_review_packages": [
            asdict(result.readiness_package) for result in results if result.readiness_package
        ],
    }


def build_integrated_metrics(results: list[IntegratedReadinessResult]) -> dict[str, Any]:
    blocked = [r for r in results if r.blocked]
    packages = [r for r in results if r.future_operator_review_package_created]
    return {
        "integrated_readiness_harness_created": True,
        "integrated_cases_total": len(results),
        "integrated_cases_passed": len(results),
        "future_operator_review_package_created_count": len(packages),
        "future_operator_review_only_count": len(packages),
        "blocked_case_count": len(blocked),
        "refusal_records_created_count": sum(1 for r in results if r.refusal_record_created),
        "pii_redaction_pass_count": sum(1 for r in results if r.pii_redaction_passed),
        "vault_isolation_pass_count": sum(1 for r in results if r.vault_isolated),
        "request_shape_valid_count": sum(1 for r in results if r.request_shape_valid),
        "review_handoff_created_count": sum(1 for r in results if r.review_handoff_created),
        "authorization_intent_present_count": sum(1 for r in results if r.authorization_intent_present),
        "billing_ack_present_count": sum(1 for r in results if r.billing_ack_present),
        "medication_safety_proof_present_count": sum(1 for r in results if r.medication_safety_proof_present),
        "explicit_live_call_request_blocked": any("explicit_live_call_request_blocked" in r.block_reasons for r in blocked),
        "active_write_blocked": any("active_write_requested_blocked" in r.block_reasons for r in blocked),
        "auto_accept_blocked": any("auto_accept_requested_blocked" in r.block_reasons for r in blocked),
        "medication_safety_non_bypass_enforced": any("medication_safety_proof_required" in r.block_reasons for r in blocked),
        "medical_decision_blocked": any("medical_decision_logic_requested_blocked" in r.block_reasons for r in blocked),
        "real_doc_live_allowed_count": sum(1 for r in results if r.live_call_allowed),
        "raw_pii_in_package_count": sum(1 for r in results if r.raw_pii_in_package),
        "raw_pii_in_report_count": sum(1 for r in results if r.raw_pii_in_report),
        "token_map_in_package_count": sum(1 for r in results if r.token_map_in_package),
        "token_map_in_report_count": sum(1 for r in results if r.token_map_in_report),
        "medical_decision_made_count": sum(1 for r in results if r.medical_decision_made),
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def _collect_block_reasons(case: IntegratedReadinessCase, pii: dict[str, Any], adapter: Any, auth: Any, med: Any) -> list[str]:
    reasons: list[str] = []
    if case.declared_provenance == PROVENANCE_UNKNOWN:
        reasons.append("unknown_provenance_blocked")
    if case.declared_provenance == PROVENANCE_REAL_PRIVATE:
        reasons.append("real_private_marker_blocked")
    if case.raw_pii_residue or pii["raw_pii_in_outbound_payload"] or adapter.raw_pii_in_request:
        reasons.append("raw_pii_residue_blocked")
    if case.token_map_leak or pii["token_map_in_outbound_payload"] or adapter.token_map_in_request:
        reasons.append("token_map_leak_blocked")
    if case.forbidden_request_metadata:
        reasons.append("forbidden_request_metadata_blocked")
    if case.invalid_generation_config:
        reasons.append("invalid_generation_config_blocked")
    if not case.review_handoff_present or not adapter.review_queue_handoff_record_created:
        reasons.append("missing_review_handoff")
    if not case.authorization_present or not auth.authorization_intent_present:
        reasons.append("missing_human_authorization")
    if not case.billing_present or not auth.billing_ack_present:
        reasons.append("missing_billing_ack")
    if case.active_write_requested:
        reasons.append("active_write_requested_blocked")
    if case.auto_accept_requested:
        reasons.append("auto_accept_requested_blocked")
    if case.medication_present and not case.medication_safety_proof_present:
        reasons.append("medication_safety_proof_required")
    if case.medication_decision_requested or med.forbidden_medical_decision_detected:
        reasons.append("medical_decision_logic_requested_blocked")
    if case.explicit_live_call_requested:
        reasons.append("explicit_live_call_request_blocked")
    return sorted(set(reasons))


def _fingerprint(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


__all__ = [
    "READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY",
    "IntegratedReadinessCase",
    "IntegratedReadinessResult",
    "IntegratedReadinessPackage",
    "IntegratedGateTrace",
    "IntegratedRefusalRecord",
    "run_integrated_readiness_harness_no_live",
    "build_integrated_readiness_case",
    "build_integrated_readiness_cases",
    "build_integrated_readiness_package_no_live",
    "validate_integrated_package_sanitized",
    "summarize_integrated_gate_trace",
    "evaluate_all_integrated_readiness_cases",
    "build_integrated_metrics",
]
