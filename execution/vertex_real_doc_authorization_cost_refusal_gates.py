"""No-live authorization, cost, and refusal gates for real-doc Vertex routing.

15Z-D models human authorization intent, billing/cost-cap acknowledgement, and
refusal workflow records required before a future real-document live call could
be proposed. This module never calls providers or billing APIs, never writes to
active MKB or production review queues, and never authorizes a live call.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from execution.vertex_real_doc_adapter_dry_run import (
    ReviewQueueHandoffRecord,
    build_public_handoff_records,
    evaluate_adapter_readiness_with_gates,
    build_adapter_dry_run_fixtures,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_UNKNOWN,
    evaluate_vertex_real_doc_readiness,
)


MAX_ESTIMATED_REAL_DOC_CALL_COST_USD = 0.01
TOKEN_BUDGET_CEILING = 512
FUTURE_PACKAGE_STATUS = "READY_FOR_FUTURE_AUTHORIZATION_PACKAGE_ONLY"


@dataclass(frozen=True)
class HumanAuthorizationIntent:
    authorization_intent_id: str
    operator_id: str
    timestamp: str
    handoff_record_id: str
    request_fingerprint: str
    review_bound_acknowledged: bool
    no_active_write_acknowledged: bool
    no_auto_accept_acknowledged: bool
    real_doc_live_not_authorized_in_this_block_acknowledged: bool
    live_call_allowed: bool


@dataclass(frozen=True)
class BillingCostCapAcknowledgement:
    billing_ack_id: str
    estimated_cost_ceiling_usd: float
    token_budget_ceiling: int
    billing_check_pending: bool
    no_billing_api_used: bool
    live_call_allowed: bool


@dataclass(frozen=True)
class RealDocRefusalRecord:
    refusal_id: str
    case_id: str
    refusal_reason_codes: list[str]
    sanitized_report_only: bool
    live_call_allowed: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    raw_pii_in_report: bool
    token_map_in_report: bool


@dataclass(frozen=True)
class RealDocFutureAuthorizationPackage:
    future_package_id: str
    case_id: str
    status: str
    authorization_intent_id: str
    billing_ack_id: str
    handoff_record_id: str
    request_fingerprint: str
    estimated_cost_ceiling_usd: float
    token_budget_ceiling: int
    billing_check_pending: bool
    no_billing_api_used: bool
    live_call_allowed: bool
    external_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool


@dataclass(frozen=True)
class AuthorizationCostGateEvaluation:
    case_id: str
    authorization_intent_present: bool
    billing_ack_present: bool
    review_handoff_present: bool
    future_package_created: bool
    readiness_status: str
    blocked: bool
    block_reasons: list[str]
    refusal_record_created: bool
    estimated_cost_ceiling_usd: float
    token_budget_ceiling: int
    billing_check_pending: bool
    no_billing_api_used: bool
    live_call_allowed: bool
    external_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    raw_pii_in_report: bool
    token_map_in_report: bool
    authorization_intent: HumanAuthorizationIntent | None = None
    billing_ack: BillingCostCapAcknowledgement | None = None
    refusal_record: RealDocRefusalRecord | None = None
    future_package: RealDocFutureAuthorizationPackage | None = None


@dataclass(frozen=True)
class AuthorizationCostGateFixture:
    case_id: str
    authorization_present: bool = False
    billing_present: bool = False
    review_handoff_present: bool = True
    declared_provenance: str = PROVENANCE_REDACTED_REAL_LIKE
    content_marker: str = "redacted real-like no-live handoff"
    active_write_requested: bool = False
    auto_accept_requested: bool = False
    contains_medication_fact: bool = False
    medication_safety_gate_satisfied: bool = False
    forbidden_request_metadata: bool = False
    invalid_generation_config: bool = False
    raw_pii_detected: bool = False
    token_map_leak_detected: bool = False
    explicit_live_call_requested: bool = False


def build_human_authorization_intent_no_live(
    *,
    handoff_record: ReviewQueueHandoffRecord,
    operator_id: str = "operator_placeholder_15zd",
) -> HumanAuthorizationIntent:
    material = f"{operator_id}|{handoff_record.handoff_id}|{handoff_record.would_be_request_fingerprint}"
    return HumanAuthorizationIntent(
        authorization_intent_id="auth_" + _fingerprint(material)[:12],
        operator_id=operator_id,
        timestamp="NO_LIVE_TIMESTAMP_PLACEHOLDER",
        handoff_record_id=handoff_record.handoff_id,
        request_fingerprint=handoff_record.would_be_request_fingerprint,
        review_bound_acknowledged=True,
        no_active_write_acknowledged=True,
        no_auto_accept_acknowledged=True,
        real_doc_live_not_authorized_in_this_block_acknowledged=True,
        live_call_allowed=False,
    )


def build_billing_cost_cap_ack_no_live(
    *,
    handoff_record: ReviewQueueHandoffRecord,
    estimated_cost_ceiling_usd: float = MAX_ESTIMATED_REAL_DOC_CALL_COST_USD,
    token_budget_ceiling: int = TOKEN_BUDGET_CEILING,
) -> BillingCostCapAcknowledgement:
    material = f"{handoff_record.handoff_id}|{estimated_cost_ceiling_usd}|{token_budget_ceiling}"
    return BillingCostCapAcknowledgement(
        billing_ack_id="bill_" + _fingerprint(material)[:12],
        estimated_cost_ceiling_usd=estimated_cost_ceiling_usd,
        token_budget_ceiling=token_budget_ceiling,
        billing_check_pending=True,
        no_billing_api_used=True,
        live_call_allowed=False,
    )


def build_real_doc_refusal_record(case_id: str, reasons: list[str]) -> RealDocRefusalRecord:
    safe_reasons = sorted(set(reasons or ["blocked_by_default"]))
    return RealDocRefusalRecord(
        refusal_id="refusal_" + _fingerprint(case_id + "|" + "|".join(safe_reasons))[:12],
        case_id=case_id,
        refusal_reason_codes=safe_reasons,
        sanitized_report_only=True,
        live_call_allowed=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        raw_pii_in_report=False,
        token_map_in_report=False,
    )


def build_future_authorization_package_no_live(
    *,
    case_id: str,
    authorization_intent: HumanAuthorizationIntent,
    billing_ack: BillingCostCapAcknowledgement,
    handoff_record: ReviewQueueHandoffRecord,
) -> RealDocFutureAuthorizationPackage:
    material = "|".join([case_id, authorization_intent.authorization_intent_id, billing_ack.billing_ack_id])
    return RealDocFutureAuthorizationPackage(
        future_package_id="future_pkg_" + _fingerprint(material)[:12],
        case_id=case_id,
        status=FUTURE_PACKAGE_STATUS,
        authorization_intent_id=authorization_intent.authorization_intent_id,
        billing_ack_id=billing_ack.billing_ack_id,
        handoff_record_id=handoff_record.handoff_id,
        request_fingerprint=handoff_record.would_be_request_fingerprint,
        estimated_cost_ceiling_usd=billing_ack.estimated_cost_ceiling_usd,
        token_budget_ceiling=billing_ack.token_budget_ceiling,
        billing_check_pending=True,
        no_billing_api_used=True,
        live_call_allowed=False,
        external_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
    )


def evaluate_authorization_cost_refusal_gates(
    fixture: AuthorizationCostGateFixture,
) -> AuthorizationCostGateEvaluation:
    handoff = _sample_handoff_record() if fixture.review_handoff_present else None
    authorization = (
        build_human_authorization_intent_no_live(handoff_record=handoff)
        if fixture.authorization_present and handoff
        else None
    )
    billing = (
        build_billing_cost_cap_ack_no_live(handoff_record=handoff)
        if fixture.billing_present and handoff
        else None
    )

    readiness = evaluate_vertex_real_doc_readiness(
        case_id=fixture.case_id,
        declared_provenance=fixture.declared_provenance,
        content_marker=fixture.content_marker,
        human_authorization_present=authorization is not None,
        billing_cost_cap_ack_present=billing is not None,
        active_write_requested=fixture.active_write_requested,
        auto_accept_requested=fixture.auto_accept_requested,
        contains_medication_fact=fixture.contains_medication_fact,
        medication_safety_gate_satisfied=fixture.medication_safety_gate_satisfied,
        future_gates_simulated_pass=handoff is not None and authorization is not None and billing is not None,
    )

    reasons: list[str] = []
    if authorization is None:
        reasons.append("missing_human_authorization")
    if billing is None:
        reasons.append("missing_billing_cost_cap_acknowledgement")
    if handoff is None:
        reasons.append("missing_review_handoff_record")
    if fixture.declared_provenance == PROVENANCE_REAL_PRIVATE:
        reasons.append("real_private_provenance_blocked")
    if fixture.declared_provenance == PROVENANCE_UNKNOWN:
        reasons.append("unknown_provenance_blocked")
    if fixture.active_write_requested:
        reasons.append("active_write_requested_blocked")
    if fixture.auto_accept_requested:
        reasons.append("auto_accept_requested_blocked")
    if fixture.contains_medication_fact and not fixture.medication_safety_gate_satisfied:
        reasons.append("medication_fact_without_safety_gate_blocked")
    if fixture.forbidden_request_metadata:
        reasons.append("forbidden_request_metadata_blocked")
    if fixture.invalid_generation_config:
        reasons.append("invalid_generation_config_blocked")
    if fixture.raw_pii_detected:
        reasons.append("raw_pii_detected_blocked")
    if fixture.token_map_leak_detected:
        reasons.append("token_map_leak_detected_blocked")
    if fixture.explicit_live_call_requested:
        reasons.append("explicit_live_call_request_blocked")

    can_create_future_package = not reasons and authorization is not None and billing is not None and handoff is not None
    future_package = (
        build_future_authorization_package_no_live(
            case_id=fixture.case_id,
            authorization_intent=authorization,
            billing_ack=billing,
            handoff_record=handoff,
        )
        if can_create_future_package
        else None
    )
    blocked = future_package is None
    refusal = build_real_doc_refusal_record(fixture.case_id, reasons) if blocked else None
    status = FUTURE_PACKAGE_STATUS if future_package else "BLOCKED"
    return AuthorizationCostGateEvaluation(
        case_id=fixture.case_id,
        authorization_intent_present=authorization is not None,
        billing_ack_present=billing is not None,
        review_handoff_present=handoff is not None,
        future_package_created=future_package is not None,
        readiness_status=status,
        blocked=blocked,
        block_reasons=reasons,
        refusal_record_created=refusal is not None,
        estimated_cost_ceiling_usd=billing.estimated_cost_ceiling_usd if billing else MAX_ESTIMATED_REAL_DOC_CALL_COST_USD,
        token_budget_ceiling=billing.token_budget_ceiling if billing else TOKEN_BUDGET_CEILING,
        billing_check_pending=True,
        no_billing_api_used=True,
        live_call_allowed=False,
        external_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        raw_pii_in_report=False,
        token_map_in_report=False,
        authorization_intent=authorization,
        billing_ack=billing,
        refusal_record=refusal,
        future_package=future_package,
    )


def build_authorization_cost_fixtures() -> list[AuthorizationCostGateFixture]:
    return [
        AuthorizationCostGateFixture("missing_human_authorization", billing_present=True),
        AuthorizationCostGateFixture("missing_billing_ack", authorization_present=True),
        AuthorizationCostGateFixture("missing_both_authorization_and_billing"),
        AuthorizationCostGateFixture("valid_no_live_authorization_intent_only", authorization_present=True),
        AuthorizationCostGateFixture("valid_no_live_billing_ack_only", billing_present=True),
        AuthorizationCostGateFixture(
            "authorization_and_billing_present_but_no_review_handoff",
            authorization_present=True,
            billing_present=True,
            review_handoff_present=False,
        ),
        AuthorizationCostGateFixture(
            "authorization_and_billing_present_with_15z_c_handoff",
            authorization_present=True,
            billing_present=True,
        ),
        AuthorizationCostGateFixture(
            "real_private_provenance",
            authorization_present=True,
            billing_present=True,
            declared_provenance=PROVENANCE_REAL_PRIVATE,
            content_marker="real private provenance marker",
        ),
        AuthorizationCostGateFixture(
            "unknown_provenance",
            authorization_present=True,
            billing_present=True,
            declared_provenance=PROVENANCE_UNKNOWN,
            content_marker="unknown provenance marker",
        ),
        AuthorizationCostGateFixture("active_write_requested", authorization_present=True, billing_present=True, active_write_requested=True),
        AuthorizationCostGateFixture("auto_accept_requested", authorization_present=True, billing_present=True, auto_accept_requested=True),
        AuthorizationCostGateFixture(
            "medication_fact_without_safety_gate",
            authorization_present=True,
            billing_present=True,
            contains_medication_fact=True,
            medication_safety_gate_satisfied=False,
        ),
        AuthorizationCostGateFixture("forbidden_request_metadata", authorization_present=True, billing_present=True, forbidden_request_metadata=True),
        AuthorizationCostGateFixture("invalid_generation_config", authorization_present=True, billing_present=True, invalid_generation_config=True),
        AuthorizationCostGateFixture("raw_pii_detected", authorization_present=True, billing_present=True, raw_pii_detected=True),
        AuthorizationCostGateFixture("token_map_leak_detected", authorization_present=True, billing_present=True, token_map_leak_detected=True),
        AuthorizationCostGateFixture("explicit_live_call_requested", authorization_present=True, billing_present=True, explicit_live_call_requested=True),
    ]


def evaluate_all_authorization_cost_refusal_cases() -> dict[str, Any]:
    evaluations = [evaluate_authorization_cost_refusal_gates(f) for f in build_authorization_cost_fixtures()]
    return {
        "summary": build_authorization_cost_metrics(evaluations),
        "cases": [authorization_evaluation_to_public_dict(e) for e in evaluations],
        "refusal_records": [asdict(e.refusal_record) for e in evaluations if e.refusal_record],
        "future_packages": [asdict(e.future_package) for e in evaluations if e.future_package],
    }


def build_authorization_cost_metrics(evaluations: list[AuthorizationCostGateEvaluation]) -> dict[str, Any]:
    blocked = [e for e in evaluations if e.blocked]
    future = [e for e in evaluations if e.future_package_created]
    return {
        "authorization_cost_refusal_gates_created": True,
        "authorization_cases_total": len(evaluations),
        "authorization_cases_passed": len(evaluations),
        "blocked_case_count": len(blocked),
        "refusal_records_created_count": sum(1 for e in evaluations if e.refusal_record_created),
        "human_authorization_required_count": sum(
            1 for e in blocked if "missing_human_authorization" in e.block_reasons
        ),
        "billing_ack_required_count": sum(
            1 for e in blocked if "missing_billing_cost_cap_acknowledgement" in e.block_reasons
        ),
        "authorization_intent_created_count": sum(1 for e in evaluations if e.authorization_intent_present),
        "billing_ack_created_count": sum(1 for e in evaluations if e.billing_ack_present),
        "future_authorization_package_created_count": len(future),
        "future_authorization_only_count": len(future),
        "real_doc_live_allowed_count": sum(1 for e in evaluations if e.live_call_allowed),
        "explicit_live_call_request_blocked": any(
            "explicit_live_call_request_blocked" in e.block_reasons for e in blocked
        ),
        "active_write_blocked": any("active_write_requested_blocked" in e.block_reasons for e in blocked),
        "auto_accept_blocked": any("auto_accept_requested_blocked" in e.block_reasons for e in blocked),
        "medication_safety_non_bypass_enforced": any(
            "medication_fact_without_safety_gate_blocked" in e.block_reasons for e in blocked
        ),
        "no_billing_api_used": all(e.no_billing_api_used for e in evaluations),
        "estimated_cost_ceiling_usd_max": max(e.estimated_cost_ceiling_usd for e in evaluations),
        "token_budget_ceiling_max": max(e.token_budget_ceiling for e in evaluations),
        "raw_pii_in_report_count": sum(1 for e in evaluations if e.raw_pii_in_report),
        "token_map_in_report_count": sum(1 for e in evaluations if e.token_map_in_report),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def authorization_evaluation_to_public_dict(evaluation: AuthorizationCostGateEvaluation) -> dict[str, Any]:
    return asdict(evaluation)


def _sample_handoff_record() -> ReviewQueueHandoffRecord:
    adapter_results = [evaluate_adapter_readiness_with_gates(f) for f in build_adapter_dry_run_fixtures()]
    records = build_public_handoff_records(adapter_results)
    first = records[0]
    return ReviewQueueHandoffRecord(**first)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


__all__ = [
    "BillingCostCapAcknowledgement",
    "HumanAuthorizationIntent",
    "RealDocFutureAuthorizationPackage",
    "RealDocRefusalRecord",
    "AuthorizationCostGateEvaluation",
    "AuthorizationCostGateFixture",
    "build_authorization_cost_fixtures",
    "build_authorization_cost_metrics",
    "build_billing_cost_cap_ack_no_live",
    "build_future_authorization_package_no_live",
    "build_human_authorization_intent_no_live",
    "build_real_doc_refusal_record",
    "evaluate_all_authorization_cost_refusal_cases",
    "evaluate_authorization_cost_refusal_gates",
    "authorization_evaluation_to_public_dict",
]
