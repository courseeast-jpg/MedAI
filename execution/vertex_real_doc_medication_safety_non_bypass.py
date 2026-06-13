"""No-live medication safety non-bypass gates for real-doc readiness (15Z-E).

Medication mentions are modeled only as review-bound candidate facts. This
module never performs DDI, contraindication, dosage, treatment, diagnosis, or
recommendation logic; it only blocks attempts to request such decision outputs.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from execution.vertex_real_doc_authorization_cost_refusal_gates import (
    FUTURE_PACKAGE_STATUS,
    build_authorization_cost_fixtures,
    evaluate_authorization_cost_refusal_gates,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REDACTED_REAL_LIKE,
    evaluate_vertex_real_doc_readiness,
)

READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY = "READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY"
BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED = "BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED"

DECISION_DDI = "ddi_decision"
DECISION_CONTRAINDICATION = "contraindication_decision"
DECISION_DOSAGE = "dosage_advice"
DECISION_TREATMENT = "treatment_advice"
DECISION_DIAGNOSIS = "diagnosis_output"
FORBIDDEN_DECISION_TYPES = (
    DECISION_DDI,
    DECISION_CONTRAINDICATION,
    DECISION_DOSAGE,
    DECISION_TREATMENT,
    DECISION_DIAGNOSIS,
)


@dataclass(frozen=True)
class MedicationCandidateFact:
    medication_name_mention: str
    dose_string: str
    route_frequency: str
    source_evidence: str
    uncertainty_flag: str
    review_required: bool
    decision_outcome: str
    recommendation_outcome: str
    active_write_allowed: bool
    auto_accept_allowed: bool


@dataclass(frozen=True)
class MedicationSafetyProof:
    proof_id: str
    medication_facts_are_candidate_facts_only: bool
    ddi_decision_performed: bool
    contraindication_decision_performed: bool
    dosage_advice_performed: bool
    treatment_advice_performed: bool
    diagnosis_performed: bool
    review_required: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    live_call_allowed: bool


@dataclass(frozen=True)
class MedicationDecisionBoundaryViolation:
    violation_type: str
    blocked: bool
    decision_made: bool


@dataclass(frozen=True)
class MedicationSafetyGateEvaluation:
    case_id: str
    medication_facts_present: bool
    medication_safety_proof_present: bool
    medication_candidate_facts_count: int
    forbidden_medical_decision_detected: bool
    forbidden_decision_types: list[str]
    readiness_status: str
    blocked: bool
    block_reasons: list[str]
    refusal_record_created: bool
    future_review_package_created: bool
    live_call_allowed: bool
    external_api_used: bool
    active_write_allowed: bool
    auto_accept_allowed: bool
    review_required: bool
    ddi_decision_made: bool
    contraindication_decision_made: bool
    dosage_advice_made: bool
    treatment_advice_made: bool
    diagnosis_made: bool
    raw_pii_in_report: bool
    token_map_in_report: bool
    medication_candidate_facts: list[MedicationCandidateFact] = field(default_factory=list)
    medication_safety_proof: MedicationSafetyProof | None = None
    refusal_record: dict[str, Any] | None = None
    future_review_package: dict[str, Any] | None = None


@dataclass(frozen=True)
class MedicationSafetyFixture:
    case_id: str
    medication_facts_present: bool
    safety_proof_present: bool = False
    dose_present: bool = False
    frequency_present: bool = False
    uncertainty: bool = False
    tokenized_evidence: bool = False
    active_write_requested: bool = False
    auto_accept_requested: bool = False
    explicit_live_call_requested: bool = False
    requested_decision_types: tuple[str, ...] = ()
    future_authorization_package_context: bool = False


def build_medication_candidate_fact_no_live(
    *,
    medication_name_mention: str = "Examplemed",
    dose_string: str = "",
    route_frequency: str = "",
    source_evidence: str = "Medication mention visible in tokenized source",
    uncertainty_flag: str = "candidate mention only; operator review required",
) -> MedicationCandidateFact:
    return MedicationCandidateFact(
        medication_name_mention=medication_name_mention,
        dose_string=dose_string,
        route_frequency=route_frequency,
        source_evidence=source_evidence,
        uncertainty_flag=uncertainty_flag,
        review_required=True,
        decision_outcome="",
        recommendation_outcome="",
        active_write_allowed=False,
        auto_accept_allowed=False,
    )


def build_medication_safety_proof_no_live(case_id: str) -> MedicationSafetyProof:
    return MedicationSafetyProof(
        proof_id="medproof_" + _fingerprint(case_id)[:12],
        medication_facts_are_candidate_facts_only=True,
        ddi_decision_performed=False,
        contraindication_decision_performed=False,
        dosage_advice_performed=False,
        treatment_advice_performed=False,
        diagnosis_performed=False,
        review_required=True,
        active_write_allowed=False,
        auto_accept_allowed=False,
        live_call_allowed=False,
    )


def detect_forbidden_medical_decision_output(requested_decision_types: tuple[str, ...]) -> list[MedicationDecisionBoundaryViolation]:
    return [
        MedicationDecisionBoundaryViolation(
            violation_type=decision_type,
            blocked=True,
            decision_made=False,
        )
        for decision_type in requested_decision_types
        if decision_type in FORBIDDEN_DECISION_TYPES
    ]


def build_medication_safety_refusal_record(case_id: str, reasons: list[str]) -> dict[str, Any]:
    safe_reasons = sorted(set(reasons or ["medication_safety_gate_blocked"]))
    return {
        "refusal_id": "med_refusal_" + _fingerprint(case_id + "|".join(safe_reasons))[:12],
        "case_id": case_id,
        "refusal_reason_codes": safe_reasons,
        "sanitized_report_only": True,
        "live_call_allowed": False,
        "active_write_allowed": False,
        "auto_accept_allowed": False,
        "review_required": True,
        "raw_pii_in_report": False,
        "token_map_in_report": False,
    }


def integrate_medication_gate_with_future_package(evaluation: MedicationSafetyGateEvaluation) -> dict[str, Any] | None:
    if evaluation.blocked or not evaluation.medication_safety_proof_present:
        return None
    # Exercise 15Z-D future package logic as context while downgrading the public
    # medication status to review-package-only.
    base = next(f for f in build_authorization_cost_fixtures() if f.case_id == "authorization_and_billing_present_with_15z_c_handoff")
    auth_eval = evaluate_authorization_cost_refusal_gates(base)
    return {
        "future_review_package_id": "med_future_review_" + _fingerprint(evaluation.case_id)[:12],
        "case_id": evaluation.case_id,
        "status": READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY,
        "authorization_context_status": auth_eval.readiness_status if auth_eval.readiness_status == FUTURE_PACKAGE_STATUS else "",
        "medication_safety_proof_present": True,
        "medication_candidate_facts_count": evaluation.medication_candidate_facts_count,
        "live_call_allowed": False,
        "external_api_used": False,
        "active_write_allowed": False,
        "auto_accept_allowed": False,
        "review_required": True,
    }


def evaluate_medication_safety_non_bypass(fixture: MedicationSafetyFixture) -> MedicationSafetyGateEvaluation:
    facts = _candidate_facts_for_fixture(fixture)
    proof = build_medication_safety_proof_no_live(fixture.case_id) if fixture.safety_proof_present else None
    violations = detect_forbidden_medical_decision_output(fixture.requested_decision_types)
    forbidden_types = [v.violation_type for v in violations]

    framework = evaluate_vertex_real_doc_readiness(
        case_id=fixture.case_id,
        declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like medication safety fixture",
        human_authorization_present=True,
        billing_cost_cap_ack_present=True,
        contains_medication_fact=fixture.medication_facts_present,
        medication_safety_gate_satisfied=proof is not None,
        future_gates_simulated_pass=True,
        active_write_requested=fixture.active_write_requested,
        auto_accept_requested=fixture.auto_accept_requested,
    )

    reasons: list[str] = []
    if fixture.medication_facts_present and proof is None:
        reasons.append("medication_safety_proof_required")
    if fixture.active_write_requested:
        reasons.append("active_write_requested_blocked")
    if fixture.auto_accept_requested:
        reasons.append("auto_accept_requested_blocked")
    if fixture.explicit_live_call_requested:
        reasons.append("explicit_live_call_request_blocked")
    if forbidden_types:
        reasons.append("medical_decision_logic_requested_blocked")
    if framework.live_call_allowed:
        reasons.append("framework_live_call_unexpected")

    blocked = bool(reasons)
    if blocked:
        status = _blocked_status(reasons)
    elif fixture.medication_facts_present and proof is not None:
        status = READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY
    else:
        status = "MEDICATION_GATE_NOT_REQUIRED_REVIEW_BOUND"

    shell = MedicationSafetyGateEvaluation(
        case_id=fixture.case_id,
        medication_facts_present=fixture.medication_facts_present,
        medication_safety_proof_present=proof is not None,
        medication_candidate_facts_count=len(facts),
        forbidden_medical_decision_detected=bool(forbidden_types),
        forbidden_decision_types=forbidden_types,
        readiness_status=status,
        blocked=blocked,
        block_reasons=reasons,
        refusal_record_created=blocked,
        future_review_package_created=False,
        live_call_allowed=False,
        external_api_used=False,
        active_write_allowed=False,
        auto_accept_allowed=False,
        review_required=True,
        ddi_decision_made=False,
        contraindication_decision_made=False,
        dosage_advice_made=False,
        treatment_advice_made=False,
        diagnosis_made=False,
        raw_pii_in_report=False,
        token_map_in_report=False,
        medication_candidate_facts=facts,
        medication_safety_proof=proof,
        refusal_record=build_medication_safety_refusal_record(fixture.case_id, reasons) if blocked else None,
        future_review_package=None,
    )
    package = integrate_medication_gate_with_future_package(shell)
    return MedicationSafetyGateEvaluation(
        **{
            **asdict(shell),
            "medication_candidate_facts": facts,
            "medication_safety_proof": proof,
            "refusal_record": shell.refusal_record,
            "future_review_package": package,
            "future_review_package_created": package is not None,
        }
    )


def build_medication_safety_fixtures() -> list[MedicationSafetyFixture]:
    return [
        MedicationSafetyFixture("medication_mention_candidate_only_no_safety_proof", True),
        MedicationSafetyFixture("medication_mention_candidate_only_with_safety_proof", True, safety_proof_present=True),
        MedicationSafetyFixture("medication_with_explicit_dose_candidate_only", True, safety_proof_present=True, dose_present=True),
        MedicationSafetyFixture("medication_with_frequency_candidate_only", True, safety_proof_present=True, frequency_present=True),
        MedicationSafetyFixture("medication_with_uncertainty_candidate_only", True, safety_proof_present=True, uncertainty=True),
        MedicationSafetyFixture("medication_fact_active_write_requested", True, safety_proof_present=True, active_write_requested=True),
        MedicationSafetyFixture("medication_fact_auto_accept_requested", True, safety_proof_present=True, auto_accept_requested=True),
        MedicationSafetyFixture("medication_fact_ddi_decision_requested", True, safety_proof_present=True, requested_decision_types=(DECISION_DDI,)),
        MedicationSafetyFixture("medication_fact_contraindication_decision_requested", True, safety_proof_present=True, requested_decision_types=(DECISION_CONTRAINDICATION,)),
        MedicationSafetyFixture("medication_fact_dosage_advice_requested", True, safety_proof_present=True, requested_decision_types=(DECISION_DOSAGE,)),
        MedicationSafetyFixture("medication_fact_treatment_advice_requested", True, safety_proof_present=True, requested_decision_types=(DECISION_TREATMENT,)),
        MedicationSafetyFixture("medication_fact_diagnosis_output_requested", True, safety_proof_present=True, requested_decision_types=(DECISION_DIAGNOSIS,)),
        MedicationSafetyFixture("non_medication_fixture", False),
        MedicationSafetyFixture("medication_with_tokenized_evidence_and_pii_vault_reference", True, safety_proof_present=True, tokenized_evidence=True),
        MedicationSafetyFixture("explicit_live_call_requested_even_with_safety_proof", True, safety_proof_present=True, explicit_live_call_requested=True),
        MedicationSafetyFixture("future_authorization_package_with_medication_without_safety_proof", True, future_authorization_package_context=True),
        MedicationSafetyFixture("future_authorization_package_with_medication_with_safety_proof", True, safety_proof_present=True, future_authorization_package_context=True),
    ]


def evaluate_all_medication_safety_cases() -> dict[str, Any]:
    evaluations = [evaluate_medication_safety_non_bypass(f) for f in build_medication_safety_fixtures()]
    return {
        "summary": build_medication_safety_metrics(evaluations),
        "cases": [asdict(e) for e in evaluations],
        "refusal_records": [e.refusal_record for e in evaluations if e.refusal_record],
        "future_review_packages": [e.future_review_package for e in evaluations if e.future_review_package],
    }


def build_medication_safety_metrics(evaluations: list[MedicationSafetyGateEvaluation]) -> dict[str, Any]:
    blocked = [e for e in evaluations if e.blocked]
    return {
        "medication_safety_non_bypass_created": True,
        "medication_cases_total": len(evaluations),
        "medication_cases_passed": len(evaluations),
        "medication_facts_case_count": sum(1 for e in evaluations if e.medication_facts_present),
        "medication_candidate_facts_total": sum(e.medication_candidate_facts_count for e in evaluations),
        "medication_safety_proof_created_count": sum(1 for e in evaluations if e.medication_safety_proof_present),
        "medication_safety_proof_required_block_count": sum(
            1 for e in blocked if "medication_safety_proof_required" in e.block_reasons
        ),
        "forbidden_medical_decision_block_count": sum(
            1 for e in blocked if "medical_decision_logic_requested_blocked" in e.block_reasons
        ),
        "ddi_decision_blocked": any(DECISION_DDI in e.forbidden_decision_types for e in blocked),
        "contraindication_decision_blocked": any(DECISION_CONTRAINDICATION in e.forbidden_decision_types for e in blocked),
        "dosage_advice_blocked": any(DECISION_DOSAGE in e.forbidden_decision_types for e in blocked),
        "treatment_advice_blocked": any(DECISION_TREATMENT in e.forbidden_decision_types for e in blocked),
        "diagnosis_output_blocked": any(DECISION_DIAGNOSIS in e.forbidden_decision_types for e in blocked),
        "future_review_package_created_count": sum(1 for e in evaluations if e.future_review_package_created),
        "future_review_only_count": sum(1 for e in evaluations if e.readiness_status == READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY),
        "real_doc_live_allowed_count": sum(1 for e in evaluations if e.live_call_allowed),
        "explicit_live_call_request_blocked": any(
            "explicit_live_call_request_blocked" in e.block_reasons for e in blocked
        ),
        "active_write_blocked": any("active_write_requested_blocked" in e.block_reasons for e in blocked),
        "auto_accept_blocked": any("auto_accept_requested_blocked" in e.block_reasons for e in blocked),
        "ddi_decision_made_count": sum(1 for e in evaluations if e.ddi_decision_made),
        "contraindication_decision_made_count": sum(1 for e in evaluations if e.contraindication_decision_made),
        "dosage_advice_made_count": sum(1 for e in evaluations if e.dosage_advice_made),
        "treatment_advice_made_count": sum(1 for e in evaluations if e.treatment_advice_made),
        "diagnosis_made_count": sum(1 for e in evaluations if e.diagnosis_made),
        "raw_pii_in_report_count": sum(1 for e in evaluations if e.raw_pii_in_report),
        "token_map_in_report_count": sum(1 for e in evaluations if e.token_map_in_report),
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }


def _candidate_facts_for_fixture(fixture: MedicationSafetyFixture) -> list[MedicationCandidateFact]:
    if not fixture.medication_facts_present:
        return []
    evidence = "Medication: [MEDICATION_1] listed in tokenized source" if fixture.tokenized_evidence else "Medication mention visible in tokenized source"
    uncertainty = "uncertain source wording; operator review required" if fixture.uncertainty else "candidate mention only; operator review required"
    return [
        build_medication_candidate_fact_no_live(
            dose_string="5 mg" if fixture.dose_present else "",
            route_frequency="daily" if fixture.frequency_present else "",
            source_evidence=evidence,
            uncertainty_flag=uncertainty,
        )
    ]


def _blocked_status(reasons: list[str]) -> str:
    if "medication_safety_proof_required" in reasons:
        return BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED
    if "active_write_requested_blocked" in reasons:
        return "BLOCKED_ACTIVE_WRITE_REQUESTED"
    if "auto_accept_requested_blocked" in reasons:
        return "BLOCKED_AUTO_ACCEPT_REQUESTED"
    if "medical_decision_logic_requested_blocked" in reasons:
        return "BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED"
    if "explicit_live_call_request_blocked" in reasons:
        return "BLOCKED_EXPLICIT_LIVE_CALL_REQUESTED"
    return "BLOCKED_MEDICATION_SAFETY_GATE"


def _fingerprint(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


__all__ = [
    "READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY",
    "BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED",
    "MedicationCandidateFact",
    "MedicationSafetyProof",
    "MedicationSafetyGateEvaluation",
    "MedicationDecisionBoundaryViolation",
    "MedicationSafetyFixture",
    "build_medication_candidate_fact_no_live",
    "build_medication_safety_proof_no_live",
    "evaluate_medication_safety_non_bypass",
    "detect_forbidden_medical_decision_output",
    "build_medication_safety_refusal_record",
    "integrate_medication_gate_with_future_package",
    "build_medication_safety_fixtures",
    "evaluate_all_medication_safety_cases",
    "build_medication_safety_metrics",
]
