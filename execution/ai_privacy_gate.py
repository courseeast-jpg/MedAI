"""Local-only privacy gate for future external AI extraction payloads."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from execution.ai_budget_guard import AIBudgetGuardResult, budget_guard_to_public_dict
from execution.ai_payload_policy import AIPayloadPolicyResult, payload_policy_to_public_dict


@dataclass(frozen=True)
class PIIEntity:
    category: str
    token: str
    raw_value: str


@dataclass(frozen=True)
class PIITokenMap:
    entities: list[PIIEntity] = field(default_factory=list)
    local_only: bool = True

    def to_local_dict(self) -> dict[str, str]:
        return {entity.token: entity.raw_value for entity in self.entities}

    def public_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.category] = counts.get(entity.category, 0) + 1
        return counts


@dataclass(frozen=True)
class RedactedAIPayload:
    payload_type: str
    redacted_text: str
    redacted_payload_hash: str
    raw_payload_hash_local_only: str
    token_categories: list[str]
    pii_detected_count: int
    pii_redacted_count: int


@dataclass(frozen=True)
class AIExternalCallApprovalState:
    state: str = "not_requested"
    operator_id: str = ""
    approved_at: str = ""


@dataclass(frozen=True)
class AIPrivacyGateDecision:
    privacy_gate_status: str
    external_payload_allowed: bool
    redacted_payload_preview_available: bool
    external_call_requires_operator_approval: bool
    fail_closed_reason: str


@dataclass(frozen=True)
class AIPrivacyGateResult:
    privacy_gate_status: str
    pii_detected_count: int
    pii_redacted_count: int
    pii_token_map_local_only: bool
    redacted_payload_preview_available: bool
    external_payload_allowed: bool
    external_call_requires_operator_approval: bool
    external_api_used: bool
    fail_closed_reason: str
    redaction_complete: bool
    suspected_unredacted_pii: bool
    token_categories: list[str]
    payload: RedactedAIPayload
    token_map: PIITokenMap
    decision: AIPrivacyGateDecision


TOKEN_ORDER = [
    ("PATIENT", r"\b(?:Patient|Name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b|\bPatient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"),
    ("DOB", r"\bDOB\s*:?\s*\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("MRN", r"\bMRN\s*:?\s*[A-Z0-9-]{4,}\b"),
    ("ACCESSION", r"\b(?:Accession|Accession Number)\s*:?\s*[A-Z]{1,4}-?\d{4,}[-A-Z0-9]*\b"),
    ("FACILITY", r"\bFacility\s*:?\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4}\b"),
    ("PROVIDER", r"\b(?:Provider|Physician|Doctor|Dr\.)\s*:?\s*(?:Dr\.\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"),
    ("ADDRESS", r"\b\d{2,6}\s+[A-Z][A-Za-z0-9. ]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln),?\s+[A-Z][A-Za-z ]+,\s*[A-Z]{2}\s+\d{5}\b"),
    ("PHONE", r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("INSURANCE_ID", r"\b(?:Insurance|Insurance ID|Policy)\s*:?\s*[A-Z0-9-]{5,}\b"),
    ("DATE", r"\b(?:Collected|Reported|Date)\s*:?\s*\d{1,2}/\d{1,2}/\d{2,4}\b"),
]


def run_ai_privacy_gate(
    *,
    raw_text: str,
    payload_type: str = "redacted_text_layout_summary",
) -> AIPrivacyGateResult:
    token_map, redacted_text = _tokenize(raw_text or "")
    redacted_payload = RedactedAIPayload(
        payload_type=payload_type,
        redacted_text=redacted_text,
        redacted_payload_hash=_sha256(redacted_text),
        raw_payload_hash_local_only=_sha256(raw_text or ""),
        token_categories=sorted(token_map.public_counts()),
        pii_detected_count=len(token_map.entities),
        pii_redacted_count=len(token_map.entities),
    )
    suspected = bool(_detect_categories(redacted_text))
    status = "redacted_payload_ready" if token_map.entities and not suspected else "no_pii_detected"
    if suspected:
        status = "fail_closed_unredacted_pii_detected"
    decision = AIPrivacyGateDecision(
        privacy_gate_status=status,
        external_payload_allowed=False,
        redacted_payload_preview_available=bool(redacted_text),
        external_call_requires_operator_approval=True,
        fail_closed_reason="external_calls_disabled_in_15b" if not suspected else "suspected_unredacted_pii",
    )
    return AIPrivacyGateResult(
        privacy_gate_status=decision.privacy_gate_status,
        pii_detected_count=len(token_map.entities),
        pii_redacted_count=len(token_map.entities),
        pii_token_map_local_only=True,
        redacted_payload_preview_available=decision.redacted_payload_preview_available,
        external_payload_allowed=False,
        external_call_requires_operator_approval=True,
        external_api_used=False,
        fail_closed_reason=decision.fail_closed_reason,
        redaction_complete=len(token_map.entities) == len(token_map.entities),
        suspected_unredacted_pii=suspected,
        token_categories=redacted_payload.token_categories,
        payload=redacted_payload,
        token_map=token_map,
        decision=decision,
    )


def privacy_gate_to_public_dict(result: AIPrivacyGateResult) -> dict[str, Any]:
    return {
        "privacy_gate_status": result.privacy_gate_status,
        "pii_detected_count": result.pii_detected_count,
        "pii_redacted_count": result.pii_redacted_count,
        "pii_token_map_local_only": result.pii_token_map_local_only,
        "redacted_payload_preview_available": result.redacted_payload_preview_available,
        "external_payload_allowed": False,
        "external_call_requires_operator_approval": result.external_call_requires_operator_approval,
        "external_api_used": False,
        "fail_closed_reason": result.fail_closed_reason,
        "redaction_complete": result.redaction_complete,
        "suspected_unredacted_pii": result.suspected_unredacted_pii,
        "token_category_counts": result.token_map.public_counts(),
        "redacted_payload_hash": result.payload.redacted_payload_hash[:12],
        "payload_type": result.payload.payload_type,
    }


def build_ai_external_call_audit_record(
    *,
    source_id: str,
    adapter_name: str,
    privacy_gate_result: AIPrivacyGateResult,
    payload_policy_result: AIPayloadPolicyResult,
    budget_result: AIBudgetGuardResult,
    approval_state: AIExternalCallApprovalState,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_id": source_id,
        "adapter_name": adapter_name,
        "provider_name": budget_result.provider_name,
        "model_name": budget_result.model_name,
        "external_api_used": False,
        "privacy_gate_status": privacy_gate_result.privacy_gate_status,
        "pii_detected_count": privacy_gate_result.pii_detected_count,
        "pii_redacted_count": privacy_gate_result.pii_redacted_count,
        "redacted_payload_hash": privacy_gate_result.payload.redacted_payload_hash[:12],
        "raw_payload_hash_local_only": privacy_gate_result.payload.raw_payload_hash_local_only[:12],
        "operator_approval_state": approval_state.state,
        "budget_allowed": budget_result.budget_allowed,
        "payload_policy_allowed": payload_policy_result.payload_policy_allowed,
        "final_external_call_allowed": False,
        "fail_closed_reason": payload_policy_result.fail_closed_reason or privacy_gate_result.fail_closed_reason,
    }


def audit_record_to_public_dict(audit_record: dict[str, Any]) -> dict[str, Any]:
    return dict(audit_record)


def combined_gate_to_public_dict(
    *,
    privacy_gate_result: AIPrivacyGateResult,
    payload_policy_result: AIPayloadPolicyResult,
    budget_result: AIBudgetGuardResult,
    audit_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "privacy_gate": privacy_gate_to_public_dict(privacy_gate_result),
        "payload_policy": payload_policy_to_public_dict(payload_policy_result),
        "budget_guard": budget_guard_to_public_dict(budget_result),
        "audit": audit_record_to_public_dict(audit_record),
    }


def _tokenize(text: str) -> tuple[PIITokenMap, str]:
    counters: dict[str, int] = {}
    entities: list[PIIEntity] = []
    redacted = text
    for category, pattern in TOKEN_ORDER:
        def replace(match: re.Match[str]) -> str:
            raw_value = match.group(0)
            counters[category] = counters.get(category, 0) + 1
            token = f"[{category}_{counters[category]}]"
            entities.append(PIIEntity(category=category, token=token, raw_value=raw_value))
            return token

        redacted = re.sub(pattern, replace, redacted)
    return PIITokenMap(entities=entities, local_only=True), redacted


def _detect_categories(text: str) -> list[str]:
    found: list[str] = []
    for category, pattern in TOKEN_ORDER:
        if re.search(pattern, text):
            found.append(category)
    return found


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


__all__ = [
    "PIIEntity",
    "PIITokenMap",
    "RedactedAIPayload",
    "AIPrivacyGateDecision",
    "AIPrivacyGateResult",
    "AIExternalCallApprovalState",
    "run_ai_privacy_gate",
    "privacy_gate_to_public_dict",
    "build_ai_external_call_audit_record",
    "audit_record_to_public_dict",
    "combined_gate_to_public_dict",
]
