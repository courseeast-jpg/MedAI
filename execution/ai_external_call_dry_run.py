"""No-network external AI call dry-run contract for 15E."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


APPROVAL_STATES = {"not_requested", "pending", "approved_for_dry_run", "denied", "expired"}


@dataclass(frozen=True)
class AIExternalCallDryRunRequest:
    requested_provider: str
    effective_provider: str
    model_name: str
    provider_enabled: bool
    operator_approval_state: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    redacted_payload_hash: str
    dry_run_mode: bool = False


@dataclass(frozen=True)
class AIExternalCallDryRunDecision:
    requested_provider: str
    effective_provider: str
    model_name: str
    provider_enabled: bool
    operator_approval_state: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    redacted_payload_hash: str
    dry_run_external_call_allowed: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    simulated_response_created: bool
    fail_closed_reason: str


@dataclass(frozen=True)
class AIExternalCallDryRunAudit:
    requested_provider: str
    effective_provider: str
    model_name: str
    provider_enabled: bool
    operator_approval_state: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    redacted_payload_hash: str
    dry_run_external_call_allowed: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    simulated_response_created: bool
    fail_closed_reason: str
    audit_scope: str = "public_hashes_counts_statuses_only"


@dataclass(frozen=True)
class AIExternalCallDryRunResult:
    request: AIExternalCallDryRunRequest
    decision: AIExternalCallDryRunDecision
    audit: AIExternalCallDryRunAudit


def build_ai_external_call_dry_run(
    *,
    requested_provider: str,
    effective_provider: str,
    model_name: str,
    provider_enabled: bool,
    operator_approval_state: str,
    privacy_gate_result: dict[str, Any] | None,
    payload_policy_result: dict[str, Any] | None,
    budget_guard_result: dict[str, Any] | None,
    dry_run_mode: bool,
) -> AIExternalCallDryRunResult:
    approval = str(operator_approval_state or "not_requested")
    privacy = dict(privacy_gate_result or {})
    policy = dict(payload_policy_result or {})
    budget = dict(budget_guard_result or {})
    request = AIExternalCallDryRunRequest(
        requested_provider=str(requested_provider or "fake_local"),
        effective_provider=str(effective_provider or "fake_local"),
        model_name=str(model_name or ""),
        provider_enabled=bool(provider_enabled),
        operator_approval_state=approval,
        privacy_gate_status=str(privacy.get("privacy_gate_status") or ""),
        payload_policy_allowed=bool(policy.get("payload_policy_allowed", False)),
        budget_allowed=bool(budget.get("budget_allowed", False)),
        redacted_payload_hash=str(privacy.get("redacted_payload_hash") or ""),
        dry_run_mode=bool(dry_run_mode),
    )
    reason = _fail_closed_reason(request, privacy, policy, budget)
    allowed = not bool(reason)
    decision = AIExternalCallDryRunDecision(
        requested_provider=request.requested_provider,
        effective_provider=request.effective_provider,
        model_name=request.model_name,
        provider_enabled=request.provider_enabled,
        operator_approval_state=request.operator_approval_state,
        privacy_gate_status=request.privacy_gate_status,
        payload_policy_allowed=request.payload_policy_allowed,
        budget_allowed=request.budget_allowed,
        redacted_payload_hash=request.redacted_payload_hash,
        dry_run_external_call_allowed=allowed,
        final_external_call_allowed=False,
        external_api_used=False,
        real_network_call_used=False,
        simulated_response_created=allowed,
        fail_closed_reason=reason,
    )
    audit = AIExternalCallDryRunAudit(**asdict(decision))
    return AIExternalCallDryRunResult(request=request, decision=decision, audit=audit)


def dry_run_decision_to_public_dict(result: AIExternalCallDryRunResult) -> dict[str, Any]:
    return asdict(result.decision)


def dry_run_audit_to_public_dict(result: AIExternalCallDryRunResult) -> dict[str, Any]:
    return asdict(result.audit)


def _fail_closed_reason(
    request: AIExternalCallDryRunRequest,
    privacy: dict[str, Any],
    policy: dict[str, Any],
    budget: dict[str, Any],
) -> str:
    if not request.dry_run_mode:
        return "dry_run_mode_not_enabled"
    if request.operator_approval_state not in APPROVAL_STATES:
        return "unknown_operator_approval_state"
    if request.operator_approval_state != "approved_for_dry_run":
        return "operator_approval_required_for_dry_run"
    if not privacy:
        return "missing_privacy_gate_result"
    if str(privacy.get("privacy_gate_status") or "").startswith("fail_closed"):
        return "privacy_gate_failed"
    if not request.redacted_payload_hash:
        return "missing_redacted_payload_hash"
    if not policy:
        return "missing_payload_policy_result"
    if not request.payload_policy_allowed:
        return str(policy.get("fail_closed_reason") or "payload_policy_failed")
    if not budget:
        return "missing_budget_guard_result"
    if not request.budget_allowed:
        return str(budget.get("budget_fail_reason") or "budget_exceeded")
    return ""


__all__ = [
    "APPROVAL_STATES",
    "AIExternalCallDryRunRequest",
    "AIExternalCallDryRunResult",
    "AIExternalCallDryRunAudit",
    "AIExternalCallDryRunDecision",
    "build_ai_external_call_dry_run",
    "dry_run_decision_to_public_dict",
    "dry_run_audit_to_public_dict",
]
