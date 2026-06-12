"""Unified provider operator-control surface (15K).

A single audited operator-control layer that can *show*, but not yet *enable*,
the future real-execution readiness state for all provider adapters
(gemini, claude, openai, local_ollama) plus fake_local.

15K makes ZERO real provider calls, network calls, local-model calls, and
subprocess calls. Staging an enablement request only records intent — it never
sets ``real_provider_execution_enabled`` or ``final_external_call_allowed`` and
never imports a provider SDK, opens a socket, calls localhost, or shells out.

Doctrine: adapters are source-package reconstruction adapters; AI output stays
review-bound; record counts are never a success metric.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from execution.ai_provider_enablement import evaluate_real_provider_execution_readiness
from execution.gemini_extraction_adapter import build_gemini_adapter_status
from execution.claude_extraction_adapter import build_claude_adapter_status
from execution.openai_extraction_adapter import build_openai_adapter_status
from execution.local_ollama_extraction_adapter import build_local_ollama_adapter_status


CONTROLLED_PROVIDERS = ("fake_local", "gemini", "claude", "openai", "local_ollama")
CLOUD_PROVIDERS = {"gemini", "claude", "openai"}
LOCAL_PROVIDERS = {"local_ollama"}
OPERATOR_REQUEST_STATES = ("not_requested", "staged", "denied", "expired", "invalid")

STAGED_REQUEST_NOTICE = "Staged request does not enable execution"
REAL_PROVIDER_DISABLED_NOTICE = "Real provider execution disabled by policy"
NO_EXTERNAL_CALL_NOTICE = "No external AI call was made"
NO_LOCAL_MODEL_CALL_NOTICE = "No local model call was made"


@dataclass(frozen=True)
class AIProviderOperatorEnablementRequest:
    requested_provider: str
    operator_enablement_request_state: str = "not_requested"
    is_cloud_provider: bool = False
    credential_present: bool = False
    justification_present: bool = False


@dataclass(frozen=True)
class AIProviderOperatorEnablementDecision:
    requested_provider: str
    operator_enablement_request_state: str
    operator_enablement_request_allowed: bool
    operator_enablement_request_block_reason: str
    staged_only: bool
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    local_model_call_used: bool
    subprocess_call_used: bool


@dataclass(frozen=True)
class AIProviderOperatorReadinessSummary:
    provider_name: str
    adapter_contract_available: bool
    schema_contract_available: bool
    provider_enabled_by_policy: bool
    provider_enabled_by_operator: bool
    provider_enabled_by_environment: bool
    credential_required: bool
    credential_env_var_name: str
    credential_present: bool
    credential_value_redacted: bool
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    dry_run_status: str
    dry_run_passed: bool
    operator_enablement_request_state: str
    operator_enablement_request_allowed: bool
    operator_enablement_request_block_reason: str
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    local_model_call_used: bool
    subprocess_call_used: bool
    real_call_attempted: bool
    execution_allowed: bool
    execution_block_reason: str


@dataclass(frozen=True)
class AIProviderOperatorControlState:
    requested_provider: str
    selected_provider: str
    effective_provider: str
    provider_enabled_by_policy: bool
    provider_enabled_by_operator: bool
    provider_enabled_by_environment: bool
    credential_present: bool
    credential_value_redacted: bool
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    dry_run_status: str
    adapter_contract_available: bool
    schema_contract_available: bool
    operator_enablement_request_state: str
    operator_enablement_request_allowed: bool
    operator_enablement_request_block_reason: str
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    local_model_call_used: bool
    subprocess_call_used: bool
    active_written_count: int
    auto_accept: bool
    review_required: bool
    providers: list[dict[str, Any]] = field(default_factory=list)
    staged_request_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AIProviderOperatorAuditRecord:
    requested_provider: str
    selected_provider: str
    effective_provider: str
    operator_enablement_request_state: str
    operator_enablement_request_allowed: bool
    operator_enablement_request_block_reason: str
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    local_model_call_used: bool
    subprocess_call_used: bool
    active_written_count: int
    auto_accept: bool
    review_required: bool
    audit_scope: str = "public_statuses_counts_only"


def evaluate_operator_enablement_request(
    *,
    provider_name: str,
    requested_state: str = "staged",
    privacy_gate_result: dict[str, Any] | None,
    payload_policy_result: dict[str, Any] | None,
    budget_guard_result: dict[str, Any] | None,
    dry_run_decision_result: dict[str, Any] | None,
    credential_present: bool,
    is_cloud_provider: bool,
) -> AIProviderOperatorEnablementDecision:
    """Evaluate a staged enablement request. Never enables execution."""
    provider = str(provider_name or "").strip().lower()
    if requested_state == "not_requested":
        return _decision(provider, "not_requested", allowed=False, reason="no_request")
    if provider == "fake_local":
        return _decision(provider, "invalid", allowed=False, reason="fake_local_no_enablement_request_needed")
    if requested_state not in OPERATOR_REQUEST_STATES:
        return _decision(provider, "invalid", allowed=False, reason="unknown_request_state")

    privacy = dict(privacy_gate_result or {})
    policy = dict(payload_policy_result or {})
    budget = dict(budget_guard_result or {})
    dry_run = dict(dry_run_decision_result or {})

    reason = ""
    if not privacy:
        reason = "missing_privacy_gate"
    elif str(privacy.get("privacy_gate_status") or "").startswith("fail_closed"):
        reason = "privacy_gate_failed"
    elif not policy:
        reason = "missing_payload_policy"
    elif not bool(policy.get("payload_policy_allowed")):
        reason = "payload_policy_failed"
    elif not budget:
        reason = "missing_budget_guard"
    elif not bool(budget.get("budget_allowed")):
        reason = "budget_exceeded"
    elif not bool(dry_run.get("dry_run_external_call_allowed")):
        reason = "dry_run_required"
    elif is_cloud_provider and not bool(credential_present):
        reason = "credential_missing"

    if reason:
        return _decision(provider, "invalid", allowed=False, reason=reason)
    # Gates pass: the request may be STAGED only. Execution stays disabled.
    return _decision(provider, "staged", allowed=True, reason="staged_only_execution_remains_disabled")


def build_provider_operator_control(
    *,
    requested_provider: str,
    selected_provider: str,
    effective_provider: str,
    privacy_gate_result: dict[str, Any] | None,
    payload_policy_result: dict[str, Any] | None,
    budget_guard_result: dict[str, Any] | None,
    dry_run_decision_result: dict[str, Any] | None,
    enablement_request: dict[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AIProviderOperatorControlState:
    requested = str(requested_provider or "fake_local").strip().lower() or "fake_local"
    privacy = dict(privacy_gate_result or {})
    policy = dict(payload_policy_result or {})
    budget = dict(budget_guard_result or {})
    dry_run = dict(dry_run_decision_result or {})
    dry_run_passed = bool(dry_run.get("dry_run_external_call_allowed", False))
    dry_run_status = (
        "dry-run only - real provider execution remains disabled" if dry_run_passed else "dry-run blocked"
    )
    request = dict(enablement_request or {})
    request_provider = str(request.get("provider") or request.get("requested_provider") or "").strip().lower()
    request_state = str(request.get("requested_state") or "not_requested")

    summaries: list[AIProviderOperatorReadinessSummary] = []
    for provider in CONTROLLED_PROVIDERS:
        cred = _credential_readiness(provider, environ)
        is_cloud = provider in CLOUD_PROVIDERS
        if provider == request_provider and provider != "fake_local":
            decision = evaluate_operator_enablement_request(
                provider_name=provider,
                requested_state=request_state,
                privacy_gate_result=privacy,
                payload_policy_result=policy,
                budget_guard_result=budget,
                dry_run_decision_result=dry_run,
                credential_present=cred["credential_present"],
                is_cloud_provider=is_cloud,
            )
        else:
            decision = _decision(provider, "not_requested", allowed=False, reason="no_request")
        if provider == "fake_local":
            block_reason = "fake_local_local_only_no_external_execution"
        else:
            block_reason = evaluate_real_provider_execution_readiness(
                provider_name=provider,
                operator_approval_state="approved_for_real_provider",
                dry_run_decision_result=dry_run,
                environ=environ,
            ).decision.real_provider_execution_block_reason
        summaries.append(
            AIProviderOperatorReadinessSummary(
                provider_name=provider,
                adapter_contract_available=True,
                schema_contract_available=True,
                provider_enabled_by_policy=(provider == "fake_local"),
                provider_enabled_by_operator=False,
                provider_enabled_by_environment=(provider == "fake_local"),
                credential_required=is_cloud,
                credential_env_var_name=cred["credential_env_var_name"],
                credential_present=cred["credential_present"],
                credential_value_redacted=cred["credential_present"],
                privacy_gate_status=str(privacy.get("privacy_gate_status") or ""),
                payload_policy_allowed=bool(policy.get("payload_policy_allowed", False)),
                budget_allowed=bool(budget.get("budget_allowed", False)),
                dry_run_status=dry_run_status,
                dry_run_passed=dry_run_passed,
                operator_enablement_request_state=decision.operator_enablement_request_state,
                operator_enablement_request_allowed=decision.operator_enablement_request_allowed,
                operator_enablement_request_block_reason=decision.operator_enablement_request_block_reason,
                real_provider_execution_enabled=False,
                final_external_call_allowed=False,
                external_api_used=False,
                real_network_call_used=False,
                local_model_call_used=False,
                subprocess_call_used=False,
                real_call_attempted=False,
                execution_allowed=False,
                execution_block_reason=block_reason,
            )
        )

    by_provider = {summary.provider_name: summary for summary in summaries}
    primary = by_provider.get(requested, by_provider["fake_local"])
    staged_decision = next(
        (
            evaluate_operator_enablement_request(
                provider_name=request_provider,
                requested_state=request_state,
                privacy_gate_result=privacy,
                payload_policy_result=policy,
                budget_guard_result=budget,
                dry_run_decision_result=dry_run,
                credential_present=_credential_readiness(request_provider, environ)["credential_present"],
                is_cloud_provider=request_provider in CLOUD_PROVIDERS,
            )
            for _ in [0]
            if request_provider
        ),
        _decision(requested, "not_requested", allowed=False, reason="no_request"),
    )

    return AIProviderOperatorControlState(
        requested_provider=requested,
        selected_provider=str(selected_provider or requested),
        effective_provider=str(effective_provider or "fake_local"),
        provider_enabled_by_policy=primary.provider_enabled_by_policy,
        provider_enabled_by_operator=False,
        provider_enabled_by_environment=primary.provider_enabled_by_environment,
        credential_present=primary.credential_present,
        credential_value_redacted=primary.credential_value_redacted,
        privacy_gate_status=primary.privacy_gate_status,
        payload_policy_allowed=primary.payload_policy_allowed,
        budget_allowed=primary.budget_allowed,
        dry_run_status=dry_run_status,
        adapter_contract_available=primary.adapter_contract_available,
        schema_contract_available=primary.schema_contract_available,
        operator_enablement_request_state=staged_decision.operator_enablement_request_state,
        operator_enablement_request_allowed=staged_decision.operator_enablement_request_allowed,
        operator_enablement_request_block_reason=staged_decision.operator_enablement_request_block_reason,
        real_provider_execution_enabled=False,
        final_external_call_allowed=False,
        external_api_used=False,
        real_network_call_used=False,
        local_model_call_used=False,
        subprocess_call_used=False,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        providers=[_summary_public_dict(summary) for summary in summaries],
        staged_request_decision=asdict(staged_decision),
    )


def build_operator_audit_record(state: AIProviderOperatorControlState) -> AIProviderOperatorAuditRecord:
    return AIProviderOperatorAuditRecord(
        requested_provider=state.requested_provider,
        selected_provider=state.selected_provider,
        effective_provider=state.effective_provider,
        operator_enablement_request_state=state.operator_enablement_request_state,
        operator_enablement_request_allowed=state.operator_enablement_request_allowed,
        operator_enablement_request_block_reason=state.operator_enablement_request_block_reason,
        real_provider_execution_enabled=False,
        final_external_call_allowed=False,
        external_api_used=False,
        real_network_call_used=False,
        local_model_call_used=False,
        subprocess_call_used=False,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
    )


def operator_control_to_public_dict(state: AIProviderOperatorControlState) -> dict[str, Any]:
    return asdict(state)


def operator_audit_to_public_dict(record: AIProviderOperatorAuditRecord) -> dict[str, Any]:
    return asdict(record)


def provider_readiness_matrix(state: AIProviderOperatorControlState) -> list[dict[str, Any]]:
    return list(state.providers)


def _decision(provider: str, state: str, *, allowed: bool, reason: str) -> AIProviderOperatorEnablementDecision:
    return AIProviderOperatorEnablementDecision(
        requested_provider=provider,
        operator_enablement_request_state=state,
        operator_enablement_request_allowed=allowed,
        operator_enablement_request_block_reason=reason,
        staged_only=True,
        real_provider_execution_enabled=False,
        final_external_call_allowed=False,
        external_api_used=False,
        real_network_call_used=False,
        local_model_call_used=False,
        subprocess_call_used=False,
    )


def _credential_readiness(provider: str, environ: Mapping[str, str] | None) -> dict[str, Any]:
    provider = str(provider or "").strip().lower()
    if provider == "gemini":
        status = build_gemini_adapter_status(selected_provider=provider, environ=environ)
        return {
            "credential_env_var_name": status["credential_env_var_name"],
            "credential_present": bool(status["credential_present"]),
        }
    if provider == "claude":
        status = build_claude_adapter_status(selected_provider=provider, environ=environ)
        return {
            "credential_env_var_name": status["credential_env_var_name"],
            "credential_present": bool(status["credential_present"]),
        }
    if provider == "openai":
        status = build_openai_adapter_status(selected_provider=provider, environ=environ)
        return {
            "credential_env_var_name": status["credential_env_var_name"],
            "credential_present": bool(status["credential_present"]),
        }
    if provider == "local_ollama":
        # Non-secret model/base URL config only; no credential.
        return {"credential_env_var_name": "", "credential_present": False}
    # fake_local: no external credential required.
    return {"credential_env_var_name": "", "credential_present": False}


def _summary_public_dict(summary: AIProviderOperatorReadinessSummary) -> dict[str, Any]:
    return asdict(summary)


__all__ = [
    "CONTROLLED_PROVIDERS",
    "CLOUD_PROVIDERS",
    "LOCAL_PROVIDERS",
    "OPERATOR_REQUEST_STATES",
    "STAGED_REQUEST_NOTICE",
    "REAL_PROVIDER_DISABLED_NOTICE",
    "NO_EXTERNAL_CALL_NOTICE",
    "NO_LOCAL_MODEL_CALL_NOTICE",
    "AIProviderOperatorEnablementRequest",
    "AIProviderOperatorEnablementDecision",
    "AIProviderOperatorReadinessSummary",
    "AIProviderOperatorControlState",
    "AIProviderOperatorAuditRecord",
    "evaluate_operator_enablement_request",
    "build_provider_operator_control",
    "build_operator_audit_record",
    "operator_control_to_public_dict",
    "operator_audit_to_public_dict",
    "provider_readiness_matrix",
]
