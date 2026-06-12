"""Fail-closed real-provider enablement policy for 15F."""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Mapping


REAL_PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "local_ollama": "",
}
REAL_PROVIDER_MODELS = {
    "gemini": "gemini-disabled-15f",
    "claude": "claude-disabled-15f",
    "openai": "openai-disabled-15f",
    "local_ollama": "ollama-disabled-15f",
    "fake_local": "fake-local-15f",
}


@dataclass(frozen=True)
class AIRealProviderEnablementPolicy:
    provider_name: str
    model_name: str
    provider_configured: bool
    provider_enabled_by_policy: bool
    supports_redacted_text_layout_summary: bool
    supports_raw_pdf: bool
    supports_raw_image: bool
    supports_vision: bool
    requires_operator_approval: bool
    requires_privacy_gate: bool
    requires_budget_guard: bool
    requires_payload_policy: bool
    dry_run_required_before_real_call: bool
    real_execution_supported_in_this_block: bool
    credential_env_var_name: str


@dataclass(frozen=True)
class AIRealProviderCredentialStatus:
    provider_name: str
    credential_env_var_name: str
    credential_present: bool
    credential_value_redacted: bool
    credential_fingerprint: str
    credential_printed_or_logged: bool
    provider_enabled_by_environment: bool


@dataclass(frozen=True)
class AIRealProviderSafetyChecklist:
    provider_name: str
    privacy_gate_required: bool
    payload_policy_required: bool
    budget_guard_required: bool
    operator_approval_required: bool
    dry_run_required_before_real_call: bool
    dry_run_passed: bool
    external_api_used: bool
    real_network_call_used: bool
    final_external_call_allowed: bool


@dataclass(frozen=True)
class AIRealProviderEnablementDecision:
    provider_name: str
    model_name: str
    provider_configured: bool
    provider_enabled_by_policy: bool
    provider_enabled_by_operator: bool
    provider_enabled_by_environment: bool
    credential_env_var_name: str
    credential_present: bool
    credential_value_redacted: bool
    credential_fingerprint: str
    credential_printed_or_logged: bool
    real_provider_execution_enabled: bool
    real_provider_execution_block_reason: str
    privacy_gate_required: bool
    payload_policy_required: bool
    budget_guard_required: bool
    operator_approval_required: bool
    dry_run_required_before_real_call: bool
    dry_run_passed: bool
    external_api_used: bool
    real_network_call_used: bool
    final_external_call_allowed: bool


@dataclass(frozen=True)
class AIRealProviderExecutionReadiness:
    policy: AIRealProviderEnablementPolicy
    credential_status: AIRealProviderCredentialStatus
    safety_checklist: AIRealProviderSafetyChecklist
    decision: AIRealProviderEnablementDecision


def default_real_provider_policies() -> dict[str, AIRealProviderEnablementPolicy]:
    return {
        "fake_local": _policy("fake_local", enabled=True, env_var=""),
        "gemini": _policy("gemini", env_var=REAL_PROVIDER_ENV_VARS["gemini"]),
        "claude": _policy("claude", env_var=REAL_PROVIDER_ENV_VARS["claude"]),
        "openai": _policy("openai", env_var=REAL_PROVIDER_ENV_VARS["openai"]),
        "local_ollama": _policy("local_ollama", env_var=REAL_PROVIDER_ENV_VARS["local_ollama"]),
    }


def evaluate_real_provider_execution_readiness(
    *,
    provider_name: str,
    operator_approval_state: str,
    dry_run_decision_result: dict | None,
    environ: Mapping[str, str] | None = None,
) -> AIRealProviderExecutionReadiness:
    provider = str(provider_name or "fake_local").strip().lower() or "fake_local"
    policies = default_real_provider_policies()
    policy = policies.get(provider) or _policy(provider, configured=False, env_var="")
    env = os.environ if environ is None else environ
    credential = _credential_status(policy, env)
    dry_run = dict(dry_run_decision_result or {})
    dry_run_passed = bool(dry_run.get("dry_run_external_call_allowed", False))
    provider_enabled_by_operator = str(operator_approval_state or "") == "approved_for_real_provider"
    checklist = AIRealProviderSafetyChecklist(
        provider_name=policy.provider_name,
        privacy_gate_required=policy.requires_privacy_gate,
        payload_policy_required=policy.requires_payload_policy,
        budget_guard_required=policy.requires_budget_guard,
        operator_approval_required=policy.requires_operator_approval,
        dry_run_required_before_real_call=policy.dry_run_required_before_real_call,
        dry_run_passed=dry_run_passed,
        external_api_used=False,
        real_network_call_used=False,
        final_external_call_allowed=False,
    )
    block_reason = _block_reason(policy, credential, provider_enabled_by_operator, dry_run_passed)
    decision = AIRealProviderEnablementDecision(
        provider_name=policy.provider_name,
        model_name=policy.model_name,
        provider_configured=policy.provider_configured,
        provider_enabled_by_policy=policy.provider_enabled_by_policy,
        provider_enabled_by_operator=provider_enabled_by_operator,
        provider_enabled_by_environment=credential.provider_enabled_by_environment,
        credential_env_var_name=credential.credential_env_var_name,
        credential_present=credential.credential_present,
        credential_value_redacted=credential.credential_value_redacted,
        credential_fingerprint=credential.credential_fingerprint,
        credential_printed_or_logged=False,
        real_provider_execution_enabled=False,
        real_provider_execution_block_reason=block_reason,
        privacy_gate_required=policy.requires_privacy_gate,
        payload_policy_required=policy.requires_payload_policy,
        budget_guard_required=policy.requires_budget_guard,
        operator_approval_required=policy.requires_operator_approval,
        dry_run_required_before_real_call=policy.dry_run_required_before_real_call,
        dry_run_passed=dry_run_passed,
        external_api_used=False,
        real_network_call_used=False,
        final_external_call_allowed=False,
    )
    return AIRealProviderExecutionReadiness(
        policy=policy,
        credential_status=credential,
        safety_checklist=checklist,
        decision=decision,
    )


def real_provider_enablement_to_public_dict(readiness: AIRealProviderExecutionReadiness) -> dict:
    return asdict(readiness.decision)


def real_provider_credential_to_public_dict(readiness: AIRealProviderExecutionReadiness) -> dict:
    return asdict(readiness.credential_status)


def real_provider_safety_checklist_to_public_dict(readiness: AIRealProviderExecutionReadiness) -> dict:
    return asdict(readiness.safety_checklist)


def _policy(provider_name: str, *, configured: bool = True, enabled: bool = False, env_var: str = "") -> AIRealProviderEnablementPolicy:
    provider = str(provider_name or "unknown").strip().lower() or "unknown"
    return AIRealProviderEnablementPolicy(
        provider_name=provider,
        model_name=REAL_PROVIDER_MODELS.get(provider, "unknown-disabled-15f"),
        provider_configured=configured,
        provider_enabled_by_policy=bool(enabled),
        supports_redacted_text_layout_summary=True,
        supports_raw_pdf=False,
        supports_raw_image=False,
        supports_vision=False,
        requires_operator_approval=provider != "fake_local",
        requires_privacy_gate=provider != "fake_local",
        requires_budget_guard=provider != "fake_local",
        requires_payload_policy=provider != "fake_local",
        dry_run_required_before_real_call=provider != "fake_local",
        real_execution_supported_in_this_block=False,
        credential_env_var_name=env_var,
    )


def _credential_status(
    policy: AIRealProviderEnablementPolicy,
    environ: Mapping[str, str],
) -> AIRealProviderCredentialStatus:
    env_var = policy.credential_env_var_name
    present = bool(env_var and str(environ.get(env_var) or "").strip())
    fingerprint = _credential_fingerprint(policy.provider_name, env_var, present)
    return AIRealProviderCredentialStatus(
        provider_name=policy.provider_name,
        credential_env_var_name=env_var,
        credential_present=present,
        credential_value_redacted=present,
        credential_fingerprint=fingerprint,
        credential_printed_or_logged=False,
        provider_enabled_by_environment=present if policy.provider_name != "fake_local" else True,
    )


def _credential_fingerprint(provider_name: str, env_var: str, present: bool) -> str:
    basis = f"{provider_name}:{env_var}:{'present' if present else 'missing'}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def _block_reason(
    policy: AIRealProviderEnablementPolicy,
    credential: AIRealProviderCredentialStatus,
    provider_enabled_by_operator: bool,
    dry_run_passed: bool,
) -> str:
    if not policy.provider_configured:
        return "provider_not_configured"
    if policy.provider_name == "fake_local":
        return "real_provider_not_selected"
    if not policy.provider_enabled_by_policy:
        return "real_provider_execution_disabled_by_policy"
    if not provider_enabled_by_operator:
        return "operator_real_provider_enablement_required"
    if policy.credential_env_var_name and not credential.credential_present:
        return "credential_missing"
    if policy.dry_run_required_before_real_call and not dry_run_passed:
        return "dry_run_required_before_real_call"
    return "real_provider_execution_disabled_in_15f"


__all__ = [
    "AIRealProviderEnablementPolicy",
    "AIRealProviderCredentialStatus",
    "AIRealProviderEnablementDecision",
    "AIRealProviderExecutionReadiness",
    "AIRealProviderSafetyChecklist",
    "default_real_provider_policies",
    "evaluate_real_provider_execution_readiness",
    "real_provider_enablement_to_public_dict",
    "real_provider_credential_to_public_dict",
    "real_provider_safety_checklist_to_public_dict",
]
