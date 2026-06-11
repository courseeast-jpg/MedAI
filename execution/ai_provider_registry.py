"""Disabled-by-default provider registry for 15C AI extraction adapter stubs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from execution.ai_provider_config import AIProviderConfig, default_provider_configs, provider_config_to_public_dict


@dataclass(frozen=True)
class AIProviderReadinessResult:
    provider_name: str
    model_name: str
    provider_enabled: bool
    provider_mode: str
    requires_operator_approval: bool
    operator_approval_state: str
    supports_text_layout_summary: bool
    supports_raw_pdf: bool
    supports_raw_image: bool
    supports_vision: bool
    external_network_required: bool
    sdk_import_required: bool
    provider_ready: bool
    external_api_used: bool
    final_external_call_allowed: bool
    fail_closed_reason: str


@dataclass(frozen=True)
class ProviderStubResult:
    provider_name: str
    model_name: str
    provider_ready: bool
    external_api_used: bool
    final_external_call_allowed: bool
    fail_closed_reason: str


class BlockedProviderAdapterStub:
    provider_name = "blocked"

    def __init__(self, config: AIProviderConfig):
        self.config = config
        self.provider_name = config.provider_name
        self.model_name = config.model_name

    def extract(self, *_args: Any, **_kwargs: Any) -> ProviderStubResult:
        return ProviderStubResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            provider_ready=False,
            external_api_used=False,
            final_external_call_allowed=False,
            fail_closed_reason=self.config.fail_closed_reason or "provider_disabled_by_policy",
        )


class GeminiExtractionAdapterStub(BlockedProviderAdapterStub):
    pass


class ClaudeExtractionAdapterStub(BlockedProviderAdapterStub):
    pass


class OpenAIExtractionAdapterStub(BlockedProviderAdapterStub):
    pass


class LocalOllamaExtractionAdapterStub(BlockedProviderAdapterStub):
    pass


class AIProviderRegistry:
    def __init__(self, configs: dict[str, AIProviderConfig] | None = None):
        self._configs = dict(configs or default_provider_configs())

    def list_providers(self) -> list[dict]:
        return [provider_config_to_public_dict(self._configs[name]) for name in sorted(self._configs)]

    def get_provider(self, provider_name: str) -> AIProviderConfig | None:
        return self._configs.get(str(provider_name or "").strip().lower())

    def adapter_for(self, provider_name: str) -> BlockedProviderAdapterStub | None:
        config = self.get_provider(provider_name)
        if config is None:
            return None
        if config.provider_name == "gemini":
            return GeminiExtractionAdapterStub(config)
        if config.provider_name == "claude":
            return ClaudeExtractionAdapterStub(config)
        if config.provider_name == "openai":
            return OpenAIExtractionAdapterStub(config)
        if config.provider_name == "local_ollama":
            return LocalOllamaExtractionAdapterStub(config)
        return BlockedProviderAdapterStub(config)

    def validate_provider_readiness(
        self,
        *,
        provider_name: str,
        operator_approval_state: str,
        privacy_gate_result: dict[str, Any] | None,
        payload_policy_result: dict[str, Any] | None,
        budget_guard_result: dict[str, Any] | None,
    ) -> AIProviderReadinessResult:
        config = self.get_provider(provider_name)
        if config is None:
            return _unknown_provider_result(provider_name)
        reason = ""
        if not config.provider_enabled:
            reason = config.fail_closed_reason or "provider_disabled_by_policy"
        elif config.requires_operator_approval and operator_approval_state != "approved":
            reason = "operator_approval_required"
        elif config.provider_mode not in {"fake_local", "external_candidate"}:
            reason = "provider_mode_not_allowed"
        elif not privacy_gate_result:
            reason = "missing_privacy_gate_result"
        elif str(privacy_gate_result.get("privacy_gate_status") or "").startswith("fail_closed"):
            reason = "privacy_gate_failed"
        elif not payload_policy_result:
            reason = "missing_payload_policy_result"
        elif config.provider_name != "fake_local" and not bool(payload_policy_result.get("payload_policy_allowed")):
            reason = str(payload_policy_result.get("fail_closed_reason") or "payload_policy_failed")
        elif not budget_guard_result:
            reason = "missing_budget_guard_result"
        elif not bool(budget_guard_result.get("budget_allowed")):
            reason = str(budget_guard_result.get("budget_fail_reason") or "budget_exceeded")

        provider_ready = config.provider_name == "fake_local" and not reason
        return AIProviderReadinessResult(
            provider_name=config.provider_name,
            model_name=config.model_name,
            provider_enabled=config.provider_enabled,
            provider_mode=config.provider_mode,
            requires_operator_approval=config.requires_operator_approval,
            operator_approval_state=operator_approval_state,
            supports_text_layout_summary=config.supports_text_layout_summary,
            supports_raw_pdf=config.supports_raw_pdf,
            supports_raw_image=config.supports_raw_image,
            supports_vision=config.supports_vision,
            external_network_required=config.external_network_required,
            sdk_import_required=config.sdk_import_required,
            provider_ready=provider_ready,
            external_api_used=False,
            final_external_call_allowed=False,
            fail_closed_reason=reason or "fake_local_provider_ready_no_external_call",
        )


def provider_readiness_to_public_dict(result: AIProviderReadinessResult) -> dict:
    return asdict(result)


def provider_stub_to_public_dict(result: ProviderStubResult) -> dict:
    return asdict(result)


def _unknown_provider_result(provider_name: str) -> AIProviderReadinessResult:
    return AIProviderReadinessResult(
        provider_name=str(provider_name or "unknown"),
        model_name="unknown",
        provider_enabled=False,
        provider_mode="unknown",
        requires_operator_approval=True,
        operator_approval_state="missing",
        supports_text_layout_summary=False,
        supports_raw_pdf=False,
        supports_raw_image=False,
        supports_vision=False,
        external_network_required=False,
        sdk_import_required=False,
        provider_ready=False,
        external_api_used=False,
        final_external_call_allowed=False,
        fail_closed_reason="unknown_provider",
    )


__all__ = [
    "AIProviderReadinessResult",
    "ProviderStubResult",
    "AIProviderRegistry",
    "GeminiExtractionAdapterStub",
    "ClaudeExtractionAdapterStub",
    "OpenAIExtractionAdapterStub",
    "LocalOllamaExtractionAdapterStub",
    "provider_readiness_to_public_dict",
    "provider_stub_to_public_dict",
]
