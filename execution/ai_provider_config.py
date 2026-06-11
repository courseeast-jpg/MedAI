"""Provider configuration models for disabled 15C AI adapter stubs."""
from __future__ import annotations

from dataclasses import asdict, dataclass


PROVIDER_NAMES = ("fake_local", "gemini", "claude", "openai", "local_ollama")


@dataclass(frozen=True)
class AIProviderConfig:
    provider_name: str
    model_name: str
    provider_enabled: bool
    provider_mode: str
    requires_operator_approval: bool
    supports_text_layout_summary: bool
    supports_raw_pdf: bool
    supports_raw_image: bool
    supports_vision: bool
    external_network_required: bool
    sdk_import_required: bool
    estimated_input_token_limit: int
    estimated_output_token_limit: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    fail_closed_reason: str = ""


def default_provider_configs() -> dict[str, AIProviderConfig]:
    return {
        "fake_local": AIProviderConfig(
            provider_name="fake_local",
            model_name="fake-local-15c",
            provider_enabled=True,
            provider_mode="fake_local",
            requires_operator_approval=False,
            supports_text_layout_summary=True,
            supports_raw_pdf=False,
            supports_raw_image=False,
            supports_vision=False,
            external_network_required=False,
            sdk_import_required=False,
            estimated_input_token_limit=4096,
            estimated_output_token_limit=1024,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
        ),
        "gemini": _disabled_external("gemini", "gemini-disabled-15c"),
        "claude": _disabled_external("claude", "claude-disabled-15c"),
        "openai": _disabled_external("openai", "openai-disabled-15c"),
        "local_ollama": _disabled_external("local_ollama", "ollama-disabled-15c", network=False),
    }


def provider_config_to_public_dict(config: AIProviderConfig) -> dict:
    return asdict(config)


def _disabled_external(provider_name: str, model_name: str, *, network: bool = True) -> AIProviderConfig:
    return AIProviderConfig(
        provider_name=provider_name,
        model_name=model_name,
        provider_enabled=False,
        provider_mode="disabled",
        requires_operator_approval=True,
        supports_text_layout_summary=True,
        supports_raw_pdf=False,
        supports_raw_image=False,
        supports_vision=False,
        external_network_required=network,
        sdk_import_required=False,
        estimated_input_token_limit=0,
        estimated_output_token_limit=0,
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        fail_closed_reason="provider_disabled_by_policy",
    )


__all__ = ["PROVIDER_NAMES", "AIProviderConfig", "default_provider_configs", "provider_config_to_public_dict"]
