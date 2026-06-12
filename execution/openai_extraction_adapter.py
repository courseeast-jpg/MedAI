"""Disabled-by-policy OpenAI text-extraction adapter contract (15I).

Mirrors the safe Gemini adapter pattern from 15G for the OpenAI provider,
built on the shared provider-neutral premium adapter contract. Real OpenAI
execution is disabled by policy: no SDK import, no network code, fail-closed
gates, credential presence only (never the value), output always review-bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from execution.premium_adapter_contract import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    MockPremiumClient,
    PremiumAdapterConfig,
    PremiumAdapterRequest,
    PremiumAdapterResult,
    PremiumExtractionAdapter,
    PremiumPromptContract,
    PremiumResponseSchemaValidationResult,
    build_mock_premium_responses,
    build_premium_adapter_status,
    run_premium_mock_extraction_preview,
)

OPENAI_PROVIDER_NAME = "openai"
OPENAI_DEFAULT_MODEL = "openai-disabled-15i"
OPENAI_CREDENTIAL_ENV_VAR = "OPENAI_API_KEY"
OPENAI_PROMPT_CONTRACT_VERSION = "openai-extraction-prompt-15i-v1"
OPENAI_SCHEMA_CONTRACT_VERSION = "openai-extraction-schema-15i-v1"


@dataclass(frozen=True)
class OpenAIAdapterConfig(PremiumAdapterConfig):
    pass


@dataclass(frozen=True)
class OpenAIAdapterRequest(PremiumAdapterRequest):
    pass


@dataclass(frozen=True)
class OpenAIPromptContract(PremiumPromptContract):
    pass


@dataclass(frozen=True)
class OpenAIResponseSchemaValidationResult(PremiumResponseSchemaValidationResult):
    pass


@dataclass(frozen=True)
class OpenAIAdapterResult(PremiumAdapterResult):
    pass


class MockOpenAIClient(MockPremiumClient):
    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        super().__init__(provider_name=OPENAI_PROVIDER_NAME, responses=responses)


class OpenAIExtractionAdapter(PremiumExtractionAdapter):
    provider_name = OPENAI_PROVIDER_NAME
    adapter_name = "OpenAIExtractionAdapter"
    default_model_name = OPENAI_DEFAULT_MODEL
    credential_env_var_name = OPENAI_CREDENTIAL_ENV_VAR
    credential_env_var_aliases = ()
    prompt_contract_version = OPENAI_PROMPT_CONTRACT_VERSION
    schema_contract_version = OPENAI_SCHEMA_CONTRACT_VERSION
    config_cls = OpenAIAdapterConfig
    request_cls = OpenAIAdapterRequest
    result_cls = OpenAIAdapterResult
    prompt_contract_cls = OpenAIPromptContract
    schema_result_cls = OpenAIResponseSchemaValidationResult


def build_openai_adapter_status(
    *,
    selected_provider: str,
    environ: Mapping[str, str] | None = None,
    real_provider_execution_block_reason: str = "real_provider_execution_disabled_by_policy",
) -> dict[str, Any]:
    return build_premium_adapter_status(
        OpenAIExtractionAdapter(),
        selected_provider=selected_provider,
        environ=environ,
        real_provider_execution_block_reason=real_provider_execution_block_reason,
    )


def run_openai_mock_extraction_preview(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    return run_premium_mock_extraction_preview(
        OpenAIExtractionAdapter(client=MockOpenAIClient()), environ=environ
    )


def build_mock_openai_responses() -> dict[str, dict[str, Any]]:
    return build_mock_premium_responses()


__all__ = [
    "OPENAI_PROVIDER_NAME",
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_CREDENTIAL_ENV_VAR",
    "OPENAI_PROMPT_CONTRACT_VERSION",
    "OPENAI_SCHEMA_CONTRACT_VERSION",
    "ALLOWED_PAYLOAD_TYPE",
    "FORBIDDEN_PAYLOAD_TYPES",
    "OpenAIAdapterConfig",
    "OpenAIAdapterRequest",
    "OpenAIPromptContract",
    "OpenAIResponseSchemaValidationResult",
    "OpenAIAdapterResult",
    "OpenAIExtractionAdapter",
    "MockOpenAIClient",
    "build_openai_adapter_status",
    "run_openai_mock_extraction_preview",
    "build_mock_openai_responses",
]
