"""Disabled-by-policy Claude text-extraction adapter contract (15I).

Mirrors the safe Gemini adapter pattern from 15G for the Claude provider,
built on the shared provider-neutral premium adapter contract. Real Claude
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

CLAUDE_PROVIDER_NAME = "claude"
CLAUDE_DEFAULT_MODEL = "claude-disabled-15i"
CLAUDE_CREDENTIAL_ENV_VAR = "ANTHROPIC_API_KEY"
CLAUDE_CREDENTIAL_ENV_VAR_ALIASES = ("CLAUDE_API_KEY",)
CLAUDE_PROMPT_CONTRACT_VERSION = "claude-extraction-prompt-15i-v1"
CLAUDE_SCHEMA_CONTRACT_VERSION = "claude-extraction-schema-15i-v1"


@dataclass(frozen=True)
class ClaudeAdapterConfig(PremiumAdapterConfig):
    pass


@dataclass(frozen=True)
class ClaudeAdapterRequest(PremiumAdapterRequest):
    pass


@dataclass(frozen=True)
class ClaudePromptContract(PremiumPromptContract):
    pass


@dataclass(frozen=True)
class ClaudeResponseSchemaValidationResult(PremiumResponseSchemaValidationResult):
    pass


@dataclass(frozen=True)
class ClaudeAdapterResult(PremiumAdapterResult):
    pass


class MockClaudeClient(MockPremiumClient):
    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        super().__init__(provider_name=CLAUDE_PROVIDER_NAME, responses=responses)


class ClaudeExtractionAdapter(PremiumExtractionAdapter):
    provider_name = CLAUDE_PROVIDER_NAME
    adapter_name = "ClaudeExtractionAdapter"
    default_model_name = CLAUDE_DEFAULT_MODEL
    credential_env_var_name = CLAUDE_CREDENTIAL_ENV_VAR
    credential_env_var_aliases = CLAUDE_CREDENTIAL_ENV_VAR_ALIASES
    prompt_contract_version = CLAUDE_PROMPT_CONTRACT_VERSION
    schema_contract_version = CLAUDE_SCHEMA_CONTRACT_VERSION
    config_cls = ClaudeAdapterConfig
    request_cls = ClaudeAdapterRequest
    result_cls = ClaudeAdapterResult
    prompt_contract_cls = ClaudePromptContract
    schema_result_cls = ClaudeResponseSchemaValidationResult


def build_claude_adapter_status(
    *,
    selected_provider: str,
    environ: Mapping[str, str] | None = None,
    real_provider_execution_block_reason: str = "real_provider_execution_disabled_by_policy",
) -> dict[str, Any]:
    return build_premium_adapter_status(
        ClaudeExtractionAdapter(),
        selected_provider=selected_provider,
        environ=environ,
        real_provider_execution_block_reason=real_provider_execution_block_reason,
    )


def run_claude_mock_extraction_preview(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    return run_premium_mock_extraction_preview(
        ClaudeExtractionAdapter(client=MockClaudeClient()), environ=environ
    )


def build_mock_claude_responses() -> dict[str, dict[str, Any]]:
    return build_mock_premium_responses()


__all__ = [
    "CLAUDE_PROVIDER_NAME",
    "CLAUDE_DEFAULT_MODEL",
    "CLAUDE_CREDENTIAL_ENV_VAR",
    "CLAUDE_CREDENTIAL_ENV_VAR_ALIASES",
    "CLAUDE_PROMPT_CONTRACT_VERSION",
    "CLAUDE_SCHEMA_CONTRACT_VERSION",
    "ALLOWED_PAYLOAD_TYPE",
    "FORBIDDEN_PAYLOAD_TYPES",
    "ClaudeAdapterConfig",
    "ClaudeAdapterRequest",
    "ClaudePromptContract",
    "ClaudeResponseSchemaValidationResult",
    "ClaudeAdapterResult",
    "ClaudeExtractionAdapter",
    "MockClaudeClient",
    "build_claude_adapter_status",
    "run_claude_mock_extraction_preview",
    "build_mock_claude_responses",
]
