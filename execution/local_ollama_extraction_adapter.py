"""Disabled-by-policy Local/Ollama text-extraction adapter contract (15J).

Mirrors the safe Gemini (15G) and premium (15I) adapter patterns for a local
Ollama model, built on the shared provider-neutral premium adapter contract.

Local/Ollama differs from the cloud providers: there is no API credential, and
a "real" call would be a localhost HTTP request (e.g. http://localhost:11434)
and/or a subprocess invocation. 15J keeps real local-model execution disabled
by policy:

* No provider SDK is imported (not even lazily). There is no network code, no
  socket/HTTP client, and no subprocess invocation anywhere in this module.
* ``ollama_base_url`` is carried as NON-SECRET config only and is never called.
* ``evaluate`` always fails closed: ``real_provider_execution_enabled`` and
  ``final_external_call_allowed`` are forced to ``False`` regardless of inputs;
  ``local_model_call_used``, ``subprocess_call_used`` and
  ``ollama_real_call_attempted`` are always ``False``.
* A schema-valid *mock* response (local dict, never network/subprocess) may be
  validated and converted into review-bound packages for test coverage only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from execution.ai_extraction_adapter import AIExtractionPackageDraft
from execution.premium_adapter_contract import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    MOCK_SOURCE_CLASSES,
    MockPremiumClient,
    PremiumAdapterConfig,
    PremiumAdapterRequest,
    PremiumExtractionAdapter,
    PremiumPromptContract,
    PremiumResponseSchemaValidationResult,
    build_premium_adapter_status,
    build_mock_premium_responses,
    draft_from_premium_response,
    fail_closed_reason,
    mock_request,
)

LOCAL_OLLAMA_PROVIDER_NAME = "local_ollama"
LOCAL_OLLAMA_DEFAULT_MODEL = "ollama-disabled-15j"
# Non-secret default base URL config. It is NEVER called in 15J.
LOCAL_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
LOCAL_OLLAMA_PROMPT_CONTRACT_VERSION = "local-ollama-extraction-prompt-15j-v1"
LOCAL_OLLAMA_SCHEMA_CONTRACT_VERSION = "local-ollama-extraction-schema-15j-v1"


@dataclass(frozen=True)
class LocalOllamaAdapterConfig(PremiumAdapterConfig):
    ollama_base_url: str = LOCAL_OLLAMA_DEFAULT_BASE_URL


@dataclass(frozen=True)
class LocalOllamaAdapterRequest(PremiumAdapterRequest):
    pass


@dataclass(frozen=True)
class LocalOllamaPromptContract(PremiumPromptContract):
    pass


@dataclass(frozen=True)
class LocalOllamaResponseSchemaValidationResult(PremiumResponseSchemaValidationResult):
    pass


@dataclass(frozen=True)
class LocalOllamaAdapterResult:
    provider_name: str
    model_name: str
    ollama_base_url: str
    redacted_payload_hash: str
    payload_type: str
    prompt_contract_version: str
    schema_contract_version: str
    privacy_gate_status: str
    payload_policy_allowed: bool
    budget_allowed: bool
    operator_approval_state: str
    real_provider_execution_enabled: bool
    final_external_call_allowed: bool
    external_api_used: bool
    real_network_call_used: bool
    local_model_call_used: bool
    subprocess_call_used: bool
    ollama_real_call_attempted: bool
    simulated_or_mocked_response_used: bool
    schema_valid: bool
    fail_closed_reason: str


class MockLocalOllamaClient(MockPremiumClient):
    """Deterministic local stand-in (tests only). No HTTP and no subprocess use."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        super().__init__(provider_name=LOCAL_OLLAMA_PROVIDER_NAME, responses=responses)


class LocalOllamaExtractionAdapter(PremiumExtractionAdapter):
    provider_name = LOCAL_OLLAMA_PROVIDER_NAME
    adapter_name = "LocalOllamaExtractionAdapter"
    default_model_name = LOCAL_OLLAMA_DEFAULT_MODEL
    credential_env_var_name = ""  # local model: no API credential
    credential_env_var_aliases = ()
    prompt_contract_version = LOCAL_OLLAMA_PROMPT_CONTRACT_VERSION
    schema_contract_version = LOCAL_OLLAMA_SCHEMA_CONTRACT_VERSION
    config_cls = LocalOllamaAdapterConfig
    request_cls = LocalOllamaAdapterRequest
    result_cls = LocalOllamaAdapterResult
    prompt_contract_cls = LocalOllamaPromptContract
    schema_result_cls = LocalOllamaResponseSchemaValidationResult

    @property
    def ollama_base_url(self) -> str:
        return getattr(self.config, "ollama_base_url", LOCAL_OLLAMA_DEFAULT_BASE_URL)

    # -- fail-closed gate evaluation (no HTTP, no subprocess, no client) ---
    def evaluate(self, request: PremiumAdapterRequest) -> LocalOllamaAdapterResult:
        reason = fail_closed_reason(request)
        effective_real_enabled = bool(
            request.real_provider_execution_enabled and self.config.real_provider_execution_enabled
        )
        return LocalOllamaAdapterResult(
            provider_name=self.config.provider_name,
            model_name=self.config.model_name,
            ollama_base_url=self.ollama_base_url,
            redacted_payload_hash=request.redacted_payload_hash[:12],
            payload_type=request.payload_type,
            prompt_contract_version=self.config.prompt_contract_version,
            schema_contract_version=self.config.schema_contract_version,
            privacy_gate_status=request.privacy_gate_status,
            payload_policy_allowed=bool(request.payload_policy_allowed),
            budget_allowed=bool(request.budget_allowed),
            operator_approval_state=request.operator_approval_state,
            real_provider_execution_enabled=effective_real_enabled,
            final_external_call_allowed=False,
            external_api_used=False,
            real_network_call_used=False,
            local_model_call_used=False,
            subprocess_call_used=False,
            ollama_real_call_attempted=False,
            simulated_or_mocked_response_used=False,
            schema_valid=False,
            fail_closed_reason=reason or "real_provider_execution_disabled_by_policy",
        )

    def simulate_local_mock(
        self,
        request: PremiumAdapterRequest,
        *,
        source_class: str,
    ) -> tuple[LocalOllamaAdapterResult, LocalOllamaResponseSchemaValidationResult, AIExtractionPackageDraft | None]:
        client = self._client or MockLocalOllamaClient()
        response = client.generate_extraction(source_class=source_class, prompt=self.build_prompt(request))
        validation = self.validate_mock_response(response)
        base = self.evaluate(request)
        draft = (
            draft_from_premium_response(
                response,
                safe_source_document_id=f"source_{self.provider_name}_mock_{source_class}",
                source_modality=f"{self.provider_name}_mock_local",
            )
            if validation.schema_valid
            else None
        )
        result = LocalOllamaAdapterResult(
            **{
                **asdict(base),
                "simulated_or_mocked_response_used": True,
                "schema_valid": validation.schema_valid,
                "ollama_real_call_attempted": False,
                "local_model_call_used": False,
                "subprocess_call_used": False,
                "external_api_used": False,
                "real_network_call_used": False,
                "final_external_call_allowed": False,
            }
        )
        return result, validation, draft


def build_local_ollama_adapter_status(
    *,
    selected_provider: str,
    environ: Mapping[str, str] | None = None,
    real_provider_execution_block_reason: str = "real_provider_execution_disabled_by_policy",
) -> dict[str, Any]:
    adapter = LocalOllamaExtractionAdapter()
    status = build_premium_adapter_status(
        adapter,
        selected_provider=selected_provider,
        environ=environ,
        real_provider_execution_block_reason=real_provider_execution_block_reason,
    )
    # Local/Ollama specifics: non-secret runtime config + extra invariants.
    status.update(
        {
            "adapter_status_message": "Local/Ollama adapter installed but real execution disabled by policy",
            "no_local_model_call_notice": "No local model call was made",
            "ollama_base_url": adapter.ollama_base_url,
            "local_model_call_used": False,
            "subprocess_call_used": False,
            "ollama_real_call_attempted": False,
            "credential_required": False,
        }
    )
    return status


def run_local_ollama_mock_extraction_preview(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    from app.source_extraction_packages import source_package_from_ai_draft

    adapter = LocalOllamaExtractionAdapter(client=MockLocalOllamaClient())
    packages: list[dict[str, Any]] = []
    schema_validations: list[dict[str, Any]] = []
    adapter_results: list[dict[str, Any]] = []
    for source_class in MOCK_SOURCE_CLASSES:
        request = mock_request(source_class)
        result, validation, draft = adapter.simulate_local_mock(request, source_class=source_class)
        schema_validations.append(asdict(validation))
        adapter_results.append(asdict(result))
        if draft is not None:
            packages.append(source_package_from_ai_draft(draft))
    status = build_local_ollama_adapter_status(
        selected_provider=LOCAL_OLLAMA_PROVIDER_NAME, environ=environ
    )
    return {
        "provider_name": LOCAL_OLLAMA_PROVIDER_NAME,
        "adapter_status": status,
        "schema_validations": schema_validations,
        "adapter_results": adapter_results,
        "packages": packages,
        "review_bound_package_count": len(packages),
        "schema_valid_count": sum(1 for item in schema_validations if item["schema_valid"]),
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "external_api_used": False,
        "real_network_call_used": False,
        "local_model_call_used": False,
        "subprocess_call_used": False,
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "ollama_real_call_attempted": False,
        "simulated_or_mocked_response_used": True,
    }


def build_mock_local_ollama_responses() -> dict[str, dict[str, Any]]:
    return build_mock_premium_responses()


__all__ = [
    "LOCAL_OLLAMA_PROVIDER_NAME",
    "LOCAL_OLLAMA_DEFAULT_MODEL",
    "LOCAL_OLLAMA_DEFAULT_BASE_URL",
    "LOCAL_OLLAMA_PROMPT_CONTRACT_VERSION",
    "LOCAL_OLLAMA_SCHEMA_CONTRACT_VERSION",
    "ALLOWED_PAYLOAD_TYPE",
    "FORBIDDEN_PAYLOAD_TYPES",
    "LocalOllamaAdapterConfig",
    "LocalOllamaAdapterRequest",
    "LocalOllamaPromptContract",
    "LocalOllamaResponseSchemaValidationResult",
    "LocalOllamaAdapterResult",
    "LocalOllamaExtractionAdapter",
    "MockLocalOllamaClient",
    "build_local_ollama_adapter_status",
    "run_local_ollama_mock_extraction_preview",
    "build_mock_local_ollama_responses",
]
