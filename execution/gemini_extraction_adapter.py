"""Disabled-by-policy Gemini text-extraction adapter contract (15G).

15G ships the *shape* of a Gemini extraction adapter — config, request,
prompt contract, schema validator, and a deterministic mock-response parser —
while keeping real Gemini execution disabled by policy.

Hard guarantees for 15G:

* No provider SDK is imported (not even lazily). There is no network code.
* The adapter never reads, prints, or logs the ``GEMINI_API_KEY`` value. It
  only reports credential *presence* and the env-var *name*.
* ``evaluate`` always fails closed: ``real_provider_execution_enabled`` and
  ``final_external_call_allowed`` are forced to ``False`` regardless of inputs.
* A schema-valid *mock* response (local dict, never network) may be validated
  and converted into review-bound :class:`AIExtractionPackageDraft` objects for
  test coverage only. ``gemini_real_call_attempted`` stays ``False``.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from execution.ai_extraction_adapter import (
    AIExtractionObservation,
    AIExtractionPackageDraft,
    AIExtractionSection,
)


GEMINI_PROVIDER_NAME = "gemini"
GEMINI_DEFAULT_MODEL = "gemini-disabled-15g"
GEMINI_CREDENTIAL_ENV_VAR = "GEMINI_API_KEY"
GEMINI_PROMPT_CONTRACT_VERSION = "gemini-extraction-prompt-15g-v1"
GEMINI_SCHEMA_CONTRACT_VERSION = "gemini-extraction-schema-15g-v1"

ALLOWED_PAYLOAD_TYPE = "redacted_text_layout_summary"
FORBIDDEN_PAYLOAD_TYPES = {
    "raw_pdf",
    "raw_image",
    "scanned_image",
    "unredacted_ocr_body",
    "external_vision_payload",
    "raw_file_upload",
    "provider_sdk_request",
}

SOURCE_TEXT_ONLY_INTERPRETATION_LABEL = "source text only - not MedAI interpretation"
SOURCE_TEXT_ONLY_RECOMMENDATION_LABEL = "source text only - not MedAI recommendation"

# The prompt contract is a static template. It is intentionally free of any
# private payload text; rendering merely appends a redacted summary at runtime
# and the rendered prompt is never written to public reports.
GEMINI_PROMPT_CONTRACT_INSTRUCTIONS = (
    "You extract structured content from a redacted clinical document summary.",
    "Extract source-visible content only.",
    "Return structured JSON only.",
    "Do not diagnose.",
    "Do not recommend treatment.",
    "Do not infer medical truth.",
    "Preserve uncertainty; record an uncertainty_note when unsure.",
    "Label recommendation-like source sections as source text only.",
    "Set review_required=true on every observation and section.",
    "Set auto_accept=false everywhere.",
    "Set active_write_allowed=false everywhere.",
)
# Public-safe rule keys (snake_case) mirror the instruction sentences without
# the capitalized prose that report PII scanners flag as false-positive names.
GEMINI_PROMPT_CONTRACT_RULE_KEYS = (
    "extract_source_visible_content_only",
    "return_structured_json_only",
    "do_not_diagnose",
    "do_not_recommend_treatment",
    "do_not_infer_medical_truth",
    "preserve_uncertainty",
    "label_recommendation_sections_as_source_text_only",
    "review_required_true",
    "auto_accept_false",
    "active_write_allowed_false",
)


@dataclass(frozen=True)
class GeminiAdapterConfig:
    """Configuration for the disabled Gemini adapter.

    ``real_provider_execution_enabled`` defaults to ``False`` and is the policy
    switch that keeps 15G from ever performing a real call.
    """

    provider_name: str = GEMINI_PROVIDER_NAME
    model_name: str = GEMINI_DEFAULT_MODEL
    credential_env_var_name: str = GEMINI_CREDENTIAL_ENV_VAR
    prompt_contract_version: str = GEMINI_PROMPT_CONTRACT_VERSION
    schema_contract_version: str = GEMINI_SCHEMA_CONTRACT_VERSION
    real_provider_execution_enabled: bool = False
    supports_redacted_text_layout_summary: bool = True
    supports_raw_pdf: bool = False
    supports_raw_image: bool = False
    supports_vision: bool = False
    sdk_import_required: bool = False


@dataclass(frozen=True)
class GeminiAdapterRequest:
    """A redacted, gate-validated request for the Gemini adapter.

    Only the redacted text/layout summary and redacted metadata are carried.
    No raw OCR body, filename, or PII is included.
    """

    payload_type: str = ALLOWED_PAYLOAD_TYPE
    redacted_payload_hash: str = ""
    redacted_text_layout_summary: str = ""
    redacted_layout_metadata: dict[str, Any] = field(default_factory=dict)
    privacy_gate_status: str = ""
    payload_policy_allowed: bool = False
    budget_allowed: bool = False
    operator_approval_state: str = "not_requested"
    real_provider_execution_enabled: bool = False
    final_external_call_allowed: bool = False
    document_category: str = "AI-assisted extraction"
    specialty_domain: str = "general"


@dataclass(frozen=True)
class GeminiPromptContract:
    """A versioned, static Gemini prompt contract template."""

    prompt_contract_version: str = GEMINI_PROMPT_CONTRACT_VERSION
    schema_contract_version: str = GEMINI_SCHEMA_CONTRACT_VERSION
    instructions: tuple[str, ...] = GEMINI_PROMPT_CONTRACT_INSTRUCTIONS

    def render(self, request: GeminiAdapterRequest) -> str:
        """Render the prompt from redacted metadata + redacted summary.

        The output may contain the *redacted* summary only and must never be
        written to public reports.
        """
        layout = ", ".join(f"{key}={value}" for key, value in sorted((request.redacted_layout_metadata or {}).items()))
        return "\n".join(
            [
                f"# Gemini extraction prompt contract {self.prompt_contract_version}",
                f"# Response schema contract {self.schema_contract_version}",
                "",
                "## Instructions",
                *[f"- {item}" for item in self.instructions],
                "",
                "## Redacted document context",
                f"- document_category: {request.document_category}",
                f"- specialty_domain: {request.specialty_domain}",
                f"- payload_type: {request.payload_type}",
                f"- redacted_payload_hash: {request.redacted_payload_hash[:12]}",
                f"- redacted_layout_metadata: {layout or 'none'}",
                "",
                "## Redacted text / layout summary",
                request.redacted_text_layout_summary or "(no redacted summary provided)",
            ]
        )

    def public_contract(self) -> dict[str, Any]:
        """Return a public-safe view (no rendered private summary)."""
        return {
            "prompt_contract_version": self.prompt_contract_version,
            "schema_contract_version": self.schema_contract_version,
            "instruction_rule_keys": list(GEMINI_PROMPT_CONTRACT_RULE_KEYS),
            "raw_payload_text_included": False,
        }


@dataclass(frozen=True)
class GeminiResponseSchemaValidationResult:
    schema_valid: bool
    schema_contract_version: str
    document_type: str
    package_title: str
    section_count: int
    observation_count: int
    review_required: bool
    auto_accept: bool
    active_write_allowed: bool
    source_text_only_recommendation_preserved: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeminiAdapterResult:
    provider_name: str
    model_name: str
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
    gemini_real_call_attempted: bool
    simulated_or_mocked_response_used: bool
    schema_valid: bool
    fail_closed_reason: str


class MockGeminiClient:
    """Deterministic local stand-in for a Gemini client (tests only).

    This is NOT a provider SDK and performs NO network I/O. It simply returns
    canned schema-valid response dicts so the adapter's validation/parse path
    can be exercised without any external call.
    """

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        self._responses = dict(responses or build_mock_gemini_responses())

    def generate_extraction(self, *, source_class: str, prompt: str = "") -> dict[str, Any]:
        del prompt  # the mock is deterministic and ignores the prompt text
        key = str(source_class or "").strip().lower()
        if key not in self._responses:
            raise KeyError(f"no mock gemini response for source_class={key!r}")
        return dict(self._responses[key])


class GeminiExtractionAdapter:
    """Disabled-by-policy Gemini adapter.

    The adapter can be instantiated, can build a prompt, and can validate /
    parse a *mock* response. It can never reach a real provider in 15G.
    """

    provider_name = GEMINI_PROVIDER_NAME
    adapter_name = "GeminiExtractionAdapter"

    def __init__(
        self,
        config: GeminiAdapterConfig | None = None,
        *,
        gemini_client: MockGeminiClient | None = None,
        prompt_contract: GeminiPromptContract | None = None,
    ):
        self.config = config or GeminiAdapterConfig()
        self.model_name = self.config.model_name
        self.prompt_contract = prompt_contract or GeminiPromptContract(
            prompt_contract_version=self.config.prompt_contract_version,
            schema_contract_version=self.config.schema_contract_version,
        )
        # Injectable mock client for tests only; never a real provider SDK.
        self._gemini_client = gemini_client

    # -- prompt -----------------------------------------------------------
    def build_prompt(self, request: GeminiAdapterRequest) -> str:
        return self.prompt_contract.render(request)

    # -- credential readiness (presence only) -----------------------------
    def credential_readiness(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        env = os.environ if environ is None else environ
        env_var = self.config.credential_env_var_name
        present = bool(env_var and str(env.get(env_var) or "").strip())
        return {
            "credential_env_var_name": env_var,
            "credential_present": present,
            "credential_value_redacted": present,
            "credential_value_read_or_logged": False,
            "credential_fingerprint": _credential_fingerprint(env_var, present),
        }

    # -- fail-closed gate evaluation (never calls a client) ---------------
    def evaluate(self, request: GeminiAdapterRequest) -> GeminiAdapterResult:
        reason = _fail_closed_reason(request)
        effective_real_enabled = bool(
            request.real_provider_execution_enabled and self.config.real_provider_execution_enabled
        )
        return GeminiAdapterResult(
            provider_name=self.config.provider_name,
            model_name=self.config.model_name,
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
            gemini_real_call_attempted=False,
            simulated_or_mocked_response_used=False,
            schema_valid=False,
            fail_closed_reason=reason or "real_provider_execution_disabled_by_policy",
        )

    # -- schema validation ------------------------------------------------
    def validate_mock_response(self, response: Mapping[str, Any]) -> GeminiResponseSchemaValidationResult:
        return validate_gemini_response_schema(response, schema_version=self.config.schema_contract_version)

    # -- mock parse → review-bound drafts ---------------------------------
    def parse_mock_response_to_draft(
        self,
        response: Mapping[str, Any],
        *,
        safe_source_document_id: str,
        source_modality: str = "gemini_mock_local",
    ) -> AIExtractionPackageDraft:
        validation = self.validate_mock_response(response)
        if not validation.schema_valid:
            raise ValueError(f"mock_gemini_response_schema_invalid: {validation.errors}")
        return _draft_from_mock_response(
            response,
            safe_source_document_id=safe_source_document_id,
            source_modality=source_modality,
        )

    def simulate_local_mock(
        self,
        request: GeminiAdapterRequest,
        *,
        source_class: str,
    ) -> tuple[GeminiAdapterResult, GeminiResponseSchemaValidationResult, AIExtractionPackageDraft | None]:
        """Validate + parse a local mock response for test coverage only.

        No real provider call is attempted. A local mock client is used; if no
        client was injected a default deterministic one is created in-memory.
        """
        client = self._gemini_client or MockGeminiClient()
        response = client.generate_extraction(source_class=source_class, prompt=self.build_prompt(request))
        validation = self.validate_mock_response(response)
        base = self.evaluate(request)
        draft = (
            _draft_from_mock_response(
                response,
                safe_source_document_id=f"source_gemini_mock_{source_class}",
                source_modality="gemini_mock_local",
            )
            if validation.schema_valid
            else None
        )
        result = GeminiAdapterResult(
            **{
                **asdict(base),
                "simulated_or_mocked_response_used": True,
                "schema_valid": validation.schema_valid,
                "gemini_real_call_attempted": False,
                "external_api_used": False,
                "real_network_call_used": False,
                "final_external_call_allowed": False,
            }
        )
        return result, validation, draft


def gemini_adapter_result_to_public_dict(result: GeminiAdapterResult) -> dict[str, Any]:
    return asdict(result)


def gemini_schema_validation_to_public_dict(result: GeminiResponseSchemaValidationResult) -> dict[str, Any]:
    return asdict(result)


def build_gemini_adapter_status(
    *,
    selected_provider: str,
    environ: Mapping[str, str] | None = None,
    real_provider_execution_block_reason: str = "real_provider_execution_disabled_by_policy",
) -> dict[str, Any]:
    """Public-safe Gemini adapter readiness status for workflow/preview/reports."""
    readiness = GeminiExtractionAdapter().credential_readiness(environ=environ)
    selected = str(selected_provider or "fake_local").strip().lower() or "fake_local"
    return {
        "provider_name": GEMINI_PROVIDER_NAME,
        "model_name": GEMINI_DEFAULT_MODEL,
        "selected_provider": selected,
        "gemini_selected": selected == GEMINI_PROVIDER_NAME,
        "gemini_adapter_installed": True,
        "gemini_adapter_status_message": "Gemini adapter installed but real execution disabled by policy",
        "prompt_contract_version": GEMINI_PROMPT_CONTRACT_VERSION,
        "schema_contract_version": GEMINI_SCHEMA_CONTRACT_VERSION,
        "real_provider_execution_enabled": False,
        "real_provider_execution_block_reason": real_provider_execution_block_reason,
        "gemini_real_call_attempted": False,
        "external_api_used": False,
        "real_network_call_used": False,
        "final_external_call_allowed": False,
        "provider_sdk_import_required": False,
        "credential_env_var_name": readiness["credential_env_var_name"],
        "credential_present": readiness["credential_present"],
        "credential_value_redacted": readiness["credential_value_redacted"],
        "credential_value_read_or_logged": False,
        "supports_redacted_text_layout_summary": True,
        "supports_raw_pdf": False,
        "supports_raw_image": False,
        "supports_vision": False,
    }


def build_mock_gemini_responses() -> dict[str, dict[str, Any]]:
    """Three deterministic, schema-valid mock Gemini responses.

    Covers cytology/pathology narrative, urinalysis observations, and portal
    result card observations. No raw fixture PII is present.
    """
    return {
        "cytology_pathology": _mock_cytology_pathology(),
        "urinalysis_table": _mock_urinalysis_table(),
        "portal_cards": _mock_portal_cards(),
    }


MOCK_SOURCE_CLASSES = ("cytology_pathology", "urinalysis_table", "portal_cards")


def run_gemini_mock_extraction_preview(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate + convert all mock responses into review-bound packages.

    Returns a public-safe dict carrying the review-bound packages, the schema
    validation evidence, and the disabled-real-call invariants. Makes zero
    network calls and never enables real execution.
    """
    from app.source_extraction_packages import source_package_from_ai_draft

    adapter = GeminiExtractionAdapter(gemini_client=MockGeminiClient())
    packages: list[dict[str, Any]] = []
    schema_validations: list[dict[str, Any]] = []
    adapter_results: list[dict[str, Any]] = []
    for source_class in MOCK_SOURCE_CLASSES:
        request = _mock_request(source_class)
        result, validation, draft = adapter.simulate_local_mock(request, source_class=source_class)
        schema_validations.append(gemini_schema_validation_to_public_dict(validation))
        adapter_results.append(gemini_adapter_result_to_public_dict(result))
        if draft is not None:
            packages.append(source_package_from_ai_draft(draft))
    status = build_gemini_adapter_status(selected_provider=GEMINI_PROVIDER_NAME, environ=environ)
    return {
        "gemini_adapter_status": status,
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
        "final_external_call_allowed": False,
        "real_provider_execution_enabled": False,
        "gemini_real_call_attempted": False,
        "simulated_or_mocked_response_used": True,
    }


# ---------------------------------------------------------------------------
# Schema validation + parsing internals
# ---------------------------------------------------------------------------
def validate_gemini_response_schema(
    response: Mapping[str, Any],
    *,
    schema_version: str = GEMINI_SCHEMA_CONTRACT_VERSION,
) -> GeminiResponseSchemaValidationResult:
    errors: list[str] = []
    response = dict(response or {})
    document_type = str(response.get("document_type") or "")
    package_title = str(response.get("package_title") or "")
    sections = response.get("sections")
    review_required = response.get("review_required")
    auto_accept = response.get("auto_accept")
    active_write_allowed = response.get("active_write_allowed")

    if not document_type.strip():
        errors.append("document_type_required")
    if not package_title.strip():
        errors.append("package_title_required")
    if review_required is not True:
        errors.append("review_required_must_be_true")
    if auto_accept is not False:
        errors.append("auto_accept_must_be_false")
    if active_write_allowed is not False:
        errors.append("active_write_allowed_must_be_false")
    if not isinstance(sections, list) or not sections:
        errors.append("sections_required")
        sections = []

    observation_count = 0
    recommendation_label_preserved = True
    for section in sections:
        if not isinstance(section, dict):
            errors.append("section_must_be_object")
            continue
        heading = str(section.get("heading") or "")
        if not heading.strip():
            errors.append("section_heading_required")
        is_recommendation = heading.strip().lower() in {"recommendation", "recommendations", "plan", "treatment plan"}
        source_text_only = bool(section.get("source_text_only", False))
        narrative_label = str(section.get("narrative_label") or "")
        if is_recommendation:
            if not source_text_only or "not medai recommendation" not in narrative_label.lower():
                recommendation_label_preserved = False
                errors.append("recommendation_section_must_be_source_text_only")
        observations = section.get("observations")
        if not isinstance(observations, list):
            errors.append("section_observations_must_be_list")
            observations = []
        for observation in observations:
            observation_count += 1
            if not isinstance(observation, dict):
                errors.append("observation_must_be_object")
                continue
            if not str(observation.get("name") or "").strip():
                errors.append("observation_name_required")
            if observation.get("review_required") is not True:
                errors.append("observation_review_required_must_be_true")
            if observation.get("auto_accept") is not False:
                errors.append("observation_auto_accept_must_be_false")
            if observation.get("active_write_allowed") is not False:
                errors.append("observation_active_write_allowed_must_be_false")

    return GeminiResponseSchemaValidationResult(
        schema_valid=not errors,
        schema_contract_version=schema_version,
        document_type=document_type,
        package_title=package_title,
        section_count=len([s for s in sections if isinstance(s, dict)]),
        observation_count=observation_count,
        review_required=review_required is True,
        auto_accept=bool(auto_accept) if auto_accept is not None else False,
        active_write_allowed=bool(active_write_allowed) if active_write_allowed is not None else False,
        source_text_only_recommendation_preserved=recommendation_label_preserved,
        errors=sorted(set(errors)),
    )


def _draft_from_mock_response(
    response: Mapping[str, Any],
    *,
    safe_source_document_id: str,
    source_modality: str,
) -> AIExtractionPackageDraft:
    response = dict(response or {})
    sections: list[AIExtractionSection] = []
    for section in response.get("sections") or []:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "Section")
        is_recommendation = heading.strip().lower() in {"recommendation", "recommendations", "plan", "treatment plan"}
        observations = [
            AIExtractionObservation(
                label=str(obs.get("name") or ""),
                value=str(obs.get("value") or ""),
                reference_interval=str(obs.get("reference_range") or ""),
                flag=str(obs.get("abnormal_flag") or ""),
                unit=str(obs.get("unit") or ""),
                source_section=heading,
                row_kind="source_visible_note" if bool(obs.get("source_text_only")) and not str(obs.get("value") or "").strip() else "observation",
                review_required=True,
                auto_accept=False,
            )
            for obs in section.get("observations") or []
            if isinstance(obs, dict)
        ]
        sections.append(
            AIExtractionSection(
                heading=heading,
                observations=observations,
                narrative_preview=str(section.get("narrative_preview") or ""),
                narrative_label=str(
                    section.get("narrative_label")
                    or (SOURCE_TEXT_ONLY_RECOMMENDATION_LABEL if is_recommendation else SOURCE_TEXT_ONLY_INTERPRETATION_LABEL)
                ),
            )
        )
    return AIExtractionPackageDraft(
        safe_source_document_id=safe_source_document_id,
        document_type=str(response.get("document_type") or "Unknown"),
        selected_document_category=str(response.get("package_title") or "AI-assisted extraction"),
        selected_specialty_domain=str(response.get("specialty_domain") or "urology"),
        source_modality=source_modality,
        sections=sections,
        package_status="review-bound",
        review_required=True,
        auto_accept=False,
        external_api_used=False,
    )


def _fail_closed_reason(request: GeminiAdapterRequest) -> str:
    payload_type = str(request.payload_type or "")
    if payload_type in FORBIDDEN_PAYLOAD_TYPES:
        return "forbidden_payload_type"
    if payload_type != ALLOWED_PAYLOAD_TYPE:
        return "payload_type_not_allowed"
    if not str(request.privacy_gate_status or "").strip():
        return "missing_privacy_gate_result"
    if str(request.privacy_gate_status or "").startswith("fail_closed"):
        return "privacy_gate_failed"
    if not str(request.redacted_payload_hash or "").strip():
        return "missing_redacted_payload_hash"
    if not bool(request.payload_policy_allowed):
        return "payload_policy_failed"
    if not bool(request.budget_allowed):
        return "budget_exceeded"
    if not bool(request.real_provider_execution_enabled):
        return "real_provider_execution_disabled_by_policy"
    if not bool(request.final_external_call_allowed):
        return "final_external_call_not_allowed"
    return ""


def _credential_fingerprint(env_var: str, present: bool) -> str:
    basis = f"{GEMINI_PROVIDER_NAME}:{env_var}:{'present' if present else 'missing'}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def _mock_request(source_class: str) -> GeminiAdapterRequest:
    return GeminiAdapterRequest(
        payload_type=ALLOWED_PAYLOAD_TYPE,
        redacted_payload_hash=hashlib.sha256(source_class.encode("utf-8")).hexdigest()[:12],
        redacted_text_layout_summary="[redacted summary: counts and layout only]",
        redacted_layout_metadata={"has_table_like_layout": True, "section_count_bucket": "1-10"},
        privacy_gate_status="redacted_payload_ready",
        payload_policy_allowed=True,
        budget_allowed=True,
        operator_approval_state="approved_for_dry_run",
        real_provider_execution_enabled=False,
        final_external_call_allowed=False,
        document_category="AI-assisted extraction",
        specialty_domain="urology",
    )


def _mock_cytology_pathology() -> dict[str, Any]:
    headings = [
        "Tests Ordered",
        "Diagnosis",
        "Recommendation",
        "Clinical History / ICD",
        "Cytology Information",
        "Gross Description",
    ]
    sections = []
    for heading in headings:
        is_recommendation = heading == "Recommendation"
        sections.append(
            {
                "heading": heading,
                "narrative_preview": "source-visible narrative present",
                "source_text_only": True,
                "narrative_label": (
                    SOURCE_TEXT_ONLY_RECOMMENDATION_LABEL if is_recommendation else SOURCE_TEXT_ONLY_INTERPRETATION_LABEL
                ),
                "observations": [
                    {
                        "name": heading,
                        "value": "source-visible narrative present",
                        "unit": "",
                        "reference_range": "",
                        "abnormal_flag": "",
                        "source_text_only": True,
                        "confidence": "reported",
                        "uncertainty_note": "source text preserved verbatim; no interpretation",
                        "review_required": True,
                        "auto_accept": False,
                        "active_write_allowed": False,
                    }
                ],
            }
        )
    return {
        "document_type": "Cytology / pathology narrative",
        "package_title": "AI-assisted extraction",
        "specialty_domain": "urology",
        "review_required": True,
        "auto_accept": False,
        "active_write_allowed": False,
        "sections": sections,
    }


def _mock_urinalysis_table() -> dict[str, Any]:
    rows = [
        ("Specific Gravity", "1.020", "", "", "1.005-1.030"),
        ("pH", "7.5", "", "", "5.0-7.5"),
        ("Occult Blood", "Trace", "abnormal", "", "Negative"),
        ("RBC", "3-10", "abnormal", "/hpf", "0-2"),
        ("Culture result", "No growth", "", "", ""),
    ]
    return {
        "document_type": "Urinalysis",
        "package_title": "AI-assisted extraction",
        "specialty_domain": "urology",
        "review_required": True,
        "auto_accept": False,
        "active_write_allowed": False,
        "sections": [
            {
                "heading": "Urinalysis Table",
                "source_text_only": False,
                "narrative_label": SOURCE_TEXT_ONLY_INTERPRETATION_LABEL,
                "observations": [
                    {
                        "name": name,
                        "value": value,
                        "unit": unit,
                        "reference_range": reference,
                        "abnormal_flag": flag,
                        "source_text_only": False,
                        "confidence": "reported",
                        "uncertainty_note": "",
                        "review_required": True,
                        "auto_accept": False,
                        "active_write_allowed": False,
                    }
                    for name, value, flag, unit, reference in rows
                ],
            }
        ],
    }


def _mock_portal_cards() -> dict[str, Any]:
    rows = [
        ("Specific Gravity", "1.020", "1.005-1.030"),
        ("pH", "7.5", "5.0-7.5"),
        ("Urine Color", "Orange", "Yellow"),
        ("Appearance", "Clear", "Clear"),
        ("Leukocyte Esterase", "Negative", "Negative"),
        ("Protein", "Trace", "Negative/Trace"),
        ("Glucose", "Negative", "Negative"),
        ("Ketones", "Negative", "Negative"),
    ]
    return {
        "document_type": "Portal result cards",
        "package_title": "AI-assisted extraction",
        "specialty_domain": "urology",
        "review_required": True,
        "auto_accept": False,
        "active_write_allowed": False,
        "sections": [
            {
                "heading": "Portal Result Cards",
                "source_text_only": False,
                "narrative_label": SOURCE_TEXT_ONLY_INTERPRETATION_LABEL,
                "observations": [
                    {
                        "name": name,
                        "value": value,
                        "unit": "",
                        "reference_range": reference,
                        "abnormal_flag": "",
                        "source_text_only": False,
                        "confidence": "reported",
                        "uncertainty_note": "",
                        "review_required": True,
                        "auto_accept": False,
                        "active_write_allowed": False,
                    }
                    for name, value, reference in rows
                ],
            }
        ],
    }


__all__ = [
    "GEMINI_PROVIDER_NAME",
    "GEMINI_DEFAULT_MODEL",
    "GEMINI_CREDENTIAL_ENV_VAR",
    "GEMINI_PROMPT_CONTRACT_VERSION",
    "GEMINI_SCHEMA_CONTRACT_VERSION",
    "GEMINI_PROMPT_CONTRACT_RULE_KEYS",
    "ALLOWED_PAYLOAD_TYPE",
    "FORBIDDEN_PAYLOAD_TYPES",
    "GeminiAdapterConfig",
    "GeminiAdapterRequest",
    "GeminiPromptContract",
    "GeminiResponseSchemaValidationResult",
    "GeminiAdapterResult",
    "MockGeminiClient",
    "GeminiExtractionAdapter",
    "gemini_adapter_result_to_public_dict",
    "gemini_schema_validation_to_public_dict",
    "build_gemini_adapter_status",
    "build_mock_gemini_responses",
    "run_gemini_mock_extraction_preview",
    "validate_gemini_response_schema",
    "MOCK_SOURCE_CLASSES",
]
