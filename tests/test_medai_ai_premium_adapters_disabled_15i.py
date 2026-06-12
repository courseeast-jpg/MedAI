"""Focused tests for MEDAI-AI-PREMIUM-ADAPTERS-DISABLED-15I."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import AIExtractionPackageDraft, ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow
from execution.claude_extraction_adapter import (
    CLAUDE_CREDENTIAL_ENV_VAR,
    ClaudeAdapterConfig,
    ClaudeAdapterRequest,
    ClaudeExtractionAdapter,
    MockClaudeClient,
    build_mock_claude_responses,
    run_claude_mock_extraction_preview,
)
from execution.openai_extraction_adapter import (
    OPENAI_CREDENTIAL_ENV_VAR,
    OpenAIAdapterConfig,
    OpenAIAdapterRequest,
    OpenAIExtractionAdapter,
    MockOpenAIClient,
    build_mock_openai_responses,
    run_openai_mock_extraction_preview,
)
from execution.premium_adapter_contract import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    validate_premium_response_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
CLAUDE_SECRET = "claude-unit-test-secret-value-15i"
OPENAI_SECRET = "openai-unit-test-secret-value-15i"
RAW_FIXTURE_VALUES = [
    "Jane Example",
    "01/02/1970",
    "123456",
    "CY-2026-0001",
    "Park Medical Center",
    "Alice Clinician",
    "123 Main Street",
    "555-123-4567",
    "jane.example@example.com",
    "INS-ABC-12345",
    "06/10/2026",
]
DOCTRINE_PHRASES = [
    "Do not assign semantic/layout understanding to OCR or deterministic rules.",
    (
        "A MedAI extraction block is not successful because records were created. It is "
        "successful only if the operator can compare a source-faithful package against the "
        "original document quickly and safely."
    ),
]

ADAPTERS = [
    ("claude", ClaudeExtractionAdapter, ClaudeAdapterRequest, CLAUDE_CREDENTIAL_ENV_VAR, MockClaudeClient, build_mock_claude_responses),
    ("openai", OpenAIExtractionAdapter, OpenAIAdapterRequest, OPENAI_CREDENTIAL_ENV_VAR, MockOpenAIClient, build_mock_openai_responses),
]


def _passing_request(request_cls, **overrides):
    values = {
        "payload_type": ALLOWED_PAYLOAD_TYPE,
        "redacted_payload_hash": "deadbeef0001",
        "redacted_text_layout_summary": "[redacted summary]",
        "privacy_gate_status": "redacted_payload_ready",
        "payload_policy_allowed": True,
        "budget_allowed": True,
        "operator_approval_state": "approved_for_dry_run",
        "real_provider_execution_enabled": True,
        "final_external_call_allowed": True,
    }
    values.update(overrides)
    return request_cls(**values)


# 1-2. Adapters exist and instantiate without provider SDK import.
def test_adapters_instantiate_without_provider_sdk() -> None:
    assert ClaudeExtractionAdapter().provider_name == "claude"
    assert OpenAIExtractionAdapter().provider_name == "openai"
    for module in ("anthropic", "openai", "google.generativeai", "vertexai"):
        assert module not in sys.modules


# 3. Configs default to disabled real execution.
def test_configs_default_disabled_real_execution() -> None:
    for cfg in (ClaudeExtractionAdapter().config, OpenAIExtractionAdapter().config):
        assert cfg.real_provider_execution_enabled is False
        assert cfg.sdk_import_required is False
        assert cfg.supports_raw_pdf is False
        assert cfg.supports_raw_image is False
        assert cfg.supports_vision is False


# 4-5. No credential value required; env var name reported, value never exposed.
def test_credential_presence_only_no_value_exposed() -> None:
    claude = ClaudeExtractionAdapter().credential_readiness(environ={CLAUDE_CREDENTIAL_ENV_VAR: CLAUDE_SECRET})
    openai = OpenAIExtractionAdapter().credential_readiness(environ={OPENAI_CREDENTIAL_ENV_VAR: OPENAI_SECRET})
    assert claude["credential_env_var_name"] == "ANTHROPIC_API_KEY"
    assert openai["credential_env_var_name"] == "OPENAI_API_KEY"
    assert claude["credential_present"] is True and openai["credential_present"] is True
    assert claude["credential_value_read_or_logged"] is False
    assert CLAUDE_SECRET not in json.dumps(claude)
    assert OPENAI_SECRET not in json.dumps(openai)
    # no value required for tests
    assert ClaudeExtractionAdapter().credential_readiness(environ={})["credential_present"] is False
    assert OpenAIExtractionAdapter().credential_readiness(environ={})["credential_present"] is False


# 6-9. Raw PDF / image / vision payloads rejected for both adapters.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,_mock", ADAPTERS)
@pytest.mark.parametrize("payload_type", ["raw_pdf", "raw_image", "external_vision_payload"])
def test_adapters_reject_raw_and_vision_payloads(provider, adapter_cls, request_cls, _env, _client, _mock, payload_type) -> None:
    result = adapter_cls().evaluate(_passing_request(request_cls, payload_type=payload_type))
    assert result.fail_closed_reason == "forbidden_payload_type"
    assert result.provider_real_call_attempted is False
    assert result.final_external_call_allowed is False


# 10. Both adapters accept only redacted_text_layout_summary.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,_mock", ADAPTERS)
def test_accepts_only_redacted_text_layout_summary(provider, adapter_cls, request_cls, _env, _client, _mock) -> None:
    accepted = adapter_cls().evaluate(_passing_request(request_cls, payload_type=ALLOWED_PAYLOAD_TYPE))
    assert accepted.fail_closed_reason not in {"forbidden_payload_type", "payload_type_not_allowed"}
    other = adapter_cls().evaluate(_passing_request(request_cls, payload_type="other_type"))
    assert other.fail_closed_reason == "payload_type_not_allowed"


# 11-19. Gate fail-closed + invariants for both adapters.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,_mock", ADAPTERS)
def test_gate_fail_closed_and_invariants(provider, adapter_cls, request_cls, _env, _client, _mock) -> None:
    adapter = adapter_cls()
    assert adapter.evaluate(_passing_request(request_cls, privacy_gate_status="")).fail_closed_reason == "missing_privacy_gate_result"
    assert adapter.evaluate(_passing_request(request_cls, payload_policy_allowed=False)).fail_closed_reason == "payload_policy_failed"
    assert adapter.evaluate(_passing_request(request_cls, budget_allowed=False)).fail_closed_reason == "budget_exceeded"
    assert adapter.evaluate(_passing_request(request_cls, real_provider_execution_enabled=False)).fail_closed_reason == "real_provider_execution_disabled_by_policy"
    assert adapter.evaluate(_passing_request(request_cls, final_external_call_allowed=False)).fail_closed_reason == "final_external_call_not_allowed"
    result = adapter.evaluate(_passing_request(request_cls))
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.provider_real_call_attempted is False
    assert result.real_provider_execution_enabled is False
    assert result.final_external_call_allowed is False


# 20-21. Mock parsers accept valid review-bound schema.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,mock_builder", ADAPTERS)
def test_mock_parser_accepts_valid_schema(provider, adapter_cls, request_cls, _env, _client, mock_builder) -> None:
    adapter = adapter_cls()
    for response in mock_builder().values():
        validation = adapter.validate_mock_response(response)
        assert validation.schema_valid is True, validation.errors
        assert validation.review_required is True
        assert validation.auto_accept is False
        assert validation.active_write_allowed is False


# 22. Parsers reject auto_accept=true.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,mock_builder", ADAPTERS)
def test_parser_rejects_auto_accept_true(provider, adapter_cls, request_cls, _env, _client, mock_builder) -> None:
    response = {**mock_builder()["urinalysis_table"], "auto_accept": True}
    validation = adapter_cls().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "auto_accept_must_be_false" in validation.errors


# 23. Parsers reject active_write_allowed=true.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,mock_builder", ADAPTERS)
def test_parser_rejects_active_write_allowed_true(provider, adapter_cls, request_cls, _env, _client, mock_builder) -> None:
    response = {**mock_builder()["urinalysis_table"], "active_write_allowed": True}
    validation = adapter_cls().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "active_write_allowed_must_be_false" in validation.errors


# 24. Parsers require review_required=true.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,mock_builder", ADAPTERS)
def test_parser_requires_review_required_true(provider, adapter_cls, request_cls, _env, _client, mock_builder) -> None:
    response = {**mock_builder()["urinalysis_table"], "review_required": False}
    validation = adapter_cls().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "review_required_must_be_true" in validation.errors


# 25. Parsers preserve source-text-only recommendation label.
@pytest.mark.parametrize("provider,adapter_cls,request_cls,_env,_client,mock_builder", ADAPTERS)
def test_parser_preserves_recommendation_label(provider, adapter_cls, request_cls, _env, _client, mock_builder) -> None:
    adapter = adapter_cls()
    response = mock_builder()["cytology_pathology"]
    validation = adapter.validate_mock_response(response)
    assert validation.schema_valid is True
    assert validation.source_text_only_recommendation_preserved is True
    draft = adapter.parse_mock_response_to_draft(response, safe_source_document_id=f"src_{provider}")
    recommendation = next(s for s in draft.sections if s.heading == "Recommendation")
    assert recommendation.narrative_label == "source text only - not MedAI recommendation"
    broken = {
        **response,
        "sections": [
            {**s, "source_text_only": False, "narrative_label": "interpreted"} if s["heading"] == "Recommendation" else s
            for s in response["sections"]
        ],
    }
    broken_validation = adapter.validate_mock_response(broken)
    assert broken_validation.schema_valid is False
    assert "recommendation_section_must_be_source_text_only" in broken_validation.errors


# 26-29. Converted mock output creates visible review-bound packages.
@pytest.mark.parametrize("preview_fn", [run_claude_mock_extraction_preview, run_openai_mock_extraction_preview])
def test_converted_mock_output_creates_review_bound_packages(preview_fn) -> None:
    preview = preview_fn()
    assert preview["review_bound_package_count"] >= 3
    assert preview["active_written_count"] == 0
    assert preview["auto_accept"] is False
    assert preview["review_required"] is True
    assert preview["provider_real_call_attempted"] is False
    for package in preview["packages"]:
        assert package["package_status"] == "review-bound"
        assert package["auto_accept_allowed"] is False
        assert package["review_required"] is True
        assert package["active_written_count"] == 0
        assert package["record_count"] >= 1


def test_simulate_local_mock_returns_draft_without_real_call() -> None:
    for adapter, request_cls, client in (
        (ClaudeExtractionAdapter(client=MockClaudeClient()), ClaudeAdapterRequest, "claude"),
        (OpenAIExtractionAdapter(client=MockOpenAIClient()), OpenAIAdapterRequest, "openai"),
    ):
        request = _passing_request(request_cls, real_provider_execution_enabled=False, final_external_call_allowed=False)
        result, validation, draft = adapter.simulate_local_mock(request, source_class="urinalysis_table")
        assert isinstance(draft, AIExtractionPackageDraft)
        assert validation.schema_valid is True
        assert result.simulated_or_mocked_response_used is True
        assert result.provider_real_call_attempted is False
        assert result.external_api_used is False
        assert result.real_network_call_used is False
        assert result.final_external_call_allowed is False


# 30-32. Workflow preview: no token map / PII / credential values.
def test_workflow_preview_has_no_pii_token_map_or_credentials(monkeypatch) -> None:
    monkeypatch.setenv(CLAUDE_CREDENTIAL_ENV_VAR, CLAUDE_SECRET)
    monkeypatch.setenv(OPENAI_CREDENTIAL_ENV_VAR, OPENAI_SECRET)
    result = run_ai_extraction_workflow(_context(provider_name="claude"))
    preview = result.operator_preview
    serialized = json.dumps(preview, sort_keys=True)
    assert preview["claude_adapter_installed"] is True
    assert preview["claude_adapter_status_message"] == "Claude adapter installed but real execution disabled by policy"
    assert preview["openai_adapter_status_message"] == "OpenAI adapter installed but real execution disabled by policy"
    assert preview["claude_real_call_attempted"] is False
    assert preview["openai_real_call_attempted"] is False
    assert result.claude_adapter_status_result["real_provider_execution_enabled"] is False
    assert result.openai_adapter_status_result["real_provider_execution_enabled"] is False
    assert CLAUDE_SECRET not in serialized
    assert OPENAI_SECRET not in serialized
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized
    assert check_public_report_payload(preview).passed


def test_workflow_invariants_with_premium_providers() -> None:
    for provider in ("claude", "openai"):
        result = run_ai_extraction_workflow(_context(provider_name=provider))
        assert result.active_written_count == 0
        assert result.auto_accept is False
        assert result.review_required is True
        assert result.external_api_used is False
        assert result.final_external_call_allowed is False
        assert result.review_bound_package_count == 1


# No provider SDK / network / key values in 15I sources.
def test_no_provider_sdk_imports_network_or_key_values_in_sources() -> None:
    source = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in [
            "execution/premium_adapter_contract.py",
            "execution/claude_extraction_adapter.py",
            "execution/openai_extraction_adapter.py",
            "execution/extraction_workflow.py",
        ]
    )
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "from " + "anthropic",
        "from " + "openai",
        "import " + "google." + "generativeai",
        "google." + "generativeai",
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "api." + "openai.com",
        "api." + "anthropic.com",
        CLAUDE_SECRET,
        OPENAI_SECRET,
    ]
    assert not any(item in source for item in forbidden)


# 33. Doctrine file exists and required exact phrases remain present.
def test_capability_boundary_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 30-32 (reports). 15I reports are public-safe.
def test_15i_reports_are_public_safe(monkeypatch) -> None:
    monkeypatch.setenv(CLAUDE_CREDENTIAL_ENV_VAR, CLAUDE_SECRET)
    monkeypatch.setenv(OPENAI_CREDENTIAL_ENV_VAR, OPENAI_SECRET)
    env = dict(os.environ)
    env["MEDAI_15I_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_premium_adapters_disabled_15i.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_premium_adapters_disabled_15i"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert CLAUDE_SECRET not in text
        assert OPENAI_SECRET not in text
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


# 34-42. Prior-block regressions still pass.
def test_15g_through_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_gemini_adapter_disabled_15g.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_real_provider_enablement_gate_15f.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_external_call_dry_run_15e.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_selection_ui_15d.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_provider_adapter_stub_15c.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_privacy_gate_15b.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
        [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"],
        [sys.executable, "-m", "pytest", "tests/test_medai_operator_usability_polish_13c.py"],
    ]
    for command in commands:
        proc = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr


def _context(**overrides):
    values = {
        "source_class": "urinalysis_table",
        "safe_source_document_id": "source_fake_15i",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
        "provider_name": "claude",
        "provider_mode": "disabled",
        "operator_approval_state": "approved_for_dry_run",
        "external_call_mode": "dry_run",
        "real_provider_enablement_mode": "readiness_check",
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
