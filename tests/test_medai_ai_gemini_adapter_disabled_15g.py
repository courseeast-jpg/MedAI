"""Focused tests for MEDAI-AI-GEMINI-ADAPTER-DISABLED-15G."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import AIExtractionPackageDraft, ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow
from execution.gemini_extraction_adapter import (
    ALLOWED_PAYLOAD_TYPE,
    FORBIDDEN_PAYLOAD_TYPES,
    GEMINI_CREDENTIAL_ENV_VAR,
    GeminiAdapterConfig,
    GeminiAdapterRequest,
    GeminiExtractionAdapter,
    MockGeminiClient,
    build_mock_gemini_responses,
    run_gemini_mock_extraction_preview,
    validate_gemini_response_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
SECRET_VALUE = "gemini-unit-test-secret-value-15g"
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


def _passing_request(**overrides) -> GeminiAdapterRequest:
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
    return GeminiAdapterRequest(**values)


# 1. Adapter exists and instantiates without provider SDK import.
def test_adapter_instantiates_without_provider_sdk() -> None:
    adapter = GeminiExtractionAdapter()
    assert adapter.provider_name == "gemini"
    # The Gemini provider SDK must never be required/imported for the test path.
    assert "google.generativeai" not in sys.modules
    assert "vertexai" not in sys.modules
    assert adapter.config.sdk_import_required is False


# 2. Config defaults to disabled real execution.
def test_config_defaults_to_disabled_real_execution() -> None:
    config = GeminiAdapterConfig()
    assert config.real_provider_execution_enabled is False
    assert config.sdk_import_required is False
    assert config.supports_raw_pdf is False
    assert config.supports_raw_image is False
    assert config.supports_vision is False


# 3. No GEMINI_API_KEY value required for tests.
def test_no_api_key_required(monkeypatch) -> None:
    monkeypatch.delenv(GEMINI_CREDENTIAL_ENV_VAR, raising=False)
    adapter = GeminiExtractionAdapter()
    readiness = adapter.credential_readiness(environ={})
    assert readiness["credential_present"] is False
    preview = run_gemini_mock_extraction_preview(environ={})
    assert preview["review_bound_package_count"] >= 3


# 4. Credential env var name reported; value never exposed.
def test_credential_env_var_name_reported_value_never_exposed() -> None:
    adapter = GeminiExtractionAdapter()
    readiness = adapter.credential_readiness(environ={GEMINI_CREDENTIAL_ENV_VAR: SECRET_VALUE})
    serialized = json.dumps(readiness, sort_keys=True)
    assert readiness["credential_env_var_name"] == GEMINI_CREDENTIAL_ENV_VAR
    assert readiness["credential_present"] is True
    assert readiness["credential_value_read_or_logged"] is False
    assert SECRET_VALUE not in serialized


# 5. Adapter rejects raw PDF payload.
def test_adapter_rejects_raw_pdf_payload() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(payload_type="raw_pdf"))
    assert result.fail_closed_reason == "forbidden_payload_type"
    assert result.final_external_call_allowed is False


# 6. Adapter rejects raw image / vision payload.
def test_adapter_rejects_raw_image_and_vision_payload() -> None:
    for payload_type in ("raw_image", "external_vision_payload"):
        result = GeminiExtractionAdapter().evaluate(_passing_request(payload_type=payload_type))
        assert result.fail_closed_reason == "forbidden_payload_type"
        assert result.gemini_real_call_attempted is False


# 7. Accepts only redacted_text_layout_summary payload type.
def test_accepts_only_redacted_text_layout_summary() -> None:
    accepted = GeminiExtractionAdapter().evaluate(_passing_request(payload_type=ALLOWED_PAYLOAD_TYPE))
    # payload gate is not the blocker for the allowed type
    assert accepted.fail_closed_reason not in {"forbidden_payload_type", "payload_type_not_allowed"}
    other = GeminiExtractionAdapter().evaluate(_passing_request(payload_type="some_other_type"))
    assert other.fail_closed_reason == "payload_type_not_allowed"


# 8. Missing privacy gate result fails closed.
def test_missing_privacy_gate_fails_closed() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(privacy_gate_status=""))
    assert result.fail_closed_reason == "missing_privacy_gate_result"
    assert result.final_external_call_allowed is False


# 9. Failed payload policy fails closed.
def test_failed_payload_policy_fails_closed() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(payload_policy_allowed=False))
    assert result.fail_closed_reason == "payload_policy_failed"


# 10. Budget failure fails closed.
def test_budget_failure_fails_closed() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(budget_allowed=False))
    assert result.fail_closed_reason == "budget_exceeded"


# 11. real_provider_execution_enabled=false fails closed.
def test_real_provider_execution_disabled_fails_closed() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(real_provider_execution_enabled=False))
    assert result.fail_closed_reason == "real_provider_execution_disabled_by_policy"
    assert result.real_provider_execution_enabled is False


# 12. final_external_call_allowed=false fails closed.
def test_final_external_call_disabled_fails_closed() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request(final_external_call_allowed=False))
    assert result.fail_closed_reason == "final_external_call_not_allowed"
    assert result.final_external_call_allowed is False


# 13-15. external/network/real-call invariants even when inputs request execution.
def test_external_network_and_real_call_invariants_hold() -> None:
    result = GeminiExtractionAdapter().evaluate(_passing_request())
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.gemini_real_call_attempted is False
    # config policy forces disabled even when the request asks for execution
    assert result.real_provider_execution_enabled is False
    assert result.final_external_call_allowed is False


# 16. Mock response parser accepts valid review-bound schema.
def test_mock_parser_accepts_valid_review_bound_schema() -> None:
    adapter = GeminiExtractionAdapter()
    for response in build_mock_gemini_responses().values():
        validation = adapter.validate_mock_response(response)
        assert validation.schema_valid is True, validation.errors
        assert validation.review_required is True
        assert validation.auto_accept is False
        assert validation.active_write_allowed is False


# 17. Parser rejects auto_accept=true.
def test_parser_rejects_auto_accept_true() -> None:
    response = build_mock_gemini_responses()["urinalysis_table"]
    response = {**response, "auto_accept": True}
    validation = validate_gemini_response_schema(response)
    assert validation.schema_valid is False
    assert "auto_accept_must_be_false" in validation.errors


# 18. Parser rejects active_write_allowed=true.
def test_parser_rejects_active_write_allowed_true() -> None:
    response = build_mock_gemini_responses()["urinalysis_table"]
    response = {**response, "active_write_allowed": True}
    validation = validate_gemini_response_schema(response)
    assert validation.schema_valid is False
    assert "active_write_allowed_must_be_false" in validation.errors


# 19. Parser requires review_required=true.
def test_parser_requires_review_required_true() -> None:
    response = build_mock_gemini_responses()["urinalysis_table"]
    response = {**response, "review_required": False}
    validation = validate_gemini_response_schema(response)
    assert validation.schema_valid is False
    assert "review_required_must_be_true" in validation.errors


# 20. Parser preserves source-text-only recommendation label.
def test_parser_preserves_source_text_only_recommendation_label() -> None:
    response = build_mock_gemini_responses()["cytology_pathology"]
    validation = validate_gemini_response_schema(response)
    assert validation.schema_valid is True
    assert validation.source_text_only_recommendation_preserved is True

    draft = GeminiExtractionAdapter().parse_mock_response_to_draft(
        response, safe_source_document_id="source_gemini_mock_cytology"
    )
    recommendation = next(section for section in draft.sections if section.heading == "Recommendation")
    assert recommendation.narrative_label == "source text only - not MedAI recommendation"

    broken = {
        **response,
        "sections": [
            {**section, "source_text_only": False, "narrative_label": "interpreted"}
            if section["heading"] == "Recommendation"
            else section
            for section in response["sections"]
        ],
    }
    broken_validation = validate_gemini_response_schema(broken)
    assert broken_validation.schema_valid is False
    assert "recommendation_section_must_be_source_text_only" in broken_validation.errors


# 21-24. Converted mock output creates visible review-bound packages.
def test_converted_mock_output_creates_review_bound_packages() -> None:
    preview = run_gemini_mock_extraction_preview()
    assert preview["review_bound_package_count"] >= 3
    assert preview["active_written_count"] == 0
    assert preview["auto_accept"] is False
    assert preview["review_required"] is True
    for package in preview["packages"]:
        assert package["package_status"] == "review-bound"
        assert package["auto_accept_allowed"] is False
        assert package["review_required"] is True
        assert package["active_written_count"] == 0
        assert package["record_count"] >= 1


def test_simulate_local_mock_returns_draft_without_real_call() -> None:
    adapter = GeminiExtractionAdapter(gemini_client=MockGeminiClient())
    request = _passing_request(real_provider_execution_enabled=False, final_external_call_allowed=False)
    result, validation, draft = adapter.simulate_local_mock(request, source_class="urinalysis_table")
    assert isinstance(draft, AIExtractionPackageDraft)
    assert validation.schema_valid is True
    assert result.simulated_or_mocked_response_used is True
    assert result.gemini_real_call_attempted is False
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.final_external_call_allowed is False


# 25-27. Workflow + preview: token map / PII / credential values absent.
def test_workflow_preview_has_no_pii_token_map_or_credentials(monkeypatch) -> None:
    monkeypatch.setenv(GEMINI_CREDENTIAL_ENV_VAR, SECRET_VALUE)
    result = run_ai_extraction_workflow(_context())
    preview = result.operator_preview
    serialized = json.dumps(preview, sort_keys=True)
    status = result.gemini_adapter_status_result

    assert preview["gemini_adapter_installed"] is True
    assert preview["gemini_adapter_status_message"] == "Gemini adapter installed but real execution disabled by policy"
    assert preview["gemini_real_call_attempted"] is False
    assert preview["gemini_real_provider_execution_enabled"] is False
    assert status["real_provider_execution_enabled"] is False
    assert status["gemini_real_call_attempted"] is False
    assert status["external_api_used"] is False
    assert status["real_network_call_used"] is False
    assert status["final_external_call_allowed"] is False
    assert status["credential_env_var_name"] == GEMINI_CREDENTIAL_ENV_VAR
    assert status["credential_present"] is True
    assert SECRET_VALUE not in serialized
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized
    check = check_public_report_payload(preview)
    assert check.passed, check.leak_examples_redacted


def test_workflow_active_write_auto_accept_and_review_invariants() -> None:
    result = run_ai_extraction_workflow(_context())
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.review_bound_package_count == 1


# No provider SDK import or network call code in 15G sources.
def test_no_provider_sdk_imports_network_or_key_values_in_sources() -> None:
    source = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in [
            "execution/gemini_extraction_adapter.py",
            "execution/extraction_workflow.py",
        ]
    )
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "import " + "google.generativeai",
        "from " + "google",
        "google." + "generativeai",
        "requests" + ".get",
        "requests" + ".post",
        "urllib" + ".request",
        "httpx" + ".",
        "generativelanguage." + "googleapis.com",
        SECRET_VALUE,
    ]
    assert not any(item in source for item in forbidden)


# 25-27 (reports). 15G reports are public-safe.
def test_15g_reports_are_public_safe(monkeypatch) -> None:
    monkeypatch.setenv(GEMINI_CREDENTIAL_ENV_VAR, SECRET_VALUE)
    env = dict(os.environ)
    env["MEDAI_15G_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_gemini_adapter_disabled_15g.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_gemini_adapter_disabled_15g"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert SECRET_VALUE not in text
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


# 28-35. Prior-block regressions still pass.
def test_15f_through_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
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
        "safe_source_document_id": "source_fake_15g",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
        "provider_name": "gemini",
        "provider_mode": "disabled",
        "operator_approval_state": "approved_for_dry_run",
        "external_call_mode": "dry_run",
        "real_provider_enablement_mode": "readiness_check",
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
