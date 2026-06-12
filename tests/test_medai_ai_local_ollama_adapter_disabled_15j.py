"""Focused tests for MEDAI-AI-LOCAL-OLLAMA-ADAPTER-DISABLED-15J."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import AIExtractionPackageDraft, ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow
from execution.local_ollama_extraction_adapter import (
    ALLOWED_PAYLOAD_TYPE,
    LocalOllamaAdapterConfig,
    LocalOllamaAdapterRequest,
    LocalOllamaExtractionAdapter,
    MockLocalOllamaClient,
    build_mock_local_ollama_responses,
    run_local_ollama_mock_extraction_preview,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTRINE = REPO_ROOT / "docs/architecture/MEDAI_CAPABILITY_BOUNDARY_DOCTRINE.md"
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
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


def _passing_request(**overrides) -> LocalOllamaAdapterRequest:
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
    return LocalOllamaAdapterRequest(**values)


# 1. Adapter exists and instantiates without provider SDK import.
def test_adapter_instantiates_without_provider_sdk() -> None:
    adapter = LocalOllamaExtractionAdapter()
    assert adapter.provider_name == "local_ollama"
    for module in ("ollama", "anthropic", "openai", "google.generativeai", "vertexai"):
        assert module not in sys.modules


# 2. Config defaults to disabled real execution.
def test_config_defaults_disabled_real_execution() -> None:
    config = LocalOllamaExtractionAdapter().config
    assert config.real_provider_execution_enabled is False
    assert config.sdk_import_required is False
    assert config.supports_raw_pdf is False
    assert config.supports_raw_image is False
    assert config.supports_vision is False
    assert isinstance(config, LocalOllamaAdapterConfig)


# 3-5. No local runtime required; no localhost/subprocess calls.
def test_no_local_runtime_required_no_localhost_or_subprocess() -> None:
    # mock preview runs with no Ollama running, no env, no subprocess
    preview = run_local_ollama_mock_extraction_preview()
    assert preview["review_bound_package_count"] >= 3
    assert preview["local_model_call_used"] is False
    assert preview["subprocess_call_used"] is False
    assert preview["ollama_real_call_attempted"] is False
    # adapter source contains no network/subprocess call machinery
    source = (REPO_ROOT / "execution/local_ollama_extraction_adapter.py").read_text(encoding="utf-8")
    for marker in (
        "import " + "subprocess",
        "subprocess" + ".run",
        "import " + "socket",
        "socket" + ".",
        "import " + "requests",
        "requests" + ".",
        "urllib" + ".request",
        "urlopen",
        "httpx" + ".",
        "http" + ".client",
    ):
        assert marker not in source, marker


# 6. Adapter rejects raw PDF payload.
def test_rejects_raw_pdf_payload() -> None:
    result = LocalOllamaExtractionAdapter().evaluate(_passing_request(payload_type="raw_pdf"))
    assert result.fail_closed_reason == "forbidden_payload_type"
    assert result.final_external_call_allowed is False


# 7. Adapter rejects raw image / vision payload.
def test_rejects_raw_image_and_vision_payload() -> None:
    for payload_type in ("raw_image", "external_vision_payload"):
        result = LocalOllamaExtractionAdapter().evaluate(_passing_request(payload_type=payload_type))
        assert result.fail_closed_reason == "forbidden_payload_type"
        assert result.ollama_real_call_attempted is False


# 8. Accepts only redacted_text_layout_summary.
def test_accepts_only_redacted_text_layout_summary() -> None:
    accepted = LocalOllamaExtractionAdapter().evaluate(_passing_request(payload_type=ALLOWED_PAYLOAD_TYPE))
    assert accepted.fail_closed_reason not in {"forbidden_payload_type", "payload_type_not_allowed"}
    other = LocalOllamaExtractionAdapter().evaluate(_passing_request(payload_type="other_type"))
    assert other.fail_closed_reason == "payload_type_not_allowed"


# 9-13. Gate fail-closed ordering.
def test_gate_fail_closed_ordering() -> None:
    adapter = LocalOllamaExtractionAdapter()
    assert adapter.evaluate(_passing_request(privacy_gate_status="")).fail_closed_reason == "missing_privacy_gate_result"
    assert adapter.evaluate(_passing_request(payload_policy_allowed=False)).fail_closed_reason == "payload_policy_failed"
    assert adapter.evaluate(_passing_request(budget_allowed=False)).fail_closed_reason == "budget_exceeded"
    assert adapter.evaluate(_passing_request(real_provider_execution_enabled=False)).fail_closed_reason == "real_provider_execution_disabled_by_policy"
    assert adapter.evaluate(_passing_request(final_external_call_allowed=False)).fail_closed_reason == "final_external_call_not_allowed"


# 14-18. Invariants hold even when inputs request execution.
def test_invariants_hold() -> None:
    result = LocalOllamaExtractionAdapter().evaluate(_passing_request())
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.local_model_call_used is False
    assert result.subprocess_call_used is False
    assert result.ollama_real_call_attempted is False
    assert result.real_provider_execution_enabled is False
    assert result.final_external_call_allowed is False
    assert result.ollama_base_url == "http://localhost:11434"


# 19. Mock parser accepts valid review-bound schema.
def test_mock_parser_accepts_valid_schema() -> None:
    adapter = LocalOllamaExtractionAdapter()
    for response in build_mock_local_ollama_responses().values():
        validation = adapter.validate_mock_response(response)
        assert validation.schema_valid is True, validation.errors
        assert validation.review_required is True
        assert validation.auto_accept is False
        assert validation.active_write_allowed is False


# 20. Parser rejects auto_accept=true.
def test_parser_rejects_auto_accept_true() -> None:
    response = {**build_mock_local_ollama_responses()["urinalysis_table"], "auto_accept": True}
    validation = LocalOllamaExtractionAdapter().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "auto_accept_must_be_false" in validation.errors


# 21. Parser rejects active_write_allowed=true.
def test_parser_rejects_active_write_allowed_true() -> None:
    response = {**build_mock_local_ollama_responses()["urinalysis_table"], "active_write_allowed": True}
    validation = LocalOllamaExtractionAdapter().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "active_write_allowed_must_be_false" in validation.errors


# 22. Parser requires review_required=true.
def test_parser_requires_review_required_true() -> None:
    response = {**build_mock_local_ollama_responses()["urinalysis_table"], "review_required": False}
    validation = LocalOllamaExtractionAdapter().validate_mock_response(response)
    assert validation.schema_valid is False
    assert "review_required_must_be_true" in validation.errors


# 23. Parser preserves source-text-only recommendation label.
def test_parser_preserves_recommendation_label() -> None:
    adapter = LocalOllamaExtractionAdapter()
    response = build_mock_local_ollama_responses()["cytology_pathology"]
    validation = adapter.validate_mock_response(response)
    assert validation.schema_valid is True
    assert validation.source_text_only_recommendation_preserved is True
    draft = adapter.parse_mock_response_to_draft(response, safe_source_document_id="src_ollama")
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


# 24-27. Converted mock output creates visible review-bound packages.
def test_converted_mock_output_creates_review_bound_packages() -> None:
    preview = run_local_ollama_mock_extraction_preview()
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
    adapter = LocalOllamaExtractionAdapter(client=MockLocalOllamaClient())
    request = _passing_request(real_provider_execution_enabled=False, final_external_call_allowed=False)
    result, validation, draft = adapter.simulate_local_mock(request, source_class="urinalysis_table")
    assert isinstance(draft, AIExtractionPackageDraft)
    assert validation.schema_valid is True
    assert result.simulated_or_mocked_response_used is True
    assert result.ollama_real_call_attempted is False
    assert result.local_model_call_used is False
    assert result.subprocess_call_used is False
    assert result.external_api_used is False
    assert result.real_network_call_used is False
    assert result.final_external_call_allowed is False


# 28-30. Workflow preview: no token map / PII / credential values.
def test_workflow_preview_has_no_pii_token_map_or_credentials() -> None:
    result = run_ai_extraction_workflow(_context())
    preview = result.operator_preview
    serialized = json.dumps(preview, sort_keys=True)
    status = result.local_ollama_adapter_status_result
    assert preview["local_ollama_adapter_installed"] is True
    assert preview["local_ollama_adapter_status_message"] == "Local/Ollama adapter installed but real execution disabled by policy"
    assert preview["local_ollama_no_local_model_call_notice"] == "No local model call was made"
    assert preview["local_ollama_real_call_attempted"] is False
    assert preview["local_ollama_local_model_call_used"] is False
    assert preview["local_ollama_subprocess_call_used"] is False
    assert status["real_provider_execution_enabled"] is False
    assert status["local_model_call_used"] is False
    assert status["subprocess_call_used"] is False
    assert status["ollama_real_call_attempted"] is False
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized
    assert check_public_report_payload(preview).passed


def test_workflow_invariants_with_local_ollama() -> None:
    result = run_ai_extraction_workflow(_context())
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.review_bound_package_count == 1


# No SDK / network / subprocess / localhost call code in 15J sources.
def test_no_sdk_network_subprocess_in_sources() -> None:
    adapter_source = (REPO_ROOT / "execution/local_ollama_extraction_adapter.py").read_text(encoding="utf-8")
    forbidden = [
        "import " + "ollama",
        "from " + "ollama",
        "import " + "subprocess",
        "subprocess" + ".",
        "import " + "socket",
        "import " + "requests",
        "requests" + ".",
        "urllib" + ".request",
        "httpx" + ".",
        "http" + ".client",
        "urlopen",
    ]
    assert not any(item in adapter_source for item in forbidden)


# 31. Doctrine file exists and required exact phrases remain present.
def test_capability_boundary_doctrine_phrases_present() -> None:
    assert DOCTRINE.exists()
    text = DOCTRINE.read_text(encoding="utf-8")
    for phrase in DOCTRINE_PHRASES:
        assert phrase in text


# 28-30 (reports). 15J reports are public-safe.
def test_15j_reports_are_public_safe() -> None:
    env = dict(os.environ)
    env["MEDAI_15J_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_local_ollama_adapter_disabled_15j.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_local_ollama_adapter_disabled_15j"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


# 32-41. Prior-block regressions still pass.
def test_15i_through_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_premium_adapters_disabled_15i.py"],
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
        "safe_source_document_id": "source_fake_15j",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
        "provider_name": "local_ollama",
        "provider_mode": "disabled",
        "operator_approval_state": "approved_for_dry_run",
        "external_call_mode": "dry_run",
        "real_provider_enablement_mode": "readiness_check",
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
