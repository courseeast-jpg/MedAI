"""Focused tests for MEDAI-AI-PROVIDER-ADAPTER-STUB-15C."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_budget_guard import AIBudgetGuard
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.ai_payload_policy import AIPayloadPolicy
from execution.ai_privacy_gate import AIExternalCallApprovalState, run_ai_privacy_gate
from execution.ai_provider_config import AIProviderConfig
from execution.ai_provider_registry import AIProviderRegistry
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _gate_inputs():
    privacy = run_ai_privacy_gate(raw_text=PRIVATE_TEXT)
    budget = AIBudgetGuard(provider_name="disabled", model_name="disabled").evaluate(
        estimated_input_tokens=256,
        estimated_output_tokens=512,
    )
    policy = AIPayloadPolicy().evaluate(
        payload_type="redacted_text_layout_summary",
        privacy_gate_result=privacy,
        operator_approval_state=AIExternalCallApprovalState(state="not_requested"),
        budget_result=budget,
        provider_mode="disabled",
        provider_name="disabled",
    )
    return privacy, policy, budget


def test_provider_registry_lists_required_providers_and_defaults() -> None:
    providers = {item["provider_name"]: item for item in AIProviderRegistry().list_providers()}

    assert set(providers) == {"fake_local", "gemini", "claude", "openai", "local_ollama"}
    assert providers["fake_local"]["provider_enabled"] is True
    for name in ("gemini", "claude", "openai", "local_ollama"):
        assert providers[name]["provider_enabled"] is False
        assert providers[name]["fail_closed_reason"] == "provider_disabled_by_policy"


def test_unknown_provider_fails_closed() -> None:
    privacy, policy, budget = _gate_inputs()
    result = AIProviderRegistry().validate_provider_readiness(
        provider_name="unknown_ai",
        operator_approval_state="approved",
        privacy_gate_result=workflow_result_to_public_dict(run_ai_extraction_workflow(_context()))["privacy_gate_result"],
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result={"budget_allowed": True},
    )

    assert privacy.external_api_used is False
    assert policy.external_api_used is False
    assert budget.budget_allowed is True
    assert result.provider_ready is False
    assert result.fail_closed_reason == "unknown_provider"
    assert result.final_external_call_allowed is False


def test_disabled_real_providers_fail_closed_and_stubs_do_not_extract() -> None:
    registry = AIProviderRegistry()
    for provider_name in ("gemini", "claude", "openai", "local_ollama"):
        result = run_ai_extraction_workflow(_context(provider_name=provider_name, provider_mode="disabled"))
        stub = registry.adapter_for(provider_name).extract()
        assert result.provider_registry_result["provider_enabled"] is False
        assert result.provider_registry_result["fail_closed_reason"] == "provider_disabled_by_policy"
        assert stub.external_api_used is False
        assert stub.final_external_call_allowed is False


def test_missing_operator_approval_missing_privacy_failed_policy_and_budget_fail_closed() -> None:
    registry = AIProviderRegistry()
    privacy, policy, budget = _gate_inputs()
    public_privacy = workflow_result_to_public_dict(run_ai_extraction_workflow(_context()))["privacy_gate_result"]

    approval_registry = AIProviderRegistry(
        {
            "approval_stub": AIProviderConfig(
                provider_name="approval_stub",
                model_name="approval-stub",
                provider_enabled=True,
                provider_mode="external_candidate",
                requires_operator_approval=True,
                supports_text_layout_summary=True,
                supports_raw_pdf=False,
                supports_raw_image=False,
                supports_vision=False,
                external_network_required=False,
                sdk_import_required=False,
                estimated_input_token_limit=1000,
                estimated_output_token_limit=1000,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
            )
        }
    )
    missing_approval = approval_registry.validate_provider_readiness(
        provider_name="approval_stub",
        operator_approval_state="not_requested",
        privacy_gate_result=public_privacy,
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result={"budget_allowed": True},
    )
    missing_privacy = registry.validate_provider_readiness(
        provider_name="fake_local",
        operator_approval_state="approved",
        privacy_gate_result=None,
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result={"budget_allowed": True},
    )
    failed_policy = registry.validate_provider_readiness(
        provider_name="gemini",
        operator_approval_state="approved",
        privacy_gate_result=public_privacy,
        payload_policy_result={"payload_policy_allowed": False, "fail_closed_reason": "payload_policy_failed"},
        budget_guard_result={"budget_allowed": True},
    )
    budget_exceeded = registry.validate_provider_readiness(
        provider_name="fake_local",
        operator_approval_state="approved",
        privacy_gate_result=public_privacy,
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result={"budget_allowed": False, "budget_fail_reason": "session_budget_exceeded"},
    )

    assert privacy.external_api_used is False
    assert policy.final_external_call_allowed is False
    assert budget.external_api_used is False if hasattr(budget, "external_api_used") else True
    assert missing_approval.provider_ready is False
    assert missing_approval.fail_closed_reason == "operator_approval_required"
    assert missing_privacy.fail_closed_reason == "missing_privacy_gate_result"
    assert failed_policy.fail_closed_reason == "provider_disabled_by_policy"
    assert budget_exceeded.fail_closed_reason == "session_budget_exceeded"


def test_provider_capabilities_have_no_raw_pdf_image_or_vision_in_15c() -> None:
    for provider in AIProviderRegistry().list_providers():
        assert provider["supports_raw_pdf"] is False
        assert provider["supports_raw_image"] is False
        assert provider["supports_vision"] is False
        assert provider["sdk_import_required"] is False


def test_no_provider_sdk_imports_or_network_calls_in_15c_files() -> None:
    source = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in [
            "execution/ai_provider_config.py",
            "execution/ai_provider_registry.py",
            "execution/extraction_workflow.py",
        ]
    )
    forbidden = ["import openai", "import anthropic", "google.generativeai", "requests.", "urllib.request"]
    assert not any(item in source for item in forbidden)


def test_fake_local_path_still_produces_review_bound_packages_and_safety_invariants() -> None:
    result = run_ai_extraction_workflow(_context(provider_name="fake_local", provider_mode="fake_local"))

    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.review_bound_package_count == 1
    assert result.operator_preview["visible"] is True
    assert result.provider_registry_result["provider_name"] == "fake_local"


def test_operator_preview_shows_provider_status_without_private_values() -> None:
    preview = run_ai_extraction_workflow(_context(provider_name="gemini", provider_mode="disabled")).operator_preview
    serialized = json.dumps(preview, sort_keys=True)

    assert preview["operator_notice"] == "No external AI call was made"
    assert preview["selected_provider"] == "gemini"
    assert preview["provider_enabled"] is False
    assert preview["provider_message"] == "Provider disabled by policy"
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_public_reports_have_no_token_map_or_raw_fixture_pii() -> None:
    env = dict(os.environ)
    env["MEDAI_15C_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_provider_adapter_stub_15c.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_provider_adapter_stub_15c"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert '"token_map":' not in text
        assert "C:\\" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_15b_15a_and_12a_13c_regressions_still_pass() -> None:
    commands = [
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
        "safe_source_document_id": "source_fake_15c",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
