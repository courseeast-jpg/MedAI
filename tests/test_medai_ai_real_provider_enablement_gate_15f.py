"""Focused tests for MEDAI-AI-REAL-PROVIDER-ENABLEMENT-GATE-15F."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.ai_payload_policy import FORBIDDEN_PAYLOAD_TYPES
from execution.ai_provider_enablement import (
    default_real_provider_policies,
    evaluate_real_provider_execution_readiness,
)
from execution.ai_provider_registry import AIProviderRegistry
from execution.extraction_workflow import run_ai_extraction_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026"
)
SECRET_VALUE = "unit-test-secret-value-15f"
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


def test_all_real_providers_default_disabled_and_fake_local_enabled() -> None:
    policies = default_real_provider_policies()
    registry = {item["provider_name"]: item for item in AIProviderRegistry().list_providers()}

    assert policies["fake_local"].provider_enabled_by_policy is True
    assert registry["fake_local"]["provider_enabled"] is True
    for provider in ("gemini", "claude", "openai", "local_ollama"):
        assert policies[provider].provider_enabled_by_policy is False
        assert policies[provider].real_execution_supported_in_this_block is False
        assert registry[provider]["provider_enabled"] is False


def test_missing_credential_produces_blocked_readiness() -> None:
    readiness = evaluate_real_provider_execution_readiness(
        provider_name="gemini",
        operator_approval_state="approved_for_real_provider",
        dry_run_decision_result={"dry_run_external_call_allowed": True},
        environ={},
    )

    assert readiness.credential_status.credential_env_var_name == "GEMINI_API_KEY"
    assert readiness.credential_status.credential_present is False
    assert readiness.decision.real_provider_execution_enabled is False
    assert readiness.decision.final_external_call_allowed is False
    assert readiness.decision.external_api_used is False
    assert readiness.decision.real_network_call_used is False


def test_present_credential_is_detected_without_value_exposure() -> None:
    readiness = evaluate_real_provider_execution_readiness(
        provider_name="openai",
        operator_approval_state="approved_for_real_provider",
        dry_run_decision_result={"dry_run_external_call_allowed": True},
        environ={"OPENAI_API_KEY": SECRET_VALUE},
    )
    serialized = json.dumps(readiness.decision.__dict__, sort_keys=True)

    assert readiness.credential_status.credential_present is True
    assert readiness.credential_status.credential_value_redacted is True
    assert readiness.credential_status.credential_printed_or_logged is False
    assert SECRET_VALUE not in serialized
    assert readiness.decision.real_provider_execution_enabled is False
    assert readiness.decision.real_provider_execution_block_reason == "real_provider_execution_disabled_by_policy"


def test_credential_readiness_does_not_allow_execution_in_workflow(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_VALUE)
    result = run_ai_extraction_workflow(_context())
    preview = result.operator_preview
    serialized = json.dumps(preview, sort_keys=True)

    assert result.credential_readiness_result["credential_present"] is True
    assert result.real_provider_enablement_result["real_provider_execution_enabled"] is False
    assert result.real_provider_enablement_result["final_external_call_allowed"] is False
    assert result.real_provider_enablement_result["external_api_used"] is False
    assert result.real_provider_enablement_result["real_network_call_used"] is False
    assert preview["credential_env_var_name"] == "GEMINI_API_KEY"
    assert preview["credential_present"] is True
    assert preview["real_provider_execution_enabled"] is False
    assert preview["real_provider_execution_notice"] == "Real provider execution disabled by policy"
    assert SECRET_VALUE not in serialized


def test_raw_pdf_image_and_vision_payloads_remain_forbidden() -> None:
    for payload_type in FORBIDDEN_PAYLOAD_TYPES:
        result = run_ai_extraction_workflow(_context(payload_type=payload_type))
        assert result.payload_policy_result["payload_policy_allowed"] is False
        assert result.final_external_call_allowed is False


def test_dry_run_and_fake_local_review_bound_package_output_preserved() -> None:
    dry_run = run_ai_extraction_workflow(_context())
    fake_local = run_ai_extraction_workflow(_context(provider_name="fake_local", provider_mode="fake_local"))

    assert dry_run.dry_run_decision_result["dry_run_external_call_allowed"] is True
    assert dry_run.external_api_used is False
    assert dry_run.real_provider_enablement_result["real_provider_execution_enabled"] is False
    assert fake_local.review_bound_package_count == 1
    assert fake_local.active_written_count == 0
    assert fake_local.auto_accept is False
    assert fake_local.review_required is True
    assert fake_local.operator_preview["visible"] is True


def test_no_provider_sdk_imports_network_calls_or_key_values_in_changed_sources() -> None:
    source = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in [
            "execution/ai_provider_enablement.py",
            "execution/ai_external_call_dry_run.py",
            "execution/extraction_workflow.py",
        ]
    )
    forbidden = [
        "import " + "openai",
        "import " + "anthropic",
        "google." + "generativeai",
        "requests" + ".",
        "urllib" + ".request",
        "httpx" + ".",
        SECRET_VALUE,
    ]
    assert not any(item in source for item in forbidden)


def test_15f_reports_are_public_safe(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", SECRET_VALUE)
    env = dict(os.environ)
    env["MEDAI_15F_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_real_provider_enablement_gate_15f.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_real_provider_enablement_gate_15f"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert SECRET_VALUE not in text
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "sk-" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_15e_15d_15c_15b_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
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
        "safe_source_document_id": "source_fake_15f",
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
