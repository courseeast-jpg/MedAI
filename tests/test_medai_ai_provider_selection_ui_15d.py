"""Focused tests for MEDAI-AI-PROVIDER-SELECTION-UI-15D."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import ExtractionWorkflowContext
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


def test_default_selection_is_fake_local_and_enabled() -> None:
    result = run_ai_extraction_workflow(_context())

    assert result.provider_selection_result["selected_provider"] == "fake_local"
    assert result.provider_selection_result["effective_provider"] == "fake_local"
    assert result.provider_registry_result["provider_enabled"] is True
    assert result.provider_selection_result["provider_execution_allowed"] is False


def test_real_providers_remain_disabled_by_default() -> None:
    providers = {item["provider_name"]: item for item in AIProviderRegistry().list_providers()}

    for name in ("gemini", "claude", "openai", "local_ollama"):
        assert providers[name]["provider_enabled"] is False
        assert providers[name]["fail_closed_reason"] == "provider_disabled_by_policy"


def test_selecting_disabled_provider_records_intent_but_effective_provider_stays_fake_local() -> None:
    result = run_ai_extraction_workflow(_context(provider_name="gemini", provider_mode="disabled"))
    selection = result.provider_selection_result

    assert selection["requested_provider"] == "gemini"
    assert selection["selected_provider"] == "gemini"
    assert selection["effective_provider"] == "fake_local"
    assert selection["provider_execution_allowed"] is False
    assert selection["provider_execution_block_reason"] == "provider_disabled_by_policy"
    assert result.final_external_call_allowed is False


def test_operator_preview_contains_provider_status_and_no_external_notice() -> None:
    preview = run_ai_extraction_workflow(_context(provider_name="openai", provider_mode="disabled")).operator_preview

    assert preview["requested_provider"] == "openai"
    assert preview["effective_provider"] == "fake_local"
    assert preview["provider_message"] == "Provider disabled by policy"
    assert preview["operator_notice"] == "No external AI call was made"
    assert preview["provider_execution_allowed"] is False
    assert preview["privacy_gate_status"]
    assert preview["budget_allowed"] is True


def test_operator_preview_has_no_raw_pii_token_map_keys_or_credentials() -> None:
    preview = run_ai_extraction_workflow(_context(provider_name="claude", provider_mode="disabled")).operator_preview
    serialized = json.dumps(preview, sort_keys=True)

    assert '"token_map":' not in serialized
    assert "api_key" not in serialized.lower()
    assert "credential" not in serialized.lower()
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_safety_invariants_and_fake_local_package_output_preserved() -> None:
    result = run_ai_extraction_workflow(_context())

    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.review_bound_package_count == 1
    assert result.operator_preview["packages"][0]["sections"][0]["observations"]


def test_15d_reports_are_public_safe() -> None:
    env = dict(os.environ)
    env["MEDAI_15D_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_provider_selection_ui_15d.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    for path in (REPO_ROOT / "reports" / "medai_ai_provider_selection_ui_15d").iterdir():
        text = path.read_text(encoding="utf-8")
        assert '"token_map":' not in text
        assert "C:\\" not in text
        assert "api_key" not in text.lower()
        assert "credential" not in text.lower()
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_15c_15b_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
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
        "safe_source_document_id": "source_fake_15d",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
