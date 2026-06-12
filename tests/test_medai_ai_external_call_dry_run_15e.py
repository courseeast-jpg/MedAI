"""Focused tests for MEDAI-AI-EXTERNAL-CALL-DRY-RUN-15E."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_external_call_dry_run import build_ai_external_call_dry_run
from execution.ai_extraction_adapter import ExtractionWorkflowContext
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


def test_missing_operator_approval_fails_closed() -> None:
    result = run_ai_extraction_workflow(_context(external_call_mode="dry_run", operator_approval_state="pending"))

    assert result.dry_run_decision_result["dry_run_external_call_allowed"] is False
    assert result.dry_run_decision_result["fail_closed_reason"] == "operator_approval_required_for_dry_run"
    assert result.dry_run_decision_result["simulated_response_created"] is False


def test_approved_for_dry_run_allows_only_local_simulation() -> None:
    result = run_ai_extraction_workflow(_context())
    decision = result.dry_run_decision_result

    assert decision["requested_provider"] == "gemini"
    assert decision["effective_provider"] == "fake_local"
    assert decision["operator_approval_state"] == "approved_for_dry_run"
    assert decision["dry_run_external_call_allowed"] is True
    assert decision["simulated_response_created"] is True
    assert decision["final_external_call_allowed"] is False
    assert decision["external_api_used"] is False
    assert decision["real_network_call_used"] is False


def test_dry_run_allowed_only_in_explicit_dry_run_mode() -> None:
    result = run_ai_extraction_workflow(_context(external_call_mode="disabled"))

    assert result.dry_run_decision_result["dry_run_external_call_allowed"] is False
    assert result.dry_run_decision_result["fail_closed_reason"] == "dry_run_mode_not_enabled"


def test_required_gate_results_are_enforced_by_dry_run_contract() -> None:
    missing_privacy = build_ai_external_call_dry_run(
        requested_provider="gemini",
        effective_provider="fake_local",
        model_name="gemini-disabled-15c",
        provider_enabled=False,
        operator_approval_state="approved_for_dry_run",
        privacy_gate_result=None,
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result={"budget_allowed": True},
        dry_run_mode=True,
    )
    missing_policy = build_ai_external_call_dry_run(
        requested_provider="gemini",
        effective_provider="fake_local",
        model_name="gemini-disabled-15c",
        provider_enabled=False,
        operator_approval_state="approved_for_dry_run",
        privacy_gate_result={"privacy_gate_status": "redacted_payload_ready", "redacted_payload_hash": "abc123"},
        payload_policy_result=None,
        budget_guard_result={"budget_allowed": True},
        dry_run_mode=True,
    )
    missing_budget = build_ai_external_call_dry_run(
        requested_provider="gemini",
        effective_provider="fake_local",
        model_name="gemini-disabled-15c",
        provider_enabled=False,
        operator_approval_state="approved_for_dry_run",
        privacy_gate_result={"privacy_gate_status": "redacted_payload_ready", "redacted_payload_hash": "abc123"},
        payload_policy_result={"payload_policy_allowed": True},
        budget_guard_result=None,
        dry_run_mode=True,
    )

    assert missing_privacy.decision.fail_closed_reason == "missing_privacy_gate_result"
    assert missing_policy.decision.fail_closed_reason == "missing_payload_policy_result"
    assert missing_budget.decision.fail_closed_reason == "missing_budget_guard_result"


def test_forbidden_raw_pdf_image_and_vision_payloads_remain_blocked() -> None:
    for payload_type in ("raw_pdf", "raw_image", "scanned_image", "external_vision_payload"):
        result = run_ai_extraction_workflow(_context(payload_type=payload_type))
        assert result.payload_policy_result["payload_policy_allowed"] is False
        assert result.dry_run_decision_result["dry_run_external_call_allowed"] is False
        assert result.dry_run_decision_result["fail_closed_reason"] == "forbidden_payload_type"


def test_operator_preview_contains_dry_run_notice_and_no_private_values() -> None:
    preview = run_ai_extraction_workflow(_context()).operator_preview
    serialized = json.dumps(preview, sort_keys=True)

    assert preview["operator_notice_sentence"] == "No external AI call was made."
    assert preview["dry_run_mode_status"] == "dry-run only - real provider execution remains disabled"
    assert preview["dry_run_external_call_allowed"] is True
    assert preview["final_external_call_allowed"] is False
    assert '"token_map":' not in serialized
    assert "api_key" not in serialized.lower()
    assert "credential" not in serialized.lower()
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_audit_sample_contains_public_hashes_counts_statuses_only() -> None:
    audit = run_ai_extraction_workflow(_context()).dry_run_audit_result
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["redacted_payload_hash"]
    assert audit["audit_scope"] == "public_hashes_counts_statuses_only"
    assert audit["external_api_used"] is False
    assert audit["real_network_call_used"] is False
    assert audit["final_external_call_allowed"] is False
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_fake_local_review_bound_package_output_and_safety_invariants_preserved() -> None:
    result = run_ai_extraction_workflow(_context(provider_name="fake_local", provider_mode="fake_local"))

    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True
    assert result.review_bound_package_count == 1
    assert result.operator_preview["packages"][0]["sections"][0]["observations"]


def test_no_provider_sdk_imports_network_calls_or_key_lookup_are_required() -> None:
    source = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in [
            "execution/ai_external_call_dry_run.py",
            "execution/ai_provider_config.py",
            "execution/ai_provider_registry.py",
            "execution/extraction_workflow.py",
        ]
    )
    forbidden = [
        "import openai",
        "import anthropic",
        "google.generativeai",
        "requests.",
        "urllib.request",
        "httpx.",
        "os.environ.get(\"OPENAI",
        "os.environ.get(\"GEMINI",
        "os.environ.get(\"ANTHROPIC",
    ]

    assert not any(item in source for item in forbidden)


def test_15e_reports_are_public_safe() -> None:
    env = dict(os.environ)
    env["MEDAI_15E_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_external_call_dry_run_15e.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_external_call_dry_run_15e"
    for path in report_dir.iterdir():
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


def test_15d_15c_15b_15a_12a_and_13c_regressions_still_pass() -> None:
    commands = [
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
        "safe_source_document_id": "source_fake_15e",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
        "provider_name": "gemini",
        "provider_mode": "disabled",
        "operator_approval_state": "approved_for_dry_run",
        "external_call_mode": "dry_run",
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)
