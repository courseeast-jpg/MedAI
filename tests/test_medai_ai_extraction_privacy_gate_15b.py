"""Focused tests for MEDAI-AI-EXTRACTION-PRIVACY-GATE-15B."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.source_extraction_packages import package_action_plan
from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_budget_guard import AIBudgetGuard
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.ai_payload_policy import AIPayloadPolicy
from execution.ai_privacy_gate import AIExternalCallApprovalState, run_ai_privacy_gate
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT = (
    "Patient Jane Example; DOB 01/02/1970; MRN 123456; Accession CY-2026-0001; "
    "Facility Park Medical Center; Provider Dr. Alice Clinician; "
    "123 Main Street, Springfield, NY 10001; 555-123-4567; jane.example@example.com; "
    "Insurance INS-ABC-12345; Collected 06/10/2026; Reported 06/11/2026"
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


def _context(source_class: str = "urinalysis_table", **overrides):
    values = {
        "source_class": source_class,
        "safe_source_document_id": "source_fake_15b",
        "selected_document_category": "AI-assisted extraction",
        "selected_specialty_domain": "urology",
        "source_modality": "fake_local_adapter",
        "raw_text_local_only": PRIVATE_TEXT,
    }
    values.update(overrides)
    return ExtractionWorkflowContext(**values)


def test_privacy_gate_detects_and_tokenizes_required_categories() -> None:
    result = run_ai_privacy_gate(raw_text=PRIVATE_TEXT)
    counts = result.token_map.public_counts()

    assert result.pii_detected_count >= 11
    for category in {
        "PATIENT",
        "DOB",
        "DATE",
        "MRN",
        "ACCESSION",
        "FACILITY",
        "PROVIDER",
        "ADDRESS",
        "PHONE",
        "EMAIL",
        "INSURANCE_ID",
    }:
        assert counts.get(category, 0) >= 1


def test_token_map_is_local_only_and_absent_from_public_reports() -> None:
    result = workflow_result_to_public_dict(run_ai_extraction_workflow(_context()))
    serialized = json.dumps(result, sort_keys=True)

    assert result["privacy_gate_result"]["pii_token_map_local_only"] is True
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_redacted_payload_contains_tokens_but_no_raw_fixture_values() -> None:
    result = run_ai_privacy_gate(raw_text=PRIVATE_TEXT)
    text = result.payload.redacted_text

    assert "[PATIENT_1]" in text
    assert "[DOB_1]" in text
    assert "[MRN_1]" in text
    assert "[ACCESSION_1]" in text
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in text


def test_unknown_payload_type_fails_closed() -> None:
    result = run_ai_extraction_workflow(_context(payload_type="mystery_blob"))

    assert result.payload_policy_result["payload_policy_allowed"] is False
    assert result.payload_policy_result["fail_closed_reason"] == "unknown_payload_type"
    assert result.final_external_call_allowed is False


def test_raw_pdf_image_and_vision_payloads_are_forbidden() -> None:
    for payload_type in ("raw_pdf", "raw_image", "scanned_image", "external_vision_payload"):
        result = run_ai_extraction_workflow(_context(payload_type=payload_type))
        assert result.payload_policy_result["payload_policy_allowed"] is False
        assert result.payload_policy_result["fail_closed_reason"] == "forbidden_payload_type"


def test_missing_operator_approval_fails_closed() -> None:
    result = run_ai_extraction_workflow(
        _context(provider_mode="external_candidate", provider_name="disabled", operator_approval_state="not_requested")
    )

    assert result.payload_policy_result["payload_policy_allowed"] is False
    assert result.payload_policy_result["fail_closed_reason"] == "operator_approval_required"


def test_budget_exceeded_fails_closed() -> None:
    gate = run_ai_privacy_gate(raw_text=PRIVATE_TEXT)
    budget = AIBudgetGuard(session_budget_cap_usd=0.0001).evaluate(estimated_input_tokens=100000, estimated_output_tokens=100000)
    policy = AIPayloadPolicy().evaluate(
        payload_type="redacted_text_layout_summary",
        privacy_gate_result=gate,
        operator_approval_state=AIExternalCallApprovalState(state="approved"),
        budget_result=budget,
        provider_mode="external_candidate",
        provider_name="disabled",
    )

    assert budget.budget_allowed is False
    assert policy.payload_policy_allowed is False
    assert policy.fail_closed_reason == "session_budget_exceeded"


def test_real_external_provider_selection_fails_closed_in_15b() -> None:
    result = run_ai_extraction_workflow(
        _context(provider_mode="external_candidate", provider_name="openai", operator_approval_state="approved")
    )

    assert result.external_api_used is False
    assert result.payload_policy_result["payload_policy_allowed"] is False
    assert result.payload_policy_result["fail_closed_reason"] == "real_provider_disabled_in_15b"


def test_external_api_active_writes_auto_accept_and_review_invariants() -> None:
    result = run_ai_extraction_workflow(_context())

    assert result.external_api_used is False
    assert result.final_external_call_allowed is False
    assert result.active_written_count == 0
    assert result.auto_accept is False
    assert result.review_required is True


def test_fake_adapter_still_returns_visible_packages_for_required_types() -> None:
    results = [run_ai_extraction_workflow(_context(source_class=source_class)) for source_class in ("cytology_pathology", "urinalysis_table", "portal_cards")]
    package_types = {result.package_types_tested[0] for result in results}

    assert {"Cytology / pathology narrative", "Urinalysis", "Portal result cards"}.issubset(package_types)
    assert all(result.operator_preview["visible"] for result in results)
    assert all(result.review_bound_package_count == 1 for result in results)


def test_operator_preview_shows_safety_status_but_no_raw_pii() -> None:
    preview = run_ai_extraction_workflow(_context()).operator_preview
    serialized = json.dumps(preview, sort_keys=True)

    assert preview["operator_notice"] == "No external AI call was made"
    assert preview["privacy_gate_status"] in {"redacted_payload_ready", "no_pii_detected"}
    assert preview["pii_detected_count"] >= 11
    assert preview["budget_allowed"] is True
    assert preview["final_external_call_allowed"] is False
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_audit_sample_contains_hashes_counts_statuses_only() -> None:
    audit = run_ai_extraction_workflow(_context()).audit_result
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["external_api_used"] is False
    assert audit["final_external_call_allowed"] is False
    assert audit["redacted_payload_hash"]
    assert audit["raw_payload_hash_local_only"]
    assert '"token_map":' not in serialized
    for raw_value in RAW_FIXTURE_VALUES:
        assert raw_value not in serialized


def test_15a_focused_tests_still_pass() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0


def test_accept_reject_defer_semantics_are_preserved() -> None:
    actions = package_action_plan(["record-1"])["actions"]
    assert {action["key"] for action in actions} >= {
        "accept_package_after_source_comparison",
        "reject_package",
        "defer_package",
    }


def test_package_first_review_queue_semantics_are_preserved() -> None:
    import app.main as main

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "Source extraction packages" in source
    assert "Atomic record actions" in source


def test_15b_validation_reports_are_public_safe() -> None:
    env = dict(os.environ)
    env["MEDAI_15B_SKIP_PYTEST"] = "1"
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_extraction_privacy_gate_15b.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    report_dir = REPO_ROOT / "reports" / "medai_ai_extraction_privacy_gate_15b"
    for path in report_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert '"token_map":' not in text
        assert "C:\\" not in text
        for raw_value in RAW_FIXTURE_VALUES:
            assert raw_value not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"
