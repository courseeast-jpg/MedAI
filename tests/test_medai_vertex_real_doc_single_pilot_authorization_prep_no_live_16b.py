"""No-live tests for the 16B single-pilot authorization-prep package."""
from __future__ import annotations

import inspect
import json
import re

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_single_pilot_authorization_prep_no_live_16b as mod


def _norm(text: str) -> str:
    """Collapse whitespace so prose line-wrapping never breaks substring checks."""
    return re.sub(r"\s+", " ", text)


def test_all_prep_docs_exist() -> None:
    mod.write_reports(mod.build_reports())
    for path in (
        mod.DESIGN_MD,
        mod.APPROVAL_RECORD_MD,
        mod.FUTURE_GATE_MD,
        mod.ONE_CALL_PLAN_MD,
        mod.STOP_FIRST_FAILURE_MD,
        mod.NEXT_DECISION_MD,
    ):
        assert path.exists(), path


def test_design_contains_required_boundaries() -> None:
    mod.write_reports(mod.build_reports())
    text = _norm(mod.DESIGN_MD.read_text(encoding="utf-8"))
    for token in (
        "Explicit no-live status",
        "design-only",
        "Real-document live execution is not authorized",
        "requires its own approval",
        "Human authorization required",
        mod.FUTURE_LIVE_GATE_NAME,
        "This block does not set that gate",
        "One-document maximum",
        "One-call limit",
        "Redacted/tokenized payload only",
        "No raw PII",
        "No token map outbound",
        "token map remains local",
        "Bounded token ceiling",
        "cost cap",
        "`contents` and `generationConfig`",
        "Review-required",
        "no active MKB write",
        "no auto-accept",
        "no medical decision output",
        "Medication safety non-bypass",
        "Stop-on-first-failure",
        "Refusal Conditions",
        "separate environment evidence only",
    ):
        assert token in text, token


def test_future_live_gate_spec_is_not_set_here() -> None:
    mod.write_reports(mod.build_reports())
    text = _norm(mod.FUTURE_GATE_MD.read_text(encoding="utf-8"))
    assert mod.FUTURE_LIVE_GATE_NAME in text
    assert "This block does not set the live gate" in text
    assert "no command that sets a live gate" in text
    assert "Preconditions Before The Gate May Ever Be Set" in text
    # The doc must NOT contain any shell-style gate-setting command.
    assert "export MEDAI_" not in text
    assert "set MEDAI_" not in text
    assert "$env:MEDAI_" not in text


def test_approval_record_does_not_authorize_live_execution() -> None:
    mod.write_reports(mod.build_reports())
    text = _norm(mod.APPROVAL_RECORD_MD.read_text(encoding="utf-8"))
    assert "does not itself authorize live execution" in text
    for token in (
        "Operator identity placeholder",
        "Approver identity placeholder",
        "provenance declaration",
        "PII stripping proof passed",
        "vault isolation passed",
        "no raw PII in outbound payload",
        "token map remains local",
        "bounded token ceiling accepted",
        "billing/cost cap accepted",
        "dedicated future live gate name acknowledged",
        "one-document limit",
        "one-call limit",
        "no active write",
        "no auto-accept",
        "no medical decision output",
        "review-bound only",
        "stop-on-first-failure",
    ):
        assert token.lower() in text.lower(), token


def test_one_call_plan_is_design_only() -> None:
    mod.write_reports(mod.build_reports())
    plan = _norm(mod.ONE_CALL_PLAN_MD.read_text(encoding="utf-8"))
    assert "Exactly one document selected" in plan
    assert "Exactly one call planned" in plan
    assert "does not include any command that sets a live gate" in plan
    assert "Expected Pass Criteria" in plan
    assert "Expected Fail Criteria" in plan


def test_stop_on_first_failure_handling_present() -> None:
    mod.write_reports(mod.build_reports())
    stop = _norm(mod.STOP_FIRST_FAILURE_MD.read_text(encoding="utf-8"))
    assert "Stop-on-first-failure" in stop
    assert "Trigger Conditions" in stop
    assert "Rollback / Cleanup Expectations" in stop
    assert "Require fresh explicit human authorization" in stop


def test_next_decision_requires_operator_decision() -> None:
    mod.write_reports(mod.build_reports())
    text = _norm(mod.NEXT_DECISION_MD.read_text(encoding="utf-8"))
    assert "Option A: Remain No-Live" in text
    assert "Option B: Execute Future Explicit Single-Document Live Pilot" in text
    assert "Option C: Pause/Freeze" in text
    assert "user/operator decision, not automatic live execution" in text


def test_summary_metrics_and_design_checks() -> None:
    summary = mod.build_reports()["summary"]
    assert summary["authorization_prep_design_created"] is True
    assert summary["approval_record_template_created"] is True
    assert summary["future_live_gate_spec_created"] is True
    assert summary["bounded_one_call_plan_created"] is True
    assert summary["stop_on_first_failure_plan_created"] is True
    assert summary["next_decision_memo_created"] is True
    assert summary["required_design_checks_passed"] == summary["required_design_checks_total"]
    for key in (
        "no_live_status_present",
        "own_approval_required_present",
        "future_live_gate_named_present",
        "future_live_gate_not_set_present",
        "one_document_limit_present",
        "one_call_limit_present",
        "redacted_tokenized_only_present",
        "no_raw_pii_boundary_present",
        "no_token_map_outbound_boundary_present",
        "bounded_token_ceiling_present",
        "cost_cap_boundary_present",
        "request_shape_boundary_present",
        "evidence_anchor_boundary_present",
        "label_alias_boundary_present",
        "medication_safety_boundary_present",
        "no_active_write_boundary_present",
        "no_auto_accept_boundary_present",
        "no_medical_decision_boundary_present",
        "review_required_boundary_present",
        "stop_on_first_failure_present",
        "human_authorization_required_present",
        "refusal_conditions_present",
        "next_decision_required",
        "gcp_sandbox_separate_present",
    ):
        assert summary[key] is True, key
    # The future live gate must NOT be set anywhere.
    assert summary["future_live_gate_currently_set"] is False
    assert summary["future_live_gate_set_in_this_block"] is False
    assert summary["docs_only_change"] is True
    assert summary["production_code_changed"] is False
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["real_private_document_processed"] is False
    assert summary["whole_corpus_processed"] is False
    assert summary["private_corpus_read"] is False
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0
    assert summary["medical_decision_made_count"] == 0
    assert summary["production_queue_mutated"] is False
    assert summary["sandbox_treated_as_medai_validation"] is False
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True


def test_reports_exist_and_are_public_safe() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    paths = [
        mod.DESIGN_MD,
        mod.APPROVAL_RECORD_MD,
        mod.FUTURE_GATE_MD,
        mod.ONE_CALL_PLAN_MD,
        mod.STOP_FIRST_FAILURE_MD,
        mod.NEXT_DECISION_MD,
        mod.SUMMARY_JSON,
        mod.CHECKLIST_JSON,
        mod.DESIGN_MATRIX_MD,
        mod.APPROVAL_MATRIX_MD,
        mod.IMPLEMENTATION_MD,
    ]
    for path in paths:
        payload = path.read_text(encoding="utf-8")
        assert "C:\\" not in payload
        assert "Bearer " not in payload
        assert "Authorization" + ":" not in payload
        assert "ya29." not in payload
        assert "AIza" not in payload
        assert '"[MRN_' not in payload
        assert '"[PATIENT_NAME_' not in payload
        assert not re.search(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", payload)
        assert check_public_report_payload(payload).passed


def test_validator_source_makes_no_provider_or_billing_calls() -> None:
    src = inspect.getsource(mod)
    for marker in (
        "import requests", "urllib.request", "import httpx", "generativeai",
        "google.cloud", "vertexai", "import anthropic", "import openai",
        "generate_content", "cloudbilling", "billing_v1", "import googleapiclient",
        "acquire_google_cloud_access_token",
    ):
        assert marker not in src, marker
    # The validator must not SET a live gate (only read/compare env names).
    assert "os.environ[" not in src
    assert "setenv" not in src


def test_summary_json_contains_required_metrics_after_write() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in (
        "block",
        "authorization_prep_design_created",
        "approval_record_template_created",
        "future_live_gate_spec_created",
        "bounded_one_call_plan_created",
        "stop_on_first_failure_plan_created",
        "next_decision_memo_created",
        "required_design_checks_passed",
        "required_design_checks_total",
        "future_live_gate_currently_set",
        "future_live_gate_set_in_this_block",
        "docs_only_change",
        "production_code_changed",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "real_doc_live_allowed_count",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "medical_decision_made_count",
        "production_queue_mutated",
        "sandbox_treated_as_medai_validation",
        "privacy_result",
        "billing_check_pending",
    ):
        assert key in summary, key
