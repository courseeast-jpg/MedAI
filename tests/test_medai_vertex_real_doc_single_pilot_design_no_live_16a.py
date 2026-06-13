"""No-live tests for the 16A single pilot design package."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_single_pilot_design_no_live_16a as mod


def test_all_design_docs_exist() -> None:
    mod.write_reports(mod.build_reports())
    for path in (mod.DESIGN_MD, mod.AUTH_TEMPLATE_MD, mod.GO_NO_GO_MD, mod.TEST_PLAN_MD, mod.NEXT_DECISION_MD):
        assert path.exists(), path


def test_pilot_design_contains_required_boundaries() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.DESIGN_MD.read_text(encoding="utf-8")
    for token in (
        "Explicit no-live status",
        "Real-document live execution is not authorized",
        mod.FUTURE_GATE_PLACEHOLDER,
        "One-document maximum",
        "One-call limit",
        "Redacted/tokenized payload only",
        "No raw PII",
        "No token map outbound",
        "No active MKB write",
        "No auto-accept",
        "Review-required",
        "No medical decision output",
        "Medication safety non-bypass",
        "Stop-on-first-failure",
        "cost cap",
        "Evidence anchoring requirements",
        "Declared label alias policy",
        "`contents` and `generationConfig`",
        "Report sanitization requirements",
        "Operator approval requirements",
        "Refusal Conditions",
    ):
        assert token in text


def test_authorization_template_does_not_authorize_live_execution() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.AUTH_TEMPLATE_MD.read_text(encoding="utf-8")
    assert "does not itself authorize live execution" in text
    for token in (
        "Operator identity placeholder",
        "Date/time placeholder",
        "selected document provenance",
        "PII stripping proof passed",
        "vault isolation passed",
        "no raw PII in outbound payload",
        "token map remains local",
        "billing/cost cap accepted",
        "no active write",
        "no auto-accept",
        "no medical decision output",
        "review-bound only",
        "stop-on-first-failure",
    ):
        assert token.lower() in text.lower()


def test_go_no_go_and_test_plan_are_design_only() -> None:
    mod.write_reports(mod.build_reports())
    go = mod.GO_NO_GO_MD.read_text(encoding="utf-8")
    plan = mod.TEST_PLAN_MD.read_text(encoding="utf-8")
    assert "GO Requirements" in go
    assert "NO-GO Blockers" in go
    assert "Human authorization required" in go
    assert "Cost cap acknowledgement required" in go
    assert "Dedicated future live gate required" in go
    assert "One-call limit required" in go
    assert "One-document limit required" in go
    assert "Medication safety proof required if medication facts appear" in go
    assert "does not include commands that set a live gate" in plan
    assert "Stop Conditions" in plan
    assert "Report Requirements" in plan


def test_next_decision_requires_operator_decision() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.NEXT_DECISION_MD.read_text(encoding="utf-8")
    assert "Option A: Remain No-Live" in text
    assert "Option B: Prepare Future Explicit Single-Document Live Pilot Block" in text
    assert "Option C: Pause/Freeze" in text
    assert "user/operator decision, not automatic live execution" in text


def test_summary_metrics_and_design_checks() -> None:
    summary = mod.build_reports()["summary"]
    assert summary["pilot_design_created"] is True
    assert summary["authorization_template_created"] is True
    assert summary["go_no_go_checklist_created"] is True
    assert summary["test_plan_created"] is True
    assert summary["next_decision_memo_created"] is True
    assert summary["required_design_checks_passed"] == summary["required_design_checks_total"]
    for key in (
        "no_live_status_present",
        "one_document_limit_present",
        "one_call_limit_present",
        "redacted_tokenized_only_present",
        "no_raw_pii_boundary_present",
        "no_token_map_outbound_boundary_present",
        "request_shape_boundary_present",
        "evidence_anchor_boundary_present",
        "label_alias_boundary_present",
        "medication_safety_boundary_present",
        "no_active_write_boundary_present",
        "no_auto_accept_boundary_present",
        "no_medical_decision_boundary_present",
        "review_required_boundary_present",
        "future_live_gate_required",
        "cost_cap_boundary_present",
        "stop_on_first_failure_present",
        "operator_authorization_required",
        "refusal_conditions_present",
        "next_decision_required",
    ):
        assert summary[key] is True, key
    assert summary["docs_only_change"] is True
    assert summary["production_code_changed"] is False
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0
    assert summary["medical_decision_made_count"] == 0
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True


def test_reports_exist_and_are_public_safe() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    paths = [
        mod.DESIGN_MD,
        mod.AUTH_TEMPLATE_MD,
        mod.GO_NO_GO_MD,
        mod.TEST_PLAN_MD,
        mod.NEXT_DECISION_MD,
        mod.SUMMARY_JSON,
        mod.CHECKLIST_JSON,
        mod.DESIGN_MATRIX_MD,
        mod.GO_NO_GO_MATRIX_MD,
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


def test_summary_json_contains_required_metrics_after_write() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in (
        "pilot_design_created",
        "authorization_template_created",
        "go_no_go_checklist_created",
        "test_plan_created",
        "next_decision_memo_created",
        "required_design_checks_passed",
        "no_live_status_present",
        "one_document_limit_present",
        "one_call_limit_present",
        "redacted_tokenized_only_present",
        "no_raw_pii_boundary_present",
        "no_token_map_outbound_boundary_present",
        "request_shape_boundary_present",
        "evidence_anchor_boundary_present",
        "label_alias_boundary_present",
        "medication_safety_boundary_present",
        "no_active_write_boundary_present",
        "no_auto_accept_boundary_present",
        "no_medical_decision_boundary_present",
        "review_required_boundary_present",
        "future_live_gate_required",
        "cost_cap_boundary_present",
        "stop_on_first_failure_present",
        "operator_authorization_required",
        "refusal_conditions_present",
        "next_decision_required",
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
        "privacy_result",
        "billing_check_pending",
    ):
        assert key in summary
