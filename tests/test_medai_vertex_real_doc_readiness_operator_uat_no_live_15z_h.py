"""No-live operator UAT tests for the 15Z readiness handoff."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h"


def test_15z_g_handoff_artifacts_exist() -> None:
    assert mod.HANDOFF_DOC.exists()
    g_dir = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g"
    for name in ("summary.json", "handoff_doc_checklist.json", "handoff_doc_matrix.md", "implementation_report.md"):
        assert (g_dir / name).exists(), name


def test_builds_twenty_required_uat_steps() -> None:
    steps = mod.build_reports()["steps"]["operator_uat_steps"]
    expected = [
        "operator_can_find_handoff_doc",
        "operator_can_identify_real_doc_boundary",
        "operator_can_identify_gate_inventory",
        "operator_can_run_15z_a_command_no_live",
        "operator_can_run_15z_b_command_no_live",
        "operator_can_run_15z_c_command_no_live",
        "operator_can_run_15z_d_command_no_live",
        "operator_can_run_15z_e_command_no_live",
        "operator_can_run_15z_f_command_no_live",
        "operator_can_verify_integrated_harness_pass",
        "operator_can_verify_future_review_package_only",
        "operator_can_verify_blocked_failure_injections",
        "operator_can_verify_no_provider_call",
        "operator_can_verify_no_billing_api",
        "operator_can_verify_no_active_write",
        "operator_can_verify_no_auto_accept",
        "operator_can_verify_no_medical_decision",
        "operator_can_verify_no_raw_pii_or_token_map_reports",
        "operator_can_identify_stop_conditions",
        "operator_can_identify_next_block_boundary",
    ]
    assert [step["step_name"] for step in steps] == expected
    assert all(step["status"] == "pass" for step in steps)


def test_each_step_has_required_output_fields_and_no_live_flags() -> None:
    steps = mod.build_reports()["steps"]["operator_uat_steps"]
    required = {
        "step_id",
        "step_name",
        "source_artifact_checked",
        "expected_result",
        "observed_result",
        "status",
        "operator_action_required",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_write_allowed",
        "auto_accept_allowed",
        "real_doc_live_allowed",
        "review_required",
    }
    for step in steps:
        assert required <= set(step)
        assert step["live_call_made"] is False
        assert step["external_api_used"] is False
        assert step["billing_api_used"] is False
        assert step["active_write_allowed"] is False
        assert step["auto_accept_allowed"] is False
        assert step["real_doc_live_allowed"] is False
        assert step["review_required"] is True


def test_handoff_sections_gate_inventory_commands_and_boundaries_verified() -> None:
    text = mod.HANDOFF_DOC.read_text(encoding="utf-8")
    for section in mod.REQUIRED_SECTIONS:
        assert section in text
    for gate in mod.GATES:
        assert gate in text
    for command in mod.COMMANDS:
        assert command in text
    for condition in mod.STOP_CONDITIONS:
        assert condition in text
    for boundary in mod.FUTURE_BOUNDARY:
        assert boundary in text
    assert "Real-document Vertex routing remains NOT authorized after this block." in text


def test_summary_metrics_are_expected_no_live_values() -> None:
    summary = mod.build_reports()["summary"]
    assert summary["operator_uat_created"] is True
    assert summary["operator_uat_steps_total"] == 20
    assert summary["operator_uat_steps_passed"] == 20
    assert summary["handoff_doc_found"] is True
    assert summary["gate_inventory_verified"] is True
    assert summary["operator_commands_verified"] is True
    assert summary["integrated_harness_verified"] is True
    assert summary["future_review_package_only_verified"] is True
    assert summary["blocked_failure_injections_verified"] is True
    assert summary["stop_conditions_verified"] is True
    assert summary["future_authorization_boundary_verified"] is True
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0
    assert summary["medical_decision_made_count"] == 0
    assert summary["raw_pii_in_report_count"] == 0
    assert summary["token_map_in_report_count"] == 0
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True


def test_reports_write_and_are_public_safe() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    for name in ("summary.json", "operator_uat_steps.json", "operator_uat_matrix.md", "operator_uat_transcript.md", "implementation_report.md"):
        path = REPORT_DIR / name
        assert path.exists(), name
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
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    for key in (
        "operator_uat_created",
        "operator_uat_steps_total",
        "operator_uat_steps_passed",
        "handoff_doc_found",
        "gate_inventory_verified",
        "operator_commands_verified",
        "integrated_harness_verified",
        "future_review_package_only_verified",
        "blocked_failure_injections_verified",
        "stop_conditions_verified",
        "future_authorization_boundary_verified",
        "real_doc_live_allowed_count",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "medical_decision_made_count",
        "raw_pii_in_report_count",
        "token_map_in_report_count",
        "privacy_result",
        "billing_check_pending",
    ):
        assert key in summary
