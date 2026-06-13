"""No-live release snapshot tests for the completed 15Z readiness chain."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_readiness_release_snapshot_15z_i as mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_builds_and_writes_required_release_artifacts() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    for path in (
        mod.RELEASE_MD,
        mod.VALIDATION_MD,
        mod.ARTIFACT_CSV,
        mod.CONTINUATION_MD,
        mod.NEXT_DECISION_MD,
        mod.SUMMARY_JSON,
        mod.CHECKLIST_JSON,
        mod.MATRIX_MD,
        mod.ARTIFACT_CHECK_JSON,
        mod.IMPLEMENTATION_MD,
    ):
        assert path.exists(), path


def test_15z_a_through_15z_h_artifacts_exist() -> None:
    reports = mod.build_reports()
    missing = reports["artifact_check"]["missing_artifacts"]
    assert missing == []
    assert reports["artifact_check"]["artifact_index_entries_count"] == len(mod.ARTIFACT_ROWS)


def test_release_document_contains_required_boundaries_and_evidence() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.RELEASE_MD.read_text(encoding="utf-8")
    assert "Release Purpose" in text
    assert "Current HEAD" in text
    assert mod.REMOTE_BRANCH in text
    for block, scope, status in mod.BLOCK_SUMMARY:
        assert block in text and scope in text and status in text
    for gate in mod.GATES:
        assert gate in text
    for command in mod.COMMANDS:
        assert command in text
    assert "Real-document Vertex routing remains NOT authorized" in text
    assert "future real-document live call requires a new separately gated block" in text
    assert "No active MKB write" in text
    assert "No auto-accept" in text
    assert "No medical decision system" in text


def test_validation_receipt_contains_commands_and_no_live_metrics() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.VALIDATION_MD.read_text(encoding="utf-8")
    for command in mod.TEST_COMMANDS:
        assert command in text
    for metric in (
        "live_call_made=false",
        "external_api_used=false",
        "billing_api_used=false",
        "real_doc_live_allowed_count=0",
        "active_written_count=0",
        "auto_accept_true_count=0",
        "medical_decision_made_count=0",
        "privacy_result=passed",
    ):
        assert metric in text


def test_artifact_index_covers_required_groups() -> None:
    mod.write_reports(mod.build_reports())
    rows = list(csv.DictReader(mod.ARTIFACT_CSV.read_text(encoding="utf-8").splitlines()))
    blocks = {row["block"] for row in rows}
    for block in ("15Z-A", "15Z-B", "15Z-C", "15Z-D", "15Z-E", "15Z-F", "15Z-G", "15Z-H", "15Z-I", "operator"):
        assert block in blocks
    assert len(rows) == len(mod.ARTIFACT_ROWS)
    assert all(row["exists"] == "true" for row in rows)


def test_continuation_snapshot_contains_repo_state_warning_and_next_block() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.CONTINUATION_MD.read_text(encoding="utf-8")
    assert mod.REPO_PATH_PUBLIC in text
    assert mod.LOCAL_BRANCH in text
    assert "Current remote HEAD" in text
    assert mod.NEXT_RECOMMENDED_BLOCK in text
    assert "Real-document Vertex routing remains NOT authorized" in text
    assert "Pre-existing dirty historical report files and untracked 15P-B files must remain unstaged" in text
    for command in mod.COMMANDS:
        assert command in text


def test_next_decision_memo_contains_options_and_recommendation() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.NEXT_DECISION_MD.read_text(encoding="utf-8")
    assert "Option A: Stay No-Live" in text
    assert "Option B: Design A Future Pilot" in text
    assert "Option C: Pause And Package" in text
    assert "pause/freeze or design-only pilot block, not live execution" in text


def test_summary_metrics_match_required_values() -> None:
    reports = mod.build_reports()
    summary = reports["summary"]
    assert summary["release_snapshot_created"] is True
    assert summary["release_folder_created"] is True
    assert summary["release_doc_created"] is True
    assert summary["validation_receipt_created"] is True
    assert summary["artifact_index_created"] is True
    assert summary["continuation_snapshot_created"] is True
    assert summary["next_decision_memo_created"] is True
    assert summary["required_release_checks_passed"] == summary["required_release_checks_total"]
    assert summary["artifact_index_entries_count"] == len(mod.ARTIFACT_ROWS)
    assert summary["operator_commands_present"] is True
    assert summary["gate_inventory_present"] is True
    assert summary["real_document_boundary_present"] is True
    assert summary["future_authorization_boundary_present"] is True
    assert summary["no_live_boundary_present"] is True
    assert summary["active_write_boundary_present"] is True
    assert summary["auto_accept_boundary_present"] is True
    assert summary["medication_safety_boundary_present"] is True
    assert summary["no_medical_decision_boundary_present"] is True
    assert summary["privacy_boundary_present"] is True
    assert summary["dirty_file_warning_present"] is True
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


def test_generated_public_artifacts_are_privacy_safe() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    paths = [
        mod.RELEASE_MD,
        mod.VALIDATION_MD,
        mod.ARTIFACT_CSV,
        mod.CONTINUATION_MD,
        mod.NEXT_DECISION_MD,
        mod.SUMMARY_JSON,
        mod.CHECKLIST_JSON,
        mod.MATRIX_MD,
        mod.ARTIFACT_CHECK_JSON,
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


def test_report_json_contains_required_metrics_after_write() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in (
        "release_snapshot_created",
        "release_folder_created",
        "release_doc_created",
        "validation_receipt_created",
        "artifact_index_created",
        "continuation_snapshot_created",
        "next_decision_memo_created",
        "required_release_checks_passed",
        "artifact_index_entries_count",
        "operator_commands_present",
        "gate_inventory_present",
        "real_document_boundary_present",
        "future_authorization_boundary_present",
        "no_live_boundary_present",
        "active_write_boundary_present",
        "auto_accept_boundary_present",
        "medication_safety_boundary_present",
        "no_medical_decision_boundary_present",
        "privacy_boundary_present",
        "dirty_file_warning_present",
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
