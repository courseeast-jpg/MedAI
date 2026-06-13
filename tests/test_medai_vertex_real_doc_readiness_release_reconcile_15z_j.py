"""No-live tests for 15Z-J release reconciliation and pause/freeze memo."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_readiness_release_reconcile_15z_j as mod


def test_current_local_and_remote_heads_are_captured_and_equal() -> None:
    reports = mod.build_reports()
    summary = reports["summary"]
    assert summary["local_head_short"]
    assert summary["remote_head_short"]
    assert summary["local_head_equals_remote_head"] is True


def test_stale_head_references_detected_or_absent_and_fixed() -> None:
    reports = mod.build_reports()
    summary = reports["summary"]
    assert summary["stale_head_references_found"] >= 0
    assert summary["stale_head_references_absent_after_reconcile"] is True
    assert summary["release_docs_reflect_current_head"] is True
    assert not any(stale in mod.RELEASE_MD.read_text(encoding="utf-8") for stale in mod.KNOWN_STALE_HEADS)
    assert not any(stale in mod.CONTINUATION_MD.read_text(encoding="utf-8") for stale in mod.KNOWN_STALE_HEADS)


def test_release_and_continuation_docs_preserve_hard_boundaries() -> None:
    mod.write_reports(mod.build_reports())
    text = "\n".join(path.read_text(encoding="utf-8") for path in (mod.RELEASE_MD, mod.CONTINUATION_MD, mod.PAUSE_MEMO_MD))
    for token in (
        "Real-document Vertex routing remains NOT authorized",
        "Future real-document live call requires separate gated block",
        "No provider calls",
        "No billing API calls",
        "No active MKB writes",
        "No auto-accept",
        "No medical decision output",
    ):
        assert token in text


def test_pause_freeze_memo_exists_and_has_options() -> None:
    mod.write_reports(mod.build_reports())
    text = mod.PAUSE_MEMO_MD.read_text(encoding="utf-8")
    assert "Option A: pause/freeze" in text
    assert "Option B: more no-live stress/UAT" in text
    assert "Option C: future design-only single pilot package" in text
    assert "Recommended next is not live execution" in text


def test_summary_metrics_are_no_live_and_safe() -> None:
    summary = mod.build_reports()["summary"]
    assert summary["pause_freeze_decision_memo_created"] is True
    assert summary["recommended_next_block"] == mod.RECOMMENDED_NEXT_BLOCK
    assert summary["production_code_changed"] is False
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0
    assert summary["medical_decision_made_count"] == 0
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True


def test_reports_are_public_safe_and_have_no_credentials_or_phi() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    for path in (mod.SUMMARY_JSON, mod.HEAD_JSON, mod.MATRIX_MD, mod.IMPLEMENTATION_MD, mod.PAUSE_MEMO_MD):
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


def test_summary_json_contains_required_fields_after_write() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in (
        "stale_head_references_found",
        "stale_head_references_fixed",
        "local_head_equals_remote_head",
        "release_docs_reflect_current_head",
        "pause_freeze_decision_memo_created",
        "recommended_next_block",
        "real_doc_live_allowed_count",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "medical_decision_made_count",
        "privacy_result",
        "billing_check_pending",
    ):
        assert key in summary
