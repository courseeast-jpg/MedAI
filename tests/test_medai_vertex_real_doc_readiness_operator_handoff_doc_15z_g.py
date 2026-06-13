"""Docs-only tests for the 15Z-G real-document readiness handoff."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "operator_handoff" / "MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g"


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_handoff_doc_exists() -> None:
    assert DOC.exists()


def test_all_required_sections_present() -> None:
    text = _text()
    for section in mod.REQUIRED_SECTIONS:
        assert section in text, section


def test_15z_pass_chain_present() -> None:
    text = _text()
    for block in ("15Z-A", "15Z-B", "15Z-C", "15Z-D", "15Z-E", "15Z-F"):
        assert block in text
    assert "15Z-A through 15Z-F PASS" in text
    assert "Integrated harness PASS" in text


def test_integrated_harness_metrics_present() -> None:
    text = _text()
    for key, value in mod.INTEGRATED_METRICS.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        assert f"{key}={rendered}" in text


def test_gate_inventory_present() -> None:
    text = _text()
    for gate in mod.GATES:
        assert gate in text


def test_operator_commands_present() -> None:
    text = _text()
    for command in mod.COMMANDS:
        assert command in text


def test_real_document_and_no_live_boundaries_present() -> None:
    text = _text()
    assert "does not authorize real-document Vertex routing" in text
    assert "Real-document Vertex routing remains NOT authorized after this block." in text
    assert "No live real-document routing is authorized." in text


def test_no_active_write_no_auto_accept_review_bound_present() -> None:
    text = _text()
    assert "does not authorize active MKB writes" in text
    assert "must not write active MKB" in text
    assert "does not authorize auto-accept" in text
    assert "must not auto-accept" in text
    assert "review-bound" in text


def test_medication_safety_and_no_medical_decision_boundary_present() -> None:
    text = _text()
    assert "Medication safety proof is required when medication facts are present" in text
    assert "No medication decision logic is introduced" in text
    assert "does not create a medical decision system" in text
    assert "must not make medical decisions" in text


def test_stop_conditions_present() -> None:
    text = _text()
    for condition in (
        "live provider usage",
        "billing API usage",
        "real-document live authorization",
        "active MKB write",
        "production review queue mutation",
        "`auto_accept=true`",
        "raw PII in payload or report",
        "token map leakage",
        "unknown provenance allowed",
        "real/private marker allowed",
        "forbidden request metadata accepted",
        "invalid `generationConfig` accepted",
        "medication decision output",
        "medical advice",
        "missing `review_required`",
    ):
        assert condition in text


def test_future_authorization_boundary_and_next_block_present() -> None:
    text = _text()
    assert "Any future real-document live call requires a new explicit block" in text
    assert "its own live gate" in text
    assert "bounded call count" in text
    assert "stop on first failure" in text
    assert "MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-UAT-NO-LIVE-15Z-H" in text


def test_no_credentials_paths_or_raw_private_values_in_doc() -> None:
    text = _text()
    forbidden = (
        "ya29.",
        "AIza",
        "Bearer ",
        "Authorization" + ":",
        "access_token",
        "refresh_token",
        "private_key",
        "application" + "_default" + "_credentials",
        "g" + "cloud",
        "C:\\",
        "/home/",
    )
    for token in forbidden:
        assert token not in text
    assert not re.search(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", text)
    assert '"[MRN_' not in text
    assert '"[PATIENT_NAME_' not in text


def test_script_report_passes_and_metrics_are_safe() -> None:
    reports = mod.build_reports()
    summary = reports["summary"]
    assert summary["handoff_doc_created"] is True
    assert summary["required_sections_present_count"] == len(mod.REQUIRED_SECTIONS)
    assert summary["gate_inventory_present"] is True
    assert summary["operator_commands_present"] is True
    assert summary["real_document_boundary_present"] is True
    assert summary["no_live_authorization_boundary_present"] is True
    assert summary["active_write_boundary_present"] is True
    assert summary["auto_accept_boundary_present"] is True
    assert summary["medication_safety_boundary_present"] is True
    assert summary["no_medical_decision_boundary_present"] is True
    assert summary["future_authorization_boundary_present"] is True
    assert summary["stop_conditions_present"] is True
    assert summary["recommended_next_block_present"] is True
    assert summary["docs_only_change"] is True
    assert summary["production_code_changed"] is False
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert check_public_report_payload(reports).passed


def test_generated_report_files_are_public_safe() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    for path in (
        REPORT_DIR / "summary.json",
        REPORT_DIR / "handoff_doc_checklist.json",
        REPORT_DIR / "handoff_doc_matrix.md",
        REPORT_DIR / "implementation_report.md",
    ):
        assert path.exists(), path
        payload = path.read_text(encoding="utf-8")
        assert "C:\\" not in payload
        assert "Bearer " not in payload
        assert "Authorization" + ":" not in payload
        assert "ya29." not in payload
        assert "AIza" not in payload
        assert '"[MRN_' not in payload
        assert '"[PATIENT_NAME_' not in payload
        assert check_public_report_payload(payload).passed


def test_summary_json_has_required_metrics_after_write() -> None:
    reports = mod.build_reports()
    mod.write_reports(reports)
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    for key in (
        "handoff_doc_created",
        "required_sections_present_count",
        "gate_inventory_present",
        "operator_commands_present",
        "real_document_boundary_present",
        "no_live_authorization_boundary_present",
        "active_write_boundary_present",
        "auto_accept_boundary_present",
        "medication_safety_boundary_present",
        "no_medical_decision_boundary_present",
        "future_authorization_boundary_present",
        "stop_conditions_present",
        "recommended_next_block_present",
        "docs_only_change",
        "production_code_changed",
        "live_call_made",
        "external_api_used",
        "billing_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ):
        assert key in summary
