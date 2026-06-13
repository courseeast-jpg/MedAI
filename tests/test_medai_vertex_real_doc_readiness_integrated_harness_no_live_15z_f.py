"""No-live tests for 15Z-F integrated readiness harness."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_readiness_integrated_harness import (
    READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY,
    build_integrated_readiness_cases,
    evaluate_all_integrated_readiness_cases,
    run_integrated_readiness_harness_no_live,
    validate_integrated_package_sanitized,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f"


def _case(case_id: str):
    return next(run_integrated_readiness_harness_no_live(c) for c in build_integrated_readiness_cases() if c.case_id == case_id)


def test_clean_non_medication_path_creates_future_operator_review_package_only() -> None:
    result = _case("clean_redacted_real_like_non_medication_full_path")
    assert result.integrated_status == READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY
    assert result.package_created is True
    assert result.readiness_package is not None
    assert validate_integrated_package_sanitized(result.readiness_package) is True
    assert result.live_call_allowed is False


def test_clean_medication_path_requires_safety_proof_and_is_no_live() -> None:
    result = _case("clean_redacted_real_like_medication_full_path_with_safety_proof")
    assert result.package_created is True
    assert result.medication_safety_proof_present is True
    assert result.readiness_package is not None
    assert result.readiness_package.live_call_allowed is False


def test_all_gates_simulated_pass_still_no_live() -> None:
    result = _case("all_gates_simulated_pass_still_no_live")
    assert result.package_created is True
    assert result.integrated_status == READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY
    assert result.live_call_allowed is False


def test_all_failure_injections_block_with_refusals() -> None:
    expected = {
        "unknown_provenance": "unknown_provenance_blocked",
        "real_private_marker": "real_private_marker_blocked",
        "raw_pii_residue": "raw_pii_residue_blocked",
        "token_map_leak": "token_map_leak_blocked",
        "forbidden_request_metadata": "forbidden_request_metadata_blocked",
        "invalid_generation_config": "invalid_generation_config_blocked",
        "missing_review_handoff": "missing_review_handoff",
        "missing_human_authorization": "missing_human_authorization",
        "missing_billing_ack": "missing_billing_ack",
        "active_write_requested": "active_write_requested_blocked",
        "auto_accept_requested": "auto_accept_requested_blocked",
        "medication_without_safety_proof": "medication_safety_proof_required",
        "medication_decision_requested": "medical_decision_logic_requested_blocked",
        "explicit_live_call_requested": "explicit_live_call_request_blocked",
    }
    for case_id, reason in expected.items():
        result = _case(case_id)
        assert result.blocked is True, case_id
        assert result.refusal_record_created is True
        assert reason in result.block_reasons
        assert result.live_call_allowed is False


def test_request_shape_keys_remain_contents_generation_config_for_package_cases() -> None:
    for result in (_case("clean_redacted_real_like_non_medication_full_path"), _case("clean_redacted_real_like_medication_full_path_with_safety_proof")):
        assert result.request_shape_valid is True
        assert result.gate_trace.request_shape_valid is True


def test_summary_metrics_match_required_counts() -> None:
    summary = evaluate_all_integrated_readiness_cases()["summary"]
    assert summary["integrated_readiness_harness_created"] is True
    assert summary["integrated_cases_total"] == 17
    assert summary["integrated_cases_passed"] == 17
    assert summary["future_operator_review_package_created_count"] == 3
    assert summary["future_operator_review_only_count"] == 3
    assert summary["blocked_case_count"] == 14
    assert summary["refusal_records_created_count"] == 14
    assert summary["explicit_live_call_request_blocked"] is True
    assert summary["active_write_blocked"] is True
    assert summary["auto_accept_blocked"] is True
    assert summary["medication_safety_non_bypass_enforced"] is True
    assert summary["medical_decision_blocked"] is True
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["raw_pii_in_package_count"] == 0
    assert summary["raw_pii_in_report_count"] == 0
    assert summary["token_map_in_package_count"] == 0
    assert summary["token_map_in_report_count"] == 0
    assert summary["medical_decision_made_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0


def test_no_provider_billing_live_or_decision_engine_path_exists() -> None:
    import execution.vertex_real_doc_readiness_integrated_harness as mod
    source = inspect.getsource(mod)
    for marker in (
        "requests.",
        "urllib.request",
        "httpx.",
        "generate_content",
        "google.cloud.billing",
        "cloudbilling.",
        "acquire_google_cloud_access_token",
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "calculate_interaction",
        "recommended_dose",
        "interaction_severity =",
    ):
        assert marker not in source


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f_ready" in proc.stdout
    expected = {
        "summary.json",
        "integrated_readiness_cases.json",
        "integrated_gate_trace_matrix.md",
        "integrated_refusal_records.json",
        "future_operator_review_package_preview.json",
        "implementation_report.md",
    }
    assert expected == {p.name for p in REPORT_DIR.iterdir()}
    for path in REPORT_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", '"[MRN_'):
            assert token not in text
        for advice in ("should take", "recommended dose", "contraindicated because", "interaction severity", "diagnosis is", "treatment plan"):
            assert advice not in text.lower()
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


def test_report_summary_contains_required_metrics() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert summary["future_operator_review_package_created_count"] == 3
