"""No-live tests for the 15Z-D authorization/cost/refusal gates."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_authorization_cost_refusal_gates import (
    FUTURE_PACKAGE_STATUS,
    MAX_ESTIMATED_REAL_DOC_CALL_COST_USD,
    TOKEN_BUDGET_CEILING,
    build_authorization_cost_fixtures,
    evaluate_all_authorization_cost_refusal_cases,
    evaluate_authorization_cost_refusal_gates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d"


def _case(case_id: str):
    return next(evaluate_authorization_cost_refusal_gates(f) for f in build_authorization_cost_fixtures() if f.case_id == case_id)


def test_missing_human_authorization_blocks() -> None:
    result = _case("missing_human_authorization")
    assert result.blocked is True
    assert "missing_human_authorization" in result.block_reasons
    assert result.refusal_record_created is True


def test_missing_billing_acknowledgement_blocks() -> None:
    result = _case("missing_billing_ack")
    assert result.blocked is True
    assert "missing_billing_cost_cap_acknowledgement" in result.block_reasons


def test_authorization_and_billing_alone_do_not_authorize_live_call() -> None:
    for case_id in ("valid_no_live_authorization_intent_only", "valid_no_live_billing_ack_only"):
        result = _case(case_id)
        assert result.live_call_allowed is False
        assert result.external_api_used is False
        assert result.active_write_allowed is False
        assert result.auto_accept_allowed is False
        assert result.review_required is True
        assert result.blocked is True


def test_authorization_billing_handoff_creates_future_package_only() -> None:
    result = _case("authorization_and_billing_present_with_15z_c_handoff")
    assert result.future_package_created is True
    assert result.blocked is False
    assert result.readiness_status == FUTURE_PACKAGE_STATUS
    assert result.future_package is not None
    assert result.future_package.live_call_allowed is False
    assert result.future_package.review_required is True


def test_no_review_handoff_blocks_even_with_authorization_and_billing() -> None:
    result = _case("authorization_and_billing_present_but_no_review_handoff")
    assert result.blocked is True
    assert "missing_review_handoff_record" in result.block_reasons


def test_blocked_conditions_create_refusal_records() -> None:
    expected = {
        "real_private_provenance": "real_private_provenance_blocked",
        "unknown_provenance": "unknown_provenance_blocked",
        "active_write_requested": "active_write_requested_blocked",
        "auto_accept_requested": "auto_accept_requested_blocked",
        "medication_fact_without_safety_gate": "medication_fact_without_safety_gate_blocked",
        "forbidden_request_metadata": "forbidden_request_metadata_blocked",
        "invalid_generation_config": "invalid_generation_config_blocked",
        "raw_pii_detected": "raw_pii_detected_blocked",
        "token_map_leak_detected": "token_map_leak_detected_blocked",
        "explicit_live_call_requested": "explicit_live_call_request_blocked",
    }
    for case_id, reason in expected.items():
        result = _case(case_id)
        assert result.blocked is True, case_id
        assert reason in result.block_reasons
        assert result.refusal_record_created is True
        assert result.refusal_record is not None
        assert reason in result.refusal_record.refusal_reason_codes


def test_cost_ceiling_is_conservative_and_no_billing_api_used() -> None:
    report = evaluate_all_authorization_cost_refusal_cases()
    for case in report["cases"]:
        assert case["estimated_cost_ceiling_usd"] <= MAX_ESTIMATED_REAL_DOC_CALL_COST_USD
        assert case["token_budget_ceiling"] <= TOKEN_BUDGET_CEILING
        assert case["no_billing_api_used"] is True
        assert case["billing_check_pending"] is True


def test_summary_metrics_match_required_counts() -> None:
    summary = evaluate_all_authorization_cost_refusal_cases()["summary"]
    assert summary["authorization_cost_refusal_gates_created"] is True
    assert summary["authorization_cases_total"] == 17
    assert summary["authorization_cases_passed"] == 17
    assert summary["blocked_case_count"] == 16
    assert summary["refusal_records_created_count"] == 16
    assert summary["human_authorization_required_count"] == 4
    assert summary["billing_ack_required_count"] == 4
    assert summary["future_authorization_package_created_count"] == 1
    assert summary["future_authorization_only_count"] == 1
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["explicit_live_call_request_blocked"] is True
    assert summary["active_write_blocked"] is True
    assert summary["auto_accept_blocked"] is True
    assert summary["medication_safety_non_bypass_enforced"] is True
    assert summary["no_billing_api_used"] is True
    assert summary["raw_pii_in_report_count"] == 0
    assert summary["token_map_in_report_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0


def test_no_provider_billing_call_path_or_live_gate_exists() -> None:
    import execution.vertex_real_doc_authorization_cost_refusal_gates as mod

    source = inspect.getsource(mod)
    for marker in (
        "requests.",
        "urllib.request",
        "httpx.",
        "generate_content",
        "google.cloud.billing",
        "cloudbilling.",
        "cloudbilling",
        "acquire_google_cloud_access_token",
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
    ):
        assert marker not in source


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d_ready" in proc.stdout
    expected = {
        "summary.json",
        "authorization_cost_cases.json",
        "refusal_records.json",
        "future_authorization_package_preview.json",
        "authorization_cost_matrix.md",
        "implementation_report.md",
    }
    assert expected == {p.name for p in REPORT_DIR.iterdir()}
    for path in REPORT_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", '"[MRN_'):
            assert token not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


def test_report_summary_contains_required_metrics() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert summary["future_authorization_package_created_count"] == 1


def test_all_cases_review_bound_no_active_write_no_auto_accept() -> None:
    for case in evaluate_all_authorization_cost_refusal_cases()["cases"]:
        assert case["review_required"] is True
        assert case["live_call_allowed"] is False
        assert case["external_api_used"] is False
        assert case["active_write_allowed"] is False
        assert case["auto_accept_allowed"] is False
        assert case["raw_pii_in_report"] is False
        assert case["token_map_in_report"] is False
