"""No-live tests for 15Z-E medication safety non-bypass gates."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_medication_safety_non_bypass import (
    BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED,
    READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY,
    build_medication_safety_fixtures,
    detect_forbidden_medical_decision_output,
    evaluate_all_medication_safety_cases,
    evaluate_medication_safety_non_bypass,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e"


def _case(case_id: str):
    return next(evaluate_medication_safety_non_bypass(f) for f in build_medication_safety_fixtures() if f.case_id == case_id)


def test_medication_facts_without_safety_proof_block() -> None:
    result = _case("medication_mention_candidate_only_no_safety_proof")
    assert result.blocked is True
    assert result.readiness_status == BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED
    assert "medication_safety_proof_required" in result.block_reasons


def test_medication_facts_with_safety_proof_create_future_review_package_only() -> None:
    result = _case("medication_mention_candidate_only_with_safety_proof")
    assert result.blocked is False
    assert result.readiness_status == READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY
    assert result.future_review_package_created is True
    assert result.live_call_allowed is False


def test_candidate_facts_remain_review_bound_no_decision_no_write() -> None:
    for case_id in (
        "medication_with_explicit_dose_candidate_only",
        "medication_with_frequency_candidate_only",
        "medication_with_uncertainty_candidate_only",
        "medication_with_tokenized_evidence_and_pii_vault_reference",
    ):
        result = _case(case_id)
        assert result.medication_candidate_facts_count == 1
        fact = result.medication_candidate_facts[0]
        assert fact.review_required is True
        assert fact.decision_outcome == ""
        assert fact.recommendation_outcome == ""
        assert fact.active_write_allowed is False
        assert fact.auto_accept_allowed is False
        assert result.ddi_decision_made is False
        assert result.contraindication_decision_made is False
        assert result.dosage_advice_made is False
        assert result.treatment_advice_made is False
        assert result.diagnosis_made is False


def test_active_auto_and_explicit_live_requests_block() -> None:
    expected = {
        "medication_fact_active_write_requested": "active_write_requested_blocked",
        "medication_fact_auto_accept_requested": "auto_accept_requested_blocked",
        "explicit_live_call_requested_even_with_safety_proof": "explicit_live_call_request_blocked",
    }
    for case_id, reason in expected.items():
        result = _case(case_id)
        assert result.blocked is True
        assert reason in result.block_reasons
        assert result.live_call_allowed is False


def test_forbidden_medical_decision_requests_block_without_decisions() -> None:
    expected = {
        "medication_fact_ddi_decision_requested": "ddi_decision",
        "medication_fact_contraindication_decision_requested": "contraindication_decision",
        "medication_fact_dosage_advice_requested": "dosage_advice",
        "medication_fact_treatment_advice_requested": "treatment_advice",
        "medication_fact_diagnosis_output_requested": "diagnosis_output",
    }
    for case_id, decision_type in expected.items():
        result = _case(case_id)
        assert result.blocked is True
        assert "medical_decision_logic_requested_blocked" in result.block_reasons
        assert decision_type in result.forbidden_decision_types
        assert result.ddi_decision_made is False
        assert result.contraindication_decision_made is False
        assert result.dosage_advice_made is False
        assert result.treatment_advice_made is False
        assert result.diagnosis_made is False


def test_boundary_detector_is_scanner_only() -> None:
    violations = detect_forbidden_medical_decision_output(("ddi_decision", "diagnosis_output"))
    assert [v.violation_type for v in violations] == ["ddi_decision", "diagnosis_output"]
    assert all(v.blocked is True and v.decision_made is False for v in violations)


def test_non_medication_fixture_gate_not_required_but_not_live() -> None:
    result = _case("non_medication_fixture")
    assert result.medication_facts_present is False
    assert result.blocked is False
    assert result.readiness_status == "MEDICATION_GATE_NOT_REQUIRED_REVIEW_BOUND"
    assert result.live_call_allowed is False


def test_future_authorization_package_medication_cases() -> None:
    missing = _case("future_authorization_package_with_medication_without_safety_proof")
    present = _case("future_authorization_package_with_medication_with_safety_proof")
    assert missing.blocked is True
    assert missing.future_review_package_created is False
    assert present.blocked is False
    assert present.future_review_package_created is True
    assert present.future_review_package["status"] == READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY
    assert present.future_review_package["live_call_allowed"] is False


def test_summary_metrics_match_required_counts() -> None:
    summary = evaluate_all_medication_safety_cases()["summary"]
    assert summary["medication_safety_non_bypass_created"] is True
    assert summary["medication_cases_total"] == 17
    assert summary["medication_cases_passed"] == 17
    assert summary["medication_facts_case_count"] == 16
    assert summary["medication_candidate_facts_total"] == 16
    assert summary["medication_safety_proof_required_block_count"] == 2
    assert summary["forbidden_medical_decision_block_count"] == 5
    assert summary["ddi_decision_blocked"] is True
    assert summary["contraindication_decision_blocked"] is True
    assert summary["dosage_advice_blocked"] is True
    assert summary["treatment_advice_blocked"] is True
    assert summary["diagnosis_output_blocked"] is True
    assert summary["future_review_package_created_count"] == 6
    assert summary["future_review_only_count"] == 6
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["explicit_live_call_request_blocked"] is True
    assert summary["active_write_blocked"] is True
    assert summary["auto_accept_blocked"] is True
    assert summary["ddi_decision_made_count"] == 0
    assert summary["contraindication_decision_made_count"] == 0
    assert summary["dosage_advice_made_count"] == 0
    assert summary["treatment_advice_made_count"] == 0
    assert summary["diagnosis_made_count"] == 0
    assert summary["raw_pii_in_report_count"] == 0
    assert summary["token_map_in_report_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["billing_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0


def test_no_provider_billing_live_or_decision_engine_path_exists() -> None:
    import execution.vertex_real_doc_medication_safety_non_bypass as mod
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
        "interaction_severity =",
        "contraindication_decision =",
        "recommended_dose",
    ):
        assert marker not in source


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e_ready" in proc.stdout
    expected = {
        "summary.json",
        "medication_safety_cases.json",
        "medication_refusal_records.json",
        "medication_safety_matrix.md",
        "future_review_package_preview.json",
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
        [sys.executable, "scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert summary["future_review_package_created_count"] == 6
