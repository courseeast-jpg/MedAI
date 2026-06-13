"""No-live tests for the 15Z-C Vertex real-doc adapter dry run."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_adapter_dry_run import (
    ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS,
    AdapterDryRunFixture,
    build_adapter_dry_run_fixtures,
    build_vertex_request_body_no_live,
    evaluate_adapter_readiness_with_gates,
    evaluate_all_adapter_dry_run_cases,
    validate_vertex_request_body_shape,
)
from execution.vertex_real_doc_pii_stripping_proof import (
    build_outbound_safe_payload,
    redact_pii_like_values,
)
from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REDACTED_REAL_LIKE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_adapter_dry_run_no_live_15z_c"


def _case(case_id: str):
    return next(r for r in (evaluate_adapter_readiness_with_gates(f) for f in build_adapter_dry_run_fixtures()) if r.case_id == case_id)


def test_adapter_accepts_15z_b_sanitized_outbound_safe_payload() -> None:
    fixture = build_adapter_dry_run_fixtures()[0]
    red = redact_pii_like_values(fixture.raw_text)
    outbound = build_outbound_safe_payload(red)
    body = build_vertex_request_body_no_live(outbound.outbound_text)
    validation = validate_vertex_request_body_shape(body)
    assert validation.request_shape_valid is True
    assert list(body.keys()) == list(ALLOWED_VERTEX_REQUEST_TOP_LEVEL_KEYS)
    assert "token_map" not in json.dumps(body)


def test_would_be_vertex_request_top_level_keys_exactly_contents_generation_config() -> None:
    report = evaluate_all_adapter_dry_run_cases()
    for case in report["cases"]:
        if case["adapter_status"] in {"DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED", "READY_FOR_FUTURE_AUTHORIZATION_ONLY"}:
            assert case["request_top_level_keys"] == ["contents", "generationConfig"]
            assert case["request_shape_valid"] is True


def test_forbidden_metadata_top_level_keys_rejected() -> None:
    result = _case("request_with_forbidden_metadata_top_level_key")
    assert result.blocked is True
    assert "metadata" in result.forbidden_top_level_keys_present
    assert "forbidden_top_level_metadata_key_present" in result.block_reasons


def test_provider_metadata_remains_local_only() -> None:
    body = build_vertex_request_body_no_live("Patient: [PATIENT_NAME_1]\nMRN: [MRN_1]")
    serialized = json.dumps(body)
    for key in ("provider_route", "provider_name", "model", "location", "endpoint", "project_id", "metadata"):
        assert key not in body
        assert key not in serialized


def test_generation_config_is_json_only_cost_bounded() -> None:
    body = build_vertex_request_body_no_live("Result: tokenized only")
    cfg = body["generationConfig"]
    assert cfg["temperature"] == 0
    assert cfg["maxOutputTokens"] <= 512
    assert cfg["responseMimeType"] == "application/json"
    assert validate_vertex_request_body_shape(body).generation_config_valid is True


def test_invalid_generation_config_blocks() -> None:
    result = _case("request_with_invalid_generation_config")
    assert result.blocked is True
    assert result.generation_config_valid is False
    assert "generation_config_invalid" in result.block_reasons


def test_request_and_outbound_and_vault_fingerprints_generated() -> None:
    for case in evaluate_all_adapter_dry_run_cases()["cases"]:
        assert case["outbound_payload_fingerprint"].startswith("sha256:")
        assert case["would_be_request_fingerprint"].startswith("sha256:")
        assert case["vault_record_fingerprint"].startswith("sha256:")


def test_review_handoff_record_created_and_isolated_for_allowed_cases() -> None:
    report = evaluate_all_adapter_dry_run_cases()
    handoffs = report["handoff_records"]
    assert len(handoffs) == 5
    for rec in handoffs:
        assert rec["status"] == "review_required"
        assert rec["report_only"] is True
        assert rec["active_write_allowed"] is False
        assert rec["auto_accept_allowed"] is False
        assert rec["live_call_allowed"] is False
        assert rec["review_required"] is True


def test_live_external_active_write_auto_accept_invariants() -> None:
    for case in evaluate_all_adapter_dry_run_cases()["cases"]:
        assert case["live_call_allowed"] is False
        assert case["external_api_used"] is False
        assert case["active_write_allowed"] is False
        assert case["auto_accept_allowed"] is False
        assert case["review_required"] is True


def test_unknown_real_private_raw_pii_token_map_active_auto_medication_blocks() -> None:
    expected = {
        "unknown_provenance_payload",
        "real_private_marker_payload",
        "payload_with_raw_pii_residue",
        "payload_with_token_map_leak",
        "active_write_requested",
        "auto_accept_requested",
        "medication_fact_without_safety_gate",
    }
    results = {r.case_id: r for r in (evaluate_adapter_readiness_with_gates(f) for f in build_adapter_dry_run_fixtures())}
    for case_id in expected:
        assert results[case_id].blocked is True, case_id
        assert results[case_id].adapter_status == "BLOCKED"
    assert results["payload_with_raw_pii_residue"].raw_pii_in_request is True
    assert results["payload_with_token_map_leak"].token_map_in_request is True


def test_all_future_gates_simulated_pass_is_future_authorization_only_not_live() -> None:
    result = _case("all_future_gates_simulated_pass")
    assert result.adapter_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
    assert result.readiness_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
    assert result.live_call_allowed is False
    assert result.review_queue_handoff_record_created is True


def test_summary_metrics_match_required_counts() -> None:
    summary = evaluate_all_adapter_dry_run_cases()["summary"]
    assert summary["adapter_dry_run_created"] is True
    assert summary["adapter_cases_total"] == 14
    assert summary["adapter_cases_passed"] == 14
    assert summary["request_shape_valid_count"] == 5
    assert summary["request_top_level_keys_exact_count"] == 5
    assert summary["forbidden_metadata_rejected_count"] == 1
    assert summary["generation_config_valid_count"] == 5
    assert summary["review_queue_handoff_records_created_count"] == 5
    assert summary["review_queue_handoff_report_only_count"] == 5
    assert summary["no_live_replay_allowed_count"] == 4
    assert summary["blocked_case_count"] == 9
    assert summary["future_authorization_only_count"] == 1
    assert summary["real_doc_live_allowed_count"] == 0
    assert summary["raw_pii_in_request_count"] == 0
    assert summary["raw_pii_in_report_count"] == 0
    assert summary["token_map_in_request_count"] == 0
    assert summary["token_map_in_report_count"] == 0
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["active_mkb_record_created_count"] == 0
    assert summary["auto_accept_true_count"] == 0


def test_no_provider_call_path_or_live_gate_exists() -> None:
    import execution.vertex_real_doc_adapter_dry_run as mod

    source = inspect.getsource(mod)
    for marker in (
        "requests.",
        "urllib.request",
        "httpx.",
        "generate_content",
        "acquire_google_cloud_access_token",
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
    ):
        assert marker not in source


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_vertex_real_doc_adapter_dry_run_no_live_15z_c_ready" in proc.stdout
    expected = {
        "summary.json",
        "adapter_dry_run_cases.json",
        "request_shape_matrix.md",
        "review_queue_handoff_records.json",
        "would_be_request_preview.md",
        "implementation_report.md",
    }
    assert expected == {p.name for p in REPORT_DIR.iterdir()}
    for path in REPORT_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/"):
            assert token not in text
        assert '"[MRN_1]":' not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


def test_report_summary_contains_required_metrics() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert summary["adapter_cases_total"] == 14
    assert summary["review_queue_handoff_records_created_count"] == 5


def test_manual_block_fixture_can_be_built_without_live_side_effects() -> None:
    fixture = AdapterDryRunFixture(
        case_id="manual_token_map_leak",
        raw_text="Patient: Manual Leaksample\nMRN: SYN-7777777\nNote: ok.",
        declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
        content_marker="redacted real-like note layout",
        expected_status="BLOCKED",
        inject_token_map=True,
    )
    result = evaluate_adapter_readiness_with_gates(fixture)
    assert result.blocked is True
    assert result.external_api_used is False
    assert result.live_call_allowed is False
