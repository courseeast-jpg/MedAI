"""No-live tests for the 15Z-A Vertex real-document readiness-gate framework."""
from __future__ import annotations

import inspect
import json

from execution.vertex_real_doc_readiness_gates import (
    PROVENANCE_REAL_PRIVATE,
    PROVENANCE_REDACTED_REAL_LIKE,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNKNOWN,
    READINESS_GATES,
    build_real_doc_refusal_record,
    classify_payload_provenance,
    evaluate_vertex_real_doc_readiness,
)


def _all_gate_pass_kwargs():
    return dict(
        human_authorization_present=True,
        billing_cost_cap_ack_present=True,
        active_write_requested=False,
        auto_accept_requested=False,
        contains_medication_fact=False,
        future_gates_simulated_pass=True,
    )


def test_framework_models_fourteen_gates() -> None:
    assert len(READINESS_GATES) == 14


def test_real_private_document_blocked_by_default() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="rp", declared_provenance=PROVENANCE_REAL_PRIVATE, **_all_gate_pass_kwargs())
    assert e.blocked is True
    assert e.live_call_allowed is False
    assert e.readiness_status == "BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY"


def test_unknown_provenance_blocked() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="unk", declared_provenance=PROVENANCE_UNKNOWN, **_all_gate_pass_kwargs())
    assert e.blocked is True and e.live_call_allowed is False


def test_pii_marker_blocked() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="pii", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE, content_marker="pii", **_all_gate_pass_kwargs())
    assert e.blocked is True
    assert any("pii_stripping_proof_required" in r for r in e.block_reasons)


def test_raw_pdf_marker_blocked() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="pdf", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE, content_marker="raw_pdf", **_all_gate_pass_kwargs())
    assert e.blocked is True
    assert any("no_raw_private_payload_in_reports_required" in r for r in e.block_reasons)


def test_ocr_private_marker_blocked() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="ocr", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE, content_marker="ocr_private", **_all_gate_pass_kwargs())
    assert e.blocked is True
    assert any("pii_vault_isolation_required" in r for r in e.block_reasons)


def test_missing_human_authorization_blocks() -> None:
    kw = _all_gate_pass_kwargs(); kw["human_authorization_present"] = False
    e = evaluate_vertex_real_doc_readiness(case_id="ha", declared_provenance=PROVENANCE_SYNTHETIC, **kw)
    assert e.blocked is True
    assert any("human_authorization_required_for_any_real_live_call" in r for r in e.block_reasons)


def test_missing_billing_ack_blocks() -> None:
    kw = _all_gate_pass_kwargs(); kw["billing_cost_cap_ack_present"] = False
    e = evaluate_vertex_real_doc_readiness(case_id="ba", declared_provenance=PROVENANCE_SYNTHETIC, **kw)
    assert e.blocked is True
    assert any("billing_cost_cap_ack_required" in r for r in e.block_reasons)


def test_active_write_request_blocks() -> None:
    kw = _all_gate_pass_kwargs(); kw["active_write_requested"] = True
    e = evaluate_vertex_real_doc_readiness(case_id="aw", declared_provenance=PROVENANCE_SYNTHETIC, **kw)
    assert e.blocked is True and e.active_write_allowed is False
    assert any("no_active_mkb_write_required" in r for r in e.block_reasons)


def test_auto_accept_request_blocks() -> None:
    kw = _all_gate_pass_kwargs(); kw["auto_accept_requested"] = True
    e = evaluate_vertex_real_doc_readiness(case_id="ac", declared_provenance=PROVENANCE_SYNTHETIC, **kw)
    assert e.blocked is True and e.auto_accept_allowed is False
    assert any("no_auto_accept_required" in r for r in e.block_reasons)


def test_medication_without_safety_gate_blocks() -> None:
    kw = _all_gate_pass_kwargs(); kw["contains_medication_fact"] = True; kw["medication_safety_gate_satisfied"] = False
    e = evaluate_vertex_real_doc_readiness(case_id="med", declared_provenance=PROVENANCE_SYNTHETIC, **kw)
    assert e.blocked is True
    assert any("medication_safety_non_bypass" in r for r in e.block_reasons)


def test_synthetic_fixture_dry_run_allowed_but_not_live() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="syn", declared_provenance=PROVENANCE_SYNTHETIC, content_marker="SYNTHETIC portal")
    assert e.classification_detail["no_live_dry_run_allowed"] is True
    assert e.live_call_allowed is False


def test_redacted_real_like_replay_allowed_no_live_only() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="rl", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE, content_marker="redacted real-like")
    assert e.classification_detail["no_live_dry_run_allowed"] is True
    assert e.live_call_allowed is False


def test_all_gates_simulated_pass_is_future_authorization_only_not_live() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="all", declared_provenance=PROVENANCE_REDACTED_REAL_LIKE,
                                           content_marker="redacted real-like", **_all_gate_pass_kwargs())
    assert e.readiness_status == "READY_FOR_FUTURE_AUTHORIZATION_ONLY"
    assert e.live_call_allowed is False
    assert e.blocked is True  # still not live-authorized in this block
    assert "live_real_doc_call_not_authorized_in_this_block" in e.block_reasons


def test_review_required_always_true_and_no_external_api() -> None:
    for prov in (PROVENANCE_SYNTHETIC, PROVENANCE_REDACTED_REAL_LIKE, PROVENANCE_REAL_PRIVATE, PROVENANCE_UNKNOWN):
        e = evaluate_vertex_real_doc_readiness(case_id=prov, declared_provenance=prov)
        assert e.review_required is True
        assert e.external_api_used is False
        assert e.live_call_allowed is False
        assert e.active_write_allowed is False
        assert e.auto_accept_allowed is False


def test_classification_distinguishes_four_classes() -> None:
    assert classify_payload_provenance(declared_provenance=PROVENANCE_SYNTHETIC).is_synthetic is True
    assert classify_payload_provenance(declared_provenance=PROVENANCE_REDACTED_REAL_LIKE).is_redacted_real_like is True
    assert classify_payload_provenance(declared_provenance=PROVENANCE_REAL_PRIVATE).is_real_private is True
    assert classify_payload_provenance(declared_provenance="weird-thing").is_unknown_provenance is True
    # a pii marker downgrades a redacted-real-like to real_private (forced block)
    c = classify_payload_provenance(declared_provenance=PROVENANCE_REDACTED_REAL_LIKE, content_marker="pii")
    assert c.is_real_private is True and c.forced_block_marker == "pii_marker_present"


def test_refusal_record_is_sanitized_no_raw_payload() -> None:
    e = evaluate_vertex_real_doc_readiness(case_id="rp", declared_provenance=PROVENANCE_REAL_PRIVATE, content_marker="real_private_document secret stuff")
    rec = build_real_doc_refusal_record(e)
    assert rec["raw_payload_in_report"] is False
    assert rec["sanitized_report_only"] is True
    assert rec["content_fingerprint"].startswith("sha256:")
    # the raw marker text must not appear; only fingerprint + reasons
    blob = json.dumps(rec)
    assert "secret stuff" not in blob


def test_no_provider_call_or_live_gate_in_module_source() -> None:
    import execution.vertex_real_doc_readiness_gates as mod
    src = inspect.getsource(mod)
    for marker in (
        "requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token",
        "MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED", "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
    ):
        assert marker not in src


def test_script_report_all_invariants_hold() -> None:
    import scripts.run_medai_vertex_real_doc_readiness_gates_no_live_15z_a as mod
    cases = mod.build_cases()
    m = mod._build_metrics(cases)
    assert m["readiness_cases_total"] == 13
    assert m["real_doc_live_allowed_count"] == 0
    assert m["future_authorization_only_count"] == 1
    assert m["blocked_case_count"] == 13  # every case is blocked from live routing in this block
    assert m["raw_payload_in_report_count"] == 0
    assert m["all_cases_review_required"] is True
    assert m["all_cases_live_call_blocked"] is True
    for k in ("real_private_document_blocked", "unknown_provenance_blocked", "pii_marker_blocked",
              "raw_payload_marker_blocked", "ocr_private_marker_blocked", "medication_safety_non_bypass_enforced",
              "human_authorization_required", "billing_ack_required", "active_write_blocked", "auto_accept_blocked"):
        assert m[k] is True, k
    assert m["live_call_made"] is False and m["external_api_used"] is False
    assert m["active_written_count"] == 0 and m["auto_accept_true_count"] == 0


def test_no_credentials_or_real_pii_in_report_payloads() -> None:
    import scripts.run_medai_vertex_real_doc_readiness_gates_no_live_15z_a as mod
    cases = mod.build_cases()
    blob = json.dumps([mod.evaluation_to_public_dict(c) for c in cases] + [mod.build_real_doc_refusal_record(c) for c in cases], default=str)
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "Jane Example", "jane.example@example.com"):
        assert token not in blob
