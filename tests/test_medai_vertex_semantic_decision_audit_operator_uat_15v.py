"""No-live, read-only operator UAT tests for the Vertex Decision Audit tab (15V)."""
from __future__ import annotations

import hashlib

from execution.vertex_semantic_review_decision_audit_export import DEFAULT_DECISION_STORE_PATH
from scripts.run_medai_vertex_semantic_decision_audit_operator_uat_15v import run_uat

EXPECTED_FAMILY_COUNTS = {
    "portal_result_cards": 4,
    "cytology_pathology_narrative": 3,
    "urinalysis_table_like_lab": 4,
    "mixed_narrative_numeric_result": 4,
}


def _metrics():
    return run_uat()["metrics"]


def test_uat_passes_overall() -> None:
    m = _metrics()
    assert m["uat_passed"] is True
    assert all(step["observed"] for step in m["journey_steps"])
    assert len(m["journey_steps"]) == 20


def test_operator_can_navigate_and_open_tab() -> None:
    m = _metrics()
    assert m["operator_nav_entry_found"] is True
    assert m["operator_tab_opened"] is True
    assert m["panel_rendered_from_nav"] is True


def test_panel_indicators_and_title_visible() -> None:
    m = _metrics()
    assert m["panel_title_visible"] is True
    assert m["no_live_read_only_indicator_visible"] is True
    assert m["read_only_statement_visible"] is True


def test_decision_summary_and_family_breakdown_visible() -> None:
    m = _metrics()
    assert m["total_decision_count_visible"] == 15
    assert m["accepted_for_review_count_visible"] == 13
    assert m["rejected_count_visible"] == 1
    assert m["deferred_count_visible"] == 1
    assert m["package_family_breakdown_correct"] is True
    assert m["package_family_breakdown_visible"] == EXPECTED_FAMILY_COUNTS


def test_provenance_evidence_unknown_uncertainty_visible() -> None:
    m = _metrics()
    n = m["decision_records_visible_count"]
    assert m["provider_route_visible_count"] == n
    assert m["provider_model_visible_count"] == n
    assert m["evidence_anchor_visible_count"] == n
    assert m["source_evidence_visible_count"] == n
    assert m["source_report_reference_visible_count"] == n
    assert m["audit_reason_visible_count"] == n
    assert m["unknown_values_visible_count"] == n
    assert m["uncertainty_flags_visible_count"] == n
    assert m["hallucinated_field_count_visible"] == 0


def test_export_affordances_and_artifacts_present() -> None:
    m = _metrics()
    assert m["json_export_affordance_visible"] is True
    assert m["csv_export_affordance_visible"] is True
    assert m["markdown_export_affordance_visible"] is True
    assert m["json_export_artifact_exists"] is True
    assert m["csv_export_artifact_exists"] is True
    assert m["markdown_export_artifact_exists"] is True


def test_safety_invariants_and_no_store_mutation() -> None:
    before = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    m = _metrics()
    after = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    assert before == after
    assert m["decision_store_unchanged"] is True
    assert m["active_mkb_record_created_count"] == 0
    assert m["active_written_count"] == 0
    assert m["auto_accept_true_count"] == 0
    assert m["review_required"] is True
    assert m["live_call_made"] is False
    assert m["external_api_used"] is False
    assert m["billing_check_pending"] is True


def test_no_credentials_or_private_markers_in_uat_output() -> None:
    import json

    result = run_uat()
    blob = json.dumps(result["metrics"], default=str) + json.dumps(result["view_model"], default=str) + result["preview_markdown"]
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gates_or_provider_calls_in_uat_source() -> None:
    import inspect
    import scripts.run_medai_vertex_semantic_decision_audit_operator_uat_15v as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        assert marker not in source
