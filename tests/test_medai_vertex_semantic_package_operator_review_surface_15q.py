"""No-live tests for 15Q Vertex semantic operator review surface."""
from __future__ import annotations

import json

from app.vertex_semantic_review_surface import (
    PROVIDER_MODEL,
    PROVIDER_ROUTE,
    REVIEW_ACTIONS,
    build_operator_review_report,
    build_vertex_semantic_review_drafts,
    load_recorded_vertex_results,
    render_operator_review_preview,
    simulate_review_action,
)

EXPECTED_FAMILIES = {
    "portal_result_cards",
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
}


def test_loads_four_recorded_families() -> None:
    recorded = load_recorded_vertex_results()
    assert EXPECTED_FAMILIES.issubset(set(recorded.keys()))


def test_builds_four_review_drafts() -> None:
    drafts = build_vertex_semantic_review_drafts()
    assert {d.package_family for d in drafts} == EXPECTED_FAMILIES
    assert len(drafts) == 4


def test_each_draft_has_required_operator_visible_elements() -> None:
    for d in drafts_by_family().values():
        assert d.provider_route == PROVIDER_ROUTE
        assert d.provider_model == PROVIDER_MODEL
        assert d.review_required is True
        assert d.source_visible_body.strip()
        assert d.source_sections
        assert d.vertex_semantic_findings
        assert d.candidate_facts
        assert d.evidence_anchors
        assert d.hallucinated_field_count == 0
        assert d.replay_from_recorded_report is True
        assert d.no_live_replay_indicator
        assert d.active_written_count == 0
        assert d.auto_accept is False
        assert set(["Accept for review queue only", "Reject", "Defer"]).issubset(set(d.action_controls))


def test_findings_separated_from_source_body() -> None:
    for d in drafts_by_family().values():
        # findings carry their own structured fields; source body is a separate string
        for f in d.vertex_semantic_findings:
            assert "label" in f and "evidence_text" in f
        assert isinstance(d.source_visible_body, str)


def test_unknown_and_uncertainty_explicit() -> None:
    for family, d in drafts_by_family().items():
        assert d.uncertainty_flags  # every family has visible uncertainty flags
        # unknowns explicit: list present (may be empty for mixed family) and findings carry unknown_value
        for f in d.vertex_semantic_findings:
            assert "unknown_value" in f


def test_accept_simulation_is_review_bound_no_write() -> None:
    for d in build_vertex_semantic_review_drafts():
        sim = simulate_review_action(d, "accept")
        assert sim["review_decision"] == "accepted_for_review_queue_only"
        assert sim["review_bound"] is True
        assert sim["active_written_count"] == 0
        assert sim["auto_accept"] is False
        assert sim["creates_active_mkb_record"] is False


def test_reject_and_defer_local_only() -> None:
    for d in build_vertex_semantic_review_drafts():
        rej = simulate_review_action(d, "reject")
        dfr = simulate_review_action(d, "defer")
        assert rej["review_decision"] == "rejected_local_simulation_only"
        assert dfr["review_decision"] == "deferred_local_simulation_only"
        for sim in (rej, dfr):
            assert sim["local_simulation_only"] is True
            assert sim["active_written_count"] == 0
            assert sim["creates_active_mkb_record"] is False
            assert sim["auto_accept"] is False


def test_invalid_action_is_rejected_safely() -> None:
    d = build_vertex_semantic_review_drafts()[0]
    sim = simulate_review_action(d, "delete_everything")
    assert sim["review_decision"] == "invalid_action"
    assert sim["active_written_count"] == 0
    assert sim["creates_active_mkb_record"] is False


def test_report_metrics_all_safe() -> None:
    s = build_operator_review_report()["summary"]
    assert s["package_families_rendered"] == 4
    assert s["all_required_families_rendered"] is True
    assert s["hallucinated_field_count"] == 0
    assert s["all_hallucinated_zero"] is True
    assert s["live_call_made"] is False
    assert s["external_api_used"] is False
    assert s["active_written_count"] == 0
    assert s["auto_accept"] is False
    assert s["review_required"] is True
    assert s["billing_check_pending"] is True
    for key in (
        "accept_simulation_review_bound_count",
        "reject_simulation_local_only_count",
        "defer_simulation_local_only_count",
        "accept_control_visible_count",
        "reject_control_visible_count",
        "defer_control_visible_count",
        "evidence_anchor_present_count",
        "uncertainty_flags_visible_count",
        "no_live_replay_indicator_visible_count",
    ):
        assert s[key] == 4


def test_preview_markdown_shows_core_content_not_hidden() -> None:
    md = render_operator_review_preview()
    assert "Source visible body" in md
    assert "Vertex semantic findings" in md
    assert "Evidence anchors" in md
    assert "Uncertainty flags" in md
    assert "Accept for review queue only" in md
    assert "gemini-2.5-flash-lite" in md
    assert "vertex" in md


def test_no_credentials_or_private_markers_in_outputs() -> None:
    report = build_operator_review_report()
    md = render_operator_review_preview()
    blob = json.dumps(report, default=str) + md
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gates_used() -> None:
    import inspect
    import app.vertex_semantic_review_surface as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source


def drafts_by_family():
    return {d.package_family: d for d in build_vertex_semantic_review_drafts()}
