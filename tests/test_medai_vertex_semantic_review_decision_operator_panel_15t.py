"""No-live, read-only tests for 15T Vertex semantic decision operator panel."""
from __future__ import annotations

import hashlib
import json

from app.vertex_semantic_review_decision_audit_panel import (
    EXPORT_AFFORDANCES,
    NO_LIVE_READ_ONLY_INDICATOR,
    PANEL_TITLE,
    READ_ONLY_EXPORT_STATEMENT,
    build_audit_panel_view_model,
    render_audit_panel_preview_markdown,
)
from execution.vertex_semantic_review_decision_audit_export import DEFAULT_DECISION_STORE_PATH

EXPECTED_FAMILIES = {
    "portal_result_cards",
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
}


def test_panel_title_and_indicators() -> None:
    vm = build_audit_panel_view_model()
    assert vm["title"] == PANEL_TITLE
    assert vm["no_live_read_only_indicator"] == NO_LIVE_READ_ONLY_INDICATOR
    assert vm["read_only_export_statement"] == READ_ONLY_EXPORT_STATEMENT
    assert "No active MKB records" in vm["read_only_export_statement"]


def test_provider_and_decision_summary_visible() -> None:
    vm = build_audit_panel_view_model()
    assert vm["provider_summary"]["provider_route"] == "vertex"
    assert vm["provider_summary"]["provider_model"] == "gemini-2.5-flash-lite"
    ds = vm["decision_summary"]
    assert ds["total_decisions"] == len(vm["decision_rows"])
    assert ds["accepted_for_review"] >= 1
    assert ds["rejected"] >= 1
    assert ds["deferred"] >= 1
    assert ds["accepted_for_review"] + ds["rejected"] + ds["deferred"] == ds["total_decisions"]


def test_package_family_breakdown_visible() -> None:
    vm = build_audit_panel_view_model()
    assert set(vm["package_family_breakdown"].keys()) == EXPECTED_FAMILIES
    assert sum(vm["package_family_breakdown"].values()) == vm["decision_summary"]["total_decisions"]


def test_safety_summary_visible_and_safe() -> None:
    s = build_audit_panel_view_model()["safety_summary"]
    assert s["active_written_count"] == 0
    assert s["active_mkb_record_created_count"] == 0
    assert s["auto_accept"] is False
    assert s["review_required"] is True
    assert s["hallucinated_field_count"] == 0
    assert s["export_read_only"] is True


def test_decision_rows_have_evidence_and_provenance() -> None:
    rows = build_audit_panel_view_model()["decision_rows"]
    assert rows
    for r in rows:
        assert r["provider_route"] == "vertex"
        assert r["provider_model"] == "gemini-2.5-flash-lite"
        assert str(r["evidence_anchor"]).strip()
        assert str(r["source_evidence_text"]).strip()
        assert str(r["source_report_reference"]).strip()
        assert str(r["audit_reason"]).strip()
        assert r["decision_status"] in {"review_draft", "rejected", "deferred"}


def test_export_affordances_present() -> None:
    vm = build_audit_panel_view_model()
    assert set(vm["export_affordances"].keys()) == {"json", "csv", "markdown"}
    assert all(vm["export_affordances_present"].values())


def test_preview_markdown_shows_core_content_not_hidden() -> None:
    md = render_audit_panel_preview_markdown()
    assert PANEL_TITLE in md
    assert "Package-family breakdown" in md
    assert "Safety status" in md
    assert "Decisions (evidence & provenance)" in md
    assert "Exports (read-only)" in md
    assert "gemini-2.5-flash-lite" in md
    assert READ_ONLY_EXPORT_STATEMENT in md


def test_panel_build_does_not_mutate_decision_store() -> None:
    before = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    build_audit_panel_view_model()
    render_audit_panel_preview_markdown()
    after = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    assert before == after


def test_no_credentials_or_private_markers() -> None:
    vm = build_audit_panel_view_model()
    blob = json.dumps(vm, default=str) + render_audit_panel_preview_markdown(vm)
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gates_or_provider_calls_in_source() -> None:
    import inspect
    import app.vertex_semantic_review_decision_audit_panel as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        assert marker not in source


def test_export_affordance_paths_match_15s_artifacts() -> None:
    assert EXPORT_AFFORDANCES["json"].endswith("decision_audit_export.json")
    assert EXPORT_AFFORDANCES["csv"].endswith("decision_audit_export.csv")
    assert EXPORT_AFFORDANCES["markdown"].endswith("decision_audit_summary.md")
    assert "medai_vertex_semantic_review_decision_audit_export_15s" in EXPORT_AFFORDANCES["json"]
