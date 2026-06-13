"""No-live, read-only tests for 15U Vertex semantic decision panel tab wiring."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.main import (
    ADVANCED_OPERATOR_TAB_LABELS,
    VERTEX_DECISION_AUDIT_TAB,
    operator_tab_labels,
)
from app.vertex_semantic_review_decision_audit_panel import build_audit_panel_view_model
from execution.vertex_semantic_review_decision_audit_export import DEFAULT_DECISION_STORE_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "app" / "main.py"
CORE_TABS = ["Run & Review", "MKB Explorer", "Review Queue"]


def test_nav_label_registered_in_advanced_tabs() -> None:
    assert VERTEX_DECISION_AUDIT_TAB == "Vertex Decision Audit"
    assert VERTEX_DECISION_AUDIT_TAB in ADVANCED_OPERATOR_TAB_LABELS
    assert VERTEX_DECISION_AUDIT_TAB in operator_tab_labels(True)


def test_nav_label_not_in_basic_tabs() -> None:
    assert VERTEX_DECISION_AUDIT_TAB not in operator_tab_labels(False)


def test_core_operator_tabs_preserved_and_first() -> None:
    assert operator_tab_labels(False) == CORE_TABS
    # advanced view keeps the same 3 core tabs first, in order
    assert operator_tab_labels(True)[:3] == CORE_TABS


def test_dispatch_branch_routes_to_15t_hook() -> None:
    source = MAIN_PY.read_text(encoding="utf-8")
    assert "elif label == VERTEX_DECISION_AUDIT_TAB:" in source
    assert "render_vertex_semantic_review_decision_audit_panel_hook()" in source
    # the hook function exists and imports the 15T panel renderer
    assert "def render_vertex_semantic_review_decision_audit_panel_hook" in source


def test_panel_content_still_visible_when_reached() -> None:
    vm = build_audit_panel_view_model()
    assert vm["title"] == "Vertex Semantic Review Decision Audit"
    assert vm["provider_summary"]["provider_route"] == "vertex"
    assert vm["provider_summary"]["provider_model"] == "gemini-2.5-flash-lite"
    assert vm["decision_summary"]["total_decisions"] == len(vm["decision_rows"]) > 0
    assert len(vm["package_family_breakdown"]) == 4
    assert all(vm["export_affordances_present"].values())
    assert vm["safety_summary"]["auto_accept"] is False
    assert vm["safety_summary"]["review_required"] is True
    assert vm["safety_summary"]["active_written_count"] == 0
    assert vm["safety_summary"]["active_mkb_record_created_count"] == 0
    assert vm["safety_summary"]["hallucinated_field_count"] == 0


def test_reaching_panel_does_not_mutate_decision_store() -> None:
    before = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    operator_tab_labels(True)
    build_audit_panel_view_model()
    after = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    assert before == after


def test_only_one_new_advanced_tab_added() -> None:
    # the wiring is additive: exactly one new label beyond the prior five
    prior = {
        "Operator Control Panel",
        "Validation Batch Audit",
        "Validation History",
        "Safety & Governance",
        "Terminology Admin",
    }
    new = set(ADVANCED_OPERATOR_TAB_LABELS) - prior
    assert new == {VERTEX_DECISION_AUDIT_TAB}


def test_no_live_gates_in_panel_source() -> None:
    import inspect
    import app.vertex_semantic_review_decision_audit_panel as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source
