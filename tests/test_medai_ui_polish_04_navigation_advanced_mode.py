from __future__ import annotations

from pathlib import Path

from app.main import (
    ADVANCED_OPERATOR_TABS,
    PHASE52_OPERATOR_TABS,
    PRIMARY_OPERATOR_TABS,
    navigation_subtitle,
    operator_tabs,
    sidebar_status_labels,
)


APP_MAIN = Path("app/main.py")


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_default_navigation_shows_primary_pages_only() -> None:
    assert operator_tabs(show_advanced_tools=False) == [
        "Run & Review",
        "Operator Control Panel",
    ]


def test_default_navigation_hides_advanced_pages_and_legacy_labels() -> None:
    default_tabs = operator_tabs(show_advanced_tools=False)
    for label in ["Validation Batch Audit", "Validation History", "Safety & Governance", "Terminology Admin"]:
        assert label not in default_tabs
    for old_label in ["Blind Audit", "Report Archive", "Clinical Knowledge Safety", "Terminology Readiness"]:
        assert old_label not in default_tabs


def test_advanced_navigation_labels_are_available_when_enabled() -> None:
    advanced_tabs = operator_tabs(show_advanced_tools=True)
    for label in ["Run & Review", "Operator Control Panel"]:
        assert label in advanced_tabs
    for label in ["Validation Batch Audit", "Validation History", "Safety & Governance", "Terminology Admin"]:
        assert label in advanced_tabs
    assert PRIMARY_OPERATOR_TABS == ["Run & Review", "Operator Control Panel"]
    assert ADVANCED_OPERATOR_TABS == [
        "Validation Batch Audit",
        "Validation History",
        "Safety & Governance",
        "Terminology Admin",
    ]
    assert PHASE52_OPERATOR_TABS == PRIMARY_OPERATOR_TABS + ADVANCED_OPERATOR_TABS


def test_navigation_toggle_and_sidebar_operator_labels_present() -> None:
    source = app_source()
    assert "Show advanced tools" in source
    assert "Advanced tools include validation history, audit pages, safety governance, and terminology administration." in source
    labels = sidebar_status_labels(enrichment_enabled=True)
    assert labels["knowledge_base"] == "Knowledge base"
    assert labels["active"] == "Active"
    assert labels["draft_facts"] == "Draft facts"
    assert labels["connector_status"] == "Medical connector active"
    assert labels["enrichment_status"] == "Enrichment enabled"


def test_advanced_page_subtitles_are_operator_facing() -> None:
    assert navigation_subtitle("Validation Batch Audit") == "Run a controlled local test batch and review summary results."
    assert navigation_subtitle("Validation History") == "Previous validation and audit reports."
    assert navigation_subtitle("Safety & Governance") == "Safety checks for privacy, knowledge state, and controlled clinical logic."
    assert navigation_subtitle("Terminology Admin") == "Check terminology files, license status, and import readiness."


def test_old_labels_are_not_primary_navigation_or_sidebar_labels() -> None:
    old_labels = {
        "MKB Status",
        "Hypothesis",
        "Connectors: dxgpt",
        "Enrichment: ON",
        "Blind Audit",
        "Report Archive",
        "Clinical Knowledge Safety",
        "Terminology Readiness",
        "MedAI Operator Control Panel",
    }
    primary_text = "\n".join(PRIMARY_OPERATOR_TABS + operator_tabs(show_advanced_tools=False))
    sidebar_text = "\n".join(sidebar_status_labels(enrichment_enabled=True).values())
    combined = primary_text + "\n" + sidebar_text
    for label in old_labels:
        assert label not in combined


def test_page_rendering_calls_remain_available() -> None:
    source = app_source()
    assert "render_run_review_tab(sys_components)" in source
    assert "render_current_run_tab(sys_components, show_title=False)" in source
    assert "render_review_package_panel(show_title=False)" in source
    assert "render_operator_control_panel()" in source
    assert "render_blind_audit_tab(sys_components)" in source
    assert "render_report_archive_tab()" in source
    assert "render_clinical_knowledge_safety_dashboard(_cka_snapshot)" in source
    assert "render_terminology_readiness_panel()" in source
    assert "shell=True" not in source
