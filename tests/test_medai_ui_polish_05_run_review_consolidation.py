from __future__ import annotations

from pathlib import Path

from app.main import (
    ADVANCED_OPERATOR_TABS,
    PRIMARY_OPERATOR_TABS,
    RUN_REVIEW_TAB,
    operator_tabs,
)


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
APP_REVIEW = Path(__file__).resolve().parents[1] / "app" / "review_package_viewer.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def review_source() -> str:
    return APP_REVIEW.read_text(encoding="utf-8")


def test_default_navigation_uses_run_review_primary_workflow() -> None:
    assert RUN_REVIEW_TAB == "Run & Review"
    assert PRIMARY_OPERATOR_TABS == ["Run & Review", "Operator Control Panel"]
    assert operator_tabs(show_advanced_tools=False) == ["Run & Review", "Operator Control Panel"]


def test_default_navigation_no_longer_shows_separate_run_review_tabs() -> None:
    default_tabs = operator_tabs(show_advanced_tools=False)
    for label in (
        "Current Run",
        "Review Package",
        "Blind Audit",
        "Report Archive",
        "Clinical Knowledge Safety",
        "Terminology Readiness",
    ):
        assert label not in default_tabs


def test_run_review_page_contains_current_run_and_review_sections() -> None:
    combined = app_source() + "\n" + review_source()
    for text in (
        "Add documents, process them locally, and review anything that needs attention.",
        "Current run",
        "Add documents, then start a run.",
        "Supported files: PDF or TXT. Files stay local.",
        "Choose files",
        "Start run",
        "Documents waiting",
        "Files ready",
        "Waiting to start",
        "Accepted",
        "Needs review",
        "OCR / scan review",
        "No text found",
        "Errors",
        "Review summary",
        "Review status: {summary['review_status']}",
        "scan-quality items need attention when convenient",
        "No production changes recommended.",
        "Issue type",
        "Items",
        "Needs action?",
        "System change?",
        "Review details",
        "Scan quality review",
        "Unknown document type",
        "Possible combined PDF",
        "Unsupported format",
        "Completed review paths",
        "Example safe file IDs",
        "Advanced: full audit report",
        "Build / audit details",
    ):
        assert text in combined


def test_advanced_mode_preserves_admin_pages() -> None:
    advanced_tabs = operator_tabs(show_advanced_tools=True)
    for label in (
        "Run & Review",
        "Operator Control Panel",
        "Validation Batch Audit",
        "Validation History",
        "Safety & Governance",
        "Terminology Admin",
    ):
        assert label in advanced_tabs
    assert ADVANCED_OPERATOR_TABS == [
        "Validation Batch Audit",
        "Validation History",
        "Safety & Governance",
        "Terminology Admin",
    ]


def test_component_renderers_remain_available_for_direct_calls() -> None:
    source = app_source()
    assert "def render_current_run_tab(sys_components: dict, *, show_title: bool = True)" in source
    assert "def render_run_review_tab(sys_components: dict)" in source
    assert "render_current_run_tab(sys_components, show_title=False)" in source
    assert "render_review_package_panel(show_title=False)" in source
    assert "def render_review_package_panel(" in review_source()
