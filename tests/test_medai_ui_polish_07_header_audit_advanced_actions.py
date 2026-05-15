from __future__ import annotations

from pathlib import Path

from app.main import operator_tabs, sidebar_status_labels


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
APP_REVIEW = Path(__file__).resolve().parents[1] / "app" / "review_package_viewer.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def review_source() -> str:
    return APP_REVIEW.read_text(encoding="utf-8")


def test_old_sidebar_labels_are_not_restored() -> None:
    source = app_source()
    sidebar_text = "\n".join(sidebar_status_labels(enrichment_enabled=True).values())

    assert "with st.sidebar" not in source
    assert "st.sidebar" not in source
    for old_label in (
        "MKB Status",
        "Hypothesis",
        "Connectors: dxgpt",
        "Enrichment: ON",
    ):
        assert old_label not in sidebar_text


def test_build_audit_details_is_collapsed_by_default() -> None:
    source = app_source()

    assert 'st.expander("Build / audit details", expanded=False)' in source
    assert "Knowledge base" in source
    assert "Draft facts:" in source
    assert "Medical connector:" in source
    assert "Enrichment:" in source


def test_duplicate_success_banner_removed_but_status_pills_remain() -> None:
    source = app_source()

    assert 'st.success("System ready. Medical connector active.")' not in source
    assert "System ready" in source
    assert "Medical connector active" in source


def test_run_review_advanced_actions_are_specific_and_safe() -> None:
    source = app_source()

    assert 'st.expander("Advanced actions", expanded=False)' in source
    assert 'st.expander("Advanced", expanded=False)' not in source
    assert "Clear last report" in source
    assert "Removes the visible latest report only. It does not delete source documents." in source


def test_run_review_sections_and_operator_control_panel_remain_available() -> None:
    combined = app_source() + "\n" + review_source()

    for text in (
        "Run & Review",
        "Current run",
        "Review summary",
        "Review details",
        "Operator Control Panel",
    ):
        assert text in combined
    assert operator_tabs(show_advanced_tools=False) == ["Run & Review", "Operator Control Panel"]
