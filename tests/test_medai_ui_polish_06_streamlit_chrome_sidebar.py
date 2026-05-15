from __future__ import annotations

from pathlib import Path

from app.main import sidebar_status_labels


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_sidebar_labels_use_operator_wording() -> None:
    labels = sidebar_status_labels(enrichment_enabled=True)

    assert labels["knowledge_base"] == "Knowledge base"
    assert labels["active"] == "Active"
    assert labels["draft_facts"] == "Draft facts"
    assert labels["connector_status"] == "Medical connector active"
    assert labels["enrichment_status"] == "Enrichment enabled"


def test_old_sidebar_labels_are_not_primary_sidebar_text() -> None:
    sidebar_text = "\n".join(sidebar_status_labels(enrichment_enabled=True).values())

    for old_label in (
        "MKB Status",
        "Active facts",
        "Hypothesis",
        "Connectors: dxgpt",
        "Enrichment: ON",
    ):
        assert old_label not in sidebar_text


def test_raw_connector_and_enrichment_values_remain_audit_only() -> None:
    source = app_source()
    audit_index = source.index('st.expander("Build / audit details", expanded=False)')

    assert "Internal connector:" in source
    assert "Enrichment:" in source
    assert source.index("Internal connector:") > audit_index
    assert source.index("Enrichment:") > audit_index


def test_streamlit_chrome_css_is_scoped_to_framework_containers() -> None:
    source = app_source()

    assert "MEDAI-UI-POLISH-06: hide Streamlit framework chrome only." in source
    for selector in (
        '[data-testid="stToolbar"]',
        '[data-testid="stDecoration"]',
        '[data-testid="stStatusWidget"]',
        '[data-testid="stDeployButton"]',
        "#MainMenu",
        "footer",
    ):
        assert selector in source


def test_streamlit_chrome_css_does_not_hide_medai_controls() -> None:
    source = app_source()
    css_start = source.index("MEDAI-UI-POLISH-06: hide Streamlit framework chrome only.")
    css_end = source.index(".stMarkdown h1 a", css_start)
    chrome_css = source[css_start:css_end]

    for selector in ("stButton", "stTabs", "stExpander", "stFileUploader", "stForm"):
        assert selector not in chrome_css
