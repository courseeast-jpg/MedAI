from __future__ import annotations

from pathlib import Path


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_run_review_orientation_uses_plain_local_review_bound_wording() -> None:
    source = app_source()

    assert "Local operator workflow: files stay on this machine, cloud tools stay off" in source
    assert "every result remains for human review until the source document is checked" in source
    assert "Use this section for the active file queue, current run status, and per-file review cards." in source


def test_status_counts_are_explained_as_workflow_not_clinical_acceptance() -> None:
    source = app_source()

    assert "These are workflow statuses only." in source
    assert "They are not diagnosis, treatment advice, or clinical acceptance." in source


def test_result_card_clarifies_operator_summary_and_advanced_metadata() -> None:
    source = app_source()

    assert "Operator summary: confirm the document type, compare with the source" in source
    assert "keep technical metadata collapsed unless needed" in source
    assert "Safe metadata only. This section is collapsed by default" in source
    assert 'st.expander("Advanced technical details", expanded=False)' in source


def test_polish_does_not_add_new_controls_or_mutation_patterns() -> None:
    source = app_source()

    assert source.count("st.button(") == 8
    assert source.count("st.form(") == 0
    assert source.count("st.checkbox(") == 1
    assert source.count("st.radio(") == 0
    assert source.count("st.selectbox(") == 2
    assert source.count("st.text_input(") == 0
    assert source.count("st.text_area(") == 2
    assert source.count("st.number_input(") == 0
    assert source.count("st.rerun(") == 7
    assert source.count("st.experimental_rerun(") == 0
    assert "form_submit_button" not in source


def test_polish_preserves_safety_boundary_language() -> None:
    source = app_source()

    assert "Did not diagnose anything." in source
    assert "Did not send data to the cloud." in source
    assert "Did not accept lab values." in source
    assert "Cue expansion" not in source
