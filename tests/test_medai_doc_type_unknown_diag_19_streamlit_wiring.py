"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-19.

Verifies the default-off Streamlit wiring in
``app/main.py::render_run_result_card``:

1.  ``app/main.py`` contains exactly one DIAG-19 wiring block.
2.  The block lives inside the "Advanced technical details" expander,
    and after the existing DIAG-08A / DIAG-10A / DIAG-12A blocks.
3.  The block uses a try/except guarded import for the DIAG-18 helper.
4.  Both env vars are required (DIAG-18 helper enforces this; wiring
    consumes the helper's plan).
5.  Default-off in real ``os.environ``: helpers stay default-disabled.
6.  Only the both-truthy combination yields a plan from the DIAG-18
    helper.
7.  No buttons / forms / callbacks / actions / state-mutation /
    data-layer-write / document_type mutation surface area in the block.
8.  No raw text / filename / private path tokens rendered.
9.  No auto-accept semantics.
10. No clinical interpretation or clinical inference flags raised.
11. PARK-20 / PARK-21 / PARK-22 tags are not touched.
12. DIAG-17 and DIAG-18 helpers remain default-off when ``os.environ`` is
    consulted directly.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    is_pdf_text_layout_quality_impl_default_disabled,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (
    DIAG_18_UI_ENV_VAR,
    is_pdf_text_layout_quality_ui_default_disabled,
    render_plan_for_pdf_text_layout_quality,
)
from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_19_streamlit_wiring import (
    DIAG_18_IMPORT_LINE,
    DIAG_19_MARKER,
    PHASE_ID,
    RENDER_PLAN_CALL,
    build_report,
    env_combination_audit,
    static_audit_app_main,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_MAIN = REPO_ROOT / "app" / "main.py"


# ── Static-audit asserts ──────────────────────────────────────────────────


def test_app_main_exists():
    assert APP_MAIN.exists()


def test_diag_19_marker_appears_exactly_once_in_app_main():
    body = APP_MAIN.read_text(encoding="utf-8")
    assert body.count(DIAG_19_MARKER) == 1


def test_diag_19_wiring_block_inside_advanced_technical_details_expander():
    s = static_audit_app_main()
    assert s["inside_advanced_technical_details_expander"] is True


def test_diag_19_wiring_block_after_prior_diag_08a_10a_12a_blocks():
    s = static_audit_app_main()
    assert s["after_prior_diag_08a_10a_12a_blocks"] is True


def test_diag_19_wiring_block_uses_try_except_guard():
    s = static_audit_app_main()
    assert s["has_try_except_guard"] is True


def test_diag_19_wiring_block_imports_diag_18_helper_inside_try():
    s = static_audit_app_main()
    assert s["has_diag_18_import_inside_try"] is True
    assert s["has_render_plan_call_inside_try"] is True


def test_diag_19_wiring_block_has_no_forbidden_st_symbols():
    s = static_audit_app_main()
    assert s["forbidden_st_symbols_present"] == []
    assert s["unsafe_st_symbols_used_in_block"] == []


def test_diag_19_wiring_block_has_no_forbidden_tokens():
    s = static_audit_app_main()
    assert s["forbidden_tokens_present"] == []


def test_diag_19_wiring_block_only_uses_safe_st_symbols():
    s = static_audit_app_main()
    safe = {"st.markdown", "st.caption"}
    used = set(s["used_st_symbols_in_block"])
    extra = used - safe
    assert extra == set(), extra


def test_diag_19_import_line_matches_diag_18_module_path():
    assert (
        DIAG_18_IMPORT_LINE
        == "from clinical_knowledge.document_type.pdf_text_layout_quality_ui import"
    )
    assert RENDER_PLAN_CALL == "render_plan_for_pdf_text_layout_quality"


# ── Env-combination semantics (DIAG-18 helper) ────────────────────────────


def test_env_combination_neither_returns_none():
    a = env_combination_audit()
    assert a["neither_env_plan_is_none"] is True


def test_env_combination_metadata_only_returns_none():
    a = env_combination_audit()
    assert a["metadata_only_env_plan_is_none"] is True


def test_env_combination_ui_only_returns_none():
    a = env_combination_audit()
    assert a["ui_only_env_plan_is_none"] is True


def test_env_combination_both_truthy_returns_plan():
    a = env_combination_audit()
    assert a["both_env_plan_is_dict"] is True


def test_helpers_still_default_disabled_in_real_environ():
    assert (
        is_pdf_text_layout_quality_impl_default_disabled(env=os.environ) is True
    )
    assert (
        is_pdf_text_layout_quality_ui_default_disabled(env=os.environ) is True
    )


# ── Build-report fields ───────────────────────────────────────────────────


def test_build_report_identity():
    p = build_report()
    assert p["phase_id"] == PHASE_ID == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"
    assert p["mode"] == "default_off_streamlit_wiring"
    assert p["default_off"] is True
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["metadata_env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert p["ui_env_var"] == DIAG_18_UI_ENV_VAR
    assert p["requires_both_env_vars"] is True
    assert p["diag_17_commit_short"] == "ad7b2d6"
    assert p["diag_18_commit_short"] == "f3c9760"
    assert p["park_20_parking_commit_short"] == "3e46461"
    assert p["park_21_parking_commit_short"] == "9f9e22d"
    assert p["park_22_parking_commit_short"] == "f4d3cc6"


def test_build_report_required_safety_flags_false():
    p = build_report()
    for flag in [
        "default_behavior_changed",
        "runtime_behavior_changed_by_default",
        "streamlit_wiring_enabled_by_default",
        "buttons_added",
        "callbacks_added",
        "actions_added",
        "forms_added",
        "state_mutation_added",
        "data_layer_write_added",
        "document_type_mutation_added",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "raw_text_rendered",
        "raw_filenames_rendered",
        "private_paths_rendered",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_expansion_performed",
        "cue_expansion_recommended",
        "cue_expansion_performed",
        "park_20_tags_touched",
        "park_21_tags_touched",
        "park_22_tags_touched",
    ]:
        assert p[flag] is False, flag


def test_build_report_required_truthy_flags():
    p = build_report()
    assert p["behavior_changed"] is True  # wiring code now exists
    assert p["streamlit_wiring_added"] is True
    assert p["advanced_technical_details_only"] is True
    assert p["all_records_review_bound"] is True


def test_build_report_invariant_counts():
    p = build_report()
    assert p["accepted_count"] == 0
    assert p["auto_accept_allowed_count"] == 0
    assert p["external_api_used_count"] == 0


def test_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 99.98
    assert before["residual_unknown_reduction_track_remaining_pct"] == 0.02
    assert before["whole_medai_project_done_pct"] == 91
    assert before["whole_medai_project_remaining_pct"] == 9
    assert after["residual_unknown_reduction_track_done_pct"] == 99.99
    assert after["residual_unknown_reduction_track_remaining_pct"] == 0.01
    assert after["whole_medai_project_done_pct"] == 91.5
    assert after["whole_medai_project_remaining_pct"] == 8.5


# ── Public-report privacy on disk ─────────────────────────────────────────


REPORT_DIR = REPO_ROOT / "reports/medai_doc_type_unknown_diag_19_streamlit_wiring"


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_19_streamlit_wiring_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_19_streamlit_wiring_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_19_STREAMLIT_WIRING.md",
    }


def test_report_files_exist(written_files):
    for p in written_files.values():
        assert p.exists(), p


def test_written_json_passes_public_payload_privacy(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    r = check_public_report_payload(payload)
    assert r.passed, r.leak_examples_redacted


def test_written_md_passes_public_payload_privacy(written_files):
    body = written_files["md"].read_text(encoding="utf-8")
    r = check_public_report_payload(body)
    assert r.passed, r.leak_examples_redacted


def test_written_summary_passes_public_payload_privacy(written_files):
    body = written_files["summary"].read_text(encoding="utf-8")
    r = check_public_report_payload(body)
    assert r.passed, r.leak_examples_redacted


# ── Block-level structural assertions on app/main.py source ───────────────


_DIAG_19_BLOCK_RE = re.compile(
    r"# MEDAI-DOC-TYPE-UNKNOWN-DIAG-19[\s\S]+?except Exception:\s*\n\s+pass"
)


def _strip_comments(source: str) -> str:
    """Drop ``# comment`` content while keeping line structure."""
    out = []
    for line in source.splitlines():
        code, _, _ = line.partition("#")
        out.append(code)
    return "\n".join(out)


def test_diag_19_block_pattern_extracts_cleanly():
    body = APP_MAIN.read_text(encoding="utf-8")
    matches = _DIAG_19_BLOCK_RE.findall(body)
    assert len(matches) == 1


def test_diag_19_block_uses_only_st_markdown_and_st_caption():
    body = APP_MAIN.read_text(encoding="utf-8")
    matches = _DIAG_19_BLOCK_RE.findall(body)
    assert matches
    block_code = _strip_comments(matches[0])
    used = set(re.findall(r"st\.[a-zA-Z_][a-zA-Z0-9_]*", block_code))
    assert used <= {"st.markdown", "st.caption"}, used


def test_diag_19_block_carries_no_button_form_callback_action_state():
    body = APP_MAIN.read_text(encoding="utf-8")
    matches = _DIAG_19_BLOCK_RE.findall(body)
    block_code = _strip_comments(matches[0])
    for forbidden in (
        "st.button",
        "st.form",
        "st.form_submit_button",
        "st.checkbox",
        "st.radio",
        "st.selectbox",
        "st.text_input",
        "st.text_area",
        "st.number_input",
        "st.date_input",
        "st.time_input",
        "st.file_uploader",
        "st.session_state",
        "st.rerun",
        "on_click=",
        "on_change=",
        "on_submit=",
    ):
        assert forbidden not in block_code, forbidden


def test_diag_19_block_does_not_emit_raw_text_or_filename_or_path():
    body = APP_MAIN.read_text(encoding="utf-8")
    matches = _DIAG_19_BLOCK_RE.findall(body)
    block_code = _strip_comments(matches[0])
    for token in ("extracted_text", "raw_text", "filename", "/home/", "/var/"):
        assert token not in block_code


def test_diag_19_block_does_not_mention_park_tags():
    body = APP_MAIN.read_text(encoding="utf-8")
    matches = _DIAG_19_BLOCK_RE.findall(body)
    block_code = _strip_comments(matches[0])
    # PARK tags are GitHub-only metadata; the runtime block must not
    # reference them, much less touch them.
    for tag in (
        "medai-unknown-diag-language-metadata-ready",
        "medai-final-parked-post-unknown-diag-language-metadata",
        "medai-text-layer-eval-spec-ready",
        "medai-final-parked-post-diag-16",
        "medai-pdf-text-layout-quality-default-off-ready",
        "medai-final-parked-post-diag-18",
    ):
        assert tag not in block_code
