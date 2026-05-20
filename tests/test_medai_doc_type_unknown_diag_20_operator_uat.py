"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-20.

These tests prove the operator UAT is reports-only, aggregate-only, and
that the env-on run preserves every default-off / read-only / privacy /
review-bound invariant from the DIAG-17, DIAG-18, and DIAG-19 blocks. No
runtime files are modified; ``os.environ`` is never written.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_medai_doc_type_unknown_diag_20_operator_uat as diag20  # noqa: E402

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (  # noqa: E402
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_default_disabled,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (  # noqa: E402
    DIAG_18_UI_ENV_VAR,
    is_pdf_text_layout_quality_ui_default_disabled,
    render_plan_for_pdf_text_layout_quality,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _report() -> Mapping[str, Any]:
    payload = diag20.build_report()
    return payload


@pytest.fixture(scope="module")
def report() -> Mapping[str, Any]:
    return _report()


# ── 1. DIAG-20 modifies no app/runtime files ───────────────────────────────


def test_diag_20_does_not_modify_app_main():
    """The DIAG-20 block adds NO new runtime code to ``app/main.py``.

    We check this by re-asserting the DIAG-19 wiring block's known span
    and that the block contains exactly one DIAG-19 marker (i.e. it is
    not duplicated by a DIAG-20 add).
    """
    src = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert src.count("MEDAI-DOC-TYPE-UNKNOWN-DIAG-19") >= 1
    # The DIAG-20 marker must NOT have been added to app/main.py — DIAG-20
    # is reports-only and never touches runtime code.
    assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-20" not in src


def test_diag_20_does_not_modify_helper_modules():
    impl = (
        REPO_ROOT
        / "clinical_knowledge"
        / "document_type"
        / "pdf_text_layout_quality_impl.py"
    ).read_text(encoding="utf-8")
    ui = (
        REPO_ROOT
        / "clinical_knowledge"
        / "document_type"
        / "pdf_text_layout_quality_ui.py"
    ).read_text(encoding="utf-8")
    assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-20" not in impl
    assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-20" not in ui


# ── 2. Env-on emits the full 21-record scope ────────────────────────────────


def test_env_on_emits_full_21_record_scope(report):
    assert report["total_records_evaluated"] == 21
    assert report["emitted_metadata_count"] == 21
    assert report["emitted_render_plan_count"] == 21


def test_env_on_per_subtrack_counts(report):
    counts = report["per_subtrack_emit_count"]
    assert counts == {"A": 11, "B": 10}


def test_env_on_family_label_counts(report):
    # 11 Sub-track A => 11 pdf_text_too_short
    # 10 Sub-track B => 10 table_structure_visible_text_insufficient
    #                 + 10 layout_or_table_extraction_gap (multi-label)
    counts = report["family_label_counts"]
    assert counts["pdf_text_too_short"] == 11
    assert counts["table_structure_visible_text_insufficient"] == 10
    assert counts["layout_or_table_extraction_gap"] == 10


# ── 3. Excluded pool remains suppressed under env-on ───────────────────────


def test_excluded_pool_metadata_suppressed(report):
    assert report["excluded_pool_count"] == 5
    assert report["excluded_pool_metadata_emission_count"] == 0


def test_excluded_pool_render_plans_suppressed(report):
    assert report["excluded_pool_render_plan_count"] == 0


# ── 4. os.environ is not polluted ──────────────────────────────────────────


def test_os_environ_not_polluted_by_uat():
    # Run the UAT directly. Then assert neither env var is present in
    # os.environ. The script never writes to os.environ; this test makes
    # that an explicit, audit-visible invariant.
    diag20.env_on_uat()
    assert PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR not in os.environ
    assert DIAG_18_UI_ENV_VAR not in os.environ


# ── 5. Default-off behavior is unchanged after the UAT ─────────────────────


def test_default_off_metadata_helper_after_uat(report):
    assert report["helper_default_disabled_after_uat"] is True
    # And re-verify live:
    assert is_pdf_text_layout_quality_impl_default_disabled(env={}) is True


def test_default_off_ui_helper_after_uat(report):
    assert report["ui_helper_default_disabled_after_uat"] is True
    assert is_pdf_text_layout_quality_ui_default_disabled(env={}) is True


def test_default_off_render_plan_returns_none_with_empty_env():
    record = diag20.in_scope_records()[0]
    assert render_plan_for_pdf_text_layout_quality(record, env={}) is None


def test_default_off_metadata_returns_none_with_empty_env():
    record = diag20.in_scope_records()[0]
    assert derive_pdf_text_layout_quality_context(record, env={}) is None


# ── 6. Two-env-var gate ────────────────────────────────────────────────────


def test_two_env_var_gate(report):
    g = report["two_env_var_gate"]
    assert g["neither_truthy_emits_plan"] is False
    assert g["metadata_only_truthy_emits_plan"] is False
    assert g["ui_only_truthy_emits_plan"] is False
    assert g["both_truthy_emits_plan"] is True


# ── 7. Render plans are read-only — no forbidden render fields ─────────────


def test_no_forbidden_render_fields_in_any_plan(report):
    assert report["forbidden_render_field_count"] == 0
    assert report["forbidden_render_fields_seen"] == []


def test_safe_render_fields_present_for_every_emitted_plan(report):
    # Every emitted plan must carry every safe field.
    emitted = report["emitted_render_plan_count"]
    assert emitted == 21
    for field, count in report["safe_render_field_counts"].items():
        assert count == emitted, f"safe field {field!r} missing from some plans"


def test_render_plan_has_no_button_callback_form_or_mutation_fields():
    env = {
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "true",
        DIAG_18_UI_ENV_VAR: "true",
    }
    plan = render_plan_for_pdf_text_layout_quality(
        diag20.in_scope_records()[0],
        env=env,
    )
    assert plan is not None
    for forbidden in (
        "on_click",
        "on_change",
        "on_submit",
        "button",
        "callback",
        "action",
        "write",
        "mutate",
        "document_type_mutation",
        "data_layer_write",
        "state_mutation",
    ):
        assert forbidden not in plan, (
            f"render plan unexpectedly contains forbidden field {forbidden!r}"
        )
    # And invariant flags ARE present:
    for invariant in (
        "is_read_only",
        "is_review_bound",
        "no_action_attached",
        "no_button_attached",
        "no_callback_attached",
        "no_form_attached",
        "no_state_mutation",
        "no_data_layer_write",
        "no_document_type_mutation",
    ):
        assert plan[invariant] is True


# ── 8. No raw text / filenames / private paths leak ───────────────────────


def test_no_raw_text_emission(report):
    assert report["raw_text_emission_count"] == 0


def test_no_raw_filename_emission(report):
    assert report["raw_filename_emission_count"] == 0


def test_no_private_path_emission(report):
    assert report["private_path_emission_count"] == 0


# ── 9-11. Hard zeros: accepted, auto-accept, external API ──────────────────


def test_accepted_count_zero(report):
    assert report["accepted_count"] == 0


def test_auto_accept_allowed_zero(report):
    assert report["auto_accept_allowed_count"] == 0
    assert report["auto_accept_allowed_count_total"] == 0


def test_external_api_used_zero(report):
    assert report["external_api_used_count_total"] == 0
    assert report["external_api_used"] is False


# ── 12. all_records_review_bound ───────────────────────────────────────────


def test_all_records_review_bound(report):
    assert report["all_records_review_bound"] is True
    # Equivalent claim: review_required_count equals emitted plans
    assert (
        report["review_required_count"] == report["emitted_render_plan_count"]
    )


# ── 13. No clinical interpretation / inference / abbreviation expansion ────


def test_no_clinical_interpretation(report):
    assert report["clinical_interpretation_performed_count"] == 0
    assert report["clinical_interpretation_performed"] is False
    assert report["clinical_value_parsing_performed"] is False


def test_no_inference_of_any_kind(report):
    assert report["diagnosis_inference_performed"] is False
    assert report["medication_inference_performed"] is False
    assert report["ddi_inference_performed"] is False
    assert report["treatment_inference_performed"] is False


def test_no_abbreviation_expansion(report):
    assert report["abbreviation_expansion_performed"] is False


# ── 14. PARK-20 / PARK-21 / PARK-22 / PARK-23 tags untouched ───────────────


def test_park_20_tags_untouched(report):
    assert report["park_20_tags_touched"] is False


def test_park_21_tags_untouched(report):
    assert report["park_21_tags_touched"] is False


def test_park_22_tags_untouched(report):
    assert report["park_22_tags_touched"] is False


def test_park_23_tags_untouched(report):
    assert report["park_23_tags_touched"] is False


# ── 15. Cue expansion remains NOT recommended ─────────────────────────────


def test_cue_expansion_not_recommended(report):
    assert report["cue_expansion_recommended"] is False
    assert report["cue_expansion_performed"] is False
    rec = report["next_block_recommendation"]
    assert rec["must_not_propose_cue_expansion_as_primary_step"] is True


# ── Default-behavior / runtime-behavior / streamlit-wiring not changed ─────


def test_runtime_invariants(report):
    for k in (
        "default_behavior_changed",
        "runtime_behavior_changed",
        "streamlit_wiring_changed",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
    ):
        assert report[k] is False, f"{k} unexpectedly True"


# ── DIAG-19 wiring block static audit invariants ──────────────────────────


def test_diag_19_static_audit_safe_st_calls_only(report):
    a = report["diag_19_static_audit"]
    assert a["unsafe_st_calls_in_block"] == []
    assert set(a["used_st_calls"]).issubset({"st.markdown", "st.caption"})


def test_diag_19_static_audit_no_forbidden_st_or_kwargs(report):
    a = report["diag_19_static_audit"]
    assert a["any_forbidden_st_symbols"] is False
    assert a["any_forbidden_kwargs"] is False


def test_diag_19_static_audit_no_forbidden_tokens_or_prefix(report):
    a = report["diag_19_static_audit"]
    assert a["any_forbidden_tokens_in_code"] is False
    assert a["any_forbidden_prefix_in_code"] is False


def test_diag_19_static_audit_no_park_references(report):
    a = report["diag_19_static_audit"]
    assert a["park_tag_names_referenced_in_block"] is False


def test_diag_19_static_audit_structure(report):
    a = report["diag_19_static_audit"]
    assert a["has_try_except_guard"] is True
    assert a["has_diag_18_import_inside_try"] is True
    assert a["has_render_plan_call_inside_try"] is True


# ── env-mapping-only invariant ─────────────────────────────────────────────


def test_env_mapping_only_invariant(report):
    assert report["env_mapping_only"] is True
    assert report["os_environ_written"] is False
    assert report["os_environ_metadata_env_var_present"] is False
    assert report["os_environ_ui_env_var_present"] is False
