"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-18.

Verifies the default-off read-only operator-surface helper for DIAG-17
PDF text/layout quality metadata:

1.  Both env vars unset → render plan is None.
2.  Only the DIAG-17 metadata env var truthy → render plan is None.
3.  Only the DIAG-18 UI env var truthy → render plan is None.
4.  Both env vars truthy AND record matches signature → plan is produced.
5.  Render plan contains only controlled-vocabulary / safe operator wording.
6.  Render plan contains no raw text / filenames / private paths.
7.  Render plan contains no button / callback / action / state-mutation /
    accept / reject / approve / deny keys.
8.  No accept / reject / auto-accept semantics in the plan.
9.  Clinical-interpretation flags remain false.
10. No diagnosis / medication / DDI / treatment inference.
11. No abbreviation expansion.
12. DIAG-17 helper remains default-off outside explicit env-on.
13. PARK-20 / PARK-21 tags are not touched by the surface.
14. DIAG-18 UI env var is DISTINCT from every prior MEDAI doc-type env var.
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
    KNOWN_DOC_TYPE_ENV_VARS,
    OPERATOR_DISCLAIMER,
    OPERATOR_DISPLAY_HEADING,
    OPERATOR_EXPANDER_LABEL,
    OPERATOR_VOCAB_TOKEN,
    QUALITY_FAMILY_LABEL_DESCRIPTION,
    SOURCE_PHASE,
    is_pdf_text_layout_quality_ui_default_disabled,
    is_pdf_text_layout_quality_ui_enabled,
    render_plan_for_pdf_text_layout_quality,
    requires_both_env_vars_truthy,
)
from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _subtrack_a_record(**overrides) -> dict:
    r = {
        "anonymous_id": "file_001",
        "pdf_text_layer_detected": "yes",
        "image_like_pdf": "no",
        "alphabetic_content_bucket": "high",
        "native_text_length_bucket": "short",
        "table_like_structure_detected": "no",
    }
    r.update(overrides)
    return r


def _subtrack_b_record(**overrides) -> dict:
    r = {
        "anonymous_id": "file_012",
        "pdf_text_layer_detected": "yes",
        "image_like_pdf": "no",
        "alphabetic_content_bucket": "high",
        "native_text_length_bucket": "none",
        "table_like_structure_detected": "yes",
    }
    r.update(overrides)
    return r


def _both_truthy_env() -> dict:
    return {
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1",
        DIAG_18_UI_ENV_VAR: "1",
    }


# ── Env-var name distinctness ─────────────────────────────────────────────


def test_diag_18_ui_env_var_is_distinct_and_correct():
    assert DIAG_18_UI_ENV_VAR == "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED"
    assert DIAG_18_UI_ENV_VAR != PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    for other in KNOWN_DOC_TYPE_ENV_VARS:
        assert DIAG_18_UI_ENV_VAR != other


def test_known_doc_type_env_vars_set_contains_prior_four():
    expected = {
        "MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED",
        "MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED",
        "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED",
        "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED",
    }
    assert set(KNOWN_DOC_TYPE_ENV_VARS) == expected


# ── Env-gating predicates ─────────────────────────────────────────────────


def test_ui_predicate_default_disabled_when_env_empty():
    assert is_pdf_text_layout_quality_ui_default_disabled(env={}) is True
    assert is_pdf_text_layout_quality_ui_enabled(env={}) is False


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "disabled"])
def test_ui_predicate_disabled_for_falsy_env_values(val):
    env = {DIAG_18_UI_ENV_VAR: val}
    assert is_pdf_text_layout_quality_ui_enabled(env=env) is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "enabled", "TRUE", "Yes"])
def test_ui_predicate_enabled_for_truthy_env_values(val):
    env = {DIAG_18_UI_ENV_VAR: val}
    assert is_pdf_text_layout_quality_ui_enabled(env=env) is True


def test_requires_both_env_vars_truthy_helper():
    assert requires_both_env_vars_truthy(env={}) is False
    assert (
        requires_both_env_vars_truthy(
            env={PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
        )
        is False
    )
    assert (
        requires_both_env_vars_truthy(env={DIAG_18_UI_ENV_VAR: "1"}) is False
    )
    assert requires_both_env_vars_truthy(env=_both_truthy_env()) is True


# ── render_plan_for_pdf_text_layout_quality — env combinations ────────────


def test_render_plan_with_neither_env_returns_none():
    r = _subtrack_a_record()
    assert render_plan_for_pdf_text_layout_quality(r, env={}) is None


def test_render_plan_with_only_metadata_env_truthy_returns_none():
    r = _subtrack_a_record()
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    assert render_plan_for_pdf_text_layout_quality(r, env=env) is None


def test_render_plan_with_only_ui_env_truthy_returns_none():
    r = _subtrack_a_record()
    env = {DIAG_18_UI_ENV_VAR: "1"}
    assert render_plan_for_pdf_text_layout_quality(r, env=env) is None


def test_render_plan_with_both_env_vars_truthy_returns_plan():
    r = _subtrack_a_record()
    plan = render_plan_for_pdf_text_layout_quality(r, env=_both_truthy_env())
    assert plan is not None


def test_render_plan_with_both_env_vars_truthy_for_subtrack_b():
    r = _subtrack_b_record()
    plan = render_plan_for_pdf_text_layout_quality(r, env=_both_truthy_env())
    assert plan is not None


def test_render_plan_explicit_enabled_false_overrides_both_envs():
    r = _subtrack_a_record()
    plan = render_plan_for_pdf_text_layout_quality(
        r, enabled=False, env=_both_truthy_env()
    )
    assert plan is None


def test_render_plan_returns_none_when_record_does_not_match_signature():
    # Excluded-pool: image-like PDF
    r = _subtrack_a_record(image_like_pdf="yes")
    plan = render_plan_for_pdf_text_layout_quality(r, env=_both_truthy_env())
    assert plan is None


# ── Plan contents: controlled vocabulary, safe wording ────────────────────


def test_plan_contains_controlled_vocabulary_wording():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    assert plan["expander_label"] == OPERATOR_EXPANDER_LABEL
    assert plan["badge_text"] == OPERATOR_DISPLAY_HEADING
    assert plan["badge_vocab_token"] == OPERATOR_VOCAB_TOKEN
    assert plan["badge_source_block"] == SOURCE_PHASE
    assert plan["disclaimer_line"] == OPERATOR_DISCLAIMER


def test_plan_records_both_env_vars():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    assert plan["env_vars"]["metadata_env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert plan["env_vars"]["ui_env_var"] == DIAG_18_UI_ENV_VAR


def test_plan_quality_family_labels_have_descriptions():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_b_record(), env=_both_truthy_env()
    )
    for label in plan["quality_family"]:
        assert label in QUALITY_FAMILY_LABEL_DESCRIPTION


def test_plan_markdown_lines_carry_no_raw_text():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    body = "\n".join(plan["markdown_lines"])
    assert "extracted_text" not in body.lower()
    assert "raw_text" not in body.lower()
    assert "filename" not in body.lower()


# ── Plan contents: NO action / button / callback / state-mutation keys ────


_FORBIDDEN_TOKENS = frozenset(
    {"callback", "button", "submit", "approve", "deny", "reject", "mutate"}
)
_FORBIDDEN_PREFIXES = (
    "on_click",
    "on_submit",
    "on_change",
    "callback",
    "button",
    "submit",
    "approve",
    "deny",
    "reject",
    "write_",
)


def _key_is_forbidden(k: str) -> bool:
    kl = k.lower()
    if kl.startswith("no_") or kl.startswith("is_"):
        return False
    if kl in _FORBIDDEN_TOKENS:
        return True
    for prefix in _FORBIDDEN_PREFIXES:
        if kl.startswith(prefix):
            return True
    tokens = kl.split("_")
    if _FORBIDDEN_TOKENS & set(tokens):
        return True
    return False


def test_plan_has_no_forbidden_action_keys():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    bad = [k for k in plan.keys() if _key_is_forbidden(k)]
    assert bad == [], bad


def test_plan_invariant_flags_assert_read_only():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    assert plan["is_read_only"] is True
    assert plan["is_review_bound"] is True
    assert plan["no_action_attached"] is True
    assert plan["no_button_attached"] is True
    assert plan["no_callback_attached"] is True
    assert plan["no_form_attached"] is True
    assert plan["no_state_mutation"] is True
    assert plan["no_data_layer_write"] is True
    assert plan["no_document_type_mutation"] is True


def test_plan_invariant_flags_assert_no_clinical_or_inference():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    for k in [
        "is_auto_accept",
        "is_clinical_classification",
        "is_active_clinical_fact",
        "is_final_document_type",
        "is_data_layer_document_type_change",
        "clinical_interpretation_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_parsed",
        "abbreviation_expanded",
        "lab_value_parsed",
        "external_api_used",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "classifier_behavior_changed",
        "thresholds_or_scoring_changed",
        "cue_packs_added",
    ]:
        assert plan[k] is False, k


def test_plan_invariant_flags_park_tags_untouched():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    assert plan["park_20_tags_touched"] is False
    assert plan["park_21_tags_touched"] is False


# ── Plan privacy: no raw text / filenames / private paths in serialization ─


_RAW_FILENAME_HINTS = (
    re.compile(r"\.pdf\b", re.IGNORECASE),
    re.compile(r"\.docx?\b", re.IGNORECASE),
    re.compile(r"\.xlsx?\b", re.IGNORECASE),
    re.compile(r"\.png\b", re.IGNORECASE),
    re.compile(r"\.jpe?g\b", re.IGNORECASE),
)


def test_plan_serialization_carries_no_raw_filenames():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    blob = json.dumps(plan)
    for rx in _RAW_FILENAME_HINTS:
        assert not rx.search(blob)


def test_plan_serialization_carries_no_unix_private_paths():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    blob = json.dumps(plan)
    for needle in ("/home/", "/var/", "/etc/", "/usr/", "/tmp/", "/opt/"):
        assert needle not in blob


def test_plan_rendering_flags_carry_no_raw_or_private_data():
    plan = render_plan_for_pdf_text_layout_quality(
        _subtrack_a_record(), env=_both_truthy_env()
    )
    for k in [
        "raw_text_rendered",
        "raw_ocr_text_rendered",
        "raw_document_text_rendered",
        "raw_filename_rendered",
        "private_path_rendered",
        "phi_rendered",
        "secret_rendered",
    ]:
        assert plan[k] is False, k


# ── Purity: record is not mutated ─────────────────────────────────────────


def test_render_plan_does_not_mutate_record():
    r = _subtrack_a_record()
    snapshot = dict(r)
    render_plan_for_pdf_text_layout_quality(r, env=_both_truthy_env())
    assert r == snapshot


def test_render_plan_default_off_does_not_mutate_record():
    r = _subtrack_a_record()
    snapshot = dict(r)
    render_plan_for_pdf_text_layout_quality(r, env={})
    assert r == snapshot


# ── DIAG-17 helper default-off survives DIAG-18 import ────────────────────


def test_diag_17_helper_still_default_off_when_real_env_has_no_diag_17_flag():
    # The DIAG-17 metadata env var should NOT be set in the real os.environ
    # by DIAG-18 imports / module load.
    assert (
        is_pdf_text_layout_quality_impl_default_disabled(env=os.environ) is True
    )


# ── No Streamlit imports in the helper or in app/main.py ──────────────────


def test_ui_helper_does_not_import_streamlit():
    ui_path = (
        Path(__file__).resolve().parents[1]
        / "clinical_knowledge"
        / "document_type"
        / "pdf_text_layout_quality_ui.py"
    )
    body = ui_path.read_text(encoding="utf-8")
    assert "import streamlit" not in body
    assert "from streamlit" not in body


def test_diag_18_helper_wiring_only_inside_diag_19_marked_block():
    """The DIAG-18 helper must only be referenced inside an explicit
    MEDAI-DOC-TYPE-UNKNOWN-DIAG-19 wiring block — never elsewhere in
    app/main.py. Prior to DIAG-19 this test asserted the helper was
    absent entirely; DIAG-19 deliberately wires it through, so the
    invariant becomes "wired only via the DIAG-19 block".
    """
    app_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    # Locate the DIAG-19 block boundaries (marker → next `except Exception: pass`).
    marker = "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"
    idx = body.find(marker)
    if idx == -1:
        # Pre-DIAG-19 world: helper must not be referenced.
        assert "pdf_text_layout_quality_ui" not in body
        assert "render_plan_for_pdf_text_layout_quality" not in body
        return
    m = re.search(r"except Exception:\s*\n\s+pass", body[idx:])
    assert m is not None, "DIAG-19 marker present without try/except guard"
    diag_19_block = body[idx : idx + m.end()]
    outside_diag_19 = body[:idx] + body[idx + m.end() :]
    # The helper module path and the render-plan function name must NOT
    # appear anywhere outside the DIAG-19 block.
    assert "pdf_text_layout_quality_ui" not in outside_diag_19
    assert "render_plan_for_pdf_text_layout_quality" not in outside_diag_19
    # And they MUST appear inside the DIAG-19 block (positive invariant).
    assert "pdf_text_layout_quality_ui" in diag_19_block
    assert "render_plan_for_pdf_text_layout_quality" in diag_19_block


def test_diag_18_helper_not_re_exported_from_document_type_init():
    init_path = (
        Path(__file__).resolve().parents[1]
        / "clinical_knowledge"
        / "document_type"
        / "__init__.py"
    )
    if not init_path.exists():
        pytest.skip("document_type/__init__.py absent")
    body = init_path.read_text(encoding="utf-8")
    assert "pdf_text_layout_quality_ui" not in body


# ── on-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_18_PDF_TEXT_LAYOUT_QUALITY_UI.md",
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


def test_written_json_records_required_fields(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["phase_id"] == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-18"
    assert payload["mode"] == "default_off_read_only_operator_surface"
    assert payload["default_off"] is True
    assert payload["read_only"] is True
    assert payload["metadata_env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert payload["ui_env_var"] == DIAG_18_UI_ENV_VAR
    assert payload["requires_both_env_vars"] is True


def test_written_json_safety_invariants(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    for flag in [
        "default_behavior_changed",
        "runtime_behavior_changed_by_default",
        "operator_ui_surface_enabled_by_default",
        "buttons_added",
        "callbacks_added",
        "actions_added",
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
    ]:
        assert payload[flag] is False, flag


def test_written_json_invariant_counts(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["accepted_count"] == 0
    assert payload["auto_accept_allowed_count"] == 0
    assert payload["external_api_used_count"] == 0
    assert payload["all_records_review_bound"] is True
    assert payload["behavior_changed"] is True
    assert payload["operator_ui_surface_added"] is True


def test_written_json_default_off_and_read_only_proofs(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    proof = payload["default_off_proof"]
    assert proof["neither_env_in_scope_emissions_must_be_zero"] is True
    assert proof["metadata_only_env_in_scope_emissions_must_be_zero"] is True
    assert proof["ui_only_env_in_scope_emissions_must_be_zero"] is True
    assert proof["both_env_in_scope_emissions_must_equal_21"] is True
    assert proof["both_env_excluded_pool_emissions_must_be_zero"] is True
    assert proof["helper_still_default_disabled_after_audit"] is True
    ro = payload["read_only_proof"]
    assert ro["forbidden_plan_keys_present"] == []
    for key in [
        "is_read_only_count_must_equal_21",
        "no_action_attached_count_must_equal_21",
        "no_button_attached_count_must_equal_21",
        "no_callback_attached_count_must_equal_21",
        "no_state_mutation_count_must_equal_21",
        "no_data_layer_write_count_must_equal_21",
        "no_document_type_mutation_count_must_equal_21",
        "raw_text_rendered_count_must_be_zero",
        "raw_filename_rendered_count_must_be_zero",
        "private_path_rendered_count_must_be_zero",
        "clinical_interpretation_count_must_be_zero",
        "auto_accept_count_must_be_zero",
        "diagnosis_medication_ddi_treatment_inference_counts_must_be_zero",
        "abbreviation_expanded_count_must_be_zero",
        "external_api_used_count_must_be_zero",
        "park_20_touch_count_must_be_zero",
        "park_21_touch_count_must_be_zero",
    ]:
        assert ro[key] is True, key
