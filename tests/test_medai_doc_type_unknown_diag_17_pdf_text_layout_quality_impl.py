"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-17.

Verifies the default-off PDF text/layout quality metadata helper:

1.  Default-off returns no metadata / no-op.
2.  Env-on emits only controlled-vocabulary metadata.
3.  No raw text / filenames / private paths in output.
4.  Unknown/review-bound invariants are preserved.
5.  accepted_count remains 0.
6.  auto_accept_allowed_count remains 0.
7.  external_api_used_count remains 0.
8.  No clinical interpretation flags are true.
9.  Env-var separation holds (DIAG-17 env var is DISTINCT from
    DIAG-07A/09A/11A env vars).
10. PARK-20 / PARK-21 tags are not referenced or touched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    DISCLAIMER,
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    SOURCE_PHASE,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_default_disabled,
    is_pdf_text_layout_quality_impl_enabled,
    matches_pdf_text_layout_quality_signature,
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


# ── Env-gating predicate ───────────────────────────────────────────────────


def test_env_var_name_is_distinct_and_correct():
    assert (
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
        == "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED"
    )
    for other in [
        "MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED",
        "MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED",
        "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED",
    ]:
        assert PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR != other


def test_predicate_default_disabled_when_env_empty():
    assert is_pdf_text_layout_quality_impl_default_disabled(env={}) is True
    assert is_pdf_text_layout_quality_impl_enabled(env={}) is False


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "disabled"])
def test_predicate_disabled_for_falsy_env_values(val):
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: val}
    assert is_pdf_text_layout_quality_impl_enabled(env=env) is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "enabled", "TRUE", "Yes"])
def test_predicate_enabled_for_truthy_env_values(val):
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: val}
    assert is_pdf_text_layout_quality_impl_enabled(env=env) is True


def test_other_env_vars_do_not_enable_diag_17_helper():
    env = {
        "MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED": "1",
        "MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED": "1",
        "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED": "1",
    }
    assert is_pdf_text_layout_quality_impl_enabled(env=env) is False


# ── Positive-signature predicate ───────────────────────────────────────────


def test_signature_accepts_subtrack_a_record():
    assert matches_pdf_text_layout_quality_signature(_subtrack_a_record()) is True


def test_signature_accepts_subtrack_b_record():
    assert matches_pdf_text_layout_quality_signature(_subtrack_b_record()) is True


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("pdf_text_layer_detected", "no"),
        ("image_like_pdf", "yes"),
        ("alphabetic_content_bucket", "low"),
        ("alphabetic_content_bucket", "medium"),
        ("native_text_length_bucket", "long"),
        ("table_like_structure_detected", "unknown"),
    ],
)
def test_signature_rejects_excluded_pool(field, bad_value):
    r = _subtrack_a_record()
    r[field] = bad_value
    assert matches_pdf_text_layout_quality_signature(r) is False


# ── derive_pdf_text_layout_quality_context — default-off path ─────────────


def test_default_off_returns_none_for_matching_record():
    r = _subtrack_a_record()
    assert derive_pdf_text_layout_quality_context(r, env={}) is None


def test_default_off_returns_none_for_subtrack_b():
    r = _subtrack_b_record()
    assert derive_pdf_text_layout_quality_context(r, env={}) is None


def test_explicit_off_overrides_truthy_env():
    r = _subtrack_a_record()
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    assert derive_pdf_text_layout_quality_context(r, enabled=False, env=env) is None


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "disabled", ""])
def test_falsy_env_returns_none(val):
    r = _subtrack_a_record()
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: val}
    assert derive_pdf_text_layout_quality_context(r, env=env) is None


# ── derive_pdf_text_layout_quality_context — explicit-on path ─────────────


def test_explicit_on_kwarg_emits_for_subtrack_a():
    r = _subtrack_a_record()
    plan = derive_pdf_text_layout_quality_context(r, enabled=True, env={})
    assert plan is not None
    assert plan["enabled"] is True
    assert plan["source_phase"] == SOURCE_PHASE == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17"
    assert plan["env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR


def test_explicit_on_env_emits_for_subtrack_a():
    r = _subtrack_a_record()
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    plan = derive_pdf_text_layout_quality_context(r, env=env)
    assert plan is not None


def test_explicit_on_env_emits_for_subtrack_b():
    r = _subtrack_b_record()
    env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    plan = derive_pdf_text_layout_quality_context(r, env=env)
    assert plan is not None


# ── Quality-family vocabulary ──────────────────────────────────────────────


def test_subtrack_a_emits_pdf_text_too_short():
    r = _subtrack_a_record()
    plan = derive_pdf_text_layout_quality_context(r, enabled=True, env={})
    assert "pdf_text_too_short" in plan["quality_family"]


def test_subtrack_b_emits_table_and_layout_labels():
    r = _subtrack_b_record()
    plan = derive_pdf_text_layout_quality_context(r, enabled=True, env={})
    fam = plan["quality_family"]
    assert "table_structure_visible_text_insufficient" in fam
    assert "layout_or_table_extraction_gap" in fam


def test_quality_family_labels_are_from_controlled_vocab():
    for r in [_subtrack_a_record(), _subtrack_b_record()]:
        plan = derive_pdf_text_layout_quality_context(r, enabled=True, env={})
        assert plan is not None
        for v in plan["quality_family"]:
            assert v in QUALITY_FAMILY_VALUES


# ── Review-bound / no-auto-accept / no clinical interpretation ────────────


def test_emitted_plan_marks_review_required_true():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    assert plan["review_required"] is True


def test_emitted_plan_marks_auto_accept_allowed_false():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    assert plan["auto_accept_allowed"] is False


def test_emitted_plan_marks_no_clinical_interpretation():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    for k in [
        "clinical_interpretation_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_parsed",
        "abbreviation_expanded",
        "lab_value_parsed",
    ]:
        assert plan[k] is False, k


def test_emitted_plan_marks_no_environment_or_extraction_changes():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    for k in [
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "classifier_behavior_changed",
        "thresholds_or_scoring_changed",
        "cue_packs_added",
        "external_api_used",
        "data_layer_document_type_changed",
        "raw_language_detector_output_changed",
    ]:
        assert plan[k] is False, k


def test_emitted_plan_marks_no_park_tag_touch():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    assert plan["park_20_tags_touched"] is False
    assert plan["park_21_tags_touched"] is False


# ── Privacy: no raw text / filenames / private paths in emitted plan ──────


def test_emitted_plan_flags_no_raw_or_private_data():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    for k in [
        "raw_text_emitted",
        "raw_ocr_text_emitted",
        "raw_filename_emitted",
        "private_path_emitted",
        "phi_emitted",
        "secret_emitted",
    ]:
        assert plan[k] is False, k


_RAW_FILENAME_HINTS = (
    re.compile(r"\.pdf\b", re.IGNORECASE),
    re.compile(r"\.docx?\b", re.IGNORECASE),
    re.compile(r"\.xlsx?\b", re.IGNORECASE),
    re.compile(r"\.png\b", re.IGNORECASE),
    re.compile(r"\.jpe?g\b", re.IGNORECASE),
)


def test_emitted_plan_serialization_carries_no_raw_filenames():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    blob = json.dumps(plan)
    for rx in _RAW_FILENAME_HINTS:
        assert not rx.search(blob)


def test_emitted_plan_serialization_carries_no_unix_private_paths():
    plan = derive_pdf_text_layout_quality_context(
        _subtrack_a_record(), enabled=True, env={}
    )
    blob = json.dumps(plan)
    for needle in ("/home/", "/var/", "/etc/", "/usr/", "/tmp/", "/opt/"):
        assert needle not in blob


# ── Purity: record is not mutated ─────────────────────────────────────────


def test_helper_does_not_mutate_record():
    r = _subtrack_a_record()
    snapshot = dict(r)
    derive_pdf_text_layout_quality_context(r, enabled=True, env={})
    assert r == snapshot


def test_disabled_helper_does_not_mutate_record():
    r = _subtrack_a_record()
    snapshot = dict(r)
    derive_pdf_text_layout_quality_context(r, env={})
    assert r == snapshot


# ── On-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_17_PDF_TEXT_LAYOUT_QUALITY_IMPL.md",
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
    assert payload["phase_id"] == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17"
    assert payload["mode"] == "default_off_implementation"
    assert payload["default_off"] is True
    assert payload["env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert payload["total_records_in_scope"] == 21
    assert payload["subtrack_a_records"] == 11
    assert payload["subtrack_b_records"] == 10


def test_written_json_safety_invariants(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    for flag in [
        "default_behavior_changed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "runtime_behavior_changed_by_default",
        "extraction_behavior_changed_by_default",
        "pdf_text_extraction_behavior_changed_by_default",
        "layout_extraction_behavior_changed_by_default",
        "table_extraction_behavior_changed_by_default",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "cue_expansion_performed",
        "operator_ui_surface_added",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
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
    assert payload["behavior_changed"] is True  # new code exists


def test_written_json_default_off_proof(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    proof = payload["default_off_proof"]
    assert proof["default_off_emissions_must_be_zero"] is True
    assert proof["explicit_off_emissions_must_be_zero"] is True
    assert proof["explicit_on_target_emissions_must_equal_21"] is True
    assert proof["explicit_on_excluded_emissions_must_be_zero"] is True
    assert proof["review_required_all_true"] is True
    assert proof["auto_accept_allowed_any_true"] is False
    assert proof["clinical_interpretation_any_true"] is False
    assert proof["raw_text_any_emitted"] is False
    assert proof["raw_filename_any_emitted"] is False
    assert proof["private_path_any_emitted"] is False


# ── Runtime wiring guard: helper must NOT be auto-invoked ─────────────────


def test_helper_is_not_imported_by_app_main():
    app_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert "pdf_text_layout_quality_impl" not in body
    assert "derive_pdf_text_layout_quality_context" not in body


def test_helper_is_not_re_exported_from_document_type_init():
    init_path = (
        Path(__file__).resolve().parents[1]
        / "clinical_knowledge"
        / "document_type"
        / "__init__.py"
    )
    if not init_path.exists():
        pytest.skip("document_type/__init__.py absent")
    body = init_path.read_text(encoding="utf-8")
    # Intentional: DIAG-17 keeps blast radius minimal by not re-exporting
    # from the package __init__.py. The helper is reachable only via fully-
    # qualified import. If a future block wants to re-export, that's a
    # separate change.
    assert "pdf_text_layout_quality_impl" not in body, (
        "DIAG-17 must not be auto-imported via the package __init__.py"
    )
