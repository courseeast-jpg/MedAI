"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_METADATA_DISCLAIMER,
    LATIN_ABBREVIATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_METADATA_LABEL,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    derive_language_propagation_metadata_label,
    derive_latin_medical_abbreviation_metadata_label,
    derive_numeric_table_safe_default_label,
    is_latin_abbreviation_metadata_default_disabled,
    is_latin_abbreviation_metadata_enabled,
)
from clinical_knowledge.document_type.latin_abbreviation_metadata import (
    EXCLUSION_RULES,
    POSITIVE_SIGNAL_PATTERN,
    matches_positive_signal_pattern,
    violates_any_exclusion_rule,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_11a_implementation import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixture ────────────────────────────────────────────────────────────────

def _matching_record(**overrides) -> dict:
    """Baseline abbreviation-pool record."""
    base = {
        "predicted_document_type": "Unknown",
        "unknown_failure_bucket": "insufficient_text_visibility",
        "unknown_ocr_routing_bucket": "language_visibility_unknown",
        "language_visibility_status": "latin_visible_language_unknown",
        "dominant_script": "latin",
        "pdf_text_layer_detected": "yes",
        "image_like_pdf": "no",
        "ocr_fallback_eligible": "no",
        "ocr_fallback_not_triggered_reason": "language_visibility_unknown",
        "table_like_structure_detected": "yes",
        "section_heading_shape_detected": "no",
        "medical_abbreviation_shape_detected": "yes",
        "lab_table_shape_detected": "no",
        "administrative_form_shape_detected": "no",
        "date_or_schedule_shape_detected": "no",
        "imaging_modality_shape_detected": "no",
        "native_text_length_bucket": "medium",
        "alphabetic_content_bucket": "high",
        "numeric_content_bucket": "low",
        "language_detector_attempted": "yes",
        "language_detector_input_bucket": "sufficient",
        "language_script_visibility": "latin_visible_language_unknown",
        "language_script_detector_unknown_bucket":
            "script_detectable_language_unknown",
        "script_detection_attempted": "yes",
        "script_detection_result": "latin",
        "symbol_content_bucket": "medium",
        "detector_confidence_bucket": "high",
        "cyrillic_visibility_status": "unknown",
        "review_status": "review",
    }
    base.update(overrides)
    return base


# ── Helper contract ────────────────────────────────────────────────────────

def test_helper_is_default_disabled():
    assert is_latin_abbreviation_metadata_default_disabled() is True
    r = _matching_record()
    assert derive_latin_medical_abbreviation_metadata_label(r, env={}) is None
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=False) is None


def test_exact_14_field_match_returns_label_when_enabled():
    r = _matching_record()
    assert (derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
            == LATIN_ABBREVIATION_METADATA_LABEL)


def test_helper_does_not_mutate_record():
    r = _matching_record()
    snapshot = dict(r)
    derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
    assert r == snapshot


def test_label_value_is_exact():
    assert LATIN_ABBREVIATION_METADATA_LABEL == "latin_medical_abbreviation_context"


def test_raw_detector_fields_unchanged():
    r = _matching_record()
    snap = {k: r.get(k) for k in (
        "language_detector_attempted",
        "language_detector_input_bucket",
        "detector_confidence_bucket",
        "script_detection_result",
        "dominant_script",
        "language_visibility_status",
    )}
    derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
    for k, v in snap.items():
        assert r.get(k) == v


def test_data_layer_document_type_unchanged():
    r = _matching_record()
    snap = r.get("predicted_document_type")
    derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
    assert r.get("predicted_document_type") == snap == "Unknown"


# ── Three-way flag separation ──────────────────────────────────────────────

def test_abbrev_env_enables_helper():
    r = _matching_record()
    assert (derive_latin_medical_abbreviation_metadata_label(
        r, env={LATIN_ABBREVIATION_METADATA_ENV_VAR: "1"}
    ) == LATIN_ABBREVIATION_METADATA_LABEL)


def test_operator_review_env_does_not_enable_abbreviation_helper():
    """Setting MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED must NOT enable
    the abbreviation helper."""
    r = _matching_record()
    assert derive_latin_medical_abbreviation_metadata_label(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    ) is None


def test_language_propagation_env_does_not_enable_abbreviation_helper():
    """Setting MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED must
    NOT enable the abbreviation helper."""
    r = _matching_record()
    assert derive_latin_medical_abbreviation_metadata_label(
        r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    ) is None


def test_three_env_vars_are_distinct():
    assert len({
        LATIN_ABBREVIATION_METADATA_ENV_VAR,
        LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
        OPERATOR_REVIEW_BADGE_ENV_VAR,
    }) == 3


def test_explicit_false_overrides_truthy_env():
    r = _matching_record()
    assert derive_latin_medical_abbreviation_metadata_label(
        r, enabled=False,
        env={LATIN_ABBREVIATION_METADATA_ENV_VAR: "1"},
    ) is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_truthy_env_values_enable_helper(value):
    assert is_latin_abbreviation_metadata_enabled(
        env={LATIN_ABBREVIATION_METADATA_ENV_VAR: value}
    ) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "disabled"])
def test_falsy_env_values_disable_helper(value):
    assert is_latin_abbreviation_metadata_enabled(
        env={LATIN_ABBREVIATION_METADATA_ENV_VAR: value}
    ) is False


def test_all_three_env_vars_together_each_label_their_own_slice():
    """When all three env vars are truthy, the abbreviation helper labels
    only its own slice (regardless of the others' env state)."""
    r = _matching_record()
    env = {
        LATIN_ABBREVIATION_METADATA_ENV_VAR: "1",
        LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1",
        OPERATOR_REVIEW_BADGE_ENV_VAR: "1",
    }
    assert (derive_latin_medical_abbreviation_metadata_label(r, env=env)
            == LATIN_ABBREVIATION_METADATA_LABEL)
    # The propagation helper must not pick this record up (the abbreviation
    # exclusion in the propagation helper would block it; in our fixture
    # med_abbrev=yes, so propagation excludes it via the abbreviation
    # safeguard).
    assert derive_language_propagation_metadata_label(r, env=env) is None
    # Numeric-table helper has no env arg; bare call with enabled=True
    # should not label this record either.
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


# ── Positive signal pattern ───────────────────────────────────────────────

@pytest.mark.parametrize("key,expected", POSITIVE_SIGNAL_PATTERN)
def test_missing_or_wrong_positive_field_prevents_label(key, expected):
    raw_field_for_signature_key = {
        "detector_attempted":                  "language_detector_attempted",
        "detector_input_bucket":               "language_detector_input_bucket",
        "detector_confidence_bucket":          "detector_confidence_bucket",
        "script_detection_result":             "script_detection_result",
        "dominant_script":                     "dominant_script",
        "language_visibility_status":          "language_visibility_status",
        "latin_medical_abbrev_visible":        "medical_abbreviation_shape_detected",
        "medical_abbreviation_shape_detected": "medical_abbreviation_shape_detected",
        "alphabetic_ratio_sufficient_for_language": "alphabetic_content_bucket",
        "no_cyrillic_dominant_signal":         "dominant_script",
        "no_mixed_script_signal":              "dominant_script",
        "no_low_confidence_detector_signal":   "detector_confidence_bucket",
        "not_already_handled_by_numeric_table_safe_default_helper":
            "administrative_form_shape_detected",  # see special-case below
        "not_already_handled_by_language_propagation_helper":
            "language_detector_attempted",  # see special-case below
    }
    if key == "no_cyrillic_dominant_signal":
        r = _matching_record(dominant_script="cyrillic")
    elif key == "no_mixed_script_signal":
        r = _matching_record(dominant_script="mixed")
    elif key == "no_low_confidence_detector_signal":
        r = _matching_record(detector_confidence_bucket="low")
    elif key == "not_already_handled_by_numeric_table_safe_default_helper":
        # Build a record that ALSO matches the DIAG-06A numeric-table pool.
        # The abbreviation helper must refuse to label it. We need to also
        # clear medical_abbreviation_shape_detected so the numeric-table
        # helper's "no medical abbreviation" safeguard does not exclude it.
        r = _matching_record(
            medical_abbreviation_shape_detected="no",
            numeric_content_bucket="medium",
            administrative_form_shape_detected="yes",
            date_or_schedule_shape_detected="yes",
        )
    elif key == "not_already_handled_by_language_propagation_helper":
        # Build a record that matches the DIAG-09A propagation pool but
        # NOT the abbreviation pool (so med_abbrev=no).
        r = _matching_record(
            medical_abbreviation_shape_detected="no",
        )
    else:
        field = raw_field_for_signature_key[key]
        r = _matching_record(**{field: ""})

    assert matches_positive_signal_pattern(r) is False
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


# ── Exclusion rules ───────────────────────────────────────────────────────

@pytest.mark.parametrize("rule", EXCLUSION_RULES)
def test_each_exclusion_rule_prevents_label(rule):
    overrides_for_rule = {
        "exclude_cyrillic_dominant_records": {"dominant_script": "cyrillic"},
        "exclude_mixed_script_records": {"dominant_script": "mixed"},
        "exclude_low_detector_confidence_records":
            {"detector_confidence_bucket": "low"},
        "exclude_insufficient_detector_input_records":
            {"language_detector_input_bucket": "insufficient"},
        "exclude_no_text_layer_records": {"pdf_text_layer_detected": "no"},
        "exclude_image_like_but_not_routed_records": {"image_like_pdf": "yes"},
        "exclude_table_heavy_numeric_safe_default_records_already_handled":
            {
                "medical_abbreviation_shape_detected": "no",
                "numeric_content_bucket": "medium",
                "administrative_form_shape_detected": "yes",
                "date_or_schedule_shape_detected": "yes",
            },
        "exclude_language_propagation_records_already_handled":
            {"medical_abbreviation_shape_detected": "no"},
        "exclude_table_header_only_special_case_record":
            {
                "medical_abbreviation_shape_detected": "no",
                "section_heading_shape_detected": "yes",
            },
        "exclude_ambiguous_below_threshold_records":
            {"unknown_failure_bucket": "ambiguous_below_threshold"},
        "exclude_fallback_ran_but_no_family_match_records":
            {"unknown_failure_bucket": "fallback_ran_but_no_family_match"},
        "exclude_medication_dose_or_ddi_interpretation":
            {"parsed_medications": ["metformin"]},
        "exclude_lab_value_parsing":
            {"parsed_lab_values": [{"name": "Hb", "value": "13.5"}]},
        "exclude_records_with_insufficient_safe_metadata":
            {"dominant_script": ""},
    }
    r = _matching_record(**overrides_for_rule[rule])
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


# ── Implementation safeguards ──────────────────────────────────────────────

def test_non_unknown_predicted_type_prevents_label():
    r = _matching_record(predicted_document_type="Lab result")
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


def test_other_failure_bucket_prevents_label():
    r = _matching_record(unknown_failure_bucket="ambiguous_below_threshold")
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


def test_non_language_visibility_routing_prevents_label():
    r = _matching_record(
        unknown_ocr_routing_bucket="text_layer_present_but_too_short"
    )
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


def test_medical_abbreviation_no_prevents_label():
    """Without med_abbrev=yes, no label."""
    r = _matching_record(medical_abbreviation_shape_detected="no")
    assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


# ── Cross-helper overlap invariants ───────────────────────────────────────

def test_numeric_table_safe_default_record_does_not_receive_abbreviation_label():
    """A record that DIAG-06A labels must NOT also receive the abbreviation
    label."""
    nt_record = _matching_record(
        medical_abbreviation_shape_detected="no",
        numeric_content_bucket="medium",
        administrative_form_shape_detected="yes",
        date_or_schedule_shape_detected="yes",
    )
    assert derive_numeric_table_safe_default_label(nt_record, enabled=True) is not None
    assert derive_latin_medical_abbreviation_metadata_label(nt_record, enabled=True) is None


def test_language_propagation_record_does_not_receive_abbreviation_label():
    """A record that DIAG-09A labels must NOT also receive the abbreviation
    label."""
    prop_record = _matching_record(medical_abbreviation_shape_detected="no")
    assert derive_language_propagation_metadata_label(prop_record, enabled=True) is not None
    assert derive_latin_medical_abbreviation_metadata_label(prop_record, enabled=True) is None


# ── Helper contract: no auto-accept / clinical / parsing ──────────────────

def test_helper_does_not_introduce_auto_accept():
    r = _matching_record()
    label = derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
    assert label == LATIN_ABBREVIATION_METADATA_LABEL
    assert "auto_accept_allowed" not in r or r["auto_accept_allowed"] is False


def test_helper_does_not_classify_clinical_meaning():
    label = derive_latin_medical_abbreviation_metadata_label(_matching_record(), enabled=True)
    forbidden = (
        "diagnosis", "diagnosed", "lab result", "medication",
        "dose", "treatment", "imaging finding", "ddi",
    )
    for tok in forbidden:
        assert tok not in label.lower()


def test_helper_does_not_parse_or_expand_abbreviation():
    """The label must not include the parsed or expanded form of any
    abbreviation. It must be a single controlled-vocabulary token."""
    label = derive_latin_medical_abbreviation_metadata_label(_matching_record(), enabled=True)
    # The label is a single snake_case token, never a parsed value.
    assert label == "latin_medical_abbreviation_context"
    # The label must not contain a parsed dose, value, or expanded form.
    assert "mg" not in label and "mcg" not in label and "ml" not in label
    assert " = " not in label and "::" not in label


def test_helper_does_not_parse_labs_or_medications():
    for forbidden_payload in (
        {"parsed_lab_values": [{"name": "Hb"}]},
        {"parsed_medications": ["metformin"]},
        {"parsed_ddi_findings": ["x"]},
        {"parsed_doses": ["500mg"]},
        {"parsed_frequencies": ["bid"]},
    ):
        r = _matching_record(**forbidden_payload)
        assert derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is None


def test_review_bound_status_preserved():
    r = _matching_record()
    snap = dict(r)
    derive_latin_medical_abbreviation_metadata_label(r, enabled=True)
    assert r.get("review_status") == snap.get("review_status") == "review"


# ── Audit-script outputs (synthetic) ──────────────────────────────────────

def _synthetic_per_file_table() -> list[dict]:
    rows: list[dict] = []
    # 4 abbreviation-pool records
    for i in range(4):
        rows.append(_matching_record(file_id=f"file_{i + 1:03d}"))
    # 1 numeric-table record (different lever)
    rows.append(_matching_record(
        file_id="file_010",
        medical_abbreviation_shape_detected="no",
        numeric_content_bucket="medium",
        administrative_form_shape_detected="yes",
        date_or_schedule_shape_detected="yes",
    ))
    # 1 propagation record (different lever)
    rows.append(_matching_record(file_id="file_011",
                                 medical_abbreviation_shape_detected="no"))
    # Non-Unknown
    rows.append(_matching_record(file_id="file_013",
                                 predicted_document_type="Lab result"))
    return rows


@pytest.fixture
def synthetic_payload() -> dict:
    return {"anonymous_per_file_table": _synthetic_per_file_table()}


def test_eight_record_replay_summary(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    rep = report.eight_record_replay
    assert rep["priority_slice_size"] == 4
    assert rep["enabled_labeled_count"] == 4
    assert rep["disabled_labeled_count"] == 0
    assert rep["default_off_labeled_count"] == 0
    assert rep["matches_priority_slice_exactly"] is True


def test_aggregate_no_false_positives(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    agg = report.five_hundred_seven_file_aggregate
    assert agg["default_off_labeled_count"] == 0
    assert agg["abbreviation_env_only_labeled_count"] == 4
    assert agg["propagation_env_only_labeled_count"] == 0
    assert agg["operator_review_env_only_labeled_count"] == 0
    assert agg["all_three_env_vars_labeled_count"] == 4
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True


def test_three_way_flag_separation_holds(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    audit = report.flag_separation_audit
    assert audit["default_off_yields_zero_abbreviation_labels"] is True
    assert audit["abbrev_env_only_yields_priority_slice"] is True
    assert audit["propagation_env_only_yields_zero_abbreviation_labels"] is True
    assert audit["operator_review_env_only_yields_zero_abbreviation_labels"] is True
    assert audit["all_three_env_vars_yields_priority_slice_for_abbreviation"] is True
    assert audit["all_three_env_vars_are_distinct"] is True
    assert audit["three_way_flag_separation_holds"] is True


def test_no_overlap_with_existing_helpers(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.overlap_with_numeric_table_safe_default_pool == 0
    assert report.no_overlap_with_numeric_table_safe_default_pool is True
    assert report.overlap_with_language_propagation_pool == 0
    assert report.no_overlap_with_language_propagation_pool is True


def test_false_positive_audit_clean(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for k, v in report.false_positive_audit.items():
        assert v == 0, f"false-positive expansion in {k}: {v}"
    assert report.no_false_positive_expansion is True


def test_review_bound_preserved(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.review_bound_preserved is True


def test_counts_remain_zero(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.accepted_count == 0
    assert report.auto_accept_allowed_count == 0
    assert report.external_api_used_count == 0


def test_raw_detector_and_doc_type_unchanged(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.raw_detector_output_unchanged is True
    assert report.data_layer_document_type_unchanged is True


def test_unknown_count_impact_is_metadata_only(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.unknown_count_impact_delta == 0
    assert report.unknown_count_before == report.unknown_count_after


def test_helper_default_disabled_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.helper_default_disabled is True


# ── Report invariants ──────────────────────────────────────────────────────

def test_report_flags(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is True
    assert report.clinical_behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    assert report.abbreviation_parsing_or_expansion is False
    sp = report.safety_privacy
    assert sp["behavior_changed_strictly_limited_to_safe_abbreviation_metadata"] is True
    assert sp["raw_detector_output_unchanged"] is True
    assert sp["data_layer_document_type_unchanged"] is True
    assert sp["clinical_behavior_changed"] is False
    assert sp["abbreviation_parsing_or_expansion_added"] is False
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["helper_default_disabled"] is True
    assert sp["rollback_path_present"] is True
    assert sp["abbreviation_env_var_distinct_from_propagation_env_var"] is True
    assert sp["abbreviation_env_var_distinct_from_operator_review_env_var"] is True
    assert sp["three_way_flag_separation_holds"] is True
    assert sp["no_overlap_with_numeric_table_safe_default_pool"] is True
    assert sp["no_overlap_with_language_propagation_pool"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_impl_unknown_track_done_pct",
        "approximately 90%",
        "approximately 10%",
        "after_impl_unknown_track_done_pct",
        "approximately 93%",
        "approximately 7%",
        "after_impl_project_done_pct",
        "approximately 84%",
        "approximately 16%",
    ):
        assert token in md


# ── Privacy invariants ─────────────────────────────────────────────────────

_RAW_FILENAME_RE = re.compile(r"\.(?:pdf|docx?|xlsx?|png|jpe?g)\b", re.IGNORECASE)
_PRIVATE_PATH_RE = re.compile(r"(?:/home/|/users/|c:\\)", re.IGNORECASE)


def _all_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _all_strings(k)
            yield from _all_strings(v)
    elif isinstance(node, (list, tuple, set)):
        for item in node:
            yield from _all_strings(item)


def test_report_emits_no_raw_filenames(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    payload = json.loads(json.dumps(report, default=lambda x: x.__dict__))
    for s in _all_strings(payload):
        assert not _RAW_FILENAME_RE.search(s), f"raw filename leaked: {s!r}"


def test_report_emits_no_private_paths(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    payload = json.loads(json.dumps(report, default=lambda x: x.__dict__))
    for s in _all_strings(payload):
        assert not _PRIVATE_PATH_RE.search(s), f"private path leaked: {s!r}"


def test_rendered_markdowns_emit_no_raw_filenames_or_paths(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for md in (render_markdown_summary(report), render_markdown_long(report)):
        assert not _RAW_FILENAME_RE.search(md)
        assert not _PRIVATE_PATH_RE.search(md)


def test_safety_guard_blocks_raw_filename():
    with pytest.raises(RuntimeError):
        assert_safe_public_payload({"doc": "the file lab_results.pdf was opened"})


def test_safety_guard_blocks_private_path():
    with pytest.raises(RuntimeError):
        assert_safe_public_payload({"path": "/home/user/private/x.pdf"})


def test_safety_guard_blocks_explicit_secret_pattern():
    with pytest.raises(RuntimeError):
        assert_safe_public_payload({"creds": "password=hunter2hunter2"})


def test_check_public_report_payload_passes_on_synthetic_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    json_payload = json.loads(json.dumps(report, default=lambda x: x.__dict__))
    summary = render_markdown_summary(report)
    long_md = render_markdown_long(report)
    for payload in (json_payload, summary, long_md):
        result = check_public_report_payload(payload)
        assert result.passed, (
            f"privacy check failed: phi={result.raw_phi_logged_in_public_reports}, "
            f"paths={result.private_filename_path_leaks}, "
            f"secrets={result.secret_leaks}, "
            f"examples={result.leak_examples_redacted}"
        )


# ── End-to-end ────────────────────────────────────────────────────────────

def test_write_reports_to_tmp(tmp_path: Path, synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    paths = write_reports(report, out_dir=tmp_path)
    assert set(paths.keys()) == {"json", "md_summary", "md_main"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    json_doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert json_doc["behavior_changed"] is True
    assert json_doc["clinical_behavior_changed"] is False
    assert json_doc["external_api_used"] is False
    assert json_doc["cue_expansion_recommended"] is False
    assert json_doc["abbreviation_parsing_or_expansion"] is False


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()


def test_real_corpus_eight_records_with_zero_overlap():
    """Sanity check on the real FAMILY-04 public report."""
    data = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    from scripts.run_medai_doc_type_unknown_diag_11a import select_abbreviation_pool
    pool = select_abbreviation_pool(data["anonymous_per_file_table"])
    assert len(pool) == 8
    nt_overlap = sum(
        1 for r in pool
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    prop_overlap = sum(
        1 for r in pool
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    )
    assert nt_overlap == 0
    assert prop_overlap == 0
    enabled_count = sum(
        1 for r in pool
        if derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is not None
    )
    assert enabled_count == 8


def test_disclaimer_marks_metadata_only():
    assert "metadata only" in LATIN_ABBREVIATION_METADATA_DISCLAIMER.lower()
    assert "not clinical interpretation" in LATIN_ABBREVIATION_METADATA_DISCLAIMER.lower()
    assert "raw detector output unchanged" in LATIN_ABBREVIATION_METADATA_DISCLAIMER.lower()
