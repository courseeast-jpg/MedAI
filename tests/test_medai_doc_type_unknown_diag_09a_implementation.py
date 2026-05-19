"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    PROPAGATED_METADATA_DISCLAIMER,
    PROPAGATED_METADATA_LABEL,
    derive_language_propagation_metadata_label,
    derive_numeric_table_safe_default_label,
    is_language_propagation_metadata_default_disabled,
    is_language_propagation_metadata_enabled,
)
from clinical_knowledge.document_type.language_propagation_metadata import (
    EXCLUSION_RULES,
    POSITIVE_SIGNAL_PATTERN,
    matches_positive_signal_pattern,
    violates_any_exclusion_rule,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_09a_implementation import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixture: synthetic propagation-pool record ──────────────────────────────

def _matching_record(**overrides) -> dict:
    """Baseline record that satisfies all 11 positive-signal fields AND every
    implementation safeguard."""
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
        "medical_abbreviation_shape_detected": "no",
        "lab_table_shape_detected": "no",
        "administrative_form_shape_detected": "no",
        "date_or_schedule_shape_detected": "no",
        "imaging_modality_shape_detected": "no",
        "native_text_length_bucket": "medium",
        "alphabetic_content_bucket": "high",
        "numeric_content_bucket": "low",          # avoids table-heavy DIAG-03
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


# ── Helper contract ─────────────────────────────────────────────────────────

def test_helper_is_default_disabled():
    assert is_language_propagation_metadata_default_disabled() is True
    r = _matching_record()
    # No kwarg, empty env -> None
    assert derive_language_propagation_metadata_label(r, env={}) is None
    # Explicit False -> None
    assert derive_language_propagation_metadata_label(r, enabled=False) is None


def test_exact_11_field_match_returns_label_when_enabled():
    r = _matching_record()
    assert (derive_language_propagation_metadata_label(r, enabled=True)
            == PROPAGATED_METADATA_LABEL)


def test_helper_does_not_mutate_record():
    r = _matching_record()
    snapshot = dict(r)
    derive_language_propagation_metadata_label(r, enabled=True)
    assert r == snapshot


def test_propagated_label_value_is_exact():
    assert PROPAGATED_METADATA_LABEL == "latin_detector_likely_english_context"


def test_raw_detector_fields_unchanged_by_helper():
    """The helper must never touch raw detector fields."""
    r = _matching_record()
    snapshot_detector_fields = {
        k: r.get(k) for k in (
            "language_detector_attempted",
            "language_detector_input_bucket",
            "detector_confidence_bucket",
            "script_detection_result",
            "dominant_script",
            "language_visibility_status",
        )
    }
    derive_language_propagation_metadata_label(r, enabled=True)
    for k, v in snapshot_detector_fields.items():
        assert r.get(k) == v


# ── Flag separation ─────────────────────────────────────────────────────────

def test_propagation_env_var_enables_helper():
    r = _matching_record()
    badge = derive_language_propagation_metadata_label(
        r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"},
    )
    assert badge == PROPAGATED_METADATA_LABEL


def test_operator_review_env_var_does_not_enable_propagation_helper():
    """Setting MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED must NOT enable
    the propagation helper. The flags are deliberately separate."""
    r = _matching_record()
    assert derive_language_propagation_metadata_label(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"},
    ) is None


def test_propagation_env_var_is_distinct_from_operator_review_env_var():
    assert (LANGUAGE_PROPAGATION_METADATA_ENV_VAR
            != OPERATOR_REVIEW_BADGE_ENV_VAR)


def test_explicit_false_overrides_truthy_propagation_env():
    r = _matching_record()
    assert derive_language_propagation_metadata_label(
        r, enabled=False,
        env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"},
    ) is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_truthy_env_values_enable_helper(value):
    assert is_language_propagation_metadata_enabled(
        env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: value}
    ) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "disabled"])
def test_falsy_env_values_disable_helper(value):
    assert is_language_propagation_metadata_enabled(
        env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: value}
    ) is False


# ── Positive signal pattern: every field must hold ─────────────────────────

@pytest.mark.parametrize("key,expected", POSITIVE_SIGNAL_PATTERN)
def test_missing_or_wrong_positive_field_prevents_label(key, expected):
    raw_field_for_signature_key = {
        "detector_attempted":                  "language_detector_attempted",
        "detector_input_bucket":               "language_detector_input_bucket",
        "detector_confidence_bucket":          "detector_confidence_bucket",
        "script_detection_result":             "script_detection_result",
        "dominant_script":                     "dominant_script",
        "language_visibility_status":          "language_visibility_status",
        "detector_output_not_propagated":      "detector_confidence_bucket",
        "alphabetic_ratio_sufficient_for_language": "alphabetic_content_bucket",
        "no_cyrillic_dominant_signal":         "dominant_script",
        "no_mixed_script_signal":              "dominant_script",
        "no_low_confidence_detector_signal":   "detector_confidence_bucket",
    }
    field = raw_field_for_signature_key[key]
    # Break the underlying field; the predicate should fail.
    if key == "no_cyrillic_dominant_signal":
        r = _matching_record(dominant_script="cyrillic")
    elif key == "no_mixed_script_signal":
        r = _matching_record(dominant_script="mixed")
    elif key == "no_low_confidence_detector_signal":
        r = _matching_record(detector_confidence_bucket="low")
    else:
        r = _matching_record(**{field: ""})
    assert matches_positive_signal_pattern(r) is False
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


# ── Exclusion rules: each one prevents the label ───────────────────────────

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
            {  # build a numeric-table safe-default-shaped record
                "numeric_content_bucket": "medium",
                "administrative_form_shape_detected": "yes",
                "date_or_schedule_shape_detected": "yes",
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
    # Either the exclusion rule itself fires, OR the implementation
    # safeguard (e.g. for the numeric-table overlap rule) fires. Either way
    # the label must be suppressed.
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


# ── Implementation safeguards (cross-lever guards) ─────────────────────────

def test_non_unknown_predicted_type_prevents_label():
    r = _matching_record(predicted_document_type="Lab result")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_other_failure_bucket_prevents_label():
    r = _matching_record(unknown_failure_bucket="ambiguous_below_threshold")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_non_language_visibility_routing_prevents_label():
    r = _matching_record(unknown_ocr_routing_bucket="text_layer_present_but_too_short")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_medical_abbreviation_yes_prevents_label():
    r = _matching_record(medical_abbreviation_shape_detected="yes")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_table_heavy_diag03_sub_pool_prevents_label():
    """A record with table_like=yes AND numeric in {medium, high} belongs to
    DIAG-03's table-heavy sub-pool; it must not receive the propagation label."""
    r = _matching_record(numeric_content_bucket="medium")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None
    r = _matching_record(numeric_content_bucket="high")
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_numeric_table_safe_default_overlap_prevents_label():
    """If a record would receive the DIAG-06A numeric-table safe-default
    label, the propagation helper must not also label it."""
    r = _matching_record(
        numeric_content_bucket="medium",
        administrative_form_shape_detected="yes",
        date_or_schedule_shape_detected="yes",
    )
    assert (derive_numeric_table_safe_default_label(r, enabled=True)
            is not None)
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


# ── Helper contract: no auto-accept / clinical / parsing ───────────────────

def test_helper_does_not_introduce_auto_accept():
    r = _matching_record()
    label = derive_language_propagation_metadata_label(r, enabled=True)
    assert label == PROPAGATED_METADATA_LABEL
    # The record's auto_accept_allowed must remain False (or absent).
    assert "auto_accept_allowed" not in r or r["auto_accept_allowed"] is False


def test_helper_does_not_classify_clinical_meaning():
    r = _matching_record()
    label = derive_language_propagation_metadata_label(r, enabled=True)
    forbidden_clinical_tokens = (
        "diagnosis", "diagnosed", "lab result", "medication",
        "dose", "treatment", "imaging finding", "ddi",
    )
    for tok in forbidden_clinical_tokens:
        assert tok not in label.lower()


def test_helper_does_not_parse_lab_values():
    r = _matching_record(parsed_lab_values=[{"name": "Hb", "value": "13.5"}])
    assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_helper_does_not_parse_medications_or_ddi():
    for forbidden_payload in (
        {"parsed_medications": ["metformin"]},
        {"parsed_ddi_findings": ["finding"]},
        {"parsed_doses": ["500mg"]},
        {"parsed_frequencies": ["bid"]},
    ):
        r = _matching_record(**forbidden_payload)
        assert derive_language_propagation_metadata_label(r, enabled=True) is None


def test_review_bound_status_is_preserved():
    r = _matching_record()
    snapshot = dict(r)
    derive_language_propagation_metadata_label(r, enabled=True)
    assert r.get("review_status") == snapshot.get("review_status") == "review"


def test_data_layer_document_type_unchanged():
    """The helper does not change predicted_document_type."""
    r = _matching_record()
    snapshot_type = r.get("predicted_document_type")
    derive_language_propagation_metadata_label(r, enabled=True)
    assert r.get("predicted_document_type") == snapshot_type == "Unknown"


# ── Implementation-validation script (synthetic) ───────────────────────────

def _synthetic_per_file_table() -> list[dict]:
    rows: list[dict] = []
    # 4 propagation-pool candidates
    for i in range(4):
        rows.append(_matching_record(file_id=f"file_{i + 1:03d}"))
    # 1 numeric-table safe-default record (must be excluded)
    rows.append(_matching_record(
        file_id="file_010",
        numeric_content_bucket="medium",
        administrative_form_shape_detected="yes",
        date_or_schedule_shape_detected="yes",
    ))
    # 1 abbreviation-pool record (must be excluded)
    rows.append(_matching_record(file_id="file_011",
                                 medical_abbreviation_shape_detected="yes"))
    # 1 non-Unknown record (must be excluded)
    rows.append(_matching_record(file_id="file_013",
                                 predicted_document_type="Lab result"))
    return rows


@pytest.fixture
def synthetic_payload() -> dict:
    return {"anonymous_per_file_table": _synthetic_per_file_table()}


def test_eleven_record_replay_summary(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    summary = report.eleven_record_replay
    assert summary["priority_slice_size"] == 4
    assert summary["enabled_labeled_count"] == 4
    assert summary["disabled_labeled_count"] == 0
    assert summary["default_off_labeled_count"] == 0
    assert summary["matches_priority_slice_exactly"] is True


def test_aggregate_replay_no_false_positives(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    agg = report.five_hundred_seven_file_aggregate
    assert agg["default_off_labeled_count"] == 0
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True
    assert agg["flag_separation_holds"] is True
    assert agg["operator_review_env_only_labeled_count"] == 0


def test_no_overlap_with_numeric_table_safe_default_pool(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.overlap_with_numeric_table_safe_default_pool == 0
    assert report.no_overlap_with_numeric_table_safe_default_pool is True


def test_false_positive_audit_is_clean(synthetic_payload):
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
    sp = report.safety_privacy
    assert (
        sp["behavior_changed_strictly_limited_to_safe_metadata_propagation"] is True
    )
    assert sp["raw_detector_output_unchanged"] is True
    assert sp["clinical_behavior_changed"] is False
    assert sp["data_layer_document_type_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["helper_default_disabled"] is True
    assert sp["rollback_path_present"] is True
    assert (
        sp["propagation_env_var_separate_from_operator_badge_env_var"] is True
    )
    assert sp["no_overlap_with_numeric_table_safe_default_pool"] is True


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
        "approximately 80%",
        "approximately 20%",
        "after_impl_unknown_track_done_pct",
        "approximately 84%",
        "approximately 16%",
        "after_impl_project_done_pct",
        "approximately 81%",
        "approximately 19%",
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


# ── End-to-end ─────────────────────────────────────────────────────────────

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


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()
