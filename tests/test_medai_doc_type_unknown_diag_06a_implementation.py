"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION.

Covers:
    * The runtime helper ``derive_numeric_table_safe_default_label``.
    * The implementation-validation script's replay and audit logic.
    * Privacy invariants on the public implementation report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    DERIVED_LABEL,
    EXCLUSION_RULES,
    POSITIVE_SIGNATURE,
    derive_numeric_table_safe_default_label,
    is_disabled_by_default,
    matches_positive_signature,
    violates_any_exclusion_rule,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_06a_implementation import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixture: synthetic record that satisfies the 14 positive fields ─────────

def _matching_record(**overrides) -> dict:
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
        "administrative_form_shape_detected": "yes",
        "date_or_schedule_shape_detected": "yes",
        "imaging_modality_shape_detected": "no",
        "native_text_length_bucket": "medium",
        "alphabetic_content_bucket": "high",
        "numeric_content_bucket": "medium",
        "language_detector_attempted": "yes",
        "language_detector_input_bucket": "sufficient",
        "language_script_visibility": "latin_visible_language_unknown",
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
    assert is_disabled_by_default() is True
    r = _matching_record()
    # No `enabled` kwarg -> disabled by default -> None
    assert derive_numeric_table_safe_default_label(r) is None
    # Explicit False -> None (rollback path)
    assert derive_numeric_table_safe_default_label(r, enabled=False) is None


def test_exact_14_field_match_returns_label():
    r = _matching_record()
    assert derive_numeric_table_safe_default_label(r, enabled=True) == DERIVED_LABEL


def test_helper_does_not_mutate_record():
    r = _matching_record()
    snapshot = dict(r)
    derive_numeric_table_safe_default_label(r, enabled=True)
    assert r == snapshot


def test_derived_label_value_is_exact():
    assert DERIVED_LABEL == "latin_script_likely_english_table_context"


# ── Positive signature: every field must be present ─────────────────────────

@pytest.mark.parametrize("key,expected", POSITIVE_SIGNATURE)
def test_missing_or_wrong_positive_field_prevents_label(key, expected):
    # Map signature key to the raw record field the predicate consults.
    # If the predicate is multi-field, break the first source field to invalidate.
    raw_field_for_signature_key = {
        "table_like_structure_detected":            "table_like_structure_detected",
        "high_table_density_required":              "table_like_structure_detected",
        "repeated_row_pattern_visible":             "date_or_schedule_shape_detected",
        "numeric_content_bucket":                   "numeric_content_bucket",
        "numeric_units_or_ranges_visible":          "symbol_content_bucket",
        "script_detection_result":                  "script_detection_result",
        "dominant_script":                          "dominant_script",
        "detector_confidence_bucket":               "detector_confidence_bucket",
        "alphabetic_ratio_sufficient_for_language": "alphabetic_content_bucket",
        "administrative_table_shape":               "administrative_form_shape_detected",
        "treatment_schedule_table_shape":           "date_or_schedule_shape_detected",
        "language_detector_attempted":              "language_detector_attempted",
        "language_detector_input_bucket":           "language_detector_input_bucket",
        "language_visibility_status":               "language_visibility_status",
    }
    field = raw_field_for_signature_key[key]
    # Break the field by clearing it; matches_positive_signature should fail.
    r = _matching_record(**{field: ""})
    assert matches_positive_signature(r) is False
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


# ── Each exclusion rule must individually prevent the label ────────────────

@pytest.mark.parametrize("rule", EXCLUSION_RULES)
def test_each_exclusion_rule_prevents_label(rule):
    overrides_for_rule = {
        "exclude_cyrillic_dominant_records": {"dominant_script": "cyrillic"},
        "exclude_mixed_script_records": {"dominant_script": "mixed"},
        "exclude_low_alphabetic_ratio_records": {"alphabetic_content_bucket": "low"},
        "exclude_no_text_layer_records": {"pdf_text_layer_detected": "no"},
        "exclude_image_like_but_not_routed_records": {"image_like_pdf": "yes"},
        "exclude_ambiguous_below_threshold_records": {
            "unknown_failure_bucket": "ambiguous_below_threshold"
        },
        "exclude_fallback_ran_but_no_family_match_records": {
            "unknown_failure_bucket": "fallback_ran_but_no_family_match"
        },
        "exclude_medication_dose_or_ddi_interpretation": {
            "parsed_medications": ["metformin"]
        },
        "exclude_lab_value_parsing": {
            "parsed_lab_values": [{"name": "Hb", "value": "13.5"}]
        },
        "exclude_records_with_insufficient_safe_metadata": {
            "dominant_script": "",
        },
    }
    overrides = overrides_for_rule[rule]
    r = _matching_record(**overrides)
    assert violates_any_exclusion_rule(r) is True
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


# ── Implementation-level safeguards (cross-lever guards) ────────────────────

def test_section_heading_yes_prevents_label():
    """Records that DIAG-05 routes to the table-header lever must not get
    the numeric-table label, even when the spec's 14 fields all match."""
    r = _matching_record(section_heading_shape_detected="yes")
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_medical_abbreviation_yes_prevents_label():
    r = _matching_record(medical_abbreviation_shape_detected="yes")
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_text_layer_too_short_routing_prevents_label():
    r = _matching_record(
        unknown_ocr_routing_bucket="text_layer_present_but_too_short"
    )
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_non_unknown_predicted_type_prevents_label():
    r = _matching_record(predicted_document_type="Lab result")
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_other_failure_bucket_prevents_label():
    r = _matching_record(unknown_failure_bucket="ambiguous_below_threshold")
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


# ── Helper contract: no auto-accept / clinical / parsing side-effects ──────

def test_helper_does_not_introduce_auto_accept():
    r = _matching_record()
    # Caller-side: the label is metadata; auto_accept_allowed must remain
    # whatever the caller set, NOT True.
    out = derive_numeric_table_safe_default_label(r, enabled=True)
    assert out == DERIVED_LABEL
    assert "auto_accept_allowed" not in r or r["auto_accept_allowed"] is False


def test_helper_does_not_classify_clinical_meaning():
    r = _matching_record()
    out = derive_numeric_table_safe_default_label(r, enabled=True)
    # The returned label must not name a clinical concept; it is a routing
    # metadata token only.
    assert out == "latin_script_likely_english_table_context"
    forbidden_clinical_tokens = (
        "diagnosis", "diagnosed", "lab result", "medication",
        "dose", "treatment", "imaging finding", "ddi",
    )
    for tok in forbidden_clinical_tokens:
        assert tok not in out.lower()


def test_helper_does_not_parse_lab_values():
    r = _matching_record(parsed_lab_values=[{"name": "Hb", "value": "13.5"}])
    # Caller passed parsed labs -> exclusion fires -> None
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_helper_does_not_parse_medications_or_ddi():
    r = _matching_record(parsed_medications=["metformin"])
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None
    r = _matching_record(parsed_ddi_findings=["finding"])
    assert derive_numeric_table_safe_default_label(r, enabled=True) is None


def test_review_bound_status_is_preserved():
    r = _matching_record()
    snapshot = dict(r)
    derive_numeric_table_safe_default_label(r, enabled=True)
    assert r.get("review_status") == snapshot.get("review_status") == "review"


# ── Implementation-validation script (synthetic) ────────────────────────────

def _synthetic_per_file_table() -> list[dict]:
    rows: list[dict] = []
    # 4 records that pass the helper.
    for i in range(4):
        rows.append(_matching_record(file_id=f"file_{i + 1:03d}"))
    # Section heading -> excluded by implementation safeguard, AND DIAG-05
    # routes the record to the table-header lever so it's outside the
    # priority slice.
    rows.append(_matching_record(file_id="file_010",
                                 section_heading_shape_detected="yes"))
    # Medical abbreviation -> excluded by implementation safeguard, AND
    # DIAG-04 routes the record to the abbreviation lever so it's outside
    # the priority slice.
    rows.append(_matching_record(file_id="file_011",
                                 medical_abbreviation_shape_detected="yes"))
    # Non-Unknown predicted type -> dropped by DIAG-02 selection, also
    # blocked by the helper's "predicted_document_type==unknown" safeguard.
    rows.append(_matching_record(file_id="file_013",
                                 predicted_document_type="Lab result"))
    return rows


@pytest.fixture
def synthetic_payload() -> dict:
    return {"anonymous_per_file_table": _synthetic_per_file_table()}


def test_eleven_record_replay_summary(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    summary = report.eleven_record_replay
    # 4 matching synthetic records.
    assert summary["priority_slice_size"] == 4
    assert summary["enabled_labeled_count"] == 4
    assert summary["disabled_labeled_count"] == 0
    assert summary["matches_priority_slice_exactly"] is True


def test_aggregate_replay_no_false_positives(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    agg = report.five_hundred_seven_file_aggregate
    assert agg["disabled_labeled_count"] == 0
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True


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


def test_helper_default_disabled_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.helper_default_disabled is True


def test_unknown_count_impact_is_metadata_only(synthetic_payload):
    """The label is metadata-only; the data-layer unknown count does not move."""
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.unknown_count_impact_delta == 0
    assert report.unknown_count_before == report.unknown_count_after


# ── Report invariants ──────────────────────────────────────────────────────

def test_report_flags(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is True
    assert report.clinical_behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    sp = report.safety_privacy
    assert sp["behavior_changed_strictly_limited_to_safe_metadata_label"] is True
    assert sp["clinical_behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["helper_default_disabled"] is True
    assert sp["rollback_path_present"] is True


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
        "approximately 62%",
        "approximately 38%",
        "after_impl_unknown_track_done_pct",
        "approximately 68%",
        "approximately 32%",
        "before_impl_project_done_pct",
        "approximately 76%",
        "approximately 24%",
        "after_impl_project_done_pct",
        "approximately 77%",
        "approximately 23%",
    ):
        assert token in md


# ── Privacy invariants ──────────────────────────────────────────────────────

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


def test_safety_guard_allows_module_paths():
    """The guard must not false-positive on `clinical_knowledge.document_type`."""
    payload = {
        "impl": "Adds clinical_knowledge.document_type.derive_numeric_table_safe_default_label",
    }
    # No exception
    assert_safe_public_payload(payload)


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


# ── End-to-end on tmp dir ───────────────────────────────────────────────────

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
    assert SOURCE_REPORT.exists(), (
        f"Expected FAMILY-04 batch-eval public report at {SOURCE_REPORT}."
    )
