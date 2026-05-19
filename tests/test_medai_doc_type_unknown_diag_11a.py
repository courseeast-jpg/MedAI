"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    derive_language_propagation_metadata_label,
    derive_numeric_table_safe_default_label,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_11a import (
    EXCLUSION_RULES,
    FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA,
    FUTURE_VALIDATION_REQUIREMENTS,
    POSITIVE_SIGNAL_PATTERN,
    PROPOSED_FUTURE_BEHAVIOR,
    PROPOSED_LABEL,
    SOURCE_REPORT,
    SUGGESTED_FUTURE_ENV_VAR,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    select_abbreviation_pool,
    write_reports,
)


# ── Synthetic fixture ──────────────────────────────────────────────────────

def _abbreviation_record(**overrides) -> dict:
    """Baseline record that matches the abbreviation-pool signature.
    Table_like=yes + numeric=low + alphabetic=high makes DIAG-03 classify
    this in the latin sub-pool (not table-heavy). med_abbrev=yes triggers
    DIAG-04's latin abbreviation lever first."""
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


def _numeric_table_record(**overrides) -> dict:
    """Record that DIAG-06A labels (numeric-table safe-default)."""
    base = _abbreviation_record()
    base.update({
        "medical_abbreviation_shape_detected": "no",  # not abbreviation pool
        "numeric_content_bucket": "medium",
        "administrative_form_shape_detected": "yes",
        "date_or_schedule_shape_detected": "yes",
    })
    base.update(overrides)
    return base


def _propagation_record(**overrides) -> dict:
    """Record that DIAG-09A-IMPL labels (language propagation)."""
    base = _abbreviation_record()
    base.update({
        "medical_abbreviation_shape_detected": "no",  # not abbreviation pool
        "numeric_content_bucket": "low",
        "administrative_form_shape_detected": "no",
        "date_or_schedule_shape_detected": "no",
    })
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_payload() -> dict:
    rows: list[dict] = []
    # 4 abbreviation-pool records
    for i in range(4):
        rows.append(_abbreviation_record(file_id=f"file_{i + 1:03d}"))
    # 1 numeric-table record (handled by DIAG-06A)
    rows.append(_numeric_table_record(file_id="file_010"))
    # 1 propagation record (handled by DIAG-09A)
    rows.append(_propagation_record(file_id="file_011"))
    # 1 table-header special case (deferred)
    rows.append(_propagation_record(file_id="file_012",
                                    section_heading_shape_detected="yes"))
    # Out-of-scope rows
    rows.append(_abbreviation_record(file_id="file_020",
                                     unknown_failure_bucket="fallback_ran_but_no_family_match"))
    rows.append(_abbreviation_record(file_id="file_021",
                                     unknown_failure_bucket="ambiguous_below_threshold"))
    rows.append(_abbreviation_record(file_id="file_022",
                                     unknown_ocr_routing_bucket="text_layer_present_but_too_short",
                                     native_text_length_bucket="none"))
    rows.append({"file_id": "file_030",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})
    return {"anonymous_per_file_table": rows}


# ── Slice selection ───────────────────────────────────────────────────────

def test_only_abbreviation_pool_records_are_targeted(synthetic_payload):
    pool = select_abbreviation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert ids == {"file_001", "file_002", "file_003", "file_004"}


def test_numeric_table_records_excluded(synthetic_payload):
    pool = select_abbreviation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert "file_010" not in ids


def test_language_propagation_records_excluded(synthetic_payload):
    pool = select_abbreviation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert "file_011" not in ids


def test_table_header_special_case_deferred(synthetic_payload):
    pool = select_abbreviation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert "file_012" not in ids


def test_other_failure_buckets_excluded(synthetic_payload):
    pool = select_abbreviation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    for excluded in ("file_020", "file_021", "file_022", "file_030"):
        assert excluded not in ids


def test_no_overlap_with_numeric_table_pool(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.overlap_with_numeric_table_safe_default_pool == 0
    assert report.no_overlap_with_numeric_table_safe_default_pool is True


def test_no_overlap_with_language_propagation_pool(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.overlap_with_language_propagation_pool == 0
    assert report.no_overlap_with_language_propagation_pool is True


def test_deferred_subsets_listed(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for key in (
        "numeric_table_safe_default_pool_already_handled",
        "language_propagation_pool_already_handled",
        "candidate_table_header_language_policy_record",
        "likely_text_layer_issue",
        "fallback_ran_but_no_family_match",
        "ambiguous_below_threshold",
    ):
        assert key in report.deferred_subsets


# ── Positive signal pattern ──────────────────────────────────────────────

def test_positive_signal_pattern_keys_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    rendered = {item["key"] for item in report.positive_signal_pattern}
    expected = {k for k, _ in POSITIVE_SIGNAL_PATTERN}
    assert rendered == expected


def test_positive_signal_holds_on_synthetic_slice(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.positive_signal_holds_on_all_priority_records is True
    for info in report.positive_signal_match_report.values():
        assert info["fully_matches"] is True


# ── Exclusion rules ──────────────────────────────────────────────────────

def test_exclusion_rules_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert set(report.exclusion_rules) == set(EXCLUSION_RULES)


def test_exclusion_audit_clean(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.no_priority_record_violates_any_exclusion_rule is True
    for row in report.exclusion_audit:
        assert row["violating_record_count"] == 0


# ── Proposed future behavior ─────────────────────────────────────────────

def test_proposed_future_behavior_contract(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    d = report.proposed_future_behavior
    # Every must_not_/must_be_/must_keep_ guard must be True.
    for k in (
        "must_not_classify_clinical_meaning",
        "must_not_parse_the_abbreviation",
        "must_not_expand_the_abbreviation",
        "must_not_parse_values",
        "must_not_auto_accept",
        "must_not_write_active_clinical_facts",
        "must_keep_document_review_bound",
        "must_be_default_off_behind_separate_env_flag_or_operator_setting",
        "must_not_overlap_with_numeric_table_safe_default_pool",
        "must_not_overlap_with_language_propagation_pool",
        "must_not_alter_raw_detector_output",
        "must_not_change_data_layer_document_type",
    ):
        assert d[k] is True
    assert d["proposed_label"] == PROPOSED_LABEL
    assert d["suggested_future_env_var"] == SUGGESTED_FUTURE_ENV_VAR
    assert d is PROPOSED_FUTURE_BEHAVIOR or d == PROPOSED_FUTURE_BEHAVIOR


def test_proposed_label_value():
    assert PROPOSED_LABEL == "latin_medical_abbreviation_context"


# ── Acceptance criteria + validation requirements ────────────────────────

def test_acceptance_criteria_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_implementation_acceptance_criteria)
            == set(FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA))


def test_validation_requirements_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_validation_requirements)
            == set(FUTURE_VALIDATION_REQUIREMENTS))


# ── Progress estimate ────────────────────────────────────────────────────

def test_progress_estimate_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    pe = report.progress_estimate
    for k in (
        "before_11a_unknown_track_done_pct",
        "before_11a_unknown_track_remaining_pct",
        "before_11a_project_done_pct",
        "before_11a_project_remaining_pct",
        "after_11a_unknown_track_done_pct",
        "after_11a_unknown_track_remaining_pct",
        "after_11a_project_done_pct",
        "after_11a_project_remaining_pct",
        "note",
    ):
        assert k in pe and pe[k]


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_11a_unknown_track_done_pct",
        "approximately 87%",
        "approximately 13%",
        "after_11a_unknown_track_done_pct",
        "approximately 90%",
        "approximately 10%",
        "before_11a_project_done_pct",
        "approximately 82%",
        "approximately 18%",
        "after_11a_project_done_pct",
        "approximately 83%",
        "approximately 17%",
    ):
        assert token in md


# ── Implementation-status invariants ─────────────────────────────────────

def test_no_implementation_marked(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    assert report.abbreviation_handling_implemented_in_this_block is False
    sp = report.safety_privacy
    assert sp["behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["raw_language_detector_behavior_changed"] is False
    assert sp["abbreviation_handling_behavior_changed"] is False
    assert sp["classifier_behavior_changed"] is False
    assert sp["cue_packs_changed"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["abbreviation_handling_implemented_in_this_block"] is False
    assert sp["abbreviation_parsing_or_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["no_overlap_with_numeric_table_safe_default_pool"] is True
    assert sp["no_overlap_with_language_propagation_pool"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


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


def test_no_forbidden_recommendation_in_rendered_output(synthetic_payload):
    """The report must never AFFIRMATIVELY recommend forbidden actions.

    The proposed_future_behavior dict legitimately mentions
    ``must_not_<action>`` flag keys (rendered as ``must_not_<action>: True``);
    those are explicit negations of the same actions and are fine. The
    check below filters out lines that lead with ``- must_not_`` or similar
    negation contexts before pattern-matching, so affirmative
    recommendations are still caught.
    """
    report = build_diagnostic_from_report(synthetic_payload)
    forbidden_phrases = (
        "recommend cue expansion",
        "add new cue",
        "expand cue pack",
        "expand the cue",
        "change ocr routing",
        "modify ocr routing",
        "change the ocr engine",
        "parse lab values",
        "parse medications",
        "implements the abbreviation",
        "we implement",
    )
    affirmative_abbrev_phrases = (
        # Lines that AFFIRMATIVELY tell the reader to expand / parse the
        # abbreviation. The negated forms ("must not expand the
        # abbreviation", "must_not_expand_the_abbreviation") are fine.
        re.compile(r"\brecommend (?:expanding|parsing) the abbreviation\b", re.I),
        re.compile(r"\bautomatically (?:expand|parse) the abbreviation\b", re.I),
        re.compile(r"\bshould expand the abbreviation\b", re.I),
        re.compile(r"\bshould parse the abbreviation\b", re.I),
    )
    for md in (render_markdown_summary(report), render_markdown_long(report)):
        low = md.lower()
        for forbidden in forbidden_phrases:
            assert forbidden not in low, (
                f"forbidden recommendation present in rendered output: "
                f"{forbidden!r}"
            )
        for pat in affirmative_abbrev_phrases:
            assert not pat.search(md), (
                f"affirmative abbreviation-action recommendation present: "
                f"{pat.pattern!r}"
            )


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
    assert json_doc["behavior_changed"] is False
    assert json_doc["external_api_used"] is False
    assert json_doc["cue_expansion_recommended"] is False
    assert json_doc["abbreviation_handling_implemented_in_this_block"] is False


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()


# ── Sanity check: the real corpus produces 8 records ───────────────────────

def test_real_corpus_yields_exactly_eight_records():
    """Sanity check against the actual FAMILY-04 public report."""
    data = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    pool = select_abbreviation_pool(data["anonymous_per_file_table"])
    assert len(pool) == 8
    # Zero overlap with both existing helpers on the real corpus.
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
