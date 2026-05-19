"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_09a import (
    EXCLUSION_RULES,
    FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA,
    FUTURE_VALIDATION_REQUIREMENTS,
    POSITIVE_SIGNAL_PATTERN,
    PROPOSED_FUTURE_PROPAGATION,
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    select_propagation_pool,
    write_reports,
)


# ── Synthetic per-file table fixture ────────────────────────────────────────

def _propagation_record(**overrides) -> dict:
    """Baseline record that satisfies the propagation-pool signature.
    DIAG-04 will route this record to language_detector_metadata_propagation_audit.
    Distinguished from the numeric-table pool by ``numeric_content_bucket=low``
    (so DIAG-03's table-heavy root-cause does not fire) but with
    ``alphabetic_content_bucket=medium`` so DIAG-03 still classifies it into
    the table-heavy sub-pool... actually we want the latin sub-pool. The
    helper code below makes the record fall into the latin_visible_language_unknown
    sub-pool by leaving table_like=no AND keeping latin visibility."""
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
    """A record that matches the DIAG-06A numeric-table safe-default helper.
    Distinguished from the propagation pool by ``numeric_content_bucket=medium``
    plus the administrative/schedule shape signals."""
    base = _propagation_record()
    base.update({
        "numeric_content_bucket": "medium",
        "administrative_form_shape_detected": "yes",
        "date_or_schedule_shape_detected": "yes",
    })
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_payload() -> dict:
    rows: list[dict] = []
    # 4 propagation-pool candidates
    for i in range(4):
        rows.append(_propagation_record(file_id=f"file_{i + 1:03d}"))
    # 1 numeric-table safe-default record (must be excluded - DIAG-06A handled)
    rows.append(_numeric_table_record(file_id="file_010"))
    # 1 abbreviation-pool record (deferred)
    rows.append(_propagation_record(file_id="file_011",
                                    medical_abbreviation_shape_detected="yes"))
    # 1 cyrillic record (exclusion fires; also won't enter the upstream slice)
    rows.append(_propagation_record(file_id="file_012",
                                    dominant_script="cyrillic"))
    # Out-of-scope rows
    rows.append(_propagation_record(file_id="file_020",
                                    unknown_failure_bucket="fallback_ran_but_no_family_match"))
    rows.append(_propagation_record(file_id="file_021",
                                    unknown_failure_bucket="ambiguous_below_threshold"))
    rows.append(_propagation_record(file_id="file_022",
                                    unknown_ocr_routing_bucket="text_layer_present_but_too_short",
                                    native_text_length_bucket="none"))
    rows.append({"file_id": "file_030",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})
    return {"anonymous_per_file_table": rows}


# ── Slice selection ────────────────────────────────────────────────────────

def test_only_propagation_pool_records_are_targeted(synthetic_payload):
    pool = select_propagation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert ids == {"file_001", "file_002", "file_003", "file_004"}


def test_numeric_table_safe_default_records_excluded(synthetic_payload):
    """The 11-record numeric-table pool must not enter the propagation slice."""
    pool = select_propagation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert "file_010" not in ids


def test_abbreviation_pool_records_deferred(synthetic_payload):
    pool = select_propagation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    assert "file_011" not in ids


def test_other_failure_buckets_excluded(synthetic_payload):
    pool = select_propagation_pool(synthetic_payload["anonymous_per_file_table"])
    ids = {r.get("file_id") for r in pool}
    for excluded in ("file_020", "file_021", "file_022", "file_030"):
        assert excluded not in ids


def test_no_overlap_with_numeric_table_safe_default_pool(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.overlap_with_numeric_table_safe_default_pool == 0
    assert report.no_overlap_with_numeric_table_safe_default_pool is True


def test_deferred_subsets_listed(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for key in (
        "numeric_table_safe_default_pool_already_handled",
        "candidate_latin_medical_abbreviation_handling_audit_pool",
        "candidate_table_header_language_policy_record",
        "likely_text_layer_issue",
        "fallback_ran_but_no_family_match",
        "ambiguous_below_threshold",
    ):
        assert key in report.deferred_subsets


# ── Positive signal pattern ─────────────────────────────────────────────────

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


# ── Exclusion rules ─────────────────────────────────────────────────────────

def test_exclusion_rules_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert set(report.exclusion_rules) == set(EXCLUSION_RULES)


def test_exclusion_audit_clean(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.no_priority_record_violates_any_exclusion_rule is True
    for row in report.exclusion_audit:
        assert row["violating_record_count"] == 0


# ── Proposed future propagation ────────────────────────────────────────────

def test_future_propagation_contract(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    d = report.proposed_future_propagation
    # Every guardrail flag must be True (preserves safety).
    assert d["must_not_alter_raw_detector_output"] is True
    assert d["must_not_auto_accept"] is True
    assert d["must_not_classify_clinical_meaning"] is True
    assert d["must_not_parse_values"] is True
    assert d["must_not_write_active_clinical_facts"] is True
    assert d["must_keep_document_review_bound"] is True
    assert d["must_be_default_off_behind_separate_env_flag_or_operator_setting"] is True
    assert d["must_not_overlap_with_numeric_table_safe_default_pool"] is True
    assert d is PROPOSED_FUTURE_PROPAGATION or d == PROPOSED_FUTURE_PROPAGATION


# ── Acceptance criteria + validation requirements ──────────────────────────

def test_acceptance_criteria_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_implementation_acceptance_criteria)
            == set(FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA))


def test_validation_requirements_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_validation_requirements)
            == set(FUTURE_VALIDATION_REQUIREMENTS))


# ── Progress estimate ──────────────────────────────────────────────────────

def test_progress_estimate_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    pe = report.progress_estimate
    for k in (
        "before_09a_unknown_track_done_pct",
        "before_09a_unknown_track_remaining_pct",
        "before_09a_project_done_pct",
        "before_09a_project_remaining_pct",
        "after_09a_unknown_track_done_pct",
        "after_09a_unknown_track_remaining_pct",
        "after_09a_project_done_pct",
        "after_09a_project_remaining_pct",
        "note",
    ):
        assert k in pe and pe[k]


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_09a_unknown_track_done_pct",
        "approximately 76%",
        "approximately 24%",
        "after_09a_unknown_track_done_pct",
        "approximately 80%",
        "approximately 20%",
        "before_09a_project_done_pct",
        "approximately 79%",
        "approximately 21%",
        "after_09a_project_done_pct",
    ):
        assert token in md


# ── Implementation-status invariants ────────────────────────────────────────

def test_no_implementation_marked(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    assert report.propagation_implemented_in_this_block is False
    sp = report.safety_privacy
    assert sp["behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["raw_language_detector_behavior_changed"] is False
    assert sp["language_metadata_propagation_behavior_changed"] is False
    assert sp["classifier_behavior_changed"] is False
    assert sp["cue_packs_changed"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["propagation_implemented_in_this_block"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["no_overlap_with_numeric_table_safe_default_pool"] is True


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
    report = build_diagnostic_from_report(synthetic_payload)
    for md in (render_markdown_summary(report), render_markdown_long(report)):
        low = md.lower()
        for forbidden in (
            "recommend cue expansion",
            "add new cue",
            "expand cue pack",
            "expand the cue",
            "change ocr routing",
            "modify ocr routing",
            "change the ocr engine",
            "parse lab values",
            "parse medications",
            "implements the propagation",
            "we implement",
            "we modify the detector",
        ):
            assert forbidden not in low


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


# ── End-to-end ──────────────────────────────────────────────────────────────

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
    assert json_doc["propagation_implemented_in_this_block"] is False


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()
