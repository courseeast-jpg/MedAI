"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-05."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_05 import (
    SOURCE_REPORT,
    _ALPHA_SCRIPT_VOCAB,
    _NUMERIC_DISTRIBUTION_VOCAB,
    _POLICY_CANDIDATE_VOCAB,
    _SECTION_SHAPE_VOCAB,
    _TABLE_STRUCTURE_VOCAB,
    alphabetic_script_flags,
    assert_safe_public_payload,
    assign_future_policy_candidate,
    build_diagnostic_from_report,
    numeric_distribution_flags,
    render_markdown_long,
    render_markdown_summary,
    section_shape_flags,
    select_target_records,
    table_structure_flags,
    write_reports,
)


# ── Synthetic per-file table fixture ─────────────────────────────────────────

def _row(**overrides) -> dict:
    """A baseline 'table-heavy Latin policy' candidate record with the same
    signature the FAMILY-04 corpus showed (table_like=yes, numeric=medium,
    alphabetic=high, dominant=latin, detector_confidence=high, ...). Overrides
    let individual tests dial in specific deviations."""
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
        "language_script_detector_unknown_bucket": "script_detectable_language_unknown",
        "script_detection_attempted": "yes",
        "script_detection_result": "latin",
        "symbol_content_bucket": "medium",
        "detector_confidence_bucket": "high",
        "cyrillic_visibility_status": "unknown",
        "page_count_bucket": "few",
        "size_bucket": "medium",
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_payload() -> dict:
    """A small synthetic per-file table that, after DIAG-02 -> DIAG-03 ->
    DIAG-04 -> DIAG-05 filtering, contains 5 table-heavy policy candidates
    plus several rows that must be excluded."""
    rows: list[dict] = []
    # 5 table-heavy policy candidates with deliberately varied evidence
    rows.append(_row(file_id="file_001"))                                    # plain
    rows.append(_row(file_id="file_002",
                     section_heading_shape_detected="yes"))                  # headers visible
    rows.append(_row(file_id="file_003",
                     symbol_content_bucket="low",
                     numeric_content_bucket="high"))                         # high numeric, no units flag
    rows.append(_row(file_id="file_004",
                     administrative_form_shape_detected="no",
                     date_or_schedule_shape_detected="no"))                  # generic table
    rows.append(_row(file_id="file_005",
                     lab_table_shape_detected="yes"))                        # lab table shape

    # Records that DIAG-04 would route to OTHER levers (must be excluded):
    # latin_visible_language_unknown sub-pool, but with med-abbrev flag so the
    # DIAG-04 lever is "latin_medical_abbreviation_handling_audit", NOT
    # "table_heavy_language_detection_policy_audit".
    rows.append(_row(file_id="file_020",
                     table_like_structure_detected="no",
                     numeric_content_bucket="low",
                     medical_abbreviation_shape_detected="yes"))

    # Out-of-scope rows from other failure buckets / non-Unknown docs.
    rows.append(_row(file_id="file_030",
                     unknown_failure_bucket="fallback_ran_but_no_family_match",
                     unknown_ocr_routing_bucket="fallback_executed"))
    rows.append(_row(file_id="file_031",
                     unknown_failure_bucket="ambiguous_below_threshold",
                     unknown_ocr_routing_bucket="ambiguous_candidates_below_threshold"))
    rows.append(_row(file_id="file_032",
                     unknown_failure_bucket="insufficient_text_visibility",
                     unknown_ocr_routing_bucket="text_layer_present_but_too_short",
                     native_text_length_bucket="none"))                      # text-layer subset
    rows.append({"file_id": "file_040",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})

    return {"anonymous_per_file_table": rows}


# ── Selection: only the 12-style records reach the classifier ───────────────

def test_only_table_heavy_policy_records_are_targeted(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.total_table_heavy_policy_records == 5
    assert report.target_lever_label == (
        "table_heavy_language_detection_policy_audit"
    )


def test_other_lever_pools_and_other_failure_buckets_are_deferred(
    synthetic_payload,
):
    """Records DIAG-04 routes elsewhere AND records DIAG-02 places in
    other failure buckets must not contribute to any sub-pool."""
    target = select_target_records(synthetic_payload["anonymous_per_file_table"])
    file_ids = {r.get("file_id") for r in target}
    assert file_ids == {"file_001", "file_002", "file_003", "file_004", "file_005"}
    # file_020 is in the latin pool with med-abbrev -> abbreviation lever
    # file_030/031/032 are out of the language-detector chain entirely
    # file_040 is not Unknown
    for excluded in ("file_020", "file_030", "file_031", "file_032", "file_040"):
        assert excluded not in file_ids


def test_deferred_subsets_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for key in (
        "language_detector_metadata_propagation_audit_pool",
        "latin_medical_abbreviation_handling_audit_pool",
        "likely_text_layer_issue",
        "fallback_ran_but_no_family_match",
        "ambiguous_below_threshold",
    ):
        assert key in report.deferred_subsets


# ── Evidence-flag dispatch (per-view) ───────────────────────────────────────

def test_table_structure_flags_dispatch():
    r = _row()
    flags = set(table_structure_flags(r))
    assert "high_table_density" in flags
    assert "repeated_row_pattern_visible" in flags        # table + schedule
    r = _row(section_heading_shape_detected="yes")
    assert "table_headers_visible" in set(table_structure_flags(r))
    # Missing all table metadata
    r = _row(table_like_structure_detected="",
             section_heading_shape_detected="")
    flags = set(table_structure_flags(r))
    assert "insufficient_table_metadata" in flags


def test_numeric_distribution_flags_dispatch():
    r = _row()
    flags = set(numeric_distribution_flags(r))
    assert "medium_numeric_ratio" in flags
    assert "numeric_units_or_ranges_visible" in flags     # symbol=medium
    r = _row(numeric_content_bucket="high")
    assert "high_numeric_ratio" in set(numeric_distribution_flags(r))
    r = _row(numeric_content_bucket="low")
    assert "low_numeric_ratio" in set(numeric_distribution_flags(r))
    r = _row(numeric_content_bucket="")
    flags = set(numeric_distribution_flags(r))
    assert flags == {"insufficient_numeric_metadata"}
    r = _row(alphabetic_content_bucket="low")
    assert "sparse_alpha_dense_numeric" in set(numeric_distribution_flags(r))


def test_alphabetic_script_flags_dispatch():
    r = _row()
    flags = set(alphabetic_script_flags(r))
    assert "latin_script_high_confidence" in flags
    assert "alphabetic_ratio_sufficient_for_language" in flags
    r = _row(detector_confidence_bucket="medium")
    assert "latin_script_medium_confidence" in set(alphabetic_script_flags(r))
    r = _row(alphabetic_content_bucket="low")
    assert "alphabetic_ratio_too_low_for_language" in set(alphabetic_script_flags(r))
    r = _row(dominant_script="", alphabetic_content_bucket="",
             detector_confidence_bucket="")
    assert set(alphabetic_script_flags(r)) == {"insufficient_script_metadata"}


def test_section_shape_flags_dispatch():
    r = _row()
    flags = set(section_shape_flags(r))
    assert "administrative_table_shape" in flags
    assert "treatment_schedule_table_shape" in flags
    r = _row(lab_table_shape_detected="yes")
    assert "lab_or_result_section_shape" in set(section_shape_flags(r))
    r = _row(administrative_form_shape_detected="no",
             date_or_schedule_shape_detected="no")
    assert "generic_table_shape" in set(section_shape_flags(r))
    r = _row(administrative_form_shape_detected="no",
             date_or_schedule_shape_detected="no",
             table_like_structure_detected="no")
    assert "no_section_shape_available" in set(section_shape_flags(r))


# ── Vocabulary coverage ─────────────────────────────────────────────────────

def test_evidence_counts_use_controlled_vocabularies(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert set(report.table_structure_evidence_counts.keys()) == set(_TABLE_STRUCTURE_VOCAB)
    assert set(report.numeric_distribution_evidence_counts.keys()) == set(_NUMERIC_DISTRIBUTION_VOCAB)
    assert set(report.alphabetic_script_evidence_counts.keys()) == set(_ALPHA_SCRIPT_VOCAB)
    assert set(report.section_shape_evidence_counts.keys()) == set(_SECTION_SHAPE_VOCAB)
    assert set(report.future_policy_candidate_counts.keys()) == set(_POLICY_CANDIDATE_VOCAB)


def test_evidence_counts_for_synthetic_payload(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # All 5 synthetic targets have table_like=yes
    assert report.table_structure_evidence_counts["high_table_density"] == 5
    # 1 has section_heading=yes (file_002)
    assert report.table_structure_evidence_counts["table_headers_visible"] == 1
    # Numeric: file_003 has numeric=high, others medium -> 4 medium / 1 high
    assert report.numeric_distribution_evidence_counts["medium_numeric_ratio"] == 4
    assert report.numeric_distribution_evidence_counts["high_numeric_ratio"] == 1
    # Alphabetic high on all 5 -> 5 sufficient
    assert (report.alphabetic_script_evidence_counts[
        "alphabetic_ratio_sufficient_for_language"
    ] == 5)
    # latin high confidence on all 5
    assert (report.alphabetic_script_evidence_counts[
        "latin_script_high_confidence"
    ] == 5)
    # Section shape: file_005 has lab_table=yes; file_004 has admin=no, schedule=no
    assert report.section_shape_evidence_counts["lab_or_result_section_shape"] == 1
    # admin and schedule = 4 each (all but file_004)
    assert report.section_shape_evidence_counts["administrative_table_shape"] == 4
    assert (report.section_shape_evidence_counts[
        "treatment_schedule_table_shape"
    ] == 4)


# ── Future-lever candidate assignment ───────────────────────────────────────

def test_assign_future_lever_candidate_priority_order():
    # 1. table_headers_visible wins
    r = _row(section_heading_shape_detected="yes")
    assert (assign_future_policy_candidate(r)
            == "candidate_table_header_language_policy")
    # 2. numeric medium + table_like + no header -> numeric_table_safe_default
    r = _row()
    assert (assign_future_policy_candidate(r)
            == "candidate_numeric_table_safe_default_policy")
    # 3. table_heavy_latin without numeric medium / header
    r = _row(numeric_content_bucket="low",
             symbol_content_bucket="low")
    assert (assign_future_policy_candidate(r)
            == "candidate_table_heavy_latin_policy")
    # 4. metadata propagation when no policy bucket applies but the detector
    # signature is present (no table-like, but latin visible language unknown).
    r = _row(table_like_structure_detected="no",
             alphabetic_content_bucket="low",
             numeric_content_bucket="low",
             symbol_content_bucket="low")
    assert (assign_future_policy_candidate(r)
            == "candidate_metadata_propagation_audit")
    # 5. insufficient -> leave manual review only when ALL views unusable
    r = _row(
        table_like_structure_detected="",
        section_heading_shape_detected="",
        numeric_content_bucket="",
        alphabetic_content_bucket="",
        dominant_script="",
        detector_confidence_bucket="",
        language_visibility_status="",
        language_detector_attempted="",
        language_detector_input_bucket="",
    )
    assert (assign_future_policy_candidate(r)
            == "insufficient_metadata_for_next_action")


def test_future_lever_candidate_counts_on_synthetic(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.future_policy_candidate_counts
    # file_001 (plain) -> numeric_table_safe_default
    # file_002 (header) -> table_header_language_policy
    # file_003 (high numeric, no units) -> numeric_table_safe_default
    # file_004 (generic shape) -> numeric_table_safe_default
    # file_005 (lab_table=yes) -> numeric_table_safe_default
    assert counts["candidate_table_header_language_policy"] == 1
    assert counts["candidate_numeric_table_safe_default_policy"] == 4
    assert counts["candidate_table_heavy_latin_policy"] == 0
    assert counts["candidate_metadata_propagation_audit"] == 0
    assert counts["leave_manual_review"] == 0
    assert counts["insufficient_metadata_for_next_action"] == 0
    assert sum(counts.values()) == report.total_table_heavy_policy_records


# ── Implementation-block decision logic ─────────────────────────────────────

def test_implementation_block_choice_below_threshold(synthetic_payload):
    """4 numeric candidates < 5 -> E (no lever clears the threshold)."""
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.implementation_block_choice == "E"
    assert report.implementation_block_justified is False


def test_implementation_block_choice_above_threshold():
    """When >=5 candidates concentrate in one lever, that lever is chosen."""
    rows = [_row(file_id=f"file_{i:03d}") for i in range(6)]
    report = build_diagnostic_from_report({"anonymous_per_file_table": rows})
    assert report.implementation_block_choice in {"A", "B", "C", "D"}
    assert report.implementation_block_justified is True


# ── Behavior / external-API invariants ──────────────────────────────────────

def test_report_marks_behavior_unchanged(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    sp = report.safety_privacy
    assert sp["behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["language_detector_behavior_changed"] is False
    assert sp["ocr_routing_changed"] is False
    assert sp["ocr_engine_changed"] is False
    assert sp["classifier_behavior_changed"] is False
    assert sp["cue_packs_changed"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["policy_implemented_in_this_block"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["metadata_propagation_audit_pool_deferred"] is True
    assert sp["latin_medical_abbreviation_pool_deferred"] is True
    assert sp["likely_text_layer_issue_deferred"] is True
    assert sp["fallback_ran_but_no_family_match_deferred"] is True
    assert sp["ambiguous_below_threshold_excluded"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


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
            "clinical interpretation added",
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


# ── End-to-end on tmp dir ───────────────────────────────────────────────────

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


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists(), (
        f"Expected FAMILY-04 batch-eval public report at {SOURCE_REPORT}."
    )
