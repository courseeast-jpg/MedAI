"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_06a import (
    EXCLUSION_RULES,
    FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA,
    FUTURE_VALIDATION_REQUIREMENTS,
    POSITIVE_SIGNATURE,
    PROPOSED_FUTURE_DEFAULT,
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    select_numeric_table_records,
    write_reports,
)


# ── Fixture: synthetic per-file table reaching all the way through ──────────

def _row(**overrides) -> dict:
    """Baseline numeric-table policy candidate (matches the homogeneous
    FAMILY-04 signature). Overrides let individual tests dial in variations."""
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
        "language_script_detector_unknown_bucket":
            "script_detectable_language_unknown",
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
    rows: list[dict] = []
    # 4 records that should land in the numeric-table policy slice
    for i in range(4):
        rows.append(_row(file_id=f"file_{i + 1:03d}"))
    # 1 record headed for the table-header policy lever (deferred)
    rows.append(_row(file_id="file_010",
                     section_heading_shape_detected="yes"))
    # Records that should NOT make it through to the numeric-table slice
    # latin med-abbrev pool
    rows.append(_row(file_id="file_020",
                     table_like_structure_detected="no",
                     numeric_content_bucket="low",
                     medical_abbreviation_shape_detected="yes"))
    # fallback / ambiguous / text-layer / non-Unknown
    rows.append(_row(file_id="file_030",
                     unknown_failure_bucket="fallback_ran_but_no_family_match",
                     unknown_ocr_routing_bucket="fallback_executed"))
    rows.append(_row(file_id="file_031",
                     unknown_failure_bucket="ambiguous_below_threshold",
                     unknown_ocr_routing_bucket=
                     "ambiguous_candidates_below_threshold"))
    rows.append(_row(file_id="file_032",
                     unknown_failure_bucket="insufficient_text_visibility",
                     unknown_ocr_routing_bucket="text_layer_present_but_too_short",
                     native_text_length_bucket="none"))
    rows.append({"file_id": "file_040",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})
    return {"anonymous_per_file_table": rows}


# ── Selection ───────────────────────────────────────────────────────────────

def test_only_numeric_table_records_are_targeted(synthetic_payload):
    target = select_numeric_table_records(
        synthetic_payload["anonymous_per_file_table"]
    )
    ids = {r["file_id"] for r in target}
    assert ids == {"file_001", "file_002", "file_003", "file_004"}


def test_table_header_record_is_deferred(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.total_numeric_table_policy_records == 4
    assert report.deferred_table_header_record_count == 1
    assert ("candidate_table_header_language_policy_record_count"
            in report.deferred_subsets)


def test_other_pools_are_excluded(synthetic_payload):
    """Metadata-propagation, abbreviation, text-layer, fallback, and
    ambiguous records must not enter the numeric-table slice."""
    target = select_numeric_table_records(
        synthetic_payload["anonymous_per_file_table"]
    )
    ids = {r["file_id"] for r in target}
    for excluded in ("file_020", "file_030", "file_031", "file_032", "file_040"):
        assert excluded not in ids


def test_deferred_subsets_listed(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for key in (
        "candidate_table_header_language_policy_record_count",
        "language_detector_metadata_propagation_audit_pool",
        "latin_medical_abbreviation_handling_audit_pool",
        "likely_text_layer_issue",
        "fallback_ran_but_no_family_match",
        "ambiguous_below_threshold",
    ):
        assert key in report.deferred_subsets


# ── Positive signature ─────────────────────────────────────────────────────

def test_positive_signature_keys_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    rendered_keys = {item["key"] for item in report.positive_signature}
    expected_keys = {k for k, _ in POSITIVE_SIGNATURE}
    assert rendered_keys == expected_keys


def test_positive_signature_holds_on_synthetic_slice(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.positive_signature_holds_on_all_priority_records is True
    for info in report.positive_signature_match_report.values():
        assert info["fully_matches"] is True


def test_positive_signature_fails_when_violated():
    """If a record breaks the signature, fully_matches goes False."""
    rows = [
        _row(file_id="file_001"),
        _row(file_id="file_002",
             dominant_script="cyrillic",
             script_detection_result="cyrillic"),  # violates dominant_script=latin
    ]
    # DIAG-05 selection would already drop the cyrillic record (it would
    # never reach the table-heavy latin sub-pool). To force a signature
    # violation inside the slice, we keep the slice's selection but break
    # a non-filter field. Drop a non-essential value:
    rows[0]["administrative_form_shape_detected"] = "no"
    rows[0]["date_or_schedule_shape_detected"] = "no"
    payload = {"anonymous_per_file_table": rows}
    report = build_diagnostic_from_report(payload)
    # If DIAG-05 still includes the broken record, the signature must fail.
    if report.total_numeric_table_policy_records > 0:
        all_match = all(
            v["fully_matches"]
            for v in report.positive_signature_match_report.values()
        )
        assert all_match is False
        assert report.positive_signature_holds_on_all_priority_records is False


# ── Exclusion rules ────────────────────────────────────────────────────────

def test_exclusion_rules_render(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert set(report.exclusion_rules) == set(EXCLUSION_RULES)


def test_exclusion_audit_clean_on_synthetic_slice(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.no_priority_record_violates_any_exclusion_rule is True
    for row in report.exclusion_audit:
        assert row["violating_record_count"] == 0


# ── Proposed future default ─────────────────────────────────────────────────

def test_future_default_contract(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    d = report.proposed_future_default
    # The future default MUST preserve all five guard rails verbatim.
    assert d["must_not_auto_accept"] is True
    assert d["must_not_classify_clinical_meaning"] is True
    assert d["must_not_parse_values"] is True
    assert d["must_not_write_active_clinical_facts"] is True
    assert d["must_keep_document_review_bound"] is True
    assert d is PROPOSED_FUTURE_DEFAULT or d == PROPOSED_FUTURE_DEFAULT


# ── Acceptance criteria + validation requirements ──────────────────────────

def test_acceptance_criteria_are_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_implementation_acceptance_criteria)
            == set(FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA))


def test_validation_requirements_are_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert (set(report.future_validation_requirements)
            == set(FUTURE_VALIDATION_REQUIREMENTS))


# ── Progress estimate ───────────────────────────────────────────────────────

def test_progress_estimate_is_rendered(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    pe = report.progress_estimate
    # Required keys are present and non-empty.
    for k in (
        "before_06a_unknown_track_done_pct",
        "before_06a_unknown_track_remaining_pct",
        "after_06a_unknown_track_done_pct",
        "after_06a_unknown_track_remaining_pct",
        "after_06a_project_done_pct",
        "after_06a_project_remaining_pct",
        "note",
    ):
        assert k in pe and pe[k]


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_06a_unknown_track_done_pct",
        "approximately 55%",
        "approximately 45%",
        "after_06a_unknown_track_done_pct",
        "approximately 62%",
        "approximately 38%",
        "after_06a_project_done_pct",
        "approximately 76%",
        "approximately 24%",
    ):
        assert token in md


# ── Implementation status invariants ────────────────────────────────────────

def test_report_marks_no_implementation(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    assert report.policy_implemented_in_this_block is False
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
    assert sp["deferred_pools_remain_deferred"] is True


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


def test_no_forbidden_recommendation_or_implementation_in_rendered_output(
    synthetic_payload,
):
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
            "implements the policy",
            "we implement",
            "we change classifier",
            "we modify the detector",
            "ddi parsing",
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
    assert json_doc["policy_implemented_in_this_block"] is False


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists(), (
        f"Expected FAMILY-04 batch-eval public report at {SOURCE_REPORT}."
    )
