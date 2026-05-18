"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-04."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_04 import (
    SOURCE_REPORT,
    _EVIDENCE_LATIN_LANG_UNKNOWN,
    _EVIDENCE_TABLE_HEAVY,
    _FUTURE_LEVERS,
    _RC_LATIN,
    _RC_TABLE_HEAVY,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    evidence_flags_for_latin_lang,
    evidence_flags_for_table_heavy,
    next_lever_for_latin_lang,
    next_lever_for_table_heavy,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Synthetic per-file table fixture ─────────────────────────────────────────

def _row(**overrides) -> dict:
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
        "table_like_structure_detected": "no",
        "section_heading_shape_detected": "no",
        "medical_abbreviation_shape_detected": "no",
        "lab_table_shape_detected": "no",
        "native_text_length_bucket": "medium",
        "alphabetic_content_bucket": "high",
        "numeric_content_bucket": "low",
        "language_detector_attempted": "yes",
        "language_detector_input_bucket": "sufficient",
        "language_script_visibility": "latin_visible_language_unknown",
        "language_script_detector_unknown_bucket": "script_detectable_language_unknown",
        "script_detection_attempted": "yes",
        "script_detection_result": "latin",
        "symbol_content_bucket": "medium",
        "detector_confidence_bucket": "high",
        "cyrillic_visibility_status": "unknown",
    }
    base.update(overrides)
    return base


def _table_heavy(**overrides) -> dict:
    """A record that DIAG-03 would classify as
    ``numeric_or_table_heavy_language_detector_gap`` (table_like=yes,
    numeric=medium)."""
    defaults = dict(table_like_structure_detected="yes",
                    numeric_content_bucket="medium")
    defaults.update(overrides)
    return _row(**defaults)


def _latin_lang(**overrides) -> dict:
    """A record that DIAG-03 would classify as
    ``latin_visible_language_unknown`` (no table-heavy condition, numeric=low)."""
    defaults = dict(table_like_structure_detected="no",
                    numeric_content_bucket="low")
    defaults.update(overrides)
    return _row(**defaults)


@pytest.fixture
def synthetic_payload() -> dict:
    """Synthetic per-file table that, after DIAG-02 -> DIAG-03 -> DIAG-04
    filtering, contains exactly the right counts in each sub-pool."""
    rows: list[dict] = []
    # Table-heavy sub-pool: 6 records
    rows.append(_table_heavy(file_id="file_001"))                        # plain table-heavy
    rows.append(_table_heavy(file_id="file_002", lab_table_shape_detected="yes"))
    rows.append(_table_heavy(file_id="file_003",
                             medical_abbreviation_shape_detected="yes"))
    rows.append(_table_heavy(file_id="file_004",
                             language_script_detector_unknown_bucket=
                             "detector_input_garbled_or_mojibake"))
    rows.append(_table_heavy(file_id="file_005", symbol_content_bucket="high"))
    rows.append(_table_heavy(file_id="file_006",
                             alphabetic_content_bucket="low",
                             numeric_content_bucket="high"))
    # Latin-language sub-pool: 5 records
    rows.append(_latin_lang(file_id="file_010"))                          # plain
    rows.append(_latin_lang(file_id="file_011",
                            medical_abbreviation_shape_detected="yes"))
    rows.append(_latin_lang(file_id="file_012",
                            table_like_structure_detected="yes"))
    rows.append(_latin_lang(file_id="file_013",
                            detector_confidence_bucket="low"))            # no propagation
    rows.append(_latin_lang(file_id="file_014",
                            language_detector_input_bucket="insufficient"))  # no propagation
    # Out-of-scope rows (must be ignored)
    rows.append(_row(file_id="file_020",
                     unknown_failure_bucket="fallback_ran_but_no_family_match",
                     unknown_ocr_routing_bucket="fallback_executed"))
    rows.append(_row(file_id="file_021",
                     unknown_failure_bucket="ambiguous_below_threshold",
                     unknown_ocr_routing_bucket="ambiguous_candidates_below_threshold"))
    rows.append(_row(file_id="file_022",
                     unknown_failure_bucket="insufficient_text_visibility",
                     unknown_ocr_routing_bucket="text_layer_present_but_too_short",
                     pdf_text_layer_detected="yes",
                     native_text_length_bucket="none"))  # likely_text_layer_issue
    rows.append({"file_id": "file_030",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})
    return {"anonymous_per_file_table": rows}


# ── Selection: only 31-style records reach the classifier ───────────────────

def test_only_language_detector_priority_records_are_targeted(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.sub_pool_counts[_RC_TABLE_HEAVY] == 6
    assert report.sub_pool_counts[_RC_LATIN] == 5
    assert report.total_language_detector_records == 11


def test_text_layer_and_fallback_records_are_excluded(synthetic_payload):
    """text-layer / fallback / ambiguous rows must NOT contribute to any
    sub-pool. They are explicitly deferred."""
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.deferred_subsets["likely_text_layer_issue"]
    assert report.deferred_subsets["fallback_ran_but_no_family_match"]
    assert report.deferred_subsets["ambiguous_below_threshold"]
    # Total still excludes the file_020, file_021, file_022, file_030 rows.
    assert report.total_language_detector_records == 11


# ── Evidence flag classifiers ───────────────────────────────────────────────

def test_evidence_flags_for_table_heavy_dispatch():
    # Plain table-heavy -> table_heavy + numeric_heavy
    r = _table_heavy()
    flags = set(evidence_flags_for_table_heavy(r))
    assert "table_heavy_latin_visible" in flags
    assert "numeric_heavy_latin_visible" in flags
    # lab table shape
    r = _table_heavy(lab_table_shape_detected="yes")
    assert "lab_table_shape_latin_visible" in evidence_flags_for_table_heavy(r)
    # sparse_words_many_numbers
    r = _table_heavy(alphabetic_content_bucket="low",
                     numeric_content_bucket="high")
    assert "sparse_words_many_numbers" in evidence_flags_for_table_heavy(r)
    # detector_input_too_structural (garbled)
    r = _table_heavy(language_script_detector_unknown_bucket=
                     "detector_input_garbled_or_mojibake")
    assert "detector_input_too_structural" in evidence_flags_for_table_heavy(r)


def test_evidence_flags_for_latin_lang_dispatch():
    r = _latin_lang()
    flags = set(evidence_flags_for_latin_lang(r))
    assert "latin_words_visible_detector_unknown" in flags
    assert "latin_script_detected_language_missing" in flags
    assert "detector_output_not_propagated" in flags
    # latin_medical_abbrev_visible
    r = _latin_lang(medical_abbreviation_shape_detected="yes")
    assert "latin_medical_abbrev_visible" in evidence_flags_for_latin_lang(r)
    # latin_table_headers_visible
    r = _latin_lang(table_like_structure_detected="yes")
    assert "latin_table_headers_visible" in evidence_flags_for_latin_lang(r)
    # detector_output_not_propagated requires sufficient input + medium/high
    # confidence; flip one and the flag goes away.
    r = _latin_lang(detector_confidence_bucket="low")
    assert "detector_output_not_propagated" not in evidence_flags_for_latin_lang(r)


def test_evidence_counts_use_controlled_vocab(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    table_heavy_counts = report.evidence_bucket_counts[_RC_TABLE_HEAVY]
    latin_counts = report.evidence_bucket_counts[_RC_LATIN]
    assert set(table_heavy_counts.keys()) == set(_EVIDENCE_TABLE_HEAVY)
    assert set(latin_counts.keys()) == set(_EVIDENCE_LATIN_LANG_UNKNOWN)


def test_evidence_counts_for_table_heavy_synthetic(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.evidence_bucket_counts[_RC_TABLE_HEAVY]
    # 6 records, all table_like=yes + numeric=medium-or-high.
    assert counts["table_heavy_latin_visible"] == 6
    assert counts["numeric_heavy_latin_visible"] == 6
    assert counts["lab_table_shape_latin_visible"] == 1
    assert counts["sparse_words_many_numbers"] == 1
    # 2 records contribute: file_004 (garbled) and file_005 (symbol high).
    assert counts["detector_input_too_structural"] == 2
    assert counts["insufficient_safe_metadata"] == 0


def test_evidence_counts_for_latin_lang_synthetic(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.evidence_bucket_counts[_RC_LATIN]
    # 5 records.
    assert counts["latin_words_visible_detector_unknown"] == 5
    assert counts["latin_script_detected_language_missing"] == 5
    assert counts["latin_medical_abbrev_visible"] == 1
    assert counts["latin_table_headers_visible"] == 1
    # file_013 has detector_confidence_bucket=low, file_014 input=insufficient
    # -> 3 records still have the propagation signature.
    assert counts["detector_output_not_propagated"] == 3
    assert counts["insufficient_safe_metadata"] == 0


# ── Next-action levers ──────────────────────────────────────────────────────

def test_next_lever_for_table_heavy_dispatch():
    # Medical abbreviation on the record takes precedence over policy audit.
    # `latin_medical_abbrev_visible` is not in the table_heavy evidence
    # vocabulary, so the lever decision consults the raw record signal.
    record = {"medical_abbreviation_shape_detected": "yes"}
    flags = ["table_heavy_latin_visible"]
    assert (next_lever_for_table_heavy(record, flags)
            == "latin_medical_abbreviation_handling_audit")
    # Pure table-heavy (no abbrev signal) -> policy audit
    flags = ["table_heavy_latin_visible", "numeric_heavy_latin_visible"]
    assert (next_lever_for_table_heavy({}, flags)
            == "table_heavy_language_detection_policy_audit")
    # Insufficient metadata
    flags = ["insufficient_safe_metadata"]
    assert (next_lever_for_table_heavy({}, flags)
            == "insufficient_metadata_for_next_action")


def test_next_lever_for_latin_lang_dispatch():
    # Medical abbreviation takes precedence over propagation
    flags = ["latin_words_visible_detector_unknown", "latin_medical_abbrev_visible",
             "detector_output_not_propagated"]
    assert (next_lever_for_latin_lang({}, flags)
            == "latin_medical_abbreviation_handling_audit")
    # No abbrev -> propagation audit
    flags = ["latin_words_visible_detector_unknown",
             "detector_output_not_propagated"]
    assert (next_lever_for_latin_lang({}, flags)
            == "language_detector_metadata_propagation_audit")
    # No propagation -> table headers audit
    flags = ["latin_words_visible_detector_unknown", "latin_table_headers_visible"]
    assert (next_lever_for_latin_lang({}, flags)
            == "table_heavy_language_detection_policy_audit")


def test_future_lever_counts_use_controlled_vocab(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert set(report.future_lever_counts.keys()) == set(_FUTURE_LEVERS)


def test_future_lever_counts_for_synthetic_payload(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.future_lever_counts
    # The synthetic payload routes:
    #   * 6 table-heavy: 5 -> table_heavy_lang_detection_policy_audit;
    #     1 -> latin_medical_abbreviation_handling_audit (file_003).
    #   * 5 latin: 1 -> latin_medical_abbreviation_handling_audit (file_011);
    #     2 -> language_detector_metadata_propagation_audit (file_010, file_013-not? wait);
    # The propagation count is 3 in evidence; among latin records,
    # 3 have detector_output_not_propagated (file_010, file_011, file_012).
    # Lever priority for latin: abbrev > propagation > table_headers.
    # file_010 (plain)        -> propagation
    # file_011 (abbrev)       -> abbrev
    # file_012 (table headers + propagation) -> propagation
    # file_013 (low conf, no propagation, no abbrev, no table) -> leave_manual_review
    # file_014 (insufficient input, no propagation, no abbrev) -> leave_manual_review
    assert counts["language_detector_metadata_propagation_audit"] == 2
    assert counts["table_heavy_language_detection_policy_audit"] == 5
    assert counts["latin_medical_abbreviation_handling_audit"] == 2
    assert counts["leave_manual_review"] == 2
    assert counts["insufficient_metadata_for_next_action"] == 0
    # Total should equal sum of sub-pool counts.
    assert sum(counts.values()) == report.total_language_detector_records


# ── Implementation-block decision logic ─────────────────────────────────────

def test_implementation_block_choice_on_synthetic(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # The synthetic levers: 2 propagation, 5 table-policy, 2 abbrev.
    # Only table-policy (5) >= threshold 5; chose B.
    assert report.implementation_block_choice == "B"
    assert report.implementation_block_justified is True


def test_decision_falls_back_to_D_when_no_lever_clears_threshold():
    """All four pools below threshold -> D."""
    rows = [
        _table_heavy(file_id="file_001"),
        _table_heavy(file_id="file_002"),
        _latin_lang(file_id="file_010"),
    ]
    payload = {"anonymous_per_file_table": rows}
    report = build_diagnostic_from_report(payload)
    assert report.implementation_block_choice == "D"
    assert report.implementation_block_justified is False


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
    assert sp["all_records_remain_review_bound"] is True
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


def test_no_cue_expansion_or_ocr_change_recommendation_in_rendered_output(
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
