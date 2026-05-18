"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-03."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_03 import (
    SOURCE_REPORT,
    _LANG_DETECTOR_LABELS,
    _NEXT_ACTION_BUCKETS,
    _TEXT_LAYER_LABELS,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    classify_language_detector_root_cause,
    classify_text_layer_root_cause,
    next_action_for_language_root_cause,
    next_action_for_text_layer_root_cause,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Synthetic per-file table fixture ─────────────────────────────────────────

def _row(
    *,
    file_id: str,
    bucket: str = "insufficient_text_visibility",
    routing: str,
    visibility: str = "latin_visible_language_unknown",
    dominant_script: str = "latin",
    text_layer: str = "yes",
    image_like: str = "no",
    fallback_eligible: str = "no",
    fallback_reason: str = "",
    table_like: str = "no",
    section_heading: str = "no",
    medical_abbreviation: str = "no",
    native_length: str = "medium",
    alphabetic: str = "high",
    numeric: str = "low",
    cyrillic_visibility: str = "unknown",
) -> dict:
    return {
        "file_id": file_id,
        "predicted_document_type": "Unknown",
        "unknown_failure_bucket": bucket,
        "unknown_ocr_routing_bucket": routing,
        "language_visibility_status": visibility,
        "dominant_script": dominant_script,
        "pdf_text_layer_detected": text_layer,
        "image_like_pdf": image_like,
        "ocr_fallback_eligible": fallback_eligible,
        "ocr_fallback_not_triggered_reason": fallback_reason,
        "table_like_structure_detected": table_like,
        "section_heading_shape_detected": section_heading,
        "medical_abbreviation_shape_detected": medical_abbreviation,
        "native_text_length_bucket": native_length,
        "alphabetic_content_bucket": alphabetic,
        "numeric_content_bucket": numeric,
        "cyrillic_visibility_status": cyrillic_visibility,
    }


def _lang_record(file_id: str, **overrides) -> dict:
    defaults: dict = dict(
        file_id=file_id,
        routing="language_visibility_unknown",
        visibility="latin_visible_language_unknown",
        dominant_script="latin",
        text_layer="yes",
        image_like="no",
        fallback_eligible="no",
        fallback_reason="language_visibility_unknown",
        native_length="medium",
        alphabetic="high",
    )
    defaults.update(overrides)
    return _row(**defaults)


def _text_record(file_id: str, **overrides) -> dict:
    defaults: dict = dict(
        file_id=file_id,
        routing="text_layer_present_but_too_short",
        visibility="latin_visible_language_unknown",
        dominant_script="latin",
        text_layer="yes",
        image_like="no",
        fallback_eligible="no",
        fallback_reason="text_layer_present_but_too_short",
        native_length="none",
        alphabetic="high",
    )
    defaults.update(overrides)
    return _row(**defaults)


@pytest.fixture
def synthetic_payload() -> dict:
    """Synthetic per-file table. All rows here pass the DIAG-02 upstream
    filter so they reach the DIAG-03 per-subset classifier.

    Rows that would be excluded by the upstream filter (empty visibility
    metadata, image_like=yes) are tested separately via the direct
    classifier-dispatch tests.
    """
    rows: list[dict] = [
        # ── language_script_visible_detector_unresolved subset ──
        # Numeric/table-heavy variants → numeric_or_table_heavy_*
        _lang_record("file_001", table_like="yes", numeric="medium"),
        _lang_record("file_002", table_like="yes", numeric="high"),
        # Plain latin-visible-language-unknown
        _lang_record("file_003"),
        _lang_record("file_004"),
        # Cyrillic-visible-language-unknown
        _lang_record("file_005",
                     visibility="cyrillic_visible_language_unknown",
                     dominant_script="cyrillic"),
        # Mixed script
        _lang_record("file_006",
                     visibility="mixed_visible_language_unknown",
                     dominant_script="mixed"),
        # ── likely_text_layer_issue subset ──
        # text_layer_too_short via "none"
        _text_record("file_010", native_length="none"),
        # short with table → table_structure_visible_but_text_insufficient
        _text_record("file_011", native_length="short", table_like="yes"),
        # tiny without table → text_layer_too_short
        _text_record("file_012", native_length="tiny"),
        # ── Out-of-scope rows: must be ignored ──
        # fallback bucket
        _row(file_id="file_020",
             bucket="fallback_ran_but_no_family_match",
             routing="fallback_executed",
             visibility="latin_visible_language_unknown"),
        # ambiguous bucket
        _row(file_id="file_021",
             bucket="ambiguous_below_threshold",
             routing="ambiguous_candidates_below_threshold",
             visibility="latin_visible_language_unknown"),
        # Non-Unknown row
        {"file_id": "file_030",
         "predicted_document_type": "Lab result",
         "unknown_failure_bucket": ""},
    ]
    return {"anonymous_per_file_table": rows}


# ── Subset selection ────────────────────────────────────────────────────────

def test_only_priority_unknown_rows_are_analyzed(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    lang = report.target_subset_counts["language_script_visible_detector_unresolved"]
    text = report.target_subset_counts["likely_text_layer_issue"]
    assert lang == 6
    assert text == 3
    assert report.total_priority_analyzed == lang + text


# ── Language detector root causes ───────────────────────────────────────────

def test_classify_language_detector_root_cause_dispatch():
    # numeric_or_table_heavy_language_detector_gap takes precedence
    r = _lang_record("file_a", table_like="yes", numeric="high")
    assert (classify_language_detector_root_cause(r)
            == "numeric_or_table_heavy_language_detector_gap")
    # Cyrillic visible language unknown
    r = _lang_record("file_b",
                     visibility="cyrillic_visible_language_unknown",
                     dominant_script="cyrillic")
    assert (classify_language_detector_root_cause(r)
            == "cyrillic_visible_language_unknown")
    # Mixed script
    r = _lang_record("file_c",
                     visibility="mixed_visible_language_unknown",
                     dominant_script="mixed")
    assert classify_language_detector_root_cause(r) == "mixed_script_detector_gap"
    # Plain Latin visible language unknown
    r = _lang_record("file_d")
    assert (classify_language_detector_root_cause(r)
            == "latin_visible_language_unknown")
    # Insufficient metadata
    r = _lang_record("file_e", visibility="", dominant_script="")
    assert classify_language_detector_root_cause(r) == "insufficient_safe_metadata"


def test_language_subset_root_cause_breakdown(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.root_cause_counts[
        "language_script_visible_detector_unresolved"
    ]
    # All declared labels must be present in the breakdown
    for label in _LANG_DETECTOR_LABELS:
        assert label in counts
    assert counts["numeric_or_table_heavy_language_detector_gap"] == 2
    assert counts["latin_visible_language_unknown"] == 2
    assert counts["cyrillic_visible_language_unknown"] == 1
    assert counts["mixed_script_detector_gap"] == 1
    # insufficient_safe_metadata and script_visible_language_detector_gap are
    # defensive branches; unreachable for the synthetic pipeline payload.
    assert counts["insufficient_safe_metadata"] == 0
    assert counts["script_visible_language_detector_gap"] == 0
    assert sum(counts.values()) == 6


# ── Text-layer root causes ──────────────────────────────────────────────────

def test_classify_text_layer_root_cause_dispatch():
    # none + no table -> text_layer_too_short
    r = _text_record("file_a", native_length="none")
    assert classify_text_layer_root_cause(r) == "text_layer_too_short"
    # short + table -> table_structure_visible_but_text_insufficient
    r = _text_record("file_b", native_length="short", table_like="yes")
    assert (classify_text_layer_root_cause(r)
            == "table_structure_visible_but_text_insufficient")
    # image-like overrides
    r = _text_record("file_c", image_like="yes")
    assert classify_text_layer_root_cause(r) == "image_like_with_partial_text"


def test_text_layer_subset_root_cause_breakdown(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.root_cause_counts["likely_text_layer_issue"]
    for label in _TEXT_LAYER_LABELS:
        assert label in counts
    assert counts["text_layer_too_short"] == 2
    assert counts["table_structure_visible_but_text_insufficient"] == 1
    # Defensive branches not reachable via the synthetic pipeline payload.
    assert counts["image_like_with_partial_text"] == 0
    assert counts["no_safe_text_visibility_metadata"] == 0
    assert sum(counts.values()) == 3


# ── Next-action buckets ─────────────────────────────────────────────────────

def test_next_action_mapping_for_language():
    assert (next_action_for_language_root_cause("latin_visible_language_unknown")
            == "candidate_language_detector_diagnostic")
    assert (next_action_for_language_root_cause(
        "numeric_or_table_heavy_language_detector_gap")
            == "candidate_language_detector_diagnostic")
    assert (next_action_for_language_root_cause("insufficient_safe_metadata")
            == "insufficient_metadata_for_next_action")


def test_next_action_mapping_for_text_layer():
    assert (next_action_for_text_layer_root_cause("text_layer_too_short")
            == "candidate_text_layer_extraction_diagnostic")
    assert (next_action_for_text_layer_root_cause("image_like_with_partial_text")
            == "candidate_ocr_routing_diagnostic")
    assert (next_action_for_text_layer_root_cause("no_safe_text_visibility_metadata")
            == "insufficient_metadata_for_next_action")


def test_next_action_bucket_counts_are_controlled_vocab(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    counts = report.next_action_bucket_counts
    assert set(counts.keys()) == set(_NEXT_ACTION_BUCKETS)
    # 6 language records all route to candidate_language_detector_diagnostic
    assert counts["candidate_language_detector_diagnostic"] == 6
    # 3 text-layer records all route to candidate_text_layer_extraction_diagnostic
    assert counts["candidate_text_layer_extraction_diagnostic"] == 3
    # Defensive next-action buckets must remain at zero for this payload.
    assert counts["candidate_ocr_routing_diagnostic"] == 0
    assert counts["insufficient_metadata_for_next_action"] == 0
    assert counts["leave_manual_review"] == 0
    assert sum(counts.values()) == 9


# ── Implementation-block decision logic ─────────────────────────────────────

def test_implementation_block_decision_uses_thresholded_logic(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # On the synthetic payload (6 lang + 3 text-layer + 1 routing): only the
    # language-detector pool clears the threshold, so choice should be A.
    assert report.implementation_block_choice in {"A", "B", "C", "D"}
    assert report.implementation_block_choice == "A"
    assert report.implementation_block_justified is True


def test_decision_falls_back_to_D_when_all_pools_below_threshold():
    payload = {"anonymous_per_file_table": [
        _lang_record(f"file_{i:03d}") for i in range(2)
    ]}
    report = build_diagnostic_from_report(payload)
    # 2 lang candidates only; below the 5-record threshold.
    assert report.implementation_block_choice == "D"
    assert report.implementation_block_justified is False


# ── Behavior / external-API invariants ──────────────────────────────────────

def test_report_marks_behavior_unchanged(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    sp = report.safety_privacy
    assert sp["behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["language_detector_behavior_changed"] is False
    assert sp["ocr_routing_changed"] is False
    assert sp["classifier_behavior_changed"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["cue_packs_changed"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["ambiguous_below_threshold_excluded"] is True
    assert sp["fallback_ran_but_no_family_match_deferred"] is True


def test_deferred_subsets_listed(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert "ambiguous_below_threshold" in report.deferred_subsets
    assert "fallback_ran_but_no_family_match" in report.deferred_subsets
    for note in report.deferred_subsets.values():
        # The deferred notes must not propose cue expansion.
        low = note.lower()
        for token in ("add cue", "expand cue", "implement"):
            assert token not in low


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


def test_no_cue_expansion_recommendation_in_rendered_output(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for md in (render_markdown_summary(report), render_markdown_long(report)):
        low = md.lower()
        # Allow phrases like "cue expansion is not recommended" but never
        # the affirmative form.
        for affirmative in (
            "recommend cue expansion",
            "add new cue",
            "expand cue pack",
            "expand the cue",
        ):
            assert affirmative not in low


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


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists(), (
        f"Expected FAMILY-04 batch-eval public report at {SOURCE_REPORT}."
    )
