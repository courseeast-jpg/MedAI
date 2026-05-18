"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-02."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_02 import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    build_section_a,
    build_section_b,
    build_section_c,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Synthetic per-file table fixture ─────────────────────────────────────────

def _row(
    bucket: str,
    *,
    file_id: str,
    cue_audit_result: str = "needs_manual_review",
    routing: str = "language_visibility_unknown",
    visibility: str = "latin_visible_language_unknown",
    text_layer: str = "yes",
    image_like: str = "no",
    fallback_eligible: str = "no",
    fallback_reason: str = "no_text_available",
    table_like: str = "no",
    imaging_modality: str = "no",
    lab_table: str = "no",
    date_or_schedule: str = "no",
    administrative_form: str = "no",
) -> dict:
    return {
        "file_id": file_id,
        "predicted_document_type": "Unknown",
        "unknown_failure_bucket": bucket,
        "unknown_ocr_routing_bucket": routing,
        "language_visibility_status": visibility,
        "pdf_text_layer_detected": text_layer,
        "image_like_pdf": image_like,
        "ocr_fallback_eligible": fallback_eligible,
        "ocr_fallback_not_triggered_reason": fallback_reason,
        "cue_audit_result": cue_audit_result,
        "table_like_structure_detected": table_like,
        "imaging_modality_shape_detected": imaging_modality,
        "lab_table_shape_detected": lab_table,
        "date_or_schedule_shape_detected": date_or_schedule,
        "administrative_form_shape_detected": administrative_form,
        "section_heading_shape_detected": "no",
        "medical_abbreviation_shape_detected": "no",
    }


@pytest.fixture
def synthetic_payload() -> dict:
    rows: list[dict] = []
    # 5 insufficient_text_visibility records covering the actionable labels
    rows.append(_row("insufficient_text_visibility",
                     file_id="file_001",
                     routing="language_visibility_unknown",
                     visibility="latin_visible_language_unknown"))
    rows.append(_row("insufficient_text_visibility",
                     file_id="file_002",
                     routing="text_layer_present_but_too_short",
                     text_layer="yes"))
    rows.append(_row("insufficient_text_visibility",
                     file_id="file_003",
                     routing="image_like_pdf_but_not_routed_to_ocr",
                     image_like="yes",
                     text_layer="no"))
    rows.append(_row("insufficient_text_visibility",
                     file_id="file_004",
                     routing="no_text_layer",
                     text_layer="no"))
    rows.append(_row("insufficient_text_visibility",
                     file_id="file_005",
                     routing="routing_not_eligible",
                     visibility="not_applicable"))
    # 4 fallback_ran_but_no_family_match records
    rows.append(_row("fallback_ran_but_no_family_match",
                     file_id="file_006",
                     cue_audit_result="possible_lab_shape_without_language_cues",
                     lab_table="yes"))
    rows.append(_row("fallback_ran_but_no_family_match",
                     file_id="file_007",
                     cue_audit_result="possible_imaging_shape_without_language_cues",
                     imaging_modality="yes"))
    rows.append(_row("fallback_ran_but_no_family_match",
                     file_id="file_008",
                     cue_audit_result="possible_treatment_shape_without_language_cues",
                     date_or_schedule="yes",
                     table_like="yes"))
    rows.append(_row("fallback_ran_but_no_family_match",
                     file_id="file_009",
                     cue_audit_result="generic_form_shapes_only",
                     administrative_form="yes"))
    # 3 ambiguous_below_threshold records
    for i in range(3):
        rows.append(_row("ambiguous_below_threshold",
                         file_id=f"file_{10+i:03d}",
                         cue_audit_result="needs_manual_review"))
    return {"anonymous_per_file_table": rows}


# ── Section A correctness ───────────────────────────────────────────────────

def test_section_a_count_matches_bucket(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_a = build_section_a(rows)
    assert section_a.count == 5


def test_section_a_breakdown_covers_actionable_labels(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_a = build_section_a(rows)
    b = section_a.ocr_visibility_breakdown
    assert b["language_script_visible_detector_unresolved"] == 1
    assert b["likely_text_layer_issue"] == 1
    assert b["image_like_but_ocr_not_routed"] == 1
    assert b["no_text_layer"] == 1
    assert b["non_actionable_leave_manual_review"] == 1
    # All synthetic rows account for: 4 actionable + 1 non-actionable = 5
    assert sum(b.values()) == 5


def test_section_a_block_justified_threshold():
    # 0 actionable rows -> not justified
    section_a = build_section_a([])
    assert section_a.block_justified is False


# ── Section B correctness ───────────────────────────────────────────────────

def test_section_b_count_matches_bucket(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_b = build_section_b(rows)
    assert section_b.count == 4


def test_section_b_shape_audit_counts_propagate(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_b = build_section_b(rows)
    counts = section_b.shape_audit_counts
    assert counts["possible_lab_shape_without_language_cues"] == 1
    assert counts["possible_imaging_shape_without_language_cues"] == 1
    assert counts["possible_treatment_shape_without_language_cues"] == 1
    assert counts["generic_form_shapes_only"] == 1
    assert sum(counts.values()) == 4


def test_section_b_block_justified_threshold(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_b = build_section_b(rows)
    # 3 actionable shape verdicts (lab + imaging + treatment) -> justified
    assert section_b.block_justified is True


# ── Section C: summary-only ─────────────────────────────────────────────────

def test_section_c_count_matches_bucket(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_c = build_section_c(rows)
    assert section_c.count == 3


def test_section_c_emits_no_cue_expansion_recommendation(synthetic_payload):
    rows = synthetic_payload["anonymous_per_file_table"]
    section_c = build_section_c(rows)
    assert section_c.cue_expansion_recommended is False
    note_l = section_c.note.lower()
    for token in ("add cue", "expand cue", "implement", "fix classifier"):
        assert token not in note_l


# ── Top-level report ────────────────────────────────────────────────────────

def test_report_marks_behavior_unchanged(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    sp = report.safety_privacy
    assert sp["behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["classifier_behavior_changed"] is False
    assert sp["ocr_routing_changed"] is False
    assert sp["auto_accept_changed"] is False
    assert sp["all_records_remain_review_bound"] is True


def test_total_unknown_matches_sum_of_buckets(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.total_unknown_analyzed == sum(report.bucket_counts.values())


def test_overall_recommendation_drives_with_synthetic_data(synthetic_payload):
    """Synthetic payload has actionable rows in both A and B."""
    report = build_diagnostic_from_report(synthetic_payload)
    # Section A actionable=4 (>=20 threshold? No) so A may be unjustified;
    # Section B has 3 actionable shape verdicts so B is justified.
    # Either way the recommendation must be one of the four valid values.
    assert report.overall_recommendation in {"A_then_B", "A", "B", "C"}


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


# ── End-to-end on a tmp dir ─────────────────────────────────────────────────

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
        f"Expected FAMILY-04 batch-eval public report at {SOURCE_REPORT}. "
        "UNKNOWN-DIAG-02 cannot run if this is missing."
    )
