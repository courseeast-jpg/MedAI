"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-15.

Verifies the aggregate-only PDF text-extraction quality audit:

* targets only the 11 Sub-track A records (Sub-track B layout/table audit
  excluded; every other Unknown pool excluded),
* renders the controlled-vocabulary buckets,
* records all safety invariants false,
* emits no raw filenames / no raw OCR or document text / no private paths,
* keeps PARK-20 tags untouched and all records review-bound.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit import (
    PDF_TEXT_QUALITY_BUCKETS,
    PHASE_ID,
    SUBTRACK_ID,
    audit_sub_track_a,
    build_report,
    excludes_other_pools_audit,
    render_markdown,
    render_short_summary,
)


def _synthetic_diag14_payload() -> dict:
    return {
        "pdf_text_extraction_quality_audit_pool_count": 11,
        "layout_table_extraction_audit_pool_count": 10,
    }


# ── audit_sub_track_a ─────────────────────────────────────────────────────


def test_targets_only_11_sub_track_a_records():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    assert a["total_records_analyzed"] == 11


def test_pdf_text_quality_buckets_present_and_named():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    assert set(a["pdf_text_extraction_quality_bucket_counts"].keys()) == set(
        PDF_TEXT_QUALITY_BUCKETS
    )


def test_pdf_text_quality_definitional_buckets():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    q = a["pdf_text_extraction_quality_bucket_counts"]
    assert q["text_layer_present_but_extracted_text_too_short"] == 11
    assert q["extraction_length_below_family_classifier_minimum"] == 11
    assert q["pdf_text_layer_detected_but_semantically_insufficient"] == 11
    assert q["metadata_sufficient_for_future_extraction_audit"] == 11


def test_pdf_text_quality_negative_buckets_are_zero():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    q = a["pdf_text_extraction_quality_bucket_counts"]
    assert q["alphabetic_visibility_too_low"] == 0
    assert q["numeric_or_table_heavy_but_text_sparse"] == 0
    assert q["possible_encoding_or_glyph_extraction_issue"] == 0
    assert q["possible_scanned_or_image_dominant_pdf_mislabeled_as_text_layer"] == 0
    assert q["insufficient_safe_metadata"] == 0


def test_anonymous_ids_are_file_nnn_only_and_count_is_11():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    ids = a["anonymous_ids_used"]
    assert len(ids) == 11
    assert ids[0] == "file_001"
    assert ids[-1] == "file_011"
    for ident in ids:
        assert re.fullmatch(r"file_\d{3}", ident), ident


def test_bucket_semantics_is_multi_label():
    a = audit_sub_track_a(_synthetic_diag14_payload())
    assert a["bucket_semantics"] == "multi_label_flags_per_record_pool"


# ── exclusion ─────────────────────────────────────────────────────────────


def test_exclusion_audit_excludes_sub_track_b_and_other_pools():
    e = excludes_other_pools_audit()
    for key in [
        "layout_table_extraction_audit_pool_excluded",
        "non_text_layer_pools_excluded_from_diag15_buckets",
        "numeric_table_safe_default_pool_excluded",
        "language_propagation_pool_excluded",
        "latin_abbreviation_pool_excluded",
        "fallback_ran_but_no_family_match_pool_excluded",
        "table_header_special_case_pool_excluded",
        "ambiguous_below_threshold_pool_excluded",
        "image_like_pdfs_requiring_ocr_routing_excluded",
        "no_text_layer_records_excluded",
        "records_requiring_lab_value_parsing_excluded",
        "records_requiring_medication_or_ddi_parsing_excluded",
        "records_requiring_abbreviation_parsing_or_expansion_excluded",
        "records_with_insufficient_safe_metadata_excluded",
    ]:
        assert e[key] is True, key


# ── build_report ──────────────────────────────────────────────────────────


def test_build_report_identity():
    p = build_report()
    assert p["phase_id"] == PHASE_ID == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15"
    assert p["mode"] == "evaluation_only"
    assert p["aggregate_only"] is True
    assert p["subtrack"] == SUBTRACK_ID == "A_pdf_text_extraction_quality_audit"
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["diag_14_commit_short"] == "832a5fe"
    assert p["park_20_parking_commit_short"] == "3e46461"


def test_build_report_total_records_analyzed():
    p = build_report()
    assert p["total_records_analyzed"] == 11


def test_build_report_safety_flags_required_by_task_spec_are_false():
    p = build_report()
    for flag in [
        "behavior_changed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "runtime_behavior_changed",
        "extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "park_20_tags_touched",
        "unknown_count_changed",
    ]:
        assert p[flag] is False, flag


def test_build_report_extra_safety_flags_false():
    p = build_report()
    for flag in [
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_table_extraction_behavior_changed",
        "raw_language_detector_changed",
        "thresholds_or_scoring_changed",
        "cue_packs_added",
        "lab_values_parsed",
        "medications_dose_frequency_duration_or_ddi_parsed",
        "abbreviations_parsed_or_expanded",
        "clinical_interpretation_added",
        "b07_changed",
        "route_fix_changed",
        "db_schema_changed",
        "command_allowlist_changed",
        "external_api_enabled",
        "source_documents_staged",
        "private_files_staged",
        "raw_ocr_text_in_public_reports",
        "raw_document_text_in_public_reports",
        "raw_filenames_in_public_reports",
        "private_paths_in_public_reports",
        "secrets_in_public_reports",
    ]:
        assert p[flag] is False, flag


def test_build_report_invariant_counts():
    p = build_report()
    assert p["accepted_count"] == 0
    assert p["auto_accept_allowed_count"] == 0
    assert p["external_api_used_count"] == 0
    assert p["all_records_review_bound"] is True


def test_build_report_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 99
    assert before["residual_unknown_reduction_track_remaining_pct"] == 1
    assert before["whole_medai_project_done_pct_lower"] == 88
    assert before["whole_medai_project_done_pct_upper"] == 89
    assert after["residual_unknown_reduction_track_done_pct"] == 99.5
    assert after["residual_unknown_reduction_track_remaining_pct"] == 0.5
    assert after["whole_medai_project_done_pct"] == 89
    assert after["whole_medai_project_remaining_pct"] == 11


def test_build_report_park_20_tag_status_mentions_origin():
    p = build_report()
    s = p["park_20_tag_status"]
    assert "PARK-20" in s
    assert "3e46461" in s
    assert "tag" in s.lower()
    assert "origin" in s.lower()


def test_build_report_recommendation_avoids_cue_expansion():
    p = build_report()
    rec = p["next_block_recommendation"]
    assert rec["must_not_propose_cue_expansion_as_primary_step"] is True
    assert rec["must_remain_evaluation_only"] is True
    assert rec["must_remain_aggregate_only"] is True
    assert rec["must_not_change_extraction_behavior_in_first_pass"] is True
    # check overall flag too
    assert p["cue_expansion_recommended"] is False


# ── rendered outputs ──────────────────────────────────────────────────────


def test_rendered_markdown_mentions_all_buckets():
    p = build_report()
    md = render_markdown(p)
    assert "PDF text-extraction quality bucket counts" in md
    for b in PDF_TEXT_QUALITY_BUCKETS:
        assert b in md, b


def test_rendered_short_summary_includes_progress():
    p = build_report()
    short = render_short_summary(p)
    assert "99%" in short and "1%" in short
    assert "99.5%" in short and "0.5%" in short


_RAW_FILENAME_HINTS = (
    re.compile(r"\.pdf\b", re.IGNORECASE),
    re.compile(r"\.docx?\b", re.IGNORECASE),
    re.compile(r"\.xlsx?\b", re.IGNORECASE),
    re.compile(r"\.png\b", re.IGNORECASE),
    re.compile(r"\.jpe?g\b", re.IGNORECASE),
)


def test_rendered_outputs_have_no_raw_filenames():
    p = build_report()
    for blob in (render_markdown(p), render_short_summary(p), json.dumps(p)):
        for rx in _RAW_FILENAME_HINTS:
            assert not rx.search(blob)


def test_rendered_outputs_have_no_unix_private_paths():
    p = build_report()
    for blob in (render_markdown(p), render_short_summary(p), json.dumps(p)):
        for needle in ("/home/", "/var/", "/etc/", "/usr/", "/tmp/", "/opt/"):
            assert needle not in blob


# ── on-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_15_PDF_TEXT_EXTRACTION_QUALITY_AUDIT.md",
    }


def test_report_files_exist(written_files):
    for p in written_files.values():
        assert p.exists(), p


def test_written_json_passes_public_payload_privacy(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    r = check_public_report_payload(payload)
    assert r.passed, r.leak_examples_redacted


def test_written_md_passes_public_payload_privacy(written_files):
    body = written_files["md"].read_text(encoding="utf-8")
    r = check_public_report_payload(body)
    assert r.passed, r.leak_examples_redacted


def test_written_summary_passes_public_payload_privacy(written_files):
    body = written_files["summary"].read_text(encoding="utf-8")
    r = check_public_report_payload(body)
    assert r.passed, r.leak_examples_redacted


def test_written_json_records_correct_counts(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["total_records_analyzed"] == 11
    assert payload["phase_id"] == PHASE_ID
    assert payload["mode"] == "evaluation_only"
    assert payload["aggregate_only"] is True
    assert payload["subtrack"] == SUBTRACK_ID


def test_written_json_safety_flags_false(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["behavior_changed"] is False
    assert payload["implementation_started"] is False
    assert payload["extraction_behavior_changed"] is False
    assert payload["runtime_helper_added"] is False
    assert payload["operator_ui_surface_added"] is False
    assert payload["cue_expansion_recommended"] is False
    assert payload["park_20_tags_touched"] is False


# ── ensure DIAG-15 added no runtime helper or UI surface ──────────────────


def test_no_diag_15_runtime_helper_imported_into_app():
    app_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert "run_medai_doc_type_unknown_diag_15" not in body
    assert "render_plan_for_pdf_text_quality" not in body
    assert "render_plan_for_pdf_text_extraction" not in body


def test_no_clinical_knowledge_helper_for_diag_15():
    ck_dir = Path(__file__).resolve().parents[1] / "clinical_knowledge"
    for path in ck_dir.rglob("*.py"):
        body = path.read_text(encoding="utf-8", errors="ignore")
        assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15" not in body, (
            f"DIAG-15 added a runtime helper at {path}, but it must remain "
            "evaluation-only."
        )
