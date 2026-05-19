"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-14.

Verifies the aggregate-only text-layer extraction specification:

* targets only the 21 text-layer records,
* splits them 11 + 10 into the two sub-tracks,
* renders the positive signatures and exclusion rules,
* records all safety invariants false,
* emits no raw filenames / no raw OCR or document text / no private paths.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_14_text_layer_extraction_spec import (
    EXCLUSION_RULES,
    FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA,
    FUTURE_VALIDATION_REQUIREMENTS,
    PROPOSED_FUTURE_DIAGNOSTIC_BEHAVIOR,
    ROLLBACK_SAFETY_BOUNDARIES,
    SUB_TRACK_A_NAME,
    SUB_TRACK_A_POSITIVE_SIGNATURE,
    SUB_TRACK_B_NAME,
    SUB_TRACK_B_POSITIVE_SIGNATURE,
    build_report,
    exclusion_audit,
    render_markdown,
    render_short_summary,
    split_text_layer_pool,
)


# ── Synthetic safe inputs ──────────────────────────────────────────────────


def _synthetic_diag13a_payload() -> dict:
    return {
        "total_text_layer_records_analyzed": 21,
        "root_cause_candidate_counts": {
            "text_layer_too_short": 11,
            "table_structure_visible_but_text_insufficient": 10,
            "text_layer_present_but_low_signal": 0,
            "image_like_with_partial_text": 0,
            "no_safe_text_visibility_metadata": 0,
            "leave_manual_review": 0,
        },
    }


# ── split_text_layer_pool ─────────────────────────────────────────────────


def test_split_targets_only_21_records():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    assert s["total_text_layer_records_analyzed"] == 21


def test_split_pdf_audit_pool_is_11():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    assert s["sub_tracks"][SUB_TRACK_A_NAME]["pool_count"] == 11


def test_split_layout_audit_pool_is_10():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    assert s["sub_tracks"][SUB_TRACK_B_NAME]["pool_count"] == 10


def test_split_sum_equals_total():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    assert s["split_sum_equals_total"] is True
    assert (
        s["sub_tracks"][SUB_TRACK_A_NAME]["pool_count"]
        + s["sub_tracks"][SUB_TRACK_B_NAME]["pool_count"]
        == s["total_text_layer_records_analyzed"]
    )


def test_anonymous_ids_are_file_nnn_only():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    for sub in s["sub_tracks"].values():
        for ident in sub["anonymous_ids"]:
            assert re.fullmatch(r"file_\d{3}", ident), ident


def test_anonymous_id_ranges_do_not_overlap():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    a_ids = set(s["sub_tracks"][SUB_TRACK_A_NAME]["anonymous_ids"])
    b_ids = set(s["sub_tracks"][SUB_TRACK_B_NAME]["anonymous_ids"])
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) + len(b_ids) == 21


def test_positive_signature_for_sub_track_a():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    sig = s["sub_tracks"][SUB_TRACK_A_NAME]["positive_signature"]
    assert sig == list(SUB_TRACK_A_POSITIVE_SIGNATURE)
    joined = "\n".join(sig)
    assert "pdf_text_layer_detected = yes" in joined
    assert "image_like_pdf = no" in joined
    assert "text_layer_too_short = yes" in joined
    assert "review-bound" in joined


def test_positive_signature_for_sub_track_b():
    s = split_text_layer_pool(_synthetic_diag13a_payload())
    sig = s["sub_tracks"][SUB_TRACK_B_NAME]["positive_signature"]
    assert sig == list(SUB_TRACK_B_POSITIVE_SIGNATURE)
    joined = "\n".join(sig)
    assert "pdf_text_layer_detected = yes" in joined
    assert "image_like_pdf = no" in joined
    assert "table_structure_visible = yes" in joined
    assert "table_structure_visible_but_text_insufficient = yes" in joined
    assert "review-bound" in joined


# ── exclusion_audit ────────────────────────────────────────────────────────


def test_exclusion_rules_render_completely():
    e = exclusion_audit(_synthetic_diag13a_payload())
    assert e["rules"] == list(EXCLUSION_RULES)
    joined = "\n".join(e["rules"])
    assert "image-like PDFs" in joined
    assert "no-text-layer" in joined
    assert "fallback_ran_but_no_family_match" in joined
    assert "ambiguous_below_threshold" in joined
    assert "numeric-table safe-default" in joined
    assert "language-propagation" in joined
    assert "latin-abbreviation" in joined
    assert "table-header" in joined
    assert "lab-value parsing" in joined
    assert "medication" in joined and "DDI" in joined
    assert "abbreviation parsing or expansion" in joined
    assert "insufficient safe metadata" in joined


def test_exclusion_audit_flags_are_all_true():
    e = exclusion_audit(_synthetic_diag13a_payload())
    for key in [
        "non_text_layer_pools_excluded_from_diag14_buckets",
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


# ── build_report ───────────────────────────────────────────────────────────


def test_build_report_basic_identity():
    p = build_report()
    assert p["conclusion"] == (
        "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_ready"
    )
    assert p["block_id"] == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-14"
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["diag_13a_commit_short"] == "7866ba4"
    assert p["park_20_parking_commit_short"] == "3e46461"


def test_build_report_records_correct_counts():
    p = build_report()
    assert p["total_text_layer_records_analyzed"] == 21
    assert p["pdf_text_extraction_quality_audit_pool_count"] == 11
    assert p["layout_table_extraction_audit_pool_count"] == 10
    assert p["split_sum_equals_total"] is True


def test_build_report_includes_tag_caveat():
    p = build_report()
    cav = p["remote_tag_caveat"]
    assert "3e46461" in cav
    assert "403" in cav
    assert "tag" in cav.lower()


def test_build_report_includes_four_source_blocks():
    p = build_report()
    srcs = p["source_reports_used"]
    assert len(srcs) == 4
    joined = "\n".join(srcs)
    assert "DIAG-02" in joined
    assert "DIAG-03" in joined
    assert "DIAG-13-PREFLIGHT" in joined
    assert "DIAG-13A" in joined


def test_build_report_safety_flags_are_false():
    p = build_report()
    for flag in [
        "behavior_changed",
        "external_api_used",
        "cue_expansion_recommended",
        "extraction_behavior_changed",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_table_extraction_behavior_changed",
        "raw_language_detector_changed",
        "classifier_behavior_changed",
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
        "park_20_tags_touched",
        "source_documents_staged",
        "private_files_staged",
        "raw_ocr_text_in_public_reports",
        "raw_document_text_in_public_reports",
        "raw_filenames_in_public_reports",
        "private_paths_in_public_reports",
        "secrets_in_public_reports",
    ]:
        assert p[flag] is False, flag


def test_build_report_no_extraction_or_runtime_in_this_block():
    p = build_report()
    assert p["no_extraction_behavior_implemented_in_this_block"] is True
    assert p["no_runtime_behavior_changed_in_this_block"] is True


def test_build_report_includes_future_lists_in_full():
    p = build_report()
    assert p["proposed_future_diagnostic_behavior"] == list(
        PROPOSED_FUTURE_DIAGNOSTIC_BEHAVIOR
    )
    assert p["future_implementation_acceptance_criteria"] == list(
        FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA
    )
    assert p["future_validation_requirements"] == list(
        FUTURE_VALIDATION_REQUIREMENTS
    )
    assert p["rollback_safety_boundaries"] == list(
        ROLLBACK_SAFETY_BOUNDARIES
    )


def test_build_report_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 98
    assert before["residual_unknown_reduction_track_remaining_pct"] == 2
    assert before["whole_medai_project_done_pct"] == 87
    assert before["whole_medai_project_remaining_pct"] == 13
    assert after["residual_unknown_reduction_track_done_pct"] == 99
    assert after["residual_unknown_reduction_track_remaining_pct"] == 1
    assert after["whole_medai_project_done_pct"] == 88
    assert after["whole_medai_project_remaining_pct"] == 12


def test_build_report_does_not_recommend_cue_expansion_as_primary():
    p = build_report()
    rec = p["next_block_recommendation"]
    assert rec["must_not_propose_cue_expansion_as_primary_step"] is True
    assert rec["must_not_change_extraction_behavior_in_first_pass"] is True
    assert rec["must_remain_evaluation_only"] is True
    assert rec["must_remain_aggregate_only"] is True


# ── rendered outputs do not leak raw filenames / private paths ────────────


_RAW_FILENAME_EXT_HINTS = (
    re.compile(r"\.pdf\b", re.IGNORECASE),
    re.compile(r"\.docx?\b", re.IGNORECASE),
    re.compile(r"\.xlsx?\b", re.IGNORECASE),
    re.compile(r"\.png\b", re.IGNORECASE),
    re.compile(r"\.jpe?g\b", re.IGNORECASE),
)


def test_rendered_outputs_have_no_raw_filenames():
    p = build_report()
    for blob in (
        render_markdown(p),
        render_short_summary(p),
        json.dumps(p),
    ):
        for rx in _RAW_FILENAME_EXT_HINTS:
            assert not rx.search(blob)


def test_rendered_outputs_have_no_unix_private_paths():
    p = build_report()
    for blob in (
        render_markdown(p),
        render_short_summary(p),
        json.dumps(p),
    ):
        for needle in ("/home/", "/var/", "/etc/", "/usr/", "/tmp/", "/opt/"):
            assert needle not in blob


def test_rendered_markdown_includes_positive_signatures_and_exclusions():
    p = build_report()
    md = render_markdown(p)
    assert "Positive signature — sub-track A" in md
    assert "Positive signature — sub-track B" in md
    assert "Exclusion rules" in md
    for rule in EXCLUSION_RULES:
        assert rule in md
    for item in SUB_TRACK_A_POSITIVE_SIGNATURE:
        assert item in md
    for item in SUB_TRACK_B_POSITIVE_SIGNATURE:
        assert item in md


def test_rendered_markdown_includes_progress_percentages():
    p = build_report()
    md = render_markdown(p)
    assert "98%" in md and "2%" in md
    assert "99%" in md and "1%" in md
    assert "87%" in md and "13%" in md
    assert "88%" in md and "12%" in md


def test_rendered_short_summary_includes_split_and_progress():
    p = build_report()
    body = render_short_summary(p)
    assert "11" in body and "10" in body
    assert "98%" in body and "99%" in body
    assert "extraction_behavior_changed" in body


# ── on-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_14_text_layer_extraction_spec"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_14_TEXT_LAYER_EXTRACTION_SPEC.md",
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
    assert payload["total_text_layer_records_analyzed"] == 21
    assert payload["pdf_text_extraction_quality_audit_pool_count"] == 11
    assert payload["layout_table_extraction_audit_pool_count"] == 10


def test_written_json_safety_flags_false(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["behavior_changed"] is False
    assert payload["implementation_started"] is False
    assert payload["extraction_behavior_changed"] is False
    assert payload["runtime_helper_added"] is False
    assert payload["operator_ui_surface_added"] is False
    assert payload["cue_expansion_recommended"] is False


# ── ensure DIAG-14 added no runtime helper or UI surface ──────────────────


def test_no_diag_14_runtime_helper_imported_into_app():
    app_main = (
        Path(__file__).resolve().parents[1] / "app" / "main.py"
    )
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert (
        "run_medai_doc_type_unknown_diag_14" not in body
    ), "DIAG-14 script must not be imported into runtime"
    assert (
        "render_plan_for_text_layer_extraction" not in body
    ), "DIAG-14 must not add an operator UI surface"


def test_no_clinical_knowledge_helper_for_diag_14():
    ck_dir = (
        Path(__file__).resolve().parents[1] / "clinical_knowledge"
    )
    for path in ck_dir.rglob("*.py"):
        body = path.read_text(encoding="utf-8", errors="ignore")
        assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-14" not in body, (
            f"DIAG-14 added a runtime helper at {path}, but it must remain "
            "evaluation-only."
        )
