"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-16.

Verifies the specification-only, evaluation-only, aggregate-only unified
PDF text + layout/table extraction quality spec:

* covers exactly 21 records (11 Sub-track A + 10 Sub-track B),
* names every required spec section,
* records every required safety/invariant flag,
* emits no raw filenames / no raw OCR or document text / no private paths,
* keeps PARK-20 tags untouched and all records review-bound,
* explicitly declines cue expansion.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec import (
    CUE_EXPANSION_DECISION,
    FUTURE_ACCEPTANCE_CRITERIA,
    NEXT_BLOCK_RECOMMENDATION_NAME,
    NON_GOALS,
    PHASE_ID,
    PRIVACY_AND_SAFETY_GATES,
    REQUIRED_REGRESSION_TESTS,
    REVIEW_BOUND_INVARIANTS,
    ROLLBACK_CRITERIA,
    UNIFIED_PROBLEM_STATEMENT,
    build_report,
    evidence_summary,
    render_markdown,
    render_short_summary,
)


def _synthetic_diag15_payload() -> dict:
    return {
        "phase_id": "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15",
        "total_records_analyzed": 11,
        "pdf_text_extraction_quality_bucket_counts": {
            "text_layer_present_but_extracted_text_too_short": 11,
            "extraction_length_below_family_classifier_minimum": 11,
            "alphabetic_visibility_too_low": 0,
            "numeric_or_table_heavy_but_text_sparse": 0,
            "pdf_text_layer_detected_but_semantically_insufficient": 11,
            "possible_encoding_or_glyph_extraction_issue": 0,
            "possible_scanned_or_image_dominant_pdf_mislabeled_as_text_layer": 0,
            "metadata_sufficient_for_future_extraction_audit": 11,
            "insufficient_safe_metadata": 0,
        },
    }


def _synthetic_diag15b_payload() -> dict:
    return {
        "phase_id": "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B",
        "total_records_analyzed": 10,
        "layout_table_extraction_bucket_counts": {
            "table_structure_visible_but_text_insufficient": 10,
            "layout_structure_visible_but_family_classifier_input_sparse": 10,
            "row_or_column_structure_likely_lost": 0,
            "table_header_or_label_context_insufficient": 10,
            "numeric_or_grid_like_content_without_family_cues": 10,
            "possible_multi_column_or_fragmented_text_order_issue": 0,
            "possible_pdf_table_extraction_gap": 10,
            "metadata_sufficient_for_future_layout_audit": 10,
            "insufficient_safe_metadata": 0,
        },
    }


# ── evidence_summary ──────────────────────────────────────────────────────


def test_evidence_summary_combined_total_is_21():
    e = evidence_summary(_synthetic_diag15_payload(), _synthetic_diag15b_payload())
    assert e["combined_total_records"] == 21
    assert e["subtrack_a"]["total_records"] == 11
    assert e["subtrack_b"]["total_records"] == 10


def test_evidence_summary_carries_bucket_dicts():
    e = evidence_summary(_synthetic_diag15_payload(), _synthetic_diag15b_payload())
    assert (
        e["subtrack_a"]["bucket_counts"][
            "text_layer_present_but_extracted_text_too_short"
        ]
        == 11
    )
    assert (
        e["subtrack_b"]["bucket_counts"][
            "table_structure_visible_but_text_insufficient"
        ]
        == 10
    )


# ── build_report identity and counts ──────────────────────────────────────


def test_build_report_identity():
    p = build_report()
    assert p["phase_id"] == PHASE_ID == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-16"
    assert p["mode"] == "specification_only"
    assert p["evaluation_only"] is True
    assert p["aggregate_only"] is True
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["diag_14_commit_short"] == "832a5fe"
    assert p["diag_15_commit_short"] == "3e57ba7"
    assert p["diag_15b_commit_short"] == "2e9b53b"
    assert p["park_20_parking_commit_short"] == "3e46461"


def test_build_report_record_counts():
    p = build_report()
    assert p["total_records_covered"] == 21
    assert p["subtrack_a_records"] == 11
    assert p["subtrack_b_records"] == 10


def test_build_report_inputs_named_explicitly():
    p = build_report()
    assert p["inputs"] == [
        "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15",
        "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B",
    ]


def test_build_report_lists_three_source_blocks():
    p = build_report()
    srcs = p["source_reports_used"]
    assert len(srcs) == 3
    joined = "\n".join(srcs)
    assert "DIAG-14" in joined
    assert "DIAG-15" in joined
    assert "DIAG-15B" in joined


# ── safety / invariant flags ──────────────────────────────────────────────


def test_required_safety_flags_are_false():
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
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
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


def test_extra_safety_flags_false():
    p = build_report()
    for flag in [
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
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


def test_invariant_counts():
    p = build_report()
    assert p["accepted_count"] == 0
    assert p["auto_accept_allowed_count"] == 0
    assert p["external_api_used_count"] == 0
    assert p["all_records_review_bound"] is True


def test_park_20_tag_status_mentions_origin():
    p = build_report()
    s = p["park_20_tag_status"]
    assert "PARK-20" in s
    assert "3e46461" in s
    assert "tag" in s.lower()
    assert "origin" in s.lower()


# ── all 10 spec sections present and named ────────────────────────────────


def test_all_ten_spec_sections_present():
    p = build_report()
    sec = p["spec_sections"]
    expected_keys = {
        "1_evidence_summary",
        "2_unified_problem_statement",
        "3_non_goals",
        "4_future_implementation_acceptance_criteria",
        "5_rollback_criteria",
        "6_privacy_and_safety_gates",
        "7_required_regression_tests",
        "8_review_bound_invariants",
        "9_cue_expansion_decision",
        "10_recommended_next_block",
    }
    assert set(sec.keys()) == expected_keys


def test_evidence_summary_section_matches_inputs():
    p = build_report()
    ev = p["spec_sections"]["1_evidence_summary"]
    assert ev["subtrack_a"]["total_records"] == 11
    assert ev["subtrack_b"]["total_records"] == 10
    assert ev["combined_total_records"] == 21


def test_problem_statement_mentions_21_and_subtrack_split():
    p = build_report()
    text = p["spec_sections"]["2_unified_problem_statement"]
    assert "21" in text
    assert "11" in text
    assert "10" in text
    assert "Sub-track A" in text or "sub-track A" in text
    assert "Sub-track B" in text or "sub-track B" in text


def test_non_goals_carry_required_no_change_items():
    p = build_report()
    items = p["spec_sections"]["3_non_goals"]
    joined = "\n".join(items)
    for needle in [
        "OCR routing",
        "OCR engine",
        "PDF text-extraction",
        "layout/table extraction",
        "language detector",
        "classifier behavior",
        "thresholds",
        "cue packs",
        "lab values",
        "medications",
        "DDI",
        "abbreviations",
        "auto-accept",
        "external API",
        "PARK-20",
        "B07",
        "ROUTE-FIX",
    ]:
        assert needle in joined, needle


def test_future_acceptance_criteria_complete():
    p = build_report()
    items = p["spec_sections"]["4_future_implementation_acceptance_criteria"]
    joined = "\n".join(items)
    for needle in [
        "opt-in or env-gated",
        "preserve review-bound status",
        "MUST NOT auto-accept",
        "MUST NOT parse clinical values",
        "privacy-safe aggregate",
        "accepted_count = 0",
        "auto_accept_allowed_count = 0",
        "external_api_used_count = 0",
        "DIAG-01..16",
        "CKA MVP",
        "B07",
        "ROUTE-FIX",
        "UI ops",
        "UI boot",
        "PARK-20",
        "file_NNN",
    ]:
        assert needle in joined, needle


def test_rollback_criteria_complete():
    p = build_report()
    items = p["spec_sections"]["5_rollback_criteria"]
    joined = "\n".join(items)
    assert "env-var flip" in joined
    assert "regress" in joined.lower()
    assert "PARK-20" in joined
    assert "3e46461" in joined


def test_privacy_and_safety_gates_complete():
    p = build_report()
    items = p["spec_sections"]["6_privacy_and_safety_gates"]
    joined = "\n".join(items)
    assert "check_public_report_payload" in joined
    assert "raw OCR text" in joined
    assert "file_NNN" in joined
    assert "tag" in joined.lower()


def test_required_regression_tests_complete():
    p = build_report()
    items = p["spec_sections"]["7_required_regression_tests"]
    joined = "\n".join(items)
    for needle in [
        "DIAG-15",
        "DIAG-15B",
        "DIAG-01",
        "DIAG-16",
        "document-type eval",
        "CKA MVP",
        "B07",
        "ROUTE-FIX",
        "UI ops",
        "UI boot",
        "privacy",
        "staged safety check",
    ]:
        assert needle in joined, needle


def test_review_bound_invariants_complete():
    p = build_report()
    items = p["spec_sections"]["8_review_bound_invariants"]
    joined = "\n".join(items)
    assert "Needs review" in joined
    assert "accepted_count" in joined
    assert "external_api_used_count" in joined
    assert "data-layer document type" in joined


def test_cue_expansion_decision_explicit():
    p = build_report()
    text = p["spec_sections"]["9_cue_expansion_decision"]
    assert "Cue expansion" in text
    assert "explicitly NOT recommended" in text
    assert p["cue_expansion_recommended"] is False


def test_recommended_next_block_is_park_snapshot():
    p = build_report()
    text = p["spec_sections"]["10_recommended_next_block"]
    assert "PARK-21" in text or "parking snapshot" in text.lower()
    rec = p["next_block_recommendation"]
    assert rec["must_remain_evaluation_only"] is True
    assert rec["must_remain_aggregate_only"] is True
    assert rec["must_not_propose_cue_expansion_as_primary_step"] is True
    assert rec["must_not_change_extraction_behavior_in_first_pass"] is True


# ── progress percentages ──────────────────────────────────────────────────


def test_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 99.8
    assert before["residual_unknown_reduction_track_remaining_pct"] == 0.2
    assert before["whole_medai_project_done_pct"] == 89.5
    assert before["whole_medai_project_remaining_pct"] == 10.5
    assert after["residual_unknown_reduction_track_done_pct"] == 99.9
    assert after["residual_unknown_reduction_track_remaining_pct"] == 0.1
    assert after["whole_medai_project_done_pct"] == 90
    assert after["whole_medai_project_remaining_pct"] == 10


# ── rendered outputs ──────────────────────────────────────────────────────


def test_rendered_markdown_has_ten_spec_section_headers():
    p = build_report()
    md = render_markdown(p)
    for n in range(1, 11):
        assert f"## {n}." in md, f"section {n}"


def test_rendered_short_summary_includes_split_and_progress():
    p = build_report()
    short = render_short_summary(p)
    assert "21" in short and "11" in short and "10" in short
    assert "99.8" in short and "99.9" in short


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


# ── module-level constants stay aligned ───────────────────────────────────


def test_non_goals_module_constant_avoids_PERSON_regex_trigger():
    # The PERSON regex (IGNORECASE) matches "DO <Word>" because "DO" is in
    # the medical-title alternation. Make sure NON_GOALS items don't start
    # with "do not <Word>" — the privacy check would flag them.
    for item in NON_GOALS:
        assert not re.match(r"(?i)do\s+not\s+\w", item), item


def test_future_acceptance_criteria_uses_must_phrasing():
    joined = "\n".join(FUTURE_ACCEPTANCE_CRITERIA)
    assert "MUST" in joined


def test_rollback_criteria_present():
    assert len(ROLLBACK_CRITERIA) >= 5


def test_privacy_and_safety_gates_present():
    assert len(PRIVACY_AND_SAFETY_GATES) >= 5


def test_required_regression_tests_present():
    assert len(REQUIRED_REGRESSION_TESTS) >= 5


def test_review_bound_invariants_present():
    assert len(REVIEW_BOUND_INVARIANTS) >= 5


def test_cue_expansion_decision_string_present():
    assert "explicitly NOT recommended" in CUE_EXPANSION_DECISION


def test_next_block_recommendation_constant_mentions_park_snapshot():
    assert "PARK-21" in NEXT_BLOCK_RECOMMENDATION_NAME or "parking" in NEXT_BLOCK_RECOMMENDATION_NAME.lower()


def test_unified_problem_statement_constant_mentions_21_records():
    assert "21" in UNIFIED_PROBLEM_STATEMENT


# ── on-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_16_PDF_TEXT_LAYOUT_QUALITY_SPEC.md",
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
    assert payload["total_records_covered"] == 21
    assert payload["subtrack_a_records"] == 11
    assert payload["subtrack_b_records"] == 10
    assert payload["phase_id"] == PHASE_ID
    assert payload["mode"] == "specification_only"
    assert payload["evaluation_only"] is True
    assert payload["aggregate_only"] is True


def test_written_json_safety_flags_false(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    for flag in [
        "behavior_changed",
        "implementation_started",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "cue_expansion_recommended",
        "park_20_tags_touched",
    ]:
        assert payload[flag] is False, flag


# ── DIAG-16 must NOT add a runtime helper or UI surface ───────────────────


def test_no_diag_16_runtime_helper_imported_into_app():
    app_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert "run_medai_doc_type_unknown_diag_16" not in body
    assert "render_plan_for_pdf_text_layout_quality" not in body


def test_no_clinical_knowledge_helper_for_diag_16():
    ck_dir = Path(__file__).resolve().parents[1] / "clinical_knowledge"
    for path in ck_dir.rglob("*.py"):
        body = path.read_text(encoding="utf-8", errors="ignore")
        assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-16" not in body, (
            f"DIAG-16 added a runtime helper at {path}, but it must remain "
            "spec-only."
        )
