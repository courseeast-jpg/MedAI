"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A.

These tests verify the aggregate-only text-layer diagnostic:

* targets ONLY the 21 ``likely_text_layer_issue`` records,
* excludes every other pool from its bucket counts,
* emits anonymized ``file_NNN`` IDs only,
* emits no raw filenames / no raw OCR or document text / no private paths,
* records the safety invariants (``behavior_changed=False`` etc.),
* recommends a future diagnostic/spec block (recommendation_code='A') and
  does NOT recommend cue expansion as the primary next step,
* includes the required progress percentages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_13a_text_layer_diagnostic import (
    FUTURE_LEVER_BUCKETS,
    ROOT_CAUSE_BUCKETS,
    STRUCTURAL_SHAPE_BUCKETS,
    build_report,
    classify_text_layer_records,
    decide_future_block,
    excludes_other_pools_audit,
    render_markdown,
    render_short_summary,
)


# ── Synthetic safe inputs ──────────────────────────────────────────────────


def _synthetic_text_layer_subset() -> dict:
    """Match the DIAG-03 aggregate shape exactly, with 21 records."""
    return {
        "name": "likely_text_layer_issue",
        "count": 21,
        "diag02_label": "likely_text_layer_issue",
        "raw_signal_counts": {
            "alphabetic_content_bucket_counts": {"high": 21},
            "image_like_pdf_counts": {"no": 21},
            "native_text_length_bucket_counts": {
                "none": 10,
                "short": 8,
                "tiny": 3,
            },
            "pdf_text_layer_detected_counts": {"yes": 21},
            "table_like_structure_detected_counts": {"no": 11, "yes": 10},
        },
        "root_cause_counts": {
            "image_like_with_partial_text": 0,
            "leave_manual_review": 0,
            "no_safe_text_visibility_metadata": 0,
            "table_structure_visible_but_text_insufficient": 10,
            "text_layer_present_but_low_signal": 0,
            "text_layer_too_short": 11,
        },
    }


def _synthetic_diag03_payload() -> dict:
    """Synthetic DIAG-03 envelope including the non-text-layer pool."""
    return {
        "subsets": {
            "likely_text_layer_issue": _synthetic_text_layer_subset(),
            "no_text_layer": {"count": 11},
        }
    }


# ── classify_text_layer_records ────────────────────────────────────────────


def test_targets_only_21_text_layer_records():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert c["total_text_layer_records_analyzed"] == 21


def test_root_cause_counts_match_diag03_aggregate():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    rc = c["root_cause_candidate_counts"]
    assert rc["text_layer_too_short"] == 11
    assert rc["table_structure_visible_but_text_insufficient"] == 10
    assert rc["text_layer_present_but_low_signal"] == 0
    assert rc["image_like_with_partial_text"] == 0
    assert rc["no_safe_text_visibility_metadata"] == 0
    assert rc["leave_manual_review"] == 0


def test_root_cause_buckets_sum_to_21():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert sum(c["root_cause_candidate_counts"].values()) == 21


def test_all_root_cause_buckets_present():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert set(c["root_cause_candidate_counts"].keys()) == set(
        ROOT_CAUSE_BUCKETS
    )


def test_structural_shape_buckets_present_and_named():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert set(c["structural_shape_counts"].keys()) == set(
        STRUCTURAL_SHAPE_BUCKETS
    )


def test_structural_shape_table_like_split():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    ss = c["structural_shape_counts"]
    assert ss["table_like_structure_visible"] == 10
    assert ss["no_known_shape_visible"] == 11
    for b in [
        "row_or_column_pattern_visible",
        "section_heading_shape_visible",
        "lab_or_result_shape_possible",
        "treatment_or_schedule_shape_possible",
        "administrative_or_form_shape_possible",
    ]:
        assert ss[b] == 0


def test_structural_shape_table_like_split_sums_to_21():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert sum(c["structural_shape_counts"].values()) == 21


def test_future_lever_buckets_present_and_named():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert set(c["future_lever_counts"].keys()) == set(FUTURE_LEVER_BUCKETS)


def test_future_lever_text_layer_extraction_diagnostic_covers_all_21():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert (
        c["future_lever_counts"]["candidate_text_layer_extraction_diagnostic"]
        == 21
    )


def test_future_lever_subtrack_pdf_text_quality_audit_matches_too_short():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert (
        c["future_lever_counts"][
            "candidate_pdf_text_extraction_quality_audit"
        ]
        == 11
    )


def test_future_lever_subtrack_layout_table_audit_matches_table_insufficient():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert (
        c["future_lever_counts"]["candidate_layout_table_extraction_audit"]
        == 10
    )


def test_future_lever_manual_review_and_insufficient_are_zero():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    assert c["future_lever_counts"]["candidate_manual_review_only"] == 0
    assert (
        c["future_lever_counts"]["insufficient_metadata_for_next_action"] == 0
    )


def test_anonymous_ids_are_file_nnn_only():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    ids = c["anonymous_ids_used"]
    assert len(ids) == 21
    assert ids[0] == "file_001"
    assert ids[-1] == "file_021"
    for ident in ids:
        assert re.fullmatch(r"file_\d{3}", ident), ident


# ── excludes_other_pools_audit ────────────────────────────────────────────


def test_exclusion_audit_excludes_every_non_text_layer_pool():
    audit = excludes_other_pools_audit(_synthetic_diag03_payload())
    assert audit["likely_text_layer_issue_count"] == 21
    assert audit["no_text_layer_count"] == 11
    for key in [
        "non_text_layer_pools_excluded_from_diag13a_buckets",
        "numeric_table_safe_default_pool_excluded",
        "language_propagation_pool_excluded",
        "latin_abbreviation_pool_excluded",
        "fallback_ran_but_no_family_match_pool_excluded",
        "table_header_special_case_pool_excluded",
        "ambiguous_below_threshold_pool_excluded",
    ]:
        assert audit[key] is True


# ── decide_future_block ────────────────────────────────────────────────────


def test_decision_is_A_text_layer_extraction_diagnostic_spec():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    d = decide_future_block(c)
    assert d["future_block_justified"] is True
    assert d["recommendation_code"] == "A"
    assert d["recommendation_name"] == "text_layer_extraction_diagnostic_spec"
    assert d["cue_expansion_recommended_as_primary_next_step"] is False


def test_decision_letter_options_complete():
    c = classify_text_layer_records(_synthetic_text_layer_subset())
    d = decide_future_block(c)
    opts = d["recommendation_letter_options"]
    assert set(opts.keys()) == {"A", "B", "C", "D", "E"}
    assert "cue" not in " ".join(opts.values()).lower()


def test_decision_falls_back_to_E_when_total_zero():
    empty = {
        "name": "likely_text_layer_issue",
        "count": 0,
        "raw_signal_counts": {
            "table_like_structure_detected_counts": {"yes": 0, "no": 0}
        },
        "root_cause_counts": {b: 0 for b in ROOT_CAUSE_BUCKETS},
    }
    c = classify_text_layer_records(empty)
    d = decide_future_block(c)
    assert d["recommendation_code"] == "E"


def test_decision_falls_back_to_D_when_no_actionable_root_cause():
    inert = {
        "name": "likely_text_layer_issue",
        "count": 5,
        "raw_signal_counts": {
            "table_like_structure_detected_counts": {"yes": 0, "no": 5}
        },
        "root_cause_counts": {b: 0 for b in ROOT_CAUSE_BUCKETS}
        | {"no_safe_text_visibility_metadata": 5},
    }
    c = classify_text_layer_records(inert)
    d = decide_future_block(c)
    assert d["recommendation_code"] == "D"


# ── build_report ───────────────────────────────────────────────────────────


def test_build_report_top_level_invariants():
    p = build_report()
    assert p["conclusion"] == (
        "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_ready"
    )
    assert p["block_id"] == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A"
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["park_20_parking_commit_short"] == "3e46461"
    assert p["diag_13_preflight_commit_short"] == "01c2b69"


def test_build_report_safety_flags_are_false():
    p = build_report()
    for flag in [
        "behavior_changed",
        "external_api_used",
        "cue_expansion_recommended",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
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


def test_build_report_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 97
    assert before["residual_unknown_reduction_track_remaining_pct"] == 3
    assert before["whole_medai_project_done_pct"] == 86
    assert before["whole_medai_project_remaining_pct"] == 14
    assert after["residual_unknown_reduction_track_done_pct"] == 98
    assert after["residual_unknown_reduction_track_remaining_pct"] == 2
    assert after["whole_medai_project_done_pct"] == 87
    assert after["whole_medai_project_remaining_pct"] == 13


def test_build_report_includes_tag_caveat():
    p = build_report()
    caveat = p["remote_tag_caveat"]
    assert "3e46461" in caveat
    assert "403" in caveat
    assert "tag" in caveat.lower()


def test_build_report_includes_total_21():
    p = build_report()
    assert p["total_text_layer_records_analyzed"] == 21


def test_build_report_recommendation_is_A():
    p = build_report()
    assert p["next_block_recommendation"]["recommended_letter"] == "A"
    assert (
        "cue" not in p["next_block_recommendation"]["rationale_summary"].lower()
        or "cue expansion" in p["next_block_recommendation"][
            "rationale_summary"
        ].lower()
    )


def test_build_report_does_not_emit_cue_expansion_as_primary():
    p = build_report()
    d = p["future_block_decision"]
    assert d["cue_expansion_recommended_as_primary_next_step"] is False
    assert p["cue_expansion_recommended"] is False


# ── render outputs ─────────────────────────────────────────────────────────


def test_render_markdown_mentions_buckets_and_counts():
    p = build_report()
    md = render_markdown(p)
    assert "Root-cause candidate counts" in md
    assert "Structural-shape counts" in md
    assert "Future-lever counts" in md
    for b in ROOT_CAUSE_BUCKETS + STRUCTURAL_SHAPE_BUCKETS + FUTURE_LEVER_BUCKETS:
        assert b in md, b


def test_render_short_summary_includes_progress_and_decision():
    p = build_report()
    short = render_short_summary(p)
    assert "97%" in short and "3%" in short
    assert "98%" in short and "2%" in short
    assert "A" in short


# ── no raw filenames / OCR text / private paths in rendered output ────────


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


def test_anonymous_ids_in_json_are_file_nnn():
    p = build_report()
    s = json.dumps(p)
    for ident in p["anonymous_ids_used"]:
        assert re.fullmatch(r"file_\d{3}", ident)
        assert ident in s


# ── on-disk artifacts ─────────────────────────────────────────────────────

REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_13a_text_layer_diagnostic"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_13A_TEXT_LAYER_DIAGNOSTIC.md",
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
    rc = payload["root_cause_candidate_counts"]
    assert rc["text_layer_too_short"] == 11
    assert rc["table_structure_visible_but_text_insufficient"] == 10


def test_written_json_records_no_runtime_behavior(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["behavior_changed"] is False
    assert payload["implementation_started"] is False
    assert payload["runtime_helper_added"] is False
    assert payload["operator_ui_surface_added"] is False
    assert payload["ocr_routing_changed"] is False
    assert payload["ocr_engine_behavior_changed"] is False
    assert payload["classifier_behavior_changed"] is False
    assert payload["cue_expansion_recommended"] is False


def test_written_json_lists_three_source_blocks(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    srcs = payload["source_reports_used"]
    assert len(srcs) == 3
    joined = "\n".join(srcs)
    assert "DIAG-02" in joined
    assert "DIAG-03" in joined
    assert "DIAG-13-PREFLIGHT" in joined


def test_written_md_contains_progress_table(written_files):
    md = written_files["md"].read_text(encoding="utf-8")
    assert "97%" in md and "3%" in md
    assert "98%" in md and "2%" in md
    assert "86%" in md and "14%" in md
    assert "87%" in md and "13%" in md


def test_written_summary_contains_progress(written_files):
    body = written_files["summary"].read_text(encoding="utf-8")
    assert "97%" in body and "3%" in body
    assert "98%" in body and "2%" in body


def test_no_diag_13a_runtime_helper_imported_into_app():
    app_main = (
        Path(__file__).resolve().parents[1] / "app" / "main.py"
    )
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert (
        "run_medai_doc_type_unknown_diag_13a" not in body
    ), "DIAG-13A script must not be imported into runtime"
    assert (
        "render_plan_for_text_layer" not in body
    ), "DIAG-13A must not add an operator UI surface"


def test_no_clinical_knowledge_helper_for_diag_13a():
    ck_dir = (
        Path(__file__).resolve().parents[1] / "clinical_knowledge"
    )
    for path in ck_dir.rglob("*.py"):
        body = path.read_text(encoding="utf-8", errors="ignore")
        assert "MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A" not in body, (
            f"DIAG-13A added a runtime helper at {path}, but it must remain "
            "evaluation-only."
        )
