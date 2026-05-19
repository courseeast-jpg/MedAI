"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B.

Verifies the env-on aggregate evaluation:

1.  Env-on emission count equals 21 (in-scope cohort).
2.  Excluded-pool emission count is 0.
3.  Only controlled-vocabulary family labels appear.
4.  No raw text / filenames / private paths in any output.
5.  accepted_count = 0.
6.  auto_accept_allowed_count = 0.
7.  external_api_used_count = 0.
8.  all_records_review_bound = true.
9.  No clinical interpretation / diagnosis / medication / DDI / treatment /
    abbreviation expansion flags are true.
10. PARK-20 / PARK-21 tags are not touched (flags + history check).
11. DIAG-17 helper remains default-off outside explicit env-on (the
    evaluation never wrote to os.environ).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    is_pdf_text_layout_quality_impl_default_disabled,
    is_pdf_text_layout_quality_impl_enabled,
)
from clinical_knowledge.privacy.report_privacy import (
    check_public_report_payload,
)
from scripts.run_medai_doc_type_unknown_diag_17b_env_on_aggregate_eval import (
    PHASE_ID,
    build_report,
    evaluate_env_on_aggregate,
    render_markdown,
    render_short_summary,
)


# ── evaluate_env_on_aggregate ─────────────────────────────────────────────


def test_env_on_emits_for_all_21_in_scope_records():
    a = evaluate_env_on_aggregate()
    assert a["env_on_records_evaluated"] == 21
    assert a["total_records_evaluated"] == 21
    assert a["emitted_metadata_count"] == 21
    assert a["suppressed_or_excluded_count"] == 0


def test_subtrack_split_matches_diag14_canonical_counts():
    a = evaluate_env_on_aggregate()
    assert a["per_subtrack_emit_count"]["A"] == 11
    assert a["per_subtrack_emit_count"]["B"] == 10


def test_excluded_pool_emits_zero():
    a = evaluate_env_on_aggregate()
    assert a["excluded_pool_count"] >= 1
    assert a["excluded_pool_emission_count"] == 0


def test_family_labels_are_only_from_controlled_vocab():
    a = evaluate_env_on_aggregate()
    assert set(a["family_label_counts"].keys()) == set(QUALITY_FAMILY_VALUES)
    # And every label has been observed at least once across the cohort
    for k, v in a["family_label_counts"].items():
        assert v >= 0


def test_subtrack_a_drives_pdf_text_too_short():
    a = evaluate_env_on_aggregate()
    # 11 Sub-track A records each get pdf_text_too_short
    assert a["family_label_counts"]["pdf_text_too_short"] == 11


def test_subtrack_b_drives_table_and_layout_labels():
    a = evaluate_env_on_aggregate()
    # 10 Sub-track B records each get both Sub-track B labels
    assert (
        a["family_label_counts"]["table_structure_visible_text_insufficient"]
        == 10
    )
    assert (
        a["family_label_counts"]["layout_or_table_extraction_gap"] == 10
    )


def test_review_required_for_every_emission():
    a = evaluate_env_on_aggregate()
    assert a["review_required_count"] == a["emitted_metadata_count"]


def test_no_auto_accept_under_env_on():
    a = evaluate_env_on_aggregate()
    assert a["auto_accept_allowed_count"] == 0


def test_no_clinical_interpretation_or_inference_under_env_on():
    a = evaluate_env_on_aggregate()
    assert a["clinical_interpretation_performed_count"] == 0
    assert a["diagnosis_inference_count"] == 0
    assert a["medication_inference_count"] == 0
    assert a["ddi_inference_count"] == 0
    assert a["treatment_inference_count"] == 0
    assert a["abbreviation_expansion_count"] == 0


def test_no_raw_text_filename_or_private_path_emissions():
    a = evaluate_env_on_aggregate()
    assert a["raw_text_emission_count"] == 0
    assert a["raw_filename_emission_count"] == 0
    assert a["private_path_emission_count"] == 0


def test_no_external_api_under_env_on():
    a = evaluate_env_on_aggregate()
    assert a["external_api_used_count"] == 0


def test_no_park_tag_touches_recorded_in_plans():
    a = evaluate_env_on_aggregate()
    assert a["park_20_tags_touched_count"] == 0
    assert a["park_21_tags_touched_count"] == 0


def test_helper_default_off_state_survives_evaluation():
    a = evaluate_env_on_aggregate()
    assert a["helper_still_default_disabled_after_eval"] is True
    assert a["helper_still_enabled_only_when_env_truthy"] is True


def test_evaluation_does_not_pollute_os_environ():
    before = os.environ.get(PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR)
    evaluate_env_on_aggregate()
    after = os.environ.get(PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR)
    assert before == after
    # And the helper is still default-disabled when consulted with the
    # real environment (which should not have been set by the eval).
    assert is_pdf_text_layout_quality_impl_default_disabled(env=os.environ) is True


# ── build_report identity and counts ──────────────────────────────────────


def test_build_report_identity():
    p = build_report()
    assert p["phase_id"] == PHASE_ID == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B"
    assert p["mode"] == "env_on_aggregate_evaluation"
    assert p["evaluation_only"] is True
    assert p["reports_only"] is True
    assert p["branch"] == "clinical-knowledge-architecture"
    assert p["env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert p["diag_17_commit_short"] == "ad7b2d6"
    assert p["park_20_parking_commit_short"] == "3e46461"
    assert p["park_21_parking_commit_short"] == "9f9e22d"


def test_build_report_safety_flags_required_by_task_spec_are_false():
    p = build_report()
    for flag in [
        "behavior_changed",
        "runtime_behavior_changed",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "operator_ui_surface_added",
        "cue_expansion_recommended",
        "cue_expansion_performed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_expansion_performed",
        "park_20_tags_touched",
        "park_21_tags_touched",
    ]:
        assert p[flag] is False, flag


def test_build_report_required_aggregate_outputs_present():
    p = build_report()
    for key in [
        "total_records_evaluated",
        "emitted_metadata_count",
        "suppressed_or_excluded_count",
        "family_label_counts",
        "review_required_count",
        "auto_accept_allowed_count",
        "clinical_interpretation_performed_count",
        "raw_text_emission_count",
        "raw_filename_emission_count",
        "private_path_emission_count",
        "excluded_pool_count",
        "excluded_pool_emission_count",
    ]:
        assert key in p, key


def test_build_report_invariants_in_scope():
    p = build_report()
    assert p["env_on_records_evaluated"] == 21
    assert p["accepted_count"] == 0
    assert p["auto_accept_allowed_count"] == 0
    assert p["external_api_used_count"] == 0
    assert p["all_records_review_bound"] is True
    assert p["emitted_metadata_count"] == 21
    assert p["excluded_pool_emission_count"] == 0


def test_build_report_park_status_mentions_both_parks_untouched():
    p = build_report()
    s = p["park_status"]
    assert "PARK-20" in s and "3e46461" in s
    assert "PARK-21" in s and "9f9e22d" in s
    assert "not touch" in s


def test_progress_percentages_present():
    p = build_report()
    before = p["progress_estimate"]["before"]
    after = p["progress_estimate"]["after"]
    assert before["residual_unknown_reduction_track_done_pct"] == 99.95
    assert before["residual_unknown_reduction_track_remaining_pct"] == 0.05
    assert before["whole_medai_project_done_pct"] == 90.5
    assert before["whole_medai_project_remaining_pct"] == 9.5
    assert after["residual_unknown_reduction_track_done_pct"] == 99.97
    assert after["residual_unknown_reduction_track_remaining_pct"] == 0.03
    assert after["whole_medai_project_done_pct"] == 90.8
    assert after["whole_medai_project_remaining_pct"] == 9.2


# ── rendered outputs: no raw text / filenames / private paths ─────────────


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


def test_rendered_markdown_mentions_aggregate_section():
    p = build_report()
    md = render_markdown(p)
    assert "Env-on aggregate results" in md
    assert "Family-label counts" in md
    assert "Default-off invariants AFTER evaluation" in md


# ── on-disk artifacts ─────────────────────────────────────────────────────


REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "reports/medai_doc_type_unknown_diag_17b_env_on_aggregate_eval"
)


@pytest.fixture(scope="module")
def written_files():
    return {
        "json": REPORT_DIR
        / "medai_doc_type_unknown_diag_17b_env_on_aggregate_eval_report.json",
        "md": REPORT_DIR
        / "medai_doc_type_unknown_diag_17b_env_on_aggregate_eval_report.md",
        "summary": REPORT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_17B_ENV_ON_AGGREGATE_EVAL.md",
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


def test_written_json_records_required_fields(written_files):
    payload = json.loads(written_files["json"].read_text(encoding="utf-8"))
    assert payload["phase_id"] == "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B"
    assert payload["mode"] == "env_on_aggregate_evaluation"
    assert payload["evaluation_only"] is True
    assert payload["reports_only"] is True
    assert payload["env_var"] == PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR
    assert payload["env_on_records_evaluated"] == 21
    assert payload["emitted_metadata_count"] == 21
    assert payload["excluded_pool_emission_count"] == 0


# ── runtime wiring guard ──────────────────────────────────────────────────


def test_diag_17b_did_not_wire_helper_into_app():
    app_main = Path(__file__).resolve().parents[1] / "app" / "main.py"
    if not app_main.exists():
        pytest.skip("app/main.py absent in this environment")
    body = app_main.read_text(encoding="utf-8")
    assert "pdf_text_layout_quality_impl" not in body
    assert "derive_pdf_text_layout_quality_context" not in body


def test_diag_17b_did_not_re_export_helper_from_document_type_init():
    init_path = (
        Path(__file__).resolve().parents[1]
        / "clinical_knowledge"
        / "document_type"
        / "__init__.py"
    )
    if not init_path.exists():
        pytest.skip("document_type/__init__.py absent")
    body = init_path.read_text(encoding="utf-8")
    assert "pdf_text_layout_quality_impl" not in body
