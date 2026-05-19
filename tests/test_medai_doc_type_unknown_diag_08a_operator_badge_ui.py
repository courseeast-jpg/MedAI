"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    OPERATOR_BADGE_UI_DISCLAIMER,
    OPERATOR_BADGE_UI_EXPANDER_LABEL,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    operator_badge_ui_is_enabled,
    render_plan_for_operator_badge,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_08a_operator_badge_ui import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixture ─────────────────────────────────────────────────────────────────

def _matching_record(**overrides) -> dict:
    """Record that satisfies every gate down to the DIAG-08A render plan."""
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
        "script_detection_attempted": "yes",
        "script_detection_result": "latin",
        "symbol_content_bucket": "medium",
        "detector_confidence_bucket": "high",
        "cyrillic_visibility_status": "unknown",
        "review_status": "review",
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_payload() -> dict:
    rows: list[dict] = []
    # 4 records that satisfy every gate
    for i in range(4):
        rows.append(_matching_record(file_id=f"file_{i + 1:03d}"))
    # Section-heading -> excluded by safeguard
    rows.append(_matching_record(file_id="file_010",
                                 section_heading_shape_detected="yes"))
    # Non-Unknown -> excluded by safeguard
    rows.append(_matching_record(file_id="file_013",
                                 predicted_document_type="Lab result"))
    return {"anonymous_per_file_table": rows}


# ── Plan gating: env-var + kwarg ────────────────────────────────────────────

def test_no_plan_when_disabled_default():
    r = _matching_record()
    # No kwarg, empty env -> default off
    assert render_plan_for_operator_badge(r, env={}) is None


def test_no_plan_when_env_var_falsy():
    r = _matching_record()
    for value in ["", "0", "false", "no", "off", "disabled", "random"]:
        assert render_plan_for_operator_badge(
            r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: value}
        ) is None, f"falsy env value {value!r} should not enable the plan"


def test_plan_returned_when_env_var_truthy():
    r = _matching_record()
    for value in ["1", "true", "TRUE", "yes", "on", "enabled"]:
        plan = render_plan_for_operator_badge(
            r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: value}
        )
        assert plan is not None, f"truthy env value {value!r} should enable plan"
        assert plan["expander_label"] == OPERATOR_BADGE_UI_EXPANDER_LABEL
        assert plan["disclaimer_line"] == OPERATOR_BADGE_UI_DISCLAIMER


def test_plan_returned_when_explicit_enabled():
    r = _matching_record()
    plan = render_plan_for_operator_badge(r, enabled=True)
    assert plan is not None


def test_explicit_disable_overrides_env():
    r = _matching_record()
    # explicit False even when env is truthy
    assert render_plan_for_operator_badge(
        r, enabled=False, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    ) is None


def test_operator_badge_ui_is_enabled_consults_env():
    assert operator_badge_ui_is_enabled(env={}) is False
    assert operator_badge_ui_is_enabled(
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: "yes"}
    ) is True


# ── Plan shape: read-only metadata, no operator action ──────────────────────

def test_plan_carries_no_action_handle():
    plan = render_plan_for_operator_badge(_matching_record(), enabled=True)
    assert plan is not None
    assert plan["is_read_only"] is True
    assert plan["no_action_attached"] is True
    # Forbidden action handles must not be present.
    for forbidden in (
        "on_click", "on_change", "callback", "button_handle", "action",
        "submit_handler", "click_handler", "form_handle",
    ):
        assert forbidden not in plan, (
            f"render plan must not carry an operator action handle "
            f"({forbidden!r})"
        )


def test_plan_marks_review_bound_not_clinical():
    plan = render_plan_for_operator_badge(_matching_record(), enabled=True)
    assert plan["review_bound"] is True
    assert plan["is_clinical_classification"] is False
    assert plan["is_final_document_type"] is False
    assert plan["is_auto_accept"] is False
    assert plan["is_active_clinical_fact"] is False


def test_plan_disclaimer_line_text():
    plan = render_plan_for_operator_badge(_matching_record(), enabled=True)
    text = plan["disclaimer_line"].lower()
    assert "review metadata only" in text
    assert "not a final document type" in text
    assert "not clinical interpretation" in text


def test_plan_markdown_does_not_promote_to_final_type():
    plan = render_plan_for_operator_badge(_matching_record(), enabled=True)
    joined = "\n".join(plan["markdown_lines"]).lower()
    # The plan must never name a final document family.
    for forbidden in (
        "lab result", "imaging report", "treatment plan", "medication plan",
        "discharge summary", "pathology report", "procedure report",
    ):
        assert forbidden not in joined


def test_plan_does_not_mutate_record():
    r = _matching_record()
    snapshot = dict(r)
    render_plan_for_operator_badge(r, enabled=True)
    assert r == snapshot


# ── Priority-slice gating: records outside the slice never render ──────────

@pytest.mark.parametrize("override", [
    {"section_heading_shape_detected": "yes"},
    {"medical_abbreviation_shape_detected": "yes"},
    {"unknown_ocr_routing_bucket": "text_layer_present_but_too_short"},
    {"predicted_document_type": "Lab result"},
    {"dominant_script": "cyrillic"},
    {"dominant_script": "mixed"},
    {"alphabetic_content_bucket": "low"},
    {"pdf_text_layer_detected": "no"},
    {"image_like_pdf": "yes"},
    {"unknown_failure_bucket": "ambiguous_below_threshold"},
    {"unknown_failure_bucket": "fallback_ran_but_no_family_match"},
    {"parsed_medications": ["metformin"]},
    {"parsed_lab_values": [{"name": "Hb", "value": "13.5"}]},
    {"parsed_ddi_findings": ["finding"]},
])
def test_no_plan_for_excluded_records(override):
    r = _matching_record(**override)
    assert render_plan_for_operator_badge(r, enabled=True) is None


# ── Hard guardrails (no parsing / no clinical / no external API) ───────────

def test_helper_does_not_parse_labs_meds_or_ddi():
    """Caller passing parsed labs/meds/DDIs forces the plan to None."""
    for forbidden_payload in (
        {"parsed_lab_values": [{"name": "Hb"}]},
        {"parsed_medications": ["metformin"]},
        {"parsed_ddi_findings": ["x"]},
    ):
        r = _matching_record(**forbidden_payload)
        assert render_plan_for_operator_badge(r, enabled=True) is None


def test_external_api_used_remains_false_on_plan():
    plan = render_plan_for_operator_badge(_matching_record(), enabled=True)
    assert plan is not None
    assert "external_api_used" not in plan or plan["external_api_used"] is False


def test_rollback_paths_all_disable_plan():
    r = _matching_record()
    # 1. Default (no kwarg, no env)
    assert render_plan_for_operator_badge(r) is None
    # 2. Explicit False
    assert render_plan_for_operator_badge(r, enabled=False) is None
    # 3. Falsy env
    assert render_plan_for_operator_badge(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "0"}
    ) is None
    # 4. Explicit False overrides truthy env
    assert render_plan_for_operator_badge(
        r, enabled=False, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    ) is None


# ── Audit-script outputs ────────────────────────────────────────────────────

def test_audit_default_off(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.default_off_audit["plan_count"] == 0


def test_audit_env_and_explicit_match_priority_slice(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.env_enabled_audit["plan_count"] == 4
    assert report.explicit_enabled_audit["plan_count"] == 4
    assert report.env_enabled_audit["matches_priority_slice_exactly"] is True
    assert report.explicit_enabled_audit["matches_priority_slice_exactly"] is True


def test_eleven_record_replay_summary(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    rep = report.eleven_record_replay
    assert rep["priority_slice_size"] == 4
    assert rep["enabled_true_plan_count"] == 4
    assert rep["enabled_false_plan_count"] == 0
    assert rep["default_off_plan_count"] == 0
    assert rep["matches_priority_slice_exactly"] is True


def test_aggregate_no_false_positives(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    agg = report.five_hundred_seven_file_aggregate
    assert agg["default_off_plan_count"] == 0
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True


def test_operator_badge_display_count_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.operator_badge_display_count == 4
    assert report.unknown_count_at_data_layer_delta == 0


def test_counts_remain_zero(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.accepted_count == 0
    assert report.auto_accept_allowed_count == 0
    assert report.external_api_used_count == 0


def test_review_bound_preserved(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.review_bound_preserved is True


def test_no_action_attached_to_badge(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.no_action_attached_to_badge is True


def test_false_positive_audit_clean(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    for k, v in report.false_positive_audit.items():
        assert v == 0, f"false-positive expansion in {k}: {v}"
    assert report.no_false_positive_expansion is True


def test_report_flags(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.behavior_changed is True
    assert report.clinical_behavior_changed is False
    assert report.external_api_used is False
    assert report.cue_expansion_recommended is False
    sp = report.safety_privacy
    assert (
        sp["behavior_changed_strictly_limited_to_read_only_ui_display"] is True
    )
    assert sp["clinical_behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["operator_badge_ui_default_disabled"] is True
    assert sp["rollback_path_present"] is True
    assert sp["no_action_attached_to_badge"] is True
    assert sp["no_button_or_callback_in_render_plan"] is True
    assert sp["ui_render_failure_is_silently_swallowed"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_08a_unknown_track_done_pct",
        "approximately 72%",
        "approximately 28%",
        "after_08a_unknown_track_done_pct",
        "approximately 76%",
        "approximately 24%",
        "before_08a_project_done_pct",
        "approximately 78%",
        "approximately 22%",
        "after_08a_project_done_pct",
        "approximately 79%",
        "approximately 21%",
    ):
        assert token in md


def test_rendered_markdown_mentions_ui_surface(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    assert "render_run_result_card" in md
    assert "Advanced technical details" in md


# ── Privacy ────────────────────────────────────────────────────────────────

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


# ── End-to-end ──────────────────────────────────────────────────────────────

def test_write_reports_to_tmp(tmp_path: Path, synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    paths = write_reports(report, out_dir=tmp_path)
    assert set(paths.keys()) == {"json", "md_summary", "md_main"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    json_doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert json_doc["behavior_changed"] is True
    assert json_doc["clinical_behavior_changed"] is False
    assert json_doc["external_api_used"] is False
    assert json_doc["cue_expansion_recommended"] is False
    assert json_doc["no_action_attached_to_badge"] is True


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()


# ── app/main.py call-site assertions (static, no Streamlit needed) ──────────

def test_main_py_call_site_present_and_guarded():
    """The app/main.py call site must:
       1. import `render_plan_for_operator_badge`
       2. live inside an Advanced technical details expander
       3. be wrapped in a try/except so the optional integration can never
          break the main UI render path.
       4. attach no operator action (no st.button / st.form / on_click)
          to the badge."""
    text = Path("app/main.py").read_text(encoding="utf-8")
    assert "render_plan_for_operator_badge" in text
    assert "Advanced technical details" in text
    # The call site must live inside a try/except.
    assert ("from clinical_knowledge.document_type.operator_badge_ui import"
            in text)
    # The Streamlit widgets immediately around the badge call site must be
    # render-only (markdown/caption); no button / form / on_click should
    # appear between the import line and the matching except.
    start = text.find("from clinical_knowledge.document_type.operator_badge_ui")
    end = text.find("except Exception", start)
    assert start != -1 and end != -1, (
        "operator badge call site not found inside a try/except"
    )
    region = text[start:end]
    for forbidden in ("st.button(", "st.form(", "on_click=", "on_change="):
        assert forbidden not in region, (
            f"badge call site must not attach an operator action ({forbidden!r})"
        )
