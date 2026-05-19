"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    LANGUAGE_PROPAGATION_DISCLAIMER,
    LANGUAGE_PROPAGATION_DISPLAY_TEXT,
    LANGUAGE_PROPAGATION_EXPANDER_LABEL,
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    LANGUAGE_PROPAGATION_VOCAB_TOKEN,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    derive_numeric_table_safe_default_label,
    language_propagation_operator_surface_is_enabled,
    render_plan_for_language_propagation,
    render_plan_for_operator_badge,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_10a_language_propagation_operator_surface import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixtures: synthetic records ────────────────────────────────────────────

def _propagation_record(**overrides) -> dict:
    """Record that DIAG-09A-IMPLEMENTATION labels."""
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
        "administrative_form_shape_detected": "no",
        "date_or_schedule_shape_detected": "no",
        "imaging_modality_shape_detected": "no",
        "native_text_length_bucket": "medium",
        "alphabetic_content_bucket": "high",
        "numeric_content_bucket": "low",
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
        "review_status": "review",
    }
    base.update(overrides)
    return base


def _numeric_table_record(**overrides) -> dict:
    """Record that DIAG-06A-IMPLEMENTATION labels (matches numeric-table safe-default)."""
    base = _propagation_record()
    base.update({
        "numeric_content_bucket": "medium",
        "administrative_form_shape_detected": "yes",
        "date_or_schedule_shape_detected": "yes",
    })
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_payload() -> dict:
    rows: list[dict] = []
    # 4 propagation-pool records
    for i in range(4):
        rows.append(_propagation_record(file_id=f"file_{i + 1:03d}"))
    # 3 numeric-table safe-default records (separate lever)
    for i in range(3):
        rows.append(_numeric_table_record(file_id=f"file_{10 + i:03d}"))
    # Out-of-scope rows
    rows.append(_propagation_record(file_id="file_020",
                                    medical_abbreviation_shape_detected="yes"))
    rows.append(_propagation_record(file_id="file_021",
                                    unknown_failure_bucket="ambiguous_below_threshold"))
    rows.append({"file_id": "file_030",
                 "predicted_document_type": "Lab result",
                 "unknown_failure_bucket": ""})
    return {"anonymous_per_file_table": rows}


# ── Plan gating: env-var + kwarg ───────────────────────────────────────────

def test_no_plan_when_default_off():
    r = _propagation_record()
    assert render_plan_for_language_propagation(r, env={}) is None


def test_no_plan_when_env_var_falsy():
    r = _propagation_record()
    for value in ["", "0", "false", "no", "off", "disabled", "random"]:
        assert render_plan_for_language_propagation(
            r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: value}
        ) is None


def test_plan_returned_when_propagation_env_truthy():
    r = _propagation_record()
    for value in ["1", "true", "TRUE", "yes", "on", "enabled"]:
        plan = render_plan_for_language_propagation(
            r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: value}
        )
        assert plan is not None, f"truthy env value {value!r} should enable"
        assert plan["expander_label"] == LANGUAGE_PROPAGATION_EXPANDER_LABEL
        assert plan["disclaimer_line"] == LANGUAGE_PROPAGATION_DISCLAIMER


def test_plan_returned_when_explicit_enabled():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    assert plan is not None


def test_explicit_disable_overrides_env():
    assert render_plan_for_language_propagation(
        _propagation_record(),
        enabled=False,
        env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"},
    ) is None


def test_language_propagation_operator_surface_is_enabled_consults_env():
    assert language_propagation_operator_surface_is_enabled(env={}) is False
    assert language_propagation_operator_surface_is_enabled(
        env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "yes"}
    ) is True


# ── Flag separation ───────────────────────────────────────────────────────

def test_operator_review_env_does_not_enable_propagation_plan():
    """Setting MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED must NOT enable
    the propagation display."""
    r = _propagation_record()
    assert render_plan_for_language_propagation(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    ) is None


def test_propagation_env_does_not_enable_operator_badge_plan():
    """Setting MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED must NOT
    enable the DIAG-08A numeric-table operator badge."""
    r = _numeric_table_record()
    assert render_plan_for_operator_badge(
        r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    ) is None


def test_env_vars_are_distinct():
    assert (LANGUAGE_PROPAGATION_METADATA_ENV_VAR
            != OPERATOR_REVIEW_BADGE_ENV_VAR)


def test_both_env_vars_together_show_both_displays_separately():
    """When both env vars are set, each lever should render its own
    priority slice without cross-contamination."""
    prop_r = _propagation_record(file_id="file_p")
    nt_r = _numeric_table_record(file_id="file_n")
    env = {
        LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1",
        OPERATOR_REVIEW_BADGE_ENV_VAR: "1",
    }
    # Propagation plan for the propagation record
    pp = render_plan_for_language_propagation(prop_r, env=env)
    assert pp is not None
    # Operator badge for the numeric-table record
    nb = render_plan_for_operator_badge(nt_r, env=env)
    assert nb is not None
    # Cross-contamination: each plan only applies to its own lever's slice
    assert render_plan_for_language_propagation(nt_r, env=env) is None
    assert render_plan_for_operator_badge(prop_r, env=env) is None


# ── Plan shape: read-only, no operator action ─────────────────────────────

def test_plan_carries_no_action_handle():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    assert plan is not None
    assert plan["is_read_only"] is True
    assert plan["no_action_attached"] is True
    for forbidden in (
        "on_click", "on_change", "callback", "button_handle", "action",
        "submit_handler", "click_handler", "form_handle",
    ):
        assert forbidden not in plan


def test_plan_marks_review_bound_not_clinical():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    assert plan["review_bound"] is True
    assert plan["is_clinical_classification"] is False
    assert plan["is_final_document_type"] is False
    assert plan["is_auto_accept"] is False
    assert plan["is_active_clinical_fact"] is False
    assert plan["is_data_layer_document_type_change"] is False
    assert plan["raw_detector_output_unchanged"] is True


def test_plan_disclaimer_line_text():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    text = plan["disclaimer_line"].lower()
    assert "review metadata only" in text
    assert "not a final document type" in text
    assert "not clinical interpretation" in text


def test_plan_markdown_does_not_promote_to_final_type():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    joined = "\n".join(plan["markdown_lines"]).lower()
    for forbidden in (
        "lab result", "imaging report", "treatment plan", "medication plan",
        "discharge summary", "pathology report", "procedure report",
    ):
        assert forbidden not in joined


def test_plan_does_not_mutate_record():
    r = _propagation_record()
    snapshot = dict(r)
    render_plan_for_language_propagation(r, enabled=True)
    assert r == snapshot


def test_raw_detector_fields_unchanged_by_render_plan():
    r = _propagation_record()
    snap = {k: r.get(k) for k in (
        "language_detector_attempted",
        "language_detector_input_bucket",
        "detector_confidence_bucket",
        "script_detection_result",
        "dominant_script",
        "language_visibility_status",
    )}
    render_plan_for_language_propagation(r, enabled=True)
    for k, v in snap.items():
        assert r.get(k) == v


def test_data_layer_document_type_unchanged_by_render_plan():
    r = _propagation_record()
    snap = r.get("predicted_document_type")
    render_plan_for_language_propagation(r, enabled=True)
    assert r.get("predicted_document_type") == snap == "Unknown"


# ── Priority-slice gating ─────────────────────────────────────────────────

@pytest.mark.parametrize("override", [
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
    # Numeric-table safe-default overlap: must not also receive propagation
    {"numeric_content_bucket": "medium",
     "administrative_form_shape_detected": "yes",
     "date_or_schedule_shape_detected": "yes"},
])
def test_no_plan_for_excluded_records(override):
    r = _propagation_record(**override)
    assert render_plan_for_language_propagation(r, enabled=True) is None


def test_numeric_table_safe_default_record_does_not_display_propagation():
    """A numeric-table safe-default record must NOT receive the propagation
    display (it has its own DIAG-08A operator badge instead)."""
    r = _numeric_table_record()
    # Confirm the DIAG-06A helper does label it.
    assert derive_numeric_table_safe_default_label(r, enabled=True) is not None
    # But the propagation display must not.
    assert render_plan_for_language_propagation(r, enabled=True) is None


# ── Hard guardrails ───────────────────────────────────────────────────────

def test_helper_does_not_introduce_auto_accept():
    r = _propagation_record()
    plan = render_plan_for_language_propagation(r, enabled=True)
    assert plan is not None
    assert plan["is_auto_accept"] is False


def test_helper_does_not_promote_document_type():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    assert plan["is_final_document_type"] is False
    assert plan["is_data_layer_document_type_change"] is False


def test_helper_does_not_classify_clinical_meaning():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    text = plan["badge_text"].lower()
    forbidden = (
        "diagnosis", "diagnosed", "treatment of", "lab value", "ddi",
        "imaging finding", "medication", "dose",
    )
    for tok in forbidden:
        assert tok not in text


def test_helper_does_not_parse_labs_meds_or_ddi():
    for forbidden_payload in (
        {"parsed_lab_values": [{"name": "Hb"}]},
        {"parsed_medications": ["metformin"]},
        {"parsed_ddi_findings": ["x"]},
        {"parsed_doses": ["500mg"]},
        {"parsed_frequencies": ["bid"]},
    ):
        r = _propagation_record(**forbidden_payload)
        assert render_plan_for_language_propagation(r, enabled=True) is None


def test_external_api_used_remains_false_on_plan():
    plan = render_plan_for_language_propagation(_propagation_record(), enabled=True)
    assert plan is not None
    assert "external_api_used" not in plan or plan["external_api_used"] is False


def test_rollback_paths_all_disable_plan():
    r = _propagation_record()
    # 1. Default (no kwarg, no env)
    assert render_plan_for_language_propagation(r) is None
    # 2. Explicit False
    assert render_plan_for_language_propagation(r, enabled=False) is None
    # 3. Falsy env
    assert render_plan_for_language_propagation(
        r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "0"}
    ) is None
    # 4. Explicit False overrides truthy env
    assert render_plan_for_language_propagation(
        r, enabled=False, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    ) is None


# ── Audit-script outputs ──────────────────────────────────────────────────

def test_audit_default_off(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.default_off_audit["propagation_plan_count"] == 0
    assert report.default_off_audit["operator_badge_plan_count"] == 0


def test_audit_prop_env_only(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # 4 propagation-pool records labeled, 0 operator badges
    assert report.propagation_env_only_audit["propagation_plan_count"] == 4
    assert report.propagation_env_only_audit["operator_badge_plan_count"] == 0


def test_audit_op_env_only(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # 0 propagation plans, 3 numeric-table operator badges
    assert report.operator_env_only_audit["propagation_plan_count"] == 0
    assert report.operator_env_only_audit["operator_badge_plan_count"] == 3


def test_audit_both_env(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.both_env_audit["propagation_plan_count"] == 4
    assert report.both_env_audit["operator_badge_plan_count"] == 3
    # Each lever rendered only its own priority slice.
    assert (report.both_env_audit[
        "propagation_matches_priority_slice_exactly"] is True)
    assert (report.both_env_audit[
        "operator_badge_matches_priority_slice_exactly"] is True)


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
    assert agg["default_off_propagation_plan_count"] == 0
    assert agg["op_env_only_propagation_plan_count"] == 0
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True


def test_flag_separation_audit_clean(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    audit = report.flag_separation_audit
    assert audit["default_off_propagation_plans_zero"] is True
    assert audit["default_off_operator_badge_plans_zero"] is True
    assert audit["prop_env_only_yields_propagation_plans"] is True
    assert audit["prop_env_only_yields_zero_operator_badge_plans"] is True
    assert audit["op_env_only_yields_zero_propagation_plans"] is True
    assert audit["op_env_only_yields_operator_badge_plans"] is True
    assert audit["both_env_yields_both_levers_with_correct_priority_slices"] is True
    assert audit["flag_separation_holds_in_all_modes"] is True


def test_propagation_display_count_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.propagation_metadata_display_count == 4
    assert report.numeric_table_badge_display_count == 3


def test_counts_remain_zero(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.accepted_count == 0
    assert report.auto_accept_allowed_count == 0
    assert report.external_api_used_count == 0


def test_review_bound_preserved(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.review_bound_preserved is True


def test_no_action_attached(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.no_action_attached_to_plan is True


def test_raw_detector_and_doc_type_unchanged(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.raw_detector_output_unchanged is True
    assert report.data_layer_document_type_unchanged is True


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
    assert sp["raw_detector_output_unchanged"] is True
    assert sp["data_layer_document_type_unchanged"] is True
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["operator_surface_default_disabled"] is True
    assert sp["rollback_path_present"] is True
    assert sp["no_action_attached_to_plan"] is True
    assert sp["no_button_or_callback_in_render_plan"] is True
    assert sp["flag_separation_holds"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_10a_unknown_track_done_pct",
        "approximately 84%",
        "approximately 16%",
        "after_10a_unknown_track_done_pct",
        "approximately 87%",
        "approximately 13%",
        "before_10a_project_done_pct",
        "approximately 81%",
        "approximately 19%",
        "after_10a_project_done_pct",
        "approximately 82%",
        "approximately 18%",
    ):
        assert token in md


def test_rendered_markdown_mentions_ui_surface(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    assert "render_run_result_card" in md
    assert "Advanced technical details" in md


# ── Privacy invariants ────────────────────────────────────────────────────

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


# ── End-to-end ────────────────────────────────────────────────────────────

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
    assert json_doc["raw_detector_output_unchanged"] is True
    assert json_doc["data_layer_document_type_unchanged"] is True


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()


# ── app/main.py call-site assertions (static) ─────────────────────────────

def test_main_py_call_site_present_and_guarded():
    """The DIAG-10A call site in app/main.py must:
       1. Import `render_plan_for_language_propagation`
       2. Live inside the Advanced technical details expander
       3. Be wrapped in a try/except
       4. Attach no operator action to the badge
       5. Live alongside but distinct from the DIAG-08A operator-badge call site."""
    text = Path("app/main.py").read_text(encoding="utf-8")
    assert "render_plan_for_language_propagation" in text
    assert "Advanced technical details" in text
    assert ("from clinical_knowledge.document_type.language_propagation_operator_surface "
            "import" in text)
    # Both call sites should be present.
    assert "render_plan_for_operator_badge" in text
    # Locate the propagation call site region.
    start = text.find(
        "from clinical_knowledge.document_type.language_propagation_operator_surface"
    )
    assert start != -1
    end = text.find("except Exception", start)
    assert end != -1
    region = text[start:end]
    for forbidden in ("st.button(", "st.form(", "on_click=", "on_change="):
        assert forbidden not in region, (
            f"propagation call site must not attach an operator action ({forbidden!r})"
        )
