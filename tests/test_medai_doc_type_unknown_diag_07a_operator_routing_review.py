"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.document_type import (
    OPERATOR_REVIEW_BADGE_DISCLAIMER,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    OPERATOR_REVIEW_BADGE_TEXT,
    OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
    derive_operator_review_badge,
    is_operator_review_badge_default_disabled,
    is_operator_review_badge_enabled,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_07a_operator_routing_review import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Fixture ─────────────────────────────────────────────────────────────────

def _matching_record(**overrides) -> dict:
    """Baseline record matching all 14 positive-signature fields, ready to
    receive the operator-review badge when the flag is enabled."""
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
    # 4 records that match the priority slice
    for i in range(4):
        rows.append(_matching_record(file_id=f"file_{i + 1:03d}"))
    # 1 record with section heading -> excluded by safeguard
    rows.append(_matching_record(file_id="file_010",
                                 section_heading_shape_detected="yes"))
    # 1 record outside Unknown -> excluded by safeguard
    rows.append(_matching_record(file_id="file_013",
                                 predicted_document_type="Lab result"))
    return {"anonymous_per_file_table": rows}


# ── Flag plumbing ───────────────────────────────────────────────────────────

def test_helper_is_default_disabled():
    assert is_operator_review_badge_default_disabled() is True


def test_env_flag_disabled_when_var_missing():
    assert is_operator_review_badge_enabled(env={}) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on", "enabled"])
def test_env_flag_truthy_values(value):
    assert is_operator_review_badge_enabled(
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: value}
    ) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "disabled", "random"])
def test_env_flag_falsy_values(value):
    assert is_operator_review_badge_enabled(
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: value}
    ) is False


# ── Badge gating ────────────────────────────────────────────────────────────

def test_default_off_returns_none_even_for_matching_record():
    r = _matching_record()
    # No kwarg, empty env -> default off -> None
    assert derive_operator_review_badge(r, env={}) is None


def test_explicit_disable_returns_none_even_for_matching_record():
    r = _matching_record()
    # Explicit False overrides any env setting
    assert derive_operator_review_badge(
        r, enabled=False,
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"},
    ) is None


def test_explicit_enable_returns_badge_for_matching_record():
    r = _matching_record()
    badge = derive_operator_review_badge(r, enabled=True)
    assert badge is not None
    assert badge["badge_vocab_token"] == OPERATOR_REVIEW_BADGE_VOCAB_TOKEN
    assert badge["badge_text"] == OPERATOR_REVIEW_BADGE_TEXT
    assert badge["badge_disclaimer"] == OPERATOR_REVIEW_BADGE_DISCLAIMER


def test_env_enable_returns_badge_for_matching_record():
    r = _matching_record()
    badge = derive_operator_review_badge(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "yes"},
    )
    assert badge is not None
    assert badge["badge_vocab_token"] == OPERATOR_REVIEW_BADGE_VOCAB_TOKEN


# ── Hard-guardrail invariants on the returned badge ─────────────────────────

def test_badge_marks_review_bound_and_not_clinical():
    badge = derive_operator_review_badge(_matching_record(), enabled=True)
    assert badge is not None
    assert badge["review_bound"] is True
    assert badge["is_clinical_classification"] is False
    assert badge["is_final_document_type"] is False
    assert badge["is_auto_accept"] is False
    assert badge["is_active_clinical_fact"] is False


def test_badge_does_not_mutate_record():
    r = _matching_record()
    snapshot = dict(r)
    derive_operator_review_badge(r, enabled=True)
    assert r == snapshot


def test_badge_text_carries_review_required_phrasing():
    """The plain-language badge must include 'review' so the operator UI
    surface cannot mistake it for a final classification."""
    assert "review" in OPERATOR_REVIEW_BADGE_TEXT.lower()
    assert "metadata" in OPERATOR_REVIEW_BADGE_TEXT.lower()


# ── Records outside the priority slice get no badge ─────────────────────────

@pytest.mark.parametrize("override,reason", [
    ({"section_heading_shape_detected": "yes"},
     "table-header lever, not numeric-table"),
    ({"medical_abbreviation_shape_detected": "yes"},
     "abbreviation lever, not numeric-table"),
    ({"unknown_ocr_routing_bucket": "text_layer_present_but_too_short"},
     "text-layer subset"),
    ({"predicted_document_type": "Lab result"},
     "non-Unknown predicted type"),
    ({"dominant_script": "cyrillic"}, "exclusion rule fires"),
    ({"dominant_script": "mixed"},    "exclusion rule fires"),
    ({"alphabetic_content_bucket": "low"}, "exclusion rule fires"),
    ({"pdf_text_layer_detected": "no"},     "exclusion rule fires"),
    ({"image_like_pdf": "yes"},             "exclusion rule fires"),
    ({"unknown_failure_bucket": "ambiguous_below_threshold"},
     "exclusion rule fires"),
    ({"unknown_failure_bucket": "fallback_ran_but_no_family_match"},
     "exclusion rule fires"),
    ({"parsed_medications": ["metformin"]},
     "medication/DDI exclusion rule fires"),
    ({"parsed_lab_values": [{"name": "Hb", "value": "13.5"}]},
     "lab-value exclusion rule fires"),
])
def test_no_badge_outside_priority_slice(override, reason):
    r = _matching_record(**override)
    assert derive_operator_review_badge(r, enabled=True) is None, (
        f"badge issued for excluded record ({reason})"
    )


# ── Helper integration: no auto-accept / promotion / parsing introduced ─────

def test_badge_does_not_introduce_auto_accept():
    badge = derive_operator_review_badge(_matching_record(), enabled=True)
    assert badge is not None
    assert badge["is_auto_accept"] is False
    # The badge dict must not carry any "accept" or "promote" decision.
    for key in badge.keys():
        assert "auto_accept" not in key or key == "is_auto_accept"
        assert "promote" not in key


def test_badge_does_not_promote_document_type():
    badge = derive_operator_review_badge(_matching_record(), enabled=True)
    assert badge is not None
    assert badge["is_final_document_type"] is False
    # The badge text must not name a final family.
    forbidden_final_types = (
        "lab result", "imaging report", "treatment plan", "medication plan",
        "discharge summary", "pathology report", "procedure report",
    )
    for tok in forbidden_final_types:
        assert tok not in badge["badge_text"].lower()


def test_badge_does_not_introduce_clinical_classification():
    badge = derive_operator_review_badge(_matching_record(), enabled=True)
    assert badge["is_clinical_classification"] is False
    # The badge text must not name a clinical concept.
    forbidden_clinical_tokens = (
        "diagnosis", "diagnosed", "ddi", "dose", "mg ", "mcg ",
        "treatment of", "imaging finding", "lab value",
    )
    for tok in forbidden_clinical_tokens:
        assert tok not in badge["badge_text"].lower()


def test_helper_does_not_parse_labs_or_medications():
    # Caller passes parsed labs/meds -> exclusion rule fires -> None badge
    assert derive_operator_review_badge(
        _matching_record(parsed_lab_values=[{"name": "Hb", "value": "13"}]),
        enabled=True,
    ) is None
    assert derive_operator_review_badge(
        _matching_record(parsed_medications=["metformin"]),
        enabled=True,
    ) is None
    assert derive_operator_review_badge(
        _matching_record(parsed_ddi_findings=["x"]),
        enabled=True,
    ) is None


def test_external_api_used_remains_false():
    """No external API call should be made by the helper."""
    badge = derive_operator_review_badge(_matching_record(), enabled=True)
    assert badge is not None
    # The badge dict must not record any external-API use.
    assert "external_api_used" not in badge or badge["external_api_used"] is False


# ── Rollback paths ──────────────────────────────────────────────────────────

def test_rollback_via_env_var_unset():
    r = _matching_record()
    assert derive_operator_review_badge(r, env={}) is None


def test_rollback_via_explicit_false():
    r = _matching_record()
    assert derive_operator_review_badge(
        r, enabled=False,
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"},
    ) is None


def test_rollback_path_disable_even_for_matching_record():
    r = _matching_record()
    # All four rollback paths
    assert derive_operator_review_badge(r) is None  # default
    assert derive_operator_review_badge(r, enabled=None, env={}) is None
    assert derive_operator_review_badge(r, enabled=False) is None
    assert derive_operator_review_badge(
        r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "0"},
    ) is None


# ── Audit report builder ────────────────────────────────────────────────────

def test_audit_default_off_yields_zero_badges(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.default_off_audit["badge_count"] == 0
    assert report.default_off_audit["extras_outside_priority_count"] == 0


def test_audit_env_and_explicit_match_priority_slice(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    # 4 priority records in the synthetic fixture
    assert report.env_enabled_audit["badge_count"] == 4
    assert report.explicit_enabled_audit["badge_count"] == 4
    assert report.env_enabled_audit["matches_priority_slice_exactly"] is True
    assert report.explicit_enabled_audit["matches_priority_slice_exactly"] is True


def test_eleven_record_replay_summary(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    rep = report.eleven_record_replay
    assert rep["priority_slice_size"] == 4
    assert rep["enabled_true_badge_count"] == 4
    assert rep["enabled_false_badge_count"] == 0
    assert rep["default_off_badge_count"] == 0
    assert rep["matches_priority_slice_exactly"] is True


def test_aggregate_no_false_positives(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    agg = report.five_hundred_seven_file_aggregate
    assert agg["default_off_badge_count"] == 0
    assert agg["no_false_positive_outside_priority"] is True
    assert agg["no_false_negative_inside_priority"] is True


def test_operator_badge_display_count_recorded(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.operator_badge_display_count == 4
    # Distinct from data-layer Unknown count
    assert report.unknown_count_at_data_layer_delta == 0
    assert (
        report.operator_review_metadata_display_count
        == report.operator_badge_display_count
    )


def test_counts_remain_zero(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.accepted_count == 0
    assert report.auto_accept_allowed_count == 0
    assert report.external_api_used_count == 0


def test_review_bound_preserved(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert report.review_bound_preserved is True


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
        sp["behavior_changed_strictly_limited_to_operator_review_display"] is True
    )
    assert sp["clinical_behavior_changed"] is False
    assert sp["external_api_used"] is False
    assert sp["cue_expansion_recommended"] is False
    assert sp["all_records_remain_review_bound"] is True
    assert sp["operator_review_badge_default_disabled"] is True
    assert sp["rollback_path_present"] is True
    assert sp["underlying_helper_default_disabled_outside_call_site"] is True


def test_short_commit_hash_policy(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


def test_rendered_markdown_includes_progress_percentages(synthetic_payload):
    report = build_diagnostic_from_report(synthetic_payload)
    md = render_markdown_summary(report)
    for token in (
        "before_07a_unknown_track_done_pct",
        "approximately 68%",
        "approximately 32%",
        "after_07a_unknown_track_done_pct",
        "approximately 72%",
        "approximately 28%",
        "before_07a_project_done_pct",
        "approximately 77%",
        "approximately 23%",
        "after_07a_project_done_pct",
        "approximately 78%",
        "approximately 22%",
    ):
        assert token in md


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


def test_real_source_report_exists():
    assert SOURCE_REPORT.exists()
