"""Focused tests for MEDAI-DOC-TYPE-UNKNOWN-DIAG-01.

Tests verify the diagnostic block's privacy invariants and aggregation logic
using synthetic, in-memory inputs only. No real public report is mutated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from scripts.run_medai_doc_type_unknown_diag_01 import (
    SOURCE_REPORT,
    assert_safe_public_payload,
    build_diagnostic_from_report,
    render_markdown_long,
    render_markdown_summary,
    write_reports,
)


# ── Synthetic source-payload fixture ─────────────────────────────────────────

@pytest.fixture
def synthetic_family04_payload() -> dict:
    """Mimics the FAMILY-04 public report schema with deterministic counts."""
    return {
        "total_files_evaluated": 507,
        "unknown_count": 107,
        "unknown_failure_buckets": {
            "insufficient_text_visibility": 75,
            "fallback_ran_but_no_family_match": 17,
            "ambiguous_below_threshold": 15,
        },
        "unknown_ocr_routing_diagnostics": {
            "fallback_false_unknown_bucket_counts": {
                "image_like_pdf_but_not_routed_to_ocr": 11,
                "language_visibility_unknown": 42,
                "no_text_layer": 11,
                "routing_not_eligible": 1,
                "text_layer_present_but_too_short": 21,
            },
            "unknown_not_fallback_eligible_reason_counts": {
                "language_visibility_unknown": 42,
                "no_text_layer": 11,
                "routing_not_eligible": 1,
                "text_layer_present_but_too_short": 21,
            },
            "unknown_image_like_pdfs_not_routed_to_ocr": 11,
            "unknown_text_layer_pdfs_with_too_little_text": 21,
            "unknown_files_with_extraction_errors": 0,
            "unknown_files_eligible_for_fallback_but_not_triggered": 11,
        },
        "false_positive_risk_audit": {
            "unknown_ambiguous_candidate_sets": {
                "('Imaging report', 'Medication plan', 'Treatment plan')": 7,
                "('Administrative / Insurance', 'Referral / Order')": 2,
                "('Imaging report', 'Medication plan')": 4,
                "('Administrative / Insurance', 'Medication plan',"
                " 'Referral / Order')": 1,
                "('Administrative / Insurance', 'Medication plan')": 1,
            }
        },
    }


# ── Aggregation correctness ──────────────────────────────────────────────────

def test_aggregation_total_unknown_matches_buckets(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    assert report.total_unknown_analyzed == 107
    assert report.bucket_counts == {
        "insufficient_text_visibility": 75,
        "fallback_ran_but_no_family_match": 17,
        "ambiguous_below_threshold": 15,
    }


def test_insufficient_text_visibility_sub_breakdown(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    insufficient = report.buckets["insufficient_text_visibility"]
    sub = insufficient["sub_breakdown"]
    # Sub-keys come straight from the source report and may overlap (e.g.
    # image_like_pdf and no_text_layer can both apply to the same file), so we
    # validate individual counts rather than the sum.
    assert sub["language_visibility_unknown"] == 42
    assert sub["text_layer_present_but_too_short"] == 21
    assert sub["image_like_pdf_but_not_routed_to_ocr"] == 11
    assert sub["no_text_layer"] == 11
    assert sub["routing_not_eligible"] == 1
    # The bucket parent count remains the authoritative total.
    assert insufficient["count"] == 75


def test_fallback_bucket_aggregate_only(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    fb = report.buckets["fallback_ran_but_no_family_match"]
    assert fb["count"] == 17
    # The bucket must be aggregate-only — no per-file or per-shape breakdown
    # is published from this block.
    assert list(fb["sub_breakdown"].keys()) == [
        "fallback_ran_no_family_match_aggregate_only"
    ]
    assert fb["sub_breakdown"]["fallback_ran_no_family_match_aggregate_only"] == 17


def test_ambiguous_bucket_summary_only(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    amb = report.buckets["ambiguous_below_threshold"]
    assert amb["count"] == 15
    assert sum(amb["sub_breakdown"].values()) == 15
    # Must steer reviewers away from cue expansion.
    assert "review-bound" in amb["likely_next_action"]
    # Must not contain an implementation recommendation.
    forbidden = {"add cue", "expand cue", "implement", "fix classifier"}
    for token in forbidden:
        assert token not in amb["likely_next_action"].lower()
        for note in amb["notes"]:
            assert token not in note.lower()


# ── Behavior / external-API invariants ──────────────────────────────────────

def test_report_marks_behavior_unchanged(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    assert report.behavior_changed is False
    assert report.external_api_used is False
    assert report.safety_privacy["behavior_changed"] is False
    assert report.safety_privacy["external_api_used"] is False
    assert report.safety_privacy["classifier_behavior_changed"] is False
    assert report.safety_privacy["ocr_routing_changed"] is False
    assert report.safety_privacy["auto_accept_changed"] is False
    assert report.safety_privacy["all_records_remain_review_bound"] is True


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


def test_report_emits_no_raw_filenames(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    payload = json.loads(json.dumps(report, default=lambda x: x.__dict__))
    for s in _all_strings(payload):
        assert not _RAW_FILENAME_RE.search(s), f"raw filename leaked in: {s!r}"


def test_report_emits_no_private_paths(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    payload = json.loads(json.dumps(report, default=lambda x: x.__dict__))
    for s in _all_strings(payload):
        assert not _PRIVATE_PATH_RE.search(s), f"private path leaked in: {s!r}"


def test_sample_ids_are_anonymized(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    sample_id_re = re.compile(r"^[a-z_]+_\d{3}$")
    for bucket in report.buckets.values():
        for sid in bucket["sample_ids"]:
            assert sample_id_re.fullmatch(sid), (
                f"sample id {sid!r} is not anonymized in the form 'label_NNN'"
            )


def test_rendered_markdown_summary_emits_no_raw_filenames(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    md = render_markdown_summary(report)
    assert not _RAW_FILENAME_RE.search(md)
    assert not _PRIVATE_PATH_RE.search(md)


def test_rendered_markdown_long_emits_no_raw_filenames(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    md = render_markdown_long(report)
    assert not _RAW_FILENAME_RE.search(md)
    assert not _PRIVATE_PATH_RE.search(md)


def test_short_commit_hash_policy(synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    # Both reported hashes must be short (12 chars) per the public-report policy.
    assert len(report.park_19_commit_short) == 12
    assert len(report.head_commit_short) <= 12
    assert report.public_report_commit_hash_policy == "short_hashes_only"


def test_safety_guard_blocks_raw_filename():
    payload = {
        "label": "demo",
        "doc": "the file lab_results.pdf was opened",
    }
    with pytest.raises(RuntimeError):
        assert_safe_public_payload(payload)


def test_safety_guard_blocks_private_path():
    with pytest.raises(RuntimeError):
        assert_safe_public_payload({"path": "/home/user/private/x.pdf"})


def test_safety_guard_blocks_explicit_secret_pattern():
    with pytest.raises(RuntimeError):
        assert_safe_public_payload({"creds": "password=hunter2hunter2"})


# ── Public-report privacy check on the actual rendered output ───────────────

def test_check_public_report_payload_passes_on_synthetic_render(
    synthetic_family04_payload,
):
    report = build_diagnostic_from_report(synthetic_family04_payload)
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


# ── End-to-end write to a tmp dir ───────────────────────────────────────────

def test_write_reports_to_tmp(tmp_path: Path, synthetic_family04_payload):
    report = build_diagnostic_from_report(synthetic_family04_payload)
    paths = write_reports(report, out_dir=tmp_path)
    assert set(paths.keys()) == {"json", "md_summary", "md_main"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0

    json_doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert json_doc["total_unknown_analyzed"] == 107
    assert json_doc["behavior_changed"] is False


def test_real_source_report_exists():
    """The real FAMILY-04 public report must be present on the branch."""
    assert SOURCE_REPORT.exists(), (
        f"Expected FAMILY-04 public report at {SOURCE_REPORT}. "
        "UNKNOWN-DIAG-01 cannot run if this is missing."
    )
