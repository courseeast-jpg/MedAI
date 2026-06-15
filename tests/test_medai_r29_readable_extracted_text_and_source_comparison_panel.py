"""Tests for R29 readable extracted-text + source comparison panel.

Validates the live-UI proof artifacts produced by the R29 script and the readable-view
builder. Content-free assertions only: counts, heading presence, proof booleans, and
privacy scans. Never serializes raw clinical text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.mkb_all_records_qa_comparator import (
    build_readable_markdown,
    readable_record_view,
    representative_proof_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r29_readable_extracted_text_and_source_comparison_panel"
TEXT_REPORTS = ("summary.json", "ui_evidence.json", "queue_counts_public.json", "rendered_ui_text_probe.txt")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
    re.compile(r"\bMRN\b"),
    re.compile(r"\bDOB\b"),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def _ui_evidence() -> dict:
    return json.loads((REPORT_DIR / "ui_evidence.json").read_text(encoding="utf-8"))


def test_artifacts_exist():
    for name in (*TEXT_REPORTS, "screenshot_proof.png"):
        assert (REPORT_DIR / name).exists(), name


def test_overall_pass_and_counts():
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["live_ui_proof_ran"] is True
    assert s["extracted_payload_qa_queue_count"] == 179
    assert s["not_extracted_failure_qa_queue_count"] == 317
    assert s["total_staging_records"] == 496


def test_ui_checks_all_pass():
    e = _ui_evidence()
    assert e["ui_proof_ran"] is True
    checks = e["checks"]
    for key in (
        "extracted_queue_count_179",
        "not_extracted_queue_count_317",
        "extracted_content_heading_present",
        "full_schema_content_heading",
        "full_schema_nonplaceholder_sections_items",
        "minimal_review_readable",
        "not_extracted_terminal_reason",
        "source_evidence_visible",
        "raw_json_not_only_detail",
    ):
        assert checks.get(key) is True, key
    assert e["queue_counts"] == {"extracted": 179, "not_extracted": 317}


def test_required_headings_present_in_evidence():
    e = _ui_evidence()
    headings = e["headings_present"]
    for h in ("Extracted content", "Extracted sections", "Extracted items / facts",
              "Source evidence / original preview", "QA decision", "Advanced raw payload"):
        assert headings.get(h) is True, h


def test_readable_view_full_schema_nonplaceholder():
    reps = representative_proof_records()
    assert reps["full_schema"]
    view = readable_record_view(reps["full_schema"])
    assert view["is_extracted"] is True
    assert "Extracted content" in view["headings"]
    pm = view["proof_metrics"]
    assert pm["sections_rendered"] > 0
    assert pm["nonplaceholder_chars"] > 0
    assert pm["readable_present"] is True
    assert pm["source_evidence_visible"] is True


def test_readable_view_not_extracted_terminal_reason():
    reps = representative_proof_records()
    assert reps["not_extracted"]
    view = readable_record_view(reps["not_extracted"])
    assert view["is_extracted"] is False
    assert view["terminal_reason"]
    assert view["not_extracted_explanation"]
    assert view["proof_metrics"]["terminal_reason_present"] is True


def test_minimal_review_readable():
    reps = representative_proof_records()
    if not reps["minimal_review"]:
        pytest.skip("no minimal_review representative")
    view = readable_record_view(reps["minimal_review"])
    pm = view["proof_metrics"]
    assert pm["readable_present"] is True or pm["items_rendered"] > 0


def test_build_readable_markdown_counts():
    md, sec_n, item_n, nonplaceholder = build_readable_markdown(
        {}, [{"section": "labs", "items": ["glucose 5.4", "sodium 140"]}], ["impression: stable"]
    )
    assert sec_n == 1 and item_n == 3 and nonplaceholder > 0
    assert "labs" in md and "glucose" in md


def test_safety_and_no_promotion():
    s = _summary()
    assert s["provider_model_call_made"] is False
    assert s["live_extraction_started"] is False
    assert s["active_verified_records_created"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["source_pdfs_or_images_committed"] is False


def test_public_reports_no_leaks():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in TEXT_REPORTS:
        text = (REPORT_DIR / name).read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            assert not pat.search(text), (name, pat.pattern)


def test_rendered_probe_is_sanitized():
    # The probe keeps only whitelisted summary/marker lines (no raw clinical content).
    text = (REPORT_DIR / "rendered_ui_text_probe.txt").read_text(encoding="utf-8", errors="ignore")
    assert "R29PROOF|queue" in text
    assert "Extracted Payload QA Queue" in text
