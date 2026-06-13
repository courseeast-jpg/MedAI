"""Focused tests for MEDAI-AI-PACKAGE-QUALITY-EVAL-FAKE-LOCAL-15O-A."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_package_quality_eval import (
    PACKAGE_FAMILIES,
    build_package_quality_fixtures,
    evaluate_all_package_quality,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_quality_eval_fake_local_15o_a"


def test_all_required_fake_local_package_families_exist() -> None:
    fixtures = build_package_quality_fixtures()
    assert [fixture.family_key for fixture in fixtures] == list(PACKAGE_FAMILIES)
    assert len(fixtures) == 4
    for fixture in fixtures:
        assert fixture.source_visible_body
        assert fixture.evidence_anchors
        assert fixture.draft.external_api_used is False
        assert fixture.draft.auto_accept is False
        assert fixture.draft.review_required is True


def test_package_quality_metrics_pass_operator_compare_target() -> None:
    result = evaluate_all_package_quality()
    assert result["summary"]["under_1_minute_compare_pass_count"] == 4
    assert result["summary"]["hallucinated_field_count"] == 0
    assert result["summary"]["all_cases_under_1_minute"] is True
    for case in result["cases"]:
        metrics = case["metrics"]
        assert metrics["source_visible_body_present"] is True
        assert metrics["source_section_grouping_present"] is True
        assert metrics["evidence_anchor_present"] is True
        assert metrics["candidate_facts_separated"] is True
        assert metrics["unknown_values_explicit"] is True
        assert metrics["estimated_human_compare_seconds"] <= 60


def test_safety_invariants_hold_no_live_no_write_no_auto_accept() -> None:
    result = evaluate_all_package_quality()
    assert result["summary"]["live_call_made"] is False
    assert result["summary"]["external_api_used"] is False
    assert result["summary"]["active_written_count"] == 0
    assert result["summary"]["auto_accept"] is False
    assert result["summary"]["review_required"] is True
    assert result["summary"]["all_safety_invariants_passed"] is True


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_quality_eval_fake_local_15o_a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected = {
        "summary.json",
        "implementation_report.md",
        "package_quality_cases.json",
        "package_quality_matrix.md",
    }
    assert expected == {path.name for path in REPORT_DIR.iterdir()}
    for path in REPORT_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "GEMINI_API_KEY" not in text
        assert "Bearer " not in text
        assert "ya29." not in text
        assert "C:\\" not in text
        assert "DOB" not in text
        assert "MRN:" not in text
        payload = json.loads(text) if path.suffix == ".json" else text
        check = check_public_report_payload(payload)
        assert check.passed, f"{path.name}: {check.leak_examples_redacted}"


def test_report_summary_contains_required_closeout_fields() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_quality_eval_fake_local_15o_a.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["billing_check_pending"] is True
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True
    assert summary["under_1_minute_compare_pass_count"] == 4
    assert summary["hallucinated_field_count"] == 0
