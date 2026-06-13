"""Focused tests for MEDAI-AI-PACKAGE-RUN-REVIEW-OPERATOR-UAT-15O-D."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from scripts.run_medai_ai_package_run_review_operator_uat_15o_d import (
    REPORT_DIR,
    build_uat_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_operator_uat_method_is_bounded_no_live() -> None:
    report = build_uat_report()
    summary = report["summary"]
    assert summary["uat_method_used"] == "bounded_source_reachability_plus_deterministic_view_model_markdown_uat"
    assert summary["streamlit_live_launch_used"] is False
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True


def test_all_operator_visible_elements_pass_for_all_families() -> None:
    report = build_uat_report()
    cases = report["cases"]["cases"]
    assert len(cases) == 4
    required = [
        "preview_entry_point_reachable",
        "package_family_label_visible",
        "review_required_visible",
        "source_visible_body_present",
        "source_visible_body_not_collapsed_only",
        "source_sections_visible",
        "candidate_facts_separated",
        "evidence_anchor_present",
        "unknown_values_explicit",
        "uncertainty_flags_visible",
        "no_live_indicator_visible",
        "active_written_count_indicator_visible",
        "auto_accept_false_indicator_visible",
        "compact_for_operator_compare",
        "under_1_minute_compare_preserved",
    ]
    for case in cases:
        for key in required:
            assert case[key] is True, f"{case['package_family']} missing {key}"
        assert case["hallucinated_field_count"] == 0


def test_summary_counts_match_required_acceptance() -> None:
    summary = build_uat_report()["summary"]
    assert summary["run_review_preview_entry_point_reachable_count"] == 4
    assert summary["source_visible_body_present_count"] == 4
    assert summary["evidence_anchor_present_count"] == 4
    assert summary["candidate_facts_separated_count"] == 4
    assert summary["unknown_values_explicit_count"] == 4
    assert summary["uncertainty_flags_visible_count"] == 4
    assert summary["no_live_indicator_visible_count"] == 4
    assert summary["active_written_count_indicator_visible_count"] == 4
    assert summary["auto_accept_false_indicator_visible_count"] == 4
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0
    assert summary["all_operator_uat_checks_passed"] is True


def test_old_provider_gates_not_required() -> None:
    source = (REPO_ROOT / "scripts" / "run_medai_ai_package_run_review_operator_uat_15o_d.py").read_text(
        encoding="utf-8"
    )
    assert "run_gemini_vertex_live_smoke" not in source
    assert "run_gemini_live_smoke" not in source
    assert "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED" not in (REPO_ROOT / "app" / "ai_package_run_review_preview.py").read_text(
        encoding="utf-8"
    )


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_run_review_operator_uat_15o_d.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected = {
        "summary.json",
        "implementation_report.md",
        "operator_uat_cases.json",
        "operator_uat_matrix.md",
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


def test_script_summary_contains_required_final_report_fields() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_run_review_operator_uat_15o_d.py"],
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
