"""Focused tests for MEDAI-AI-PACKAGE-QUALITY-REVIEW-SURFACE-WIRING-15O-B."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.ai_package_review_surface import (
    build_all_review_surface_view_models,
    build_review_surface_report,
    render_review_surface_markdown,
    review_surface_to_public_dict,
)
from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_package_quality_eval import PACKAGE_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_quality_review_surface_15o_b"


def test_view_models_cover_all_15o_a_package_families() -> None:
    models = build_all_review_surface_view_models()
    assert [model.package_family for model in models] == list(PACKAGE_FAMILIES)
    for model in models:
        assert model.package_id
        assert model.package_family_label
        assert model.source_visible_body
        assert model.source_sections
        assert model.candidate_facts
        assert model.evidence_anchors
        assert model.review_required is True


def test_review_surface_separates_source_body_candidate_facts_and_anchors() -> None:
    for model in build_all_review_surface_view_models():
        public = review_surface_to_public_dict(model)
        assert public["source_visible_body"]
        assert public["source_sections"]
        assert public["candidate_facts"]
        assert public["evidence_anchors"]
        for fact in public["candidate_facts"]:
            assert fact["source_section"]
            assert fact["evidence_anchor_id"]
            assert fact["evidence_snippet"]
            assert fact["uncertainty"]
        assert public["source_visible_body"] not in json.dumps(public["candidate_facts"])


def test_unknown_and_uncertainty_are_visible() -> None:
    models = build_all_review_surface_view_models()
    mixed = next(model for model in models if model.package_family == "mixed_narrative_numeric_result")
    assert "Collection Time" in mixed.unknown_values
    assert any("missing or not visible" in flag for flag in mixed.uncertainty_flags)
    for model in models:
        assert model.uncertainty_flags


def test_safety_invariants_preserved() -> None:
    report = build_review_surface_report()
    summary = report["summary"]
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True
    assert summary["all_review_surface_invariants_passed"] is True
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0


def test_markdown_preview_is_visible_not_collapsed_only() -> None:
    for model in build_all_review_surface_view_models():
        markdown = render_review_surface_markdown(model)
        assert "### Source Visible Body" in markdown
        assert "### Candidate Facts" in markdown
        assert "### Unknown Values" in markdown
        assert "Review required" in markdown
        assert "Auto-accept" in markdown
        assert "Live call made" in markdown
        assert check_public_report_payload(markdown).passed


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_quality_review_surface_15o_b.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected = {
        "summary.json",
        "implementation_report.md",
        "review_surface_cases.json",
        "review_surface_matrix.md",
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


def test_report_summary_contains_required_counts() -> None:
    subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_quality_review_surface_15o_b.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["source_visible_body_present_count"] == 4
    assert summary["evidence_anchor_present_count"] == 4
    assert summary["candidate_facts_separated_count"] == 4
    assert summary["unknown_values_explicit_count"] == 4
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0
    assert summary["billing_check_pending"] is True
