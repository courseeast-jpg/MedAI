"""Focused tests for MEDAI-AI-PACKAGE-QUALITY-RUN-REVIEW-WIRING-15O-C."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import app.main as main
from app.ai_package_run_review_preview import (
    build_run_review_package_previews,
    build_run_review_preview_report,
    render_run_review_preview_markdown,
    run_review_preview_to_public_dict,
)
from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_package_quality_eval import PACKAGE_FAMILIES

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_package_quality_run_review_wiring_15o_c"


def test_run_review_preview_models_cover_all_package_families() -> None:
    previews = build_run_review_package_previews()
    assert [item.package_family for item in previews] == list(PACKAGE_FAMILIES)
    for preview in previews:
        assert preview.entry_point_label == "AI package review preview"
        assert preview.package_family_label
        assert preview.source_visible_body
        assert preview.source_sections
        assert preview.candidate_facts
        assert preview.evidence_anchors
        assert preview.uncertainty_flags


def test_run_review_preview_has_required_operator_visible_elements() -> None:
    for preview in build_run_review_package_previews():
        public = run_review_preview_to_public_dict(preview)
        assert public["package_family_label"]
        assert public["review_required"] is True
        assert public["source_visible_body"]
        assert public["source_sections"]
        assert public["candidate_facts"]
        assert public["evidence_anchors"]
        assert public["uncertainty_flags"]
        assert public["no_live_provider_off_indicator"] == "no-live/provider-off"
        assert public["active_written_count_indicator"] == "active_written_count=0"
        assert public["auto_accept_indicator"] == "auto_accept=false"
        assert public["source_visible_body"] not in json.dumps(public["candidate_facts"])


def test_unknown_values_and_uncertainty_visible() -> None:
    previews = build_run_review_package_previews()
    mixed = next(item for item in previews if item.package_family == "mixed_narrative_numeric_result")
    assert "Collection Time" in mixed.unknown_values
    assert any("missing or not visible" in flag for flag in mixed.uncertainty_flags)
    for preview in previews:
        markdown = render_run_review_preview_markdown(preview)
        assert "### Source Visible Body" in markdown
        assert "### Candidate Facts" in markdown
        assert "### Unknown / Missing Values" in markdown
        assert "### Uncertainty Flags" in markdown
        assert check_public_report_payload(markdown).passed


def test_safety_invariants_preserved() -> None:
    report = build_run_review_preview_report()
    summary = report["summary"]
    assert summary["live_call_made"] is False
    assert summary["external_api_used"] is False
    assert summary["active_written_count"] == 0
    assert summary["auto_accept"] is False
    assert summary["review_required"] is True
    assert summary["all_run_review_preview_invariants_passed"] is True
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0


def test_run_review_tab_contains_preview_entry_point() -> None:
    source = inspect.getsource(main.render_run_review_tab)
    assert "render_ai_package_run_review_preview_panel" in source
    assert "app.ai_package_run_review_preview" in source


def test_old_provider_gates_are_not_required_or_changed() -> None:
    source = (REPO_ROOT / "app" / "ai_package_run_review_preview.py").read_text(encoding="utf-8")
    assert "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED" not in source
    assert "GEMINI_API_KEY" not in source
    assert "run_gemini_vertex_live_smoke" not in source
    assert "run_gemini_live_smoke" not in source


def test_reports_generated_and_public_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_ai_package_quality_run_review_wiring_15o_c.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected = {
        "summary.json",
        "implementation_report.md",
        "run_review_surface_cases.json",
        "run_review_surface_matrix.md",
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
        [sys.executable, "scripts/run_medai_ai_package_quality_run_review_wiring_15o_c.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["privacy_result"] == "passed"
    assert summary["run_review_package_preview_entry_point_added"] is True
    assert summary["source_visible_body_present_count"] == 4
    assert summary["evidence_anchor_present_count"] == 4
    assert summary["candidate_facts_separated_count"] == 4
    assert summary["unknown_values_explicit_count"] == 4
    assert summary["uncertainty_flags_visible_count"] == 4
    assert summary["under_1_minute_compare_preserved_count"] == 4
    assert summary["hallucinated_field_count"] == 0
    assert summary["billing_check_pending"] is True
