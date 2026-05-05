from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import run_phase73_operator_feedback_bypass_decision as phase73


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase70(tmp_path: Path) -> Path:
    p = tmp_path / "phase70" / "phase70_post_diagnostics_decision_audit_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 70,
        "conclusion": "post_diagnostics_decision_audit_complete",
        "recommended_next_phase": "Phase71 Operator Feedback Completion",
        "closed_branches": [
            {"branch": "pdf_ocr_preprocessing",
             "status": "closed_to_manual_review_boundary",
             "evidence": "Phase67 retained manual-review boundary."},
        ],
        "open_branches": [
            {"branch": "operator_feedback_completion", "status": "open",
             "evidence": "Phase54 reviewed_files=0; not_reviewed_files=15."},
            {"branch": "manual_review_package_improvement", "status": "open",
             "evidence": "Diagnostics repeatedly retain review boundaries."},
        ],
        "deferred_branches": [
            {"branch": "docx_support_triage_or_prototype", "status": "deferred",
             "evidence": "No evidence shows it outranks operator feedback."},
            {"branch": "another_ocr_sandbox", "status": "deferred",
             "evidence": "Phase67 and Phase69 both retained manual-review boundary."},
            {"branch": "production_ocr_or_extractor_change",
             "status": "deferred_blocked_by_evidence",
             "evidence": "No diagnostic justifies changing OCR or extraction."},
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _make_phase72(tmp_path: Path, pending: int = 15, unresolved_hp: int = 3) -> Path:
    p = tmp_path / "phase72" / "phase72_operator_feedback_collection_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 72,
        "conclusion": "operator_feedback_collection_initialized",
        "pending_count": pending,
        "reviewed_count": 0,
        "unresolved_high_priority_count": unresolved_hp,
        "pending_safe_ids": [f"file_{i:03d}" for i in range(1, pending + 1)],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _run(tmp_path: Path, **kwargs) -> dict:
    p70 = kwargs.pop("phase70_path", _make_phase70(tmp_path))
    p72 = kwargs.pop("phase72_path", _make_phase72(tmp_path))
    report_dir = kwargs.pop("report_dir", tmp_path / "reports")
    return phase73.run_decision(
        phase70_path=p70,
        phase72_path=p72,
        report_dir=report_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Script runs from current repo state
# ---------------------------------------------------------------------------


def test_phase73_runs_with_real_reports():
    """Phase73 can run against the real repo reports without error."""
    report = phase73.run_decision()
    assert report["phase"] == 73
    assert report["conclusion"] == "operator_feedback_bypass_ready"


# ---------------------------------------------------------------------------
# 2. Does not require operator feedback to be complete
# ---------------------------------------------------------------------------


def test_does_not_require_complete_operator_feedback(tmp_path: Path):
    report = _run(tmp_path)
    assert report["operator_feedback_required_for_next_phase"] is False
    assert report["pending_operator_review_count"] > 0


# ---------------------------------------------------------------------------
# 3. Fails clearly if Phase70 or Phase72 reports are missing
# ---------------------------------------------------------------------------


def test_fails_if_phase70_missing(tmp_path: Path):
    p72 = _make_phase72(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase70"):
        phase73.run_decision(
            phase70_path=tmp_path / "no_such_phase70.json",
            phase72_path=p72,
            report_dir=tmp_path / "reports",
        )


def test_fails_if_phase72_missing(tmp_path: Path):
    p70 = _make_phase70(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase72"):
        phase73.run_decision(
            phase70_path=p70,
            phase72_path=tmp_path / "no_such_phase72.json",
            report_dir=tmp_path / "reports",
        )


# ---------------------------------------------------------------------------
# 4. operator_feedback_status is "deferred_by_user"
# ---------------------------------------------------------------------------


def test_operator_feedback_status_deferred(tmp_path: Path):
    report = _run(tmp_path)
    assert report["operator_feedback_status"] == "deferred_by_user"


# ---------------------------------------------------------------------------
# 5. operator_feedback_required_for_next_phase is false
# ---------------------------------------------------------------------------


def test_operator_feedback_not_required(tmp_path: Path):
    report = _run(tmp_path)
    assert report["operator_feedback_required_for_next_phase"] is False


# ---------------------------------------------------------------------------
# 6. labels_fabricated is false
# ---------------------------------------------------------------------------


def test_labels_fabricated_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["labels_fabricated"] is False


# ---------------------------------------------------------------------------
# 7. private_feedback_file_modified is false
# ---------------------------------------------------------------------------


def test_private_feedback_file_not_modified(tmp_path: Path):
    report = _run(tmp_path)
    assert report["private_feedback_file_modified"] is False


def test_real_private_feedback_file_unchanged():
    """The real operator_feedback_PRIVATE.json must not be touched by Phase73."""
    real_private = (
        Path(__file__).resolve().parents[1]
        / "reports" / "phase72_operator_feedback_collection"
        / "operator_feedback_PRIVATE.json"
    )
    if not real_private.exists():
        pytest.skip("Real private feedback file not present (gitignored)")
    mtime_before = real_private.stat().st_mtime
    phase73.run_decision()
    mtime_after = real_private.stat().st_mtime
    assert mtime_before == mtime_after


# ---------------------------------------------------------------------------
# 8–10. Production flags are false
# ---------------------------------------------------------------------------


def test_production_extractor_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_extractor_should_change_yet"] is False


def test_production_ocr_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_ocr_should_change_yet"] is False


def test_safety_gates_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["safety_gates_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 11. Manual-review boundary retained
# ---------------------------------------------------------------------------


def test_manual_review_boundary_retained(tmp_path: Path):
    report = _run(tmp_path)
    assert report["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# 12. External API used is false
# ---------------------------------------------------------------------------


def test_external_api_used_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


# ---------------------------------------------------------------------------
# 13–17. Privacy checks on public reports
# ---------------------------------------------------------------------------


def test_public_reports_no_raw_filenames(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = (
        (report_dir / phase73.JSON_REPORT.name).read_text(encoding="utf-8")
        + (report_dir / phase73.MD_REPORT.name).read_text(encoding="utf-8")
    )
    assert "Patient Jane Doe" not in combined
    assert "local_filename_mapping_PRIVATE" not in combined


def test_public_reports_no_raw_paths(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = (
        (report_dir / phase73.JSON_REPORT.name).read_text(encoding="utf-8")
        + (report_dir / phase73.MD_REPORT.name).read_text(encoding="utf-8")
    )
    assert "full_corpus_input" not in combined
    assert "original_relative_path" not in combined


def test_public_reports_no_ocr_text(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = (
        (report_dir / phase73.JSON_REPORT.name).read_text(encoding="utf-8")
        + (report_dir / phase73.MD_REPORT.name).read_text(encoding="utf-8")
    )
    assert '"ocr_text":' not in combined
    assert "Glucose 103" not in combined


def test_public_reports_no_extracted_medical_text(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = (
        (report_dir / phase73.JSON_REPORT.name).read_text(encoding="utf-8")
        + (report_dir / phase73.MD_REPORT.name).read_text(encoding="utf-8")
    )
    assert '"extracted_text":' not in combined


def test_public_reports_no_phi(tmp_path: Path):
    report = _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = (
        (report_dir / phase73.JSON_REPORT.name).read_text(encoding="utf-8")
        + (report_dir / phase73.MD_REPORT.name).read_text(encoding="utf-8")
    )
    assert "SSN 999" not in combined
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


# ---------------------------------------------------------------------------
# 18–20. Decision matrix exclusions
# ---------------------------------------------------------------------------


def _selected(report: dict) -> str:
    for c in report["autonomous_decision_matrix"]:
        if c["selected"]:
            return c["branch"]
    return ""


def _all_branches(report: dict) -> list[str]:
    return [c["branch"] for c in report["autonomous_decision_matrix"]]


def test_resume_operator_review_not_selected(tmp_path: Path):
    report = _run(tmp_path)
    assert _selected(report) != "resume_operator_review"


def test_another_ocr_sandbox_not_selected(tmp_path: Path):
    report = _run(tmp_path)
    assert _selected(report) != "another_ocr_sandbox"


def test_production_change_not_selected(tmp_path: Path):
    report = _run(tmp_path)
    assert _selected(report) != "production_ocr_or_extractor_change"


# ---------------------------------------------------------------------------
# 21. Recommended next phase is deterministic
# ---------------------------------------------------------------------------


def test_recommended_next_phase_is_deterministic(tmp_path: Path):
    p70 = _make_phase70(tmp_path)
    p72 = _make_phase72(tmp_path)

    r1 = phase73.run_decision(
        phase70_path=p70, phase72_path=p72, report_dir=tmp_path / "r1"
    )
    r2 = phase73.run_decision(
        phase70_path=p70, phase72_path=p72, report_dir=tmp_path / "r2"
    )
    assert r1["recommended_next_phase"] == r2["recommended_next_phase"]
    assert r1["recommended_next_phase"] == "Phase74 Manual Review Package Auto-Improvement"


# ---------------------------------------------------------------------------
# 22. Phase73 tests pass even if test_phase2_validation is failing
# ---------------------------------------------------------------------------


def test_phase73_conclusion_is_bypass_ready(tmp_path: Path):
    """Isolated Phase73 passes regardless of test_phase2_validation state."""
    report = _run(tmp_path)
    assert report["conclusion"] == "operator_feedback_bypass_ready"
    assert report["phase"] == 73
    assert report["phase_name"] == "Operator Feedback Bypass + Autonomous Next-Action Selection"
