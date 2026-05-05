from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_phase72b_operator_review_console as phase72b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue(tmp_path: Path, count: int = 5) -> Path:
    items = []
    for i in range(1, count + 1):
        tier = 1 if i <= 3 else 2
        items.append({
            "safe_file_id": f"file_{i:03d}",
            "priority_rank": i,
            "priority_tier": tier,
            "suspected_problem_class": (
                "ocr_quality_gate_trigger" if i == 1 else "unknown_document_class"
            ),
            "source_phase": "phase54",
            "review_goal": f"Review goal for file {i}.",
            "operator_question": f"Question for file {i}?",
            "allowed_answers": list(phase72b.ALLOWED_ANSWERS),
            "development_impact": "test impact",
            "should_open_original_file": True,
            "notes_allowed_private_only": True,
        })
    queue_path = tmp_path / "operator_review_queue_SAFE.json"
    queue_path.write_text(json.dumps({"review_queue": items}), encoding="utf-8")
    return queue_path


def _make_feedback(tmp_path: Path, queue_path: Path) -> Path:
    """Create an empty private feedback file mirroring the queue."""
    items = json.loads(queue_path.read_text(encoding="utf-8"))["review_queue"]
    records = [
        {
            "safe_file_id": it["safe_file_id"],
            "priority_tier": it["priority_tier"],
            "priority_rank": it["priority_rank"],
            "suspected_problem_class": it["suspected_problem_class"],
            "source_phase": it["source_phase"],
            "review_goal": it["review_goal"],
            "operator_question": it["operator_question"],
            "status": "pending",
            "answer": None,
            "private_note": None,
            "reviewed_at": None,
        }
        for it in items
    ]
    private_path = tmp_path / "operator_feedback_PRIVATE.json"
    private_path.write_text(
        json.dumps({"phase": 72, "feedback": records}), encoding="utf-8"
    )
    return private_path


def _run(tmp_path: Path, **kwargs) -> dict:
    queue_path = kwargs.pop("queue_path", _make_queue(tmp_path))
    private_path = kwargs.pop("private_path", _make_feedback(tmp_path, queue_path))
    report_dir = kwargs.pop("report_dir", tmp_path / "reports")
    return phase72b.run_console(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=report_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Script exists and runs in non-interactive summary mode
# ---------------------------------------------------------------------------


def test_phase72b_runs_summary_mode(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["phase"] == "72B"
    assert report["phase_name"] == "Human-Minimized Operator Review Console"
    assert report["conclusion"] in (
        "operator_review_console_ready",
        "operator_review_console_complete",
    )


# ---------------------------------------------------------------------------
# 2. Default scope is tier_1_only
# ---------------------------------------------------------------------------


def test_default_scope_is_tier1_only(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["console_default_scope"] == "tier_1_only"


# ---------------------------------------------------------------------------
# 3. Loads Phase71/Phase72 queue correctly
# ---------------------------------------------------------------------------


def test_loads_queue_correctly(tmp_path: Path):
    queue_path = _make_queue(tmp_path, count=5)
    private_path = _make_feedback(tmp_path, queue_path)
    report = phase72b.run_console(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=tmp_path / "reports",
        mode="summary",
    )
    assert report["review_queue_count"] == 5
    assert report["tier_1_pending_count"] == 3
    assert report["tier_2_pending_count"] == 2


# ---------------------------------------------------------------------------
# 4. Does not require reviewing all items at once
# ---------------------------------------------------------------------------


def test_partial_review_is_valid(tmp_path: Path):
    queue_path = _make_queue(tmp_path, count=5)
    private_path = _make_feedback(tmp_path, queue_path)

    phase72b.record_answer_to_file("file_001", "correct_review", private_path=private_path)

    report = phase72b.run_console(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=tmp_path / "reports",
        mode="summary",
    )
    assert report["reviewed_count"] == 1
    assert report["pending_count"] == 4
    assert report["conclusion"] == "operator_review_console_ready"


# ---------------------------------------------------------------------------
# 5. Next pending tier-1 item is deterministic
# ---------------------------------------------------------------------------


def test_get_next_pending_tier1_is_deterministic(tmp_path: Path):
    queue_path = _make_queue(tmp_path, count=5)
    private_path_a = _make_feedback(tmp_path, queue_path)
    private_path_b = tmp_path / "fb_b.json"
    private_path_b.write_text(private_path_a.read_text(encoding="utf-8"), encoding="utf-8")

    def _next(p):
        fb = phase72b.load_private_feedback(p)
        return phase72b.get_next_pending_tier1(fb.get("feedback") or [])

    a = _next(private_path_a)
    b = _next(private_path_b)
    assert a is not None
    assert b is not None
    assert a["safe_file_id"] == b["safe_file_id"]


# ---------------------------------------------------------------------------
# 6. Validates allowed answers
# ---------------------------------------------------------------------------


def test_validate_answer_accepts_all_allowed(tmp_path: Path):
    for ans in phase72b.ALLOWED_ANSWERS:
        phase72b.validate_answer(ans)  # should not raise


# ---------------------------------------------------------------------------
# 7. Rejects invalid answers
# ---------------------------------------------------------------------------


def test_validate_answer_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid answer"):
        phase72b.validate_answer("totally_bogus")


def test_record_answer_rejects_invalid_answer(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = _make_feedback(tmp_path, queue_path)

    with pytest.raises(ValueError, match="Invalid answer"):
        phase72b.record_answer_to_file("file_001", "not_a_real_answer",
                                       private_path=private_path)


# ---------------------------------------------------------------------------
# 8. Writes only to private feedback path when recording
# ---------------------------------------------------------------------------


def test_record_writes_only_to_private_path(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = _make_feedback(tmp_path, queue_path)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    files_before = set(report_dir.iterdir()) if report_dir.exists() else set()
    phase72b.record_answer_to_file("file_001", "correct_accept",
                                   private_path=private_path,
                                   private_note="secret note")
    files_after = set(report_dir.iterdir()) if report_dir.exists() else set()

    assert files_before == files_after  # no new files in report_dir
    payload = json.loads(private_path.read_text(encoding="utf-8"))
    rec = next(r for r in payload["feedback"] if r["safe_file_id"] == "file_001")
    assert rec["answer"] == "correct_accept"
    assert rec["private_note"] == "secret note"


# ---------------------------------------------------------------------------
# 9. Generates safe public summary
# ---------------------------------------------------------------------------


def test_generates_public_summary_without_private_notes(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = _make_feedback(tmp_path, queue_path)
    report_dir = tmp_path / "reports"

    phase72b.record_answer_to_file("file_001", "correct_review",
                                   private_path=private_path,
                                   private_note="very secret")

    phase72b.run_console(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=report_dir,
        mode="summary",
    )

    json_text = (report_dir / phase72b.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72b.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "very secret" not in combined


# ---------------------------------------------------------------------------
# 10. Public reports contain no raw filenames
# ---------------------------------------------------------------------------


def test_public_reports_no_raw_filenames(tmp_path: Path):
    _run(tmp_path, mode="summary")
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72b.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72b.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "Patient Jane Doe" not in combined
    assert "local_filename_mapping_PRIVATE" not in combined


# ---------------------------------------------------------------------------
# 11. Public reports contain no raw paths
# ---------------------------------------------------------------------------


def test_public_reports_no_raw_paths(tmp_path: Path):
    _run(tmp_path, mode="summary")
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72b.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72b.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "full_corpus_input" not in combined
    assert "original_relative_path" not in combined


# ---------------------------------------------------------------------------
# 12. Public reports contain no OCR or extracted text
# ---------------------------------------------------------------------------


def test_public_reports_no_ocr_or_extracted_text(tmp_path: Path):
    _run(tmp_path, mode="summary")
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72b.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72b.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "Glucose 103" not in combined
    assert "SSN 999" not in combined
    assert '"ocr_text":' not in combined
    assert '"extracted_text":' not in combined


# ---------------------------------------------------------------------------
# 13. raw_phi_logged_in_public_reports is False
# ---------------------------------------------------------------------------


def test_raw_phi_logged_is_false(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


# ---------------------------------------------------------------------------
# 14. external_api_used is False
# ---------------------------------------------------------------------------


def test_external_api_used_is_false(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


# ---------------------------------------------------------------------------
# 15. production_extractor_should_change_yet is False
# ---------------------------------------------------------------------------


def test_production_extractor_should_not_change(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["production_extractor_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 16. production_ocr_should_change_yet is False
# ---------------------------------------------------------------------------


def test_production_ocr_should_not_change(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["production_ocr_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 17. safety_gates_should_change_yet is False
# ---------------------------------------------------------------------------


def test_safety_gates_should_not_change(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["safety_gates_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 18. manual_review_boundary_retained is True
# ---------------------------------------------------------------------------


def test_manual_review_boundary_retained(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    assert report["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# 19. real operator_feedback_PRIVATE.json is gitignored
# ---------------------------------------------------------------------------


def test_private_feedback_file_is_gitignored():
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = repo_root / ".gitignore"
    assert gitignore.exists()
    text = gitignore.read_text(encoding="utf-8")
    assert (
        "operator_feedback_PRIVATE.json" in text
        or "phase72_operator_feedback_collection/operator_feedback_PRIVATE" in text
    )


def test_phase72b_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase72b_operator_review_console"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (
        ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
        ".bmp", ".webp", ".docx", ".rtf",
    )
    bad = [line for line in result.stdout.splitlines()
           if line.lower().endswith(forbidden)]
    assert bad == []


# ---------------------------------------------------------------------------
# 20. No OCR sandbox or production change recommended
# ---------------------------------------------------------------------------


def test_ocr_sandbox_not_recommended(tmp_path: Path):
    report = _run(tmp_path, mode="summary")
    rec = report["recommended_next_phase"].lower()
    assert "ocr_sandbox" not in rec
    assert report["production_ocr_should_change_yet"] is False
    assert report["production_extractor_should_change_yet"] is False
