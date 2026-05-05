from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_phase72_operator_feedback_collection as phase72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue(tmp_path: Path, count: int = 5) -> Path:
    items = []
    for i in range(1, count + 1):
        tier = 1 if i <= 2 else 2
        items.append({
            "safe_file_id": f"file_{i:03d}",
            "priority_rank": i,
            "priority_tier": tier,
            "suspected_problem_class": "ocr_quality_gate_trigger" if i == 1 else "unknown_document_class",
            "source_phase": "phase54",
            "review_goal": f"Review goal for file {i}.",
            "operator_question": f"Question for file {i}?",
            "allowed_answers": list(phase72.ALLOWED_ANSWERS),
            "development_impact": "test impact",
            "should_open_original_file": True,
            "notes_allowed_private_only": True,
        })
    queue_path = tmp_path / "operator_review_queue_SAFE.json"
    queue_path.write_text(json.dumps({"review_queue": items}), encoding="utf-8")
    return queue_path


def _run(tmp_path: Path, mode: str = "init_and_summarize", **kwargs) -> dict:
    queue_path = kwargs.pop("queue_path", _make_queue(tmp_path))
    private_path = kwargs.pop("private_path", tmp_path / "operator_feedback_PRIVATE.json")
    report_dir = kwargs.pop("report_dir", tmp_path / "reports")
    return phase72.run_collection(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=report_dir,
        mode=mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------------


def test_phase72_runs_with_existing_reports(tmp_path: Path):
    report = _run(tmp_path)

    assert report["phase"] == 72
    assert report["phase_name"] == "Operator Feedback Collection Pass"
    assert report["conclusion"] == "operator_feedback_collection_initialized"
    assert (tmp_path / "reports" / phase72.JSON_REPORT.name).exists()
    assert (tmp_path / "reports" / phase72.MD_REPORT.name).exists()


def test_phase72_fails_clearly_if_phase71_queue_missing(tmp_path: Path):
    missing = tmp_path / "no_such_queue.json"

    with pytest.raises(FileNotFoundError, match="Phase71 review queue not found"):
        phase72.run_collection(
            queue_path=missing,
            private_path=tmp_path / "feedback_PRIVATE.json",
            report_dir=tmp_path / "reports",
        )


def test_init_creates_private_feedback_structure(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = tmp_path / "feedback_PRIVATE.json"

    phase72.run_collection(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=tmp_path / "reports",
        mode="init",
    )

    assert private_path.exists()
    payload = json.loads(private_path.read_text(encoding="utf-8"))
    assert "feedback" in payload
    assert payload["phase"] == 72
    assert all(r["status"] == "pending" for r in payload["feedback"])
    assert all(r["answer"] is None for r in payload["feedback"])


def test_record_accepts_valid_answer(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = tmp_path / "feedback_PRIVATE.json"

    _run(tmp_path, "init", queue_path=queue_path, private_path=private_path,
         report_dir=tmp_path / "r")

    phase72.record_answer("file_001", "correct_review", private_path=private_path)

    payload = json.loads(private_path.read_text(encoding="utf-8"))
    rec = next(r for r in payload["feedback"] if r["safe_file_id"] == "file_001")
    assert rec["answer"] == "correct_review"
    assert rec["status"] == "reviewed"
    assert rec["reviewed_at"] is not None


def test_record_rejects_invalid_answer(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = tmp_path / "feedback_PRIVATE.json"

    _run(tmp_path, "init", queue_path=queue_path, private_path=private_path,
         report_dir=tmp_path / "r")

    with pytest.raises(ValueError, match="Invalid answer"):
        phase72.record_answer("file_001", "totally_wrong", private_path=private_path)


def test_summarize_generates_public_report_without_private_notes(tmp_path: Path):
    queue_path = _make_queue(tmp_path)
    private_path = tmp_path / "feedback_PRIVATE.json"
    report_dir = tmp_path / "reports"

    _run(tmp_path, "init", queue_path=queue_path, private_path=private_path,
         report_dir=report_dir)
    phase72.record_answer("file_001", "correct_review",
                          private_path=private_path, private_note="my secret note")

    report = phase72.run_collection(
        queue_path=queue_path,
        private_path=private_path,
        report_dir=report_dir,
        mode="summarize",
    )

    json_text = (report_dir / phase72.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "my secret note" not in combined
    assert report["reviewed_count"] == 1
    assert report["pending_count"] == 4
    assert report["conclusion"] == "operator_feedback_collection_in_progress"


# ---------------------------------------------------------------------------
# Privacy tests
# ---------------------------------------------------------------------------


def test_public_reports_contain_no_raw_filenames(tmp_path: Path):
    report = _run(tmp_path)
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "Patient Jane Doe" not in combined
    assert "local_filename_mapping_PRIVATE" not in combined


def test_public_reports_contain_no_raw_paths(tmp_path: Path):
    report = _run(tmp_path)
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "full_corpus_input" not in combined
    assert "original_relative_path" not in combined


def test_public_reports_contain_no_ocr_or_extracted_text(tmp_path: Path):
    report = _run(tmp_path)
    report_dir = tmp_path / "reports"
    json_text = (report_dir / phase72.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase72.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "Glucose 103" not in combined
    assert "SSN 999" not in combined
    assert '"ocr_text":' not in combined
    assert '"extracted_text":' not in combined


# ---------------------------------------------------------------------------
# Safety flag tests
# ---------------------------------------------------------------------------


def test_external_api_used_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


def test_production_extractor_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_extractor_should_change_yet"] is False


def test_production_ocr_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_ocr_should_change_yet"] is False


def test_safety_gates_should_not_change(tmp_path: Path):
    report = _run(tmp_path)
    assert report["safety_gates_should_change_yet"] is False


def test_manual_review_boundary_retained(tmp_path: Path):
    report = _run(tmp_path)
    assert report["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# Gitignore test
# ---------------------------------------------------------------------------


def test_private_feedback_file_pattern_is_gitignored():
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = repo_root / ".gitignore"
    assert gitignore.exists()
    text = gitignore.read_text(encoding="utf-8")
    assert (
        "operator_feedback_PRIVATE.json" in text
        or "phase72_operator_feedback_collection/operator_feedback_PRIVATE" in text
    )


# ---------------------------------------------------------------------------
# Determinism / ID tests
# ---------------------------------------------------------------------------


def test_queue_ordering_remains_deterministic(tmp_path: Path):
    queue_path = _make_queue(tmp_path, count=10)
    private_a = tmp_path / "feedback_a.json"
    private_b = tmp_path / "feedback_b.json"

    phase72.run_collection(
        queue_path=queue_path, private_path=private_a,
        report_dir=tmp_path / "r_a", mode="init",
    )
    phase72.run_collection(
        queue_path=queue_path, private_path=private_b,
        report_dir=tmp_path / "r_b", mode="init",
    )

    a_ids = [r["safe_file_id"] for r in json.loads(private_a.read_text())["feedback"]]
    b_ids = [r["safe_file_id"] for r in json.loads(private_b.read_text())["feedback"]]
    assert a_ids == b_ids


def test_only_safe_file_ids_in_queue_output(tmp_path: Path):
    report = _run(tmp_path)
    report_dir = tmp_path / "reports"
    queue_text = json.dumps(report.get("pending_safe_ids", []))

    assert ".pdf" not in queue_text
    assert ".jpg" not in queue_text
    assert "original_relative_path" not in queue_text
    assert "Patient" not in queue_text


# ---------------------------------------------------------------------------
# Branch recommendation tests
# ---------------------------------------------------------------------------


def test_ocr_sandbox_not_recommended(tmp_path: Path):
    report = _run(tmp_path)
    rec = report["recommended_next_phase"].lower()
    assert "ocr_sandbox" not in rec
    assert report["production_ocr_should_change_yet"] is False


def test_production_extractor_change_not_recommended(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False


def test_phase72_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase72_operator_feedback_collection"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []
