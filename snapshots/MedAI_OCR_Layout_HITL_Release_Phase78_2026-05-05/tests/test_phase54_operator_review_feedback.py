from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.run_phase54_operator_review_feedback_summary import (
    JSON_REPORT,
    PRIVATE_FEEDBACK,
    run_summary,
)


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase54_operator_review_feedback_summary.py"


def write_phase53_report(path: Path) -> dict:
    payload = {
        "run_id": "synthetic-phase53",
        "total_files": 3,
        "results": [
            {
                "file_id": "file_001",
                "filename_hash": "hash001",
                "status": "accepted",
                "validation_status": "accepted",
                "ocr_status": "good",
            },
            {
                "file_id": "file_002",
                "filename_hash": "hash002",
                "status": "review",
                "validation_status": "needs_review",
                "ocr_status": "good",
            },
            {
                "file_id": "file_003",
                "filename_hash": "hash003",
                "status": "review_ocr_quality",
                "validation_status": "needs_review",
                "ocr_status": "poor_ocr",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_phase54_handles_missing_phase53_report(tmp_path: Path):
    report = run_summary(
        phase53_report_path=tmp_path / "missing_phase53.json",
        report_dir=tmp_path / "reports" / "phase54",
    )

    assert report["conclusion"] == "phase53_report_missing"
    assert report["external_api_used"] is False
    assert report["phase53_report_modified"] is False


def test_phase54_handles_phase53_report_with_no_feedback(tmp_path: Path):
    phase53 = tmp_path / "phase53.json"
    report_dir = tmp_path / "reports" / "phase54"
    write_phase53_report(phase53)

    report = run_summary(phase53_report_path=phase53, report_dir=report_dir)

    assert report["global_summary"]["total_files"] == 3
    assert report["global_summary"]["not_reviewed_files"] == 3
    assert report["class_summary"]["unknown_other"]["not_reviewed_count"] == 3
    assert report["conclusion"] == "review_feedback_incomplete"


def test_phase54_merges_feedback_by_safe_file_id_and_filename_hash(tmp_path: Path):
    phase53 = tmp_path / "phase53.json"
    report_dir = tmp_path / "reports" / "phase54"
    feedback = report_dir / PRIVATE_FEEDBACK.name
    write_phase53_report(phase53)
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text(
        json.dumps(
            {
                "feedback": [
                    {
                        "safe_file_id": "file_001",
                        "filename_hash": "hash001",
                        "operator_verdict": "incorrect",
                        "operator_document_class": "lab_report",
                        "operator_reason": "false_accept",
                        "operator_note": "PRIVATE: patient name should never be public",
                        "reviewed_at": "2026-05-03T00:00:00+00:00",
                    },
                    {
                        "safe_file_id": "file_002",
                        "filename_hash": "hash002",
                        "operator_verdict": "correct",
                        "operator_document_class": "lab_report",
                        "operator_reason": "false_review",
                        "operator_note": "PRIVATE: DOB should never be public",
                        "reviewed_at": "2026-05-03T00:01:00+00:00",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = run_summary(phase53_report_path=phase53, report_dir=report_dir, private_feedback_path=feedback)

    records = {row["safe_file_id"]: row for row in report["records"]}
    assert records["file_001"]["operator_verdict"] == "incorrect"
    assert records["file_001"]["operator_document_class"] == "lab_report"
    assert records["file_002"]["operator_reason"] == "false_review"
    assert records["file_003"]["operator_verdict"] == "not_reviewed"


def test_phase54_public_reports_exclude_raw_filenames_and_private_notes(tmp_path: Path):
    phase53 = tmp_path / "phase53.json"
    report_dir = tmp_path / "reports" / "phase54"
    payload = write_phase53_report(phase53)
    payload["results"][0]["raw_filename_should_not_exist"] = "Jane Doe Labs.pdf"
    phase53.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    feedback = report_dir / PRIVATE_FEEDBACK.name
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text(
        json.dumps(
            {
                "feedback": [
                    {
                        "safe_file_id": "file_001",
                        "filename_hash": "hash001",
                        "operator_verdict": "uncertain",
                        "operator_document_class": "unknown_other",
                        "operator_reason": "other",
                        "operator_note": "PRIVATE John Smith DOB 01/02/1970",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_summary(phase53_report_path=phase53, report_dir=report_dir, private_feedback_path=feedback)
    public_json = (report_dir / JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / "phase54_operator_review_feedback_report.md").read_text(encoding="utf-8")

    assert "Jane Doe Labs.pdf" not in public_json
    assert "PRIVATE John Smith" not in public_json
    assert "DOB 01/02/1970" not in public_md


def test_phase54_private_feedback_file_is_ignored_by_git():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase54_operator_review_feedback/operator_feedback_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0


def test_phase54_class_level_and_false_counts_are_correct(tmp_path: Path):
    phase53 = tmp_path / "phase53.json"
    report_dir = tmp_path / "reports" / "phase54"
    feedback = report_dir / PRIVATE_FEEDBACK.name
    write_phase53_report(phase53)
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text(
        json.dumps(
            {
                "feedback": [
                    {
                        "safe_file_id": "file_001",
                        "filename_hash": "hash001",
                        "operator_verdict": "incorrect",
                        "operator_document_class": "lab_report",
                        "operator_reason": "false_accept",
                    },
                    {
                        "safe_file_id": "file_002",
                        "filename_hash": "hash002",
                        "operator_verdict": "correct",
                        "operator_document_class": "lab_report",
                        "operator_reason": "false_review",
                    },
                    {
                        "safe_file_id": "file_003",
                        "filename_hash": "hash003",
                        "operator_verdict": "uncertain",
                        "operator_document_class": "prescription",
                        "operator_reason": "ocr_quality_issue",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_summary(phase53_report_path=phase53, report_dir=report_dir, private_feedback_path=feedback)

    assert report["class_summary"]["lab_report"]["total"] == 2
    assert report["class_summary"]["lab_report"]["accepted"] == 1
    assert report["class_summary"]["lab_report"]["review"] == 1
    assert report["class_summary"]["lab_report"]["false_accept_count"] == 1
    assert report["class_summary"]["lab_report"]["false_review_count"] == 1
    assert report["class_summary"]["prescription"]["review_ocr_quality"] == 1
    assert report["class_summary"]["prescription"]["ocr_quality_issue_count"] == 1
    assert report["global_summary"]["false_accept_count"] == 1
    assert report["global_summary"]["false_review_count"] == 1


def test_phase54_ui_labels_reference_phase54_correctly():
    source = APP_MAIN.read_text(encoding="utf-8")

    assert "Phase54 Operator Review Feedback" in source
    assert "Generate Phase54 Class-Level Review Summary" in source
    assert "operator_feedback_PRIVATE.json" in source


def test_phase54_does_not_call_external_apis_or_alter_phase53_results(tmp_path: Path):
    phase53 = tmp_path / "phase53.json"
    report_dir = tmp_path / "reports" / "phase54"
    original = write_phase53_report(phase53)
    before = phase53.read_text(encoding="utf-8")

    report = run_summary(phase53_report_path=phase53, report_dir=report_dir)
    after = phase53.read_text(encoding="utf-8")
    script_source = SCRIPT_PATH.read_text(encoding="utf-8").lower()

    assert report["external_api_used"] is False
    assert report["phase53_report_modified"] is False
    assert before == after
    assert json.loads(after) == original
    assert "generativeai" not in script_source
    assert "openai" not in script_source
    assert "anthropic" not in script_source
