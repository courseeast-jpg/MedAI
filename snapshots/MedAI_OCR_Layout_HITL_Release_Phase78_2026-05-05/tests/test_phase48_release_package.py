from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase48_release_snapshot_validation import REQUIRED_FILES, run_validation
from validation_baselines.compare_holdout_baseline import (
    tracked_report_archive_or_review_files,
    tracked_report_phi_files,
)


PACKAGE_DIR = Path("reports/phase48_operator_release_package")


def test_phase48_required_release_files_exist():
    for path in REQUIRED_FILES.values():
        assert path.exists(), path


def test_phase48_release_snapshot_schema_valid():
    snapshot = json.loads((PACKAGE_DIR / "release_snapshot.json").read_text(encoding="utf-8"))

    assert snapshot["snapshot_id"] == "MedAI_Snapshot_Phase48_2026-05-01"
    assert snapshot["release_name"] == "MedAI v2 OCR/Layout HITL Release"
    assert snapshot["completed_phases"] == list(range(37, 49))
    assert snapshot["final_metrics"]["accepted"] == 2
    assert snapshot["final_metrics"]["review"] == 6


def test_phase48_no_report_pdfs_or_phi_artifacts_tracked():
    assert tracked_report_phi_files() == []
    assert tracked_report_archive_or_review_files() == []


def test_phase48_count_convention_documented():
    guide = (PACKAGE_DIR / "operator_review_guide.md").read_text(encoding="utf-8")
    snapshot = (PACKAGE_DIR / "release_snapshot.md").read_text(encoding="utf-8")

    assert "total == accepted + review" in guide
    assert "review_ocr_quality" in guide
    assert "subsets of `review`" in snapshot


def test_phase48_safety_guarantees_documented():
    summary = (PACKAGE_DIR / "medai_v2_ocr_layout_release_summary.md").read_text(encoding="utf-8")

    assert "Poor OCR cannot become accepted" in summary
    assert "Empty extraction cannot become accepted" in summary
    assert "Lab normalization cannot produce accepted status" in summary
    assert "Cyrillic non-lab reconciliation cannot produce accepted status" in summary


def test_phase48_release_validation_passes():
    report = run_validation()

    assert report["conclusion"] == "release_snapshot_ready"
    assert report["phase47_release_candidate_ready"] is True
    assert report["phi_artifact_check"]["passed"] is True
    assert report["release_snapshot_schema_valid"] is True
