"""Tests for Phase 78 — Final HITL Release Snapshot / Release Freeze.

Tests 9–25 of the Phase76–78 combined block spec.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from scripts import run_phase78_final_hitl_release_freeze as phase78


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_src(tmp_path: Path) -> Path:
    """Create a minimal source tree to snapshot (no PDFs, no private files)."""
    src = tmp_path / "src"
    # Safe code file
    (src / "scripts").mkdir(parents=True)
    (src / "scripts" / "run_demo.py").write_text("print('hello')", encoding="utf-8")
    # Safe report
    (src / "reports" / "phase74_test").mkdir(parents=True)
    (src / "reports" / "phase74_test" / "report.json").write_text(
        json.dumps({"phase": 74, "conclusion": "ok"}), encoding="utf-8"
    )
    # Release docs
    (src / "RELEASE_OPERATOR_GUIDE.md").write_text(
        "local-only. not a medical device. not production-autonomous. "
        "manual-review boundary is retained. review package.",
        encoding="utf-8"
    )
    # Things that should be excluded
    (src / "real_validation_input").mkdir(parents=True)
    (src / "real_validation_input" / "patient.pdf").write_bytes(b"%PDF")
    priv = src / "reports" / "phase72"
    priv.mkdir(parents=True)
    (priv / "operator_feedback_PRIVATE.json").write_text("{}", encoding="utf-8")
    return src


def _run(tmp_path: Path, **kwargs) -> dict:
    src = kwargs.pop("_root", _minimal_src(tmp_path))
    snap_dir = kwargs.pop("snapshots_dir", tmp_path / "snapshots")
    report_dir = kwargs.pop("report_dir", tmp_path / "phase78_reports")
    return phase78.run_freeze(
        _root=src,
        snapshots_dir=snap_dir,
        report_dir=report_dir,
        final_test_count=kwargs.pop("final_test_count", 782),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 9. Phase78 script exists and runs
# ---------------------------------------------------------------------------


def test_phase78_script_exists():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_phase78_final_hitl_release_freeze.py"
    assert p.exists()


def test_phase78_runs(tmp_path: Path):
    report = _run(tmp_path)
    assert report["phase"] == 78
    assert report["conclusion"] == "hitl_release_frozen"


# ---------------------------------------------------------------------------
# 10. Phase78 creates snapshot folder
# ---------------------------------------------------------------------------


def test_snapshot_folder_created(tmp_path: Path):
    report = _run(tmp_path)
    assert report["snapshot_created"] is True
    assert Path(report["snapshot_path"]).exists()


# ---------------------------------------------------------------------------
# 11. Phase78 creates snapshot zip
# ---------------------------------------------------------------------------


def test_snapshot_zip_created(tmp_path: Path):
    report = _run(tmp_path)
    assert report["snapshot_zip_created"] is True
    assert Path(report["snapshot_zip_path"]).exists()
    # Verify it's a valid zip
    assert zipfile.is_zipfile(report["snapshot_zip_path"])


# ---------------------------------------------------------------------------
# 12. Snapshot excludes PDFs/images/private mappings/.env
# ---------------------------------------------------------------------------


def test_snapshot_excludes_pdfs(tmp_path: Path):
    report = _run(tmp_path)
    snap = Path(report["snapshot_path"])
    bad = [f for f in snap.rglob("*") if f.suffix.lower() in (".pdf", ".png", ".jpg")]
    assert bad == [], f"Snapshot contains forbidden files: {bad}"


def test_snapshot_excludes_private_mappings(tmp_path: Path):
    report = _run(tmp_path)
    snap = Path(report["snapshot_path"])
    priv = [
        f for f in snap.rglob("*")
        if "PRIVATE" in f.name and not f.name.endswith(".example.json")
    ]
    assert priv == [], f"Snapshot contains private files: {priv}"


def test_snapshot_excludes_real_patient_data(tmp_path: Path):
    report = _run(tmp_path)
    snap = Path(report["snapshot_path"])
    real_input = snap / "real_validation_input"
    pdfs_in_real = list(real_input.glob("*.pdf")) if real_input.exists() else []
    assert pdfs_in_real == []


# ---------------------------------------------------------------------------
# 13. Snapshot metadata includes release status and commit hash
# ---------------------------------------------------------------------------


def test_snapshot_metadata_has_release_status(tmp_path: Path):
    report = _run(tmp_path)
    meta_file = Path(report["snapshot_path"]) / "SNAPSHOT_METADATA.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["release_status"] == "FROZEN_HITL_RELEASE"
    assert meta["release_name"] == "MedAI OCR/Layout HITL Release"
    assert "commit_hash" in meta
    assert meta["not_production_autonomous"] is True
    assert meta["not_medical_device"] is True
    assert meta["clinical_diagnosis_provided"] is False
    assert meta["local_only_default"] is True
    assert meta["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# 14–18. Public reports privacy checks
# ---------------------------------------------------------------------------


def _all_public_text(tmp_path: Path, report: dict) -> str:
    combined = ""
    for dir_path in [tmp_path / "phase76_reports", tmp_path / "phase77_reports",
                     tmp_path / "phase78_reports"]:
        if dir_path.exists():
            for f in dir_path.glob("*.json"):
                combined += f.read_text(encoding="utf-8", errors="replace")
            for f in dir_path.glob("*.md"):
                combined += f.read_text(encoding="utf-8", errors="replace")
    return combined


def test_public_reports_no_raw_filenames(tmp_path: Path):
    report = _run(tmp_path)
    text = _all_public_text(tmp_path, report)
    assert "Patient Jane Doe" not in text
    assert "local_filename_mapping_PRIVATE" not in text


def test_public_reports_no_raw_paths(tmp_path: Path):
    report = _run(tmp_path)
    text = _all_public_text(tmp_path, report)
    assert "original_relative_path" not in text


def test_public_reports_no_ocr_text(tmp_path: Path):
    report = _run(tmp_path)
    text = _all_public_text(tmp_path, report)
    assert '"ocr_text":' not in text
    assert "Glucose 103" not in text


def test_public_reports_no_extracted_medical_text(tmp_path: Path):
    report = _run(tmp_path)
    text = _all_public_text(tmp_path, report)
    assert '"extracted_text":' not in text


def test_public_reports_no_phi(tmp_path: Path):
    report = _run(tmp_path)
    text = _all_public_text(tmp_path, report)
    assert "SSN 999" not in text
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


# ---------------------------------------------------------------------------
# 19–23. Safety flags throughout
# ---------------------------------------------------------------------------


def test_production_extractor_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_extractor_should_change_yet"] is False


def test_production_ocr_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_ocr_should_change_yet"] is False


def test_safety_gates_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["safety_gates_should_change_yet"] is False


def test_manual_review_boundary_retained(tmp_path: Path):
    report = _run(tmp_path)
    assert report["manual_review_boundary_retained"] is True


def test_external_api_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"


# ---------------------------------------------------------------------------
# 24. Final recommendation does not start a new OCR/extractor branch
# ---------------------------------------------------------------------------


def test_no_new_ocr_branch_recommended(tmp_path: Path):
    report = _run(tmp_path)
    json_text = json.dumps(report)
    assert "ocr_sandbox" not in json_text.lower()
    assert report["production_ocr_should_change_yet"] is False
    assert report["production_extractor_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 25. Final recommendation says branch is complete/frozen
# ---------------------------------------------------------------------------


def test_branch_is_frozen(tmp_path: Path):
    report = _run(tmp_path)
    assert report["release_frozen"] is True
    assert report["conclusion"] == "hitl_release_frozen"
    assert report["release_status"] == "FROZEN_HITL_RELEASE"
