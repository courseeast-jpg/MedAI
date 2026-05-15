"""Tests for Phase 77 — Operator-Facing Local Release Polish."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import run_phase77_operator_release_polish_validation as phase77


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_docs(tmp_path: Path) -> tuple[Path, Path, Path]:
    guide = tmp_path / "RELEASE_OPERATOR_GUIDE.md"
    quick = tmp_path / "RELEASE_QUICKSTART_LOCAL_ONLY.md"
    limits = tmp_path / "RELEASE_LIMITATIONS_AND_SAFETY.md"

    combined = (
        "local-only mode. "
        "This is not a medical device and does not provide clinical diagnosis. "
        "Not production-autonomous. Human review is required before downstream use. "
        "The manual-review boundary is retained. "
        "Review Package groups pending review cases."
    )
    guide.write_text(combined, encoding="utf-8")
    quick.write_text(combined, encoding="utf-8")
    limits.write_text(combined, encoding="utf-8")
    return guide, quick, limits


def _run(tmp_path: Path, **kwargs) -> dict:
    guide, quick, limits = _make_docs(tmp_path)
    report_dir = kwargs.pop("report_dir", tmp_path / "phase77_reports")
    return phase77.run_polish_validation(
        operator_guide_path=guide,
        quickstart_path=quick,
        limitations_path=limits,
        report_dir=report_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 4. Operator guide exists
# ---------------------------------------------------------------------------


def test_real_operator_guide_exists():
    from scripts.run_phase77_operator_release_polish_validation import OPERATOR_GUIDE
    assert OPERATOR_GUIDE.exists(), f"Missing: {OPERATOR_GUIDE}"


# ---------------------------------------------------------------------------
# 5. Quickstart exists
# ---------------------------------------------------------------------------


def test_real_quickstart_exists():
    from scripts.run_phase77_operator_release_polish_validation import QUICKSTART
    assert QUICKSTART.exists(), f"Missing: {QUICKSTART}"


# ---------------------------------------------------------------------------
# 6. Limitations/safety doc exists
# ---------------------------------------------------------------------------


def test_real_limitations_exists():
    from scripts.run_phase77_operator_release_polish_validation import LIMITATIONS
    assert LIMITATIONS.exists(), f"Missing: {LIMITATIONS}"


# ---------------------------------------------------------------------------
# 7. Docs include required statements
# ---------------------------------------------------------------------------


def test_docs_contain_local_only(tmp_path: Path):
    report = _run(tmp_path)
    assert report["local_only_message_present"] is True


def test_docs_contain_not_medical_device(tmp_path: Path):
    report = _run(tmp_path)
    assert report["not_medical_device_message_present"] is True


def test_docs_contain_not_production_autonomous(tmp_path: Path):
    report = _run(tmp_path)
    assert report["not_production_autonomous_message_present"] is True


def test_docs_contain_manual_review_boundary(tmp_path: Path):
    report = _run(tmp_path)
    assert report["manual_review_boundary_message_present"] is True


def test_docs_explain_review_package(tmp_path: Path):
    report = _run(tmp_path)
    assert report["review_package_explained"] is True


# ---------------------------------------------------------------------------
# 7. Real docs contain all required statements
# ---------------------------------------------------------------------------


def test_real_docs_contain_all_required_statements():
    report = phase77.run_polish_validation()
    assert report["local_only_message_present"], "Missing local-only statement"
    assert report["not_medical_device_message_present"], "Missing not-medical-device statement"
    assert report["not_production_autonomous_message_present"], "Missing not-production-autonomous"
    assert report["manual_review_boundary_message_present"], "Missing manual-review boundary"
    assert report["review_package_explained"], "Missing review package explanation"
    assert report["conclusion"] == "operator_release_polish_ready"


# ---------------------------------------------------------------------------
# 8. App/helper imports still work
# ---------------------------------------------------------------------------


def test_app_imports_still_work():
    from app.review_package_viewer import get_bucket_summary, load_review_package
    from app.main import PHASE52_OPERATOR_TABS
    assert "Run & Review" in PHASE52_OPERATOR_TABS
    assert callable(get_bucket_summary)
    assert callable(load_review_package)


# ---------------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------------


def test_phase77_safety_flags(tmp_path: Path):
    report = _run(tmp_path)
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False
    assert report["safety_gates_should_change_yet"] is False
    assert report["manual_review_boundary_retained"] is True
    assert report["external_api_used"] is False
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


def test_phase77_conclusion(tmp_path: Path):
    report = _run(tmp_path)
    assert report["conclusion"] == "operator_release_polish_ready"
