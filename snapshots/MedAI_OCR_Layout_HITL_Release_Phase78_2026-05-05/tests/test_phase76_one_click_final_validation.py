"""Tests for Phase 76 — One-Click Final Validation / Release Check."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import run_phase76_one_click_final_validation as phase76


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ready_reports(tmp_path: Path) -> dict[str, Path]:
    """Create minimal upstream reports that each return a 'ready' conclusion."""
    paths = {}

    for name, subdir, filename, conclusion in [
        ("p47", "phase47_final_regression_hardening",
         "phase47_final_regression_hardening_report.json", "release_candidate_ready"),
        ("p48", "phase48_operator_release_package",
         "phase48_release_validation_report.json", "release_snapshot_ready"),
        ("p49", "phase49_privacy_ui",
         "phase49_privacy_gate_report.json", "privacy_gate_ready"),
        ("p75", "phase75_review_package_ui_launcher",
         "phase75_review_package_ui_launcher_report.json", "review_package_ui_launcher_ready"),
    ]:
        d = tmp_path / "reports" / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / filename
        p.write_text(json.dumps({"conclusion": conclusion}), encoding="utf-8")
        paths[name] = p

    return paths


def _run(tmp_path: Path, **kwargs) -> dict:
    paths = _make_ready_reports(tmp_path)
    report_dir = kwargs.pop("report_dir", tmp_path / "phase76_reports")
    return phase76.run_validation(
        p47_path=paths["p47"],
        p48_path=paths["p48"],
        p49_path=paths["p49"],
        p75_path=paths["p75"],
        report_dir=report_dir,
        _root=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. Phase76 script runs
# ---------------------------------------------------------------------------


def test_phase76_runs_with_real_reports():
    report = phase76.run_validation()
    assert report["phase"] == 76
    assert report["conclusion"] in ("final_validation_ready", "final_validation_blocked")


# ---------------------------------------------------------------------------
# 2. Phase76 report confirms readiness
# ---------------------------------------------------------------------------


def test_phase76_confirms_phase47_48_49_75_readiness(tmp_path: Path):
    report = _run(tmp_path)
    assert report["phase47_ready"] is True
    assert report["phase48_ready"] is True
    assert report["phase49_ready"] is True
    assert report["phase75_ready"] is True


def test_phase76_full_suite_passed_when_all_ready(tmp_path: Path):
    report = _run(tmp_path)
    assert report["full_suite_passed"] is True


# ---------------------------------------------------------------------------
# 3. Privacy/artifact checks
# ---------------------------------------------------------------------------


def test_phase76_phi_artifact_check_passes(tmp_path: Path):
    report = _run(tmp_path)
    assert report["phi_artifact_check_passed"] is True
    assert report["no_tracked_medical_artifacts"] is True
    assert report["no_private_mapping_tracked"] is True
    assert report["public_report_privacy_check_passed"] is True


def test_phase76_report_has_no_phi(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "phase76_reports"
    combined = "".join(
        (report_dir / f).read_text(encoding="utf-8")
        for f in [phase76.JSON_REPORT.name, phase76.MD_REPORT.name]
        if (report_dir / f).exists()
    )
    assert "SSN 999" not in combined
    assert "Patient Jane Doe" not in combined
    assert '"ocr_text":' not in combined
    assert "full_corpus_input" not in combined


# ---------------------------------------------------------------------------
# Safety flags
# ---------------------------------------------------------------------------


def test_phase76_safety_flags(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False
    assert report["safety_gates_should_change_yet"] is False
    assert report["manual_review_boundary_retained"] is True
    assert report["local_only_default"] is True
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
