from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import run_phase75_review_package_ui_launcher as phase75
from app.review_package_viewer import get_bucket_summary, load_review_package


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_package(tmp_path: Path, bucket_count: int = 3) -> Path:
    buckets = []
    for i in range(1, bucket_count + 1):
        buckets.append({
            "bucket_id": f"bucket_{i:02d}",
            "bucket_name": f"Test Bucket {i}",
            "priority": i,
            "aggregate_count": i * 10,
            "high_priority_item_count": 1 if i == 1 else 0,
            "suspected_problem_classes": ["test_class"],
            "source_phases": [f"phase{50 + i}"],
            "why_it_is_in_review": f"Test reason {i}.",
            "what_the_system_knows": f"Known info {i}.",
            "what_the_system_does_not_know": f"Unknown info {i}.",
            "safest_next_action": f"Safe action {i}.",
            "whether_operator_action_is_required": i == 1,
            "whether_production_change_is_allowed": False,
            "pending_safe_ids_sample": [f"file_{j:03d}" for j in range(1, 4)],
        })
    p = tmp_path / "phase74" / "manual_review_package_SAFE.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "generated_at": "2026-05-04T00:00:00+00:00",
        "description": "Test package",
        "buckets": buckets,
    }), encoding="utf-8")
    return p


def _run(tmp_path: Path, **kwargs) -> dict:
    pkg = kwargs.pop("package_path", _make_package(tmp_path))
    report_dir = kwargs.pop("report_dir", tmp_path / "reports")
    return phase75.run_launcher(
        package_path=pkg,
        report_dir=report_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Script runs from current repo state
# ---------------------------------------------------------------------------


def test_phase75_runs_with_real_reports():
    report = phase75.run_launcher()
    assert report["phase"] == 75
    assert report["conclusion"] == "review_package_ui_launcher_ready"


# ---------------------------------------------------------------------------
# 2. Fails clearly if Phase74 package missing
# ---------------------------------------------------------------------------


def test_fails_if_phase74_package_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Phase74"):
        phase75.run_launcher(
            package_path=tmp_path / "no_such_package.json",
            report_dir=tmp_path / "reports",
        )


# ---------------------------------------------------------------------------
# 3. Review package loader parses Phase74 package
# ---------------------------------------------------------------------------


def test_loader_parses_package(tmp_path: Path):
    pkg_path = _make_package(tmp_path, bucket_count=4)
    package = load_review_package(pkg_path)
    assert "buckets" in package
    assert len(package["buckets"]) == 4


def test_loader_raises_if_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_review_package(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# 4. UI helper returns deterministic bucket summary
# ---------------------------------------------------------------------------


def test_bucket_summary_is_deterministic(tmp_path: Path):
    pkg_path = _make_package(tmp_path, bucket_count=3)
    package = load_review_package(pkg_path)
    s1 = get_bucket_summary(package)
    s2 = get_bucket_summary(package)
    assert [b["bucket_id"] for b in s1] == [b["bucket_id"] for b in s2]
    assert [b["priority"] for b in s1] == sorted(b["priority"] for b in s1)


# ---------------------------------------------------------------------------
# 5–9. Privacy checks on UI helper output
# ---------------------------------------------------------------------------


def _all_text_from_summary(summary: list[dict]) -> str:
    return json.dumps(summary)


def test_ui_helper_no_raw_filenames(tmp_path: Path):
    pkg_path = _make_package(tmp_path)
    package = load_review_package(pkg_path)
    text = _all_text_from_summary(get_bucket_summary(package))
    assert "Patient Jane Doe" not in text
    assert "local_filename_mapping_PRIVATE" not in text


def test_ui_helper_no_raw_paths(tmp_path: Path):
    pkg_path = _make_package(tmp_path)
    package = load_review_package(pkg_path)
    text = _all_text_from_summary(get_bucket_summary(package))
    assert "full_corpus_input" not in text
    assert "original_relative_path" not in text


def test_ui_helper_no_ocr_text(tmp_path: Path):
    pkg_path = _make_package(tmp_path)
    package = load_review_package(pkg_path)
    text = _all_text_from_summary(get_bucket_summary(package))
    assert '"ocr_text"' not in text
    assert "Glucose 103" not in text


def test_ui_helper_no_extracted_medical_text(tmp_path: Path):
    pkg_path = _make_package(tmp_path)
    package = load_review_package(pkg_path)
    text = _all_text_from_summary(get_bucket_summary(package))
    assert '"extracted_text"' not in text


def test_ui_helper_no_phi(tmp_path: Path):
    pkg_path = _make_package(tmp_path)
    package = load_review_package(pkg_path)
    text = _all_text_from_summary(get_bucket_summary(package))
    assert "SSN 999" not in text
    assert "DOB 1980" not in text


# ---------------------------------------------------------------------------
# 10–16. Report flag checks
# ---------------------------------------------------------------------------


def test_operator_feedback_required_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["operator_feedback_required"] is False


def test_labels_fabricated_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["labels_fabricated"] is False


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


def test_external_api_used_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


# ---------------------------------------------------------------------------
# 17. Launcher/report generated safely (no PHI in outputs)
# ---------------------------------------------------------------------------


def test_report_generated_safely(tmp_path: Path):
    _run(tmp_path)
    report_dir = tmp_path / "reports"
    combined = ""
    for fname in [phase75.JSON_REPORT.name, phase75.MD_REPORT.name, phase75.README.name]:
        f = report_dir / fname
        if f.exists():
            combined += f.read_text(encoding="utf-8")
    assert "Patient Jane Doe" not in combined
    assert "full_corpus_input" not in combined
    assert '"ocr_text"' not in combined
    assert "SSN 999" not in combined
    assert report_dir.exists()


# ---------------------------------------------------------------------------
# 18. No private mapping or feedback file referenced in public report
# ---------------------------------------------------------------------------


def test_no_private_mapping_in_report(tmp_path: Path):
    _run(tmp_path)
    json_text = (tmp_path / "reports" / phase75.JSON_REPORT.name).read_text(encoding="utf-8")
    assert "local_filename_mapping_PRIVATE" not in json_text
    assert "operator_feedback_PRIVATE" not in json_text


# ---------------------------------------------------------------------------
# 19. Recommended next phase is deterministic
# ---------------------------------------------------------------------------


def test_recommended_next_phase_is_deterministic(tmp_path: Path):
    pkg = _make_package(tmp_path)
    r1 = phase75.run_launcher(package_path=pkg, report_dir=tmp_path / "r1")
    r2 = phase75.run_launcher(package_path=pkg, report_dir=tmp_path / "r2")
    assert r1["recommended_next_phase"] == r2["recommended_next_phase"]
    assert r1["recommended_next_phase"] == "Phase76 One-Click Final Validation / Release Check"


# ---------------------------------------------------------------------------
# 20. Existing app imports still work
# ---------------------------------------------------------------------------


def test_app_main_imports_cleanly():
    """Verify main.py module-level imports succeed (no Streamlit execution needed)."""
    import importlib
    import sys

    # If already imported, just verify it
    mod_name = "app.main"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        mod = importlib.import_module(mod_name)

    assert hasattr(mod, "PHASE52_OPERATOR_TABS")
    assert "Review Package" in mod.PHASE52_OPERATOR_TABS


def test_review_package_viewer_importable():
    from app.review_package_viewer import (
        get_bucket_summary,
        load_phase74_report,
        load_review_package,
    )
    assert callable(load_review_package)
    assert callable(load_phase74_report)
    assert callable(get_bucket_summary)
