from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import run_phase74_manual_review_package_auto_improvement as phase74


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase57(tmp_path: Path) -> Path:
    p = tmp_path / "phase57" / "phase57_full_corpus_inventory_audit_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 57,
        "total_discovered": 20,
        "total_supported": 18,
        "accepted": 5,
        "review": 10,
        "review_ocr_quality": 2,
        "empty": 4,
        "errors": 1,
        "unsupported_count": 2,
        "problem_clusters": {
            "empty_extraction": [f"corpus_file_{i:06d}" for i in range(4)],
            "pdf_ocr_low_quality": [f"corpus_file_{i:06d}" for i in range(4, 6)],
            "image_ocr_low_quality": ["corpus_file_000006"],
            "unknown_other": ["corpus_file_000007"],
            "rules_based_low_confidence": [f"corpus_file_{i:06d}" for i in range(8, 14)],
            "possible_lab_table_failure": [f"corpus_file_{i:06d}" for i in range(14, 18)],
            "unsupported_format": ["corpus_file_000018", "corpus_file_000019"],
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _make_phase71(tmp_path: Path) -> Path:
    p = tmp_path / "phase71" / "phase71_operator_feedback_prioritization_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 71,
        "review_queue_count": 5,
        "priority_distribution": {"tier_1": 2, "tier_2": 3},
        "problem_class_distribution": {
            "ocr_quality_gate_trigger": 1,
            "borderline_ocr_quality": 1,
            "unknown_document_class": 3,
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _make_phase72(tmp_path: Path) -> Path:
    p = tmp_path / "phase72" / "phase72_operator_feedback_collection_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": 72,
        "pending_count": 5,
        "reviewed_count": 0,
        "unresolved_high_priority_count": 2,
        "pending_safe_ids": [f"file_{i:03d}" for i in range(1, 6)],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _run(tmp_path: Path, **kwargs) -> dict:
    p57 = kwargs.pop("phase57_path", _make_phase57(tmp_path))
    p71 = kwargs.pop("phase71_path", _make_phase71(tmp_path))
    p72 = kwargs.pop("phase72_path", _make_phase72(tmp_path))
    report_dir = kwargs.pop("report_dir", tmp_path / "reports")
    return phase74.run_improvement(
        phase57_path=p57,
        phase71_path=p71,
        phase72_path=p72,
        report_dir=report_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Script runs from current repository state
# ---------------------------------------------------------------------------


def test_phase74_runs_with_real_reports():
    report = phase74.run_improvement()
    assert report["phase"] == 74
    assert report["conclusion"] == "manual_review_package_auto_improvement_ready"


# ---------------------------------------------------------------------------
# 2–3. Does not require private feedback or operator labels
# ---------------------------------------------------------------------------


def test_does_not_require_private_feedback(tmp_path: Path):
    """Running with no phase72 private file present still succeeds."""
    p57 = _make_phase57(tmp_path)
    report = phase74.run_improvement(
        phase57_path=p57,
        report_dir=tmp_path / "reports",
        # phase72 not provided → optional
    )
    assert report["operator_feedback_required"] is False


def test_does_not_require_operator_labels(tmp_path: Path):
    report = _run(tmp_path)
    assert report["labels_fabricated"] is False
    assert report["operator_feedback_required"] is False


# ---------------------------------------------------------------------------
# 4. labels_fabricated is false
# ---------------------------------------------------------------------------


def test_labels_fabricated_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["labels_fabricated"] is False


# ---------------------------------------------------------------------------
# 5. operator_feedback_required is false
# ---------------------------------------------------------------------------


def test_operator_feedback_required_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["operator_feedback_required"] is False


# ---------------------------------------------------------------------------
# 6–8. Production / safety flags
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
# 9. manual_review_boundary_retained is true
# ---------------------------------------------------------------------------


def test_manual_review_boundary_retained(tmp_path: Path):
    report = _run(tmp_path)
    assert report["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# 10. external_api_used is false
# ---------------------------------------------------------------------------


def test_external_api_used_is_false(tmp_path: Path):
    report = _run(tmp_path)
    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


# ---------------------------------------------------------------------------
# 11–15. Privacy checks on public reports
# ---------------------------------------------------------------------------


def _read_all_outputs(tmp_path: Path) -> str:
    report_dir = tmp_path / "reports"
    combined = ""
    for fname in [
        phase74.JSON_REPORT.name,
        phase74.MD_REPORT.name,
        phase74.PACKAGE_JSON.name,
        phase74.PACKAGE_MD.name,
    ]:
        f = report_dir / fname
        if f.exists():
            combined += f.read_text(encoding="utf-8")
    return combined


def test_public_reports_no_raw_filenames(tmp_path: Path):
    _run(tmp_path)
    combined = _read_all_outputs(tmp_path)
    assert "Patient Jane Doe" not in combined
    assert "local_filename_mapping_PRIVATE" not in combined


def test_public_reports_no_raw_paths(tmp_path: Path):
    _run(tmp_path)
    combined = _read_all_outputs(tmp_path)
    assert "full_corpus_input" not in combined
    assert "original_relative_path" not in combined


def test_public_reports_no_ocr_text(tmp_path: Path):
    _run(tmp_path)
    combined = _read_all_outputs(tmp_path)
    assert '"ocr_text":' not in combined
    assert "Glucose 103" not in combined


def test_public_reports_no_extracted_medical_text(tmp_path: Path):
    _run(tmp_path)
    combined = _read_all_outputs(tmp_path)
    assert '"extracted_text":' not in combined


def test_public_reports_no_phi(tmp_path: Path):
    report = _run(tmp_path)
    combined = _read_all_outputs(tmp_path)
    assert "SSN 999" not in combined
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


# ---------------------------------------------------------------------------
# 16. manual_review_package_SAFE.json is generated
# ---------------------------------------------------------------------------


def test_package_json_is_generated(tmp_path: Path):
    _run(tmp_path)
    pkg = tmp_path / "reports" / phase74.PACKAGE_JSON.name
    assert pkg.exists()
    payload = json.loads(pkg.read_text(encoding="utf-8"))
    assert "buckets" in payload
    assert len(payload["buckets"]) > 0


# ---------------------------------------------------------------------------
# 17. manual_review_package_SAFE.md is generated
# ---------------------------------------------------------------------------


def test_package_md_is_generated(tmp_path: Path):
    _run(tmp_path)
    pkg = tmp_path / "reports" / phase74.PACKAGE_MD.name
    assert pkg.exists()
    text = pkg.read_text(encoding="utf-8")
    assert "Manual Review Package" in text
    assert "bucket_id" in text


# ---------------------------------------------------------------------------
# 18. Buckets are deterministic
# ---------------------------------------------------------------------------


def test_buckets_are_deterministic(tmp_path: Path):
    p57 = _make_phase57(tmp_path)
    p71 = _make_phase71(tmp_path)
    p72 = _make_phase72(tmp_path)

    r1 = phase74.run_improvement(
        phase57_path=p57, phase71_path=p71, phase72_path=p72,
        report_dir=tmp_path / "r1",
    )
    r2 = phase74.run_improvement(
        phase57_path=p57, phase71_path=p71, phase72_path=p72,
        report_dir=tmp_path / "r2",
    )
    ids1 = [b["bucket_id"] for b in r1["safe_review_buckets"]]
    ids2 = [b["bucket_id"] for b in r2["safe_review_buckets"]]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# 19. Bucket priorities are deterministic
# ---------------------------------------------------------------------------


def test_bucket_priorities_are_deterministic(tmp_path: Path):
    report = _run(tmp_path)
    priorities = [b["priority"] for b in report["safe_review_buckets"]]
    assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# 20. No production OCR/extractor change is recommended
# ---------------------------------------------------------------------------


def test_no_production_change_recommended(tmp_path: Path):
    report = _run(tmp_path)
    for b in report["safe_review_buckets"]:
        assert b["whether_production_change_is_allowed"] is False
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False


# ---------------------------------------------------------------------------
# 21. No OCR sandbox is recommended
# ---------------------------------------------------------------------------


def test_no_ocr_sandbox_recommended(tmp_path: Path):
    report = _run(tmp_path)
    rec = report["recommended_next_phase"].lower()
    assert "ocr_sandbox" not in rec
    assert "ocr sandbox" not in rec
    for b in report["safe_review_buckets"]:
        assert "ocr_sandbox" not in b["safest_next_action"].lower()


# ---------------------------------------------------------------------------
# 22. Recommended next phase is deterministic
# ---------------------------------------------------------------------------


def test_recommended_next_phase_is_deterministic(tmp_path: Path):
    p57 = _make_phase57(tmp_path)
    r1 = phase74.run_improvement(phase57_path=p57, report_dir=tmp_path / "r1")
    r2 = phase74.run_improvement(phase57_path=p57, report_dir=tmp_path / "r2")
    assert r1["recommended_next_phase"] == r2["recommended_next_phase"]
    assert r1["recommended_next_phase"] == "Phase75 Review Package UI/Launcher Integration"
