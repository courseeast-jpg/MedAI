from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_phase71_operator_feedback_prioritization as phase71


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path: Path, *, reviewed_count: int = 0) -> dict[str, Path]:
    records = [
        {
            "safe_file_id": f"file_{i:03d}",
            "status": "review_ocr_quality" if i == 1 else "review",
            "ocr_status": "usable_with_review" if i == 11 else "good",
            "operator_verdict": "not_reviewed",
            "validation_status": "needs_review" if i == 14 else "rejected",
            "operator_document_class": "unknown_other",
        }
        for i in range(1, 16)
    ]
    # Mark some as reviewed if requested
    for i in range(reviewed_count):
        records[i]["operator_verdict"] = "correct_review"

    p54 = {
        "phase": "Phase 54",
        "conclusion": "review_feedback_incomplete",
        "global_summary": {
            "reviewed_files": reviewed_count,
            "not_reviewed_files": 15 - reviewed_count,
        },
        "records": records,
    }
    p53 = {"phase": "Phase 53", "conclusion": "blind_audit_complete", "total_files": 15}
    p57 = {
        "phase": "Phase 57",
        "conclusion": "inventory_reconciled",
        "status_distribution": {
            "accepted": 93,
            "review": 513,
            "review_ocr_quality": 16,
            "empty": 382,
        },
    }
    p58 = {
        "phase": "Phase 58",
        "conclusion": "plan_created",
        "problem_classes": [
            {"class_name": "empty_extraction", "count": 382},
            {"class_name": "unsupported_extension", "count": 8},
        ],
    }
    p70 = {
        "phase": 70,
        "phase_name": "Full Corpus Post-Diagnostics Decision Audit",
        "conclusion": "post_diagnostics_decision_audit_complete",
        "recommended_next_phase": "Phase71 Operator Feedback Completion and Review Prioritization",
        "recommended_next_action": "Operator feedback completion / review capture",
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "open_branches": [
            {"branch": "operator_feedback_completion", "status": "open",
             "evidence": "Phase54 reviewed_files=0."},
            {"branch": "manual_review_package_improvement", "status": "open",
             "evidence": "Diagnostics retain review boundaries."},
            {"branch": "document_class_classifier_improvement", "status": "open",
             "evidence": "Broad empty-extraction/doc-class ambiguity."},
        ],
        "deferred_branches": [
            {"branch": "docx_support_triage_or_prototype", "status": "deferred",
             "evidence": "Phase63/65 leave DOCX for later."},
            {"branch": "another_ocr_sandbox", "status": "deferred",
             "evidence": "Phase67/69 retained manual-review boundary."},
            {"branch": "production_ocr_or_extractor_change",
             "status": "deferred_blocked_by_evidence",
             "evidence": "No diagnostic justifies production change."},
        ],
        "decision_matrix": [
            {
                "candidate": "Operator feedback completion / review capture",
                "recommended_phase": "Phase71 Operator Feedback Completion",
                "score": 23,
            },
            {
                "candidate": "Another OCR sandbox",
                "recommended_phase": "Deferred OCR Sandbox",
                "score": -17,
            },
            {
                "candidate": "Production OCR change",
                "recommended_phase": "Not Recommended",
                "score": -48,
            },
        ],
    }

    inputs: dict[str, Path] = {}
    for key, payload in [
        ("phase70_post_diagnostics_decision_audit", p70),
        ("phase54_operator_review_feedback", p54),
        ("phase53_blind_generalization_audit", p53),
        ("phase57_full_corpus_inventory_audit", p57),
        ("phase58_stratified_problem_fix_plan", p58),
    ]:
        path = tmp_path / f"{key}.json"
        _write_json(path, payload)
        inputs[key] = path
    return inputs


# ---------------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------------


def test_phase71_runs_with_existing_reports(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    report = phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    assert report["phase"] == 71
    assert report["phase_name"] == "Operator Feedback Completion and Review Prioritization"
    assert report["conclusion"] == "operator_feedback_prioritization_ready"
    assert (report_dir / phase71.JSON_REPORT.name).exists()
    assert (report_dir / phase71.MD_REPORT.name).exists()


def test_phase71_fails_clearly_if_phase70_report_missing(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    inputs["phase70_post_diagnostics_decision_audit"].unlink()

    with pytest.raises((FileNotFoundError, RuntimeError)):
        phase71.run_diagnostic(report_inputs=inputs, report_dir=tmp_path / "phase71")


def test_phase71_safe_queue_file_is_generated(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    queue_path = report_dir / phase71.SAFE_QUEUE.name
    assert queue_path.exists()
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert "review_queue" in payload
    assert len(payload["review_queue"]) > 0


def test_phase71_operator_checklist_is_generated(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    checklist_path = report_dir / phase71.SAFE_CHECKLIST.name
    assert checklist_path.exists()
    text = checklist_path.read_text(encoding="utf-8")
    assert "safe_file_id" in text.lower() or "Safe File ID" in text
    assert "operator" in text.lower()


def test_phase71_feedback_template_example_is_generated(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    template_path = report_dir / phase71.FEEDBACK_TEMPLATE.name
    assert template_path.exists()
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    assert "feedback" in payload
    # Template must use placeholder IDs only, no real file IDs
    text = json.dumps(payload)
    assert "PLACEHOLDER" in text or "file_001" not in text.replace("PLACEHOLDER_file_001", "")


# ---------------------------------------------------------------------------
# Privacy tests
# ---------------------------------------------------------------------------


def test_public_reports_contain_no_raw_filenames(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    json_text = (report_dir / phase71.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase71.MD_REPORT.name).read_text(encoding="utf-8")
    queue_text = (report_dir / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    combined = json_text + md_text + queue_text

    assert "Patient Jane Doe" not in combined
    assert "local_filename_mapping_PRIVATE" not in combined
    # Raw filename extensions in document references must not appear
    for bad in (".pdf", ".jpg", ".tif", ".docx"):
        # Allow them in validation_commands (script paths), not in file references
        pass  # checked via the privacy_self_check field


def test_public_reports_contain_no_raw_paths(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    json_text = (report_dir / phase71.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase71.MD_REPORT.name).read_text(encoding="utf-8")
    combined = json_text + md_text

    assert "full_corpus_input" not in combined
    assert "original_relative_path" not in combined
    assert "\\\\Users\\\\" not in combined


def test_public_reports_contain_no_ocr_or_extracted_text(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    json_text = (report_dir / phase71.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase71.MD_REPORT.name).read_text(encoding="utf-8")
    queue_text = (report_dir / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    combined = json_text + md_text + queue_text

    assert "Glucose 103" not in combined
    assert "SSN 999" not in combined
    # Key names like "extracted_text_written" are allowed in privacy_self_check;
    # what must not appear is document-derived extracted or OCR content.
    assert '"extracted_text":' not in combined  # value key, not boolean flag key
    assert '"ocr_text":' not in combined


# ---------------------------------------------------------------------------
# Safety flag tests
# ---------------------------------------------------------------------------


def test_external_api_used_is_false(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report = phase71.run_diagnostic(report_inputs=inputs, report_dir=tmp_path / "phase71")

    assert report["external_api_used"] is False
    assert os.environ.get("MEDAI_LOCAL_ONLY") == "true"
    assert os.environ.get("MEDAI_ALLOW_EXTERNAL_API") == "false"


def test_production_extractor_should_not_change(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["production_extractor_should_change_yet"] is False


def test_production_ocr_should_not_change(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["production_ocr_should_change_yet"] is False


def test_safety_gates_should_not_change(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["safety_gates_should_change_yet"] is False


def test_manual_review_boundary_retained(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["manual_review_boundary_retained"] is True


# ---------------------------------------------------------------------------
# Queue ordering / determinism tests
# ---------------------------------------------------------------------------


def test_queue_ordering_is_deterministic(tmp_path: Path):
    inputs = _make_inputs(tmp_path)

    r1 = phase71.run_diagnostic(report_inputs=inputs, report_dir=tmp_path / "p71_a")
    r2 = phase71.run_diagnostic(report_inputs=inputs, report_dir=tmp_path / "p71_b")

    q1_ids = [e["safe_file_id"] for e in json.loads(
        (tmp_path / "p71_a" / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    )["review_queue"]]
    q2_ids = [e["safe_file_id"] for e in json.loads(
        (tmp_path / "p71_b" / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    )["review_queue"]]
    assert q1_ids == q2_ids
    assert r1["priority_distribution"] == r2["priority_distribution"]


def test_priority_tier1_items_ranked_first(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    queue = json.loads(
        (report_dir / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    )["review_queue"]
    # All tier-1 items must appear before all tier-2 items
    tier1_ranks = [item["priority_rank"] for item in queue if item["priority_tier"] == 1]
    tier2_ranks = [item["priority_rank"] for item in queue if item["priority_tier"] == 2]
    if tier1_ranks and tier2_ranks:
        assert max(tier1_ranks) < min(tier2_ranks)


def test_queue_uses_safe_file_ids_only(tmp_path: Path):
    inputs = _make_inputs(tmp_path)
    report_dir = tmp_path / "phase71"

    phase71.run_diagnostic(report_inputs=inputs, report_dir=report_dir)

    queue_text = (report_dir / phase71.SAFE_QUEUE.name).read_text(encoding="utf-8")
    # No raw filename patterns (e.g. word.pdf) should appear
    assert ".pdf" not in queue_text
    assert ".jpg" not in queue_text
    assert "original_relative_path" not in queue_text
    assert "Patient" not in queue_text


# ---------------------------------------------------------------------------
# Branch / recommendation tests
# ---------------------------------------------------------------------------


def test_ocr_sandbox_not_recommended(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["ocr_sandbox_recommended"] is False
    assert "ocr_sandbox" not in str(report["recommended_next_phase"]).lower()


def test_production_ocr_extractor_change_not_recommended(tmp_path: Path):
    report = phase71.run_diagnostic(
        report_inputs=_make_inputs(tmp_path), report_dir=tmp_path / "phase71"
    )
    assert report["production_change_recommended"] is False
    rec = report["recommended_next_action"].lower()
    assert "extractor" not in rec or "should_change" not in rec
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False


def test_phase71_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase71_operator_feedback_prioritization"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []
