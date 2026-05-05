from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_phase70_post_diagnostics_decision_audit as phase70


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase70_script_runs_with_existing_reports(tmp_path: Path):
    inputs = write_inputs(tmp_path)
    report_dir = tmp_path / "phase70"

    report = phase70.run_audit(report_inputs=inputs, report_dir=report_dir)

    assert report["phase"] == 70
    assert report["phase_name"] == "Full Corpus Post-Diagnostics Decision Audit"
    assert report["conclusion"] == "post_diagnostics_decision_audit_complete"
    assert report["recommended_next_action"] == "Operator feedback completion / review capture"
    assert (report_dir / phase70.JSON_REPORT.name).exists()
    assert (report_dir / phase70.MD_REPORT.name).exists()


def test_phase70_fails_clearly_if_phase69_report_missing(tmp_path: Path):
    inputs = write_inputs(tmp_path)
    inputs["phase69_image_ocr_preprocessing_comparison"].unlink()

    with pytest.raises(FileNotFoundError, match="Required Phase69 report is missing"):
        phase70.run_audit(report_inputs=inputs, report_dir=tmp_path / "phase70")


def test_phase70_report_contains_no_raw_filenames_paths_or_text(tmp_path: Path):
    inputs = write_inputs(tmp_path)
    report_dir = tmp_path / "phase70"

    report = phase70.run_audit(report_inputs=inputs, report_dir=report_dir)
    public_json = (report_dir / phase70.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase70.MD_REPORT.name).read_text(encoding="utf-8")
    public = public_json + "\n" + public_md

    assert "Patient Jane Doe" not in public
    assert "full_corpus_input" not in public
    assert "filename_hash" not in public
    assert "content_hash" not in public
    assert "OCR TEXT" not in public
    assert "EXTRACTED TEXT" not in public
    assert ".pdf" not in public
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


def test_phase70_safety_flags_are_false_or_retained(tmp_path: Path):
    report = phase70.run_audit(report_inputs=write_inputs(tmp_path), report_dir=tmp_path / "phase70")

    assert report["external_api_used"] is False
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False
    assert report["safety_gates_should_change_yet"] is False
    assert report["manual_review_boundary_retained"] is True
    assert phase70.app_config.MEDAI_LOCAL_ONLY is True
    assert phase70.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase70_ocr_sandbox_not_recommended_when_phase69_says_no(tmp_path: Path):
    report = phase70.run_audit(report_inputs=write_inputs(tmp_path), report_dir=tmp_path / "phase70")

    assert report["recommended_next_action"] != "Another OCR sandbox"
    assert report["recommended_next_action"] != "Production OCR change"
    sandbox = next(item for item in report["decision_matrix"] if item["candidate"] == "Another OCR sandbox")
    production = next(item for item in report["decision_matrix"] if item["candidate"] == "Production OCR change")
    selected = report["decision_matrix"][0]
    assert sandbox["score"] < selected["score"]
    assert production["score"] < sandbox["score"]


def test_phase70_closed_open_deferred_branches_are_present(tmp_path: Path):
    report = phase70.run_audit(report_inputs=write_inputs(tmp_path), report_dir=tmp_path / "phase70")

    closed = {item["branch"] for item in report["closed_branches"]}
    opened = {item["branch"] for item in report["open_branches"]}
    deferred = {item["branch"] for item in report["deferred_branches"]}
    assert "pdf_geometry_header_inference" in closed
    assert "pdf_ocr_preprocessing" in closed
    assert "image_ocr_preprocessing" in closed
    assert "rtf_narrow_format_support" in closed
    assert "operator_feedback_completion" in opened
    assert "docx_support_triage_or_prototype" in deferred
    assert "production_ocr_or_extractor_change" in deferred


def test_phase70_selected_next_action_is_deterministic(tmp_path: Path):
    inputs = write_inputs(tmp_path)

    report1 = phase70.run_audit(report_inputs=inputs, report_dir=tmp_path / "phase70_a")
    report2 = phase70.run_audit(report_inputs=inputs, report_dir=tmp_path / "phase70_b")

    assert report1["recommended_next_action"] == report2["recommended_next_action"]
    assert report1["decision_matrix"] == report2["decision_matrix"]


def test_phase70_markdown_generated_from_json_safe_content(tmp_path: Path):
    inputs = write_inputs(tmp_path)
    report_dir = tmp_path / "phase70"
    report = phase70.run_audit(report_inputs=inputs, report_dir=report_dir)

    markdown = (report_dir / phase70.MD_REPORT.name).read_text(encoding="utf-8")
    assert report["recommended_next_action"] in markdown
    assert "Operator feedback completion / review capture" in markdown
    assert "Patient Jane Doe" not in markdown
    assert "local_filename_mapping_PRIVATE" not in markdown


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    payloads = {
        "phase54_operator_review_feedback": {
            "phase": "Phase 54",
            "conclusion": "review_feedback_incomplete",
            "global_summary": {"reviewed_files": 0, "not_reviewed_files": 15},
        },
        "phase58_stratified_problem_fix_plan": {"phase": "Phase 58", "conclusion": "plan_created"},
        "phase59_empty_extraction_forensics": {"phase": "Phase 59", "conclusion": "forensic_subset_audited"},
        "phase60_text_extraction_gap_diagnostic": {
            "phase": "Phase 60",
            "conclusion": "text_extraction_gap_diagnosed",
            "production_extractor_should_change_yet": False,
        },
        "phase61_header_label_inference_diagnostic": {
            "phase": "Phase 61",
            "conclusion": "header_label_inference_diagnosed",
            "production_extractor_should_change_yet": False,
        },
        "phase62_table_geometry_header_inference_prototype": {
            "phase": "Phase 62",
            "conclusion": "table_geometry_prototype_assessed",
            "production_extractor_should_change_yet": False,
        },
        "phase63_unsupported_extension_triage": {
            "phase": "Phase 63",
            "conclusion": "triage_complete_manual_or_later_only",
            "production_extractor_should_change_yet": False,
        },
        "phase64_rtf_local_text_parser": {"phase": "Phase 64", "conclusion": "rtf_local_parser_prototype_ready"},
        "phase65_full_corpus_delta_after_rtf": {
            "phase": "Phase 65",
            "conclusion": "rtf_delta_measured_no_safety_regression",
            "production_extractor_should_change_yet": False,
        },
        "phase66_pdf_ocr_low_quality_diagnostic": {
            "phase": "Phase 66",
            "conclusion": "pdf_ocr_low_quality_diagnostic_complete",
            "manual_review_boundary_preserved": True,
            "production_extractor_should_change_yet": False,
            "production_ocr_should_change_yet": False,
        },
        "phase67_ocr_preprocessing_comparison": {
            "phase": "Phase 67",
            "conclusion": "manual_review_boundary_retained",
            "recommended_next_action": "keep_manual_review_boundary",
            "phase68_controlled_ocr_fallback_sandbox_recommended": False,
            "production_extractor_should_change_yet": False,
            "production_ocr_should_change_yet": False,
        },
        "phase68_image_ocr_low_quality_diagnostic": {
            "phase": "Phase 68",
            "conclusion": "image_preprocessing_prototype_justified",
            "manual_review_boundary_preserved": True,
            "production_extractor_should_change_yet": False,
            "production_ocr_should_change_yet": False,
        },
        "phase69_image_ocr_preprocessing_comparison": {
            "phase": "Phase 69",
            "conclusion": "manual_review_boundary_retained",
            "candidate_count": 5,
            "meaningful_improvement_file_count": 2,
            "recommended_next_action": "keep_manual_review_boundary",
            "phase70_controlled_image_ocr_fallback_sandbox_recommended": False,
            "manual_review_boundary_preserved": True,
            "production_extractor_should_change_yet": False,
            "production_ocr_should_change_yet": False,
        },
    }
    inputs: dict[str, Path] = {}
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.json"
        write_json(path, payload)
        inputs[key] = path
    return inputs
