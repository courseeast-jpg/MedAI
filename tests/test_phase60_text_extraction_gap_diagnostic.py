from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from scripts import run_phase60_text_extraction_gap_diagnostic as phase60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase57_report(tmp_path: Path, results: list[dict]) -> Path:
    payload = {
        "results": results,
        "filesystem_reconciliation": {
            "total_filesystem_files": len(results),
            "total_supported_processed": len(results),
        },
    }
    path = tmp_path / "phase57_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_phase59_report(tmp_path: Path, subset: list[dict]) -> Path:
    payload = {"subset": subset}
    path = tmp_path / "phase59_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_private_mapping(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"files": mapping}), encoding="utf-8")
    return path


def _make_text_pdf(target: Path, text: str) -> None:
    import fitz  # type: ignore[import-untyped]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Insert text in chunks to fit on page
    y = 50
    for line in text.splitlines() or [text]:
        page.insert_text((36, y), line[:120], fontsize=10)
        y += 14
        if y > 760:
            page = doc.new_page(width=612, height=792)
            y = 50
    doc.save(str(target))
    doc.close()


def _build_pipeline_inputs(tmp_path: Path, files: list[tuple[str, str]]) -> dict:
    """For each (filename, text), make a PDF and a Phase 57/59 entry."""
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    results = []
    mapping = {}
    phase59_subset = []
    for i, (fname, text) in enumerate(files, start=1):
        pdf_path = input_dir / fname
        _make_text_pdf(pdf_path, text)
        sid = f"corpus_file_{i:06d}"
        results.append({
            "safe_file_id": sid,
            "file_id": sid,
            "file_type": "pdf",
            "extension": ".pdf",
            "file_size_bytes": pdf_path.stat().st_size,
            "ocr_status": "good",
            "entity_count": 0,
            "empty_extraction_flag": True,
        })
        mapping[sid] = {"original_relative_path": pdf_path.name}
        phase59_subset.append({
            "safe_file_id": sid,
            "root_cause_bucket": "pdf_text_extraction_gap",
            "file_type": "pdf",
        })
    p57 = _make_phase57_report(tmp_path, results)
    p59 = _make_phase59_report(tmp_path, phase59_subset)
    pmap = _make_private_mapping(tmp_path, mapping)
    return {
        "input_dir": input_dir,
        "phase57": p57,
        "phase59": p59,
        "private_mapping": pmap,
    }


# ---------------------------------------------------------------------------
# Vocabulary measurement tests
# ---------------------------------------------------------------------------


def test_synthetic_lab_text_classifies_as_lab_table_vocabulary_gap(tmp_path: Path):
    text = "\n".join([
        "Glucose 103 mg/dL  65-99",
        "Creatinine 1.1 mg/dL  0.6-1.2",
        "Hemoglobin 14.2 g/dL  12-16",
        "Sodium 140 mmol/L  135-145",
        "Potassium 4.0 mmol/L  3.5-5.1",
    ])
    inputs = _build_pipeline_inputs(tmp_path, [("labs.pdf", text)])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    entry = report["subset"][0]
    assert entry["likely_document_class_guess"] == "lab_report"
    assert entry["likely_gap_type"] in {"lab_table_vocabulary_gap", "medical_vocabulary_gap"}
    assert entry["has_common_lab_units"] is True
    assert entry["has_reference_range_patterns"] is True


def test_synthetic_imaging_text_classifies_as_imaging_gap(tmp_path: Path):
    text = "\n".join([
        "CT chest with contrast.",
        "Findings: no acute abnormality.",
        "Impression: stable.",
        "Radiology report follows.",
        "X-ray comparison from prior study.",
    ])
    inputs = _build_pipeline_inputs(tmp_path, [("imaging.pdf", text)])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    entry = report["subset"][0]
    assert entry["likely_document_class_guess"] == "radiology_or_imaging"
    assert entry["likely_gap_type"] == "imaging_report_vocabulary_gap"
    assert entry["has_radiology_terms"] is True


def test_synthetic_admin_text_classifies_as_admin_not_target(tmp_path: Path):
    text = "\n".join([
        "Insurance Statement",
        "Account: 12345",
        "Policy holder: subscriber name",
        "Balance due, deductible, copay applied",
        "Billing inquiry to payer",
        "Claim ID: 998877",
    ])
    inputs = _build_pipeline_inputs(tmp_path, [("admin.pdf", text)])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    entry = report["subset"][0]
    assert entry["likely_document_class_guess"] == "admin_or_billing"
    assert entry["likely_gap_type"] == "admin_document_not_target"
    assert entry["has_admin_terms"] is True


def test_numeric_heavy_table_without_labels_detected(tmp_path: Path):
    text = "\n".join([
        "   1.2   3.4   5.6",
        "   2.3   4.5   6.7",
        "   3.4   5.6   7.8",
        "   4.5   6.7   8.9",
        "   5.6   7.8   9.0",
        "   1.1   2.2   3.3",
    ])
    inputs = _build_pipeline_inputs(tmp_path, [("numbers.pdf", text)])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    entry = report["subset"][0]
    # Without medical vocab, this becomes numeric_table_without_labels or
    # tokenization_or_layout_issue depending on text length thresholds.
    assert entry["likely_gap_type"] in {
        "numeric_table_without_labels",
        "tokenization_or_layout_issue",
        "unknown",
    }
    # Public output must NOT contain the literal numeric tokens beyond what
    # safe id paths legitimately use.
    json_text = (report_dir / phase60.JSON_REPORT.name).read_text(encoding="utf-8")
    # The exact line "1.2   3.4   5.6" must not appear
    assert "1.2   3.4" not in json_text


# ---------------------------------------------------------------------------
# Privacy tests
# ---------------------------------------------------------------------------


def test_public_reports_use_safe_ids_only(tmp_path: Path):
    text = "Glucose 103 mg/dL"
    inputs = _build_pipeline_inputs(tmp_path, [("Patient John Doe Glucose Mar 2024.pdf", text)])
    report_dir = tmp_path / "phase60_out"

    phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase60.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase60.MD_REPORT.name).read_text(encoding="utf-8")

    for needle in ["Patient John Doe", "Glucose Mar 2024"]:
        assert needle not in json_text
        assert needle not in md_text


def test_public_reports_do_not_contain_raw_paths(tmp_path: Path):
    text = "Glucose 103 mg/dL"
    inputs = _build_pipeline_inputs(tmp_path, [("labs.pdf", text)])
    # Move the file into a sub-folder so the relative path includes folder name
    sub = inputs["input_dir"] / "Private Folder With Personal Info"
    sub.mkdir()
    moved = sub / "labs.pdf"
    (inputs["input_dir"] / "labs.pdf").rename(moved)
    # Update mapping to reflect new path
    mapping_payload = json.loads(inputs["private_mapping"].read_text(encoding="utf-8"))
    sid = next(iter(mapping_payload["files"]))
    mapping_payload["files"][sid]["original_relative_path"] = "Private Folder With Personal Info/labs.pdf"
    inputs["private_mapping"].write_text(json.dumps(mapping_payload), encoding="utf-8")

    report_dir = tmp_path / "phase60_out"
    phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase60.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase60.MD_REPORT.name).read_text(encoding="utf-8")
    assert "Private Folder With Personal Info" not in json_text
    assert "Private Folder With Personal Info" not in md_text


def test_public_reports_do_not_contain_extracted_or_ocr_text(tmp_path: Path):
    secret_text = "Patient SSN 123-45-6789 Glucose 103 mg/dL Cholesterol 200 mg/dL"
    inputs = _build_pipeline_inputs(tmp_path, [("labs.pdf", secret_text)])
    report_dir = tmp_path / "phase60_out"

    phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase60.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase60.MD_REPORT.name).read_text(encoding="utf-8")
    assert "SSN 123-45-6789" not in json_text
    assert "SSN 123-45-6789" not in md_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in json_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in md_text


# ---------------------------------------------------------------------------
# Stability / safety tests
# ---------------------------------------------------------------------------


def test_subset_selection_is_deterministic(tmp_path: Path):
    files = [(f"f{i:03d}.pdf", f"Glucose 103 mg/dL line {i}") for i in range(1, 25)]
    inputs = _build_pipeline_inputs(tmp_path, files)
    report_dir = tmp_path / "phase60_out"

    r1 = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
        subset_size=10,
    )
    r2 = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
        subset_size=10,
    )
    ids1 = [e["safe_file_id"] for e in r1["subset"]]
    ids2 = [e["safe_file_id"] for e in r2["subset"]]
    assert ids1 == ids2


def test_external_apis_remain_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEDAI_LOCAL_ONLY", "true")
    monkeypatch.setenv("MEDAI_ALLOW_EXTERNAL_API", "false")
    inputs = _build_pipeline_inputs(tmp_path, [("labs.pdf", "Glucose 103 mg/dL")])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )
    assert report["privacy_safety"]["external_api_used"] is False
    assert os.environ["MEDAI_LOCAL_ONLY"] == "true"
    assert os.environ["MEDAI_ALLOW_EXTERNAL_API"] == "false"


def test_script_does_not_modify_extraction_logic_or_thresholds():
    src = Path(phase60.__file__).read_text(encoding="utf-8")
    forbidden = [
        "ExecutionPipeline.run",
        "from execution.pipeline import",
        "MEDAI_CONFIDENCE_THRESHOLD =",
        "execution_router.set_threshold",
    ]
    for token in forbidden:
        assert token not in src


def test_missing_phase59_report_handled_safely(tmp_path: Path):
    p57 = _make_phase57_report(tmp_path, [])
    pmap = _make_private_mapping(tmp_path, {})
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=p57,
        phase59_report_path=tmp_path / "missing_phase59.json",
        private_mapping_path=pmap,
        input_dir=tmp_path / "no_corpus",
        report_dir=report_dir,
    )
    assert report["subset_size_actual"] == 0
    assert report["conclusion"] == "no_text_gap_files"


def test_empty_target_bucket_yields_safe_conclusion(tmp_path: Path):
    p57 = _make_phase57_report(tmp_path, [])
    p59 = _make_phase59_report(tmp_path, [])
    pmap = _make_private_mapping(tmp_path, {})
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=p57,
        phase59_report_path=p59,
        private_mapping_path=pmap,
        input_dir=tmp_path,
        report_dir=report_dir,
    )
    assert report["conclusion"] == "no_text_gap_files"


def test_phase60_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase60_text_extraction_gap_diagnostic"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []


def test_recommendation_default_is_no_production_change(tmp_path: Path):
    inputs = _build_pipeline_inputs(tmp_path, [("labs.pdf", "Glucose 103 mg/dL  65-99")])
    report_dir = tmp_path / "phase60_out"

    report = phase60.run_diagnostic(
        phase57_report_path=inputs["phase57"],
        phase59_report_path=inputs["phase59"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    rec = report["recommended_phase61_target"]
    assert rec["production_extractor_should_change_yet"] is False
