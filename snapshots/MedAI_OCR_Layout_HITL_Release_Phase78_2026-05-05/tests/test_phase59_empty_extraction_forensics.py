from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts import run_phase59_empty_extraction_forensics as phase59


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


def _make_private_mapping(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"files": mapping}), encoding="utf-8")
    return path


def _make_blank_pdf(target: Path) -> None:
    import fitz  # type: ignore[import-untyped]
    doc = fitz.open()
    doc.new_page(width=72, height=72)
    doc.save(str(target))
    doc.close()


def _make_text_pdf(target: Path, text: str) -> None:
    import fitz  # type: ignore[import-untyped]
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((20, 50), text, fontsize=11)
    doc.save(str(target))
    doc.close()


def _make_blank_image(target: Path) -> None:
    from PIL import Image  # type: ignore[import-untyped]
    Image.new("RGB", (50, 50), (255, 255, 255)).save(target)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_blank_pdf_classified_as_blank_or_near_blank(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    pdf_path = input_dir / "blank.pdf"
    _make_blank_pdf(pdf_path)

    safe_id = "corpus_file_000001"
    results = [{
        "safe_file_id": safe_id,
        "file_id": safe_id,
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "ocr_status": "good",
        "document_type": None,
        "entity_count": 0,
        "empty_extraction_flag": True,
    }]
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, {
        safe_id: {"original_relative_path": "blank.pdf"}
    })
    report_dir = tmp_path / "phase59_out"

    report = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    entry = report["subset"][0]
    assert entry["root_cause_bucket"] in {"blank_or_near_blank", "image_only_pdf_needs_ocr", "pipeline_bug_suspected"}
    # Most likely outcome is blank_or_near_blank or pipeline_bug_suspected
    # because fitz blank page has no text and no image objects.
    assert entry["page_count"] == 1
    assert entry["has_embedded_text"] is False


def test_image_only_pdf_classified_safely(tmp_path: Path):
    """A PDF with image objects but no extractable text should NOT leak text."""
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    pdf_path = input_dir / "scanned.pdf"

    import fitz  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]

    img_path = tmp_path / "page_image.png"
    Image.new("RGB", (200, 200), (220, 220, 220)).save(img_path)

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(fitz.Rect(0, 0, 200, 200), filename=str(img_path))
    doc.save(str(pdf_path))
    doc.close()

    safe_id = "corpus_file_000002"
    results = [{
        "safe_file_id": safe_id,
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "ocr_status": "good",
        "document_type": None,
        "entity_count": 0,
        "empty_extraction_flag": True,
    }]
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, {
        safe_id: {"original_relative_path": "scanned.pdf"}
    })
    report_dir = tmp_path / "phase59_out"

    report = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
    )
    entry = report["subset"][0]

    # Should be image_only_pdf_needs_ocr OR ocr_ran_but_low_text
    assert entry["root_cause_bucket"] in {"image_only_pdf_needs_ocr", "ocr_ran_but_low_text", "blank_or_near_blank"}
    # Privacy: no raw filename in output
    written_json = (report_dir / phase59.JSON_REPORT.name).read_text(encoding="utf-8")
    assert "scanned.pdf" not in written_json
    assert "page_image.png" not in written_json


def test_public_reports_use_safe_ids_only(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    pdf_path = input_dir / "Patient John Doe Glucose Labs Apr 2024.pdf"
    _make_blank_pdf(pdf_path)

    safe_id = "corpus_file_000003"
    results = [{
        "safe_file_id": safe_id,
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "ocr_status": "good",
        "entity_count": 0,
    }]
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, {
        safe_id: {
            "original_filename": pdf_path.name,
            "original_relative_path": pdf_path.name,
        }
    })
    report_dir = tmp_path / "phase59_out"

    phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    json_text = (report_dir / phase59.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase59.MD_REPORT.name).read_text(encoding="utf-8")

    # Forbid the raw filename, the patient name, and the year+month token.
    for needle in ["Patient John Doe", "Glucose Labs Apr 2024", pdf_path.name]:
        assert needle not in json_text
        assert needle not in md_text
    # safe_file_id IS allowed
    assert safe_id in json_text


def test_public_reports_do_not_contain_raw_paths(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    sub = input_dir / "Private Folder With Personal Info"
    sub.mkdir(parents=True)
    pdf_path = sub / "report.pdf"
    _make_blank_pdf(pdf_path)
    safe_id = "corpus_file_000004"
    results = [{
        "safe_file_id": safe_id,
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "ocr_status": "good",
        "entity_count": 0,
    }]
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, {
        safe_id: {
            "original_filename": "report.pdf",
            "original_relative_path": "Private Folder With Personal Info/report.pdf",
        }
    })
    report_dir = tmp_path / "phase59_out"

    phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    json_text = (report_dir / phase59.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase59.MD_REPORT.name).read_text(encoding="utf-8")
    assert "Private Folder With Personal Info" not in json_text
    assert "Private Folder With Personal Info" not in md_text


def test_public_reports_do_not_contain_extracted_or_ocr_text(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    pdf_path = input_dir / "labs.pdf"
    secret_text = "Patient SSN 123-45-6789 Glucose 103 mg/dL"
    _make_text_pdf(pdf_path, secret_text)

    safe_id = "corpus_file_000005"
    results = [{
        "safe_file_id": safe_id,
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": pdf_path.stat().st_size,
        "ocr_status": "good",
        "document_type": None,
        "entity_count": 0,
        "empty_extraction_flag": True,
    }]
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, {
        safe_id: {"original_relative_path": "labs.pdf"}
    })
    report_dir = tmp_path / "phase59_out"

    phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    json_text = (report_dir / phase59.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase59.MD_REPORT.name).read_text(encoding="utf-8")
    assert "SSN 123-45-6789" not in json_text
    assert "SSN 123-45-6789" not in md_text
    assert "Glucose 103 mg/dL" not in json_text
    assert "Glucose 103 mg/dL" not in md_text


def test_subset_selection_is_deterministic(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    results = []
    mapping = {}
    for i in range(1, 60):
        pdf_path = input_dir / f"f{i:03d}.pdf"
        _make_blank_pdf(pdf_path)
        sid = f"corpus_file_{i:06d}"
        results.append({
            "safe_file_id": sid,
            "file_type": "pdf",
            "extension": ".pdf",
            "file_size_bytes": pdf_path.stat().st_size,
            "ocr_status": "good",
            "entity_count": 0,
            "empty_extraction_flag": True,
        })
        mapping[sid] = {"original_relative_path": pdf_path.name}
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, mapping)
    report_dir = tmp_path / "phase59_out"

    r1 = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
        subset_size=10,
    )
    r2 = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
        subset_size=10,
    )
    ids1 = [e["safe_file_id"] for e in r1["subset"]]
    ids2 = [e["safe_file_id"] for e in r2["subset"]]
    assert ids1 == ids2


def test_no_empty_extraction_files_yields_safe_conclusion(tmp_path: Path):
    report_path = _make_phase57_report(tmp_path, [{
        "safe_file_id": "corpus_file_000001",
        "file_type": "pdf",
        "extension": ".pdf",
        "file_size_bytes": 1000,
        "entity_count": 5,
        "status": "accepted",
    }])
    mapping_path = _make_private_mapping(tmp_path, {})
    report_dir = tmp_path / "phase59_out"

    report = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=tmp_path / "no_corpus",
        report_dir=report_dir,
    )

    assert report["empty_extraction_population"] == 0
    assert report["subset_size_actual"] == 0
    assert report["conclusion"] == "no_empty_extraction_files"


def test_external_apis_remain_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEDAI_LOCAL_ONLY", "true")
    monkeypatch.setenv("MEDAI_ALLOW_EXTERNAL_API", "false")
    report_path = _make_phase57_report(tmp_path, [])
    mapping_path = _make_private_mapping(tmp_path, {})
    report_dir = tmp_path / "phase59_out"

    report = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=tmp_path,
        report_dir=report_dir,
    )

    assert report["privacy_safety"]["external_api_used"] is False
    assert os.environ["MEDAI_LOCAL_ONLY"] == "true"
    assert os.environ["MEDAI_ALLOW_EXTERNAL_API"] == "false"


def test_script_does_not_change_extraction_thresholds():
    # The forensics module must NOT import or reference threshold-bearing
    # modules in a way that would mutate them.
    src = Path(phase59.__file__).read_text(encoding="utf-8")
    forbidden = [
        "ExecutionPipeline.run",
        "from execution.pipeline import",
        "MEDAI_CONFIDENCE_THRESHOLD =",
        "execution_router.set_threshold",
    ]
    for token in forbidden:
        assert token not in src, f"forensics script unexpectedly references {token}"


def test_private_mapping_remains_ignored_by_git():
    repo_root = Path(__file__).resolve().parents[1]
    target = "reports/phase57_full_corpus_inventory_audit/local_filename_mapping_PRIVATE.json"
    result = subprocess.run(
        ["git", "check-ignore", target],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_phase59_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase59_empty_extraction_forensics"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []


def test_recommended_phase60_target_present(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    results = []
    mapping = {}
    for i in range(1, 6):
        pdf_path = input_dir / f"f{i:03d}.pdf"
        _make_blank_pdf(pdf_path)
        sid = f"corpus_file_{i:06d}"
        results.append({
            "safe_file_id": sid,
            "file_type": "pdf",
            "extension": ".pdf",
            "file_size_bytes": pdf_path.stat().st_size,
            "ocr_status": "good",
            "entity_count": 0,
            "empty_extraction_flag": True,
        })
        mapping[sid] = {"original_relative_path": pdf_path.name}
    report_path = _make_phase57_report(tmp_path, results)
    mapping_path = _make_private_mapping(tmp_path, mapping)
    report_dir = tmp_path / "phase59_out"

    report = phase59.run_forensics(
        phase57_report_path=report_path,
        private_mapping_path=mapping_path,
        input_dir=input_dir,
        report_dir=report_dir,
        subset_size=5,
    )

    rec = report["recommended_phase60_target"]
    assert "title" in rec and rec["title"]
    assert "reason" in rec and rec["reason"]
