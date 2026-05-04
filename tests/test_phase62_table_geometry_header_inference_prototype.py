from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts import run_phase62_table_geometry_header_inference_prototype as phase62


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_pdf(target: Path, text: str) -> None:
    import fitz  # type: ignore[import-untyped]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 40
    for line in text.splitlines() or [text]:
        page.insert_text((36, y), line[:200], fontsize=10)
        y += 13
        if y > 760:
            page = doc.new_page(width=612, height=792)
            y = 40
    doc.save(str(target))
    doc.close()


def _make_phase61_report(tmp_path: Path, files: list[tuple[str, str]]) -> dict:
    """For each (filename, text), make a PDF and a synthetic Phase 61 entry
    flagged as table_geometry_header_inference."""
    input_dir = tmp_path / "corpus"
    input_dir.mkdir()
    subset = []
    mapping = {}
    for i, (fname, text) in enumerate(files, start=1):
        pdf_path = input_dir / fname
        _make_text_pdf(pdf_path, text)
        sid = f"corpus_file_{i:06d}"
        subset.append({
            "safe_file_id": sid,
            "file_type": "pdf",
            "extension": ".pdf",
            "recommended_strategy": "table_geometry_header_inference",
            "confidence_band": "high",
        })
        mapping[sid] = {"original_relative_path": pdf_path.name}
    p61 = tmp_path / "phase61_report.json"
    p61.write_text(json.dumps({"subset": subset}), encoding="utf-8")
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": mapping}), encoding="utf-8")
    return {
        "input_dir": input_dir,
        "phase61": p61,
        "private_mapping": pmap,
    }


_ALIGNED_NUMERIC_TABLE = "\n".join([
    "  103     1.1    14.2    140",
    "   98     0.9    13.5    141",
    "  105     1.2    15.0    139",
    "  104     0.8    12.0    142",
    "  100     1.0    13.8    138",
    "   99     1.1    14.0    140",
])


# ---------------------------------------------------------------------------
# Geometry measurement tests
# ---------------------------------------------------------------------------


def test_aligned_numeric_table_detected():
    sigs = phase62.measure_geometry_signals(_ALIGNED_NUMERIC_TABLE)

    assert sigs["column_alignment_detected"] is True
    assert sigs["deep_table_block_present"] is True
    assert sigs["geometry_signal_strength"] in {"high", "medium"}


def test_recoverable_candidate_on_aligned_table():
    sigs = phase62.measure_geometry_signals(_ALIGNED_NUMERIC_TABLE)

    assert sigs["recoverable_table_candidate"] is True
    assert sigs["recovery_confidence_band"] in {"high", "medium"}
    assert sigs["safe_next_action"] == "prototype_candidate"


def test_shallow_block_not_recoverable():
    text = "\n".join([
        "  1.2   3.4   5.6",
        "  2.3   4.5   6.7",
    ])
    sigs = phase62.measure_geometry_signals(text)

    assert sigs["deep_table_block_present"] is False
    assert sigs["recoverable_table_candidate"] is False


def test_multi_block_structure_detected():
    text = "\n".join([
        "  1.2   3.4   5.6",
        "  2.3   4.5   6.7",
        "  3.4   5.6   7.8",
        "  4.5   6.7   8.9",
        "",
        "some label text here",
        "",
        "  5.6   7.8   9.0",
        "  6.7   8.9  10.1",
        "  7.8  10.1  11.2",
        "  8.9  11.2  12.3",
    ])
    sigs = phase62.measure_geometry_signals(text)

    assert sigs["multi_block_structure"] is True
    assert sigs["table_block_count_bucket"] != "zero"


def test_empty_text_returns_insufficient_geometry():
    sigs = phase62.measure_geometry_signals("")

    assert sigs["geometry_signal_strength"] == "none"
    assert sigs["recoverable_table_candidate"] is False
    assert sigs["safe_next_action"] == "insufficient_geometry"


def test_pure_text_no_geometry_signal():
    text = "\n".join([
        "This is a patient discharge summary.",
        "The patient presented with chest pain.",
        "Diagnosis: stable angina.",
        "Follow up in two weeks.",
    ])
    sigs = phase62.measure_geometry_signals(text)

    assert sigs["column_alignment_detected"] is False
    assert sigs["geometry_signal_strength"] in {"none", "low"}


def test_repeated_numeric_column_pattern_counted():
    text = "\n".join([
        "  103     1.1    14.2    140",
        "   98     0.9    13.5    141",
        "  105     1.2    15.0    139",
        "  104     0.8    12.0    142",
    ])
    sigs = phase62.measure_geometry_signals(text)

    assert sigs["repeated_numeric_column_line_count_bucket"] != "zero"


# ---------------------------------------------------------------------------
# Privacy tests
# ---------------------------------------------------------------------------


def test_public_reports_use_safe_ids_only(tmp_path: Path):
    inputs = _make_phase61_report(
        tmp_path,
        [("Patient Jane Doe 2024.pdf", _ALIGNED_NUMERIC_TABLE)],
    )
    report_dir = tmp_path / "phase62_out"

    phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase62.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase62.MD_REPORT.name).read_text(encoding="utf-8")

    assert "Patient Jane Doe" not in json_text
    assert "Patient Jane Doe" not in md_text
    assert "corpus_file_000001" in json_text


def test_public_reports_do_not_contain_raw_paths(tmp_path: Path):
    inputs = _make_phase61_report(tmp_path, [("labs.pdf", _ALIGNED_NUMERIC_TABLE)])
    sub = inputs["input_dir"] / "Private Folder With Personal Info"
    sub.mkdir()
    moved = sub / "labs.pdf"
    (inputs["input_dir"] / "labs.pdf").rename(moved)
    payload = json.loads(inputs["private_mapping"].read_text(encoding="utf-8"))
    sid = next(iter(payload["files"]))
    payload["files"][sid]["original_relative_path"] = "Private Folder With Personal Info/labs.pdf"
    inputs["private_mapping"].write_text(json.dumps(payload), encoding="utf-8")

    report_dir = tmp_path / "phase62_out"
    phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase62.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase62.MD_REPORT.name).read_text(encoding="utf-8")
    assert "Private Folder With Personal Info" not in json_text
    assert "Private Folder With Personal Info" not in md_text


def test_public_reports_do_not_contain_extracted_or_ocr_text(tmp_path: Path):
    secret = "Patient SSN 999-88-7777 Glucose 103 mg/dL Cholesterol 200 mg/dL"
    inputs = _make_phase61_report(tmp_path, [("labs.pdf", secret)])
    report_dir = tmp_path / "phase62_out"

    phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase62.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase62.MD_REPORT.name).read_text(encoding="utf-8")
    assert "SSN 999-88-7777" not in json_text
    assert "SSN 999-88-7777" not in md_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in json_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in md_text


def test_public_reports_do_not_contain_table_rows_or_inferred_labels(tmp_path: Path):
    text = "\n".join([
        "  103     1.1    14.2    140",
        "   98     0.9    13.5    141",
        "  105     1.2    15.0    139",
    ])
    inputs = _make_phase61_report(tmp_path, [("labs.pdf", text)])
    report_dir = tmp_path / "phase62_out"

    phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase62.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase62.MD_REPORT.name).read_text(encoding="utf-8")
    assert "103     1.1" not in json_text
    assert "103     1.1" not in md_text
    assert "98     0.9" not in json_text
    assert "98     0.9" not in md_text


# ---------------------------------------------------------------------------
# Stability and safety tests
# ---------------------------------------------------------------------------


def test_subset_selection_is_deterministic(tmp_path: Path):
    files = [
        (f"f{i:03d}.pdf", _ALIGNED_NUMERIC_TABLE + f"\n  {i}.0   {i+1}.0   {i+2}.0   {i+3}.0")
        for i in range(1, 25)
    ]
    inputs = _make_phase61_report(tmp_path, files)
    report_dir = tmp_path / "phase62_out"

    r1 = phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
        subset_size=10,
    )
    r2 = phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
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
    inputs = _make_phase61_report(tmp_path, [("labs.pdf", _ALIGNED_NUMERIC_TABLE)])
    report_dir = tmp_path / "phase62_out"

    report = phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )
    assert report["privacy_safety"]["external_api_used"] is False
    assert os.environ["MEDAI_LOCAL_ONLY"] == "true"
    assert os.environ["MEDAI_ALLOW_EXTERNAL_API"] == "false"


def test_missing_phase61_report_handled_safely(tmp_path: Path):
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": {}}), encoding="utf-8")
    report_dir = tmp_path / "phase62_out"

    report = phase62.run_diagnostic(
        phase61_report_path=tmp_path / "missing_phase61.json",
        private_mapping_path=pmap,
        input_dir=tmp_path / "no_corpus",
        report_dir=report_dir,
    )
    assert report["subset_size_actual"] == 0
    assert report["conclusion"] == "no_geometry_inference_files"


def test_empty_target_bucket_yields_safe_conclusion(tmp_path: Path):
    p61 = tmp_path / "phase61_report.json"
    p61.write_text(json.dumps({"subset": []}), encoding="utf-8")
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": {}}), encoding="utf-8")
    report_dir = tmp_path / "phase62_out"

    report = phase62.run_diagnostic(
        phase61_report_path=p61,
        private_mapping_path=pmap,
        input_dir=tmp_path,
        report_dir=report_dir,
    )
    assert report["conclusion"] == "no_geometry_inference_files"


def test_phase62_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase62_table_geometry_header_inference_prototype"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []


def test_default_recommendation_is_no_production_change(tmp_path: Path):
    inputs = _make_phase61_report(tmp_path, [("labs.pdf", _ALIGNED_NUMERIC_TABLE)])
    report_dir = tmp_path / "phase62_out"

    report = phase62.run_diagnostic(
        phase61_report_path=inputs["phase61"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )
    rec = report["recommended_phase63_target"]
    assert rec["production_extractor_should_change_yet"] is False
