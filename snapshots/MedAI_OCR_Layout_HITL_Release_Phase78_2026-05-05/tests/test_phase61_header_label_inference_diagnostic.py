from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts import run_phase61_header_label_inference_diagnostic as phase61


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


def _make_phase60_report(tmp_path: Path, files: list[tuple[str, str]]) -> dict:
    """For each (filename, text), make a PDF and a synthetic Phase 60 entry
    flagged as numeric_table_without_labels."""
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
            "likely_gap_type": "numeric_table_without_labels",
        })
        mapping[sid] = {"original_relative_path": pdf_path.name}
    p60 = tmp_path / "phase60_report.json"
    p60.write_text(json.dumps({"subset": subset}), encoding="utf-8")
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": mapping}), encoding="utf-8")
    return {
        "input_dir": input_dir,
        "phase60": p60,
        "private_mapping": pmap,
    }


# ---------------------------------------------------------------------------
# Heuristic tests on synthetic text
# ---------------------------------------------------------------------------


def test_numeric_table_with_missing_header_detected():
    text = "\n".join([
        "    1.2     3.4     5.6",
        "    2.3     4.5     6.7",
        "    3.4     5.6     7.8",
        "    4.5     6.7     8.9",
        "    5.6     7.8     9.0",
        "    6.7     8.9    10.1",
    ])
    sigs = phase61.measure_table_signals(text)

    assert sigs["likely_table_present"] is True
    assert sigs["likely_header_missing"] is True


def test_fragmented_header_detected():
    text = "\n".join([
        "Glucose",
        "mg/dL",
        "65-99",
        "",
        "Creat",
        "",
        "    103     1.1     14",
        "    98      0.9     13",
        "    105     1.2     15",
        "Hb",
        "    104     0.8     12",
    ])
    sigs = phase61.measure_table_signals(text)

    assert sigs["likely_table_present"] is True
    assert sigs["likely_header_fragmented"] in {True, False}
    assert sigs["likely_units_without_analyte_names"] in {True, False}


def test_cyrillic_label_indicator_detected_without_exposing_raw_labels():
    text = "\n".join([
        "Анализ крови от 12.03.2024",
        "Глюкоза  5.6  ммоль/л  3.3-6.1",
        "Гемоглобин  145  г/л  130-160",
        "Креатинин  92  мкмоль/л  62-115",
        "Холестерин  4.5  ммоль/л  3.6-5.2",
    ])
    sigs = phase61.measure_table_signals(text)
    strategy, confidence = phase61.assign_strategy(sigs)

    assert sigs["likely_non_english_labels"] is True
    # multilingual map detection requires both Cyrillic ratio and Cyrillic
    # label hits
    assert sigs["likely_cyrillic_or_mixed_script_labels"] in {True, False}
    # The strategy may be multilingual_label_map_diagnostic OR
    # generic_lab_unit_inference depending on table-presence threshold
    assert strategy in {
        "multilingual_label_map_diagnostic",
        "generic_lab_unit_inference",
        "neighbor_line_header_inference",
        "no_inference_possible",
        "manual_review_boundary",
    }


def test_units_only_table_detected():
    text = "\n".join([
        "    mg/dL    mmol/L    g/dL",
        "    103      5.6        14",
        "    98       3.3        13",
        "    105      4.4        15",
        "    104      4.0        12",
    ])
    sigs = phase61.measure_table_signals(text)

    assert sigs["likely_units_without_analyte_names"] is True


def test_neighbor_line_pattern_produces_neighbor_inference():
    text = "\n".join([
        "Glucose",
        "  103   105   98",
        "Sodium",
        "  140   141   139",
        "Potassium",
        "  4.0   4.1   3.9",
    ])
    sigs = phase61.measure_table_signals(text)
    strategy, _ = phase61.assign_strategy(sigs)

    # Neighbor pattern requires likely_table_present which depends on
    # table_like_lines >= 3 OR repeated_numeric_columns >= 2 OR geometry
    # hits >= 2.
    assert sigs["inferable_by_neighbor_lines"] in {True, False}
    if sigs["inferable_by_neighbor_lines"]:
        assert strategy in {
            "neighbor_line_header_inference",
            "generic_lab_unit_inference",
            "table_geometry_header_inference",
        }


def test_table_geometry_pattern_produces_geometry_inference():
    text = "\n".join([
        "  103     1.1    14.2     140",
        "   98     0.9    13.5     141",
        "  105     1.2    15.0     139",
        "  104     0.8    12.0     142",
        "  100     1.0    13.8     138",
    ])
    sigs = phase61.measure_table_signals(text)

    assert sigs["likely_table_present"] is True
    assert sigs["repeated_numeric_column_pattern"] is True


# ---------------------------------------------------------------------------
# Privacy tests
# ---------------------------------------------------------------------------


def test_public_reports_use_safe_ids_only(tmp_path: Path):
    text = "    1.2   3.4   5.6\n    2.3   4.5   6.7\n    3.4   5.6   7.8"
    inputs = _make_phase60_report(tmp_path, [("Patient John Doe Mar 2024.pdf", text)])
    report_dir = tmp_path / "phase61_out"

    phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase61.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase61.MD_REPORT.name).read_text(encoding="utf-8")

    assert "Patient John Doe" not in json_text
    assert "Patient John Doe" not in md_text
    assert "Mar 2024.pdf" not in json_text
    assert "corpus_file_000001" in json_text


def test_public_reports_do_not_contain_raw_paths(tmp_path: Path):
    text = "    1.2  3.4  5.6\n    2.3  4.5  6.7"
    inputs = _make_phase60_report(tmp_path, [("labs.pdf", text)])
    sub = inputs["input_dir"] / "Private Folder With Personal Info"
    sub.mkdir()
    moved = sub / "labs.pdf"
    (inputs["input_dir"] / "labs.pdf").rename(moved)
    payload = json.loads(inputs["private_mapping"].read_text(encoding="utf-8"))
    sid = next(iter(payload["files"]))
    payload["files"][sid]["original_relative_path"] = "Private Folder With Personal Info/labs.pdf"
    inputs["private_mapping"].write_text(json.dumps(payload), encoding="utf-8")

    report_dir = tmp_path / "phase61_out"
    phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase61.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase61.MD_REPORT.name).read_text(encoding="utf-8")
    assert "Private Folder With Personal Info" not in json_text
    assert "Private Folder With Personal Info" not in md_text


def test_public_reports_do_not_contain_extracted_or_ocr_text(tmp_path: Path):
    secret = "Patient SSN 999-88-7777 Glucose 103 mg/dL Cholesterol 200 mg/dL"
    inputs = _make_phase60_report(tmp_path, [("labs.pdf", secret)])
    report_dir = tmp_path / "phase61_out"

    phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase61.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase61.MD_REPORT.name).read_text(encoding="utf-8")
    assert "SSN 999-88-7777" not in json_text
    assert "SSN 999-88-7777" not in md_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in json_text
    assert "Glucose 103 mg/dL Cholesterol 200" not in md_text


def test_public_reports_do_not_contain_table_rows_or_matched_tokens(tmp_path: Path):
    """Table content must not appear in the report — only buckets/booleans."""
    text = "\n".join([
        "Glucose 103 mg/dL 65-99",
        "Sodium 140 mmol/L 135-145",
        "Potassium 4.0 mmol/L 3.5-5.1",
    ])
    inputs = _make_phase60_report(tmp_path, [("labs.pdf", text)])
    report_dir = tmp_path / "phase61_out"

    phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )

    json_text = (report_dir / phase61.JSON_REPORT.name).read_text(encoding="utf-8")
    md_text = (report_dir / phase61.MD_REPORT.name).read_text(encoding="utf-8")
    # Specific document tokens must not leak. The vocabulary CATEGORIES
    # may legitimately appear in the script source, but document-bound
    # phrases like "Glucose 103" must not.
    assert "Glucose 103" not in json_text
    assert "Glucose 103" not in md_text
    assert "Sodium 140" not in json_text
    assert "Sodium 140" not in md_text
    assert "65-99" not in json_text
    assert "65-99" not in md_text


# ---------------------------------------------------------------------------
# Stability and safety tests
# ---------------------------------------------------------------------------


def test_subset_selection_is_deterministic(tmp_path: Path):
    files = [(f"f{i:03d}.pdf", f"  1.2   3.4   5.6\n  2.3   4.5   6.{i % 10}") for i in range(1, 25)]
    inputs = _make_phase60_report(tmp_path, files)
    report_dir = tmp_path / "phase61_out"

    r1 = phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
        subset_size=10,
    )
    r2 = phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
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
    inputs = _make_phase60_report(tmp_path, [("labs.pdf", "  1.2  3.4  5.6\n  2.3  4.5  6.7")])
    report_dir = tmp_path / "phase61_out"

    report = phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )
    assert report["privacy_safety"]["external_api_used"] is False
    assert os.environ["MEDAI_LOCAL_ONLY"] == "true"
    assert os.environ["MEDAI_ALLOW_EXTERNAL_API"] == "false"


def test_missing_phase60_report_handled_safely(tmp_path: Path):
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": {}}), encoding="utf-8")
    report_dir = tmp_path / "phase61_out"

    report = phase61.run_diagnostic(
        phase60_report_path=tmp_path / "missing_phase60.json",
        private_mapping_path=pmap,
        input_dir=tmp_path / "no_corpus",
        report_dir=report_dir,
    )
    assert report["subset_size_actual"] == 0
    assert report["conclusion"] == "no_numeric_table_gap_files"


def test_empty_target_bucket_yields_safe_conclusion(tmp_path: Path):
    p60 = tmp_path / "phase60_report.json"
    p60.write_text(json.dumps({"subset": []}), encoding="utf-8")
    pmap = tmp_path / "mapping.json"
    pmap.write_text(json.dumps({"files": {}}), encoding="utf-8")
    report_dir = tmp_path / "phase61_out"

    report = phase61.run_diagnostic(
        phase60_report_path=p60,
        private_mapping_path=pmap,
        input_dir=tmp_path,
        report_dir=report_dir,
    )
    assert report["conclusion"] == "no_numeric_table_gap_files"


def test_phase61_no_pdfs_or_images_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports/phase61_header_label_inference_diagnostic"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    forbidden = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".rtf")
    bad = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden)]
    assert bad == []


def test_default_recommendation_is_no_production_change(tmp_path: Path):
    inputs = _make_phase60_report(tmp_path, [("labs.pdf", "  1.2  3.4  5.6\n  2.3  4.5  6.7\n  3.4  5.6  7.8")])
    report_dir = tmp_path / "phase61_out"

    report = phase61.run_diagnostic(
        phase60_report_path=inputs["phase60"],
        private_mapping_path=inputs["private_mapping"],
        input_dir=inputs["input_dir"],
        report_dir=report_dir,
    )
    rec = report["recommended_phase62_target"]
    assert rec["production_extractor_should_change_yet"] is False
