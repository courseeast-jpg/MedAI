from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_medai_terminology_inventory.py"


def make_synthetic_terminology_root(repo_root: Path) -> Path:
    terminology = repo_root / "terminology_data"
    (terminology / "Loinc_2.82").mkdir(parents=True)
    (terminology / "Loinc_2.82" / "Loinc.csv").write_text("LOINC_ROW_SHOULD_NOT_PRINT\n", encoding="utf-8")
    (terminology / "loinc").mkdir(parents=True)
    (terminology / "loinc" / "Loinc.csv").write_text("DUPLICATE_LOINC_ROW_SHOULD_NOT_PRINT\n", encoding="utf-8")

    for folder in ["RxNorm_full_05042026", "RxNorm_full_prescribe_05042026"]:
        root = terminology / folder
        root.mkdir(parents=True)
        for name in ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF"]:
            (root / name).write_text("RXNORM_ROW_SHOULD_NOT_PRINT\n", encoding="utf-8")

    (terminology / "umls 2026AA-full").mkdir(parents=True)
    (terminology / "umls 2026AA-full" / "synthetic.nlm").write_text("UMLS_LICENSE_TEXT_SHOULD_NOT_PRINT\n", encoding="utf-8")

    for folder in [
        "SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z",
        "SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z",
    ]:
        root = terminology / folder / "Snapshot" / "Terminology"
        root.mkdir(parents=True)
        (root / "sct2_Concept_Synthetic.txt").write_text("SNOMED_ROW_SHOULD_NOT_PRINT\n", encoding="utf-8")
        (root / "sct2_Description_Synthetic.txt").write_text("SNOMED_DESCRIPTION_SHOULD_NOT_PRINT\n", encoding="utf-8")
        (root / "sct2_Relationship_Synthetic.txt").write_text("SNOMED_RELATIONSHIP_SHOULD_NOT_PRINT\n", encoding="utf-8")
        (terminology / folder / "release_package_information.json").write_text("{}", encoding="utf-8")

    (terminology / "LICENSE_ACK_PRIVATE.json").write_text("PRIVATE_LICENSE_ACK_SHOULD_NOT_PRINT\n", encoding="utf-8")
    return terminology


def run_inventory(repo_root: Path, terminology_root: str = "terminology_data") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo_root),
            "--terminology-root",
            terminology_root,
            "--report-dir",
            str(repo_root / "reports" / "terminology_data_inventory"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def report_text(repo_root: Path) -> str:
    report_dir = repo_root / "reports" / "terminology_data_inventory"
    return (
        (report_dir / "terminology_data_inventory_report.json").read_text(encoding="utf-8")
        + "\n"
        + (report_dir / "terminology_data_inventory_report.md").read_text(encoding="utf-8")
    )


def load_report(repo_root: Path) -> dict:
    return json.loads(
        (repo_root / "reports" / "terminology_data_inventory" / "terminology_data_inventory_report.json").read_text(
            encoding="utf-8"
        )
    )


def test_inventory_is_report_only_and_no_import(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["safety"]["terminology_import_performed"] is False
    assert report["safety"]["terminology_files_modified"] is False
    assert not (tmp_path / "data" / "terminology").exists()


def test_license_ack_contents_are_not_read_or_reported(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    text = report_text(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["safety"]["license_ack_contents_read"] is False
    assert "PRIVATE_LICENSE_ACK_SHOULD_NOT_PRINT" not in text


def test_licensed_rows_are_not_printed(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    text = result.stdout + "\n" + report_text(tmp_path)

    assert result.returncode == 0, result.stderr
    for sentinel in [
        "LOINC_ROW_SHOULD_NOT_PRINT",
        "RXNORM_ROW_SHOULD_NOT_PRINT",
        "UMLS_LICENSE_TEXT_SHOULD_NOT_PRINT",
        "SNOMED_ROW_SHOULD_NOT_PRINT",
    ]:
        assert sentinel not in text


def test_reports_use_relative_paths_only(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    text = report_text(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert str(tmp_path) not in text
    assert report["terminology_root"] == "terminology_data"
    assert report["safety"]["absolute_paths_in_report"] is False


def test_duplicate_candidate_package_warning(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert any("loinc has 2 candidate folders" in warning for warning in report["duplicate_parallel_folder_warnings"])


def test_missing_terminology_root_is_safe(tmp_path: Path) -> None:
    result = run_inventory(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["summary"]["root_exists"] is False
    assert report["summary"]["file_count"] == 0
    assert report["summary"]["folder_count"] == 0
    assert report["safety"]["terminology_import_performed"] is False


def test_source_terminology_files_are_not_staged_by_tool(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)

    assert result.returncode == 0, result.stderr
    # The inventory script has no git staging code and should not create any source copies.
    assert not list((tmp_path / "reports").rglob("*.RRF"))
    assert not list((tmp_path / "reports").rglob("*.rrf"))
    assert not list((tmp_path / "reports").rglob("*.nlm"))


def test_report_privacy_clean_for_common_private_markers(tmp_path: Path) -> None:
    make_synthetic_terminology_root(tmp_path)

    result = run_inventory(tmp_path)
    text = report_text(tmp_path).lower()

    assert result.returncode == 0, result.stderr
    assert "private_license_ack_should_not_print".lower() not in text
    assert "source_response_raw" not in text
    assert "replacement_map" not in text
    assert "api_key" not in text
    assert "secret" not in text
