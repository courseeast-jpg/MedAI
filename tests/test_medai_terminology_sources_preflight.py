from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_medai_terminology_sources_preflight.py"


def write_manifest(repo_root: Path) -> None:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    source = REPO_ROOT / "config" / "terminology_sources.example.json"
    (config_dir / "terminology_sources.example.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def make_valid_synthetic_sources(repo_root: Path) -> None:
    write_manifest(repo_root)
    terminology = repo_root / "terminology_data"
    (terminology / "Loinc_2.82").mkdir(parents=True)
    (terminology / "Loinc_2.82" / "Loinc.csv").write_text("DO_NOT_PRINT_LOINC_ROW\n", encoding="utf-8")
    (terminology / "loinc").mkdir(parents=True)
    (terminology / "loinc" / "Loinc.csv").write_text("DUPLICATE_DO_NOT_PRINT\n", encoding="utf-8")

    for folder in ["RxNorm_full_05042026", "RxNorm_full_prescribe_05042026"]:
        root = terminology / folder
        root.mkdir(parents=True)
        for name in ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF"]:
            (root / name).write_text("DO_NOT_PRINT_RXNORM_ROW\n", encoding="utf-8")

    for folder in [
        "SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260301T120000Z",
        "SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z",
    ]:
        root = terminology / folder / "Snapshot" / "Terminology"
        root.mkdir(parents=True)
        (root / "sct2_Concept_Synthetic.txt").write_text("DO_NOT_PRINT_SNOMED_CONCEPT\n", encoding="utf-8")
        (root / "sct2_Description_Synthetic.txt").write_text("DO_NOT_PRINT_SNOMED_DESCRIPTION\n", encoding="utf-8")
        (root / "sct2_Relationship_Synthetic.txt").write_text("DO_NOT_PRINT_SNOMED_RELATIONSHIP\n", encoding="utf-8")
        (terminology / folder / "release_package_information.json").write_text("{}", encoding="utf-8")

    umls = terminology / "umls 2026AA-full"
    umls.mkdir(parents=True)
    (umls / "umls-synthetic.nlm").write_text("DO_NOT_PRINT_UMLS_NLM\n", encoding="utf-8")
    (terminology / "LICENSE_ACK_PRIVATE.json").write_text("PRIVATE_ACK_CONTENT_DO_NOT_PRINT\n", encoding="utf-8")


def run_preflight(repo_root: Path, report_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--repo-root", str(repo_root)]
    if report_dir is not None:
        command.extend(["--report-dir", str(report_dir)])
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def load_report(repo_root: Path, report_dir: Path | None = None) -> dict:
    base = report_dir or (repo_root / "reports" / "terminology_sources_preflight")
    return json.loads((base / "terminology_sources_preflight_report.json").read_text(encoding="utf-8"))


def combined_report_text(repo_root: Path, report_dir: Path | None = None) -> str:
    base = report_dir or (repo_root / "reports" / "terminology_sources_preflight")
    return (
        (base / "terminology_sources_preflight_report.json").read_text(encoding="utf-8")
        + "\n"
        + (base / "terminology_sources_preflight_report.md").read_text(encoding="utf-8")
    )


def test_loads_example_manifest(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)

    result = run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    report = load_report(tmp_path)
    assert report["manifest_source"] == "config/terminology_sources.example.json"
    assert report["import_performed"] is False


def test_validates_canonical_primary_and_auxiliary_sources(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    result = run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    report = load_report(tmp_path)
    by_source = {row["source"]: row for row in report["canonical_sources"]}

    assert by_source["loinc"]["ready"] is True
    assert by_source["loinc"]["canonical_path"] == "terminology_data/Loinc_2.82"
    assert by_source["rxnorm"]["ready"] is True
    assert by_source["rxnorm"]["role"] == "primary"
    assert by_source["rxnorm_prescribable"]["ready"] is True
    assert by_source["rxnorm_prescribable"]["role"] == "auxiliary"


def test_validates_snomed_roles_and_umls_future_gate(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    result = run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    report = load_report(tmp_path)
    by_source = {row["source"]: row for row in report["canonical_sources"]}

    assert by_source["snomed_ct_us"]["ready"] is True
    assert by_source["snomed_ct_us"]["role"] == "primary"
    assert by_source["snomed_ct_international"]["ready"] is True
    assert by_source["snomed_ct_international"]["role"] == "secondary"
    assert by_source["umls"]["ready"] is True
    assert by_source["umls"]["role"] == "separate_future_import"
    assert report["readiness"]["umls_present_but_future_gated"] is True


def test_license_ack_presence_only_without_content_leak(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    result = run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    report = load_report(tmp_path)
    by_source = {row["source"]: row for row in report["canonical_sources"]}
    text = combined_report_text(tmp_path)

    assert by_source["license_ack_private"]["ready"] is True
    assert by_source["license_ack_private"]["contents_read"] is False
    assert "PRIVATE_ACK_CONTENT_DO_NOT_PRINT" not in text


def test_reports_use_relative_paths_and_do_not_print_rows(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    result = run_preflight(tmp_path)

    assert result.returncode == 0, result.stderr
    text = combined_report_text(tmp_path)
    assert str(tmp_path) not in text
    assert "DO_NOT_PRINT_LOINC_ROW" not in text
    assert "DO_NOT_PRINT_RXNORM_ROW" not in text
    assert "DO_NOT_PRINT_SNOMED_CONCEPT" not in text
    assert "DO_NOT_PRINT_UMLS_NLM" not in text


def test_missing_primary_source_fails_closed(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    (tmp_path / "terminology_data" / "RxNorm_full_05042026" / "RXNCONSO.RRF").unlink()

    result = run_preflight(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 1
    assert report["conclusion"] == "blocked_missing_required_primary_sources"
    assert "rxnorm" in report["missing_required_primary_sources"]


def test_duplicate_loinc_candidates_warn_without_import(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)

    result = run_preflight(tmp_path)
    report = load_report(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["duplicate_candidate_warnings"]
    assert report["duplicate_candidate_warnings"][0]["source"] == "loinc"
    assert report["import_performed"] is False
    assert report["runtime_db_or_index_created"] is False


def test_local_manifest_overrides_example_when_present(tmp_path: Path) -> None:
    make_valid_synthetic_sources(tmp_path)
    local = tmp_path / "config" / "terminology_sources.local.json"
    example = json.loads((tmp_path / "config" / "terminology_sources.example.json").read_text(encoding="utf-8"))
    example["loinc"]["canonical_path"] = "terminology_data/loinc"
    local.write_text(json.dumps(example), encoding="utf-8")

    result = run_preflight(tmp_path)
    report = load_report(tmp_path)
    by_source = {row["source"]: row for row in report["canonical_sources"]}

    assert result.returncode == 0, result.stderr
    assert report["manifest_local_used"] is True
    assert by_source["loinc"]["canonical_path"] == "terminology_data/loinc"


@pytest.mark.parametrize("report_subdir", ["nested/reports"])
def test_custom_report_dir_is_supported_for_temp_validation(tmp_path: Path, report_subdir: str) -> None:
    make_valid_synthetic_sources(tmp_path)
    report_dir = tmp_path / report_subdir

    result = run_preflight(tmp_path, report_dir)

    assert result.returncode == 0, result.stderr
    assert (report_dir / "terminology_sources_preflight_report.json").exists()
    assert (report_dir / "terminology_sources_preflight_report.md").exists()
