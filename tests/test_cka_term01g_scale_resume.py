from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from clinical_knowledge.terminology.import_limits import build_import_limits
from clinical_knowledge.terminology.import_performance import elapsed_seconds_safe_bucket
from clinical_knowledge.terminology.import_resume import simulate_chunked_import_with_resume
from clinical_knowledge.terminology.import_scale import build_scale_fixtures, parse_scale_fixture


def test_synthetic_scale_fixtures_generate_all_systems() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_scale_fixtures(Path(tmp), rows_per_system=20)
        assert sorted(f.system for f in fixtures) == ["loinc", "rxnorm", "snomed_ct", "umls"]
        assert all(f.row_count == 20 for f in fixtures)


def test_umls_scale_fixture_streams() -> None:
    parsed = _parse_system("umls", rows=30, max_rows=12)
    assert parsed.rows_seen == 12
    assert len(parsed.concepts) == 12


def test_snomed_scale_fixture_streams() -> None:
    parsed = _parse_system("snomed_ct", rows=30, max_rows=12)
    assert parsed.rows_seen == 24
    assert len(parsed.concepts) == 12


def test_rxnorm_scale_fixture_streams() -> None:
    parsed = _parse_system("rxnorm", rows=30, max_rows=12)
    assert parsed.rows_seen == 12
    assert len(parsed.concepts) == 12


def test_loinc_scale_fixture_streams() -> None:
    parsed = _parse_system("loinc", rows=30, max_rows=12)
    assert parsed.rows_seen == 12
    assert len(parsed.concepts) == 12


def test_row_cap_enforced() -> None:
    limits = build_import_limits(max_rows_per_file=10, chunk_size=4)
    parsed = _parse_system("umls", rows=50, max_rows=limits.max_rows_per_file_default)
    metrics = simulate_chunked_import_with_resume(parsed, limits=limits)
    assert metrics.row_cap_enforced is True
    assert metrics.rows_imported == 10


def test_chunking_works() -> None:
    limits = build_import_limits(max_rows_per_file=25, chunk_size=5)
    parsed = _parse_system("loinc", rows=25, max_rows=25)
    metrics = simulate_chunked_import_with_resume(parsed, limits=limits)
    assert metrics.chunking_verified is True
    assert metrics.chunks_processed == 5


def test_interrupted_import_resumes() -> None:
    limits = build_import_limits(max_rows_per_file=25, chunk_size=5)
    parsed = _parse_system("rxnorm", rows=25, max_rows=25)
    metrics = simulate_chunked_import_with_resume(parsed, limits=limits, interrupt_after_chunks=2)
    assert metrics.resume_performed is True
    assert metrics.rows_imported == 25


def test_duplicate_prevention_works() -> None:
    limits = build_import_limits(max_rows_per_file=25, chunk_size=5)
    parsed = _parse_system("umls", rows=25, max_rows=25)
    metrics = simulate_chunked_import_with_resume(parsed, limits=limits, interrupt_after_chunks=1)
    assert metrics.duplicate_prevention_passed is True


def test_memory_safe_streaming_guard_works() -> None:
    limits = build_import_limits(max_rows_per_file=15, chunk_size=5)
    parsed = _parse_system("loinc", rows=100, max_rows=15)
    metrics = simulate_chunked_import_with_resume(parsed, limits=limits)
    assert metrics.streaming_parser_guard_passed is True
    assert metrics.rows_seen == 15


def test_elapsed_bucket_is_coarse() -> None:
    assert elapsed_seconds_safe_bucket(0.01) == "lt_250ms"
    assert elapsed_seconds_safe_bucket(2.0) == "lt_5s"


def test_cli_runs_without_real_data() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/cka_terminology_synthetic_scale_test.py", "--rows", "20", "--max-rows-per-file", "10", "--chunk-size", "5"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["no_real_terminology_import_performed"] is True
    assert payload["no_real_terminology_files_used"] is True
    assert payload["external_api_used"] is False


def test_validation_script_generates_safe_report() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term01g_scale_resume_validation.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term01g_scale_resume_ready" in proc.stdout
    report = repo / "reports" / "cka_term01g_scale_resume" / "cka_term01g_scale_resume_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["raw_phi_logged_in_public_reports"] is False
    assert payload["private_filename_path_leaks"] == 0
    assert payload["secret_leaks"] == 0
    serialized = json.dumps(payload)
    assert "C:\\" not in serialized
    assert "source_response_raw" not in serialized


def test_no_real_import_data_or_staged_files() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    staged = proc.stdout.replace("\\", "/")
    assert "terminology_data/" not in staged
    assert "data/terminology/" not in staged


def _parse_system(system: str, *, rows: int, max_rows: int):
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_scale_fixtures(Path(tmp), rows_per_system=rows)
        fixture = next(f for f in fixtures if f.system == system)
        return parse_scale_fixture(fixture, max_rows=max_rows)
