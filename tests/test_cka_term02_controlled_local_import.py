from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.import_limits import build_import_limits
from clinical_knowledge.terminology.license_gate import license_acknowledged_for
from clinical_knowledge.terminology.models import TerminologySystem
from clinical_knowledge.terminology.term02_controlled_import import (
    TERM02_DB_RELATIVE,
    Term02ImportBlocked,
    _iter_loinc_concepts,
    _iter_rxnorm_concepts,
    run_controlled_local_import,
)
from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_blocks_when_ack_missing(tmp_path: Path) -> None:
    terminology_root = _synthetic_root(tmp_path, ack=False)
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=terminology_root)
    assert result.allowed is False
    assert "license_ack_private_missing" in result.reason_codes


def test_preflight_passes_with_rxnorm_and_loinc_ack(tmp_path: Path) -> None:
    terminology_root = _synthetic_root(tmp_path, ack=True)
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=terminology_root)
    assert result.allowed is True
    assert sorted(result.systems_import_ready) == ["loinc", "rxnorm"]


def test_rxnorm_streaming_parser_handles_synthetic_rrf(tmp_path: Path) -> None:
    path = tmp_path / "RXNCONSO.RRF"
    path.write_text(_rxnorm_rows(5), encoding="utf-8")
    concepts = list(_iter_rxnorm_concepts(path, source_safe_id="term02_test", max_rows=3))
    assert len(concepts) == 3
    assert concepts[0][1] is not None
    assert concepts[0][1].system == TerminologySystem.RXNORM


def test_loinc_streaming_parser_handles_synthetic_csv(tmp_path: Path) -> None:
    path = tmp_path / "Loinc.csv"
    path.write_text(_loinc_rows(5), encoding="utf-8")
    concepts = list(_iter_loinc_concepts(path, source_safe_id="term02_test", max_rows=3))
    assert len(concepts) == 3
    assert concepts[0][1] is not None
    assert concepts[0][1].system == TerminologySystem.LOINC


def test_controlled_import_synthetic_temp_root(tmp_path: Path) -> None:
    terminology_root = _synthetic_root(tmp_path, ack=True)
    db_path = tmp_path / "data" / "terminology" / "term02.sqlite"
    result = run_controlled_local_import(
        repo_root=tmp_path,
        terminology_root=terminology_root,
        db_path=db_path,
        limits=build_import_limits(max_rows_per_file=20, chunk_size=5, checkpoint_interval_rows=5, allow_real_import=True),
    )
    assert result.term02_completed is True
    assert result.imported_systems == ("rxnorm", "loinc")
    assert result.real_import_performed is True
    assert result.external_api_used is False
    assert db_path.exists()
    assert result.lookup_validation.unknown_unmapped_passed is True
    assert result.lookup_validation.ambiguous_flag_passed is True


def test_import_is_chunked_and_bounded(tmp_path: Path) -> None:
    terminology_root = _synthetic_root(tmp_path, ack=True, rows=25)
    result = run_controlled_local_import(
        repo_root=tmp_path,
        terminology_root=terminology_root,
        db_path=tmp_path / "data" / "terminology" / "term02.sqlite",
        limits=build_import_limits(max_rows_per_file=10, chunk_size=4, checkpoint_interval_rows=4, allow_real_import=True),
    )
    summaries = {summary.system: summary for summary in result.file_summaries}
    assert summaries["rxnorm"].rows_seen == 10
    assert summaries["loinc"].rows_seen == 10
    assert summaries["rxnorm"].chunks_processed == 3
    assert summaries["loinc"].checkpoint_count >= 2


def test_report_privacy_and_no_raw_paths(tmp_path: Path) -> None:
    terminology_root = _synthetic_root(tmp_path, ack=True)
    result = run_controlled_local_import(
        repo_root=tmp_path,
        terminology_root=terminology_root,
        db_path=tmp_path / "data" / "terminology" / "term02.sqlite",
        limits=build_import_limits(max_rows_per_file=10, chunk_size=5, allow_real_import=True),
    )
    payload = result.safe_public_summary()
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert "Synthetic RxNorm" not in serialized
    check = check_public_report_payload(payload)
    assert check.passed is True


def test_no_private_or_runtime_files_staged() -> None:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True)
    staged = proc.stdout.replace("\\", "/")
    assert "terminology_data/" not in staged
    assert "data/terminology/" not in staged
    assert "LICENSE_ACK_PRIVATE" not in staged
    assert ".RRF" not in staged
    assert ".csv" not in staged.lower()
    assert ".sqlite" not in staged


def test_validation_script_runs_if_real_preflight_available() -> None:
    preflight = subprocess.run([sys.executable, "scripts/cka_term02_preflight_gate.py", "--json"], cwd=ROOT, text=True, capture_output=True)
    payload = json.loads(preflight.stdout)
    if not payload.get("term02_may_start"):
        return
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term02_controlled_local_import_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term02_controlled_local_import_ready" in proc.stdout


def test_current_license_ack_covers_rxnorm_and_loinc_if_present() -> None:
    ack = ROOT / "terminology_data" / "LICENSE_ACK_PRIVATE.json"
    if not ack.exists():
        return
    assert license_acknowledged_for(TerminologySystem.RXNORM, local_ack_file=ack)
    assert license_acknowledged_for(TerminologySystem.LOINC, local_ack_file=ack)


def _synthetic_root(tmp_path: Path, *, ack: bool, rows: int = 8) -> Path:
    root = tmp_path / "terminology_data"
    (root / "rxnorm").mkdir(parents=True)
    (root / "loinc").mkdir(parents=True)
    (root / "rxnorm" / "RXNCONSO.RRF").write_text(_rxnorm_rows(rows), encoding="utf-8")
    (root / "loinc" / "Loinc.csv").write_text(_loinc_rows(rows), encoding="utf-8")
    if ack:
        (root / "LICENSE_ACK_PRIVATE.json").write_text(
            json.dumps({"operator_acknowledged": True, "acknowledged_systems": ["rxnorm", "loinc"]}),
            encoding="utf-8",
        )
    return root


def _rxnorm_rows(rows: int) -> str:
    out = []
    for i in range(rows):
        fields = [f"RX{i:04d}", "ENG"] + [""] * 12 + [f"Synthetic RxNorm {i}"] + ["", "", ""]
        out.append("|".join(fields))
    return "\n".join(out) + "\n"


def _loinc_rows(rows: int) -> str:
    lines = ["LOINC_NUM,LONG_COMMON_NAME,COMPONENT"]
    for i in range(rows):
        lines.append(f"{i:04d}-1,Synthetic LOINC {i},Synthetic Component {i}")
    return "\n".join(lines) + "\n"
