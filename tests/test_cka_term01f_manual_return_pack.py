from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.terminology.manual_return_pack import (
    build_manual_return_guide_text,
    build_term02_preflight_checklist_text,
    run_manual_return_pack,
)
from clinical_knowledge.terminology.synthetic_intake_rehearsal import run_synthetic_intake_rehearsal
from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate


def test_manual_return_guide_safe_and_actionable() -> None:
    text = build_manual_return_guide_text()
    assert "TERM-02 has not started" in text
    assert "terminology_data/" in text
    assert "LICENSE_ACK_PRIVATE.json" in text
    assert "Keep vendor terms out of reports and public files" in text
    assert "No clinical advice is generated" in text
    assert "dosing advice" not in text.lower()


def test_preflight_checklist_mentions_required_gate_items() -> None:
    text = build_term02_preflight_checklist_text()
    assert "operator_acknowledged" in text
    assert "acknowledged_systems" in text
    assert "No `terminology_data/` files are staged" in text


def test_preflight_blocks_without_manual_files(tmp_path: Path) -> None:
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=tmp_path / "terminology_data")
    assert result.allowed is False
    assert "terminology_data_missing" in result.reason_codes
    assert "license_ack_private_missing" in result.reason_codes


def test_preflight_blocks_files_without_ack(tmp_path: Path) -> None:
    root = tmp_path / "terminology_data"
    (root / "loinc").mkdir(parents=True)
    (root / "loinc" / "Loinc.csv").write_text("LOINC_NUM,COMPONENT\nSYN-1,Example\n", encoding="utf-8")
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=root)
    assert result.allowed is False
    assert "license_ack_private_missing" in result.reason_codes
    assert "systems_pending_acknowledgment" in result.reason_codes


def test_preflight_passes_with_synthetic_temp_ack(tmp_path: Path) -> None:
    root = tmp_path / "terminology_data"
    (root / "loinc").mkdir(parents=True)
    (root / "loinc" / "Loinc.csv").write_text("LOINC_NUM,COMPONENT\nSYN-1,Example\n", encoding="utf-8")
    (root / "LICENSE_ACK_PRIVATE.json").write_text(
        json.dumps({"operator_acknowledged": True, "acknowledged_systems": ["loinc"]}),
        encoding="utf-8",
    )
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=root)
    assert result.allowed is True
    assert result.reason_codes == []
    assert result.systems_import_ready == ["loinc"]


def test_preflight_blocks_pending_ack_system(tmp_path: Path) -> None:
    root = tmp_path / "terminology_data"
    (root / "loinc").mkdir(parents=True)
    (root / "rxnorm").mkdir()
    (root / "loinc" / "Loinc.csv").write_text("LOINC_NUM,COMPONENT\n", encoding="utf-8")
    (root / "rxnorm" / "RXNCONSO.RRF").write_text("RXCUI|LAT|STR|\n", encoding="utf-8")
    (root / "LICENSE_ACK_PRIVATE.json").write_text(
        json.dumps({"operator_acknowledged": True, "acknowledged_systems": ["loinc"]}),
        encoding="utf-8",
    )
    result = run_term02_preflight_gate(repo_root=tmp_path, terminology_root=root)
    assert result.allowed is False
    assert "systems_pending_acknowledgment" in result.reason_codes
    assert result.systems_pending_acknowledgment == ["rxnorm"]


def test_synthetic_intake_rehearsal_passes_without_repo_data_creation() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = run_synthetic_intake_rehearsal(repo_root=repo)
    assert result.classification_passed is True
    assert result.safe_entries_extracted is True
    assert result.zip_slip_protection_verified is True
    assert result.readiness_passed is True
    assert result.dry_run_passed is True
    assert result.term02_preflight_passed is True
    assert result.repo_terminology_data_created is False


def test_manual_return_pack_does_not_import_or_download() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = run_manual_return_pack(repo_root=repo, run_prepare=False, run_readiness=True, run_dry_run=True, run_preflight=True)
    summary = result.safe_public_summary()
    assert summary["real_import_performed"] is False
    assert summary["external_api_used"] is False
    assert summary["term02_preflight_checked"] is True


def test_term02_preflight_cli_outputs_safe_json() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/cka_term02_preflight_gate.py", "--json"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in {0, 2}
    payload = json.loads(proc.stdout)
    assert payload["real_import_performed"] is False
    assert payload["external_api_used"] is False
    assert "C:\\" not in proc.stdout


def test_manual_return_pack_cli_does_not_start_term02() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/cka_terminology_manual_return_pack.py", "--json"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["term02_not_started"] is True
    assert payload["real_license_ack_created"] is False
    assert payload["no_real_terminology_import_performed"] is True


def test_validation_script_generates_safe_public_report() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term01f_manual_return_pack_validation.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "cka_term01f_manual_return_pack_ready" in proc.stdout
    report = repo / "reports" / "cka_term01f_manual_return_pack" / "cka_term01f_manual_return_pack_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["raw_phi_logged_in_public_reports"] is False
    assert payload["private_filename_path_leaks"] == 0
    assert payload["secret_leaks"] == 0
    assert payload["real_license_ack_created"] is False
    serialized = json.dumps(payload)
    assert "C:\\" not in serialized
    assert "source_response_raw" not in serialized


def test_no_terminology_or_data_files_staged() -> None:
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
