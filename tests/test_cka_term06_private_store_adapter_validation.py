from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologyConcept, TerminologySystem
from clinical_knowledge.terminology.term06_private_store_adapter_validation import (
    Term06ValidationBlocked,
    run_private_store_adapter_validation,
)

ROOT = Path(__file__).resolve().parents[1]


def _private_like_store(path: Path) -> None:
    store = LocalTerminologyStore(str(path))
    rx = store.register_source(TerminologySystem.RXNORM, safe_source_id="term06_rx", license_confirmed=True)
    loinc = store.register_source(TerminologySystem.LOINC, safe_source_id="term06_loinc", license_confirmed=True)
    store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term06_rx_exact",
                system=TerminologySystem.RXNORM,
                code="RX06A",
                display="term06 rx private exact",
                synthetic=False,
            ),
            TerminologyConcept(
                concept_id="term06_rx_amb",
                system=TerminologySystem.RXNORM,
                code="RX06AMB",
                display="term06 shared private",
                synthetic=False,
            ),
        ],
        rx,
    )
    store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term06_loinc_exact",
                system=TerminologySystem.LOINC,
                code="L06A",
                display="term06 loinc private exact",
                synthetic=False,
            ),
            TerminologyConcept(
                concept_id="term06_loinc_amb",
                system=TerminologySystem.LOINC,
                code="L06AMB",
                display="term06 shared private",
                synthetic=False,
            ),
        ],
        loinc,
    )


def test_private_store_validation_on_temp_store(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    result = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path)
    assert result.conclusion == "cka_term06_private_store_adapter_validation_ready"
    assert result.store_opened_read_only is True
    assert result.write_attempt_blocked is True
    assert result.cases_failed == 0
    assert result.private_store_accessed_read_only is True


def test_missing_store_blocks(tmp_path: Path) -> None:
    with pytest.raises(Term06ValidationBlocked, match="term02_local_store_missing"):
        run_private_store_adapter_validation(repo_root=ROOT, db_path=tmp_path / "missing.sqlite")


def test_exact_code_unknown_ambiguous_and_filter_behaviors(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    result = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path)
    assert result.exact_rxnorm_passed is True
    assert result.exact_loinc_passed is True
    assert result.code_lookup_passed is True
    assert result.source_filter_isolation_passed is True
    assert result.unknown_unmapped_passed is True
    assert result.ambiguous_manual_review_passed is True


def test_determinism_and_normalization(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    result = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path)
    assert result.determinism_passed is True
    assert result.normalization_passed is True


def test_read_only_store_rejects_write(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    result = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path)
    assert result.write_attempt_blocked is True
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'term06_write_probe'"
        ).fetchone()
    assert row is None


def test_public_report_privacy_clean(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    payload = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path).safe_public_summary()
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "data/terminology" not in rendered.replace("\\", "/")
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered
    check = check_public_report_payload(payload)
    assert check.passed is True


def test_no_external_api_or_clinical_side_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "term06.sqlite"
    _private_like_store(db_path)
    result = run_private_store_adapter_validation(repo_root=ROOT, db_path=db_path)
    assert result.external_api_used is False
    assert result.clinical_advice_generated is False
    assert result.dosing_advice_generated is False
    assert result.mkb_write_performed is False
    assert result.automatic_annotation_created is False
    assert result.b07_integrated is False
    assert result.ddi_status_cleared is False
    assert result.hypothesis_promoted is False
    assert result.real_import_performed is False
    assert result.store_recreated is False


def test_no_private_or_runtime_files_staged() -> None:
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert staged.returncode == 0
    lowered = staged.stdout.lower().replace("\\", "/")
    assert "terminology_data" not in lowered
    assert "data/terminology" not in lowered
    assert "license_ack_private" not in lowered
    assert ".rrf" not in lowered
    assert ".csv" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered


def test_validation_script_runs_when_real_store_exists() -> None:
    db_path = ROOT / "data" / "terminology" / "term02_local_terminology.sqlite"
    if not db_path.exists():
        pytest.skip("TERM-02 local store is not available in this environment")
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term06_private_store_adapter_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cka_term06_private_store_adapter_validation_ready" in proc.stdout
