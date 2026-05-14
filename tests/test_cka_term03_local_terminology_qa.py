from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologyConcept, TerminologySystem
from clinical_knowledge.terminology.term03_local_qa import Term03QABlocked, run_local_terminology_qa


def _synthetic_store() -> LocalTerminologyStore:
    store = LocalTerminologyStore()
    rx_sid = store.register_source(TerminologySystem.RXNORM, safe_source_id="test_rx", license_confirmed=True)
    loinc_sid = store.register_source(TerminologySystem.LOINC, safe_source_id="test_loinc", license_confirmed=True)
    store.add_concepts(
        [
            TerminologyConcept.synthetic_for(
                TerminologySystem.RXNORM,
                code="RX03A",
                display="term03 rxnorm exact",
                synonyms=["term03 rx alias"],
                source_safe_id="test_rx",
            ),
            TerminologyConcept.synthetic_for(
                TerminologySystem.RXNORM,
                code="RX03B",
                display="term03 rxnorm second",
                source_safe_id="test_rx",
            ),
        ],
        rx_sid,
    )
    store.add_concepts(
        [
            TerminologyConcept.synthetic_for(
                TerminologySystem.LOINC,
                code="L03A",
                display="term03 loinc exact",
                source_safe_id="test_loinc",
            ),
            TerminologyConcept.synthetic_for(
                TerminologySystem.LOINC,
                code="L03B",
                display="term03 loinc second",
                source_safe_id="test_loinc",
            ),
        ],
        loinc_sid,
    )
    return store


def test_synthetic_rxnorm_and_loinc_exact_lookup() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    assert result.conclusion == "cka_term03_local_terminology_qa_ready"
    cases = {case.case_id: case for case in result.qa_case_results}
    assert cases["term03_rxnorm_exact"].passed is True
    assert cases["term03_loinc_exact"].passed is True


def test_unknown_unmapped_and_no_code_hallucination() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    assert result.unknown_unmapped_passed is True
    assert result.no_code_hallucinated is True
    unknown = next(case for case in result.qa_case_results if case.case_id == "term03_unknown_unmapped")
    assert unknown.observed_status == "unmapped"
    assert unknown.matches_count == 0


def test_ambiguous_no_silent_resolution_behavior() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    ambiguous = next(case for case in result.qa_case_results if case.case_id == "term03_ambiguous_manual_review")
    assert ambiguous.passed is True
    assert ambiguous.observed_status == "ambiguous"
    assert ambiguous.ambiguous is True


def test_deterministic_repeated_lookup_and_normalization() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    det = next(case for case in result.qa_case_results if case.case_id == "term03_deterministic_normalization")
    assert det.passed is True
    assert result.determinism_passed is True


def test_source_filter_isolation() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    assert result.source_filter_isolation_passed is True
    filter_cases = [case for case in result.qa_case_results if case.case_type == "source_filter_isolation"]
    assert filter_cases
    assert all(case.observed_status == "unmapped" for case in filter_cases)


def test_public_report_privacy_cleanliness() -> None:
    payload = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store()).safe_public_summary()
    rendered = json.dumps(payload)
    assert "terminology_data/" not in rendered.replace("\\", "/")
    assert "LICENSE_ACK_PRIVATE" not in rendered
    assert "RXNCONSO" not in rendered
    assert "Loinc.csv" not in rendered
    check = check_public_report_payload(payload)
    assert check.passed is True


def test_no_external_api_and_no_clinical_advice_flags() -> None:
    result = run_local_terminology_qa(repo_root=ROOT, synthetic_store=_synthetic_store())
    assert result.external_api_used is False
    assert result.clinical_recommendations_generated is False
    assert result.prescription_dosing_advice_generated is False
    assert result.coding_promotes_hypothesis is False
    assert result.coding_clears_ddi_status is False


def test_missing_real_store_blocks(tmp_path: Path) -> None:
    with pytest.raises(Term03QABlocked, match="term02_local_store_missing"):
        run_local_terminology_qa(repo_root=tmp_path, db_path=tmp_path / "missing.sqlite")


def test_no_private_or_licensed_files_staged() -> None:
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert staged.returncode == 0
    lowered = staged.stdout.lower()
    assert "terminology_data" not in lowered
    assert "license_ack_private" not in lowered
    assert "data/terminology" not in lowered.replace("\\", "/")
    assert ".rrf" not in lowered
    assert ".csv" not in lowered
    assert ".sqlite" not in lowered
    assert ".db" not in lowered


def test_validation_script_runs_when_real_store_exists() -> None:
    db_path = ROOT / "data" / "terminology" / "term02_local_terminology.sqlite"
    if not db_path.exists():
        pytest.skip("TERM-02 local store is not available in this environment")
    proc = subprocess.run(
        [sys.executable, "scripts/run_cka_term03_local_terminology_qa_validation.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cka_term03_local_terminology_qa_ready" in proc.stdout
