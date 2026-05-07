from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology import (
    LocalTerminologyStore,
    RealTerminologyImportBlocked,
    TerminologyImportExecutor,
    TerminologyImportLimits,
    TerminologyLookupService,
    TerminologyLookupStatus,
    TerminologySystem,
    code_entity_via_local_terminology,
    parse_loinc_csv,
    parse_rxnorm_rxnconso,
    parse_snomed_concept_description,
    parse_umls_mrconso,
    safe_b07_boundary_summary,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


UMLS_TEXT = "\n".join([
    "C0001|ENG|P|L1|PF|S1|Y|A1||||MTH|PT|U1|hypertension||N|",
    "C0001|ENG|P|L1|PF|S2|N|A2||||MTH|SY|U2|high blood pressure||N|",
    "C0002|ENG|P|L2|PF|S3|Y|A3||||MTH|PT|U3|fatigue||N|",
])
SNOMED_CONCEPT = "\n".join([
    "100\t20240101\t1\tm\t900",
    "200\t20240101\t1\tm\t900",
])
SNOMED_DESCRIPTION = "\n".join([
    "d1\t20240101\t1\tm\t100\ten\t900000000000003001\tDiabetes mellitus type 2 (disorder)\tp",
    "d2\t20240101\t1\tm\t100\ten\t900000000000013009\ttype 2 diabetes\tp",
    "d3\t20240101\t1\tm\t200\ten\t900000000000003001\tFatigue (finding)\tp",
])
RXNORM_TEXT = "\n".join([
    "R001|ENG|P|L1|PF|S1|Y|A1||||RXNORM|IN|RX1|aspirin||N|",
    "R002|ENG|P|L2|PF|S2|Y|A2||||RXNORM|IN|RX2|metformin||N|",
])
LOINC_TEXT = "\n".join([
    "LOINC_NUM,COMPONENT,LONG_COMMON_NAME",
    "1-1,Glucose,Glucose synthetic lab",
    "2-2,Hemoglobin,Hemoglobin synthetic lab",
])


def _executor(limits: TerminologyImportLimits | None = None) -> TerminologyImportExecutor:
    return TerminologyImportExecutor(store=LocalTerminologyStore(), limits=limits)


def test_synthetic_umls_import_transactional() -> None:
    result = _executor().execute_synthetic(parse_umls_mrconso(text=UMLS_TEXT), source_safe_id="term_src_safe_umls")
    summary = result.safe_public_summary()
    assert summary["audit"]["import_completed"] is True
    assert summary["audit"]["records_imported"] == 2
    assert summary["store_summary"]["sources_count"] == 1
    assert summary["store_summary"]["concepts_count"] == 2
    assert summary["store_summary"]["synonyms_count"] == 1
    assert summary["store_summary"]["import_events_count"] == 1


def test_synthetic_snomed_import_transactional() -> None:
    parsed = parse_snomed_concept_description(
        concept_text=SNOMED_CONCEPT,
        description_text=SNOMED_DESCRIPTION,
    )
    result = _executor().execute_synthetic(parsed, source_safe_id="term_src_safe_snomed")
    assert result.audit.import_completed is True
    assert result.store_summary["concepts_count"] == 2
    assert result.store_summary["synonyms_count"] == 1


def test_synthetic_rxnorm_import_transactional() -> None:
    result = _executor().execute_synthetic(parse_rxnorm_rxnconso(text=RXNORM_TEXT), source_safe_id="term_src_safe_rxnorm")
    assert result.audit.import_completed is True
    assert result.store_summary["concepts_count"] == 2


def test_synthetic_loinc_import_transactional() -> None:
    result = _executor().execute_synthetic(parse_loinc_csv(text=LOINC_TEXT), source_safe_id="term_src_safe_loinc")
    assert result.audit.import_completed is True
    assert result.store_summary["concepts_count"] == 2


def test_dry_run_does_not_write_store() -> None:
    executor = _executor()
    result = executor.dry_run(parse_loinc_csv(text=LOINC_TEXT), source_safe_id="term_src_safe_loinc")
    assert result.audit.import_mode == "dry_run"
    assert result.audit.import_completed is True
    assert result.store_summary["concepts_count"] == 0
    assert result.store_summary["sources_count"] == 0


def test_row_cap_stops_import_safely() -> None:
    limits = TerminologyImportLimits(max_rows_per_file_default=1, chunk_size=1, checkpoint_interval_rows=1)
    result = _executor(limits).execute_synthetic(parse_loinc_csv(text=LOINC_TEXT), source_safe_id="term_src_safe_loinc")
    assert result.audit.import_completed is True
    assert result.audit.records_imported == 1
    assert result.audit.records_skipped == 1
    assert result.store_summary["concepts_count"] == 1


def test_simulated_failure_rolls_back() -> None:
    executor = _executor()
    result = executor.execute_synthetic(
        parse_rxnorm_rxnconso(text=RXNORM_TEXT),
        source_safe_id="term_src_safe_rxnorm",
        simulate_failure_after_source=True,
    )
    assert result.audit.rollback_performed is True
    assert result.audit.import_completed is False
    assert result.store_summary["sources_count"] == 0
    assert result.store_summary["concepts_count"] == 0
    assert result.store_summary["import_events_count"] == 0


def test_checkpoint_simulation_records_progress_safely() -> None:
    limits = TerminologyImportLimits(checkpoint_interval_rows=1, chunk_size=1)
    result = _executor(limits).execute_synthetic(parse_umls_mrconso(text=UMLS_TEXT), source_safe_id="term_src_safe_umls")
    summary = result.safe_public_summary()
    assert summary["audit"]["checkpoint_count"] >= 2
    assert summary["checkpoints"]
    text = json.dumps(summary)
    assert ":\\" not in text
    assert "/tmp/" not in text


def test_real_import_blocked_by_default() -> None:
    with pytest.raises(RealTerminologyImportBlocked):
        _executor().execute_real_import_blocked()


def test_b07_boundary_unchanged_and_no_code_hallucinated() -> None:
    executor = _executor()
    executor.execute_synthetic(parse_rxnorm_rxnconso(text=RXNORM_TEXT), source_safe_id="term_src_safe_rxnorm")
    service = TerminologyLookupService(executor.store)
    mapped = code_entity_via_local_terminology("metformin", service, systems=[TerminologySystem.RXNORM])
    unknown = code_entity_via_local_terminology("zzzz unknown", service, systems=[TerminologySystem.RXNORM])
    boundary = safe_b07_boundary_summary()
    assert mapped.status == TerminologyLookupStatus.EXACT
    assert unknown.status == TerminologyLookupStatus.UNMAPPED
    assert unknown.matches == []
    assert boundary["default_b07_behavior_unchanged"] is True
    assert boundary["coding_promotes_hypothesis"] is False
    assert boundary["coding_clears_ddi_status"] is False
    assert boundary["no_code_hallucinated"] is True


def test_executor_public_summary_privacy_clean() -> None:
    result = _executor().execute_synthetic(parse_umls_mrconso(text=UMLS_TEXT), source_safe_id="term_src_safe_umls")
    check = check_public_report_payload(result.safe_public_summary())
    assert check.passed


def test_validation_script_runs_and_report_privacy_clean() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_cka_term01c_import_executor_validation.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=360,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report_path = REPO_ROOT / "reports" / "cka_term01c_import_executor" / "cka_term01c_import_executor_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["conclusion"] == "cka_term01c_synthetic_import_executor_ready"
    assert report["real_import_performed"] is False
    assert check_public_report_payload(report).passed


def test_no_terminology_data_or_index_staged() -> None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0
    staged = result.stdout.splitlines()
    assert not any(line.startswith("terminology_data/") for line in staged)
    assert not any(line.startswith("data/terminology/") for line in staged)
