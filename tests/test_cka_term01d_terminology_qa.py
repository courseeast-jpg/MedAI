from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.terminology import (
    TerminologyLookupService,
    TerminologyLookupStatus,
    TerminologySystem,
    build_synthetic_qa_store,
    code_entity_via_local_terminology,
    run_synthetic_terminology_qa,
    safe_b07_boundary_summary,
    synthetic_golden_cases,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_qa_fixtures_load() -> None:
    store, metadata = build_synthetic_qa_store()
    assert sorted(metadata["systems_loaded"]) == ["loinc", "rxnorm", "snomed_ct", "umls"]
    assert metadata["malformed_codes_skipped"] == 1
    assert metadata["real_terminology_imported"] is False
    assert store.safe_public_summary()["sources_count"] == 4
    assert len(synthetic_golden_cases()) >= 7


def test_exact_match_passes() -> None:
    report = run_synthetic_terminology_qa()
    assert report.metrics.exact_match_passed is True


def test_synonym_match_passes() -> None:
    report = run_synthetic_terminology_qa()
    assert report.metrics.synonym_match_passed is True


def test_ambiguous_term_is_flagged() -> None:
    report = run_synthetic_terminology_qa()
    assert report.metrics.ambiguous_flag_passed is True


def test_unknown_term_is_unmapped_and_no_code_hallucinated() -> None:
    report = run_synthetic_terminology_qa()
    assert report.metrics.unmapped_no_hallucination_passed is True
    unknown_cases = [
        result for result in report.case_results
        if "unmapped_term" in result.tags
    ]
    assert unknown_cases
    assert all(result.actual_codes_count == 0 for result in unknown_cases)


def test_multi_system_duplicate_is_ambiguous_not_resolved() -> None:
    store, _ = build_synthetic_qa_store()
    result = TerminologyLookupService(store).lookup("shared duplicate")
    assert result.status == TerminologyLookupStatus.AMBIGUOUS
    assert result.ambiguous is True


def test_malformed_code_skipped() -> None:
    store, metadata = build_synthetic_qa_store()
    result = TerminologyLookupService(store).lookup("malformed skipped")
    assert metadata["malformed_codes_skipped"] == 1
    assert result.status == TerminologyLookupStatus.UNMAPPED
    assert result.matches == []


def test_inactive_concept_excluded() -> None:
    store, metadata = build_synthetic_qa_store()
    result = TerminologyLookupService(store).lookup("inactive hidden")
    assert metadata["inactive_concepts_loaded"] == 1
    assert result.status == TerminologyLookupStatus.UNMAPPED
    assert result.matches == []


def test_b07_opt_in_helper_preserves_boundary() -> None:
    store, _ = build_synthetic_qa_store()
    service = TerminologyLookupService(store)
    result = code_entity_via_local_terminology("hypertension", service, systems=[TerminologySystem.UMLS])
    boundary = safe_b07_boundary_summary()
    assert result.status == TerminologyLookupStatus.EXACT
    assert boundary["default_b07_behavior_unchanged"] is True
    assert boundary["coding_promotes_hypothesis"] is False
    assert boundary["coding_clears_ddi_status"] is False
    assert boundary["no_code_hallucinated"] is True


def test_qa_runner_metrics_complete() -> None:
    report = run_synthetic_terminology_qa()
    summary = report.safe_public_summary()
    metrics = summary["metrics"]
    assert metrics["total_cases"] == 7
    assert metrics["passed_cases"] == 7
    assert metrics["failed_cases"] == 0
    assert metrics["b07_boundary_passed"] is True
    assert metrics["external_api_used"] is False
    assert metrics["real_terminology_imported"] is False


def test_qa_cli_json_runs_without_external_api() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/cka_terminology_run_qa.py", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["metrics"]["failed_cases"] == 0
    assert payload["metrics"]["external_api_used"] is False


def test_validation_script_generates_privacy_clean_report() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_cka_term01d_qa_validation.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=620,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report_path = REPO_ROOT / "reports" / "cka_term01d_terminology_qa" / "cka_term01d_terminology_qa_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["conclusion"] == "cka_term01d_terminology_qa_ready"
    assert report["real_terminology_imported"] is False
    assert report["external_api_used"] is False
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
