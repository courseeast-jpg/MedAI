"""CKA-TERM-01D synthetic terminology QA runner."""
from __future__ import annotations

from dataclasses import dataclass

from clinical_knowledge.terminology.integration import code_entity_via_local_terminology, safe_b07_boundary_summary
from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import TerminologyLookupStatus, TerminologySystem
from clinical_knowledge.terminology.qa_golden import (
    TerminologyGoldenCase,
    build_synthetic_qa_store,
    synthetic_golden_cases,
)
from clinical_knowledge.terminology.qa_metrics import TerminologyQAMetrics


@dataclass(frozen=True)
class TerminologyQACaseResult:
    case_id: str
    passed: bool
    expected_status: str
    actual_status: str
    expected_codes_count: int
    actual_codes_count: int
    no_code_hallucinated: bool
    tags: tuple[str, ...]

    def safe_public_summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "expected_codes_count": self.expected_codes_count,
            "actual_codes_count": self.actual_codes_count,
            "no_code_hallucinated": self.no_code_hallucinated,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class TerminologyQAReport:
    metrics: TerminologyQAMetrics
    case_results: list[TerminologyQACaseResult]
    fixture_metadata: dict
    b07_boundary: dict

    def safe_public_summary(self) -> dict:
        return {
            "metrics": self.metrics.safe_public_summary(),
            "case_results": [result.safe_public_summary() for result in self.case_results],
            "fixture_metadata": dict(self.fixture_metadata),
            "b07_boundary": dict(self.b07_boundary),
        }


def run_synthetic_terminology_qa() -> TerminologyQAReport:
    store, metadata = build_synthetic_qa_store()
    service = TerminologyLookupService(store)
    cases = synthetic_golden_cases()
    case_results = [_run_case(service, case) for case in cases]
    boundary = safe_b07_boundary_summary()
    b07_lookup = code_entity_via_local_terminology(
        "hypertension",
        service,
        systems=[TerminologySystem.UMLS],
    )
    b07_boundary_passed = (
        boundary["default_b07_behavior_unchanged"] is True
        and boundary["coding_promotes_hypothesis"] is False
        and boundary["coding_clears_ddi_status"] is False
        and boundary["no_code_hallucinated"] is True
        and b07_lookup.status == TerminologyLookupStatus.EXACT
    )
    metrics = _build_metrics(case_results, b07_boundary_passed=b07_boundary_passed)
    return TerminologyQAReport(
        metrics=metrics,
        case_results=case_results,
        fixture_metadata=metadata,
        b07_boundary=boundary,
    )


def _run_case(service: TerminologyLookupService, case: TerminologyGoldenCase) -> TerminologyQACaseResult:
    systems = list(case.systems) if case.systems else None
    result = service.lookup(case.query, systems=systems)
    actual_codes = tuple(sorted(match.code for match in result.matches))
    expected_codes = tuple(sorted(case.expected_codes))
    status_ok = result.status == case.expected_status
    if case.expected_status == TerminologyLookupStatus.AMBIGUOUS:
        codes_ok = set(expected_codes).issubset(set(actual_codes)) and result.ambiguous
    else:
        codes_ok = actual_codes == expected_codes
    no_hallucination = not (
        case.expected_status == TerminologyLookupStatus.UNMAPPED
        and result.matches
    )
    return TerminologyQACaseResult(
        case_id=case.case_id,
        passed=status_ok and codes_ok and no_hallucination,
        expected_status=case.expected_status.value,
        actual_status=result.status.value,
        expected_codes_count=len(expected_codes),
        actual_codes_count=len(actual_codes),
        no_code_hallucinated=no_hallucination and result.no_code_hallucinated,
        tags=case.tags,
    )


def _build_metrics(
    case_results: list[TerminologyQACaseResult],
    *,
    b07_boundary_passed: bool,
) -> TerminologyQAMetrics:
    failed = [result.case_id for result in case_results if not result.passed]
    return TerminologyQAMetrics(
        total_cases=len(case_results),
        passed_cases=sum(1 for result in case_results if result.passed),
        failed_cases=len(failed),
        exact_match_passed=_tag_passed(case_results, "exact_match"),
        synonym_match_passed=_tag_passed(case_results, "synonym_match"),
        ambiguous_flag_passed=_tag_passed(case_results, "ambiguous_match"),
        unmapped_no_hallucination_passed=_tag_passed(case_results, "unmapped_term"),
        b07_boundary_passed=b07_boundary_passed,
        external_api_used=False,
        real_terminology_imported=False,
        failed_case_ids=failed,
    )


def _tag_passed(case_results: list[TerminologyQACaseResult], tag: str) -> bool:
    tagged = [result for result in case_results if tag in result.tags]
    return bool(tagged) and all(result.passed for result in tagged)
