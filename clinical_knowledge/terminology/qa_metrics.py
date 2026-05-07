"""CKA-TERM-01D terminology QA metrics."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TerminologyQAMetrics:
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    exact_match_passed: bool = False
    synonym_match_passed: bool = False
    ambiguous_flag_passed: bool = False
    unmapped_no_hallucination_passed: bool = False
    b07_boundary_passed: bool = False
    external_api_used: bool = False
    real_terminology_imported: bool = False
    failed_case_ids: list[str] = field(default_factory=list)

    def safe_public_summary(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "exact_match_passed": self.exact_match_passed,
            "synonym_match_passed": self.synonym_match_passed,
            "ambiguous_flag_passed": self.ambiguous_flag_passed,
            "unmapped_no_hallucination_passed": self.unmapped_no_hallucination_passed,
            "b07_boundary_passed": self.b07_boundary_passed,
            "external_api_used": self.external_api_used,
            "real_terminology_imported": self.real_terminology_imported,
            "failed_case_ids": list(self.failed_case_ids),
        }
