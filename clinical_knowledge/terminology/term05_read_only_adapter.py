"""CKA-TERM-05 synthetic read-only terminology adapter.

This adapter defines the future terminology lookup interface using only
synthetic in-memory fixtures. It does not access the private TERM-02 store,
does not write to the MKB, and does not integrate with B07.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import (
    TerminologyLookupStatus,
    TerminologySystem,
    normalize_query,
)
from clinical_knowledge.terminology.qa_golden import build_synthetic_qa_store


@dataclass(frozen=True)
class Term05AdapterMatch:
    system: str
    code: str
    display_normalized: str
    synthetic: bool = True

    def safe_public_summary(self) -> dict:
        return {
            "system": self.system,
            "code": self.code,
            "display_normalized": self.display_normalized,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class Term05AdapterResult:
    query: str
    normalized_query: str
    source_filter: tuple[str, ...]
    status: str
    matches: tuple[Term05AdapterMatch, ...] = field(default_factory=tuple)
    confidence_label: str = "none"
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    read_only: bool = True
    external_api_used: bool = False
    clinical_advice_generated: bool = False
    dosing_advice_generated: bool = False
    mkb_write_performed: bool = False
    b07_integrated: bool = False
    ddi_status_cleared: bool = False
    hypothesis_promoted: bool = False
    no_code_hallucinated: bool = True

    def safe_public_summary(self) -> dict:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "source_filter": list(self.source_filter),
            "status": self.status,
            "matches": [match.safe_public_summary() for match in self.matches],
            "confidence_label": self.confidence_label,
            "reason_codes": list(self.reason_codes),
            "read_only": self.read_only,
            "external_api_used": self.external_api_used,
            "clinical_advice_generated": self.clinical_advice_generated,
            "dosing_advice_generated": self.dosing_advice_generated,
            "mkb_write_performed": self.mkb_write_performed,
            "b07_integrated": self.b07_integrated,
            "ddi_status_cleared": self.ddi_status_cleared,
            "hypothesis_promoted": self.hypothesis_promoted,
            "no_code_hallucinated": self.no_code_hallucinated,
        }


@dataclass(frozen=True)
class Term05AdapterValidationResult:
    conclusion: str
    cases_total: int
    cases_passed: int
    cases_failed: int
    case_results: tuple[dict, ...]
    fixture_metadata: dict
    exact_rxnorm_passed: bool
    exact_loinc_passed: bool
    code_lookup_passed: bool
    source_filter_isolation_passed: bool
    unknown_unmapped_passed: bool
    ambiguous_manual_review_passed: bool
    determinism_passed: bool
    normalization_passed: bool
    read_only: bool = True
    synthetic_only: bool = True
    private_store_accessed: bool = False
    terminology_data_accessed: bool = False
    data_terminology_accessed: bool = False
    external_api_used: bool = False
    clinical_advice_generated: bool = False
    dosing_advice_generated: bool = False
    mkb_write_performed: bool = False
    b07_integrated: bool = False
    ddi_status_cleared: bool = False
    hypothesis_promoted: bool = False
    no_code_hallucinated: bool = True

    def safe_public_summary(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "cases_total": self.cases_total,
            "cases_passed": self.cases_passed,
            "cases_failed": self.cases_failed,
            "case_results": list(self.case_results),
            "fixture_metadata": dict(self.fixture_metadata),
            "exact_rxnorm_passed": self.exact_rxnorm_passed,
            "exact_loinc_passed": self.exact_loinc_passed,
            "code_lookup_passed": self.code_lookup_passed,
            "source_filter_isolation_passed": self.source_filter_isolation_passed,
            "unknown_unmapped_passed": self.unknown_unmapped_passed,
            "ambiguous_manual_review_passed": self.ambiguous_manual_review_passed,
            "determinism_passed": self.determinism_passed,
            "normalization_passed": self.normalization_passed,
            "read_only": self.read_only,
            "synthetic_only": self.synthetic_only,
            "private_store_accessed": self.private_store_accessed,
            "terminology_data_accessed": self.terminology_data_accessed,
            "data_terminology_accessed": self.data_terminology_accessed,
            "external_api_used": self.external_api_used,
            "clinical_advice_generated": self.clinical_advice_generated,
            "dosing_advice_generated": self.dosing_advice_generated,
            "mkb_write_performed": self.mkb_write_performed,
            "b07_integrated": self.b07_integrated,
            "ddi_status_cleared": self.ddi_status_cleared,
            "hypothesis_promoted": self.hypothesis_promoted,
            "no_code_hallucinated": self.no_code_hallucinated,
        }


class SyntheticReadOnlyTerminologyAdapter:
    """Read-only synthetic adapter for future private-store validation."""

    def __init__(self, store: LocalTerminologyStore | None = None, fixture_metadata: dict | None = None) -> None:
        if store is None:
            store, metadata = build_synthetic_qa_store()
        else:
            metadata = fixture_metadata or {"synthetic_only": True}
        self.store = store
        self.fixture_metadata = dict(metadata)
        self.service = TerminologyLookupService(store)
        self.read_only = True
        self.external_api_used = False

    def lookup(
        self,
        query: str,
        *,
        source_filter: Iterable[str | TerminologySystem] | None = None,
        max_results: int = 10,
    ) -> Term05AdapterResult:
        systems = _parse_system_filter(source_filter)
        result = self.service.lookup(query, systems=systems or None, max_results=max_results)
        status = _adapter_status(result.status)
        reason_codes = _reason_codes_for_status(result.status, result.ambiguous)
        return Term05AdapterResult(
            query=query,
            normalized_query=normalize_query(query),
            source_filter=tuple(system.value for system in systems),
            status=status,
            matches=tuple(_match_from_concept(match) for match in result.matches),
            confidence_label=_confidence_for_status(result.status),
            reason_codes=reason_codes,
            no_code_hallucinated=result.no_code_hallucinated and not (
                result.status == TerminologyLookupStatus.UNMAPPED and bool(result.matches)
            ),
        )

    def lookup_code(
        self,
        code: str,
        *,
        source_filter: Iterable[str | TerminologySystem] | None = None,
        max_results: int = 10,
    ) -> Term05AdapterResult:
        systems = _parse_system_filter(source_filter)
        rows = self._fetch_code(code, systems=systems, max_results=max_results)
        matches = tuple(
            Term05AdapterMatch(
                system=row["system"],
                code=row["code"],
                display_normalized=normalize_query(row["display"]),
                synthetic=bool(row["synthetic"]),
            )
            for row in rows
        )
        if not matches:
            status = "unmapped"
            confidence = "none"
            reason_codes = ("code_unmapped", "no_code_hallucinated")
        elif len({(match.system, match.code) for match in matches}) > 1:
            status = "ambiguous"
            confidence = "manual_review"
            reason_codes = ("code_ambiguous", "manual_review_required")
        else:
            status = "exact"
            confidence = "high"
            reason_codes = ("code_exact_match", "synthetic_fixture")
        return Term05AdapterResult(
            query=code,
            normalized_query=normalize_query(code),
            source_filter=tuple(system.value for system in systems),
            status=status,
            matches=matches,
            confidence_label=confidence,
            reason_codes=reason_codes,
            no_code_hallucinated=bool(matches) or status == "unmapped",
        )

    def _fetch_code(
        self,
        code: str,
        *,
        systems: list[TerminologySystem],
        max_results: int,
    ) -> list[dict]:
        params: list[str | int] = [code]
        where = "active = 1 AND code = ?"
        if systems:
            placeholders = ",".join("?" * len(systems))
            where += f" AND system IN ({placeholders})"
            params.extend(system.value for system in systems)
        params.append(int(max_results))
        with self.store._conn() as con:
            rows = con.execute(
                "SELECT system, code, display, synthetic FROM terminology_concepts "
                f"WHERE {where} ORDER BY system, code, display_norm LIMIT ?",
                params,
            ).fetchall()
        return [
            {
                "system": str(row["system"]),
                "code": str(row["code"]),
                "display": str(row["display"]),
                "synthetic": bool(row["synthetic"]),
            }
            for row in rows
        ]


def build_synthetic_read_only_adapter() -> SyntheticReadOnlyTerminologyAdapter:
    return SyntheticReadOnlyTerminologyAdapter()


def run_term05_synthetic_adapter_validation() -> Term05AdapterValidationResult:
    adapter = build_synthetic_read_only_adapter()
    cases: list[tuple[str, bool, Term05AdapterResult]] = []

    rx = adapter.lookup("aspirin", source_filter=["rxnorm"])
    cases.append(("exact_rxnorm_lookup", rx.status == "exact" and len(rx.matches) == 1, rx))
    loinc = adapter.lookup("glucose synthetic lab", source_filter=["loinc"])
    cases.append(("exact_loinc_lookup", loinc.status == "exact" and len(loinc.matches) == 1, loinc))
    code = adapter.lookup_code("LOINC001", source_filter=["loinc"])
    cases.append(("code_lookup", code.status == "exact" and len(code.matches) == 1, code))
    isolated = adapter.lookup("glucose synthetic lab", source_filter=["rxnorm"])
    cases.append(("source_filter_isolation", isolated.status == "unmapped" and not isolated.matches, isolated))
    unknown = adapter.lookup("term05 unknown should stay unmapped")
    cases.append(("unknown_unmapped", unknown.status == "unmapped" and not unknown.matches and unknown.no_code_hallucinated, unknown))
    ambiguous = adapter.lookup("aspirin")
    cases.append(("ambiguous_manual_review", ambiguous.status == "ambiguous" and "manual_review_required" in ambiguous.reason_codes, ambiguous))
    det_a = adapter.lookup("aspirin", source_filter=["rxnorm"])
    det_b = adapter.lookup("aspirin", source_filter=["rxnorm"])
    cases.append(("determinism", det_a.safe_public_summary() == det_b.safe_public_summary(), det_b))
    norm = adapter.lookup("  ASPIRIN  ", source_filter=["rxnorm"])
    cases.append(("normalization", norm.normalized_query == "aspirin" and norm.status == "exact", norm))

    case_summaries = tuple(
        {
            "case_id": case_id,
            "passed": passed,
            "status": result.status,
            "matches_count": len(result.matches),
            "reason_codes": list(result.reason_codes),
        }
        for case_id, passed, result in cases
    )
    flags = {
        case_id: passed for case_id, passed, _ in cases
    }
    all_passed = all(flags.values())
    return Term05AdapterValidationResult(
        conclusion="cka_term05_synthetic_adapter_ready" if all_passed else "cka_term05_synthetic_adapter_blocked",
        cases_total=len(cases),
        cases_passed=sum(1 for passed in flags.values() if passed),
        cases_failed=sum(1 for passed in flags.values() if not passed),
        case_results=case_summaries,
        fixture_metadata=adapter.fixture_metadata,
        exact_rxnorm_passed=flags["exact_rxnorm_lookup"],
        exact_loinc_passed=flags["exact_loinc_lookup"],
        code_lookup_passed=flags["code_lookup"],
        source_filter_isolation_passed=flags["source_filter_isolation"],
        unknown_unmapped_passed=flags["unknown_unmapped"],
        ambiguous_manual_review_passed=flags["ambiguous_manual_review"],
        determinism_passed=flags["determinism"],
        normalization_passed=flags["normalization"],
    )


def _parse_system_filter(source_filter: Iterable[str | TerminologySystem] | None) -> list[TerminologySystem]:
    systems: list[TerminologySystem] = []
    for item in source_filter or []:
        if isinstance(item, TerminologySystem):
            systems.append(item)
        else:
            systems.append(TerminologySystem(str(item)))
    return systems


def _adapter_status(status: TerminologyLookupStatus) -> str:
    if status in (TerminologyLookupStatus.EXACT, TerminologyLookupStatus.SYNONYM):
        return "exact"
    if status == TerminologyLookupStatus.AMBIGUOUS:
        return "ambiguous"
    return "unmapped"


def _confidence_for_status(status: TerminologyLookupStatus) -> str:
    if status in (TerminologyLookupStatus.EXACT, TerminologyLookupStatus.SYNONYM):
        return "high"
    if status == TerminologyLookupStatus.AMBIGUOUS:
        return "manual_review"
    return "none"


def _reason_codes_for_status(status: TerminologyLookupStatus, ambiguous: bool) -> tuple[str, ...]:
    if status == TerminologyLookupStatus.AMBIGUOUS or ambiguous:
        return ("ambiguous_lookup", "manual_review_required")
    if status in (TerminologyLookupStatus.EXACT, TerminologyLookupStatus.SYNONYM):
        return ("exact_lookup_match", "synthetic_fixture")
    return ("unmapped_lookup", "no_code_hallucinated")


def _match_from_concept(concept) -> Term05AdapterMatch:
    return Term05AdapterMatch(
        system=concept.system.value,
        code=concept.code,
        display_normalized=normalize_query(concept.display),
        synthetic=bool(concept.synthetic),
    )
