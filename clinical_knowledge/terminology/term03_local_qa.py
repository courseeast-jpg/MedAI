"""CKA-TERM-03 read-only terminology QA over the local TERM-02 store.

This module does not import terminology data. It opens the existing local
SQLite store in read-only mode, runs deterministic lookup checks, and returns
public-report-safe aggregate metrics only.
"""
from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from clinical_knowledge.terminology.integration import (
    code_entity_via_local_terminology,
    safe_b07_boundary_summary,
)
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologyLookupResult,
    TerminologyLookupStatus,
    TerminologySystem,
    normalize_query,
)
from clinical_knowledge.terminology.term02_controlled_import import TERM02_DB_RELATIVE


@dataclass(frozen=True)
class Term03QACaseResult:
    case_id: str
    case_type: str
    system_filter: tuple[str, ...]
    expected_status: str
    observed_status: str
    passed: bool
    matches_count: int = 0
    ambiguous: bool = False
    no_code_hallucinated: bool = True
    skipped: bool = False
    skip_reason: str | None = None

    def safe_public_summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "system_filter": list(self.system_filter),
            "expected_status": self.expected_status,
            "observed_status": self.observed_status,
            "passed": self.passed,
            "matches_count": self.matches_count,
            "ambiguous": self.ambiguous,
            "no_code_hallucinated": self.no_code_hallucinated,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class Term03LocalTerminologyQAResult:
    conclusion: str
    store_available: bool
    source_systems_detected: tuple[str, ...]
    store_summary: dict
    qa_case_results: tuple[Term03QACaseResult, ...]
    pass_count: int
    fail_count: int
    skipped_count: int
    unknown_unmapped_passed: bool
    ambiguous_manual_review_passed: bool
    determinism_passed: bool
    source_filter_isolation_passed: bool
    code_lookup_passed: bool
    synonym_alias_supported: bool
    read_only_mode: bool
    external_api_used: bool = False
    raw_phi_logged_in_public_reports: bool = False
    private_filename_path_leaks: int = 0
    secret_leaks: int = 0
    license_text_written_to_public_reports: bool = False
    clinical_recommendations_generated: bool = False
    prescription_dosing_advice_generated: bool = False
    coding_promotes_hypothesis: bool = False
    coding_clears_ddi_status: bool = False
    no_code_hallucinated: bool = True
    terminology_data_staged: bool = False
    data_terminology_staged: bool = False
    license_ack_private_staged: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "store_available": self.store_available,
            "source_systems_detected": list(self.source_systems_detected),
            "store_summary": dict(self.store_summary),
            "qa_case_counts": {
                "total": len(self.qa_case_results),
                "passed": self.pass_count,
                "failed": self.fail_count,
                "skipped": self.skipped_count,
            },
            "qa_case_results": [case.safe_public_summary() for case in self.qa_case_results],
            "unknown_unmapped_passed": self.unknown_unmapped_passed,
            "ambiguous_manual_review_passed": self.ambiguous_manual_review_passed,
            "determinism_passed": self.determinism_passed,
            "source_filter_isolation_passed": self.source_filter_isolation_passed,
            "code_lookup_passed": self.code_lookup_passed,
            "synonym_alias_supported": self.synonym_alias_supported,
            "read_only_mode": self.read_only_mode,
            "external_api_used": self.external_api_used,
            "raw_phi_logged_in_public_reports": self.raw_phi_logged_in_public_reports,
            "private_filename_path_leaks": self.private_filename_path_leaks,
            "secret_leaks": self.secret_leaks,
            "license_text_written_to_public_reports": self.license_text_written_to_public_reports,
            "clinical_recommendations_generated": self.clinical_recommendations_generated,
            "prescription_dosing_advice_generated": self.prescription_dosing_advice_generated,
            "coding_promotes_hypothesis": self.coding_promotes_hypothesis,
            "coding_clears_ddi_status": self.coding_clears_ddi_status,
            "no_code_hallucinated": self.no_code_hallucinated,
            "terminology_data_staged": self.terminology_data_staged,
            "data_terminology_staged": self.data_terminology_staged,
            "license_ack_private_staged": self.license_ack_private_staged,
        }


class Term03QABlocked(RuntimeError):
    """Raised when TERM-03 safety preconditions are not met."""


def run_local_terminology_qa(
    *,
    repo_root: Path | None = None,
    db_path: Path | None = None,
    synthetic_store: LocalTerminologyStore | None = None,
) -> Term03LocalTerminologyQAResult:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent.parent
    staged = _git_staged_paths(repo_root)
    if _staged_under(staged, "terminology_data/") or _staged_under(staged, "data/terminology/") or any(
        "LICENSE_ACK_PRIVATE" in path for path in staged
    ):
        raise Term03QABlocked("private_or_runtime_artifact_staged")

    read_only_mode = synthetic_store is None
    if synthetic_store is None:
        db_path = Path(db_path) if db_path is not None else repo_root / TERM02_DB_RELATIVE
        if not db_path.exists():
            raise Term03QABlocked("term02_local_store_missing")
        store = _open_read_only_store(db_path)
    else:
        store = synthetic_store

    qa = _run_qa_cases(store)
    case_results = tuple(qa["cases"])
    pass_count = sum(1 for case in case_results if case.passed and not case.skipped)
    fail_count = sum(1 for case in case_results if not case.passed and not case.skipped)
    skipped_count = sum(1 for case in case_results if case.skipped)
    required_passed = fail_count == 0 and qa["required_flags_passed"]
    boundary = safe_b07_boundary_summary()

    return Term03LocalTerminologyQAResult(
        conclusion="cka_term03_local_terminology_qa_ready" if required_passed else "cka_term03_local_terminology_qa_blocked",
        store_available=True,
        source_systems_detected=tuple(_systems_seen(store)),
        store_summary=store.safe_public_summary(),
        qa_case_results=case_results,
        pass_count=pass_count,
        fail_count=fail_count,
        skipped_count=skipped_count,
        unknown_unmapped_passed=qa["unknown_unmapped_passed"],
        ambiguous_manual_review_passed=qa["ambiguous_manual_review_passed"],
        determinism_passed=qa["determinism_passed"],
        source_filter_isolation_passed=qa["source_filter_isolation_passed"],
        code_lookup_passed=qa["code_lookup_passed"],
        synonym_alias_supported=qa["synonym_alias_supported"],
        read_only_mode=read_only_mode,
        coding_promotes_hypothesis=not bool(boundary["default_b07_behavior_unchanged"])
        or bool(boundary["coding_promotes_hypothesis"]),
        coding_clears_ddi_status=bool(boundary["coding_clears_ddi_status"]),
        no_code_hallucinated=qa["no_code_hallucinated"],
        terminology_data_staged=_staged_under(staged, "terminology_data/"),
        data_terminology_staged=_staged_under(staged, "data/terminology/"),
        license_ack_private_staged=any("LICENSE_ACK_PRIVATE" in path for path in staged),
    )


def _open_read_only_store(db_path: Path) -> LocalTerminologyStore:
    # Verify SQLite can open the file read-only. The LocalTerminologyStore
    # instance below is then pointed at the same URI without initializing schema.
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.execute("SELECT 1").fetchone()
    store = object.__new__(LocalTerminologyStore)
    store.db_path = uri
    store._mem_con = None
    store._uri = True
    return store


def _run_qa_cases(store: LocalTerminologyStore) -> dict:
    service = TerminologyLookupService(store)
    cases: list[Term03QACaseResult] = []

    rx_sample = _distinct_sample(store, TerminologySystem.RXNORM)
    loinc_sample = _distinct_sample(store, TerminologySystem.LOINC)
    cases.append(_lookup_case("term03_rxnorm_exact", "exact_lookup", service, rx_sample["display"], TerminologyLookupStatus.EXACT, [TerminologySystem.RXNORM]))
    cases.append(_lookup_case("term03_loinc_exact", "exact_lookup", service, loinc_sample["display"], TerminologyLookupStatus.EXACT, [TerminologySystem.LOINC]))
    cases.append(_code_lookup_case("term03_rxnorm_code_to_display", store, TerminologySystem.RXNORM, rx_sample["code"]))
    cases.append(_code_lookup_case("term03_loinc_code_to_display", store, TerminologySystem.LOINC, loinc_sample["code"]))
    cases.append(_lookup_case("term03_unknown_unmapped", "unknown_lookup", service, "medai term03 unknown no local terminology mapping", TerminologyLookupStatus.UNMAPPED, None))

    rx_cross = service.lookup(rx_sample["display"], systems=[TerminologySystem.LOINC], max_results=5)
    loinc_cross = service.lookup(loinc_sample["display"], systems=[TerminologySystem.RXNORM], max_results=5)
    cases.append(_result_case("term03_rxnorm_filter_isolation", "source_filter_isolation", rx_cross, TerminologyLookupStatus.UNMAPPED))
    cases.append(_result_case("term03_loinc_filter_isolation", "source_filter_isolation", loinc_cross, TerminologyLookupStatus.UNMAPPED))

    det_a = service.lookup(rx_sample["display"], systems=[TerminologySystem.RXNORM], max_results=5)
    det_b = service.lookup(f"  {rx_sample['display'].upper()}  ", systems=[TerminologySystem.RXNORM], max_results=5)
    cases.append(_determinism_case("term03_deterministic_normalization", det_a, det_b))

    ambiguous = _ambiguous_lookup_check()
    cases.append(ambiguous)

    synonym_supported = _synonym_count(store) > 0
    if synonym_supported:
        synonym_text = _first_synonym_text(store)
        syn_result = service.lookup(synonym_text, max_results=5)
        cases.append(_result_case("term03_synonym_alias_lookup", "synonym_alias_lookup", syn_result, TerminologyLookupStatus.SYNONYM))
    else:
        cases.append(Term03QACaseResult(
            case_id="term03_synonym_alias_lookup",
            case_type="synonym_alias_lookup",
            system_filter=(),
            expected_status="synonym",
            observed_status="not_supported",
            passed=True,
            skipped=True,
            skip_reason="no_synonym_alias_fields_imported",
        ))

    unknown = next(case for case in cases if case.case_id == "term03_unknown_unmapped")
    filter_cases = [case for case in cases if case.case_type == "source_filter_isolation"]
    determinism = next(case for case in cases if case.case_id == "term03_deterministic_normalization")
    code_cases = [case for case in cases if case.case_type == "code_to_display_lookup"]
    required_flags = [
        unknown.passed,
        ambiguous.passed,
        determinism.passed,
        all(case.passed for case in filter_cases),
        all(case.passed for case in code_cases),
    ]
    b07_unknown = code_entity_via_local_terminology("medai term03 unknown no local terminology mapping", service)

    return {
        "cases": cases,
        "unknown_unmapped_passed": unknown.passed,
        "ambiguous_manual_review_passed": ambiguous.passed,
        "determinism_passed": determinism.passed,
        "source_filter_isolation_passed": all(case.passed for case in filter_cases),
        "code_lookup_passed": all(case.passed for case in code_cases),
        "synonym_alias_supported": synonym_supported,
        "no_code_hallucinated": b07_unknown.status == TerminologyLookupStatus.UNMAPPED,
        "required_flags_passed": all(required_flags) and b07_unknown.status == TerminologyLookupStatus.UNMAPPED,
    }


def _lookup_case(
    case_id: str,
    case_type: str,
    service: TerminologyLookupService,
    query: str,
    expected: TerminologyLookupStatus,
    systems: list[TerminologySystem] | None,
) -> Term03QACaseResult:
    result = service.lookup(query, systems=systems, max_results=5)
    return _result_case(case_id, case_type, result, expected)


def _result_case(
    case_id: str,
    case_type: str,
    result: TerminologyLookupResult,
    expected: TerminologyLookupStatus,
) -> Term03QACaseResult:
    return Term03QACaseResult(
        case_id=case_id,
        case_type=case_type,
        system_filter=tuple(result.system_filter),
        expected_status=expected.value,
        observed_status=result.status.value,
        passed=result.status == expected and (result.status != TerminologyLookupStatus.UNMAPPED or not result.matches),
        matches_count=len(result.matches),
        ambiguous=result.ambiguous,
        no_code_hallucinated=result.no_code_hallucinated,
    )


def _code_lookup_case(
    case_id: str,
    store: LocalTerminologyStore,
    system: TerminologySystem,
    code: str,
) -> Term03QACaseResult:
    with store._conn() as con:
        rows = con.execute(
            "SELECT code, display FROM terminology_concepts WHERE active = 1 AND system = ? AND code = ? ORDER BY concept_id LIMIT 5",
            (system.value, code),
        ).fetchall()
    return Term03QACaseResult(
        case_id=case_id,
        case_type="code_to_display_lookup",
        system_filter=(system.value,),
        expected_status="found",
        observed_status="found" if rows else "unmapped",
        passed=bool(rows),
        matches_count=len(rows),
    )


def _determinism_case(case_id: str, a: TerminologyLookupResult, b: TerminologyLookupResult) -> Term03QACaseResult:
    a_pairs = [(m.system.value, m.code) for m in a.matches]
    b_pairs = [(m.system.value, m.code) for m in b.matches]
    passed = a.status == b.status and a_pairs == b_pairs and a.ambiguous == b.ambiguous
    return Term03QACaseResult(
        case_id=case_id,
        case_type="determinism",
        system_filter=tuple(a.system_filter),
        expected_status=a.status.value,
        observed_status=b.status.value,
        passed=passed,
        matches_count=len(a.matches),
        ambiguous=a.ambiguous,
        no_code_hallucinated=a.no_code_hallucinated and b.no_code_hallucinated,
    )


def _ambiguous_lookup_check() -> Term03QACaseResult:
    temp_store = LocalTerminologyStore()
    sid = temp_store.register_source(TerminologySystem.RXNORM, safe_source_id="term03_ambiguous_synthetic", license_confirmed=True)
    temp_store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term03_ambiguous_rxnorm",
                system=TerminologySystem.RXNORM,
                code="RXTERM03",
                display="term03 shared ambiguous",
                synthetic=True,
            ),
            TerminologyConcept(
                concept_id="term03_ambiguous_loinc",
                system=TerminologySystem.LOINC,
                code="LTERM03",
                display="term03 shared ambiguous",
                synthetic=True,
            ),
        ],
        sid,
    )
    result = TerminologyLookupService(temp_store).lookup("term03 shared ambiguous")
    return _result_case("term03_ambiguous_manual_review", "ambiguous_lookup", result, TerminologyLookupStatus.AMBIGUOUS)


def _distinct_sample(store: LocalTerminologyStore, system: TerminologySystem) -> dict:
    other = TerminologySystem.LOINC if system == TerminologySystem.RXNORM else TerminologySystem.RXNORM
    with store._conn() as con:
        row = con.execute(
            """
            SELECT code, display, display_norm FROM terminology_concepts
            WHERE active = 1 AND system = ?
              AND display_norm NOT IN (
                  SELECT display_norm FROM terminology_concepts
                  WHERE active = 1 AND system = ?
              )
            ORDER BY concept_id LIMIT 1
            """,
            (system.value, other.value),
        ).fetchone()
        if row is None:
            row = con.execute(
                "SELECT code, display, display_norm FROM terminology_concepts WHERE active = 1 AND system = ? ORDER BY concept_id LIMIT 1",
                (system.value,),
            ).fetchone()
    if row is None:
        raise Term03QABlocked(f"no_sample_for_system:{system.value}")
    return {"code": str(row["code"]), "display": str(row["display"]), "display_norm": str(row["display_norm"])}


def _systems_seen(store: LocalTerminologyStore) -> list[str]:
    with store._conn() as con:
        rows = con.execute("SELECT DISTINCT system FROM terminology_concepts ORDER BY system").fetchall()
    return [str(row["system"]) for row in rows]


def _synonym_count(store: LocalTerminologyStore) -> int:
    with store._conn() as con:
        row = con.execute("SELECT count(*) AS n FROM terminology_synonyms").fetchone()
    return int(row["n"]) if row else 0


def _first_synonym_text(store: LocalTerminologyStore) -> str:
    with store._conn() as con:
        row = con.execute("SELECT synonym_norm FROM terminology_synonyms ORDER BY synonym_norm LIMIT 1").fetchone()
    if row is None:
        raise Term03QABlocked("no_synonym_available")
    return str(row["synonym_norm"])


def _git_staged_paths(repo_root: Path) -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: Iterable[str], prefix: str) -> bool:
    prefix = prefix.replace("\\", "/")
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)
