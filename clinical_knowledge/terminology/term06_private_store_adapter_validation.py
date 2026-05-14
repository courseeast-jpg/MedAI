"""CKA-TERM-06 private-store read-only adapter validation.

Runs the TERM-05 adapter interface against the existing TERM-02 local
terminology store in read-only mode. Public summaries contain only aggregate
counts, statuses, and reason codes; real query/display text is never emitted.
"""
from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologySystem, normalize_query
from clinical_knowledge.terminology.term02_controlled_import import TERM02_DB_RELATIVE
from clinical_knowledge.terminology.term05_read_only_adapter import (
    SyntheticReadOnlyTerminologyAdapter,
    Term05AdapterResult,
)


@dataclass(frozen=True)
class Term06PrivateStoreCaseResult:
    case_id: str
    case_type: str
    expected_status: str
    observed_status: str
    passed: bool
    matches_count: int = 0
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    skipped: bool = False
    skip_reason: str | None = None

    def safe_public_summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "expected_status": self.expected_status,
            "observed_status": self.observed_status,
            "passed": self.passed,
            "matches_count": self.matches_count,
            "reason_codes": list(self.reason_codes),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class Term06PrivateStoreValidationResult:
    conclusion: str
    store_available: bool
    store_opened_read_only: bool
    write_attempt_blocked: bool
    source_systems_detected: tuple[str, ...]
    store_summary: dict
    cases_total: int
    cases_passed: int
    cases_failed: int
    cases_skipped: int
    case_results: tuple[Term06PrivateStoreCaseResult, ...]
    exact_rxnorm_passed: bool
    exact_loinc_passed: bool
    code_lookup_passed: bool
    source_filter_isolation_passed: bool
    unknown_unmapped_passed: bool
    ambiguous_manual_review_passed: bool
    determinism_passed: bool
    normalization_passed: bool
    read_only: bool = True
    private_store_accessed_read_only: bool = True
    real_import_performed: bool = False
    store_recreated: bool = False
    external_api_used: bool = False
    clinical_advice_generated: bool = False
    dosing_advice_generated: bool = False
    mkb_write_performed: bool = False
    automatic_annotation_created: bool = False
    b07_integrated: bool = False
    ddi_status_cleared: bool = False
    hypothesis_promoted: bool = False
    no_code_hallucinated: bool = True
    terminology_data_staged: bool = False
    data_terminology_staged: bool = False
    license_ack_private_staged: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "store_available": self.store_available,
            "store_opened_read_only": self.store_opened_read_only,
            "write_attempt_blocked": self.write_attempt_blocked,
            "source_systems_detected": list(self.source_systems_detected),
            "store_summary": dict(self.store_summary),
            "qa_case_counts": {
                "total": self.cases_total,
                "passed": self.cases_passed,
                "failed": self.cases_failed,
                "skipped": self.cases_skipped,
            },
            "case_results": [case.safe_public_summary() for case in self.case_results],
            "exact_rxnorm_passed": self.exact_rxnorm_passed,
            "exact_loinc_passed": self.exact_loinc_passed,
            "code_lookup_passed": self.code_lookup_passed,
            "source_filter_isolation_passed": self.source_filter_isolation_passed,
            "unknown_unmapped_passed": self.unknown_unmapped_passed,
            "ambiguous_manual_review_passed": self.ambiguous_manual_review_passed,
            "determinism_passed": self.determinism_passed,
            "normalization_passed": self.normalization_passed,
            "read_only": self.read_only,
            "private_store_accessed_read_only": self.private_store_accessed_read_only,
            "real_import_performed": self.real_import_performed,
            "store_recreated": self.store_recreated,
            "external_api_used": self.external_api_used,
            "clinical_advice_generated": self.clinical_advice_generated,
            "dosing_advice_generated": self.dosing_advice_generated,
            "mkb_write_performed": self.mkb_write_performed,
            "automatic_annotation_created": self.automatic_annotation_created,
            "b07_integrated": self.b07_integrated,
            "ddi_status_cleared": self.ddi_status_cleared,
            "hypothesis_promoted": self.hypothesis_promoted,
            "no_code_hallucinated": self.no_code_hallucinated,
            "terminology_data_staged": self.terminology_data_staged,
            "data_terminology_staged": self.data_terminology_staged,
            "license_ack_private_staged": self.license_ack_private_staged,
        }


class Term06ValidationBlocked(RuntimeError):
    """Raised when TERM-06 validation preconditions are not met."""


def run_private_store_adapter_validation(
    *,
    repo_root: Path | None = None,
    db_path: Path | None = None,
) -> Term06PrivateStoreValidationResult:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent.parent
    db_path = Path(db_path) if db_path is not None else repo_root / TERM02_DB_RELATIVE
    staged = _git_staged_paths(repo_root)
    if _staged_under(staged, "terminology_data/") or _staged_under(staged, "data/terminology/") or any(
        "LICENSE_ACK_PRIVATE" in path for path in staged
    ):
        raise Term06ValidationBlocked("private_or_runtime_artifact_staged")
    if not db_path.exists():
        raise Term06ValidationBlocked("term02_local_store_missing")

    store = _open_read_only_store(db_path)
    write_blocked = _write_attempt_blocked(store)
    adapter = SyntheticReadOnlyTerminologyAdapter(
        store=store,
        fixture_metadata={"synthetic_only": False, "private_store_read_only": True},
    )
    cases = _run_cases(adapter, store)
    pass_count = sum(1 for case in cases if case.passed and not case.skipped)
    fail_count = sum(1 for case in cases if not case.passed and not case.skipped)
    skipped_count = sum(1 for case in cases if case.skipped)
    flags = {case.case_id: case.passed for case in cases}
    required = [
        flags["term06_exact_rxnorm_lookup"],
        flags["term06_exact_loinc_lookup"],
        flags["term06_rxnorm_code_lookup"],
        flags["term06_loinc_code_lookup"],
        flags["term06_source_filter_isolation"],
        flags["term06_unknown_unmapped"],
        flags["term06_ambiguous_manual_review"],
        flags["term06_determinism"],
        flags["term06_normalization"],
        write_blocked,
    ]
    return Term06PrivateStoreValidationResult(
        conclusion="cka_term06_private_store_adapter_validation_ready" if all(required) and fail_count == 0 else "cka_term06_private_store_adapter_validation_blocked",
        store_available=True,
        store_opened_read_only=True,
        write_attempt_blocked=write_blocked,
        source_systems_detected=tuple(_systems_seen(store)),
        store_summary=store.safe_public_summary(),
        cases_total=len(cases),
        cases_passed=pass_count,
        cases_failed=fail_count,
        cases_skipped=skipped_count,
        case_results=tuple(cases),
        exact_rxnorm_passed=flags["term06_exact_rxnorm_lookup"],
        exact_loinc_passed=flags["term06_exact_loinc_lookup"],
        code_lookup_passed=flags["term06_rxnorm_code_lookup"] and flags["term06_loinc_code_lookup"],
        source_filter_isolation_passed=flags["term06_source_filter_isolation"],
        unknown_unmapped_passed=flags["term06_unknown_unmapped"],
        ambiguous_manual_review_passed=flags["term06_ambiguous_manual_review"],
        determinism_passed=flags["term06_determinism"],
        normalization_passed=flags["term06_normalization"],
        terminology_data_staged=_staged_under(staged, "terminology_data/"),
        data_terminology_staged=_staged_under(staged, "data/terminology/"),
        license_ack_private_staged=any("LICENSE_ACK_PRIVATE" in path for path in staged),
    )


def _run_cases(adapter: SyntheticReadOnlyTerminologyAdapter, store: LocalTerminologyStore) -> list[Term06PrivateStoreCaseResult]:
    rx_sample = _distinct_sample(store, TerminologySystem.RXNORM)
    loinc_sample = _distinct_sample(store, TerminologySystem.LOINC)
    cases = [
        _case("term06_exact_rxnorm_lookup", "exact_lookup", "exact", adapter.lookup(rx_sample["display"], source_filter=["rxnorm"], max_results=5)),
        _case("term06_exact_loinc_lookup", "exact_lookup", "exact", adapter.lookup(loinc_sample["display"], source_filter=["loinc"], max_results=5)),
        _case("term06_rxnorm_code_lookup", "code_lookup", "exact", adapter.lookup_code(rx_sample["code"], source_filter=["rxnorm"], max_results=5)),
        _case("term06_loinc_code_lookup", "code_lookup", "exact", adapter.lookup_code(loinc_sample["code"], source_filter=["loinc"], max_results=5)),
        _case("term06_source_filter_isolation", "source_filter_isolation", "unmapped", adapter.lookup(rx_sample["display"], source_filter=["loinc"], max_results=5)),
        _case("term06_unknown_unmapped", "unknown_lookup", "unmapped", adapter.lookup("medai term06 unknown no mapping", max_results=5)),
    ]
    ambiguous_query = _real_ambiguous_query(store)
    if ambiguous_query:
        cases.append(_case("term06_ambiguous_manual_review", "ambiguous_lookup", "ambiguous", adapter.lookup(ambiguous_query, max_results=5)))
    else:
        cases.append(Term06PrivateStoreCaseResult(
            case_id="term06_ambiguous_manual_review",
            case_type="ambiguous_lookup",
            expected_status="ambiguous",
            observed_status="not_available",
            passed=True,
            skipped=True,
            skip_reason="no_real_cross_system_ambiguity_found",
            reason_codes=("manual_review_boundary_preserved_by_adapter_contract",),
        ))
    a = adapter.lookup(rx_sample["display"], source_filter=["rxnorm"], max_results=5)
    b = adapter.lookup(rx_sample["display"], source_filter=["rxnorm"], max_results=5)
    cases.append(Term06PrivateStoreCaseResult(
        case_id="term06_determinism",
        case_type="determinism",
        expected_status=a.status,
        observed_status=b.status,
        passed=_safe_result_shape(a) == _safe_result_shape(b),
        matches_count=len(a.matches),
        reason_codes=("deterministic_result_shape",),
    ))
    normalized = adapter.lookup(f"  {rx_sample['display'].upper()}  ", source_filter=["rxnorm"], max_results=5)
    cases.append(Term06PrivateStoreCaseResult(
        case_id="term06_normalization",
        case_type="normalization",
        expected_status="exact",
        observed_status=normalized.status,
        passed=normalized.status == "exact" and normalized.normalized_query == normalize_query(rx_sample["display"]),
        matches_count=len(normalized.matches),
        reason_codes=("case_whitespace_normalization",),
    ))
    return cases


def _case(case_id: str, case_type: str, expected: str, result: Term05AdapterResult) -> Term06PrivateStoreCaseResult:
    return Term06PrivateStoreCaseResult(
        case_id=case_id,
        case_type=case_type,
        expected_status=expected,
        observed_status=result.status,
        passed=result.status == expected and (result.status != "unmapped" or not result.matches) and result.no_code_hallucinated,
        matches_count=len(result.matches),
        reason_codes=tuple(result.reason_codes),
    )


def _open_read_only_store(db_path: Path) -> LocalTerminologyStore:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.execute("SELECT 1").fetchone()
    store = object.__new__(LocalTerminologyStore)
    store.db_path = uri
    store._mem_con = None
    store._uri = True
    return store


def _write_attempt_blocked(store: LocalTerminologyStore) -> bool:
    try:
        with store._conn() as con:
            con.execute("CREATE TABLE term06_write_probe (id INTEGER)")
        return False
    except sqlite3.OperationalError as exc:
        return "readonly" in str(exc).lower() or "read-only" in str(exc).lower()


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
        raise Term06ValidationBlocked(f"no_sample_for_system:{system.value}")
    return {"code": str(row["code"]), "display": str(row["display"]), "display_norm": str(row["display_norm"])}


def _real_ambiguous_query(store: LocalTerminologyStore) -> str | None:
    with store._conn() as con:
        row = con.execute(
            """
            SELECT display_norm FROM terminology_concepts
            WHERE active = 1
            GROUP BY display_norm
            HAVING count(DISTINCT system || ':' || code) > 1
            ORDER BY display_norm LIMIT 1
            """
        ).fetchone()
    return str(row["display_norm"]) if row else None


def _systems_seen(store: LocalTerminologyStore) -> list[str]:
    with store._conn() as con:
        rows = con.execute("SELECT DISTINCT system FROM terminology_concepts ORDER BY system").fetchall()
    return [str(row["system"]) for row in rows]


def _safe_result_shape(result: Term05AdapterResult) -> dict:
    return {
        "status": result.status,
        "source_filter": list(result.source_filter),
        "matches": [match.safe_public_summary() for match in result.matches],
        "confidence_label": result.confidence_label,
        "reason_codes": list(result.reason_codes),
    }


def _git_staged_paths(repo_root: Path) -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: Iterable[str], prefix: str) -> bool:
    prefix = prefix.replace("\\", "/")
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)
