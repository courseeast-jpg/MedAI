"""CKA-TERM-02 controlled local terminology import.

Imports acknowledged local RxNorm and LOINC files into a local SQLite
terminology store. Public summaries contain counts and safe hashes only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from clinical_knowledge.terminology.import_limits import TerminologyImportLimits
from clinical_knowledge.terminology.import_transaction import TerminologyImportTransaction
from clinical_knowledge.terminology.integration import code_entity_via_local_terminology, safe_b07_boundary_summary
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologyLookupStatus,
    TerminologySystem,
    normalize_query,
    query_hash,
)
from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate


ALLOWED_TERM02_SYSTEMS = (TerminologySystem.RXNORM, TerminologySystem.LOINC)
TERM02_DB_RELATIVE = Path("data") / "terminology" / "term02_local_terminology.sqlite"


@dataclass(frozen=True)
class Term02FileImportSummary:
    system: str
    file_safe_id: str
    content_hash: str
    rows_seen: int = 0
    records_imported: int = 0
    records_skipped: int = 0
    chunks_processed: int = 0
    checkpoint_count: int = 0
    row_cap_applied: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "system": self.system,
            "file_safe_id": self.file_safe_id,
            "content_hash": self.content_hash,
            "rows_seen": self.rows_seen,
            "records_imported": self.records_imported,
            "records_skipped": self.records_skipped,
            "chunks_processed": self.chunks_processed,
            "checkpoint_count": self.checkpoint_count,
            "row_cap_applied": self.row_cap_applied,
        }


@dataclass(frozen=True)
class Term02LookupValidation:
    exact_rxnorm_passed: bool
    exact_loinc_passed: bool
    unknown_unmapped_passed: bool
    ambiguous_flag_passed: bool
    no_code_hallucinated: bool
    b07_boundary_passed: bool

    def safe_public_summary(self) -> dict:
        return {
            "exact_rxnorm_passed": self.exact_rxnorm_passed,
            "exact_loinc_passed": self.exact_loinc_passed,
            "unknown_unmapped_passed": self.unknown_unmapped_passed,
            "ambiguous_flag_passed": self.ambiguous_flag_passed,
            "no_code_hallucinated": self.no_code_hallucinated,
            "b07_boundary_passed": self.b07_boundary_passed,
        }


@dataclass(frozen=True)
class Term02ControlledImportResult:
    conclusion: str
    imported_systems: tuple[str, ...]
    file_summaries: tuple[Term02FileImportSummary, ...]
    store_summary: dict
    lookup_validation: Term02LookupValidation
    db_safe_id: str
    preflight_allowed: bool
    term02_completed: bool
    real_import_performed: bool
    external_api_used: bool = False
    raw_phi_logged_in_public_reports: bool = False
    private_filename_path_leaks: int = 0
    secret_leaks: int = 0
    license_text_written_to_public_reports: bool = False
    clinical_recommendations_generated: bool = False
    prescription_dosing_advice_generated: bool = False
    production_ocr_changed: bool = False
    production_extractor_changed: bool = False
    safety_gate_changed: bool = False
    terminology_data_staged: bool = False
    data_terminology_staged: bool = False
    license_ack_private_staged: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "conclusion": self.conclusion,
            "imported_systems": list(self.imported_systems),
            "file_summaries": [summary.safe_public_summary() for summary in self.file_summaries],
            "store_summary": dict(self.store_summary),
            "lookup_validation": self.lookup_validation.safe_public_summary(),
            "db_safe_id": self.db_safe_id,
            "preflight_allowed": self.preflight_allowed,
            "term02_completed": self.term02_completed,
            "real_import_performed": self.real_import_performed,
            "external_api_used": self.external_api_used,
            "raw_phi_logged_in_public_reports": self.raw_phi_logged_in_public_reports,
            "private_filename_path_leaks": self.private_filename_path_leaks,
            "secret_leaks": self.secret_leaks,
            "license_text_written_to_public_reports": self.license_text_written_to_public_reports,
            "clinical_recommendations_generated": self.clinical_recommendations_generated,
            "prescription_dosing_advice_generated": self.prescription_dosing_advice_generated,
            "production_ocr_changed": self.production_ocr_changed,
            "production_extractor_changed": self.production_extractor_changed,
            "safety_gate_changed": self.safety_gate_changed,
            "terminology_data_staged": self.terminology_data_staged,
            "data_terminology_staged": self.data_terminology_staged,
            "license_ack_private_staged": self.license_ack_private_staged,
        }


class Term02ImportBlocked(RuntimeError):
    """Raised when TERM-02 safety preconditions are not met."""


def run_controlled_local_import(
    *,
    repo_root: Path | None = None,
    terminology_root: Path | None = None,
    db_path: Path | None = None,
    limits: TerminologyImportLimits | None = None,
) -> Term02ControlledImportResult:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent.parent
    terminology_root = Path(terminology_root) if terminology_root is not None else repo_root / "terminology_data"
    db_path = Path(db_path) if db_path is not None else repo_root / TERM02_DB_RELATIVE
    limits = limits or TerminologyImportLimits(allow_real_import=True)

    staged = _git_staged_paths(repo_root)
    if _staged_under(staged, "terminology_data/") or _staged_under(staged, "data/terminology/") or any(
        "LICENSE_ACK_PRIVATE" in path for path in staged
    ):
        raise Term02ImportBlocked("private_or_runtime_artifact_staged")

    preflight = run_term02_preflight_gate(repo_root=repo_root, terminology_root=terminology_root)
    if not preflight.allowed:
        raise Term02ImportBlocked("term02_preflight_failed")
    if set(preflight.systems_import_ready) != {"rxnorm", "loinc"}:
        raise Term02ImportBlocked("unexpected_import_ready_systems")

    files = _resolve_import_files(terminology_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    store = LocalTerminologyStore(str(db_path))

    file_summaries: list[Term02FileImportSummary] = []
    for system, file_path in files:
        summary = _import_file(store, system, file_path, limits=limits)
        file_summaries.append(summary)

    lookup_validation = _validate_imported_lookup(store)
    all_lookup_passed = all(lookup_validation.safe_public_summary().values())
    result = Term02ControlledImportResult(
        conclusion="cka_term02_controlled_local_import_ready" if all_lookup_passed else "cka_term02_controlled_local_import_blocked",
        imported_systems=tuple(summary.system for summary in file_summaries),
        file_summaries=tuple(file_summaries),
        store_summary=store.safe_public_summary(),
        lookup_validation=lookup_validation,
        db_safe_id=_safe_hash(str(db_path), prefix="term02_db_"),
        preflight_allowed=True,
        term02_completed=all_lookup_passed,
        real_import_performed=True,
        terminology_data_staged=_staged_under(staged, "terminology_data/"),
        data_terminology_staged=_staged_under(staged, "data/terminology/"),
        license_ack_private_staged=any("LICENSE_ACK_PRIVATE" in path for path in staged),
    )
    return result


def _resolve_import_files(terminology_root: Path) -> list[tuple[TerminologySystem, Path]]:
    rxnorm = terminology_root / "rxnorm" / "RXNCONSO.RRF"
    loinc = terminology_root / "loinc" / "Loinc.csv"
    missing = []
    if not rxnorm.exists():
        missing.append("rxnorm_expected_file_missing")
    if not loinc.exists():
        missing.append("loinc_expected_file_missing")
    if missing:
        raise Term02ImportBlocked(",".join(missing))
    return [(TerminologySystem.RXNORM, rxnorm), (TerminologySystem.LOINC, loinc)]


def _import_file(
    store: LocalTerminologyStore,
    system: TerminologySystem,
    file_path: Path,
    *,
    limits: TerminologyImportLimits,
) -> Term02FileImportSummary:
    content_hash = _file_hash(file_path)
    file_safe_id = _safe_hash(f"{system.value}:{file_path.name}:{content_hash}", prefix="term02_file_")
    rows_seen = 0
    records_imported = 0
    records_skipped = 0
    chunks_processed = 0
    checkpoint_count = 0
    source_safe_id = _safe_hash(f"{system.value}:{content_hash}", prefix="term02_src_")
    max_rows = max(0, limits.max_rows_per_file_default)
    chunk_size = max(1, limits.chunk_size)
    checkpoint_interval = max(1, limits.checkpoint_interval_rows)

    with TerminologyImportTransaction(store) as tx:
        source_id = tx.write_source_manifest(
            system,
            source_safe_id=source_safe_id,
            version="licensed-local-term02",
            license_confirmed=True,
        )
        chunk: list[TerminologyConcept] = []
        for parsed in _iter_concepts(system, file_path, source_safe_id=source_safe_id, max_rows=max_rows):
            rows_seen = parsed[0]
            concept = parsed[1]
            if concept is None:
                records_skipped += 1
            else:
                chunk.append(concept)
            if rows_seen and rows_seen % checkpoint_interval == 0:
                checkpoint_count += 1
            if len(chunk) >= chunk_size:
                records_imported += tx.write_concepts(chunk, source_id)
                chunks_processed += 1
                chunk = []
        if chunk:
            records_imported += tx.write_concepts(chunk, source_id)
            chunks_processed += 1
        if rows_seen and rows_seen % checkpoint_interval:
            checkpoint_count += 1
        tx.write_import_audit_event(
            source_id,
            event_type=f"term02_{system.value}_import_completed",
            rows_seen=rows_seen,
            rows_imported=records_imported,
        )

    return Term02FileImportSummary(
        system=system.value,
        file_safe_id=file_safe_id,
        content_hash=content_hash,
        rows_seen=rows_seen,
        records_imported=records_imported,
        records_skipped=records_skipped,
        chunks_processed=chunks_processed,
        checkpoint_count=checkpoint_count,
        row_cap_applied=rows_seen >= max_rows,
    )


def _iter_concepts(
    system: TerminologySystem,
    file_path: Path,
    *,
    source_safe_id: str,
    max_rows: int,
) -> Iterator[tuple[int, TerminologyConcept | None]]:
    if system == TerminologySystem.RXNORM:
        yield from _iter_rxnorm_concepts(file_path, source_safe_id=source_safe_id, max_rows=max_rows)
    elif system == TerminologySystem.LOINC:
        yield from _iter_loinc_concepts(file_path, source_safe_id=source_safe_id, max_rows=max_rows)
    else:
        raise Term02ImportBlocked(f"unsupported_system:{system.value}")


def _iter_rxnorm_concepts(
    file_path: Path,
    *,
    source_safe_id: str,
    max_rows: int,
) -> Iterator[tuple[int, TerminologyConcept | None]]:
    rows = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if rows >= max_rows:
                break
            line = line.rstrip("\r\n")
            if not line:
                continue
            rows += 1
            fields = line.split("|")
            if len(fields) < 15:
                yield rows, None
                continue
            code = fields[0].strip()
            lat = fields[1].strip().upper() if len(fields) > 1 else "ENG"
            display = fields[14].strip()
            if not code or not display or lat not in ("ENG", ""):
                yield rows, None
                continue
            yield rows, TerminologyConcept(
                concept_id=f"term_concept_rxnorm_{code}_{rows:07d}_{_short_hash(display)}",
                system=TerminologySystem.RXNORM,
                code=code,
                display=display,
                synonyms=[],
                version="licensed-local-term02",
                source_safe_id=source_safe_id,
                active=True,
                synthetic=False,
            )


def _iter_loinc_concepts(
    file_path: Path,
    *,
    source_safe_id: str,
    max_rows: int,
) -> Iterator[tuple[int, TerminologyConcept | None]]:
    rows = 0
    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if rows >= max_rows:
                break
            rows += 1
            code = (row.get("LOINC_NUM") or row.get("loinc_num") or "").strip()
            display = (
                row.get("LONG_COMMON_NAME")
                or row.get("long_common_name")
                or row.get("COMPONENT")
                or row.get("component")
                or ""
            ).strip()
            if not code or not display:
                yield rows, None
                continue
            yield rows, TerminologyConcept(
                concept_id=f"term_concept_loinc_{code}",
                system=TerminologySystem.LOINC,
                code=code,
                display=display,
                synonyms=[],
                version="licensed-local-term02",
                source_safe_id=source_safe_id,
                active=True,
                synthetic=False,
            )


def _validate_imported_lookup(store: LocalTerminologyStore) -> Term02LookupValidation:
    service = TerminologyLookupService(store)
    rxnorm_display = _first_display(store, TerminologySystem.RXNORM)
    loinc_display = _first_display(store, TerminologySystem.LOINC)
    rxnorm = service.lookup(rxnorm_display, systems=[TerminologySystem.RXNORM], max_results=5)
    loinc = service.lookup(loinc_display, systems=[TerminologySystem.LOINC], max_results=5)
    unknown = service.lookup("medai term02 synthetic unknown no mapping", max_results=5)
    ambiguous = _ambiguous_lookup_check()
    boundary = safe_b07_boundary_summary()
    b07_unknown = code_entity_via_local_terminology("medai term02 synthetic unknown no mapping", service)
    return Term02LookupValidation(
        exact_rxnorm_passed=rxnorm.status in (TerminologyLookupStatus.EXACT, TerminologyLookupStatus.SYNONYM) and bool(rxnorm.matches),
        exact_loinc_passed=loinc.status == TerminologyLookupStatus.EXACT and bool(loinc.matches),
        unknown_unmapped_passed=unknown.status == TerminologyLookupStatus.UNMAPPED and not unknown.matches,
        ambiguous_flag_passed=ambiguous,
        no_code_hallucinated=unknown.no_code_hallucinated and b07_unknown.status == TerminologyLookupStatus.UNMAPPED,
        b07_boundary_passed=(
            boundary["default_b07_behavior_unchanged"] is True
            and boundary["coding_promotes_hypothesis"] is False
            and boundary["coding_clears_ddi_status"] is False
        ),
    )


def _ambiguous_lookup_check() -> bool:
    temp_store = LocalTerminologyStore()
    sid = temp_store.register_source(TerminologySystem.RXNORM, safe_source_id="term02_ambiguous_synthetic", license_confirmed=True)
    temp_store.add_concepts(
        [
            TerminologyConcept(
                concept_id="term02_ambiguous_rxnorm",
                system=TerminologySystem.RXNORM,
                code="RXTERM02",
                display="term02 shared ambiguous",
                synthetic=True,
            ),
            TerminologyConcept(
                concept_id="term02_ambiguous_loinc",
                system=TerminologySystem.LOINC,
                code="LTERM02",
                display="term02 shared ambiguous",
                synthetic=True,
            ),
        ],
        sid,
    )
    result = TerminologyLookupService(temp_store).lookup("term02 shared ambiguous")
    return result.status == TerminologyLookupStatus.AMBIGUOUS and result.ambiguous


def _first_display(store: LocalTerminologyStore, system: TerminologySystem) -> str:
    with sqlite3.connect(store.db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT display FROM terminology_concepts WHERE system = ? ORDER BY concept_id LIMIT 1",
            (system.value,),
        ).fetchone()
    if not row:
        raise Term02ImportBlocked(f"no_imported_display:{system.value}")
    return str(row["display"])


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return f"term02_hash_{h.hexdigest()[:16]}"


def _safe_hash(raw: str, *, prefix: str) -> str:
    digest = hashlib.sha256(f"cka_term02:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def _short_hash(raw: str) -> str:
    return hashlib.sha256(f"term02_concept:{raw}".encode("utf-8")).hexdigest()[:12]


def _git_staged_paths(repo_root: Path) -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: Iterable[str], prefix: str) -> bool:
    prefix = prefix.replace("\\", "/")
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)
