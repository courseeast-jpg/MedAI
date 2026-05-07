"""CKA-TERM-01C synthetic terminology import executor scaffold.

TERM-01C prepares execution mechanics only. Real licensed imports remain
blocked by default and are left for TERM-02.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from clinical_knowledge.terminology.import_audit import TerminologyImportAuditSummary
from clinical_knowledge.terminology.import_checkpoint import TerminologyImportCheckpoint
from clinical_knowledge.terminology.import_limits import TerminologyImportLimits
from clinical_knowledge.terminology.import_transaction import TerminologyImportTransaction
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologyConcept, TerminologySystem
from clinical_knowledge.terminology.parsers import ParseResult


@dataclass(frozen=True)
class TerminologyImportExecutionResult:
    audit: TerminologyImportAuditSummary
    checkpoints: list[TerminologyImportCheckpoint]
    store_summary: dict
    real_import_performed: bool = False
    external_api_used: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "audit": self.audit.safe_public_summary(),
            "checkpoints": [checkpoint.safe_public_summary() for checkpoint in self.checkpoints],
            "store_summary": dict(self.store_summary),
            "real_import_performed": self.real_import_performed,
            "external_api_used": self.external_api_used,
        }


class RealTerminologyImportBlocked(RuntimeError):
    """Raised when a caller attempts real import before TERM-02 enables it."""


class TerminologyImportExecutor:
    def __init__(
        self,
        *,
        store: LocalTerminologyStore | None = None,
        limits: TerminologyImportLimits | None = None,
        allow_real_import: bool = False,
    ) -> None:
        self.store = store or LocalTerminologyStore()
        self.limits = limits or TerminologyImportLimits()
        self.allow_real_import = bool(allow_real_import and self.limits.allow_real_import)

    def dry_run(self, parse_result: ParseResult, *, source_safe_id: str = "term_source_synthetic") -> TerminologyImportExecutionResult:
        audit = self._audit_for_parse_result(
            parse_result,
            source_safe_id=source_safe_id,
            mode="dry_run",
            imported=0,
            completed=True,
            rollback=False,
        )
        checkpoints = self._build_checkpoints(
            system=parse_result.system.value,
            source_safe_id=source_safe_id,
            rows_seen=parse_result.rows_seen,
            rows_imported=0,
        )
        audit.checkpoint_count = len(checkpoints)
        return TerminologyImportExecutionResult(
            audit=audit,
            checkpoints=checkpoints,
            store_summary=self.store.safe_public_summary(),
        )

    def execute_synthetic(
        self,
        parse_result: ParseResult,
        *,
        source_safe_id: str = "term_source_synthetic",
        simulate_failure_after_source: bool = False,
    ) -> TerminologyImportExecutionResult:
        concepts = list(parse_result.concepts)
        row_cap = max(0, self.limits.max_rows_per_file_default)
        capped_concepts = concepts[:row_cap]
        skipped_by_cap = max(0, len(concepts) - len(capped_concepts))
        checkpoints = self._build_checkpoints(
            system=parse_result.system.value,
            source_safe_id=source_safe_id,
            rows_seen=parse_result.rows_seen,
            rows_imported=len(capped_concepts),
        )
        audit = self._audit_for_parse_result(
            parse_result,
            source_safe_id=source_safe_id,
            mode="execute_synthetic",
            imported=0,
            completed=False,
            rollback=False,
        )
        audit.records_skipped += skipped_by_cap
        audit.checkpoint_count = len(checkpoints)
        try:
            with TerminologyImportTransaction(self.store) as tx:
                source_id = tx.write_source_manifest(
                    parse_result.system,
                    source_safe_id=source_safe_id,
                    version="synthetic-test-1",
                    license_confirmed=True,
                )
                if simulate_failure_after_source:
                    raise RuntimeError("simulated_import_failure")
                imported = self._write_in_chunks(tx, capped_concepts, source_id)
                tx.write_import_audit_event(
                    source_id,
                    event_type="synthetic_import_completed",
                    rows_seen=parse_result.rows_seen,
                    rows_imported=imported,
                )
            audit.records_imported = imported
            audit.import_completed = True
        except Exception:
            audit.records_imported = 0
            audit.rollback_performed = True
            audit.import_completed = False
        return TerminologyImportExecutionResult(
            audit=audit,
            checkpoints=checkpoints,
            store_summary=self.store.safe_public_summary(),
        )

    def execute_real_import_blocked(self) -> None:
        if not self.allow_real_import:
            raise RealTerminologyImportBlocked("real_import_blocked_until_term02")

    def _write_in_chunks(
        self,
        tx: TerminologyImportTransaction,
        concepts: list[TerminologyConcept],
        source_id: str,
    ) -> int:
        chunk_size = max(1, self.limits.chunk_size)
        imported = 0
        for index in range(0, len(concepts), chunk_size):
            chunk = concepts[index:index + chunk_size]
            imported += tx.write_concepts(chunk, source_id)
        return imported

    def _audit_for_parse_result(
        self,
        parse_result: ParseResult,
        *,
        source_safe_id: str,
        mode: str,
        imported: int,
        completed: bool,
        rollback: bool,
    ) -> TerminologyImportAuditSummary:
        chunks = math.ceil(max(0, min(len(parse_result.concepts), self.limits.max_rows_per_file_default)) / max(1, self.limits.chunk_size))
        return TerminologyImportAuditSummary(
            system=parse_result.system.value,
            source_safe_id=source_safe_id,
            records_seen=parse_result.rows_seen,
            records_imported=imported,
            records_skipped=parse_result.skipped_rows,
            chunks_processed=chunks,
            checkpoint_count=0,
            rollback_performed=rollback,
            import_completed=completed,
            import_mode=mode,
        )

    def _build_checkpoints(
        self,
        *,
        system: str,
        source_safe_id: str,
        rows_seen: int,
        rows_imported: int,
    ) -> list[TerminologyImportCheckpoint]:
        interval = max(1, self.limits.checkpoint_interval_rows)
        if rows_seen <= 0:
            return []
        count = math.ceil(rows_seen / interval)
        checkpoints: list[TerminologyImportCheckpoint] = []
        for index in range(count):
            checkpoint_rows = min(rows_seen, (index + 1) * interval)
            imported_rows = min(rows_imported, checkpoint_rows)
            checkpoints.append(TerminologyImportCheckpoint(
                system=system,
                source_safe_id=source_safe_id,
                file_safe_id=f"term_file_safe_{index + 1:04d}",
                rows_seen=checkpoint_rows,
                rows_imported=imported_rows,
                chunk_index=index,
                completed=index == count - 1,
                failed=False,
            ))
        return checkpoints
