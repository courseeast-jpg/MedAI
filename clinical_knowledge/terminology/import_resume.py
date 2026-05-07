"""CKA-TERM-01G synthetic chunk/resume import simulation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from clinical_knowledge.terminology.import_limits import TerminologyImportLimits
from clinical_knowledge.terminology.import_performance import elapsed_seconds_safe_bucket
from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import TerminologyConcept, TerminologySystem
from clinical_knowledge.terminology.parsers import ParseResult


@dataclass(frozen=True)
class ScaleResumeMetrics:
    system: str
    rows_seen: int
    rows_imported: int
    chunks_processed: int
    checkpoints_written: int
    resume_performed: bool
    duplicate_prevention_passed: bool
    row_cap_enforced: bool
    chunking_verified: bool
    streaming_parser_guard_passed: bool
    elapsed_seconds_safe_bucket: str
    warnings: list[str] = field(default_factory=list)

    def safe_public_summary(self) -> dict:
        return {
            "system": self.system,
            "rows_seen": self.rows_seen,
            "rows_imported": self.rows_imported,
            "chunks_processed": self.chunks_processed,
            "checkpoints_written": self.checkpoints_written,
            "resume_performed": self.resume_performed,
            "duplicate_prevention_passed": self.duplicate_prevention_passed,
            "row_cap_enforced": self.row_cap_enforced,
            "chunking_verified": self.chunking_verified,
            "streaming_parser_guard_passed": self.streaming_parser_guard_passed,
            "elapsed_seconds_safe_bucket": self.elapsed_seconds_safe_bucket,
            "warnings": list(self.warnings),
        }


def simulate_chunked_import_with_resume(
    parse_result: ParseResult,
    *,
    limits: TerminologyImportLimits,
    interrupt_after_chunks: int = 1,
    store: LocalTerminologyStore | None = None,
) -> ScaleResumeMetrics:
    start = time.perf_counter()
    active_store = store or LocalTerminologyStore()
    concepts = list(parse_result.concepts)
    capped = concepts[: max(0, limits.max_rows_per_file_default)]
    chunk_size = max(1, limits.chunk_size)
    chunks = [capped[i:i + chunk_size] for i in range(0, len(capped), chunk_size)]
    source_id = active_store.register_source(
        parse_result.system,
        safe_source_id=f"scale_resume_{parse_result.system.value}",
        version="synthetic-scale-1",
        license_confirmed=True,
    )

    imported_ids: set[str] = set()
    checkpoints = 0
    processed = 0
    resume_performed = False

    for chunk_index, chunk in enumerate(chunks):
        if chunk_index >= interrupt_after_chunks:
            break
        active_store.add_concepts(chunk, source_id)
        imported_ids.update(c.concept_id for c in chunk)
        checkpoints += 1
        processed += 1

    resume_start = processed
    if resume_start < len(chunks):
        resume_performed = True
    for chunk in chunks[resume_start:]:
        deduped = [concept for concept in chunk if concept.concept_id not in imported_ids]
        active_store.add_concepts(deduped, source_id)
        imported_ids.update(c.concept_id for c in deduped)
        checkpoints += 1
        processed += 1

    # Replay the full capped set to prove INSERT OR REPLACE prevents duplicate
    # concept rows in the synthetic store.
    active_store.add_concepts(capped, source_id)
    final_count = active_store.count_concepts(parse_result.system)
    expected = len({c.concept_id for c in capped})
    duplicate_prevention = final_count == expected
    elapsed = time.perf_counter() - start
    return ScaleResumeMetrics(
        system=parse_result.system.value,
        rows_seen=parse_result.rows_seen,
        rows_imported=final_count,
        chunks_processed=processed,
        checkpoints_written=checkpoints,
        resume_performed=resume_performed,
        duplicate_prevention_passed=duplicate_prevention,
        row_cap_enforced=final_count <= limits.max_rows_per_file_default,
        chunking_verified=processed == len(chunks) and (len(chunks) > 1 if capped else True),
        streaming_parser_guard_passed=_streaming_guard_passed(parse_result, limits),
        elapsed_seconds_safe_bucket=elapsed_seconds_safe_bucket(elapsed),
    )


def _streaming_guard_passed(parse_result: ParseResult, limits: TerminologyImportLimits) -> bool:
    # SNOMED RF2 parsing streams concept and description files separately, so
    # rows_seen can be up to 2x the per-file cap while imported concepts remain
    # capped to max_rows_per_file_default.
    multiplier = 2 if parse_result.system == TerminologySystem.SNOMED_CT else 1
    return parse_result.rows_seen <= limits.max_rows_per_file_default * multiplier
