"""CKA-TERM-01C safe synthetic import audit summaries."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TerminologyImportAuditSummary:
    system: str
    source_safe_id: str
    records_seen: int = 0
    records_imported: int = 0
    records_skipped: int = 0
    chunks_processed: int = 0
    checkpoint_count: int = 0
    rollback_performed: bool = False
    import_completed: bool = False
    import_mode: str = "dry_run"
    audit_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.audit_id:
            self.audit_id = f"term_audit_{uuid.uuid4().hex[:12]}"

    def safe_public_summary(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "system": self.system,
            "source_safe_id": self.source_safe_id,
            "records_seen": self.records_seen,
            "records_imported": self.records_imported,
            "records_skipped": self.records_skipped,
            "chunks_processed": self.chunks_processed,
            "checkpoint_count": self.checkpoint_count,
            "rollback_performed": self.rollback_performed,
            "import_completed": self.import_completed,
            "import_mode": self.import_mode,
            "warnings": list(self.warnings),
        }
