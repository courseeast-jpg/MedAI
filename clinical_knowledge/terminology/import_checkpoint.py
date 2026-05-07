"""CKA-TERM-01B terminology import checkpoint model.

TERM-01B defines the checkpoint shape only. It does not persist production
checkpoint files and does not import terminology data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class TerminologyImportCheckpoint:
    system: str
    source_safe_id: str
    file_safe_id: str
    rows_seen: int = 0
    rows_imported: int = 0
    chunk_index: int = 0
    completed: bool = False
    failed: bool = False
    checkpoint_id: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            self.checkpoint_id = f"term_ckpt_{uuid.uuid4().hex[:12]}"

    def safe_public_summary(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "system": self.system,
            "source_safe_id": self.source_safe_id,
            "file_safe_id": self.file_safe_id,
            "rows_seen": self.rows_seen,
            "rows_imported": self.rows_imported,
            "chunk_index": self.chunk_index,
            "completed": self.completed,
            "failed": self.failed,
        }


def simulate_checkpoint_resume(
    checkpoint: TerminologyImportCheckpoint,
    *,
    additional_rows_seen: int,
    additional_rows_imported: int,
    chunk_increment: int = 1,
) -> TerminologyImportCheckpoint:
    return TerminologyImportCheckpoint(
        checkpoint_id=checkpoint.checkpoint_id,
        system=checkpoint.system,
        source_safe_id=checkpoint.source_safe_id,
        file_safe_id=checkpoint.file_safe_id,
        rows_seen=checkpoint.rows_seen + max(0, additional_rows_seen),
        rows_imported=checkpoint.rows_imported + max(0, additional_rows_imported),
        chunk_index=checkpoint.chunk_index + max(0, chunk_increment),
        completed=False,
        failed=False,
    )
