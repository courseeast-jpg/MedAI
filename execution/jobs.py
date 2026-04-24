"""Execution job/result models for the Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas import MKBRecord


@dataclass(frozen=True)
class ExecutionJob:
    """A single deterministic ingestion job."""

    text: str = ""
    pdf_path: Path | None = None
    source_name: str = "manual"
    specialty: str = "general"
    session_id: str = ""


@dataclass
class ExecutionResult:
    """Pipeline outcome. Outcome is intentionally constrained."""

    outcome: str
    validation_status: str = "accepted"
    validation_errors: list[dict] = field(default_factory=list)
    records: list[MKBRecord] = field(default_factory=list)
    queued_records: list[MKBRecord] = field(default_factory=list)
    blocked_records: list[MKBRecord] = field(default_factory=list)
    ddi_findings: list[Any] = field(default_factory=list)
    extractor_result: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def written_count(self) -> int:
        return len(self.records)

    @property
    def queued_count(self) -> int:
        return len(self.queued_records)
