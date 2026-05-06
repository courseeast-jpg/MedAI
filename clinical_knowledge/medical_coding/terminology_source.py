"""Terminology source abstraction for CKA-B07.

Defines the TerminologySource base interface and status protocol.
No external API calls. No downloads. No real licensed terminology data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from clinical_knowledge.medical_coding.models import (
    CodingResult,
    TerminologySourceStatus,
)


class TerminologySource(ABC):
    """Abstract base for all terminology sources.

    All implementations must be deterministic and local-only.
    No network calls permitted. No LLM calls.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this source."""

    @abstractmethod
    def status(self) -> TerminologySourceStatus:
        """Return current availability status of this source."""

    @abstractmethod
    def lookup(
        self,
        normalized_text: str,
        fact_type: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> CodingResult:
        """Look up normalized_text and return a CodingResult.

        Must never make network calls.
        Must never hallucinate codes.
        Must return status=unmapped if entity is not in the local table.
        """
