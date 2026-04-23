"""Base extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> dict:
        """Extract entities from text and return the Phase 1 schema."""
        raise NotImplementedError


def empty_result(extractor: str, text: str, latency_ms: int = 0, notes: list[str] | None = None) -> dict:
    return {
        "extractor": extractor,
        "entities": [],
        "confidence": 0.0,
        "latency_ms": int(latency_ms),
        "raw_text": text,
        "notes": notes or [],
    }
