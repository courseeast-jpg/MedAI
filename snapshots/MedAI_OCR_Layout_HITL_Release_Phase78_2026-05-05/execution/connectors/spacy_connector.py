"""Connector wrapper for local spaCy extraction."""

from __future__ import annotations

from extractors.spacy_extractor import SpacyExtractor


class SpacyConnector:
    name = "spacy"
    is_configured = True

    def __init__(self, extractor: SpacyExtractor | None = None):
        self.extractor = extractor or SpacyExtractor()

    def extract(self, text: str, *, specialty: str = "general") -> dict:
        del specialty
        result = dict(self.extractor.extract(text))
        result.setdefault("extractor", self.name)
        result.setdefault("actual_extractor", result.get("extractor", self.name))
        result.setdefault("notes", [])
        return result
