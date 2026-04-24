"""Connector wrapper for Gemini extraction."""

from __future__ import annotations

from extractors.gemini_extractor import GeminiExtractor


class GeminiConnector:
    name = "gemini"

    def __init__(self, extractor: GeminiExtractor):
        self.extractor = extractor

    @property
    def is_configured(self) -> bool:
        return bool(getattr(self.extractor, "gemini_available", True))

    def extract(self, text: str, *, specialty: str = "general") -> dict:
        del specialty
        result = dict(self.extractor.extract(text))
        result.setdefault("extractor", self.name)
        result.setdefault("actual_extractor", result.get("extractor", self.name))
        result.setdefault("notes", [])
        return result
