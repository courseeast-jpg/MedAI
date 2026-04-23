"""Phase 1 extractor adapters."""

from extractors.base_extractor import BaseExtractor
from extractors.gemini_extractor import GeminiExtractor
from extractors.spacy_extractor import SpacyExtractor

__all__ = ["BaseExtractor", "GeminiExtractor", "SpacyExtractor"]
