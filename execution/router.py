"""Deterministic connector routing and fallback orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from execution.connectors.gemini_connector import GeminiConnector
from execution.connectors.phi3_connector import Phi3Connector
from execution.connectors.spacy_connector import SpacyConnector


OCR_ARTIFACT_RE = re.compile(r"(\ufffd|[|]{3,}|_{4,}|\b(?:l|I){8,}\b)")


@dataclass
class RoutedExtraction:
    extractor_route: str
    extractor_actual: str
    results: list[dict[str, Any]]
    fallback_used: bool = False
    failure_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class ExecutionRouter:
    """Small routing layer that orchestrates connectors before consensus."""

    def __init__(
        self,
        *,
        spacy_connector: SpacyConnector,
        gemini_connector_factory: Callable[[str], GeminiConnector],
        phi3_connector: Phi3Connector,
        spacy_fast_path_char_limit: int = 3000,
    ):
        self.spacy_connector = spacy_connector
        self.gemini_connector_factory = gemini_connector_factory
        self.phi3_connector = phi3_connector
        self.spacy_fast_path_char_limit = spacy_fast_path_char_limit

    def select_route(self, text: str) -> str:
        if len(text) < self.spacy_fast_path_char_limit and not self._has_ocr_artifacts(text):
            return "spacy"
        return "gemini"

    def execute(self, text: str, *, specialty: str = "general") -> RoutedExtraction:
        extractor_route = self.select_route(text)
        if extractor_route == "spacy":
            spacy_result = self.spacy_connector.extract(text, specialty=specialty)
            return RoutedExtraction(
                extractor_route="spacy",
                extractor_actual=str(spacy_result.get("actual_extractor", spacy_result.get("extractor", "spacy"))),
                results=[spacy_result],
            )

        spacy_result = self.spacy_connector.extract(text, specialty=specialty)
        results: list[dict[str, Any]] = []
        if spacy_result.get("entities"):
            results.append(spacy_result)
        events: list[dict[str, Any]] = []
        gemini_connector = self.gemini_connector_factory(specialty)

        try:
            gemini_result = gemini_connector.extract(text, specialty=specialty)
            results.append(gemini_result)
            extractor_actual = str(gemini_result.get("actual_extractor", gemini_result.get("extractor", "gemini")))
            if extractor_actual != extractor_route:
                events.append({
                    "action": "route_actual_mismatch",
                    "reason": f"intended={extractor_route} actual={extractor_actual}",
                })
            return RoutedExtraction(
                extractor_route=extractor_route,
                extractor_actual=extractor_actual,
                results=results,
                events=events,
            )
        except TimeoutError as exc:
            failure_code = "timeout"
            failure_message = str(exc) or "connector_timeout"
        except Exception as exc:  # noqa: BLE001 - deterministic classification wrapper
            failure_code = self._classify_error(exc)
            failure_message = str(exc) or "connector_failure"

        fallback_result = self.phi3_connector.extract(text, specialty=specialty)
        fallback_notes = list(fallback_result.get("notes", []))
        fallback_notes.append(f"router_fallback=gemini:{failure_code}")
        if gemini_connector.is_configured:
            fallback_notes.append("gemini_configured_fallback=true")
        fallback_result["notes"] = fallback_notes

        events.append({
            "action": "fallback_invoked",
            "reason": failure_message,
            "from": "gemini",
            "to": str(fallback_result.get("actual_extractor", fallback_result.get("extractor", "phi3"))),
            "error_code": failure_code,
            "configured_gemini_fallback": gemini_connector.is_configured,
        })

        results.append(fallback_result)
        return RoutedExtraction(
            extractor_route=extractor_route,
            extractor_actual=str(fallback_result.get("actual_extractor", fallback_result.get("extractor", "phi3"))),
            results=results,
            fallback_used=True,
            failure_count=1,
            events=events,
        )

    def _has_ocr_artifacts(self, text: str) -> bool:
        if OCR_ARTIFACT_RE.search(text):
            return True
        if not text:
            return False
        non_alnum = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return (non_alnum / max(len(text), 1)) > 0.18

    def _classify_error(self, exc: Exception) -> str:
        if "timeout" in str(exc).lower():
            return "timeout"
        return "connector_error"
