"""Gemini fallback adapter around the existing extractor logic."""

from __future__ import annotations

import time

from loguru import logger

from app.config import GEMINI_API_KEY
from extractors.base_extractor import BaseExtractor


def _is_quota_or_rate_limit(message: str | None) -> bool:
    lowered = str(message or "").lower()
    return any(
        marker in lowered
        for marker in (
            "429",
            "quota exceeded",
            "current quota",
            "rate limit",
            "retry_delay",
            "generate_content_free_tier_requests",
        )
    )


def _is_privacy_gate_block(message: str | None) -> bool:
    return "privacy gate" in str(message or "").lower()


class GeminiExtractor(BaseExtractor):
    """Wraps extraction.extractor.Extractor in the Phase 1 return schema."""

    def __init__(self, specialty: str = "general", legacy_extractor=None):
        self.specialty = specialty
        self.gemini_available = bool(GEMINI_API_KEY)
        if legacy_extractor is None:
            from extraction.extractor import Extractor

            legacy_extractor = Extractor()
        self.legacy_extractor = legacy_extractor
        if not self.gemini_available:
            logger.warning("Gemini extractor unavailable: GEMINI_API_KEY is missing")

    def extract(self, text: str) -> dict:
        started = time.perf_counter()
        output = self.legacy_extractor.extract(text, self.specialty)
        entities: list[dict] = []

        for diagnosis in output.diagnoses:
            entities.append({
                "type": "diagnosis",
                "text": diagnosis.name,
                "structured": diagnosis.model_dump(exclude_none=True),
            })
        for medication in output.medications:
            entities.append({
                "type": "medication",
                "text": medication.name,
                "structured": medication.model_dump(exclude_none=True),
            })
        for test in output.test_results:
            entities.append({
                "type": "test_result",
                "text": test.test_name,
                "structured": test.model_dump(exclude_none=True),
            })
        for symptom in output.symptoms:
            if hasattr(symptom, "model_dump"):
                structured = symptom.model_dump(exclude_none=True)
                text_value = structured.get("description", "")
            else:
                structured = {"description": str(symptom)}
                text_value = str(symptom)
            entities.append({"type": "symptom", "text": text_value, "structured": structured})
        for note in output.notes:
            entities.append({"type": "note", "text": str(note), "structured": {"text": str(note)}})
        for recommendation in output.recommendations:
            entities.append({
                "type": "recommendation",
                "text": str(recommendation),
                "structured": {"text": str(recommendation)},
            })

        latency_ms = int((time.perf_counter() - started) * 1000)
        method = output.extraction_method or "gemini"
        notes = []
        if not self.gemini_available:
            notes.append("gemini_unavailable=missing_api_key")
        if method != "gemini":
            root_cause = getattr(self.legacy_extractor, "last_gemini_error_message", None)
            if self.gemini_available and _is_quota_or_rate_limit(root_cause):
                fallback_method = "spacy" if method == "rules_based" else method
                from execution.connectors.phi3_connector import Phi3Connector

                phi3_result = Phi3Connector().extract(text)
                phi3_entities = list(phi3_result.get("entities", []))
                if len(phi3_entities) > len(entities):
                    entities = phi3_entities
                    fallback_method = str(phi3_result.get("actual_extractor", "phi3"))
                notes.extend([
                    "router_fallback=gemini:gemini_quota_or_rate_limit",
                    "gemini_configured_fallback=true",
                    "fallback_reason=gemini_quota_or_rate_limit",
                ])
                return {
                    "extractor": "gemini",
                    "actual_extractor": fallback_method,
                    "primary_extractor": "gemini",
                    "fallback_extractor": fallback_method,
                    "fallback_reason": "gemini_quota_or_rate_limit",
                    "terminal_empty_prevented": bool(entities),
                    "entities": entities,
                    "confidence": float(output.confidence),
                    "latency_ms": latency_ms,
                    "raw_text": text,
                    "notes": notes,
                }
            logger.warning("Gemini route did not execute real Gemini; actual extractor={}", method)
            if self.gemini_available and _is_privacy_gate_block(root_cause):
                notes.extend([
                    "router_fallback=gemini:privacy_gate_blocked",
                    "privacy_gate_blocked=true",
                ])
            elif self.gemini_available:
                logger.error("Gemini route fallback occurred despite configured GEMINI_API_KEY")
                if root_cause:
                    raise RuntimeError(
                        f"Gemini route fallback occurred despite configured key: {method}; root_cause={root_cause}"
                    )
                raise RuntimeError(f"Gemini route fallback occurred despite configured key: {method}")
            notes.append(f"gemini_route_legacy_fallback={method}")
        return {
            "extractor": "gemini",
            "actual_extractor": method,
            "entities": entities,
            "confidence": float(output.confidence),
            "latency_ms": latency_ms,
            "raw_text": text,
            "notes": notes,
        }
