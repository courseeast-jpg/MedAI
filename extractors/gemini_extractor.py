"""Gemini fallback adapter around the existing extractor logic."""

from __future__ import annotations

import time

from extractors.base_extractor import BaseExtractor


class GeminiExtractor(BaseExtractor):
    """Wraps extraction.extractor.Extractor in the Phase 1 return schema."""

    def __init__(self, specialty: str = "general", legacy_extractor=None):
        self.specialty = specialty
        if legacy_extractor is None:
            from extraction.extractor import Extractor

            legacy_extractor = Extractor()
        self.legacy_extractor = legacy_extractor

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
        return {
            "extractor": "gemini" if method == "gemini" else method,
            "entities": entities,
            "confidence": float(output.confidence),
            "latency_ms": latency_ms,
            "raw_text": text,
            "notes": [],
        }
