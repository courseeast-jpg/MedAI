"""Deterministic spaCy/medspacy fast-path extractor."""

from __future__ import annotations

import re
import time

from loguru import logger

from extractors.base_extractor import BaseExtractor


class SpacyExtractor(BaseExtractor):
    """Local extractor for short, clean text."""

    def __init__(self):
        self.nlp = None
        self.notes: list[str] = []
        self._load_model()

    def extract(self, text: str) -> dict:
        started = time.perf_counter()
        entities = self._regex_entities(text)

        if self.nlp is not None:
            doc = self.nlp(text[:5000])
            for ent in doc.ents:
                entity_type = self._map_label(ent.label_)
                if entity_type:
                    entities.append({
                        "type": entity_type,
                        "text": ent.text.strip(),
                        "label": ent.label_,
                    })

        entities = self._dedupe(entities)
        latency_ms = int((time.perf_counter() - started) * 1000)
        confidence = 0.70 if entities else 0.35
        return {
            "extractor": "spacy",
            "entities": entities,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "raw_text": text,
            "notes": list(self.notes),
        }

    def _load_model(self) -> None:
        try:
            import spacy
        except ImportError:
            self.notes.append("spacy unavailable; regex extraction only")
            return

        for model_name in ("en_core_sci_md", "en_core_web_sm"):
            try:
                self.nlp = spacy.load(model_name)
                self.notes.append(f"loaded {model_name}")
                return
            except OSError:
                continue

        try:
            import medspacy

            self.nlp = medspacy.load()
            self.notes.append("loaded medspacy")
        except Exception as exc:
            logger.warning("No spaCy medical model available: {}", exc)
            self.notes.append("no spaCy model available; regex extraction only")

    def _regex_entities(self, text: str) -> list[dict]:
        entities: list[dict] = []

        med_pattern = re.compile(
            r"\b([A-Z][A-Za-z-]{2,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units?)\b",
            re.IGNORECASE,
        )
        for match in med_pattern.finditer(text):
            entities.append({
                "type": "medication",
                "text": match.group(1),
                "dose": f"{match.group(2)}{match.group(3)}",
            })

        diagnosis_pattern = re.compile(
            r"\b(?:diagnosis|diagnosed with|history of)\s*:?\s*([A-Za-z][A-Za-z ,/-]{3,80})",
            re.IGNORECASE,
        )
        for match in diagnosis_pattern.finditer(text):
            entities.append({"type": "diagnosis", "text": match.group(1).strip(" .;")})

        test_pattern = re.compile(
            r"\b([A-Z][A-Za-z ]{2,40})\s*:?\s+(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L|%|bpm|mmHg)?\b"
        )
        for match in test_pattern.finditer(text):
            entities.append({
                "type": "test_result",
                "text": match.group(1).strip(),
                "value": match.group(2),
                "unit": match.group(3),
            })

        return entities

    def _map_label(self, label: str) -> str | None:
        normalized = label.upper()
        if normalized in {"DISEASE", "DIAGNOSIS", "CONDITION", "PROBLEM"}:
            return "diagnosis"
        if normalized in {"DRUG", "CHEMICAL", "MEDICATION"}:
            return "medication"
        if normalized in {"TEST", "PROCEDURE"}:
            return "test_result"
        return None

    def _dedupe(self, entities: list[dict]) -> list[dict]:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for entity in entities:
            key = (entity.get("type", ""), entity.get("text", "").lower())
            if not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped
