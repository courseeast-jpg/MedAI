"""Deterministic local stub connector for future phi3 routing."""

from __future__ import annotations

import re


class Phi3Connector:
    name = "phi3"
    is_configured = True

    def __init__(self, extractor=None):
        self.extractor = extractor

    def extract(self, text: str, *, specialty: str = "general") -> dict:
        del specialty
        if self.extractor is not None:
            result = dict(self.extractor.extract(text))
            result.setdefault("extractor", self.name)
            result.setdefault("actual_extractor", result.get("extractor", self.name))
            result.setdefault("notes", [])
            return result

        entities: list[dict] = []

        diagnosis_match = re.search(r"\bdiagnosis\s*:?\s*([A-Za-z][A-Za-z ,/-]{2,80})", text, re.IGNORECASE)
        if diagnosis_match:
            entities.append({"type": "diagnosis", "text": diagnosis_match.group(1).strip(" .;")})

        medication_match = re.search(
            r"\b([A-Z][A-Za-z-]{2,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|units?)\b",
            text,
            re.IGNORECASE,
        )
        if medication_match:
            entities.append({
                "type": "medication",
                "text": medication_match.group(1),
                "dose": f"{medication_match.group(2)}{medication_match.group(3)}",
            })

        return {
            "extractor": self.name,
            "actual_extractor": self.name,
            "entities": entities,
            "confidence": 0.68 if entities else 0.4,
            "latency_ms": 1,
            "raw_text": text,
            "notes": ["phi3_stub_local_fallback"],
        }
