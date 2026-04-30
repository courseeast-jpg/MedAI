"""Deterministic local stub connector for future phi3 routing."""

from __future__ import annotations

import re

from execution.confidence_scorer import score_extraction_result
from execution.supplemental_rules import apply_supplemental_rules


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

        has_match = re.search(r"\b(?:has|with)\s+([A-Za-z][A-Za-z -]{2,60})\b", text, re.IGNORECASE)
        if has_match:
            diagnosis = has_match.group(1).strip(" .;")
            if not any(item["type"] == "diagnosis" and item["text"].lower() == diagnosis.lower() for item in entities):
                entities.append({"type": "diagnosis", "text": diagnosis})

        takes_match = re.search(r"\btakes\s+([A-Za-z][A-Za-z-]{2,40})\b", text, re.IGNORECASE)
        if takes_match:
            medication = takes_match.group(1).strip(" .;")
            if not any(item["type"] == "medication" and item["text"].lower() == medication.lower() for item in entities):
                entities.append({"type": "medication", "text": medication})

        for negated in re.findall(r"\bno\s+([A-Za-z][A-Za-z -]{2,60})", text, re.IGNORECASE):
            diagnosis = negated.strip(" .;")
            if not any(item["type"] == "diagnosis" and item["text"].lower() == diagnosis.lower() for item in entities):
                entities.append({"type": "diagnosis", "text": diagnosis, "negated": True})

        normalized = text.lower()
        if "ua blood" in normalized or "blood positive" in normalized:
            entities.append({"type": "test_result", "text": "UA Blood"})
        if re.search(r"\brbc\b", text, re.IGNORECASE):
            entities.append({"type": "test_result", "text": "RBC"})
        if "calcium oxalate crystals" in normalized:
            entities.append({"type": "test_result", "text": "Calcium Oxalate Crystals"})

        result = apply_supplemental_rules({
            "extractor": self.name,
            "actual_extractor": self.name,
            "entities": entities,
            "confidence": 0.0,
            "latency_ms": 1,
            "raw_text": text,
            "notes": ["phi3_stub_local_fallback"],
        })
        return score_extraction_result(result)
