"""Deterministic spaCy/medspacy fast-path extractor."""

from __future__ import annotations

import re
import time

from loguru import logger

from extractors.base_extractor import BaseExtractor
from execution.confidence_scorer import score_extraction_result
from execution.supplemental_rules import apply_supplemental_rules


STRUCTURED_PARSER_VERSION = "structured_lab_parser_v1"
STRUCTURED_LAB_STATUS_VALUES = {"normal", "abnormal", "high", "low", "positive", "negative"}
STRUCTURED_LAB_VALUE_PREFIXES = ("value", "result")
STRUCTURED_LAB_STATUS_PREFIXES = ("status", "flag", "interpretation")
STRUCTURED_LAB_UNIT_PREFIXES = ("unit", "units")
STRUCTURED_LAB_REFERENCE_PREFIXES = ("reference", "reference range", "normal value", "range")


class SpacyExtractor(BaseExtractor):
    """Local extractor for short, clean text."""

    def __init__(self):
        self.nlp = None
        self.notes: list[str] = []
        self._load_model()

    def extract(self, text: str) -> dict:
        started = time.perf_counter()
        structured_entities = self._extract_structured_lab_entities(text)
        entities = list(structured_entities)
        entities.extend(self._regex_entities(text))

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
        result = apply_supplemental_rules({
            "extractor": "spacy",
            "entities": entities,
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "raw_text": text,
            "notes": list(self.notes),
            "structured_parser_used": bool(structured_entities),
            "structured_entities_count": len(structured_entities),
            "structured_parser_version": STRUCTURED_PARSER_VERSION,
        })
        return score_extraction_result(result)

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
            test_name = match.group(1).strip()
            test_value = match.group(2)
            if self._is_noise_test_result(test_name, test_value):
                continue
            entities.append({
                "type": "test_result",
                "text": test_name,
                "value": test_value,
                "unit": match.group(3),
            })

        return entities

    def _extract_structured_lab_entities(self, text: str) -> list[dict]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        entities: list[dict] = []

        for line in lines:
            inline_entity = self._parse_structured_inline_lab_line(line)
            if inline_entity is not None:
                entities.append(inline_entity)

        index = 0
        while index < len(lines):
            test_name = self._normalize_lab_test_name(lines[index])
            if test_name is None:
                index += 1
                continue

            value: str | None = None
            status: str | None = None
            unit: str | None = None
            reference_range: str | None = None
            lookahead = index + 1
            while lookahead < len(lines) and lookahead <= index + 4:
                candidate = lines[lookahead]
                if self._normalize_lab_test_name(candidate) is not None:
                    break

                parsed_value, parsed_unit = self._parse_value_line(candidate, allow_ambiguous_status=value is None)
                if value is None and parsed_value is not None:
                    value = parsed_value
                    unit = unit or parsed_unit
                    lookahead += 1
                    continue

                parsed_status = self._parse_status_line(candidate)
                if status is None and parsed_status is not None:
                    status = parsed_status
                    lookahead += 1
                    continue

                parsed_reference = self._parse_reference_range_line(candidate)
                if reference_range is None and parsed_reference is not None:
                    reference_range = parsed_reference
                    lookahead += 1
                    continue

                parsed_unit_only = self._parse_unit_line(candidate)
                if unit is None and parsed_unit_only is not None:
                    unit = parsed_unit_only
                    lookahead += 1
                    continue

                break

            if value is not None:
                entities.append(self._build_structured_lab_entity(
                    test_name=test_name,
                    value=value,
                    status=status,
                    unit=unit,
                    reference_range=reference_range,
                ))
                index = lookahead
                continue

            index += 1

        return entities

    def _parse_structured_inline_lab_line(self, line: str) -> dict | None:
        normalized = re.sub(r"\s+", " ", line.strip())
        colon_match = re.match(
            r"^(?P<name>UA\s+[A-Za-z][A-Za-z0-9 /()-]{1,60}?)\s+Value:\s*(?P<value>.+?)\s+Status:\s*(?P<status>Normal|Abnormal|High|Low|Positive|Negative)\b(?:\s+Unit:\s*(?P<unit>\S+))?(?:\s+Reference(?: Range)?:\s*(?P<reference>.+))?$",
            normalized,
            re.IGNORECASE,
        )
        if colon_match:
            return self._build_structured_lab_entity(
                test_name=self._format_lab_test_name(colon_match.group("name")),
                value=colon_match.group("value").strip(),
                status=colon_match.group("status").strip(),
                unit=self._clean_optional_field(colon_match.group("unit")),
                reference_range=self._clean_optional_field(colon_match.group("reference")),
            )

        token_match = re.match(
            r"^(?P<name>UA\s+[A-Za-z][A-Za-z0-9 /()-]{1,60}?)\s+(?P<value>Trace|Negative|Positive|[0-9]+(?:-[0-9]+)?(?:\.[0-9]+)?(?:\s*/\s*[A-Za-z]+)?|[A-Za-z][A-Za-z0-9 /.-]{0,30})\s+(?P<status>Normal|Abnormal|High|Low|Positive|Negative)\b(?:\s+(?P<unit>/?[A-Za-z%]+))?(?:\s+(?P<reference>[0-9A-Za-z<>= ./%-]+))?$",
            normalized,
            re.IGNORECASE,
        )
        if token_match:
            return self._build_structured_lab_entity(
                test_name=self._format_lab_test_name(token_match.group("name")),
                value=token_match.group("value").strip(),
                status=token_match.group("status").strip(),
                unit=self._clean_optional_field(token_match.group("unit")),
                reference_range=self._clean_optional_field(token_match.group("reference")),
            )
        return None

    def _normalize_lab_test_name(self, line: str) -> str | None:
        normalized = re.sub(r"\s+", " ", line.strip().replace(":", ""))
        if not normalized:
            return None
        if not normalized.upper().startswith("UA "):
            return None
        suffix = normalized[3:].strip()
        if not suffix or any(char.isdigit() for char in suffix):
            return None
        if len(suffix) > 64:
            return None
        return self._format_lab_test_name(normalized)

    def _parse_value_line(self, line: str, *, allow_ambiguous_status: bool = False) -> tuple[str | None, str | None]:
        normalized = re.sub(r"\s+", " ", line.strip())
        lowered = normalized.lower()
        for prefix in STRUCTURED_LAB_VALUE_PREFIXES:
            prefix_token = f"{prefix}:"
            if lowered.startswith(prefix_token):
                return self._split_value_and_unit(normalized.split(":", 1)[1].strip())

        if self._parse_status_line(normalized) is not None and not allow_ambiguous_status:
            return None, None
        if self._parse_reference_range_line(normalized) is not None:
            return None, None
        if self._parse_unit_line(normalized) is not None:
            return None, None
        if len(normalized) > 40:
            return None, None
        return self._split_value_and_unit(normalized)

    def _parse_status_line(self, line: str) -> str | None:
        normalized = re.sub(r"\s+", " ", line.strip())
        lowered = normalized.lower()
        for prefix in STRUCTURED_LAB_STATUS_PREFIXES:
            prefix_token = f"{prefix}:"
            if lowered.startswith(prefix_token):
                value = normalized.split(":", 1)[1].strip()
                return value if value.lower() in STRUCTURED_LAB_STATUS_VALUES else None
        return normalized if lowered in STRUCTURED_LAB_STATUS_VALUES else None

    def _parse_unit_line(self, line: str) -> str | None:
        normalized = re.sub(r"\s+", " ", line.strip())
        lowered = normalized.lower()
        for prefix in STRUCTURED_LAB_UNIT_PREFIXES:
            prefix_token = f"{prefix}:"
            if lowered.startswith(prefix_token):
                return self._clean_optional_field(normalized.split(":", 1)[1].strip())
        return None

    def _parse_reference_range_line(self, line: str) -> str | None:
        normalized = re.sub(r"\s+", " ", line.strip())
        lowered = normalized.lower()
        for prefix in STRUCTURED_LAB_REFERENCE_PREFIXES:
            prefix_token = f"{prefix}:"
            if lowered.startswith(prefix_token):
                return self._clean_optional_field(normalized.split(":", 1)[1].strip())
        return None

    def _split_value_and_unit(self, value_text: str) -> tuple[str | None, str | None]:
        value_text = value_text.strip()
        if not value_text:
            return None, None
        match = re.match(r"^(?P<value>.+?)\s+(?P<unit>/[A-Za-z]+|[A-Za-z%/]+)$", value_text)
        if match:
            return match.group("value").strip(), match.group("unit").strip()
        return value_text, None

    def _build_structured_lab_entity(
        self,
        *,
        test_name: str,
        value: str,
        status: str | None,
        unit: str | None,
        reference_range: str | None,
    ) -> dict:
        structured = {
            "test_name": test_name,
            "value": value,
            "status": status,
            "unit": unit,
            "reference_range": reference_range,
            "source": "structured_lab_parser",
            "entity_type": "lab_result",
            "confidence": 0.85,
        }
        return {
            "type": "test_result",
            "text": test_name,
            "entity_type": "lab_result",
            "test_name": test_name,
            "value": value,
            "status": status,
            "unit": unit,
            "reference_range": reference_range,
            "source": "structured_lab_parser",
            "confidence": 0.85,
            "structured": structured,
        }

    def _format_lab_test_name(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.strip())
        if not normalized:
            return value
        parts = normalized.split(" ", 1)
        if len(parts) == 1:
            return normalized.upper()
        return f"{parts[0].upper()} {parts[1]}"

    def _clean_optional_field(self, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def _is_noise_test_result(self, test_name: str, test_value: str) -> bool:
        normalized_name = test_name.strip().lower()
        if normalized_name in {"page", "pg"} and test_value.isdigit():
            return True
        return False

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
