from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")

DIAGNOSIS_SEED_MAP = {
    "type 2 diabetes": {
        "coding_system": "SNOMED-CT-seed",
        "code": "44054006",
        "code_display": "Type 2 diabetes mellitus",
    },
    "diabetes": {
        "coding_system": "SNOMED-CT-seed",
        "code": "73211009",
        "code_display": "Diabetes mellitus",
    },
    "diabetes mellitus": {
        "coding_system": "SNOMED-CT-seed",
        "code": "73211009",
        "code_display": "Diabetes mellitus",
    },
    "myocardial infarction": {
        "coding_system": "SNOMED-CT-seed",
        "code": "22298006",
        "code_display": "Myocardial infarction",
    },
    "heart attack": {
        "coding_system": "SNOMED-CT-seed",
        "code": "22298006",
        "code_display": "Myocardial infarction",
    },
    "hypertension": {
        "coding_system": "SNOMED-CT-seed",
        "code": "38341003",
        "code_display": "Hypertensive disorder, systemic arterial",
    },
    "high blood pressure": {
        "coding_system": "SNOMED-CT-seed",
        "code": "38341003",
        "code_display": "Hypertensive disorder, systemic arterial",
    },
}

MEDICATION_SEED_MAP = {
    "metformin": {
        "coding_system": "UMLS-seed",
        "code": "C0025598",
        "code_display": "Metformin",
    },
    "aspirin": {
        "coding_system": "UMLS-seed",
        "code": "C0004057",
        "code_display": "Aspirin",
    },
}

AMBIGUOUS_SEED_MAP = {
    ("diagnosis", "mi"): {
        "coding_system": "SNOMED-CT-seed",
        "code": None,
        "code_display": "Ambiguous abbreviation for myocardial infarction",
    },
}

SUPPORTED_ENTITY_TYPES = {"diagnosis", "medication"}


@dataclass(frozen=True)
class MedicalCodingEntry:
    original_entity_text: str
    normalized_entity_text: str
    entity_type: str
    coding_system: str | None
    code: str | None
    code_display: str | None
    coding_confidence: float
    coding_source: str
    coding_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MedicalCodingResult:
    applied: bool
    coding_source: str
    entries: list[MedicalCodingEntry]
    coding_attempted_count: int
    coding_success_count: int
    coding_unmapped_count: int
    coding_ambiguous_count: int
    coding_skipped_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


def map_medical_codes(*, entities: list[dict[str, Any]]) -> MedicalCodingResult:
    source = "rules_based_seed"
    if not entities:
        return MedicalCodingResult(
            applied=False,
            coding_source=source,
            entries=[],
            coding_attempted_count=0,
            coding_success_count=0,
            coding_unmapped_count=0,
            coding_ambiguous_count=0,
            coding_skipped_count=0,
        )

    entries: list[MedicalCodingEntry] = []
    attempted_count = 0
    success_count = 0
    unmapped_count = 0
    ambiguous_count = 0
    skipped_count = 0

    for entity in entities:
        entity_type = str(entity.get("type", "unknown")).strip().lower()
        original_text = str(entity.get("text", "")).strip()
        normalized_text = _normalize_text(original_text)

        if entity_type not in SUPPORTED_ENTITY_TYPES or not normalized_text:
            skipped_count += 1
            entries.append(
                MedicalCodingEntry(
                    original_entity_text=original_text,
                    normalized_entity_text=normalized_text,
                    entity_type=entity_type,
                    coding_system=None,
                    code=None,
                    code_display=None,
                    coding_confidence=0.0,
                    coding_source=source,
                    coding_status="skipped",
                )
            )
            continue

        attempted_count += 1
        ambiguous_seed = AMBIGUOUS_SEED_MAP.get((entity_type, normalized_text))
        if ambiguous_seed is not None:
            ambiguous_count += 1
            entries.append(
                MedicalCodingEntry(
                    original_entity_text=original_text,
                    normalized_entity_text=normalized_text,
                    entity_type=entity_type,
                    coding_system=str(ambiguous_seed["coding_system"]),
                    code=None,
                    code_display=str(ambiguous_seed["code_display"]),
                    coding_confidence=0.5,
                    coding_source=source,
                    coding_status="ambiguous",
                )
            )
            continue

        seed_map = DIAGNOSIS_SEED_MAP if entity_type == "diagnosis" else MEDICATION_SEED_MAP
        seed_entry = seed_map.get(normalized_text)
        if seed_entry is None:
            unmapped_count += 1
            entries.append(
                MedicalCodingEntry(
                    original_entity_text=original_text,
                    normalized_entity_text=normalized_text,
                    entity_type=entity_type,
                    coding_system=None,
                    code=None,
                    code_display=None,
                    coding_confidence=0.0,
                    coding_source=source,
                    coding_status="unmapped",
                )
            )
            continue

        success_count += 1
        entries.append(
            MedicalCodingEntry(
                original_entity_text=original_text,
                normalized_entity_text=normalized_text,
                entity_type=entity_type,
                coding_system=str(seed_entry["coding_system"]),
                code=str(seed_entry["code"]),
                code_display=str(seed_entry["code_display"]),
                coding_confidence=0.95,
                coding_source=source,
                coding_status="coded",
            )
        )

    return MedicalCodingResult(
        applied=bool(entries),
        coding_source=source,
        entries=entries,
        coding_attempted_count=attempted_count,
        coding_success_count=success_count,
        coding_unmapped_count=unmapped_count,
        coding_ambiguous_count=ambiguous_count,
        coding_skipped_count=skipped_count,
    )


def _normalize_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    normalized = PUNCT_RE.sub(" ", lowered)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized
