"""Deterministic supplemental extraction rules for local fallback paths."""

from __future__ import annotations

import re
from typing import Any


LAB_FINDINGS: tuple[tuple[str, str], ...] = (
    (r"\bcalcium\s+oxalate\s+crystals?\b", "Calcium Oxalate Crystals"),
    (r"\bepithelial\s+cells?\b", "Epithelial Cells"),
    (r"\bblood\s+ua\b|\bua\s+blood\b", "Blood UA"),
    (r"\bbilirubin\s+ua\b|\bua\s+bilirubin\b", "Bilirubin UA"),
    (r"\burobilinogen\s+ua\b|\bua\s+urobilinogen\b", "Urobilinogen UA"),
    (r"\bnitrite\s+ua\b|\bua\s+nitrite\b", "Nitrite UA"),
    (r"\bleukocyte(?:s| esterase)?\b", "Leukocyte"),
    (r"\bnitrite\b", "Nitrite"),
    (r"\bketones?\b", "Ketones"),
    (r"\bprotein\b", "Protein"),
    (r"\bglucose\b", "Glucose"),
    (r"\bbacteria\b", "Bacteria"),
    (r"\bcrystals?\b", "Crystals"),
    (r"\bblood\b", "Blood"),
    (r"\brbc\b", "RBC"),
    (r"\bwbc\b", "WBC"),
)

REPORT_FINDINGS: tuple[tuple[str, str, str], ...] = (
    (r"\burine\s+culture(?:,\s*routine)?\b", "test_result", "Urine Culture"),
    (r"\burine\s+cytology\b|\bcytology,\s*urine\b", "test_result", "Urine Cytology"),
    (r"\bno\s+growth\b", "test_result", "No Growth"),
    (r"\bnegative\s+for\s+high-grade\s+urothelial\s+carcinoma\b", "diagnosis", "Negative for high-grade urothelial carcinoma"),
    (r"\bbenign\s+urothelial\s+cells?\b", "test_result", "Benign Urothelial Cells"),
    (r"\bred\s+blood\s+cells?\b", "test_result", "Red Blood Cells"),
    (r"\bfinal\s+report\b", "note", "Final Report"),
    (r"\bdiagnosis\s*:", "diagnosis", "Diagnosis"),
    (r"\brecommendation\s*:", "recommendation", "Recommendation"),
    (r"\bgross\s+description\s*:", "note", "Gross Description"),
    (r"\bmicroscopic\s+examination\b", "test_result", "Microscopic Examination"),
)


def apply_supplemental_rules(result: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(result)
    entities = [dict(entity) for entity in enriched.get("entities", []) if isinstance(entity, dict)]
    supplemental = supplemental_entities(str(enriched.get("raw_text") or ""), existing_entities=entities)
    entities.extend(supplemental)
    enriched["entities"] = _dedupe_entities(entities)
    enriched["supplemental_rules_applied"] = bool(supplemental)
    enriched["supplemental_entity_count"] = len(supplemental)
    enriched["final_entity_count_after_supplement"] = len(enriched["entities"])
    if supplemental:
        enriched["notes"] = list(enriched.get("notes", [])) + ["supplemental_rules_applied"]
    return enriched


def supplemental_entities(text: str, *, existing_entities: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    existing_keys = {
        _entity_key(entity)
        for entity in (existing_entities or [])
        if isinstance(entity, dict)
    }
    additions: list[dict[str, Any]] = []

    def add(entity_type: str, label: str) -> None:
        entity = {
            "type": entity_type,
            "text": label,
            "source": "supplemental_lab_rules",
            "rule_pack": "urology_cytology_culture",
        }
        key = _entity_key(entity)
        if key in existing_keys:
            return
        existing_keys.add(key)
        additions.append(entity)

    for pattern, entity_type, label in REPORT_FINDINGS:
        if re.search(pattern, text, re.IGNORECASE):
            add(entity_type, label)

    for pattern, label in LAB_FINDINGS:
        if re.search(pattern, text, re.IGNORECASE):
            add("test_result", label)

    return additions


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for entity in entities:
        key = _entity_key(entity)
        if not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return (
        str(entity.get("type", "")).strip().lower(),
        str(entity.get("text", "")).strip().lower(),
    )
