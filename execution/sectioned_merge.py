"""Local section merge validation for 17C-R2-R13.

Merges validated section JSON into the canonical review-bound package shape
without weakening schema or synthesizing clinical values.
"""
from __future__ import annotations

import json
from typing import Any

from execution.sectioned_extraction import SECTION_NAMES

TOP_LEVEL_KEYS = (
    "extracted_labs",
    "extracted_diagnoses",
    "extracted_medications",
    "needs_review",
    "source_evidence",
    "extraction_warnings",
)


def merge_sections(section_payloads: dict[str, dict[str, Any]]) -> tuple[bool, str, dict[str, Any] | None]:
    missing = [name for name in SECTION_NAMES if name not in section_payloads]
    if missing:
        return False, "missing_required_sections", None
    warnings: list[Any] = []
    evidence: list[Any] = []
    labs: list[Any] = []
    diagnoses: list[Any] = []
    medications: list[Any] = []
    for section, payload in section_payloads.items():
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return False, "section_items_not_list", None
        if section == "lab_results_and_measurements":
            labs.extend(items)
        elif section == "diagnoses_assessments_impressions":
            diagnoses.extend(items)
        elif section == "medications_treatments_orders":
            medications.extend(items)
        else:
            evidence.extend({"section": section, "item": item} for item in items)
        warnings.extend(payload.get("warnings") or [])
    merged = {
        "extracted_labs": _dedupe_exact(labs),
        "extracted_diagnoses": _dedupe_exact(diagnoses),
        "extracted_medications": _dedupe_exact(medications),
        "needs_review": True,
        "source_evidence": _dedupe_exact(evidence),
        "extraction_warnings": _dedupe_exact(warnings),
    }
    missing_top = [key for key in TOP_LEVEL_KEYS if key not in merged]
    if missing_top:
        return False, "missing_required_top_level_keys", None
    return True, "ok", merged


def _dedupe_exact(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


__all__ = ["TOP_LEVEL_KEYS", "merge_sections"]
