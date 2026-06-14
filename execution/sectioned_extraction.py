"""Sectioned extraction helpers for 17C-R2-R13 autonomous recovery.

Local helpers only. They build compact JSON-only Vertex payloads, classify strict
JSON failures, and define the bounded section ladder. They do not call a provider
and do not write private payloads into public reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from execution.strict_json import normalize_one_json_object

SECTION_NAMES = (
    "document_identity_and_metadata",
    "clinical_findings",
    "lab_results_and_measurements",
    "diagnoses_assessments_impressions",
    "medications_treatments_orders",
    "procedures_imaging_pathology",
    "followup_recommendations_and_unknowns",
)

SECTION_OUTPUT_KEYS = ("section", "items", "needs_review", "warnings")

TRANSIENT_PROVIDER_CATEGORIES = {
    "timeout",
    "rate_limit",
    "unavailable",
    "connection_reset",
    "temporarily_unavailable",
}

RECOVERABLE_JSON_CATEGORIES = {
    "truncated_or_invalid_json",
    "multiple_json_objects_or_trailing_text",
    "empty_response",
    "missing_expected_schema_fields",
    "missing_section_schema_fields",
}


@dataclass(frozen=True)
class ExtractionStrategy:
    name: str
    max_output_tokens: int
    temperature: int = 0


FULL_SCHEMA_STRATEGY = ExtractionStrategy("compact_full_schema", 8192)
SECTION_STRATEGY = ExtractionStrategy("sectioned_extraction", 2048)
SUBSECTION_STRATEGY = ExtractionStrategy("adaptive_section_window", 1024)
SKELETON_STRATEGY = ExtractionStrategy("schema_skeleton_retry", 2048)


def build_full_payload(prompt_contract: str, tokenized_content: str, *,
                       max_output_tokens: int = FULL_SCHEMA_STRATEGY.max_output_tokens) -> dict[str, Any]:
    prompt = (
        f"{prompt_contract.strip()}\n\n"
        "# DOCUMENT (tokenized)\n"
        f"{tokenized_content}\n\n"
        "# OUTPUT\n"
        "Return one compact JSON object only. No markdown. No prose. Preserve token placeholders. "
        "Use empty arrays for absent lists and null only where the schema permits it."
    )
    return _payload(prompt, max_output_tokens)


def build_section_payload(section: str, tokenized_content: str, *,
                          window_index: int | None = None,
                          window_count: int | None = None,
                          max_output_tokens: int = SECTION_STRATEGY.max_output_tokens,
                          skeleton_retry: bool = False) -> dict[str, Any]:
    if section not in SECTION_NAMES:
        raise ValueError(f"unknown_section:{section}")
    window_note = ""
    if window_index is not None and window_count is not None:
        window_note = f" Window {window_index + 1} of {window_count}; extract only evidence in this window."
    skeleton = ""
    if skeleton_retry:
        skeleton = (
            "\nRequired skeleton exactly: "
            '{"section":"%s","items":[],"needs_review":true,"warnings":[]}' % section
        )
    prompt = (
        "You are extracting a single section from a tokenized medical document for review-bound "
        "MedAI staging. Do not infer, diagnose, or synthesize missing clinical values. "
        "Return strict compact JSON object only with keys: section, items, needs_review, warnings. "
        "Use empty arrays when absent. Use short evidence anchors only, not long copied passages. "
        f"Section: {section}.{window_note}{skeleton}\n\n"
        "# DOCUMENT WINDOW (tokenized)\n"
        f"{tokenized_content}"
    )
    return _payload(prompt, max_output_tokens)


def split_tokenized_window(tokenized_content: str, max_chars: int = 6000) -> list[str]:
    text = str(tokenized_content or "")
    if len(text) <= max_chars:
        return [text]
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            pivot = text.rfind("\n", start, end)
            if pivot > start + max_chars // 2:
                end = pivot
        windows.append(text[start:end])
        start = end
    return windows


def validate_section_response(text: str, section: str) -> tuple[bool, str, dict[str, Any] | None]:
    obj, reason = normalize_one_json_object(text)
    if obj is None:
        return False, reason, None
    missing = [key for key in SECTION_OUTPUT_KEYS if key not in obj]
    if missing:
        return False, "missing_section_schema_fields", None
    if str(obj.get("section")) != section:
        return False, "section_name_mismatch", None
    if not isinstance(obj.get("items"), list) or not isinstance(obj.get("warnings"), list):
        return False, "section_schema_type_mismatch", None
    return True, "ok", obj


def salvage_one_complete_json_object(text: str) -> tuple[bool, dict[str, Any] | None, str]:
    """Recover only an exact complete JSON object with no inference.

    This intentionally rejects partial/truncated JSON and prose-wrapped JSON with
    extra text. It is a local parse convenience, not clinical repair.
    """
    obj, reason = normalize_one_json_object(text)
    if obj is None:
        return False, None, reason
    try:
        json.dumps(obj, sort_keys=True)
    except (TypeError, ValueError):
        return False, None, "not_json_serializable"
    return True, obj, "ok"


def is_transient_provider_category(category: str) -> bool:
    return str(category or "").lower() in TRANSIENT_PROVIDER_CATEGORIES


def _payload(prompt: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
        },
    }


__all__ = [
    "SECTION_NAMES",
    "FULL_SCHEMA_STRATEGY",
    "SECTION_STRATEGY",
    "SUBSECTION_STRATEGY",
    "SKELETON_STRATEGY",
    "build_full_payload",
    "build_section_payload",
    "split_tokenized_window",
    "validate_section_response",
    "salvage_one_complete_json_object",
    "is_transient_provider_category",
]
