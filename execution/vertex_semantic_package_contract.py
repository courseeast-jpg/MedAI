"""No-live Vertex semantic package contract for 15P-A.

This module defines the request/response contract MedAI can use for a later
Vertex semantic extraction route. It only uses synthetic 15O fixtures and fake
provider responses; it performs no network calls, OCR routing, MKB writes, or
review transitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from app.ai_package_run_review_preview import (
    RunReviewPackagePreview,
    build_run_review_package_previews,
    run_review_preview_to_public_dict,
)
from execution.gemini_vertex_adapter import (
    VERTEX_ENDPOINT_HOST,
    VERTEX_LOCATION,
    VERTEX_MODEL,
    VERTEX_PROVIDER_NAME,
    VERTEX_PROVIDER_ROUTE,
)


VERTEX_SEMANTIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["package_family", "semantic_findings", "review_required", "auto_accept"],
    "properties": {
        "package_family": {"type": "string"},
        "semantic_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "label",
                    "value",
                    "source_section",
                    "evidence_text",
                    "uncertainty",
                    "unknown_value",
                    "source_faithful",
                ],
            },
        },
        "review_required": {"type": "boolean"},
        "auto_accept": {"type": "boolean"},
    },
}

PROMPT_REQUIRED_PHRASES = {
    "json_only": "Return JSON only.",
    "source_faithful": "Extract only facts visible in the synthetic source body and candidate list.",
    "no_clinical_advice": "Do not provide diagnosis, treatment advice, clinical recommendations, or interpretation.",
    "evidence_required": "Every candidate fact must include exact source evidence text from the supplied synthetic source.",
    "unknowns": "Keep unknown or missing values unknown; do not infer values.",
    "uncertainty": "Set uncertainty when the source wording is uncertain or narrative-only.",
    "no_hallucinated_fields": "Do not add fields, values, sections, dates, identifiers, or facts not present in the source.",
    # 15X-R1 hardening: evidence must be a verbatim source span, never paraphrased.
    "evidence_verbatim": (
        "Copy evidence_text verbatim from the synthetic source body; do not paraphrase, "
        "reword, summarize, synthesize, or infer evidence_text. If no exact source span "
        "supports a finding, set evidence_text to null and set uncertainty with review_required true."
    ),
}

FORBIDDEN_PROMPT_MARKERS = (
    "GEMINI" + "_API_KEY",
    "Authorization",
    "Bearer ",
    "ya29.",
    "AIza",
    "token_map",
    "DOB",
    "MRN",
    "Accession",
    "C:\\",
    ".pdf",
    ".png",
    ".jpg",
    "raw OCR",
    "raw" + "_pdf",
    "raw_image",
)


@dataclass(frozen=True)
class VertexSemanticContractCase:
    package_family: str
    package_family_label: str
    source_visible_body: str
    local_candidate_facts: list[dict[str, Any]]
    request_payload: dict[str, Any]
    expected_response_schema: dict[str, Any]
    fake_vertex_response: dict[str, Any]
    schema_validation_passed: bool
    final_review_package: dict[str, Any]
    prompt_instruction_flags: dict[str, bool]
    source_visible_body_preserved: bool
    evidence_anchor_preserved: bool
    candidate_facts_separated: bool
    unknown_values_explicit: bool
    uncertainty_flags_visible: bool
    hallucinated_field_count: int
    under_1_minute_compare_preserved: bool
    live_call_made: bool
    external_api_used: bool
    active_written_count: int
    auto_accept: bool
    review_required: bool


def build_vertex_semantic_prompt(preview: RunReviewPackagePreview) -> str:
    facts = "\n".join(
        "- {label}: value={value}; section={section}; evidence={evidence}; uncertainty={uncertainty}".format(
            label=fact["label"],
            value=fact["value"],
            section=fact["source_section"],
            evidence=fact["evidence_snippet"],
            uncertainty=fact["uncertainty"],
        )
        for fact in preview.candidate_facts
    )
    anchors = "\n".join(
        f"- {anchor['anchor_id']}: {anchor['source_section']} | {anchor['snippet']}"
        for anchor in preview.evidence_anchors
    )
    required = "\n".join(f"- {phrase}" for phrase in PROMPT_REQUIRED_PHRASES.values())
    return "\n".join(
        [
            "MedAI Vertex semantic extraction contract.",
            required,
            "Output keys: package_family, semantic_findings, review_required, auto_accept.",
            "semantic_findings keys: label, value, source_section, evidence_text, uncertainty, unknown_value, source_faithful.",
            "Set review_required=true and auto_accept=false.",
            "",
            f"Synthetic package family: {preview.package_family}",
            f"Synthetic source body: {preview.source_visible_body}",
            "Evidence anchors:",
            anchors,
            "Local deterministic candidate facts:",
            facts,
        ]
    )


def build_vertex_semantic_request_payload(preview: RunReviewPackagePreview) -> dict[str, Any]:
    prompt = build_vertex_semantic_prompt(preview)
    return {
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "provider_name": VERTEX_PROVIDER_NAME,
        "model": VERTEX_MODEL,
        "location": VERTEX_LOCATION,
        "endpoint_host": VERTEX_ENDPOINT_HOST,
        "live_call_made": False,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }


def build_fake_vertex_semantic_response(preview: RunReviewPackagePreview) -> dict[str, Any]:
    return {
        "package_family": preview.package_family,
        "semantic_findings": [
            {
                "label": str(fact["label"]),
                "value": str(fact["value"]),
                "source_section": str(fact["source_section"]),
                "evidence_text": str(fact["evidence_snippet"]),
                "uncertainty": str(fact["uncertainty"]),
                "unknown_value": bool(fact["unknown_value"]),
                "source_faithful": True,
            }
            for fact in preview.candidate_facts
        ],
        "review_required": True,
        "auto_accept": False,
    }


def validate_vertex_semantic_response(
    response: Mapping[str, Any],
    preview: RunReviewPackagePreview,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if response.get("package_family") != preview.package_family:
        errors.append("package_family_mismatch")
    if response.get("review_required") is not True:
        errors.append("review_required_not_true")
    if response.get("auto_accept") is not False:
        errors.append("auto_accept_not_false")
    findings = response.get("semantic_findings")
    if not isinstance(findings, list) or not findings:
        errors.append("semantic_findings_missing")
        return False, errors
    allowed = {
        (str(fact["label"]), str(fact["source_section"]), str(fact["evidence_snippet"]))
        for fact in preview.candidate_facts
    }
    for index, item in enumerate(findings):
        if not isinstance(item, Mapping):
            errors.append(f"finding_{index}_not_object")
            continue
        missing = [
            key
            for key in (
                "label",
                "value",
                "source_section",
                "evidence_text",
                "uncertainty",
                "unknown_value",
                "source_faithful",
            )
            if key not in item
        ]
        if missing:
            errors.append(f"finding_{index}_missing_{'_'.join(missing)}")
            continue
        identity = (str(item["label"]), str(item["source_section"]), str(item["evidence_text"]))
        if identity not in allowed:
            errors.append(f"finding_{index}_not_source_anchored")
        if item.get("source_faithful") is not True:
            errors.append(f"finding_{index}_not_source_faithful")
    return not errors, errors


_UNICODE_NORMALIZE_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',  # smart quotes
    "–": "-", "—": "-", "−": "-",                  # en/em/minus dashes
    " ": " ", " ": " ", " ": " ", " ": " ",   # nbsp/thin spaces
}


def normalize_evidence_text(text: str) -> str:
    """Normalize ONLY harmless formatting: whitespace + common Unicode quote/dash
    variants. This never changes words and never accepts paraphrase."""
    import re as _re

    text = str(text or "")
    for raw, repl in _UNICODE_NORMALIZE_MAP.items():
        text = text.replace(raw, repl)
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def evidence_text_is_source_verbatim(
    evidence_text: Any,
    source_text: str,
    *,
    section_text: str | None = None,
) -> bool:
    """Strict, deterministic literal source-anchor check (substring only).

    Returns True only if the (whitespace/Unicode-normalized) ``evidence_text`` is a
    non-empty verbatim substring of the normalized source body (and of the section
    text when a section constraint is supplied). Reworded, paraphrased, or inferred
    text, or any text not literally present in the source, is rejected. No fuzzy or
    semantic scoring is used. A null/empty evidence_text is not a verbatim match
    (callers treat null as "no supporting span" + uncertainty, not a passing anchor).
    """
    if evidence_text is None:
        return False
    needle = normalize_evidence_text(str(evidence_text))
    if not needle:
        return False
    haystack = normalize_evidence_text(source_text)
    if needle not in haystack:
        return False
    if section_text is not None:
        return needle in normalize_evidence_text(section_text)
    return True


def evaluate_vertex_semantic_contract() -> dict[str, Any]:
    cases = [_evaluate_preview(preview) for preview in build_run_review_package_previews()]
    summary = {
        "package_families_checked": [case.package_family for case in cases],
        "package_family_count": len(cases),
        "prompt_json_only_instruction_present_count": sum(
            1 for case in cases if case.prompt_instruction_flags["json_only"]
        ),
        "source_faithful_instruction_present_count": sum(
            1 for case in cases if case.prompt_instruction_flags["source_faithful"]
        ),
        "no_clinical_advice_instruction_present_count": sum(
            1 for case in cases if case.prompt_instruction_flags["no_clinical_advice"]
        ),
        "evidence_required_instruction_present_count": sum(
            1 for case in cases if case.prompt_instruction_flags["evidence_required"]
        ),
        "unknowns_must_remain_unknown_instruction_present_count": sum(
            1 for case in cases if case.prompt_instruction_flags["unknowns"]
        ),
        "schema_validation_pass_count": sum(1 for case in cases if case.schema_validation_passed),
        "fake_vertex_response_valid_count": sum(1 for case in cases if case.schema_validation_passed),
        "source_visible_body_preserved_count": sum(1 for case in cases if case.source_visible_body_preserved),
        "evidence_anchor_preserved_count": sum(1 for case in cases if case.evidence_anchor_preserved),
        "candidate_facts_separated_count": sum(1 for case in cases if case.candidate_facts_separated),
        "unknown_values_explicit_count": sum(1 for case in cases if case.unknown_values_explicit),
        "uncertainty_flags_visible_count": sum(1 for case in cases if case.uncertainty_flags_visible),
        "hallucinated_field_count": sum(case.hallucinated_field_count for case in cases),
        "under_1_minute_compare_preserved_count": sum(
            1 for case in cases if case.under_1_minute_compare_preserved
        ),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
    }
    summary["all_contract_invariants_passed"] = (
        bool(cases)
        and summary["schema_validation_pass_count"] == len(cases)
        and summary["fake_vertex_response_valid_count"] == len(cases)
        and summary["source_visible_body_preserved_count"] == len(cases)
        and summary["evidence_anchor_preserved_count"] == len(cases)
        and summary["candidate_facts_separated_count"] == len(cases)
        and summary["unknown_values_explicit_count"] == len(cases)
        and summary["uncertainty_flags_visible_count"] == len(cases)
        and summary["under_1_minute_compare_preserved_count"] == len(cases)
        and summary["hallucinated_field_count"] == 0
        and summary["live_call_made"] is False
        and summary["external_api_used"] is False
        and summary["active_written_count"] == 0
        and summary["auto_accept"] is False
        and summary["review_required"] is True
    )
    return {
        "summary": summary,
        "cases": [vertex_semantic_case_to_public_dict(case) for case in cases],
    }


def vertex_semantic_case_to_public_dict(case: VertexSemanticContractCase) -> dict[str, Any]:
    data = asdict(case)
    data["request_payload"] = _public_payload_preview(case.request_payload)
    return data


def prompt_privacy_check(prompt: str) -> dict[str, Any]:
    hits = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker.lower() in prompt.lower()]
    return {
        "privacy_result": "passed" if not hits else "failed",
        "synthetic_redacted_payload_only": not hits,
        "forbidden_marker_hits": hits,
    }


def _evaluate_preview(preview: RunReviewPackagePreview) -> VertexSemanticContractCase:
    request_payload = build_vertex_semantic_request_payload(preview)
    prompt = str(request_payload["contents"][0]["parts"][0]["text"])
    prompt_flags = {
        key: phrase in prompt
        for key, phrase in PROMPT_REQUIRED_PHRASES.items()
    }
    fake_response = build_fake_vertex_semantic_response(preview)
    schema_valid, errors = validate_vertex_semantic_response(fake_response, preview)
    final_review_package = run_review_preview_to_public_dict(preview)
    final_review_package["semantic_enrichment"] = {
        "provider_route": VERTEX_PROVIDER_ROUTE,
        "provider_name": VERTEX_PROVIDER_NAME,
        "fake_response_schema_valid": schema_valid,
        "validation_errors": errors,
        "finding_count": len(fake_response["semantic_findings"]),
        "review_bound_only": True,
    }
    return VertexSemanticContractCase(
        package_family=preview.package_family,
        package_family_label=preview.package_family_label,
        source_visible_body=preview.source_visible_body,
        local_candidate_facts=list(preview.candidate_facts),
        request_payload=request_payload,
        expected_response_schema=VERTEX_SEMANTIC_RESPONSE_SCHEMA,
        fake_vertex_response=fake_response,
        schema_validation_passed=schema_valid,
        final_review_package=final_review_package,
        prompt_instruction_flags=prompt_flags,
        source_visible_body_preserved=final_review_package["source_visible_body"] == preview.source_visible_body,
        evidence_anchor_preserved=bool(final_review_package["evidence_anchors"]),
        candidate_facts_separated=bool(final_review_package["candidate_facts"]),
        unknown_values_explicit=bool(preview.unknown_values) or preview.package_family != "mixed_narrative_numeric_result",
        uncertainty_flags_visible=bool(final_review_package["uncertainty_flags"]),
        hallucinated_field_count=preview.hallucinated_field_count,
        under_1_minute_compare_preserved=preview.under_1_minute_compare_preserved,
        live_call_made=False,
        external_api_used=False,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
    )


def _public_payload_preview(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload["contents"][0]["parts"][0]["text"])
    return {
        "provider_route": payload["provider_route"],
        "provider_name": payload["provider_name"],
        "model": payload["model"],
        "location": payload["location"],
        "endpoint_host": payload["endpoint_host"],
        "live_call_made": payload["live_call_made"],
        "generationConfig": dict(payload["generationConfig"]),
        "prompt_char_count": len(prompt),
        "prompt_instruction_flags": {
            key: phrase in prompt
            for key, phrase in PROMPT_REQUIRED_PHRASES.items()
        },
        "prompt_privacy": prompt_privacy_check(prompt),
    }


__all__ = [
    "PROMPT_REQUIRED_PHRASES",
    "VERTEX_SEMANTIC_RESPONSE_SCHEMA",
    "VertexSemanticContractCase",
    "build_fake_vertex_semantic_response",
    "build_vertex_semantic_prompt",
    "build_vertex_semantic_request_payload",
    "evaluate_vertex_semantic_contract",
    "prompt_privacy_check",
    "validate_vertex_semantic_response",
    "vertex_semantic_case_to_public_dict",
    "normalize_evidence_text",
    "evidence_text_is_source_verbatim",
]
