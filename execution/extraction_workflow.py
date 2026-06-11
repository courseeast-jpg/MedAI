"""Safe AI extraction workflow seam for 15A.

This orchestrator accepts local text/OCR layout summaries, passes through a
privacy-gate placeholder, calls only the fake local adapter, validates the
returned schema, and returns review-bound source packages. It never writes
active MKB records and never calls external APIs.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.source_extraction_packages import source_package_from_ai_draft
from execution.ai_extraction_adapter import (
    AIExtractionAdapterInput,
    AIExtractionPackageDraft,
    ExtractionWorkflowContext,
    ExtractionWorkflowResult,
    FakeAIExtractionAdapter,
)


PLACEHOLDER_TOKENS = (
    "[PATIENT_1]",
    "[DOB_1]",
    "[DATE_1]",
    "[MRN_1]",
    "[ACCESSION_1]",
    "[FACILITY_1]",
    "[PROVIDER_1]",
    "[ADDRESS_1]",
    "[PHONE_1]",
    "[EMAIL_1]",
    "[INSURANCE_ID_1]",
)


def run_ai_extraction_workflow(
    context: ExtractionWorkflowContext,
    *,
    adapter: Any | None = None,
) -> ExtractionWorkflowResult:
    adapter = adapter or FakeAIExtractionAdapter()
    privacy = _privacy_gate_placeholder(context)
    adapter_input = AIExtractionAdapterInput(
        source_class=context.source_class,
        safe_source_document_id=context.safe_source_document_id,
        selected_document_category=context.selected_document_category,
        selected_specialty_domain=context.selected_specialty_domain,
        source_modality=context.source_modality,
        text_character_bucket=context.text_character_bucket,
        line_count_bucket=context.line_count_bucket,
        layout_hints=dict(context.layout_hints or {}),
    )
    draft = adapter.extract(adapter_input)
    errors = validate_ai_package_draft(draft)
    packages = [] if errors else [source_package_from_ai_draft(draft)]
    operator_preview = build_operator_preview(packages)
    return ExtractionWorkflowResult(
        adapter_name=str(getattr(adapter, "adapter_name", adapter.__class__.__name__)),
        packages=packages,
        privacy_gate_status=privacy["privacy_gate_status"],
        pii_redaction_required=privacy["pii_redaction_required"],
        external_payload_allowed=privacy["external_payload_allowed"],
        external_payload_preview_available=privacy["external_payload_preview_available"],
        external_api_used=False,
        active_written_count=0,
        auto_accept=False,
        review_required=True,
        review_bound_package_count=len(packages),
        package_types_tested=[str(getattr(draft, "document_type", "Unknown"))],
        operator_preview=operator_preview,
        validation_errors=errors,
    )


def validate_ai_package_draft(draft: AIExtractionPackageDraft) -> list[str]:
    errors: list[str] = []
    if getattr(draft, "external_api_used", False):
        errors.append("external_api_used_must_be_false")
    if getattr(draft, "auto_accept", True) is not False:
        errors.append("auto_accept_must_be_false")
    if getattr(draft, "review_required", False) is not True:
        errors.append("review_required_must_be_true")
    if str(getattr(draft, "package_status", "")) != "review-bound":
        errors.append("package_status_must_be_review_bound")
    if not list(getattr(draft, "sections", []) or []):
        errors.append("sections_required")
    for section in list(getattr(draft, "sections", []) or []):
        if not str(getattr(section, "heading", "") or "").strip():
            errors.append("section_heading_required")
        for observation in list(getattr(section, "observations", []) or []):
            if getattr(observation, "auto_accept", True) is not False:
                errors.append("observation_auto_accept_must_be_false")
            if getattr(observation, "review_required", False) is not True:
                errors.append("observation_review_required_must_be_true")
    return sorted(set(errors))


def build_operator_preview(packages: list[dict[str, Any]]) -> dict[str, Any]:
    preview_packages: list[dict[str, Any]] = []
    for package in packages:
        preview_packages.append(
            {
                "package_id": package["package_id"],
                "label": "AI-assisted extraction draft - review-bound only",
                "document_type": package["detected_document_family_type"],
                "section_count": len(package["sections"]),
                "observation_count": sum(len(section["observations"]) for section in package["sections"]),
                "sections": [
                    {
                        "heading": section["heading"],
                        "narrative_label": section["narrative_label"],
                        "observations": [
                            {
                                "label": observation["label"],
                                "value": observation["value"],
                                "flag": observation["flag"],
                                "unit": observation["unit"],
                                "reference_interval": observation["reference_interval"],
                                "review_status": observation["review_status"],
                            }
                            for observation in section["observations"]
                        ],
                    }
                    for section in package["sections"]
                ],
            }
        )
    return {
        "visible": bool(preview_packages),
        "packages": preview_packages,
        "review_required": True,
        "auto_accept": False,
    }


def workflow_result_to_public_dict(result: ExtractionWorkflowResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["external_api_used"] = False
    payload["active_written_count"] = 0
    payload["auto_accept"] = False
    payload["review_required"] = True
    return payload


def _privacy_gate_placeholder(context: ExtractionWorkflowContext) -> dict[str, Any]:
    return {
        "privacy_gate_status": "placeholder_passed_fake_local_only",
        "pii_redaction_required": False,
        "external_payload_allowed": bool(False if context.fake_local_only else False),
        "external_payload_preview_available": False,
        "placeholder_tokens_supported": list(PLACEHOLDER_TOKENS),
    }


__all__ = [
    "PLACEHOLDER_TOKENS",
    "run_ai_extraction_workflow",
    "validate_ai_package_draft",
    "build_operator_preview",
    "workflow_result_to_public_dict",
]
