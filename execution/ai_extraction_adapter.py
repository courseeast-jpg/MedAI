"""Provider-neutral AI extraction adapter seam.

15A intentionally ships only a deterministic fake adapter. It performs no
external calls and emits review-bound, source-facing package drafts for testing
the workflow boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AIExtractionObservation:
    label: str
    value: str = ""
    reference_interval: str = ""
    flag: str = ""
    unit: str = ""
    source_section: str = ""
    row_kind: str = "observation"
    review_required: bool = True
    auto_accept: bool = False


@dataclass(frozen=True)
class AIExtractionSection:
    heading: str
    observations: list[AIExtractionObservation] = field(default_factory=list)
    narrative_preview: str = ""
    narrative_label: str = "source text only - not MedAI interpretation"


@dataclass(frozen=True)
class AIExtractionPackageDraft:
    safe_source_document_id: str
    document_type: str
    selected_document_category: str
    selected_specialty_domain: str
    source_modality: str
    sections: list[AIExtractionSection]
    package_status: str = "review-bound"
    review_required: bool = True
    auto_accept: bool = False
    external_api_used: bool = False


@dataclass(frozen=True)
class AIExtractionAdapterInput:
    source_class: str
    safe_source_document_id: str
    selected_document_category: str
    selected_specialty_domain: str
    source_modality: str
    text_character_bucket: str = "unknown"
    line_count_bucket: str = "unknown"
    layout_hints: dict[str, bool] = field(default_factory=dict)


class AIExtractionAdapter(Protocol):
    adapter_name: str

    def extract(self, adapter_input: AIExtractionAdapterInput) -> AIExtractionPackageDraft:
        """Return a schema-valid package draft without external side effects."""


@dataclass(frozen=True)
class ExtractionWorkflowContext:
    source_class: str
    safe_source_document_id: str = "source_fake_001"
    selected_document_category: str = "General source document"
    selected_specialty_domain: str = "general"
    source_modality: str = "local_text"
    text_character_bucket: str = "1-500"
    line_count_bucket: str = "1-20"
    layout_hints: dict[str, bool] = field(default_factory=dict)
    fake_local_only: bool = True


@dataclass(frozen=True)
class ExtractionWorkflowResult:
    adapter_name: str
    packages: list[dict]
    privacy_gate_status: str
    pii_redaction_required: bool
    external_payload_allowed: bool
    external_payload_preview_available: bool
    external_api_used: bool
    active_written_count: int
    auto_accept: bool
    review_required: bool
    review_bound_package_count: int
    package_types_tested: list[str]
    operator_preview: dict
    validation_errors: list[str] = field(default_factory=list)


class FakeAIExtractionAdapter:
    """Deterministic local-only adapter used to prove the 15A seam."""

    adapter_name = "FakeAIExtractionAdapter"

    def extract(self, adapter_input: AIExtractionAdapterInput) -> AIExtractionPackageDraft:
        source_class = adapter_input.source_class.strip().lower()
        if source_class in {"cytology", "pathology", "cytology_pathology"}:
            return self._cytology_pathology(adapter_input)
        if source_class in {"urinalysis", "urinalysis_table"}:
            return self._urinalysis_table(adapter_input)
        if source_class in {"portal_cards", "portal_result_cards"}:
            return self._portal_cards(adapter_input)
        return AIExtractionPackageDraft(
            safe_source_document_id=adapter_input.safe_source_document_id,
            document_type="Unknown",
            selected_document_category=adapter_input.selected_document_category,
            selected_specialty_domain=adapter_input.selected_specialty_domain,
            source_modality=adapter_input.source_modality,
            sections=[],
        )

    def _cytology_pathology(self, adapter_input: AIExtractionAdapterInput) -> AIExtractionPackageDraft:
        headings = [
            "Tests Ordered",
            "Diagnosis",
            "Recommendation",
            "Clinical History / ICD",
            "Cytology Information",
            "Gross Description",
        ]
        sections = [
            AIExtractionSection(
                heading=heading,
                observations=[
                    AIExtractionObservation(
                        label=heading,
                        value="source-visible narrative present",
                        source_section=heading,
                        row_kind="source_visible_note",
                    )
                ],
                narrative_preview="source-visible narrative present",
                narrative_label=(
                    "source text only - not MedAI recommendation"
                    if heading == "Recommendation"
                    else "source text only - not MedAI interpretation"
                ),
            )
            for heading in headings
        ]
        return self._draft(adapter_input, "Cytology / pathology narrative", sections)

    def _urinalysis_table(self, adapter_input: AIExtractionAdapterInput) -> AIExtractionPackageDraft:
        rows = [
            ("Specific Gravity", "1.020", "", "", "1.005-1.030"),
            ("pH", "7.5", "", "", "5.0-7.5"),
            ("Occult Blood", "Trace", "abnormal", "", "Negative"),
            ("RBC", "3-10", "abnormal", "/hpf", "0-2"),
            ("Culture result", "No growth", "", "", ""),
        ]
        observations = [
            AIExtractionObservation(
                label=label,
                value=value,
                flag=flag,
                unit=unit,
                reference_interval=reference,
                source_section="Urinalysis Table",
            )
            for label, value, flag, unit, reference in rows
        ]
        return self._draft(
            adapter_input,
            "Urinalysis",
            [AIExtractionSection(heading="Urinalysis Table", observations=observations)],
        )

    def _portal_cards(self, adapter_input: AIExtractionAdapterInput) -> AIExtractionPackageDraft:
        rows = [
            ("Specific Gravity", "1.020", "1.005-1.030"),
            ("pH", "7.5", "5.0-7.5"),
            ("Urine Color", "Orange", "Yellow"),
            ("Appearance", "Clear", "Clear"),
            ("Leukocyte Esterase", "Negative", "Negative"),
            ("Protein", "Trace", "Negative/Trace"),
            ("Glucose", "Negative", "Negative"),
            ("Ketones", "Negative", "Negative"),
        ]
        observations = [
            AIExtractionObservation(
                label=label,
                value=value,
                reference_interval=reference,
                source_section="Portal Result Cards",
            )
            for label, value, reference in rows
        ]
        return self._draft(
            adapter_input,
            "Portal result cards",
            [AIExtractionSection(heading="Portal Result Cards", observations=observations)],
        )

    def _draft(
        self,
        adapter_input: AIExtractionAdapterInput,
        document_type: str,
        sections: list[AIExtractionSection],
    ) -> AIExtractionPackageDraft:
        return AIExtractionPackageDraft(
            safe_source_document_id=adapter_input.safe_source_document_id,
            document_type=document_type,
            selected_document_category=adapter_input.selected_document_category,
            selected_specialty_domain=adapter_input.selected_specialty_domain,
            source_modality=adapter_input.source_modality,
            sections=sections,
        )


__all__ = [
    "AIExtractionAdapter",
    "FakeAIExtractionAdapter",
    "AIExtractionAdapterInput",
    "AIExtractionPackageDraft",
    "AIExtractionSection",
    "AIExtractionObservation",
    "ExtractionWorkflowContext",
    "ExtractionWorkflowResult",
]
