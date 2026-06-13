"""Fake/local AI package quality evaluation for 15O-A.

The evaluator scores whether a review package is easy for an operator to compare
against a source summary. It does not score clinical correctness and performs no
provider calls, OCR routing, MKB writes, or auto-accept transitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.source_extraction_packages import source_package_from_ai_draft
from execution.ai_extraction_adapter import (
    AIExtractionObservation,
    AIExtractionPackageDraft,
    AIExtractionSection,
)


PACKAGE_FAMILIES = (
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "portal_result_cards",
    "mixed_narrative_numeric_result",
)


@dataclass(frozen=True)
class SourceEvidenceAnchor:
    anchor_id: str
    source_section: str
    snippet: str


@dataclass(frozen=True)
class PackageQualityFixture:
    family_key: str
    display_name: str
    source_visible_body: str
    draft: AIExtractionPackageDraft
    evidence_anchors: list[SourceEvidenceAnchor]
    expected_source_fields: list[str]
    candidate_fact_labels: list[str]
    unknown_value_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PackageQualityMetrics:
    family_key: str
    display_name: str
    package_id: str
    source_visible_body_present: bool
    source_section_grouping_present: bool
    evidence_anchor_present: bool
    candidate_facts_separated: bool
    unknown_values_explicit: bool
    hallucinated_field_count: int
    review_required: bool
    active_written_count: int
    auto_accept: bool
    external_api_used: bool
    live_call_made: bool
    estimated_human_compare_seconds: int
    under_1_minute_compare_pass: bool
    section_count: int
    observation_count: int
    evidence_anchor_count: int


def build_package_quality_fixtures() -> list[PackageQualityFixture]:
    return [
        _cytology_pathology_fixture(),
        _urinalysis_fixture(),
        _portal_cards_fixture(),
        _mixed_fixture(),
    ]


def evaluate_all_package_quality() -> dict[str, Any]:
    cases = []
    for fixture in build_package_quality_fixtures():
        package = source_package_from_ai_draft(fixture.draft)
        metrics = evaluate_package_quality(fixture, package)
        cases.append(
            {
                "fixture": _fixture_public_dict(fixture),
                "package": _package_public_preview(package, fixture),
                "metrics": asdict(metrics),
            }
        )
    summary = {
        "package_families_evaluated": [case["metrics"]["family_key"] for case in cases],
        "package_family_count": len(cases),
        "under_1_minute_compare_pass_count": sum(
            1 for case in cases if case["metrics"]["under_1_minute_compare_pass"]
        ),
        "hallucinated_field_count": sum(case["metrics"]["hallucinated_field_count"] for case in cases),
        "source_visible_body_present_count": sum(
            1 for case in cases if case["metrics"]["source_visible_body_present"]
        ),
        "evidence_anchor_present_count": sum(1 for case in cases if case["metrics"]["evidence_anchor_present"]),
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "external_api_used": False,
        "live_call_made": False,
        "billing_check_pending": True,
    }
    summary["all_cases_under_1_minute"] = summary["under_1_minute_compare_pass_count"] == len(cases)
    summary["all_safety_invariants_passed"] = all(
        case["metrics"]["review_required"] is True
        and case["metrics"]["active_written_count"] == 0
        and case["metrics"]["auto_accept"] is False
        and case["metrics"]["external_api_used"] is False
        and case["metrics"]["live_call_made"] is False
        for case in cases
    )
    return {"summary": summary, "cases": cases}


def evaluate_package_quality(fixture: PackageQualityFixture, package: dict[str, Any]) -> PackageQualityMetrics:
    sections = list(package.get("sections") or [])
    observations = [obs for section in sections for obs in section.get("observations") or []]
    package_labels = {str(obs.get("label") or "").strip() for obs in observations}
    expected_labels = {label.strip() for label in fixture.expected_source_fields}
    hallucinated = sorted(label for label in package_labels if label and label not in expected_labels)
    source_sections = {str(anchor.source_section) for anchor in fixture.evidence_anchors}
    section_headings = {str(section.get("heading") or "") for section in sections}
    unknown_labels = set(fixture.unknown_value_labels)
    unknown_values_explicit = not unknown_labels or all(
        any(obs.get("label") == label and str(obs.get("value") or "").lower() in {"unknown", "not visible", "not provided"} for obs in observations)
        for label in unknown_labels
    )
    compare_seconds = _estimate_compare_seconds(
        source_char_count=len(fixture.source_visible_body),
        section_count=len(sections),
        observation_count=len(observations),
        anchor_count=len(fixture.evidence_anchors),
    )
    return PackageQualityMetrics(
        family_key=fixture.family_key,
        display_name=fixture.display_name,
        package_id=str(package.get("package_id") or ""),
        source_visible_body_present=bool(fixture.source_visible_body.strip()),
        source_section_grouping_present=bool(sections) and source_sections.issubset(section_headings),
        evidence_anchor_present=bool(fixture.evidence_anchors),
        candidate_facts_separated=all(obs.get("row_kind") in {"observation", "source_visible_note"} for obs in observations),
        unknown_values_explicit=unknown_values_explicit,
        hallucinated_field_count=len(hallucinated),
        review_required=bool(package.get("review_required")) is True,
        active_written_count=int(package.get("active_written_count") or 0),
        auto_accept=bool(package.get("auto_accept_allowed")),
        external_api_used=bool(getattr(fixture.draft, "external_api_used", False)),
        live_call_made=False,
        estimated_human_compare_seconds=compare_seconds,
        under_1_minute_compare_pass=compare_seconds <= 60,
        section_count=len(sections),
        observation_count=len(observations),
        evidence_anchor_count=len(fixture.evidence_anchors),
    )


def _estimate_compare_seconds(*, source_char_count: int, section_count: int, observation_count: int, anchor_count: int) -> int:
    base = 12
    source_scan = min(18, max(4, source_char_count // 80))
    section_cost = section_count * 4
    observation_cost = observation_count * 2
    anchor_credit = min(10, anchor_count * 2)
    return max(12, base + source_scan + section_cost + observation_cost - anchor_credit)


def _fixture_public_dict(fixture: PackageQualityFixture) -> dict[str, Any]:
    return {
        "family_key": fixture.family_key,
        "display_name": fixture.display_name,
        "source_visible_body_present": bool(fixture.source_visible_body),
        "source_visible_body_char_count": len(fixture.source_visible_body),
        "section_headings": [section.heading for section in fixture.draft.sections],
        "evidence_anchors": [asdict(anchor) for anchor in fixture.evidence_anchors],
        "expected_source_fields": list(fixture.expected_source_fields),
        "candidate_fact_labels": list(fixture.candidate_fact_labels),
        "unknown_value_labels": list(fixture.unknown_value_labels),
        "raw_source_body_in_report": False,
    }


def _package_public_preview(package: dict[str, Any], fixture: PackageQualityFixture) -> dict[str, Any]:
    return {
        "package_id": package.get("package_id"),
        "detected_document_family_type": package.get("detected_document_family_type"),
        "package_status": package.get("package_status"),
        "record_count": package.get("record_count"),
        "section_summaries": [
            {
                "heading": section.get("heading"),
                "observation_count": len(section.get("observations") or []),
                "narrative_preview_available": section.get("narrative_preview_available"),
                "narrative_label": section.get("narrative_label"),
            }
            for section in package.get("sections") or []
        ],
        "source_body_hash_only": f"fixture_{fixture.family_key}",
        "review_required": package.get("review_required"),
        "auto_accept_allowed": package.get("auto_accept_allowed"),
        "active_written_count": package.get("active_written_count"),
    }


def _draft(
    *,
    source_id: str,
    document_type: str,
    domain: str,
    sections: list[AIExtractionSection],
) -> AIExtractionPackageDraft:
    return AIExtractionPackageDraft(
        safe_source_document_id=source_id,
        document_type=document_type,
        selected_document_category="AI-assisted extraction",
        selected_specialty_domain=domain,
        source_modality="fake_local_text_fixture",
        sections=sections,
        review_required=True,
        auto_accept=False,
        external_api_used=False,
    )


def _obs(label: str, value: str, *, section: str, reference: str = "", flag: str = "", unit: str = "") -> AIExtractionObservation:
    return AIExtractionObservation(
        label=label,
        value=value,
        reference_interval=reference,
        flag=flag,
        unit=unit,
        source_section=section,
        review_required=True,
        auto_accept=False,
    )


def _note(label: str, *, section: str) -> AIExtractionObservation:
    return AIExtractionObservation(
        label=label,
        value="source-visible narrative present",
        source_section=section,
        row_kind="source_visible_note",
        review_required=True,
        auto_accept=False,
    )


def _cytology_pathology_fixture() -> PackageQualityFixture:
    sections = [
        AIExtractionSection("Tests Ordered", [_note("Tests Ordered", section="Tests Ordered")], "source-visible narrative present"),
        AIExtractionSection("Diagnosis", [_note("Diagnosis", section="Diagnosis")], "source-visible narrative present"),
        AIExtractionSection(
            "Recommendation",
            [_note("Recommendation", section="Recommendation")],
            "source-visible narrative present",
            "source text only - not MedAI recommendation",
        ),
    ]
    return PackageQualityFixture(
        family_key="cytology_pathology_narrative",
        display_name="Cytology/pathology narrative package",
        source_visible_body="Synthetic narrative fixture with tests ordered, diagnosis, and recommendation sections.",
        draft=_draft(source_id="source_fake_15oa_cytology", document_type="Cytology / pathology narrative", domain="urology", sections=sections),
        evidence_anchors=[
            SourceEvidenceAnchor("cyto_a1", "Tests Ordered", "tests ordered section present"),
            SourceEvidenceAnchor("cyto_a2", "Diagnosis", "diagnosis section present"),
            SourceEvidenceAnchor("cyto_a3", "Recommendation", "recommendation section present"),
        ],
        expected_source_fields=["Tests Ordered", "Diagnosis", "Recommendation"],
        candidate_fact_labels=["Tests Ordered", "Diagnosis", "Recommendation"],
    )


def _urinalysis_fixture() -> PackageQualityFixture:
    heading = "Urinalysis Table"
    rows = [
        _obs("Specific Gravity", "1.020", section=heading, reference="1.005-1.030"),
        _obs("pH", "7.5", section=heading, reference="5.0-7.5"),
        _obs("Occult Blood", "Trace", section=heading, reference="Negative", flag="abnormal"),
        _obs("RBC", "3-10", section=heading, reference="0-2", flag="abnormal", unit="/hpf"),
    ]
    return PackageQualityFixture(
        family_key="urinalysis_table_like_lab",
        display_name="Urinalysis/table-like lab package",
        source_visible_body="Synthetic table fixture with specific gravity, pH, occult blood, and RBC rows.",
        draft=_draft(
            source_id="source_fake_15oa_urinalysis",
            document_type="Urinalysis",
            domain="urology",
            sections=[AIExtractionSection(heading, rows)],
        ),
        evidence_anchors=[SourceEvidenceAnchor("ua_a1", heading, "table rows grouped under urinalysis table")],
        expected_source_fields=["Specific Gravity", "pH", "Occult Blood", "RBC"],
        candidate_fact_labels=["Specific Gravity", "pH", "Occult Blood", "RBC"],
    )


def _portal_cards_fixture() -> PackageQualityFixture:
    heading = "Portal Result Cards"
    rows = [
        _obs("Urine Color", "Orange", section=heading, reference="Yellow"),
        _obs("Appearance", "Clear", section=heading, reference="Clear"),
        _obs("Leukocyte Esterase", "Negative", section=heading, reference="Negative"),
        _obs("Protein", "Trace", section=heading, reference="Negative/Trace"),
    ]
    return PackageQualityFixture(
        family_key="portal_result_cards",
        display_name="Portal result-card package",
        source_visible_body="Synthetic portal cards fixture with card labels and visible reference text.",
        draft=_draft(
            source_id="source_fake_15oa_portal",
            document_type="Portal result cards",
            domain="urology",
            sections=[AIExtractionSection(heading, rows)],
        ),
        evidence_anchors=[SourceEvidenceAnchor("portal_a1", heading, "card labels grouped as portal result cards")],
        expected_source_fields=["Urine Color", "Appearance", "Leukocyte Esterase", "Protein"],
        candidate_fact_labels=["Urine Color", "Appearance", "Leukocyte Esterase", "Protein"],
    )


def _mixed_fixture() -> PackageQualityFixture:
    narrative = AIExtractionSection(
        "Clinical Note",
        [_note("Clinical Note", section="Clinical Note")],
        "source-visible narrative present",
    )
    results = AIExtractionSection(
        "Result Summary",
        [
            _obs("Culture Result", "No growth", section="Result Summary"),
            _obs("Collection Time", "Unknown", section="Result Summary"),
            _obs("Nitrite", "Negative", section="Result Summary", reference="Negative"),
        ],
    )
    return PackageQualityFixture(
        family_key="mixed_narrative_numeric_result",
        display_name="Mixed narrative + numeric result package",
        source_visible_body="Synthetic mixed fixture with a clinical note, visible culture result, unknown collection time, and nitrite result.",
        draft=_draft(
            source_id="source_fake_15oa_mixed",
            document_type="Mixed narrative and numeric result",
            domain="urology",
            sections=[narrative, results],
        ),
        evidence_anchors=[
            SourceEvidenceAnchor("mixed_a1", "Clinical Note", "narrative section present"),
            SourceEvidenceAnchor("mixed_a2", "Result Summary", "numeric/result section present"),
        ],
        expected_source_fields=["Clinical Note", "Culture Result", "Collection Time", "Nitrite"],
        candidate_fact_labels=["Culture Result", "Collection Time", "Nitrite"],
        unknown_value_labels=["Collection Time"],
    )


__all__ = [
    "PACKAGE_FAMILIES",
    "PackageQualityFixture",
    "PackageQualityMetrics",
    "SourceEvidenceAnchor",
    "build_package_quality_fixtures",
    "evaluate_all_package_quality",
    "evaluate_package_quality",
]
