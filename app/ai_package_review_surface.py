"""Operator-visible package review surface models for fake/local 15O-B.

This module converts the deterministic 15O-A quality fixtures into compact
review view models. It is importable without Streamlit and does not touch OCR,
provider execution, runtime databases, or review actions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.source_extraction_packages import source_package_from_ai_draft
from execution.ai_package_quality_eval import (
    PackageQualityFixture,
    build_package_quality_fixtures,
    evaluate_package_quality,
)


@dataclass(frozen=True)
class ReviewSurfaceSourceSection:
    heading: str
    source_excerpt: str
    candidate_fact_count: int
    evidence_anchor_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewSurfaceCandidateFact:
    label: str
    value: str
    source_section: str
    evidence_anchor_id: str
    evidence_snippet: str
    uncertainty: str
    unknown_value: bool
    row_kind: str


@dataclass(frozen=True)
class PackageReviewSurfaceViewModel:
    package_id: str
    package_family: str
    package_family_label: str
    source_visible_body: str
    source_sections: list[ReviewSurfaceSourceSection]
    candidate_facts: list[ReviewSurfaceCandidateFact]
    evidence_anchors: list[dict[str, str]]
    uncertainty_flags: list[str]
    unknown_values: list[str]
    review_required: bool
    auto_accept: bool
    active_written_count: int
    external_api_used: bool
    live_call_made: bool
    estimated_human_compare_seconds: int
    under_1_minute_compare_preserved: bool
    hallucinated_field_count: int


def build_all_review_surface_view_models() -> list[PackageReviewSurfaceViewModel]:
    return [build_review_surface_view_model(fixture) for fixture in build_package_quality_fixtures()]


def build_review_surface_view_model(fixture: PackageQualityFixture) -> PackageReviewSurfaceViewModel:
    package = source_package_from_ai_draft(fixture.draft)
    metrics = evaluate_package_quality(fixture, package)
    anchors_by_section = _anchors_by_section(fixture)
    facts: list[ReviewSurfaceCandidateFact] = []
    source_sections: list[ReviewSurfaceSourceSection] = []
    unknown_values: list[str] = []
    uncertainty_flags: list[str] = []

    for section in package.get("sections") or []:
        heading = str(section.get("heading") or "Unsectioned")
        section_anchors = anchors_by_section.get(heading, [])
        observations = list(section.get("observations") or [])
        source_sections.append(
            ReviewSurfaceSourceSection(
                heading=heading,
                source_excerpt=_source_excerpt_for_section(fixture, heading),
                candidate_fact_count=len(observations),
                evidence_anchor_ids=[anchor["anchor_id"] for anchor in section_anchors],
            )
        )
        default_anchor = section_anchors[0] if section_anchors else {"anchor_id": "", "snippet": ""}
        for obs in observations:
            label = str(obs.get("label") or "")
            value = str(obs.get("value") or "")
            unknown = value.lower() in {"unknown", "not visible", "not provided"}
            if unknown:
                unknown_values.append(label)
            if obs.get("row_kind") == "source_visible_note":
                uncertainty = "source narrative only; no MedAI interpretation"
            elif unknown:
                uncertainty = "missing or not visible in source"
            else:
                uncertainty = "source-visible candidate fact; operator must compare"
            uncertainty_flags.append(f"{label}: {uncertainty}")
            facts.append(
                ReviewSurfaceCandidateFact(
                    label=label,
                    value=value or "source-visible narrative present",
                    source_section=heading,
                    evidence_anchor_id=str(default_anchor.get("anchor_id") or ""),
                    evidence_snippet=str(default_anchor.get("snippet") or ""),
                    uncertainty=uncertainty,
                    unknown_value=unknown,
                    row_kind=str(obs.get("row_kind") or ""),
                )
            )

    return PackageReviewSurfaceViewModel(
        package_id=str(package.get("package_id") or ""),
        package_family=fixture.family_key,
        package_family_label=fixture.display_name,
        source_visible_body=fixture.source_visible_body,
        source_sections=source_sections,
        candidate_facts=facts,
        evidence_anchors=[asdict(anchor) for anchor in fixture.evidence_anchors],
        uncertainty_flags=uncertainty_flags,
        unknown_values=unknown_values,
        review_required=True,
        auto_accept=False,
        active_written_count=0,
        external_api_used=False,
        live_call_made=False,
        estimated_human_compare_seconds=metrics.estimated_human_compare_seconds,
        under_1_minute_compare_preserved=metrics.under_1_minute_compare_pass,
        hallucinated_field_count=metrics.hallucinated_field_count,
    )


def review_surface_to_public_dict(view_model: PackageReviewSurfaceViewModel) -> dict[str, Any]:
    return {
        "package_id": view_model.package_id,
        "package_family": view_model.package_family,
        "package_family_label": view_model.package_family_label,
        "source_visible_body": view_model.source_visible_body,
        "source_sections": [asdict(section) for section in view_model.source_sections],
        "candidate_facts": [asdict(fact) for fact in view_model.candidate_facts],
        "evidence_anchors": list(view_model.evidence_anchors),
        "uncertainty_flags": list(view_model.uncertainty_flags),
        "unknown_values": list(view_model.unknown_values),
        "review_required": view_model.review_required,
        "auto_accept": view_model.auto_accept,
        "active_written_count": view_model.active_written_count,
        "external_api_used": view_model.external_api_used,
        "live_call_made": view_model.live_call_made,
        "estimated_human_compare_seconds": view_model.estimated_human_compare_seconds,
        "under_1_minute_compare_preserved": view_model.under_1_minute_compare_preserved,
        "hallucinated_field_count": view_model.hallucinated_field_count,
    }


def render_review_surface_markdown(view_model: PackageReviewSurfaceViewModel) -> str:
    lines = [
        f"## {view_model.package_family_label}",
        "",
        f"- Package ID: `{view_model.package_id}`",
        f"- Review required: `{view_model.review_required}`",
        f"- Auto-accept: `{view_model.auto_accept}`",
        f"- Active written count: `{view_model.active_written_count}`",
        f"- External API used: `{view_model.external_api_used}`",
        f"- Live call made: `{view_model.live_call_made}`",
        f"- Compare estimate: `{view_model.estimated_human_compare_seconds}s`",
        "",
        "### Source Visible Body",
        "",
        view_model.source_visible_body,
        "",
        "### Source Sections",
        "",
    ]
    for section in view_model.source_sections:
        lines.extend(
            [
                f"- **{section.heading}**",
                f"  - Source excerpt: {section.source_excerpt}",
                f"  - Candidate facts: `{section.candidate_fact_count}`",
                f"  - Evidence anchors: `{section.evidence_anchor_ids}`",
            ]
        )
    lines.extend(["", "### Candidate Facts", ""])
    for fact in view_model.candidate_facts:
        lines.extend(
            [
                f"- **{fact.label}**",
                f"  - Value: `{fact.value}`",
                f"  - Source section: `{fact.source_section}`",
                f"  - Evidence: `{fact.evidence_anchor_id}` {fact.evidence_snippet}",
                f"  - Uncertainty: {fact.uncertainty}",
            ]
        )
    lines.extend(["", "### Unknown Values", ""])
    if view_model.unknown_values:
        lines.extend(f"- {item}" for item in view_model.unknown_values)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def build_review_surface_report() -> dict[str, Any]:
    view_models = build_all_review_surface_view_models()
    cases = [review_surface_to_public_dict(item) for item in view_models]
    summary = {
        "package_families_rendered": [item.package_family for item in view_models],
        "package_family_count": len(view_models),
        "source_visible_body_present_count": sum(1 for item in view_models if item.source_visible_body.strip()),
        "evidence_anchor_present_count": sum(1 for item in view_models if item.evidence_anchors),
        "candidate_facts_separated_count": sum(1 for item in view_models if item.candidate_facts),
        "unknown_values_explicit_count": sum(1 for item in view_models if item.unknown_values or item.package_family != "mixed_narrative_numeric_result"),
        "under_1_minute_compare_preserved_count": sum(1 for item in view_models if item.under_1_minute_compare_preserved),
        "hallucinated_field_count": sum(item.hallucinated_field_count for item in view_models),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
    }
    summary["all_review_surface_invariants_passed"] = all(
        item.review_required is True
        and item.auto_accept is False
        and item.active_written_count == 0
        and item.external_api_used is False
        and item.live_call_made is False
        and item.source_visible_body.strip()
        and item.evidence_anchors
        and item.candidate_facts
        for item in view_models
    )
    return {"summary": summary, "cases": cases, "markdown": [render_review_surface_markdown(item) for item in view_models]}


def _anchors_by_section(fixture: PackageQualityFixture) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for anchor in fixture.evidence_anchors:
        grouped.setdefault(anchor.source_section, []).append(asdict(anchor))
    return grouped


def _source_excerpt_for_section(fixture: PackageQualityFixture, heading: str) -> str:
    for anchor in fixture.evidence_anchors:
        if anchor.source_section == heading:
            return anchor.snippet
    return "source section visible in synthetic fixture"


__all__ = [
    "PackageReviewSurfaceViewModel",
    "ReviewSurfaceCandidateFact",
    "ReviewSurfaceSourceSection",
    "build_all_review_surface_view_models",
    "build_review_surface_report",
    "build_review_surface_view_model",
    "render_review_surface_markdown",
    "review_surface_to_public_dict",
]
