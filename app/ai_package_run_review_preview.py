"""No-live Run & Review package preview wiring for 15O-C.

The preview adapts the 15O-B package review surface into a Run & Review shaped
view model. It uses deterministic fake/local fixtures only and never calls
providers, OCR, runtime databases, or review transition code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.ai_package_review_surface import (
    PackageReviewSurfaceViewModel,
    build_all_review_surface_view_models,
    review_surface_to_public_dict,
)


@dataclass(frozen=True)
class RunReviewPackagePreview:
    entry_point_label: str
    package_id: str
    package_family: str
    package_family_label: str
    source_visible_body: str
    source_sections: list[dict[str, Any]]
    candidate_facts: list[dict[str, Any]]
    evidence_anchors: list[dict[str, str]]
    unknown_values: list[str]
    uncertainty_flags: list[str]
    no_live_provider_off_indicator: str
    active_written_count_indicator: str
    auto_accept_indicator: str
    review_required: bool
    auto_accept: bool
    active_written_count: int
    external_api_used: bool
    live_call_made: bool
    estimated_human_compare_seconds: int
    under_1_minute_compare_preserved: bool
    hallucinated_field_count: int


def build_run_review_package_previews() -> list[RunReviewPackagePreview]:
    return [_preview_from_surface(surface) for surface in build_all_review_surface_view_models()]


def build_run_review_preview_report() -> dict[str, Any]:
    previews = build_run_review_package_previews()
    cases = [run_review_preview_to_public_dict(item) for item in previews]
    summary = {
        "run_review_package_preview_entry_point_added": True,
        "entry_point_label": "AI package review preview",
        "package_families_previewed": [item.package_family for item in previews],
        "package_family_count": len(previews),
        "source_visible_body_present_count": sum(1 for item in previews if item.source_visible_body.strip()),
        "evidence_anchor_present_count": sum(1 for item in previews if item.evidence_anchors),
        "candidate_facts_separated_count": sum(1 for item in previews if item.candidate_facts),
        "unknown_values_explicit_count": sum(
            1
            for item in previews
            if item.unknown_values or item.package_family != "mixed_narrative_numeric_result"
        ),
        "uncertainty_flags_visible_count": sum(1 for item in previews if item.uncertainty_flags),
        "under_1_minute_compare_preserved_count": sum(
            1 for item in previews if item.under_1_minute_compare_preserved
        ),
        "hallucinated_field_count": sum(item.hallucinated_field_count for item in previews),
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "billing_check_pending": True,
    }
    summary["all_run_review_preview_invariants_passed"] = all(
        item.review_required is True
        and item.auto_accept is False
        and item.active_written_count == 0
        and item.external_api_used is False
        and item.live_call_made is False
        and item.source_visible_body.strip()
        and item.evidence_anchors
        and item.candidate_facts
        and item.uncertainty_flags
        for item in previews
    )
    return {"summary": summary, "cases": cases}


def run_review_preview_to_public_dict(preview: RunReviewPackagePreview) -> dict[str, Any]:
    return asdict(preview)


def render_run_review_preview_markdown(preview: RunReviewPackagePreview) -> str:
    lines = [
        f"## {preview.entry_point_label}: {preview.package_family_label}",
        "",
        f"- Review required: `{preview.review_required}`",
        f"- Auto-accept: `{preview.auto_accept}`",
        f"- Active written count: `{preview.active_written_count}`",
        f"- External API used: `{preview.external_api_used}`",
        f"- Live call made: `{preview.live_call_made}`",
        f"- Provider status: {preview.no_live_provider_off_indicator}",
        f"- Compare estimate: `{preview.estimated_human_compare_seconds}s`",
        "",
        "### Source Visible Body",
        "",
        preview.source_visible_body,
        "",
        "### Source Sections",
        "",
    ]
    for section in preview.source_sections:
        lines.extend(
            [
                f"- **{section['heading']}**",
                f"  - Source excerpt: {section['source_excerpt']}",
                f"  - Candidate facts: `{section['candidate_fact_count']}`",
                f"  - Evidence anchors: `{section['evidence_anchor_ids']}`",
            ]
        )
    lines.extend(["", "### Candidate Facts", ""])
    for fact in preview.candidate_facts:
        lines.extend(
            [
                f"- **{fact['label']}**",
                f"  - Value: `{fact['value']}`",
                f"  - Source section: `{fact['source_section']}`",
                f"  - Evidence: `{fact['evidence_anchor_id']}` {fact['evidence_snippet']}",
                f"  - Uncertainty: {fact['uncertainty']}",
            ]
        )
    lines.extend(["", "### Unknown / Missing Values", ""])
    lines.extend(f"- {item}" for item in preview.unknown_values) if preview.unknown_values else lines.append("- None")
    lines.extend(["", "### Uncertainty Flags", ""])
    lines.extend(f"- {item}" for item in preview.uncertainty_flags)
    lines.append("")
    return "\n".join(lines)


def render_ai_package_run_review_preview_panel() -> None:
    """Render a compact Streamlit preview panel for the Run & Review tab."""
    try:
        import streamlit as st  # type: ignore[import-untyped]
    except ImportError:
        return

    previews = build_run_review_package_previews()
    st.markdown("#### AI package review preview")
    st.caption("Deterministic fake/local package preview. No provider call, no active write, no auto-accept.")
    if not previews:
        st.info("No AI package preview fixtures available.")
        return
    labels = [item.package_family_label for item in previews]
    selected_label = st.selectbox("Package family preview", labels, key="ai_package_review_preview_family")
    preview = next(item for item in previews if item.package_family_label == selected_label)
    status_cols = st.columns(4)
    status_cols[0].metric("Review required", str(preview.review_required))
    status_cols[1].metric("Auto-accept", str(preview.auto_accept))
    status_cols[2].metric("Active writes", preview.active_written_count)
    status_cols[3].metric("Provider", "off")
    st.markdown("**Source visible body**")
    st.write(preview.source_visible_body)
    st.markdown("**Source sections**")
    st.dataframe(preview.source_sections, hide_index=True, use_container_width=True)
    st.markdown("**Candidate facts**")
    st.dataframe(preview.candidate_facts, hide_index=True, use_container_width=True)
    st.markdown("**Unknown / missing values**")
    st.write(", ".join(preview.unknown_values) if preview.unknown_values else "None")
    st.markdown("**Uncertainty flags**")
    for flag in preview.uncertainty_flags:
        st.caption(flag)


def _preview_from_surface(surface: PackageReviewSurfaceViewModel) -> RunReviewPackagePreview:
    public = review_surface_to_public_dict(surface)
    return RunReviewPackagePreview(
        entry_point_label="AI package review preview",
        package_id=surface.package_id,
        package_family=surface.package_family,
        package_family_label=surface.package_family_label,
        source_visible_body=surface.source_visible_body,
        source_sections=list(public["source_sections"]),
        candidate_facts=list(public["candidate_facts"]),
        evidence_anchors=list(public["evidence_anchors"]),
        unknown_values=list(surface.unknown_values),
        uncertainty_flags=list(surface.uncertainty_flags),
        no_live_provider_off_indicator="no-live/provider-off",
        active_written_count_indicator="active_written_count=0",
        auto_accept_indicator="auto_accept=false",
        review_required=True,
        auto_accept=False,
        active_written_count=0,
        external_api_used=False,
        live_call_made=False,
        estimated_human_compare_seconds=surface.estimated_human_compare_seconds,
        under_1_minute_compare_preserved=surface.under_1_minute_compare_preserved,
        hallucinated_field_count=surface.hallucinated_field_count,
    )


__all__ = [
    "RunReviewPackagePreview",
    "build_run_review_package_previews",
    "build_run_review_preview_report",
    "render_ai_package_run_review_preview_panel",
    "render_run_review_preview_markdown",
    "run_review_preview_to_public_dict",
]
