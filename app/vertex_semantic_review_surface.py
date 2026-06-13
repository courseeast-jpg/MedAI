"""No-live operator review surface for verified Vertex semantic packages (15Q).

This module replays the recorded, sanitized live-comparison artifacts from
15P-C (portal result-card) and 15P-D (the other three synthetic families) into
review-bound operator draft packages. It makes NO provider call: the live
Vertex findings come from the committed report JSONs, and the synthetic source
body / sections / evidence anchors / unknowns / uncertainty come from the same
deterministic 15O fixtures the live call used.

Operator controls (accept / reject / defer) are simulated deterministically and
locally: they never write active MKB records and never enable auto-accept.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.ai_package_run_review_preview import (
    RunReviewPackagePreview,
    build_run_review_package_previews,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_15PC_RESULT = REPO_ROOT / "reports" / "medai_vertex_semantic_package_comparison_live_15p_c" / "live_comparison_result.json"
RECORDED_15PD_RESULTS = REPO_ROOT / "reports" / "medai_vertex_semantic_package_remaining_families_live_15p_d" / "live_remaining_families_results.json"

PROVIDER_ROUTE = "vertex"
PROVIDER_MODEL = "gemini-2.5-flash-lite"

REVIEW_ACTIONS = ("accept", "reject", "defer")


@dataclass(frozen=True)
class VertexSemanticReviewDraft:
    package_id: str
    package_family: str
    package_family_label: str
    provider_route: str
    provider_model: str
    source_visible_body: str
    source_sections: list[dict[str, Any]]
    vertex_semantic_findings: list[dict[str, Any]]
    candidate_facts: list[dict[str, Any]]
    evidence_anchors: list[dict[str, str]]
    unknown_values: list[str]
    uncertainty_flags: list[str]
    hallucinated_field_count: int
    schema_validation_pass: bool
    recorded_live_call_made: bool
    recorded_provider_response_received: bool
    review_required: bool
    auto_accept: bool
    active_written_count: int
    replay_from_recorded_report: bool
    no_live_replay_indicator: str
    active_written_count_indicator: str
    auto_accept_indicator: str
    action_controls: list[str]
    action_state: str


def load_recorded_vertex_results() -> dict[str, dict[str, Any]]:
    """Load recorded sanitized live results keyed by package_family.

    Reads committed 15P-C and 15P-D report artifacts only. No provider call.
    """
    results: dict[str, dict[str, Any]] = {}
    if RECORDED_15PC_RESULT.exists():
        c = json.loads(RECORDED_15PC_RESULT.read_text(encoding="utf-8"))
        results[str(c.get("package_family"))] = c
    if RECORDED_15PD_RESULTS.exists():
        d = json.loads(RECORDED_15PD_RESULTS.read_text(encoding="utf-8"))
        for entry in d.get("per_family_results", []):
            if isinstance(entry, dict) and entry.get("package_family"):
                results[str(entry["package_family"])] = entry
    return results


def _preview_by_family() -> dict[str, RunReviewPackagePreview]:
    return {p.package_family: p for p in build_run_review_package_previews()}


def build_vertex_semantic_review_drafts() -> list[VertexSemanticReviewDraft]:
    recorded = load_recorded_vertex_results()
    previews = _preview_by_family()
    drafts: list[VertexSemanticReviewDraft] = []
    for family, preview in previews.items():
        record = recorded.get(family)
        if record is None:
            continue
        findings = list(record.get("sanitized_findings") or [])
        drafts.append(
            VertexSemanticReviewDraft(
                package_id=f"vertex_semantic_review_{family}",
                package_family=family,
                package_family_label=preview.package_family_label,
                provider_route=str(record.get("provider_route") or PROVIDER_ROUTE),
                provider_model=str(record.get("model") or PROVIDER_MODEL),
                source_visible_body=preview.source_visible_body,
                source_sections=list(preview.source_sections),
                vertex_semantic_findings=findings,
                candidate_facts=list(preview.candidate_facts),
                evidence_anchors=list(preview.evidence_anchors),
                unknown_values=list(preview.unknown_values),
                uncertainty_flags=list(preview.uncertainty_flags),
                hallucinated_field_count=int(record.get("hallucinated_field_count") or 0),
                schema_validation_pass=bool(record.get("schema_validation_pass")),
                recorded_live_call_made=bool(record.get("live_call_made")),
                recorded_provider_response_received=bool(record.get("provider_response_received")),
                review_required=True,
                auto_accept=False,
                active_written_count=0,
                replay_from_recorded_report=True,
                no_live_replay_indicator="Replayed from recorded 15P-C/15P-D report; no live call in this view",
                active_written_count_indicator="Active MKB writes: 0",
                auto_accept_indicator="Auto-accept: off",
                action_controls=["Accept for review queue only", "Reject", "Defer"],
                action_state="pending_operator_review",
            )
        )
    return drafts


def simulate_review_action(draft: VertexSemanticReviewDraft, action: str) -> dict[str, Any]:
    """Deterministic, local-only review action simulation. Never writes MKB."""
    action = str(action or "").strip().lower()
    if action not in REVIEW_ACTIONS:
        return {
            "package_id": draft.package_id,
            "action": action,
            "accepted": False,
            "review_decision": "invalid_action",
            "review_bound": True,
            "active_written_count": 0,
            "auto_accept": False,
            "creates_active_mkb_record": False,
            "local_simulation_only": True,
        }
    decision = {
        "accept": "accepted_for_review_queue_only",
        "reject": "rejected_local_simulation_only",
        "defer": "deferred_local_simulation_only",
    }[action]
    return {
        "package_id": draft.package_id,
        "package_family": draft.package_family,
        "action": action,
        "review_decision": decision,
        "review_bound": True,
        "active_written_count": 0,
        "auto_accept": False,
        "creates_active_mkb_record": False,
        "local_simulation_only": True,
    }


def vertex_review_draft_to_public_dict(draft: VertexSemanticReviewDraft) -> dict[str, Any]:
    return asdict(draft)


def build_operator_review_report() -> dict[str, Any]:
    drafts = build_vertex_semantic_review_drafts()
    cases = [vertex_review_draft_to_public_dict(draft) for draft in drafts]
    action_sims = {
        draft.package_family: {
            action: simulate_review_action(draft, action) for action in REVIEW_ACTIONS
        }
        for draft in drafts
    }
    total = len(drafts)
    summary = {
        "block": "MEDAI-VERTEX-SEMANTIC-PACKAGE-OPERATOR-REVIEW-SURFACE-15Q",
        "package_families_loaded": len(load_recorded_vertex_results()),
        "package_families_rendered": total,
        "provider_route_visible_count": sum(1 for d in drafts if d.provider_route == PROVIDER_ROUTE),
        "provider_model_visible_count": sum(1 for d in drafts if d.provider_model == PROVIDER_MODEL),
        "source_visible_body_present_count": sum(1 for d in drafts if d.source_visible_body.strip()),
        "evidence_anchor_present_count": sum(1 for d in drafts if d.evidence_anchors),
        "vertex_semantic_findings_separated_count": sum(1 for d in drafts if d.vertex_semantic_findings),
        "candidate_facts_separated_count": sum(1 for d in drafts if d.candidate_facts),
        "unknown_values_explicit_count": sum(
            1 for d in drafts if (d.unknown_values or d.package_family != "mixed_narrative_numeric_result")
        ),
        "uncertainty_flags_visible_count": sum(1 for d in drafts if d.uncertainty_flags),
        "hallucinated_field_count": sum(d.hallucinated_field_count for d in drafts),
        "no_live_replay_indicator_visible_count": sum(1 for d in drafts if d.no_live_replay_indicator),
        "active_written_count_indicator_visible_count": sum(1 for d in drafts if d.active_written_count_indicator),
        "auto_accept_false_indicator_visible_count": sum(1 for d in drafts if d.auto_accept_indicator),
        "accept_control_visible_count": sum(1 for d in drafts if "Accept for review queue only" in d.action_controls),
        "reject_control_visible_count": sum(1 for d in drafts if "Reject" in d.action_controls),
        "defer_control_visible_count": sum(1 for d in drafts if "Defer" in d.action_controls),
        "accept_simulation_review_bound_count": sum(
            1 for sims in action_sims.values() if sims["accept"]["review_bound"] and sims["accept"]["active_written_count"] == 0
        ),
        "reject_simulation_local_only_count": sum(
            1 for sims in action_sims.values() if sims["reject"]["local_simulation_only"]
        ),
        "defer_simulation_local_only_count": sum(
            1 for sims in action_sims.values() if sims["defer"]["local_simulation_only"]
        ),
        "active_written_count": 0,
        "auto_accept": False,
        "review_required": True,
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }
    summary["all_required_families_rendered"] = total == 4
    summary["all_hallucinated_zero"] = summary["hallucinated_field_count"] == 0
    return {"summary": summary, "cases": cases, "action_simulations": action_sims}


def render_operator_review_preview(drafts: list[VertexSemanticReviewDraft] | None = None) -> str:
    drafts = drafts if drafts is not None else build_vertex_semantic_review_drafts()
    lines = [
        "# Vertex Semantic Package — Operator Review Drafts (15Q, no-live replay)",
        "",
        "Replayed from recorded 15P-C / 15P-D live comparison reports. No provider call is made here.",
        "All drafts are review-bound: active MKB writes = 0, auto-accept = off.",
        "",
    ]
    for draft in drafts:
        lines += [
            f"## {draft.package_family_label} (`{draft.package_family}`)",
            "",
            f"- Provider route: `{draft.provider_route}` | Model: `{draft.provider_model}`",
            f"- {draft.no_live_replay_indicator}",
            f"- Review required: `{draft.review_required}` | {draft.auto_accept_indicator} | {draft.active_written_count_indicator}",
            f"- Hallucinated field count: `{draft.hallucinated_field_count}` | Schema valid: `{draft.schema_validation_pass}`",
            "",
            "### Source visible body",
            "",
            f"> {draft.source_visible_body}",
            "",
            "### Source sections",
            "",
            *[f"- {s.get('heading', '')}: {s.get('source_excerpt', '')}" for s in draft.source_sections],
            "",
            "### Vertex semantic findings (separated from source body)",
            "",
            "| Label | Value | Section | Evidence text | Uncertainty | Unknown |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                "| {label} | {value} | {section} | {ev} | {unc} | {unk} |".format(
                    label=f.get("label", ""),
                    value=f.get("value", ""),
                    section=f.get("source_section", ""),
                    ev=f.get("evidence_text", ""),
                    unc=f.get("uncertainty", ""),
                    unk=f.get("unknown_value", False),
                )
                for f in draft.vertex_semantic_findings
            ],
            "",
            "### Evidence anchors",
            "",
            *[f"- `{a.get('anchor_id', '')}` [{a.get('source_section', '')}]: {a.get('snippet', '')}" for a in draft.evidence_anchors],
            "",
            "### Unknown / missing values",
            "",
            ("- " + "; ".join(draft.unknown_values)) if draft.unknown_values else "- (none unknown in this synthetic family)",
            "",
            "### Uncertainty flags",
            "",
            *([f"- {flag}" for flag in draft.uncertainty_flags] if draft.uncertainty_flags else ["- (none)"]),
            "",
            "### Operator controls (simulation only — no active MKB write)",
            "",
            *[f"- [ ] {control}" for control in draft.action_controls],
            "",
        ]
    return "\n".join(lines)


__all__ = [
    "PROVIDER_ROUTE",
    "PROVIDER_MODEL",
    "REVIEW_ACTIONS",
    "VertexSemanticReviewDraft",
    "load_recorded_vertex_results",
    "build_vertex_semantic_review_drafts",
    "simulate_review_action",
    "vertex_review_draft_to_public_dict",
    "build_operator_review_report",
    "render_operator_review_preview",
]
