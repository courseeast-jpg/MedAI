"""Read-only operator panel for the Vertex semantic review-decision audit (15T).

Builds a compact, deterministic panel view-model over the 15S read-only
audit/export layer (which reads the isolated 15R decision store). It is strictly
read-only: it never mutates the 15R store, never writes active MKB records,
never sets auto_accept=true, performs no provider/network call, and uses no
live gate. Export affordances point at the already-generated 15S artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from execution.vertex_semantic_review_decision_audit_export import (
    PROVIDER_MODEL,
    PROVIDER_ROUTE,
    build_audit_export,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PANEL_TITLE = "Vertex Semantic Review Decision Audit"
NO_LIVE_READ_ONLY_INDICATOR = "Read-only audit. No live provider call. No active MKB records are created."
READ_ONLY_EXPORT_STATEMENT = "Exports are read-only. No active MKB records are created."

# Already-generated 15S export artifacts (read-only references).
_S15_DIR = "reports/medai_vertex_semantic_review_decision_audit_export_15s"
EXPORT_AFFORDANCES = {
    "json": f"{_S15_DIR}/decision_audit_export.json",
    "csv": f"{_S15_DIR}/decision_audit_export.csv",
    "markdown": f"{_S15_DIR}/decision_audit_summary.md",
}


def build_audit_panel_view_model() -> dict[str, Any]:
    export = build_audit_export()
    s = export.summary
    records = export.json_export

    decision_rows = [
        {
            "package_family": r.get("package_family", ""),
            "decision_status": r.get("decision_status", ""),
            "action": r.get("action", ""),
            "evidence_anchor": r.get("evidence_anchor", ""),
            "source_evidence_text": r.get("source_evidence_text", ""),
            "provider_route": r.get("provider_route", ""),
            "provider_model": r.get("provider_model", ""),
            "source_report_reference": r.get("source_report_reference", ""),
            "audit_reason": r.get("audit_reason", ""),
            "unknown_values": list(r.get("unknown_values") or []),
            "uncertainty_flag_count": len(r.get("uncertainty_flags") or []),
            "hallucinated_field_count": int(r.get("hallucinated_field_count") or 0),
            "review_required": bool(r.get("review_required", True)),
            "auto_accept": bool(r.get("auto_accept", False)),
            "active_written_count_delta": int(r.get("active_written_count_delta") or 0),
        }
        for r in records
    ]

    export_paths_present = {
        kind: (REPO_ROOT / path).exists() for kind, path in EXPORT_AFFORDANCES.items()
    }

    view_model = {
        "title": PANEL_TITLE,
        "no_live_read_only_indicator": NO_LIVE_READ_ONLY_INDICATOR,
        "read_only_export_statement": READ_ONLY_EXPORT_STATEMENT,
        "provider_summary": {
            "provider_route": PROVIDER_ROUTE,
            "provider_model": PROVIDER_MODEL,
        },
        "decision_summary": {
            "total_decisions": s["decision_records_loaded"],
            "accepted_for_review": s["accepted_for_review_count"],
            "rejected": s["rejected_count"],
            "deferred": s["deferred_count"],
        },
        "package_family_breakdown": dict(s["package_family_breakdown"]),
        "safety_summary": {
            "active_written_count": s["active_written_count"],
            "active_mkb_record_created_count": s["active_mkb_record_created_count"],
            "auto_accept": False,
            "review_required": True,
            "hallucinated_field_count": s["hallucinated_field_count"],
            "export_read_only": True,
        },
        "decision_rows": decision_rows,
        "export_affordances": dict(EXPORT_AFFORDANCES),
        "export_affordances_present": export_paths_present,
        "source_summary": s,
    }
    return view_model


def render_audit_panel_preview_markdown(view_model: dict[str, Any] | None = None) -> str:
    vm = view_model if view_model is not None else build_audit_panel_view_model()
    ds = vm["decision_summary"]
    safety = vm["safety_summary"]
    lines = [
        f"# {vm['title']}",
        "",
        f"_{vm['no_live_read_only_indicator']}_",
        "",
        f"- Provider: route=`{vm['provider_summary']['provider_route']}`, model=`{vm['provider_summary']['provider_model']}`",
        f"- Total decisions: **{ds['total_decisions']}** | "
        f"Accepted for review: **{ds['accepted_for_review']}** | "
        f"Rejected: **{ds['rejected']}** | Deferred: **{ds['deferred']}**",
        "",
        "## Package-family breakdown",
        "",
        *[f"- {family}: {count}" for family, count in vm["package_family_breakdown"].items()],
        "",
        "## Safety status",
        "",
        f"- active_written_count: **{safety['active_written_count']}**",
        f"- active_mkb_record_created_count: **{safety['active_mkb_record_created_count']}**",
        f"- auto_accept: **{safety['auto_accept']}**",
        f"- review_required: **{safety['review_required']}**",
        f"- hallucinated_field_count: **{safety['hallucinated_field_count']}**",
        "",
        "## Decisions (evidence & provenance)",
        "",
        "| Family | Status | Anchor | Source evidence | Route/Model | Source report | Audit reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            "| {fam} | {st} | {anc} | {ev} | {rt}/{md} | {src} | {reason} |".format(
                fam=row["package_family"],
                st=row["decision_status"],
                anc=row["evidence_anchor"],
                ev=row["source_evidence_text"],
                rt=row["provider_route"],
                md=row["provider_model"],
                src=row["source_report_reference"],
                reason=row["audit_reason"],
            )
            for row in vm["decision_rows"]
        ],
        "",
        "## Exports (read-only)",
        "",
        *[f"- {kind.upper()}: `{path}`" for kind, path in vm["export_affordances"].items()],
        "",
        f"_{vm['read_only_export_statement']}_",
        "",
    ]
    return "\n".join(lines)


def render_vertex_semantic_review_decision_audit_panel() -> None:
    """Compact read-only Streamlit panel (lazy st import; safe if unavailable)."""
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - streamlit optional at import time
        return
    vm = build_audit_panel_view_model()
    ds = vm["decision_summary"]
    safety = vm["safety_summary"]
    st.markdown(f"#### {vm['title']}")
    st.caption(vm["no_live_read_only_indicator"])
    st.caption(
        f"Provider: {vm['provider_summary']['provider_route']} / {vm['provider_summary']['provider_model']}"
    )
    st.caption(
        f"Total: {ds['total_decisions']} | Accepted for review: {ds['accepted_for_review']} | "
        f"Rejected: {ds['rejected']} | Deferred: {ds['deferred']}"
    )
    st.caption(
        "Per family: " + ", ".join(f"{fam}={count}" for fam, count in vm["package_family_breakdown"].items())
    )
    st.caption(
        f"Active writes: {safety['active_written_count']} | Active MKB records: {safety['active_mkb_record_created_count']} | "
        f"Auto-accept: {safety['auto_accept']} | Review required: {safety['review_required']} | "
        f"Hallucinated fields: {safety['hallucinated_field_count']}"
    )
    if vm["decision_rows"]:
        st.dataframe(
            [
                {
                    "family": row["package_family"],
                    "status": row["decision_status"],
                    "anchor": row["evidence_anchor"],
                    "source_evidence": row["source_evidence_text"],
                    "route": row["provider_route"],
                    "model": row["provider_model"],
                    "source_report": row["source_report_reference"],
                    "audit_reason": row["audit_reason"],
                    "uncertainty_flags": row["uncertainty_flag_count"],
                }
                for row in vm["decision_rows"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    for kind, path in vm["export_affordances"].items():
        full = REPO_ROOT / path
        if full.exists():
            try:
                st.download_button(
                    label=f"Download {kind.upper()} export (read-only)",
                    data=full.read_bytes(),
                    file_name=full.name,
                    key=f"vertex_semantic_audit_export_{kind}",
                )
            except Exception:  # pragma: no cover - defensive
                st.caption(f"{kind.upper()} export available at: {path}")
    st.caption(vm["read_only_export_statement"])


__all__ = [
    "PANEL_TITLE",
    "NO_LIVE_READ_ONLY_INDICATOR",
    "READ_ONLY_EXPORT_STATEMENT",
    "EXPORT_AFFORDANCES",
    "build_audit_panel_view_model",
    "render_audit_panel_preview_markdown",
    "render_vertex_semantic_review_decision_audit_panel",
]
