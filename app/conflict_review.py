"""
MedAI — Streamlit UI for conflict resolution (Phase 5).

Call ``render_conflict_review(resolver)`` from any Streamlit page to display
pending conflicts and let the user resolve them.
"""
from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover — UI is optional at import time
    st = None  # type: ignore


_SEVERITY_COLORS = {
    "critical": "#d7263d",
    "high":     "#f46036",
    "medium":   "#1b998b",
    "low":      "#2e86ab",
}

_SEVERITY_ICONS = {
    "critical": "🚨",
    "high":     "🔶",
    "medium":   "🔷",
    "low":      "🔹",
}


def render_conflict_review(resolver) -> None:
    """Render the conflict review panel.

    :param resolver: a ``ConflictResolver`` instance.
    """
    if st is None:
        raise RuntimeError("Streamlit is not installed; cannot render UI.")

    st.header("Conflict Review")
    st.caption("Review and resolve facts the system could not merge automatically.")

    conflicts = resolver.list_pending()
    if not conflicts:
        st.success("No pending conflicts. MKB is consistent.")
        return

    # Severity filter.
    severities = ["all"] + sorted({c["severity"] for c in conflicts})
    selected = st.selectbox("Filter by severity", severities, index=0)
    if selected != "all":
        conflicts = [c for c in conflicts if c["severity"] == selected]

    st.caption(f"{len(conflicts)} pending conflict(s)")
    st.divider()

    for conflict in conflicts:
        _render_single(resolver, conflict)


# ── helpers ──────────────────────────────────────────────────────────────────

def _render_single(resolver, conflict: dict[str, Any]) -> None:
    severity = conflict.get("severity", "medium")
    color = _SEVERITY_COLORS.get(severity, "#888888")
    icon = _SEVERITY_ICONS.get(severity, "•")

    with st.container(border=True):
        st.markdown(
            f"<div style='border-left: 6px solid {color}; padding-left: 8px'>"
            f"<strong>{icon} {severity.upper()}</strong> · "
            f"<em>{conflict.get('conflict_type', 'unknown')}</em></div>",
            unsafe_allow_html=True,
        )
        st.caption(conflict.get("reason") or "")

        col1, col2 = st.columns(2, gap="medium")
        f1 = conflict.get("fact1_snapshot") or {}
        f2 = conflict.get("fact2_snapshot") or {}

        with col1:
            st.markdown("**Fact 1**")
            _fact_block(f1)
        with col2:
            st.markdown("**Fact 2**")
            _fact_block(f2)

        st.divider()

        with st.form(key=f"resolve_{conflict['id']}"):
            choice = st.radio(
                "Resolution",
                options=["fact1", "fact2", "both", "merge", "neither"],
                format_func=_choice_label,
                horizontal=True,
                key=f"choice_{conflict['id']}",
            )
            merged_value = st.text_input(
                "Merged value (only if merging)",
                key=f"merge_val_{conflict['id']}",
            ) if choice == "merge" else None
            notes = st.text_area(
                "Reasoning / notes",
                key=f"notes_{conflict['id']}",
                placeholder="Why did you choose this resolution?",
            )
            submitted = st.form_submit_button("Resolve", type="primary")

        if submitted:
            try:
                resolver.resolve_conflict(
                    conflict["id"],
                    {
                        "choice": choice,
                        "merged_value": merged_value,
                        "reasoning": notes,
                        "notes": notes,
                    },
                )
                st.success(f"Conflict resolved as '{choice}'.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not resolve conflict: {exc}")


def _fact_block(fact: dict[str, Any]) -> None:
    name = fact.get("entity_name") or fact.get("content") or "(no name)"
    value = fact.get("value")
    unit = fact.get("unit")
    date_ = fact.get("date")
    source = fact.get("source") or fact.get("source_name") or "unknown"
    confidence = fact.get("confidence")
    occurrences = fact.get("occurrence_count") or fact.get("provenance")

    st.markdown(f"**{name}**")
    if value is not None:
        st.caption(f"Value: {value} {unit or ''}".strip())
    if date_:
        st.caption(f"Date: {date_}")
    st.caption(f"Source: {source}")
    if confidence is not None:
        st.caption(f"Confidence: {confidence:.0%}" if isinstance(confidence, float)
                   else f"Confidence: {confidence}")
    if occurrences:
        st.caption(f"Provenance / occurrences: {occurrences}")


def _choice_label(choice: str) -> str:
    return {
        "fact1":   "Keep Fact 1",
        "fact2":   "Keep Fact 2",
        "both":    "Both correct (different contexts)",
        "merge":   "Merge into new fact",
        "neither": "Neither — reject both",
    }.get(choice, choice)
