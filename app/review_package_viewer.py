"""Phase 75 — Review Package UI helper.

Renders the Phase74 safe manual-review package inside Streamlit.
Import render_review_package_panel() from main.py, or run standalone:
    streamlit run app/review_package_viewer.py

Privacy rules
-------------
- Reads only safe public Phase74 fields (bucket IDs, counts, explanations).
- Never displays raw filenames, raw paths, PHI, OCR text, or private mappings.
- Never modifies private feedback files.
- Does not change production OCR, extraction, thresholds, or safety gates.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

PACKAGE_JSON = (
    ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
    / "manual_review_package_SAFE.json"
)
REPORT74_JSON = (
    ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
    / "phase74_manual_review_package_auto_improvement_report.json"
)
PACKAGE_MD = (
    ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
    / "manual_review_package_SAFE.md"
)


# ---------------------------------------------------------------------------
# Data loading (importable, no Streamlit dependency)
# ---------------------------------------------------------------------------


def load_review_package(path: Path | None = None) -> dict[str, Any]:
    """Load the Phase74 manual_review_package_SAFE.json."""
    p = path or PACKAGE_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"Phase74 safe review package not found: {p}. "
            "Run Phase74 (run_phase74_manual_review_package_auto_improvement.py) first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_phase74_report(path: Path | None = None) -> dict[str, Any] | None:
    p = path or REPORT74_JSON
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_bucket_summary(package: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of safe bucket summary records (no PHI, no paths)."""
    buckets = package.get("buckets") or []
    return sorted(
        [
            {
                "bucket_id": b.get("bucket_id", ""),
                "bucket_name": b.get("bucket_name", ""),
                "priority": b.get("priority", 99),
                "aggregate_count": b.get("aggregate_count", 0),
                "high_priority_item_count": b.get("high_priority_item_count", 0),
                "why_it_is_in_review": b.get("why_it_is_in_review", ""),
                "what_the_system_knows": b.get("what_the_system_knows", ""),
                "what_the_system_does_not_know": b.get("what_the_system_does_not_know", ""),
                "safest_next_action": b.get("safest_next_action", ""),
                "whether_operator_action_is_required": b.get(
                    "whether_operator_action_is_required", False
                ),
                "whether_production_change_is_allowed": b.get(
                    "whether_production_change_is_allowed", False
                ),
                "pending_safe_ids_sample": b.get("pending_safe_ids_sample") or [],
            }
            for b in buckets
        ],
        key=lambda x: x["priority"],
    )


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

_PRIORITY_COLORS = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🔵", 5: "⚪", 6: "✅"}


def render_review_package_panel(
    package_path: Path | None = None,
    report74_path: Path | None = None,
) -> None:
    """Render the Review Package panel. Safe for embedding in main.py."""
    try:
        import streamlit as st  # type: ignore[import-untyped]
    except ImportError:
        print("streamlit not available — run: pip install streamlit")
        return

    st.subheader("Review Package — Phase 74 Auto-Improvement")
    st.caption("Local-only. No PHI. No raw filenames. No production changes.")

    # Load Phase74 report for meta info
    report74 = load_phase74_report(report74_path)
    if report74:
        conclusion = report74.get("conclusion", "unknown")
        item_count = report74.get("review_package_item_count", 0)
        bucket_count = report74.get("bucket_count", 0)
        st.success(f"Phase74 conclusion: `{conclusion}` — {item_count} items across {bucket_count} buckets")

    # Plain-language status strip
    col1, col2, col3, col4 = st.columns(4)
    col1.info("No manual document review required to continue.")
    col2.info("Buckets grouped automatically from safe diagnostics.")
    col3.info("Production OCR/extractor changes not currently justified.")
    col4.info("Manual-review boundary retained.")

    # Load package
    try:
        package = load_review_package(package_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    buckets = get_bucket_summary(package)
    total = sum(b["aggregate_count"] for b in buckets)

    st.markdown("---")
    st.markdown(f"**Total review items:** `{total}`  |  **Buckets:** `{len(buckets)}`")

    # Bucket summary table
    st.markdown("#### Bucket Summary")
    header_cols = st.columns([1, 4, 3, 3, 3])
    header_cols[0].markdown("**Pri**")
    header_cols[1].markdown("**Bucket**")
    header_cols[2].markdown("**Count**")
    header_cols[3].markdown("**Operator required?**")
    header_cols[4].markdown("**Prod. change allowed?**")
    for b in buckets:
        icon = _PRIORITY_COLORS.get(b["priority"], "•")
        row = st.columns([1, 4, 3, 3, 3])
        row[0].markdown(f"{icon} `{b['priority']}`")
        row[1].markdown(f"**{b['bucket_name']}**")
        row[2].markdown(f"`{b['aggregate_count']}`")
        row[3].markdown("Yes" if b["whether_operator_action_is_required"] else "**No**")
        row[4].markdown("Yes" if b["whether_production_change_is_allowed"] else "**No**")

    st.markdown("---")
    st.markdown("#### Bucket Details")

    for b in buckets:
        icon = _PRIORITY_COLORS.get(b["priority"], "•")
        with st.expander(
            f"{icon} {b['bucket_name']}  —  {b['aggregate_count']} items",
            expanded=(b["priority"] == 1),
        ):
            st.markdown(f"**bucket_id:** `{b['bucket_id']}`")
            if b["high_priority_item_count"]:
                st.warning(
                    f"{b['high_priority_item_count']} high-priority item(s) in this bucket."
                )
            st.markdown(f"**Why in review:** {b['why_it_is_in_review']}")
            st.markdown(f"**What the system knows:** {b['what_the_system_knows']}")
            st.markdown(
                f"**What the system does not know:** {b['what_the_system_does_not_know']}"
            )
            st.success(f"**Safest next action:** {b['safest_next_action']}")

            if b["pending_safe_ids_sample"]:
                with st.expander("Safe IDs sample (public — no PHI)"):
                    st.code(", ".join(b["pending_safe_ids_sample"]), language=None)

            flag_col1, flag_col2 = st.columns(2)
            flag_col1.caption(
                f"Operator action required: {'Yes' if b['whether_operator_action_is_required'] else 'No'}"
            )
            flag_col2.caption(
                f"Production change allowed: {'Yes' if b['whether_production_change_is_allowed'] else 'No'}"
            )

    # Link to safe MD report
    st.markdown("---")
    if PACKAGE_MD.exists():
        with st.expander("View full safe review package (Markdown)"):
            st.markdown(PACKAGE_MD.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_review_package_panel()
