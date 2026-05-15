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
# Operator-facing presentation helpers
# ---------------------------------------------------------------------------

_PRIORITY_LABELS = {
    1: "[1]",
    2: "[2]",
    3: "[3]",
    4: "[4]",
    5: "[5]",
    6: "[6]",
}

_BUCKET_DISPLAY_NAMES = {
    "OCR Quality Review": "Scan quality review",
    "Empty Extraction Review": "No text found",
    "Unknown Document Class Review": "Unknown document type",
    "Possible Multi-Document PDF Review": "Possible combined PDF",
    "Unsupported or Deferred Format Review": "Unsupported format",
    "Completed Manual Boundary Branches": "Completed review paths",
}

_BUCKET_OPERATOR_COPY = {
    "ocr_quality_review": {
        "why": "Some files may have weak scan quality. MedAI may miss or misread values.",
        "knows": "The scan quality was below the safe threshold.",
        "unknown": "Whether the missed text is clinically important.",
        "next": "Review these files later or upload clearer copies. Do not lower the quality threshold.",
    },
    "empty_extraction_review": {
        "why": "MedAI could not find useful text to process.",
        "knows": "The file produced no usable extraction result.",
        "unknown": "Whether the file is blank, unreadable, unsupported, or simply not a medical document.",
        "next": "Check the source file and upload a clearer or correct copy if needed.",
    },
    "unknown_document_class_review": {
        "why": "MedAI could not confidently identify the document type.",
        "knows": "The file did not match a known safe document category.",
        "unknown": "Whether the document belongs to a supported medical category.",
        "next": "Review the file type before relying on extracted results.",
    },
    "possible_multi_document_pdf_review": {
        "why": "One PDF may contain more than one document.",
        "knows": "The file may combine multiple sections or documents.",
        "unknown": "Where one document ends and another begins.",
        "next": "Split the file into separate documents if needed.",
    },
    "unsupported_or_deferred_format_review": {
        "why": "The file type or structure is not fully supported.",
        "knows": "The current pipeline should not accept it automatically.",
        "unknown": "Whether a converted copy would process correctly.",
        "next": "Convert to PDF or TXT and try again.",
    },
    "completed_manual_boundary_branches": {
        "why": "These review paths are already completed.",
        "knows": "No action is currently needed for this group.",
        "unknown": "",
        "next": "Keep for audit history.",
    },
}


def display_bucket_name(bucket: dict[str, Any]) -> str:
    return _BUCKET_DISPLAY_NAMES.get(
        str(bucket.get("bucket_name") or ""),
        str(bucket.get("bucket_name") or "Review item"),
    )


def bucket_operator_copy(bucket: dict[str, Any]) -> dict[str, str]:
    return _BUCKET_OPERATOR_COPY.get(
        str(bucket.get("bucket_id") or ""),
        {
            "why": str(bucket.get("why_it_is_in_review") or "This group needs review before relying on it."),
            "knows": str(bucket.get("what_the_system_knows") or "MedAI has aggregate safety information only."),
            "unknown": str(bucket.get("what_the_system_does_not_know") or "Whether the source file needs operator action."),
            "next": str(bucket.get("safest_next_action") or "Review the safe file IDs before relying on the result."),
        },
    )


def review_status_summary(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(b.get("aggregate_count") or 0) for b in buckets)
    scan_quality_count = 0
    for bucket in buckets:
        if bucket.get("bucket_id") == "ocr_quality_review":
            scan_quality_count = int(bucket.get("aggregate_count") or 0)
            break
    return {
        "review_status": "No blocking review required",
        "scan_quality_attention_count": scan_quality_count,
        "production_change_recommended": False,
        "total_review_items": total,
        "review_categories": len(buckets),
    }


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------


def render_review_package_panel(
    package_path: Path | None = None,
    report74_path: Path | None = None,
    *,
    show_title: bool = True,
) -> None:
    """Render the Review Package panel. Safe for embedding in main.py."""
    try:
        import streamlit as st  # type: ignore[import-untyped]
    except ImportError:
        print("streamlit not available — run: pip install streamlit")
        return

    # Load package
    try:
        package = load_review_package(package_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    buckets = get_bucket_summary(package)
    summary = review_status_summary(buckets)
    report74 = load_phase74_report(report74_path)

    if show_title:
        st.subheader("Review Package")
        st.caption("Review items and system safety findings.")

    status_col, attention_col, production_col = st.columns(3)
    status_col.success(f"Review status: {summary['review_status']}")
    attention_col.info(
        f"{summary['scan_quality_attention_count']} scan-quality items need attention when convenient."
    )
    production_col.info("No production changes recommended.")

    with st.expander("Build / audit details", expanded=False):
        st.caption("Phase 74")
        st.caption("Auto-improvement")
        if report74:
            st.caption(f"Original conclusion: {report74.get('conclusion', 'unknown')}")
            st.caption(f"Raw report conclusion: {report74.get('conclusion', 'unknown')}")
            st.caption(f"Total item count: {report74.get('review_package_item_count', summary['total_review_items'])}")
            st.caption(f"Bucket count: {report74.get('bucket_count', summary['review_categories'])}")
        else:
            st.caption(f"Total item count: {summary['total_review_items']}")
            st.caption(f"Bucket count: {summary['review_categories']}")
        st.caption("Source report: reports/phase74_manual_review_package_auto_improvement/manual_review_package_SAFE.json")

    st.markdown("---")
    st.markdown(f"**Total review items:** `{summary['total_review_items']}`")
    st.markdown(f"**Review categories:** `{summary['review_categories']}`")

    # Bucket summary table
    st.markdown("#### Review summary")
    header_cols = st.columns([1, 4, 3, 3, 3])
    header_cols[0].markdown("**Pri**")
    header_cols[1].markdown("**Issue type**")
    header_cols[2].markdown("**Items**")
    header_cols[3].markdown("**Needs action?**")
    header_cols[4].markdown("**System change?**")
    for b in buckets:
        icon = _PRIORITY_LABELS.get(b["priority"], "[?]")
        row = st.columns([1, 4, 3, 3, 3])
        row[0].markdown(f"{icon} `{b['priority']}`")
        row[1].markdown(f"**{display_bucket_name(b)}**")
        row[2].markdown(f"`{b['aggregate_count']}`")
        row[3].markdown("Yes" if b["whether_operator_action_is_required"] else "**No**")
        row[4].markdown("Yes" if b["whether_production_change_is_allowed"] else "**No**")

    st.markdown("---")
    st.markdown("#### Review details")

    for b in buckets:
        icon = _PRIORITY_LABELS.get(b["priority"], "[?]")
        copy = bucket_operator_copy(b)
        with st.expander(
            f"{icon} {display_bucket_name(b)} - {b['aggregate_count']} items",
            expanded=(b["priority"] == 1),
        ):
            if b["high_priority_item_count"]:
                st.warning(f"{b['high_priority_item_count']} higher-priority items need attention.")
            else:
                st.info("No high-priority items.")
            if b["bucket_id"] == "completed_manual_boundary_branches":
                st.markdown(f"**Why this appears here:** {copy['why']}")
            else:
                st.markdown(f"**Why this needs review:** {copy['why']}")
            st.markdown(f"**What MedAI knows:** {copy['knows']}")
            if copy.get("unknown"):
                st.markdown(f"**What MedAI does not know:** {copy['unknown']}")
            st.success(f"**Safest next step:** {copy['next']}")

            if b["pending_safe_ids_sample"]:
                with st.expander("Example safe file IDs"):
                    st.caption("These are internal safe IDs, not patient names or real filenames.")
                    st.code(", ".join(b["pending_safe_ids_sample"]), language=None)

            with st.expander("Advanced bucket audit details", expanded=False):
                st.markdown(f"**bucket_id:** `{b['bucket_id']}`")
                st.markdown(f"**Original bucket label:** {b['bucket_name']}")
                st.markdown(f"**Original why in review:** {b['why_it_is_in_review']}")
                st.markdown(f"**Original system knows:** {b['what_the_system_knows']}")
                st.markdown(
                    f"**Original system does not know:** {b['what_the_system_does_not_know']}"
                )
                st.markdown(f"**Original safest next action:** {b['safest_next_action']}")

            flag_col1, flag_col2 = st.columns(2)
            flag_col1.caption(
                f"Needs action: {'Yes' if b['whether_operator_action_is_required'] else 'No'}"
            )
            flag_col2.caption(
                f"System change: {'Yes' if b['whether_production_change_is_allowed'] else 'No'}"
            )

    # Link to safe MD report
    st.markdown("---")
    if PACKAGE_MD.exists():
        with st.expander("Advanced: full audit report", expanded=False):
            st.markdown(PACKAGE_MD.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_review_package_panel()
