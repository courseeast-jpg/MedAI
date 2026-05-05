"""Phase 72B — Streamlit operator feedback panel.

Local-only review UI for the MedAI operator feedback workflow.
Run standalone with:
    streamlit run app/operator_feedback.py

Or import render_feedback_panel() from within app/main.py (optional).

Privacy rules:
  - Displays safe_file_id, tier, problem class, review goal, question only.
  - Raw filenames/paths shown in an ephemeral local-only expander (never logged).
  - No raw filenames or paths written to any public report or log.
  - Private notes saved only to gitignored operator_feedback_PRIVATE.json.
  - Does NOT change production extraction, OCR, classifiers, or safety gates.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

from scripts.run_phase72b_operator_review_console import (  # noqa: E402
    ALLOWED_ANSWERS,
    PHASE71_QUEUE,
    PHASE72_PRIVATE,
    REPORT_DIR,
    generate_console_report,
    get_pending,
    load_private_feedback,
    load_queue,
    record_answer_to_file,
)

_PRIVATE_MAPPING = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit"
    / "local_filename_mapping_PRIVATE.json"
)


def _load_private_mapping() -> dict[str, dict]:
    if not _PRIVATE_MAPPING.exists():
        return {}
    try:
        payload = json.loads(_PRIVATE_MAPPING.read_text(encoding="utf-8"))
        return payload.get("files") or {}
    except Exception:
        return {}


def render_feedback_panel(
    queue_path: Path | None = None,
    private_path: Path | None = None,
    report_dir: Path | None = None,
) -> None:
    """Render the operator feedback Streamlit panel."""
    try:
        import streamlit as st  # type: ignore[import-untyped]
    except ImportError:
        print("streamlit not available — run: pip install streamlit")
        return

    st.title("MedAI Operator Feedback — Phase 72B")
    st.caption("Local-only review console. Answers are saved privately.")

    try:
        queue = load_queue(queue_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    feedback = load_private_feedback(private_path)
    records = feedback.get("feedback") or []
    private_mapping = _load_private_mapping()

    tier_filter = st.radio(
        "Review scope",
        options=["Tier 1 only (recommended)", "Tier 2", "All pending"],
        horizontal=True,
    )
    tier_map = {"Tier 1 only (recommended)": 1, "Tier 2": 2, "All pending": None}
    selected_tier = tier_map[tier_filter]

    pending = get_pending(records, tier=selected_tier)
    total = len(queue)
    reviewed = sum(1 for r in records if r.get("answer") is not None)

    st.progress(reviewed / total if total else 0,
                text=f"Reviewed {reviewed}/{total}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Reviewed", reviewed)
    col2.metric("Pending", total - reviewed)
    col3.metric("Tier-1 pending",
                sum(1 for r in records
                    if r.get("priority_tier") == 1 and r.get("answer") is None))

    if not pending:
        st.success("All items in this scope have been reviewed.")
        return

    # Show one item at a time using session state index
    if "fb_index" not in st.session_state:
        st.session_state.fb_index = 0
    idx = min(st.session_state.fb_index, len(pending) - 1)
    item = pending[idx]
    sid = str(item.get("safe_file_id") or "")

    st.markdown("---")
    st.markdown(f"**Item {idx + 1} of {len(pending)}**")
    st.markdown(f"- `safe_file_id`: **{sid}**")
    st.markdown(f"- Priority tier: **{item.get('priority_tier')}**")
    st.markdown(f"- Problem class: `{item.get('suspected_problem_class')}`")
    st.markdown(f"- Source: `{item.get('source_phase')}`")
    st.markdown(f"**Review goal:** {item.get('review_goal', '')}")
    st.markdown(f"**Question:** _{item.get('operator_question', '')}_")

    # Local-only file open helper — never written to public output
    if sid in private_mapping:
        rel = str(private_mapping[sid].get("original_relative_path") or "")
        if rel:
            with st.expander("Open original file locally (private — never logged)"):
                full_path = ROOT / "full_corpus_input" / rel
                st.code(str(full_path), language=None)
                if full_path.exists():
                    import subprocess  # noqa: PLC0415
                    if st.button("Open file"):
                        subprocess.Popen(["start", "", str(full_path)], shell=True)
                else:
                    st.caption("File not found at expected local path.")

    st.markdown("---")
    note = st.text_input(
        "Private note (optional — never in public reports):",
        key=f"note_{sid}",
        placeholder="Your private note here…",
    )

    answer_cols = st.columns(len(ALLOWED_ANSWERS) // 2 + 1)
    for i, answer in enumerate(ALLOWED_ANSWERS):
        col = answer_cols[i % len(answer_cols)]
        if col.button(answer, key=f"ans_{sid}_{answer}"):
            try:
                p = private_path or PHASE72_PRIVATE
                record_answer_to_file(
                    sid, answer,
                    private_path=p,
                    private_note=note if note else None,
                )
                st.success(f"Saved: **{answer}**")
                st.session_state.fb_index = idx + 1
                # Regenerate public summary
                updated = load_private_feedback(p)
                generate_console_report(
                    queue,
                    updated.get("feedback") or [],
                    report_dir=report_dir or REPORT_DIR,
                    private_path=p,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error saving: {exc}")

    nav_col1, nav_col2 = st.columns(2)
    if nav_col1.button("← Previous"):
        st.session_state.fb_index = max(0, idx - 1)
        st.rerun()
    if nav_col2.button("Next →"):
        st.session_state.fb_index = min(len(pending) - 1, idx + 1)
        st.rerun()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_feedback_panel()
