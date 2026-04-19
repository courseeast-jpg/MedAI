"""
MedAI v1.1 — Streamlit UI (Track D)
Thin safety layer: 5 required surfaces only.
1. Uncertainty / confidence display
2. Contradiction / conflict display
3. Trust + source display
4. Degraded mode indicator
5. Medication warning indicator (DDI gate)
"""
import asyncio
import re
import streamlit as st
from pathlib import Path
from datetime import datetime

# Matches any Presidio-style placeholder written into content during ingestion.
# Used to detect records whose content was over-redacted (e.g. food names
# mis-labelled as PERSON/LOCATION by Presidio's NER).
_PLACEHOLDER_RE = re.compile(
    r'\[(?:PERSON|LOCATION|DATE|CONTACT_REMOVED|ID_REMOVED|URL_REMOVED|PHYSICIAN)\]'
)


def _display_content(record) -> tuple[str, bool]:
    """
    Return (text_to_display, is_stale).

    If record.content contains PII placeholders the record was corrupted during
    ingestion (Presidio NER false-positive on food/reference data).  We try to
    rebuild a readable string from record.structured instead — that field holds
    the raw values Gemini extracted BEFORE the content string was assembled, so
    it often has the real food name even when content only shows [PERSON].

    Returns is_stale=True when placeholders are detected so the caller can add
    a visual warning.  The UI must NEVER call pii.strip() on stored records.
    """
    if not _PLACEHOLDER_RE.search(record.content):
        return record.content, False

    s = record.structured or {}

    # ── Food guide entry ──────────────────────────────────────────────────────
    food_name = s.get("food_name", "")
    if food_name and not _PLACEHOLDER_RE.search(food_name):
        parts = [f"Food: {food_name}"]
        if s.get("food_name_ru"):
            parts.append(f"(ru: {s['food_name_ru']})")
        scores = []
        for key, label in (
            ("ibs_score", "IBS"),
            ("diverticulitis_score", "Diverticulitis"),
            ("oxalates_score", "Oxalates"),
            ("crystalluria_score", "Crystalluria"),
        ):
            if s.get(key) is not None:
                scores.append(f"{label}={s[key]}")
        if scores:
            parts.append("| " + ", ".join(scores))
        if s.get("safety_category"):
            parts.append(f"| Safety: {s['safety_category']}")
        return " ".join(parts), True

    # ── Generic structured record (diagnosis, medication, etc.) ───────────────
    name = s.get("name") or s.get("test_name") or s.get("description") or ""
    if name and not _PLACEHOLDER_RE.search(name):
        return f"{record.fact_type.replace('_', ' ').title()}: {name}", True

    # Can't reconstruct — return original with stale flag
    return record.content, True

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI v1.1",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import core system ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    ANTHROPIC_API_KEY, DB_PATH, CHROMA_PATH,
    ACTIVE_CONNECTORS, ENABLE_ENRICHMENT
)
from app.schemas import SystemState, UnifiedResponse, MKBRecord


@st.cache_resource
def load_system():
    """Load all system components once, cache for session."""
    import os
    from mkb.sqlite_store import SQLiteStore
    from mkb.vector_store import VectorStore
    from mkb.quality_gate import QualityGate
    from mkb.truth_resolution import TruthResolutionEngine
    from extraction.extractor import Extractor
    from extraction.pii_stripper import PIIStripper
    from ingestion.pdf_pipeline import PDFPipeline
    from decision.response_scorer import ResponseScorer
    from decision.medication_safety import MedicationSafetyGate
    from decision.decision_engine import DecisionEngine
    from enrichment.enrichment_engine import EnrichmentEngine
    from external_apis.connectors import build_connector_registry, ClaudeSynthesizer

    db_key = os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")
    sql = SQLiteStore(DB_PATH, db_key)
    vec = VectorStore(CHROMA_PATH)
    extractor = Extractor()
    pii = PIIStripper()
    quality_gate = QualityGate(sql, vec)
    connectors = build_connector_registry()
    ddi_connector = connectors.get("patientnotes_ddi")
    med_gate = MedicationSafetyGate(ddi_connector, sql)
    scorer = ResponseScorer(vec, med_gate)
    synthesizer = ClaudeSynthesizer(ANTHROPIC_API_KEY, "claude-sonnet-4-20250514")
    state = SystemState(
        claude_available=bool(ANTHROPIC_API_KEY),
        active_connectors=ACTIVE_CONNECTORS,
    )
    engine = DecisionEngine(sql, vec, scorer, med_gate, connectors, synthesizer, state)
    pdf_pipeline = PDFPipeline(extractor, pii)
    enrichment = EnrichmentEngine(extractor, sql, vec, quality_gate, med_gate)

    return {
        "sql": sql, "vec": vec, "extractor": extractor, "pii": pii,
        "quality_gate": quality_gate, "engine": engine,
        "pdf_pipeline": pdf_pipeline, "enrichment": enrichment,
        "state": state, "med_gate": med_gate,
    }


# ── Safety surface 4: Degraded mode banner ───────────────────────────────────
def render_system_status(state: SystemState):
    if state.safe_mode:
        st.error(
            "⚠ SAFE MODE — External AI unavailable. Operating on MKB context only. "
            f"Reason: {state.safe_mode_reason or 'API unavailable'}",
            icon="🔴"
        )
    elif not state.claude_available:
        st.warning("Claude API not configured. Add ANTHROPIC_API_KEY to .env", icon="⚠")
    else:
        st.success(f"System ready · Active connectors: {', '.join(state.active_connectors)}")


# ── Safety surface 3: Trust + source display ─────────────────────────────────
def render_mkb_record(record: MKBRecord, show_hypothesis_warning: bool = True):
    tier_color = {
        "active": "🟢", "hypothesis": "🟡",
        "quarantined": "🔴", "superseded": "⚫"
    }.get(record.tier, "⚪")
    trust_label = {1: "Clinical", 2: "Peer-review", 3: "AI", 4: "Web", 5: "Unverified"}

    # Never apply pii.strip() here — display the raw DB value.
    # If content was corrupted by over-redaction at ingestion time, fall back
    # to structured data and flag the record as stale.
    display_text, is_stale = _display_content(record)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"{tier_color} **{display_text}**")
        if is_stale:
            st.caption(
                "⚠ Content was over-redacted during ingestion (Presidio NER false positive "
                "on food/reference data). Showing structured-field fallback. "
                "Re-upload the source document to regenerate this record."
            )
        if record.tier == "hypothesis" and show_hypothesis_warning:
            st.caption("⚠ HYPOTHESIS — not clinically verified. Not used in synthesis without promotion.")
    with col2:
        st.caption(f"Trust: {trust_label.get(record.trust_level, str(record.trust_level))}")
        st.caption(f"Source: {record.source_name[:30]}")


# ── Safety surface 5: Medication DDI warning ─────────────────────────────────
def render_ddi_warning(response: UnifiedResponse):
    if not response.ddi_findings:
        return
    for finding in response.ddi_findings:
        if finding.severity == "HIGH":
            st.error(
                f"🚨 HIGH SEVERITY INTERACTION: {finding.drug_a} ↔ {finding.drug_b}\n"
                f"Mechanism: {finding.mechanism or 'See pharmacist'}\n"
                f"Management: {finding.management or 'Consult physician before proceeding'}",
                icon="🚨"
            )
        elif finding.severity == "MEDIUM":
            st.warning(
                f"⚠ MEDIUM interaction: {finding.drug_a} ↔ {finding.drug_b}. "
                f"{finding.management or 'Monitor closely.'}",
                icon="⚠"
            )
        else:
            st.info(f"ℹ Low interaction: {finding.drug_a} ↔ {finding.drug_b}", icon="ℹ")


# ── Safety surface 1: Uncertainty / confidence display ───────────────────────
def render_confidence(response: UnifiedResponse):
    band = response.confidence_band
    score = response.confidence
    color = {"high": "green", "acceptable": "blue", "low": "orange", "discarded": "red"}.get(band, "gray")
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{score:.0%}")
    cols[1].metric("Band", band.upper())
    cols[2].metric("Sources", len(response.sources_used))
    cols[3].metric("MKB facts used", len(response.mkb_facts_used))

    if band in ("low", "discarded") or response.safe_mode:
        st.warning(
            "Low confidence response. External AI results were insufficient or unavailable. "
            "Verify with a qualified clinician.",
            icon="⚠"
        )


# ── Safety surface 2: Contradiction / conflict display ───────────────────────
def render_conflicts(sys_components: dict):
    sql = sys_components["sql"]
    conflicts = sql.get_records_requiring_review()
    if not conflicts:
        return
    with st.expander(f"⚠ {len(conflicts)} record(s) require review", expanded=False):
        for record in conflicts:
            display_text, is_stale = _display_content(record)
            st.markdown(f"**{record.fact_type.upper()}**: {display_text}")
            if is_stale:
                st.caption(
                    "⚠ Content over-redacted at ingestion — showing structured fallback. "
                    "Re-upload source document to fix."
                )
            st.caption(f"Status: {record.status} | Source: {record.source_name}")
            col1, col2, col3 = st.columns(3)
            if col1.button("Accept", key=f"accept_{record.id}"):
                sql.update_status(record.id, "active", "active")
                sql.write_ledger(__import__('app.schemas', fromlist=['LedgerEvent']).LedgerEvent(
                    event_type="conflict_resolved",
                    record_id=record.id,
                    details={"action": "user_accepted"},
                ))
                st.rerun()
            if col2.button("Reject", key=f"reject_{record.id}"):
                sql.update_status(record.id, "archived", "superseded")
                st.rerun()
            if col3.button("Defer", key=f"defer_{record.id}"):
                st.info("Deferred — will resurface in 7 days")
            st.divider()


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    sys_components = load_system()
    state = sys_components["state"]

    st.title("MedAI v1.1")
    st.caption("Personal medical intelligence · Local-first · Decision support only")

    # System status banner (Safety surface 4)
    render_system_status(state)

    # Conflict panel (Safety surface 2)
    render_conflicts(sys_components)

    # Sidebar: MKB stats
    with st.sidebar:
        st.header("MKB Status")
        counts = sys_components["sql"].count_records()
        st.metric("Active facts", counts["active"])
        st.metric("Hypothesis", counts["hypothesis"])
        st.metric("Quarantined", counts["quarantined"])
        st.metric("Total", counts["total"])
        st.divider()
        st.caption(f"Connectors: {', '.join(ACTIVE_CONNECTORS)}")
        st.caption(f"Enrichment: {'ON' if ENABLE_ENRICHMENT else 'OFF'}")

    # Tabs
    tab_query, tab_upload, tab_mkb, tab_conflicts = st.tabs(
        ["Query", "Upload Document", "MKB Explorer", "Conflict Review"]
    )

    # ── Query Tab ─────────────────────────────────────────────────────────
    with tab_query:
        query = st.text_area(
            "Ask a medical question",
            placeholder="e.g. What does my EEG result mean for my epilepsy treatment?",
            height=100,
        )
        col1, col2 = st.columns([1, 4])
        submit = col1.button("Submit", type="primary")

        if submit and query.strip():
            with st.spinner("Processing..."):
                try:
                    response = asyncio.run(
                        sys_components["engine"].process(query)
                    )

                    # Safety surface 1: confidence
                    render_confidence(response)
                    st.divider()

                    # Safety surface 5: DDI warnings FIRST
                    render_ddi_warning(response)

                    # Main synthesis
                    st.subheader("Response")
                    if response.safe_mode:
                        st.info(response.synthesis)
                    else:
                        st.markdown(response.synthesis)

                    # MKB facts used (Safety surface 3)
                    if response.mkb_facts_used:
                        with st.expander("MKB records used in this response"):
                            for r in response.mkb_facts_used:
                                render_mkb_record(r)

                    # Hypothesis facts
                    if response.hypothesis_facts:
                        with st.expander(f"⚠ {len(response.hypothesis_facts)} hypothesis facts (AI-derived, unverified)"):
                            for r in response.hypothesis_facts:
                                render_mkb_record(r)

                    # Discarded responses
                    if response.discarded_responses:
                        with st.expander(f"Discarded responses ({len(response.discarded_responses)})"):
                            for d in response.discarded_responses:
                                st.caption(f"✗ {d}")

                    # Enrich from response
                    if ENABLE_ENRICHMENT and not response.safe_mode:
                        written = sys_components["enrichment"].enrich_from_response(response)
                        if written:
                            st.caption(f"Added {len(written)} hypothesis facts to MKB from this response")

                except Exception as e:
                    st.error(f"Error processing query: {e}")

    # ── Upload Tab ────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Upload Medical Document")
        specialty = st.selectbox(
            "Specialty",
            ["neurology", "epilepsy", "gastroenterology", "urology", "general"]
        )
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded and st.button("Process Document"):
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(uploaded, tmp)
                tmp_path = Path(tmp.name)

            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    candidates = sys_components["pdf_pipeline"].process(tmp_path, specialty)
                    written = []
                    blocked = []
                    for candidate in candidates:
                        # Medication gate for medication facts
                        if candidate.fact_type == "medication":
                            decision, msg, findings = sys_components["med_gate"].gate_medication_write(candidate)
                            if decision == "block":
                                blocked.append((candidate, msg, findings))
                                continue
                        approved, reason, final = sys_components["quality_gate"].check(candidate)
                        if approved and final:
                            sys_components["sql"].write_record(final)
                            sys_components["vec"].add_record(final)
                            written.append(final)

                    st.success(f"Added {len(written)} records to MKB from {uploaded.name}")

                    # Show DDI blocks (Safety surface 5)
                    for cand, msg, findings in blocked:
                        st.error(f"🚨 DDI BLOCK: {msg}")
                        render_ddi_warning(UnifiedResponse(
                            query="", specialty="", synthesis="", confidence=0,
                            confidence_band="", ddi_findings=findings,
                        ))
                        if st.button(f"Accept anyway — {cand.content[:40]}", key=f"ddi_accept_{cand.id}"):
                            cand.ddi_status = "high_accepted_by_user"
                            cand.tier = "active"
                            approved, reason, final = sys_components["quality_gate"].check(cand)
                            if approved and final:
                                sys_components["sql"].write_record(final)
                                sys_components["vec"].add_record(final)
                                st.success("Accepted by user")

                    if written:
                        st.subheader("Records added:")
                        for r in written[:10]:
                            render_mkb_record(r, show_hypothesis_warning=False)

                except Exception as e:
                    st.error(f"Processing error: {e}")
                    raise

    # ── MKB Explorer Tab ─────────────────────────────────────────────────
    with tab_mkb:
        st.subheader("MKB Explorer")
        col1, col2 = st.columns(2)
        spec_filter = col1.selectbox("Specialty", ["all", "neurology", "epilepsy", "gastroenterology", "urology"])
        tier_filter = col2.selectbox("Tier", ["all", "active", "hypothesis", "quarantined"])

        specialty_q = None if spec_filter == "all" else spec_filter
        tier_q = None if tier_filter == "all" else tier_filter

        if specialty_q:
            records = sys_components["sql"].get_by_specialty(specialty_q, tier_q)
        else:
            # Show all — limited to 50
            with sys_components["sql"]._get_conn() as conn:
                q = "SELECT * FROM records"
                params = []
                if tier_q:
                    q += " WHERE tier=?"
                    params.append(tier_q)
                q += " ORDER BY first_recorded DESC LIMIT 50"
                rows = conn.execute(q, params).fetchall()
            records = [sys_components["sql"]._row_to_record(r) for r in rows]

        st.caption(f"{len(records)} records shown")
        for r in records:
            render_mkb_record(r)

    # ── Conflict Review Tab ──────────────────────────────────────────────
    with tab_conflicts:
        try:
            from mkb.conflict_resolver import ConflictResolver
            from app.conflict_review import render_conflict_review
            resolver = ConflictResolver(DB_PATH, sql_store=sys_components["sql"])
            render_conflict_review(resolver)
        except Exception as e:
            st.error(f"Conflict review unavailable: {e}")

    st.divider()
    st.caption(
        "MedAI v1.1 · Decision support only · Not a medical device · "
        "All outputs require clinical verification"
    )


if __name__ == "__main__":
    main()
