"""MedAI Streamlit application with Phase 1 execution pipeline integration."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ACTIVE_CONNECTORS, ANTHROPIC_API_KEY, CHROMA_PATH, DB_PATH, ENABLE_ENRICHMENT
from app.schemas import MKBRecord, SystemState, UnifiedResponse
from execution.pipeline import ExecutionPipeline


PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|LOCATION|DATE|CONTACT_REMOVED|ID_REMOVED|URL_REMOVED|PHYSICIAN)\]")


st.set_page_config(
    page_title="MedAI v1.1",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_content(record: MKBRecord) -> tuple[str, bool]:
    if not PLACEHOLDER_RE.search(record.content):
        return record.content, False

    structured = record.structured or {}
    name = structured.get("name") or structured.get("test_name") or structured.get("description")
    if name and not PLACEHOLDER_RE.search(str(name)):
        return f"{record.fact_type.replace('_', ' ').title()}: {name}", True
    if structured.get("text") and not PLACEHOLDER_RE.search(str(structured["text"])):
        return str(structured["text"]), True
    return record.content, True


@st.cache_resource
def load_system() -> dict:
    from decision.decision_engine import DecisionEngine
    from decision.medication_safety import MedicationSafetyGate
    from decision.response_scorer import ResponseScorer
    from enrichment.enrichment_engine import EnrichmentEngine
    from external_apis.connectors import ClaudeSynthesizer, build_connector_registry
    from extraction.pii_stripper import PIIStripper
    from mkb.quality_gate import QualityGate
    from mkb.sqlite_store import SQLiteStore
    from mkb.vector_store import VectorStore

    db_key = os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")
    sql = SQLiteStore(DB_PATH, db_key)
    vec = VectorStore(CHROMA_PATH)
    quality_gate = QualityGate(sql, vec)
    connectors = build_connector_registry()
    medication_gate = MedicationSafetyGate(connectors.get("patientnotes_ddi"), sql)
    scorer = ResponseScorer(vec, medication_gate)
    synthesizer = ClaudeSynthesizer(ANTHROPIC_API_KEY, "claude-sonnet-4-20250514")
    state = SystemState(claude_available=bool(ANTHROPIC_API_KEY), active_connectors=ACTIVE_CONNECTORS)
    engine = DecisionEngine(sql, vec, scorer, medication_gate, connectors, synthesizer, state)
    execution = ExecutionPipeline(
        sql_store=sql,
        vector_store=vec,
        quality_gate=quality_gate,
        medication_gate=medication_gate,
        pii_stripper=PIIStripper(),
    )
    enrichment = EnrichmentEngine(None, sql, vec, quality_gate, medication_gate)

    return {
        "sql": sql,
        "vec": vec,
        "quality_gate": quality_gate,
        "engine": engine,
        "execution": execution,
        "enrichment": enrichment,
        "state": state,
        "med_gate": medication_gate,
    }


def render_system_status(state: SystemState) -> None:
    if state.safe_mode:
        st.error(
            "SAFE MODE - External AI unavailable. Operating on MKB context only. "
            f"Reason: {state.safe_mode_reason or 'API unavailable'}"
        )
    elif not state.claude_available:
        st.warning("Claude API not configured. Add ANTHROPIC_API_KEY to .env")
    else:
        st.success(f"System ready. Active connectors: {', '.join(state.active_connectors)}")


def render_mkb_record(record: MKBRecord, show_hypothesis_warning: bool = True) -> None:
    tier_label = {
        "active": "Active",
        "hypothesis": "Hypothesis",
        "quarantined": "Quarantined",
        "superseded": "Superseded",
    }.get(record.tier, record.tier)
    trust_label = {1: "Clinical", 2: "Peer-review", 3: "AI", 4: "Web", 5: "Unverified"}
    content, stale = display_content(record)

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**{content}**")
        if stale:
            st.caption("Content was over-redacted during ingestion; showing structured fallback.")
        if record.tier == "hypothesis" and show_hypothesis_warning:
            st.caption("Hypothesis fact - not clinically verified.")
    with right:
        st.caption(f"Tier: {tier_label}")
        st.caption(f"Trust: {trust_label.get(record.trust_level, str(record.trust_level))}")
        st.caption(f"Source: {record.source_name[:30]}")


def render_ddi_warning(response_or_findings) -> None:
    findings = getattr(response_or_findings, "ddi_findings", response_or_findings) or []
    for finding in findings:
        if isinstance(finding, dict):
            severity = finding.get("severity", "")
            drug_a = finding.get("drug_a", "")
            drug_b = finding.get("drug_b", "")
            management = finding.get("management", "")
        else:
            severity = getattr(finding, "severity", "")
            drug_a = getattr(finding, "drug_a", "")
            drug_b = getattr(finding, "drug_b", "")
            management = getattr(finding, "management", "")
        if severity == "HIGH":
            st.error(f"HIGH SEVERITY INTERACTION: {drug_a} <-> {drug_b}. {management or 'Consult physician.'}")
        elif severity == "MEDIUM":
            st.warning(f"MEDIUM interaction: {drug_a} <-> {drug_b}. {management or 'Monitor closely.'}")
        else:
            st.info(f"Interaction: {drug_a} <-> {drug_b}")


def render_confidence(response: UnifiedResponse) -> None:
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{response.confidence:.0%}")
    cols[1].metric("Band", response.confidence_band.upper())
    cols[2].metric("Sources", len(response.sources_used))
    cols[3].metric("MKB facts used", len(response.mkb_facts_used))
    if response.confidence_band in ("low", "discarded") or response.safe_mode:
        st.warning("Low confidence response. Verify with a qualified clinician.")


def render_conflicts(sys_components: dict) -> None:
    conflicts = sys_components["sql"].get_records_requiring_review()
    if not conflicts:
        return

    with st.expander(f"{len(conflicts)} record(s) require review", expanded=False):
        for record in conflicts:
            content, stale = display_content(record)
            st.markdown(f"**{record.fact_type.upper()}**: {content}")
            if stale:
                st.caption("Content over-redacted during ingestion; showing structured fallback.")
            st.caption(f"Status: {record.status} | Source: {record.source_name}")
            accept, reject, defer = st.columns(3)
            if accept.button("Accept", key=f"accept_{record.id}"):
                sys_components["sql"].update_status(record.id, "active", "active")
                st.rerun()
            if reject.button("Reject", key=f"reject_{record.id}"):
                sys_components["sql"].update_status(record.id, "archived", "superseded")
                st.rerun()
            if defer.button("Defer", key=f"defer_{record.id}"):
                st.info("Deferred.")
            st.divider()


def render_query_tab(sys_components: dict) -> None:
    query = st.text_area(
        "Ask a medical question",
        placeholder="e.g. What does my EEG result mean for my epilepsy treatment?",
        height=100,
    )
    submit = st.button("Submit", type="primary")

    if submit and query.strip():
        with st.spinner("Processing..."):
            try:
                response = asyncio.run(sys_components["engine"].process(query))
                render_confidence(response)
                st.divider()
                render_ddi_warning(response)
                st.subheader("Response")
                st.markdown(response.synthesis)

                if response.mkb_facts_used:
                    with st.expander("MKB records used in this response"):
                        for record in response.mkb_facts_used:
                            render_mkb_record(record)

                if response.hypothesis_facts:
                    with st.expander(f"{len(response.hypothesis_facts)} hypothesis facts"):
                        for record in response.hypothesis_facts:
                            render_mkb_record(record)

                if ENABLE_ENRICHMENT and not response.safe_mode and sys_components["enrichment"].extractor is not None:
                    written = sys_components["enrichment"].enrich_from_response(response)
                    if written:
                        st.caption(f"Added {len(written)} hypothesis facts to MKB from this response.")
            except Exception as exc:
                st.error(f"Error processing query: {exc}")


def render_upload_tab(sys_components: dict) -> None:
    st.subheader("Upload Medical Document")
    specialty = st.selectbox("Specialty", ["neurology", "epilepsy", "gastroenterology", "urology", "general"])
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.button("Process Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(uploaded, tmp)
            tmp_path = Path(tmp.name)

        with st.spinner(f"Processing {uploaded.name}..."):
            try:
                result = sys_components["execution"].process_pdf(tmp_path, specialty=specialty)
            finally:
                tmp_path.unlink(missing_ok=True)

        st.caption(
            f"Extractor: {result.audit.get('extractor', result.extractor_result.get('extractor', 'unknown'))} | "
            f"Entities: {result.audit.get('entity_count', 0)} | "
            f"Confidence: {result.audit.get('confidence', 0):.2f}"
        )

        if result.outcome == "written":
            st.success(f"Added {result.written_count} records to MKB from {uploaded.name}.")
            for record in result.records[:10]:
                render_mkb_record(record, show_hypothesis_warning=False)
        elif result.outcome == "queued_for_review":
            st.warning(f"Queued {result.queued_count} records for review from {uploaded.name}.")
            for record in result.queued_records[:10]:
                render_mkb_record(record)
        elif result.outcome == "blocked_ddi":
            st.error("Document write blocked by medication interaction safety gate.")
            render_ddi_warning(result.ddi_findings)
            for record in result.blocked_records[:10]:
                render_mkb_record(record, show_hypothesis_warning=False)
        else:
            st.error(f"Unexpected execution outcome: {result.outcome}")


def render_mkb_tab(sys_components: dict) -> None:
    st.subheader("MKB Explorer")
    specialty_filter, tier_filter = st.columns(2)
    specialty = specialty_filter.selectbox("Specialty", ["all", "neurology", "epilepsy", "gastroenterology", "urology"])
    tier = tier_filter.selectbox("Tier", ["all", "active", "hypothesis", "quarantined"])

    specialty_query = None if specialty == "all" else specialty
    tier_query = None if tier == "all" else tier

    if specialty_query:
        records = sys_components["sql"].get_by_specialty(specialty_query, tier_query)
    else:
        with sys_components["sql"]._get_conn() as conn:
            query = "SELECT * FROM records"
            params = []
            if tier_query:
                query += " WHERE tier=?"
                params.append(tier_query)
            query += " ORDER BY first_recorded DESC LIMIT 50"
            rows = conn.execute(query, params).fetchall()
        records = [sys_components["sql"]._row_to_record(row) for row in rows]

    st.caption(f"{len(records)} records shown")
    for record in records:
        render_mkb_record(record)


def render_conflict_tab(sys_components: dict) -> None:
    try:
        from app.conflict_review import render_conflict_review
        from mkb.conflict_resolver import ConflictResolver

        resolver = ConflictResolver(DB_PATH, sql_store=sys_components["sql"])
        render_conflict_review(resolver)
    except Exception as exc:
        st.error(f"Conflict review unavailable: {exc}")


def main() -> None:
    sys_components = load_system()
    st.title("MedAI v1.1")
    st.caption("Personal medical intelligence. Local-first. Decision support only.")

    render_system_status(sys_components["state"])
    render_conflicts(sys_components)

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

    tab_query, tab_upload, tab_mkb, tab_conflicts = st.tabs(
        ["Query", "Upload Document", "MKB Explorer", "Conflict Review"]
    )
    with tab_query:
        render_query_tab(sys_components)
    with tab_upload:
        render_upload_tab(sys_components)
    with tab_mkb:
        render_mkb_tab(sys_components)
    with tab_conflicts:
        render_conflict_tab(sys_components)

    st.divider()
    st.caption("MedAI v1.1. Decision support only. Not a medical device. All outputs require clinical verification.")


if __name__ == "__main__":
    main()
