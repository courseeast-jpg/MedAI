"""MedAI Streamlit application with Phase 1 execution pipeline integration."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ACTIVE_CONNECTORS, ANTHROPIC_API_KEY, CHROMA_PATH, DB_PATH, ENABLE_ENRICHMENT
from app.operator_safety import (
    PHASE52_SAFETY_WARNING,
    PRIVACY_INVARIANT_GUIDANCE,
    RELEASE_NAME,
    SNAPSHOT_ID,
    build_result_summary,
    current_commit,
    detailed_operator_guidance,
    operator_guidance_catalog,
    privacy_mode_labels,
    status_badge,
)
from app.schemas import MKBRecord, SystemState, UnifiedResponse
from app.test_launcher import (
    LATEST_MD_REPORT,
    TEST_INPUT_DIR,
    clear_latest_test_reports,
    clear_test_input,
    ensure_test_launcher_dirs,
    list_test_input_files,
    remove_test_input_file,
    run_medai_test_batch,
    save_uploaded_test_file,
)
from execution.pipeline import ExecutionPipeline


PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|LOCATION|DATE|CONTACT_REMOVED|ID_REMOVED|URL_REMOVED|PHYSICIAN)\]")


st.set_page_config(
    page_title="MedAI v1.1",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


PHASE52_OPERATOR_TABS = ["Current Run", "Blind Audit", "Report Archive"]


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


def inject_phase52_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #faf9f7; color: #1f2933; }
        .medai-card {
            border: 1px solid #d8dee5;
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            margin-bottom: .75rem;
        }
        .medai-header {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem 1.1rem;
            margin-bottom: .75rem;
        }
        .safety-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .65rem;
            margin: .75rem 0;
        }
        .safety-cell {
            border: 1px solid #d8dee5;
            border-radius: 8px;
            background: #f8fafc;
            padding: .7rem .8rem;
        }
        .safety-cell span { color: #64748b; font-size: .78rem; }
        .safety-cell strong { display: block; font-size: 1rem; margin-top: .15rem; }
        .warning-banner {
            border-left: 4px solid #d97706;
            background: #fff7ed;
            color: #7c2d12;
            padding: .75rem .9rem;
            border-radius: 8px;
            margin: .75rem 0 1rem 0;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: .25rem .65rem;
            font-size: .8rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .badge-accepted { background: #dcfce7; color: #166534; border-color: #86efac; }
        .badge-review { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
        .badge-ocr { background: #ffedd5; color: #9a3412; border-color: #fdba74; }
        .badge-empty { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
        .badge-error { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
        .badge-privacy { background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }
        .reason-chip {
            display: inline-block;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            color: #334155;
            border-radius: 999px;
            padding: .16rem .5rem;
            margin: .12rem .18rem .12rem 0;
            font-size: .78rem;
        }
        .muted-label { color: #64748b; font-size: .84rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_operator_safety_panel(run_id: str | None = None, timestamp: str | None = None) -> None:
    labels = privacy_mode_labels()
    st.markdown(
        f"""
        <div class="medai-header">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
            <div>
              <div class="muted-label">MedAI v2 · OCR / Layout HITL</div>
              <h2 style="margin:.15rem 0 .25rem 0;">Operator @local</h2>
              <div class="muted-label">Snapshot: {SNAPSHOT_ID} · Commit: {current_commit()}</div>
              <div class="muted-label">Run ID: {run_id or "not started"} · Timestamp: {timestamp or "not available"}</div>
            </div>
            <div><span class="badge badge-privacy">SAFE LOCAL MODE</span></div>
          </div>
          <div class="safety-strip">
            <div class="safety-cell"><span>Mode</span><strong>HITL</strong></div>
            <div class="safety-cell"><span>Local-only</span><strong>{labels.local_only}</strong></div>
            <div class="safety-cell"><span>External APIs</span><strong>{labels.external_apis}</strong></div>
            <div class="safety-cell"><span>PII scrub required</span><strong>{labels.pii_scrub_required}</strong></div>
          </div>
        </div>
        <div class="warning-banner">{PHASE52_SAFETY_WARNING}</div>
        """,
        unsafe_allow_html=True,
    )


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
    st.caption("Process one document through the HITL pipeline. Review all non-accepted outputs before use.")
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
            f"Confidence: {result.audit.get('confidence', 0):.2f} | "
            f"Validation: {result.validation_status}"
        )
        render_operator_result_panel(result)
        if result.validation_errors:
            st.caption(f"Validation issues: {', '.join(error['code'] for error in result.validation_errors)}")

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


def render_operator_result_panel(result) -> None:
    summary = build_result_summary(result)
    st.markdown("**Operator Review Summary**")
    status = summary["final_status"]
    if status == "accepted":
        st.success(f"{status}: {summary['operator_next_action']}")
    elif status == "review_ocr_quality" or status == "empty":
        st.error(f"{status}: {summary['operator_next_action']}")
    else:
        st.warning(f"{status}: {summary['operator_next_action']}")

    cols = st.columns(4)
    cols[0].metric("OCR/Layout", summary["ocr_layout_quality_band"])
    cols[1].metric("OCR engine", summary["selected_ocr_engine"])
    cols[2].metric("Lab rows", summary["parsed_lab_row_count"])
    cols[3].metric("Privacy gate", summary["privacy_gate_status"])
    st.caption(f"Reason codes: {', '.join(summary['reason_codes']) or 'none'}")
    st.caption(
        " | ".join(
            [
                f"Document type: {summary['document_type']}",
                f"Cyrillic ratio: {summary['cyrillic_ratio'] if summary['cyrillic_ratio'] is not None else 'unknown'}",
                f"Lab table detected: {'yes' if summary['lab_table_detected'] else 'no'}",
                f"Lab coverage: {summary['lab_coverage_band']}",
            ]
        )
    )
    st.caption(
        f"External API used: {'yes' if summary['external_api_used'] else 'no'} | "
        f"Payload redacted: {'yes' if summary['payload_redacted'] else 'no'}"
    )
    if summary["external_api_used"] and not summary["payload_redacted"]:
        st.error("Not safe for cloud processing: external use requires a redacted payload.")


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


def render_current_run_tab(sys_components: dict) -> None:
    ensure_test_launcher_dirs()
    st.subheader("Current Run")
    st.caption("Upload PDF/TXT files into `test_input/`, review the active queue, then start a new local run.")

    specialty = st.selectbox(
        "Test specialty",
        ["general", "neurology", "epilepsy", "gastroenterology", "urology"],
        key="test_launcher_specialty",
    )
    uploaded_files = st.file_uploader(
        "Add PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supported files are persisted into test_input/ for local processing.",
    )
    if uploaded_files:
        saved_uploads = st.session_state.get("test_launcher_saved_uploads", {})
        if not isinstance(saved_uploads, dict):
            saved_uploads = {}
            st.session_state["test_launcher_saved_uploads"] = saved_uploads
        saved = []
        for uploaded_file in uploaded_files:
            upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
            prior_path = Path(saved_uploads.get(upload_key, ""))
            if prior_path.exists() and prior_path.parent == TEST_INPUT_DIR:
                continue
            destination = save_uploaded_test_file(uploaded_file)
            saved.append(destination)
            saved_uploads[upload_key] = str(destination.resolve())
        if saved:
            st.success(f"Added {len(saved)} file(s) to test_input/.")

    files = list_test_input_files()
    active_run = st.session_state.get("phase52_current_run")
    run_state = "Idle"
    if active_run:
        run_state = "Complete" if not active_run.get("failed") else "Failed"

    control_cols = st.columns([1, 1, 1])
    if control_cols[0].button("Clear queue"):
        removed = clear_test_input()
        st.session_state["test_launcher_saved_uploads"] = {}
        st.session_state.pop("phase52_current_run", None)
        st.success(f"Cleared {len(removed)} queued file(s) from test_input/.")
        st.rerun()
    if control_cols[1].button("Clear latest report"):
        removed = clear_latest_test_reports()
        st.session_state.pop("phase52_current_run", None)
        st.success(f"Cleared {len(removed)} latest report file(s).")
        st.rerun()
    if control_cols[2].button("Start new run", type="primary"):
        if not files:
            st.warning("No supported files waiting in test_input/.")
        else:
            with st.spinner("Running MedAI local test batch..."):
                summary = run_medai_test_batch(sys_components["execution"], specialty=specialty)
            st.session_state["phase52_current_run"] = {
                "timestamp": summary.timestamp,
                "run_id": summary.run_id,
                "accepted_count": summary.accepted_count,
                "review_count": summary.review_count,
                "error_count": summary.error_count,
                "results": summary.results,
                "failed": summary.error_count > 0,
            }
            st.success(
                f"Run complete: {summary.accepted_count} accepted, "
                f"{summary.review_count} review, {summary.error_count} errors."
            )
            st.rerun()

    render_queue_panel(files)
    active_run = st.session_state.get("phase52_current_run")
    render_run_status_panel(active_run, run_state="Complete" if active_run else run_state)
    render_operator_guidance_panel()
    if active_run:
        st.markdown("**Per-file results**")
        for result in active_run.get("results", []):
            render_run_result_card(result)
    else:
        st.caption("No current run results. Previous reports are available in Report Archive.")


def render_queue_panel(files: list[Path]) -> None:
    st.markdown("**Active upload queue**")
    st.metric("Queue count", len(files))
    if not files:
        st.caption("No supported files waiting.")
        return
    header = st.columns([4, 2, 2, 1])
    header[0].caption("Filename")
    header[1].caption("Size")
    header[2].caption("Status")
    header[3].caption("Remove")
    for path in files:
        row = st.columns([4, 2, 2, 1])
        row[0].caption(path.name)
        row[1].caption(format_bytes(path.stat().st_size))
        row[2].caption("queued")
        if row[3].button("Remove", key=f"remove_queued_{path.name}"):
            remove_test_input_file(path.name)
            st.rerun()


def render_run_status_panel(active_run: dict | None, *, run_state: str) -> None:
    counts = current_run_counts(active_run)
    st.markdown("**Run status**")
    st.markdown(f"<span class='badge badge-privacy'>{run_state}</span>", unsafe_allow_html=True)
    cols = st.columns(5)
    metric_specs = [
        ("Accepted", counts["accepted"], "passed all gates · spot-check"),
        ("Review", counts["review"], "manual reconciliation"),
        ("OCR Review", counts["ocr_review"], "do not trust extraction"),
        ("Empty", counts["empty"], "no usable extraction"),
        ("Errors", counts["errors"], "processing failed"),
    ]
    for col, (label, value, help_text) in zip(cols, metric_specs):
        col.metric(label, value, help=help_text)


def current_run_counts(active_run: dict | None) -> dict[str, int]:
    if not active_run:
        return {"accepted": 0, "review": 0, "ocr_review": 0, "empty": 0, "errors": 0}
    results = list(active_run.get("results") or [])
    return {
        "accepted": sum(1 for item in results if item_status(item) == "accepted"),
        "review": sum(1 for item in results if item_status(item) == "review"),
        "ocr_review": sum(1 for item in results if item_status(item) == "review_ocr_quality"),
        "empty": sum(1 for item in results if item_status(item) == "empty"),
        "errors": sum(1 for item in results if item_status(item) == "error"),
    }


def item_status(item: dict) -> str:
    if item.get("status") == "error":
        return "error"
    if item.get("empty_extraction_flag") or item.get("validation_status") == "empty":
        return "empty"
    if item.get("status") == "review_ocr_quality":
        return "review_ocr_quality"
    return str(item.get("status") or "review")


def render_run_result_card(item: dict) -> None:
    status = item_status(item)
    badge = status_badge(status)
    reason_codes = item.get("reason_codes") or item.get("classification_reason_codes") or []
    if not reason_codes and item.get("error"):
        reason_codes = ["processing_error"]
    if not reason_codes and item.get("validation_status"):
        reason_codes = [str(item.get("validation_status"))]
    st.markdown("<div class='medai-card'>", unsafe_allow_html=True)
    st.markdown(
        f"### {safe_display_name(item)} &nbsp; <span class='badge {badge['class']}'>{badge['label']}</span>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Confidence", value_or_unknown(item.get("confidence")))
    cols[1].metric("Extractor", value_or_unknown(item.get("selected_extractor")))
    cols[2].metric("Document type", value_or_unknown(item.get("document_type")))
    cols[3].metric("OCR quality", value_or_unknown(item.get("ocr_quality_band") or item.get("input_quality_band")))
    details = st.columns(4)
    details[0].caption(f"Document ID: {item.get('file_id') or item.get('file_name') or 'unknown'}")
    details[1].caption(f"File size: {format_bytes(int(item.get('file_size_bytes') or 0)) if item.get('file_size_bytes') else 'unknown'}")
    details[2].caption(f"Privacy mode: {item.get('privacy_gate_mode') or 'local_only'}")
    details[3].caption(f"External API used: {'Yes' if item.get('external_api_used') else 'No'}")
    st.caption(f"PII scrub: {pii_scrub_label(item)}")
    st.caption(f"Processed time: {item.get('processed_time') or item.get('timestamp') or 'unknown'}")
    st.info(detailed_operator_guidance(status))
    if reason_codes:
        chips = " ".join(f"<span class='reason-chip'>{code}</span>" for code in reason_codes)
        st.markdown(chips, unsafe_allow_html=True)
    with st.expander("Show raw run record", expanded=False):
        st.json(item)
    st.markdown("</div>", unsafe_allow_html=True)


def render_operator_guidance_panel() -> None:
    with st.expander("Operator guidance", expanded=True):
        for title, guidance in operator_guidance_catalog().items():
            st.markdown(f"**{title}:** {guidance}")


def render_blind_audit_tab(sys_components: dict) -> None:
    st.subheader("Blind Audit")
    st.caption("Put many PDFs into real_validation_input/")
    st.caption("Supported formats: PDF, TXT, RTF, TIF, TIFF, PNG, JPG, JPEG, BMP, WEBP.")
    st.caption("Run Blind Audit workflow: Phase53 local-only processing with PHI-safe public reports.")
    st.warning("Do not tune parsers during blind audit. Run first, review report second, change code only after audit is complete.")
    try:
        from scripts.run_phase53_blind_pdf_generalization_audit import (
            INPUT_DIR as BLIND_AUDIT_INPUT_DIR,
            JSON_REPORT as BLIND_AUDIT_JSON_REPORT,
            MD_REPORT as BLIND_AUDIT_MD_REPORT,
            OPERATOR_SUMMARY as BLIND_AUDIT_OPERATOR_SUMMARY,
            run_audit as run_blind_audit,
            supported_input_files as blind_audit_input_files,
        )

        blind_files = blind_audit_input_files(BLIND_AUDIT_INPUT_DIR)
        st.caption("Folder: real_validation_input/")
        st.metric("Files found", len(blind_files))
        if st.button("Run Phase53 Blind Audit from real_validation_input/", type="primary"):
            with st.spinner("Running local-only blind audit..."):
                report = run_blind_audit(pipeline=sys_components["execution"])
            st.session_state["phase53_blind_audit"] = report
            st.success(
                f"Blind audit complete: {report['accepted_count']} accepted, "
                f"{report['review_count']} review, {report['error_count']} errors."
            )

        report = st.session_state.get("phase53_blind_audit") or load_json_file(BLIND_AUDIT_JSON_REPORT)
        if report:
            render_blind_audit_summary(report)
            st.caption(f"Operator summary: {BLIND_AUDIT_OPERATOR_SUMMARY}")
            st.caption(f"Markdown report: {BLIND_AUDIT_MD_REPORT}")
            st.caption(f"JSON report: {BLIND_AUDIT_JSON_REPORT}")
            st.markdown("**Safe file IDs requiring attention**")
            attention = [item["file_id"] for item in report.get("results", []) if item.get("status") != "accepted"]
            st.write(attention or "None")
            render_phase54_review_section()
        render_phase57_full_corpus_section()
    except Exception as exc:
        st.error(f"Blind audit unavailable: {exc}")


def render_phase57_full_corpus_section() -> None:
    st.divider()
    st.subheader("Phase57 Full Corpus Inventory Audit")
    st.caption("Folder: full_corpus_input/")
    st.caption("Supported formats: PDF, TXT, RTF, TIF, TIFF, PNG, JPG, JPEG, BMP, WEBP.")
    st.caption("Inventory/discovery only. Public reports use safe IDs and hashes, not raw filenames.")
    try:
        from scripts.run_phase57_full_corpus_inventory_audit import (
            CLUSTERS_MD as PHASE57_CLUSTERS_MD,
            JSON_REPORT as PHASE57_JSON_REPORT,
            MD_REPORT as PHASE57_MD_REPORT,
            OPERATOR_SUMMARY as PHASE57_OPERATOR_SUMMARY,
            INPUT_DIR as PHASE57_INPUT_DIR,
            discover_corpus_files,
            run_inventory_audit,
        )

        corpus_files = discover_corpus_files(PHASE57_INPUT_DIR)
        st.metric("Corpus files found", len(corpus_files))
        if st.button("Run Phase57 Full Corpus Inventory Audit", type="primary"):
            with st.spinner("Running local-only full corpus inventory audit..."):
                report = run_inventory_audit()
            st.session_state["phase57_full_corpus_inventory"] = report
            st.success(
                f"Phase57 complete: {report['total_discovered']} discovered, "
                f"{report['total_processed']} supported processed, {report['errors']} errors."
            )
        report = st.session_state.get("phase57_full_corpus_inventory") or load_json_file(PHASE57_JSON_REPORT)
        if report:
            render_phase57_summary(report)
            st.caption(f"Operator summary: {PHASE57_OPERATOR_SUMMARY}")
            st.caption(f"Markdown report: {PHASE57_MD_REPORT}")
            st.caption(f"JSON report: {PHASE57_JSON_REPORT}")
            st.caption(f"Problem clusters: {PHASE57_CLUSTERS_MD}")
    except Exception as exc:
        st.error(f"Phase57 full corpus audit unavailable: {exc}")


def render_phase57_summary(report: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Accepted", int(report.get("accepted", 0)))
    cols[1].metric("Review", int(report.get("review", 0)))
    cols[2].metric("OCR Review", int(report.get("review_ocr_quality", 0)))
    cols[3].metric("Empty", int(report.get("empty", 0)))
    cols[4].metric("Errors", int(report.get("errors", 0)))
    st.caption(f"Conclusion: {report.get('conclusion', 'unknown')}")
    st.caption(f"External API used: {'Yes' if report.get('external_api_used') else 'No'}")
    clusters = report.get("problem_clusters") or {}
    if clusters:
        st.markdown("**Problem clusters**")
        st.json({name: len(ids) for name, ids in clusters.items()}, expanded=False)


def render_blind_audit_summary(report: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Accepted", int(report.get("accepted_count", 0)))
    cols[1].metric("Review", int(report.get("review_count", 0)))
    cols[2].metric("OCR Review", int(report.get("review_ocr_quality_count", 0)))
    cols[3].metric("Empty", int(report.get("empty_count", 0)))
    cols[4].metric("Errors", int(report.get("error_count", 0)))
    st.caption(f"Conclusion: {report.get('conclusion', 'unknown')}")
    st.caption(f"External API used: {'Yes' if report.get('external_api_used') else 'No'}")


def render_phase54_review_section() -> None:
    st.divider()
    st.subheader("Phase54 Operator Review Feedback")
    st.caption("Capture correct/incorrect/uncertain review feedback using safe file IDs only. Notes are private and ignored by Git.")
    try:
        from scripts.run_phase54_operator_review_feedback_summary import (
            DOCUMENT_CLASSES,
            JSON_REPORT as PHASE54_JSON_REPORT,
            MD_REPORT as PHASE54_MD_REPORT,
            PHASE53_REPORT,
            PRIVATE_FEEDBACK,
            REASONS,
            REPORT_DIR as PHASE54_REPORT_DIR,
            VERDICTS,
            run_summary as run_phase54_summary,
        )

        phase53_report = load_json_file(PHASE53_REPORT)
        if not phase53_report:
            st.warning("Phase53 public report is missing. Run Phase53 before capturing Phase54 feedback.")
            return
        feedback_path = PHASE54_REPORT_DIR / PRIVATE_FEEDBACK.name
        st.caption("Private feedback path: reports/phase54_operator_review_feedback/operator_feedback_PRIVATE.json")
        feedback_payload = load_json_file(feedback_path) or {"feedback": []}
        feedback_by_id = {row.get("safe_file_id"): row for row in feedback_payload.get("feedback", [])}
        updated_feedback = []
        for item in phase53_report.get("results", []):
            safe_id = item.get("file_id")
            filename_hash = item.get("filename_hash")
            existing = feedback_by_id.get(safe_id, {})
            with st.expander(f"Review {safe_id} · {filename_hash}", expanded=False):
                cols = st.columns(4)
                verdict = cols[0].selectbox(
                    "operator_verdict",
                    sorted(VERDICTS),
                    index=sorted(VERDICTS).index(existing.get("operator_verdict", "not_reviewed"))
                    if existing.get("operator_verdict", "not_reviewed") in sorted(VERDICTS)
                    else sorted(VERDICTS).index("not_reviewed"),
                    key=f"phase54_verdict_{safe_id}",
                )
                doc_class = cols[1].selectbox(
                    "operator_document_class",
                    sorted(DOCUMENT_CLASSES),
                    index=sorted(DOCUMENT_CLASSES).index(existing.get("operator_document_class", "unknown_other"))
                    if existing.get("operator_document_class", "unknown_other") in sorted(DOCUMENT_CLASSES)
                    else sorted(DOCUMENT_CLASSES).index("unknown_other"),
                    key=f"phase54_class_{safe_id}",
                )
                reason = cols[2].selectbox(
                    "operator_reason",
                    sorted(REASONS),
                    index=sorted(REASONS).index(existing.get("operator_reason", "other"))
                    if existing.get("operator_reason", "other") in sorted(REASONS)
                    else sorted(REASONS).index("other"),
                    key=f"phase54_reason_{safe_id}",
                )
                cols[3].caption(f"Status: {item.get('status')}")
                note = st.text_area(
                    "operator_note_PRIVATE",
                    value=existing.get("operator_note", ""),
                    help="Private local note. Do not enter text that should appear in public reports.",
                    key=f"phase54_note_{safe_id}",
                )
                updated_feedback.append(
                    {
                        "safe_file_id": safe_id,
                        "filename_hash": filename_hash,
                        "operator_verdict": verdict,
                        "operator_document_class": doc_class,
                        "operator_reason": reason,
                        "operator_note": note,
                        "reviewed_at": existing.get("reviewed_at") if verdict == "not_reviewed" else datetime.now(UTC).isoformat(),
                    }
                )
        if st.button("Save Phase54 Private Feedback"):
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            feedback_path.write_text(json.dumps({"feedback": updated_feedback}, indent=2), encoding="utf-8")
            st.success("Saved private feedback locally. This file is ignored by Git.")
        if st.button("Generate Phase54 Class-Level Review Summary", type="primary"):
            report = run_phase54_summary()
            st.success(f"Phase54 summary generated: {report['conclusion']}")
            st.caption(f"Markdown report: {PHASE54_MD_REPORT}")
            st.caption(f"JSON report: {PHASE54_JSON_REPORT}")
        phase54_report = load_json_file(PHASE54_JSON_REPORT)
        if phase54_report:
            st.markdown("**Phase54 class-level summary**")
            st.json(phase54_report.get("class_summary", {}), expanded=False)
    except Exception as exc:
        st.error(f"Phase54 feedback summary unavailable: {exc}")


def render_report_archive_tab() -> None:
    st.subheader("Report Archive")
    st.caption("Previous reports live here so current-run counters stay separate from historical output.")
    archives = [
        ("latest test run", LATEST_MD_REPORT, LATEST_MD_REPORT.with_suffix(".json")),
        (
            "phase53 blind audit",
            Path("reports/phase53_blind_generalization_audit/phase53_blind_generalization_audit_report.md"),
            Path("reports/phase53_blind_generalization_audit/phase53_blind_generalization_audit_report.json"),
        ),
        (
            "phase54 operator review feedback",
            Path("reports/phase54_operator_review_feedback/phase54_operator_review_feedback_report.md"),
            Path("reports/phase54_operator_review_feedback/phase54_operator_review_feedback_report.json"),
        ),
    ]
    for label, md_path, json_path in archives:
        render_archive_card(label, md_path, json_path)


def render_archive_card(label: str, md_path: Path, json_path: Path) -> None:
    payload = load_json_file(json_path)
    st.markdown("<div class='medai-card'>", unsafe_allow_html=True)
    st.markdown(f"**{label.title()}**")
    st.caption(f"Markdown: {md_path}")
    st.caption(f"JSON: {json_path}")
    if payload:
        st.caption(f"Generated timestamp: {payload.get('timestamp') or payload.get('generated_at') or 'unknown'}")
        st.caption(f"Status/conclusion: {payload.get('conclusion') or payload.get('run_status') or 'available'}")
    else:
        st.caption("No report found.")
    st.markdown("</div>", unsafe_allow_html=True)


def load_json_file(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def safe_display_name(item: dict) -> str:
    if item.get("original_filename_redacted"):
        return str(item["original_filename_redacted"])
    return str(item.get("file_name") or item.get("filename") or item.get("file_id") or "document")


def pii_scrub_label(item: dict) -> str:
    if item.get("payload_redacted") or item.get("pii_scrub_passed"):
        return "Passed"
    if item.get("pii_scrub_failed"):
        return "Failed"
    return "Unknown"


def value_or_unknown(value) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def format_bytes(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def main() -> None:
    inject_phase52_styles()
    sys_components = load_system()
    render_operator_safety_panel()
    render_system_status(sys_components["state"])

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

    tab_current, tab_blind, tab_archive = st.tabs(PHASE52_OPERATOR_TABS)
    with tab_current:
        render_current_run_tab(sys_components)
    with tab_blind:
        render_blind_audit_tab(sys_components)
    with tab_archive:
        render_report_archive_tab()

    st.divider()
    st.caption(PRIVACY_INVARIANT_GUIDANCE)


if __name__ == "__main__":
    main()
