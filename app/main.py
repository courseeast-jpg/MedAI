"""MedAI Streamlit application with Phase 1 execution pipeline integration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ACTIVE_CONNECTORS, ANTHROPIC_API_KEY, CHROMA_PATH, DB_PATH, ENABLE_ENRICHMENT
from app.lab_document_metadata import reason_label_for_validation, review_reason_for_result
from app.mkb_all_records_qa_comparator import (
    QA_STATUSES,
    build_all_records_qa_comparator,
    get_comparator_record_detail,
    readable_record_view,
    representative_proof_records,
    save_qa_status,
)
from app.mkb_explorer_model import build_mkb_explorer_model
from app.mkb_staging_payload_reader import build_staging_quality_view, get_staging_detail
from app.operator_compact_styles import COMPACT_OPERATOR_CSS
from app.operator_ui_model import (
    ADVANCED_TABS,
    DEFAULT_PRIMARY_TABS,
    MKB_EXPLORER_TAB,
    REVIEW_QUEUE_TAB,
    RUN_REVIEW_TAB,
    SOURCE_COMPARISON_DISCLAIMER,
    operator_tabs as operator_ui_tabs,
)
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
from app.specialty_selection import (
    DEFAULT_SPECIALTY_KEY,
    specialty_key_from_label,
    specialty_keys_for_ui,
    specialty_label,
    specialty_labels_for_ui,
    validate_specialty_key,
)
from app.test_launcher import (
    LATEST_MD_REPORT,
    RUN_REVIEW_UPLOAD_TYPES,
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
from app.startup_preflight import StartupState, initialize_startup_state


PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|LOCATION|DATE|CONTACT_REMOVED|ID_REMOVED|URL_REMOVED|PHYSICIAN)\]")


st.set_page_config(
    page_title="MedAI v1.1",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRIMARY_OPERATOR_TABS = list(DEFAULT_PRIMARY_TABS)
ADVANCED_OPERATOR_TABS = list(ADVANCED_TABS)
DOCUMENT_CATEGORY_OPTIONS = [
    "General",
    "Lab result",
    "Urinalysis",
    "Imaging report",
    "Pathology report",
    "Treatment plan",
    "Clinical note",
    "Other / needs review",
]

TERMINOLOGY_LOOKUP_TAB = "Terminology Lookup"
VERTEX_DECISION_AUDIT_TAB = "Vertex Decision Audit"
SPECIALTY_SESSION_KEY = "medai_selected_specialty"
PERSISTED_UPLOAD_FINGERPRINTS_KEY = "test_launcher_persisted_upload_fingerprints"
PERSISTED_UPLOAD_GENERATION_KEY = "test_launcher_persisted_upload_generation"
UPLOAD_WIDGET_VERSION_KEY = "test_launcher_upload_widget_version"
TERMINOLOGY_LOOKUP_UI_ENV_VAR = "MEDAI_TERMINOLOGY_LOOKUP_UI_ENABLED"
OPERATOR_FIRST_VISIBLE_TAB_LABELS = ["Run & Review", "MKB Explorer", "Review Queue"]
ADVANCED_OPERATOR_TAB_LABELS = [
    "Operator Control Panel",
    "Validation Batch Audit",
    "Validation History",
    "Safety & Governance",
    "Terminology Admin",
    VERTEX_DECISION_AUDIT_TAB,
]


def operator_tab_labels(show_advanced_tools: bool) -> list[str]:
    """Deterministic operator tab label list (additive; testable without Streamlit)."""
    labels = [RUN_REVIEW_TAB, MKB_EXPLORER_TAB, REVIEW_QUEUE_TAB]
    if show_advanced_tools:
        labels.extend(ADVANCED_OPERATOR_TAB_LABELS)
    return labels
REVIEW_QUEUE_SOURCE_COMPARISON_DISCLAIMER = (
    "Accept only after comparing with source. This does not clinically interpret the result."
)
SUPPORTED_FILE_TYPE_COPY = "Supported file types: PDF, TXT, PNG, JPG/JPEG, TIFF/TIF, BMP, DOCX"

# Backward-compatible export for older tests/importers. These are current
# visible labels; advanced pages are shown only after the operator opts in.
PHASE52_OPERATOR_TABS = PRIMARY_OPERATOR_TABS + ADVANCED_OPERATOR_TABS


def operator_tabs(show_advanced_tools: bool = False) -> list[str]:
    return operator_ui_tabs(show_advanced_tools)


def navigation_subtitle(tab_label: str) -> str:
    subtitles = {
        "Validation Batch Audit": "Run a controlled local test batch and review summary results.",
        "Validation History": "Previous validation and audit reports.",
        "Safety & Governance": "Safety checks for privacy, knowledge state, and controlled clinical logic.",
        "Terminology Admin": "Check terminology files, license status, and import readiness.",
    }
    return subtitles.get(tab_label, "")


def sidebar_status_labels(*, enrichment_enabled: bool) -> dict[str, str]:
    return {
        "knowledge_base": "Knowledge base",
        "active": "Active",
        "draft_facts": "Draft facts",
        "connector_status": "Medical connector active",
        "enrichment_status": "Enrichment enabled" if enrichment_enabled else "Enrichment disabled",
    }


def uploaded_file_fingerprint(uploaded_file) -> str:
    content = uploaded_file_bytes_for_fingerprint(uploaded_file)
    digest = hashlib.sha256(content).hexdigest()
    name = Path(str(getattr(uploaded_file, "name", ""))).name
    size = getattr(uploaded_file, "size", len(content))
    return f"{name}:{size}:{digest}"


def uploaded_file_bytes_for_fingerprint(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    if not hasattr(uploaded_file, "read"):
        raise TypeError("Uploaded file object does not expose getbuffer() or read().")
    position = None
    if hasattr(uploaded_file, "tell"):
        try:
            position = uploaded_file.tell()
        except Exception:
            position = None
    data = bytes(uploaded_file.read())
    if position is not None and hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(position)
        except Exception:
            pass
    return data


def current_upload_generation(session_state) -> int:
    return int(session_state.get(UPLOAD_WIDGET_VERSION_KEY, 0) or 0)


def current_upload_widget_key(session_state) -> str:
    return f"test_launcher_uploads_{current_upload_generation(session_state)}"


def persisted_upload_fingerprints(session_state) -> set[str]:
    generation = current_upload_generation(session_state)
    persisted_generation = session_state.get(PERSISTED_UPLOAD_GENERATION_KEY)
    raw = session_state.get(PERSISTED_UPLOAD_FINGERPRINTS_KEY, set())
    if persisted_generation is not None and int(persisted_generation) != generation:
        return set()
    if isinstance(raw, set):
        return set(raw)
    return set(raw or [])


def set_persisted_upload_fingerprints(session_state, fingerprints: set[str]) -> None:
    session_state[PERSISTED_UPLOAD_FINGERPRINTS_KEY] = set(fingerprints)
    session_state[PERSISTED_UPLOAD_GENERATION_KEY] = current_upload_generation(session_state)


def persist_uploaded_files_once(uploaded_files, session_state, *, save_func=save_uploaded_test_file) -> list[Path]:
    persisted = persisted_upload_fingerprints(session_state)
    saved: list[Path] = []
    for uploaded_file in uploaded_files or []:
        fingerprint = uploaded_file_fingerprint(uploaded_file)
        if fingerprint in persisted:
            continue
        destination = save_func(uploaded_file)
        saved.append(destination)
        persisted.add(fingerprint)
    set_persisted_upload_fingerprints(session_state, persisted)
    return saved


def selected_upload_count(uploaded_files) -> int:
    return len(list(uploaded_files or []))


def queue_display_state(*, queued_count: int, selected_count: int) -> dict[str, object]:
    return {
        "queued_count": queued_count,
        "selected_count": selected_count,
        "start_enabled": queued_count > 0,
        "message": (
            f"Ready: {queued_count} documents waiting."
            if queued_count
            else "Files selected. Add selected files to queue."
            if selected_count
            else "No documents queued."
        ),
    }


def visible_current_run(active_run: dict | None, *, queued_count: int, selected_count: int) -> dict | None:
    if active_run and active_run.get("failed") and not queued_count and not selected_count:
        return None
    return active_run


def start_run_state_reason(queue_state: dict, *, active_run: dict | None = None) -> dict[str, object]:
    queued_count = int(queue_state.get("queued_count", 0) or 0)
    selected_count = int(queue_state.get("selected_count", 0) or 0)
    enabled = bool(queue_state.get("start_enabled", False))
    if enabled:
        reason = f"Ready: {queued_count} documents waiting."
    elif active_run:
        reason = "Run complete. Add files to start another run."
    elif selected_count:
        reason = "Start disabled: selected files must be added to the queue first."
    else:
        reason = "No documents queued. Add supported files to start."
    return {"enabled": enabled, "reason": reason}


def current_run_status_message(*, queue_state: dict, active_run: dict | None) -> str:
    if active_run:
        return "Run complete. Review results below."
    return str(queue_state.get("message") or "No documents queued.")


def operator_file_state_label(item: dict) -> str:
    status = item_status(item)
    if status == "accepted":
        return "Accepted"
    if status == "review_ocr_quality":
        return "No text found"
    if status == "error":
        return "Error"
    if str(item.get("outcome") or "") == "unsupported":
        return "Unsupported file"
    if item.get("running"):
        return "Running"
    if status == "review":
        return "Needs review"
    return "Queued"


def operator_console_redesign_static_model() -> dict[str, object]:
    return {
        "default_tabs": [RUN_REVIEW_TAB, MKB_EXPLORER_TAB, REVIEW_QUEUE_TAB],
        "safety_banner": "Review required - not for diagnosis.",
        "safety_pills": ["Local only", "Cloud APIs off", "Privacy check on", "Human review"],
        "supported_file_types": SUPPORTED_FILE_TYPE_COPY,
        "review_queue_actions": ["Accept after source comparison", "Reject", "Defer"],
        "external_api_used": False,
        "auto_accept": False,
    }


def reset_upload_persistence(session_state) -> None:
    version = int(session_state.get(UPLOAD_WIDGET_VERSION_KEY, 0) or 0)
    session_state[UPLOAD_WIDGET_VERSION_KEY] = version + 1
    set_persisted_upload_fingerprints(session_state, set())


def clear_queue_action(session_state, *, clear_func=clear_test_input) -> list[Path]:
    removed = clear_func()
    reset_upload_persistence(session_state)
    session_state.pop("phase52_current_run", None)
    return removed


def clear_last_report_action(session_state, *, clear_func=clear_latest_test_reports) -> list[Path]:
    removed = clear_func()
    session_state.pop("phase52_current_run", None)
    return removed


def selected_specialty_from_state(session_state) -> str:
    return validate_specialty_key(session_state.get(SPECIALTY_SESSION_KEY))


def render_specialty_selector(session_state, *, key: str) -> str:
    current = selected_specialty_from_state(session_state)
    labels = specialty_labels_for_ui()
    keys = specialty_keys_for_ui()
    index = keys.index(current) if current in keys else keys.index(DEFAULT_SPECIALTY_KEY)
    selected_label = st.selectbox(
        "Medical specialty / domain",
        labels,
        index=index,
        key=key,
        help="Used only to organize extracted facts in the MKB. It does not diagnose or interpret results.",
    )
    selected_key = specialty_key_from_label(selected_label)
    session_state[SPECIALTY_SESSION_KEY] = selected_key
    st.caption(
        "Used only to organize extracted facts in the MKB. It does not diagnose or interpret results."
    )
    return selected_key


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
    """Componentized startup. See MEDAI-UI-STARTUP-RESILIENCE-08.

    SQLite is critical: if it fails, the UI runs in diagnostics-only
    mode. VectorStore, QualityGate, connectors, scorer, synthesizer,
    DecisionEngine, and EnrichmentEngine are optional — any failure
    there leaves the local clinical-review workflow available with a
    "Vector/semantic index unavailable" banner. ExecutionPipeline is
    built whenever SQLite is OK, with ``vector_store=None`` and
    ``quality_gate=None`` as needed.
    """
    return build_system_components()


def build_system_components(
    *,
    db_path: Path | None = None,
    chroma_path: Path | None = None,
    db_encryption_key: str | None = None,
    sqlite_store_factory: Any | None = None,
    vector_store_factory: Any | None = None,
    quality_gate_factory: Any | None = None,
    connector_registry_factory: Any | None = None,
    medication_safety_gate_factory: Any | None = None,
    response_scorer_factory: Any | None = None,
    claude_synthesizer_factory: Any | None = None,
    decision_engine_factory: Any | None = None,
    execution_pipeline_factory: Any | None = None,
    enrichment_engine_factory: Any | None = None,
    pii_stripper_factory: Any | None = None,
) -> dict:
    """Build the system components dict with per-component try/except.

    Every dependency import sits inside its try block so a broken
    optional dependency cannot cascade into a misleading
    "MKB initialization failed" banner. The factory hooks let tests
    inject simulated failures (e.g., a VectorStore that raises a
    Pydantic ConfigValidationError) without touching production code.
    """
    db_path = db_path or DB_PATH
    chroma_path = chroma_path or CHROMA_PATH
    db_key = db_encryption_key or os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")

    component_errors: list[tuple[str, str]] = []
    sqlite_store_initialized = False
    vector_store_initialized = False
    quality_gate_initialized = False
    execution_pipeline_initialized = False

    sql = None
    vec = None
    quality_gate = None
    connectors: dict = {}
    medication_gate = None
    scorer = None
    synthesizer = None
    engine = None
    execution = None
    enrichment = None

    # 1. SQLiteStore (critical).
    try:
        if sqlite_store_factory is not None:
            sql = sqlite_store_factory(db_path, db_key)
        else:
            from mkb.sqlite_store import SQLiteStore

            sql = SQLiteStore(db_path, db_key)
        sqlite_store_initialized = True
    except Exception as exc:
        component_errors.append(("SQLiteStore", type(exc).__name__))

    # 2. VectorStore (optional).
    try:
        if sqlite_store_initialized:
            if vector_store_factory is not None:
                vec = vector_store_factory(chroma_path)
            else:
                from mkb.vector_store import VectorStore

                vec = VectorStore(chroma_path)
            vector_store_initialized = True
    except Exception as exc:
        component_errors.append(("VectorStore", type(exc).__name__))

    # 3. QualityGate (depends on SQLite + Vector).
    try:
        if sqlite_store_initialized and vector_store_initialized:
            if quality_gate_factory is not None:
                quality_gate = quality_gate_factory(sql, vec)
            else:
                from mkb.quality_gate import QualityGate

                quality_gate = QualityGate(sql, vec)
            quality_gate_initialized = True
    except Exception as exc:
        component_errors.append(("QualityGate", type(exc).__name__))

    # 4. Connector registry (optional).
    try:
        if connector_registry_factory is not None:
            connectors = connector_registry_factory() or {}
        else:
            from external_apis.connectors import build_connector_registry

            connectors = build_connector_registry() or {}
    except Exception as exc:
        component_errors.append(("ConnectorRegistry", type(exc).__name__))
        connectors = {}

    # 5. MedicationSafetyGate (depends on SQLite).
    try:
        if sqlite_store_initialized:
            if medication_safety_gate_factory is not None:
                medication_gate = medication_safety_gate_factory(
                    connectors.get("patientnotes_ddi"), sql
                )
            else:
                from decision.medication_safety import MedicationSafetyGate

                medication_gate = MedicationSafetyGate(
                    connectors.get("patientnotes_ddi"), sql
                )
    except Exception as exc:
        component_errors.append(("MedicationSafetyGate", type(exc).__name__))

    # 6. ResponseScorer (depends on Vector).
    try:
        if vector_store_initialized:
            if response_scorer_factory is not None:
                scorer = response_scorer_factory(vec, medication_gate)
            else:
                from decision.response_scorer import ResponseScorer

                scorer = ResponseScorer(vec, medication_gate)
    except Exception as exc:
        component_errors.append(("ResponseScorer", type(exc).__name__))

    # 7. ClaudeSynthesizer (always built; degraded when API key absent).
    try:
        if claude_synthesizer_factory is not None:
            synthesizer = claude_synthesizer_factory(
                ANTHROPIC_API_KEY, "claude-sonnet-4-20250514"
            )
        else:
            from external_apis.connectors import ClaudeSynthesizer

            synthesizer = ClaudeSynthesizer(
                ANTHROPIC_API_KEY, "claude-sonnet-4-20250514"
            )
    except Exception as exc:
        component_errors.append(("ClaudeSynthesizer", type(exc).__name__))

    state = SystemState(
        claude_available=bool(ANTHROPIC_API_KEY) and synthesizer is not None,
        active_connectors=ACTIVE_CONNECTORS,
    )

    # 8. DecisionEngine (depends on SQLite + Vector + others).
    try:
        if sqlite_store_initialized and vector_store_initialized and scorer is not None and synthesizer is not None:
            if decision_engine_factory is not None:
                engine = decision_engine_factory(
                    sql, vec, scorer, medication_gate, connectors, synthesizer, state
                )
            else:
                from decision.decision_engine import DecisionEngine

                engine = DecisionEngine(
                    sql, vec, scorer, medication_gate, connectors, synthesizer, state
                )
    except Exception as exc:
        component_errors.append(("DecisionEngine", type(exc).__name__))

    # 9. ExecutionPipeline (built whenever SQLite is OK).
    try:
        if sqlite_store_initialized:
            pii_stripper = None
            try:
                if pii_stripper_factory is not None:
                    pii_stripper = pii_stripper_factory()
                else:
                    from extraction.pii_stripper import PIIStripper

                    pii_stripper = PIIStripper()
            except Exception as pii_exc:
                component_errors.append(("PIIStripper", type(pii_exc).__name__))
                pii_stripper = None
            if execution_pipeline_factory is not None:
                execution = execution_pipeline_factory(
                    sql_store=sql,
                    vector_store=vec,
                    quality_gate=quality_gate,
                    medication_gate=medication_gate,
                    pii_stripper=pii_stripper,
                )
            else:
                execution = ExecutionPipeline(
                    sql_store=sql,
                    vector_store=vec,
                    quality_gate=quality_gate,
                    medication_gate=medication_gate,
                    pii_stripper=pii_stripper,
                )
            execution_pipeline_initialized = True
    except Exception as exc:
        component_errors.append(("ExecutionPipeline", type(exc).__name__))

    # 10. EnrichmentEngine (depends on SQLite + Vector).
    try:
        if sqlite_store_initialized and vector_store_initialized:
            if enrichment_engine_factory is not None:
                enrichment = enrichment_engine_factory(
                    None, sql, vec, quality_gate, medication_gate
                )
            else:
                from enrichment.enrichment_engine import EnrichmentEngine

                enrichment = EnrichmentEngine(
                    None, sql, vec, quality_gate, medication_gate
                )
    except Exception as exc:
        component_errors.append(("EnrichmentEngine", type(exc).__name__))

    return {
        "sql": sql,
        "vec": vec,
        "quality_gate": quality_gate,
        "engine": engine,
        "execution": execution,
        "enrichment": enrichment,
        "state": state,
        "med_gate": medication_gate,
        "component_status": {
            "sqlite_store_initialized": sqlite_store_initialized,
            "vector_store_initialized": vector_store_initialized,
            "quality_gate_initialized": quality_gate_initialized,
            "execution_pipeline_initialized": execution_pipeline_initialized,
            "component_errors": list(component_errors),
        },
    }


def render_system_status(state: SystemState, *, show_advanced_tools: bool = False) -> None:
    if state.safe_mode:
        st.caption(f"Local safe mode active. Cloud APIs off. Reason: {state.safe_mode_reason or 'API unavailable'}")
    elif not state.claude_available:
        if show_advanced_tools:
            st.caption("Cloud API connector not configured. Local workflow remains available.")
    else:
        return


def render_degraded_startup_panel(startup: StartupState) -> None:
    diagnostics = startup.diagnostics.safe_public_summary()
    st.error("MKB initialization failed. MedAI started in diagnostics-only mode.")
    st.warning("No clinical processing started. Avoid manual database deletion.")
    st.markdown("### Startup Diagnostics")
    st.json(diagnostics)
    st.markdown("### Safe Operator Actions")
    for item in diagnostics.get("safe_operator_guidance", []):
        st.write(f"- {item}")
    try:
        from app.operator_control_panel import render_operator_control_panel

        render_operator_control_panel()
    except Exception as _exc:
        st.error(f"Operator Control Panel unavailable: {_exc}")


def render_degraded_vector_banner(startup: StartupState) -> None:
    """MEDAI-UI-STARTUP-RESILIENCE-08: shown when SQLite is OK but the
    vector store (Chroma) failed to initialize. The local clinical
    review workflow continues; semantic / vector search is disabled.
    """
    diagnostics = startup.diagnostics.safe_public_summary()
    st.warning(
        "Vector/semantic index unavailable. SQLite MKB and local review "
        "workflow remain available."
    )
    st.caption(
        "Review required. MedAI does not diagnose, recommend treatment, "
        "interpret medications, or accept extracted values on its own."
    )
    with st.expander("Startup diagnostics (degraded vector)", expanded=False):
        st.json(diagnostics)
        for item in diagnostics.get("safe_operator_guidance", []):
            st.write(f"- {item}")


def render_degraded_pipeline_banner(startup: StartupState) -> None:
    """MEDAI-UI-STARTUP-RESILIENCE-09: shown when SQLite is OK but the
    ExecutionPipeline (or both pipeline + vector) failed to initialize.
    MKB Explorer remains available; Run & Review is degraded.
    """
    diagnostics = startup.diagnostics.safe_public_summary()
    st.warning(
        "Processing pipeline unavailable. SQLite MKB remains available. "
        "Run local validation repair if document processing is needed."
    )
    st.code("python scripts/run_medai_local_self_healing_validation_07.py", language="bash")
    st.caption(
        "Review required. MedAI does not diagnose, recommend treatment, "
        "interpret medications, or accept extracted values on its own."
    )
    with st.expander("Startup diagnostics (degraded pipeline)", expanded=False):
        st.json(diagnostics)
        for item in diagnostics.get("safe_operator_guidance", []):
            st.write(f"- {item}")


def render_run_review_unavailable_panel() -> None:
    """MEDAI-UI-STARTUP-RESILIENCE-09: replaces the Run & Review body
    when ``sys_components['execution']`` is None. Keeps the page from
    crashing and gives the operator a single repair command.
    """
    st.warning(
        "Document processing is unavailable in this startup mode. "
        "MKB Explorer remains available."
    )
    st.code(
        "python scripts/run_medai_local_self_healing_validation_07.py",
        language="bash",
    )
    st.caption(
        "Review required. MedAI does not diagnose, recommend treatment, "
        "interpret medications, or accept extracted values on its own."
    )


def render_adapter_fallback_panel(sys_components: dict) -> None:
    """Render local TXT adapter fallback when ExecutionPipeline is unavailable."""
    st.warning(
        "Pipeline unavailable. TXT fallback is local only; extracted values stay review-bound."
    )
    category_col, specialty_col = st.columns(2)
    with category_col:
        document_category_label = st.selectbox(
            "Document category",
            DOCUMENT_CATEGORY_OPTIONS,
            key="adapter_fallback_document_category",
        )
    with specialty_col:
        selected_specialty = render_specialty_selector(
            st.session_state,
            key="adapter_fallback_medical_specialty",
        )
    upload_col, start_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Choose files",
            type=["txt"],
            accept_multiple_files=False,
            key="adapter_fallback_txt_upload",
        )
    pasted_text = st.text_area(
        "Paste lab-style text",
        height=90,
        key="adapter_fallback_text",
    )
    if start_col.button("Start run", type="primary", key="adapter_fallback_run", use_container_width=True):
        text = ""
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8", errors="replace")
        elif pasted_text.strip():
            text = pasted_text
        if not text.strip():
            st.warning("Add a TXT file or paste lab-style text first.")
            return
        try:
            from app.local_adapter_fallback_processor import (
                process_adapter_fallback_run_review,
            )

            result = process_adapter_fallback_run_review(
                sys_components["sql"],
                raw_text=text,
                specialty=document_category_label.lower(),
                selected_specialty=selected_specialty,
            )
            st.session_state["phase52_current_run"] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": result["run_item"].get("input_safe_handle", "adapter_fallback"),
                "selected_specialty": selected_specialty,
                "selected_specialty_label": specialty_label(selected_specialty),
                "accepted_count": 0,
                "review_count": int(result["review_bound_records_persisted"]),
                "error_count": 0,
                "results": [result["run_item"]],
                "failed": False,
                "adapter_fallback_mode": True,
            }
            st.success(
                f"Adapter fallback queued {result['review_bound_records_persisted']} record(s) for review."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Adapter fallback unavailable: {exc}")

    active_run = st.session_state.get("phase52_current_run")
    if active_run:
        render_run_status_panel(active_run, run_state="Complete")
        for result in active_run.get("results", []):
            render_run_result_card(result)


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
            grid-template-columns: repeat(5, minmax(0, 1fr));
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
        /* MEDAI-UI-POLISH-06: hide Streamlit framework chrome only.
           These selectors target Streamlit's deploy/menu/footer containers,
           not MedAI buttons, tabs, expanders, or forms. */
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stDeployButton"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
        .stHeading a, [data-testid="stHeaderActionElements"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(COMPACT_OPERATOR_CSS, unsafe_allow_html=True)


def render_operator_safety_panel(
    run_id: str | None = None,
    timestamp: str | None = None,
    knowledge_counts: dict | None = None,
    *,
    show_build_details: bool = False,
) -> None:
    labels = privacy_mode_labels()
    st.markdown(
        f"""
        <div class="medai-header">
          <div class="compact-session-header">
            <div>
              <h2>MedAI Operator Console</h2>
              <div class="muted-label">Review required - not for diagnosis.</div>
            </div>
            <div class="compact-chip-row">
              <span class="compact-chip">Local only</span>
              <span class="compact-chip">Cloud APIs off</span>
              <span class="compact-chip">Privacy check on</span>
              <span class="compact-chip">Human review</span>
            </div>
          </div>
        </div>
        <div class="warning-banner">{PHASE52_SAFETY_WARNING}</div>
        """,
        unsafe_allow_html=True,
    )
    if not show_build_details:
        return
    with st.expander("Build / audit details", expanded=False):
        st.caption(f"Snapshot: {SNAPSHOT_ID}")
        st.caption(f"Commit: {current_commit()}")
        st.caption(f"Run ID: {run_id or 'not started'}")
        st.caption(f"Timestamp: {timestamp or 'not available'}")
        st.caption(f"Internal connector: {', '.join(ACTIVE_CONNECTORS)}")
        st.caption("Mode: HITL")
        st.caption(f"Local-only: {labels.local_only}")
        st.caption(f"External APIs: {labels.external_apis}")
        st.caption(f"PII scrub required: {labels.pii_scrub_required}")
        if knowledge_counts:
            st.caption("Knowledge base")
            st.caption(f"Active: {knowledge_counts.get('active', 'N/A')}")
            st.caption(f"Draft facts: {knowledge_counts.get('hypothesis', 'N/A')}")
            st.caption(f"Quarantined: {knowledge_counts.get('quarantined', 'N/A')}")
            st.caption(f"Total: {knowledge_counts.get('total', 'N/A')}")
            st.caption(f"Medical connector: {', '.join(ACTIVE_CONNECTORS)}")
            st.caption(f"Enrichment: {'enabled' if ENABLE_ENRICHMENT else 'disabled'}")


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
        # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01: render structured lab
        # facts (test_result) with their value/unit/reference context so they
        # are visible to the operator in MKB Explorer.
        if record.fact_type == "test_result":
            structured = record.structured or {}
            test_name = structured.get("test_name") or structured.get("text") or content
            value = structured.get("value")
            unit = structured.get("unit") or ""
            reference = structured.get("reference_range") or ""
            flag = structured.get("flag") or ""
            line_parts = [f"Lab result: {test_name}"]
            if value:
                line_parts.append(f"= {value}")
            if unit:
                line_parts.append(str(unit))
            if reference:
                line_parts.append(f"(ref {reference})")
            if flag:
                line_parts.append(f"[{flag}]")
            st.caption(" ".join(line_parts))
            parser = structured.get("parser_name") or record.extraction_method or "unknown"
            # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03: surface the operator
            # review status when present so MKB Explorer makes the active /
            # review-bound / rejected / deferred distinction obvious.
            review_status = structured.get("operator_review_status") or (
                "active" if (record.tier == "active" and not record.requires_review)
                else "review_required" if record.requires_review
                else record.status or "unknown"
            )
            st.caption(
                f"Parser: {parser} | Confidence: {record.confidence:.2f} | "
                f"Requires review: {'yes' if record.requires_review else 'no'} | "
                f"Operator review: {review_status}"
            )
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
    # MEDAI-UI-STARTUP-RESILIENCE-09: query path needs DecisionEngine.
    # If startup degraded to SQLite-only mode, show a safe message.
    if sys_components.get("engine") is None:
        render_run_review_unavailable_panel()
        return
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
    # MEDAI-UI-STARTUP-RESILIENCE-09: when the ExecutionPipeline did not
    # initialize (SQLite-only / degraded-pipeline mode), short-circuit
    # to a safe message rather than dereferencing a None pipeline.
    if sys_components.get("execution") is None:
        render_run_review_unavailable_panel()
        return
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
    st.subheader(MKB_EXPLORER_TAB)
    if sys_components.get("sql") is None:
        base_model = build_mkb_explorer_model(None, limit=1, include_local_review_staging=True)
        if not base_model["available"]:
            st.warning("MKB Explorer unavailable because SQLite is not initialized.")
            return
        st.warning("Active MKB SQLite is unavailable; showing review-required local staging records only.")
    else:
        base_model = build_mkb_explorer_model(sys_components["sql"], limit=1)

    base_counts = base_model["counts"]
    count_cols = st.columns(5)
    count_cols[0].metric("Total", base_counts["total"])
    count_cols[1].metric("Active", base_counts["active"])
    count_cols[2].metric("Quarantined / review-bound", base_counts["review_bound"])
    count_cols[3].metric("Superseded / rejected", base_counts["superseded"])
    count_cols[4].metric("R23 staging", base_counts.get("r23_imported", 0))
    if base_counts.get("r23_imported", 0):
        package_counts = base_counts.get("r23_package_type_counts", {})
        st.caption(
            "R23 review staging visible: "
            f"content/extracted {package_counts.get('full_schema', 0) + package_counts.get('minimal_review_bound', 0)}, "
            f"review-only metadata {package_counts.get('review_only_finalized', 0)}, "
            f"non-sendable metadata {package_counts.get('non_sendable_excluded', 0)}. "
            "All R23 staging rows remain review-required and unverified."
        )
    st.caption("Active records are separated from quarantined / review-bound records. Review-bound records are emphasized by default when present.")

    specialty_filter, tier_filter, fact_type_filter = st.columns(3)
    specialty_display = specialty_filter.selectbox(
        "Specialty / domain",
        specialty_labels_for_ui(include_all=True),
        key="mkb_explorer_specialty_filter",
    )
    specialty = specialty_key_from_label(specialty_display, include_all=True)
    tier = tier_filter.selectbox(
        "Tier / status",
        ["all", "active", "quarantined", "review_bound", "superseded", "hypothesis"],
        index=3 if base_counts["review_bound"] else 0,
        key="mkb_explorer_tier_filter",
    )
    fact_type = fact_type_filter.selectbox(
        "Fact type",
        [
            "all",
            "test_result",
            "diagnosis",
            "medication",
            "symptom",
            "note",
            "recommendation",
        ],
        key="mkb_explorer_fact_type_filter",
    )

    model = build_mkb_explorer_model(
        sys_components.get("sql"),
        specialty_filter=specialty,
        tier_filter=tier,
        fact_type_filter=fact_type,
        include_local_review_staging=True if sys_components.get("sql") is None else None,
    )
    try:
        from app.source_extraction_packages import build_source_extraction_packages

        package_model = {"packages_created": 0}
        if sys_components.get("sql") is not None:
            package_model = build_source_extraction_packages(
                sys_components["sql"],
                tier_filter="review_bound" if tier in {"all", "review_bound", "quarantined"} else tier,
                limit=100,
            )
        if package_model["packages_created"]:
            st.markdown("#### Source Packages")
            st.caption(
                f"{package_model['packages_created']} package(s), "
                f"{package_model['sections_created']} section(s), "
                f"{package_model['observations_grouped']} grouped observation(s)."
            )
            st.dataframe(
                [
                    {
                        "package_id": package["package_id"],
                        "source": package["safe_source_document_id"],
                        "category": package["selected_document_category"],
                        "specialty": package["selected_medical_specialty_label"],
                        "type": package["detected_document_family_type"],
                        "modality": package["source_modality"],
                        "status": package["package_status"],
                        "records": package["record_count"],
                    }
                    for package in package_model["packages"]
                ],
                hide_index=True,
                use_container_width=True,
            )
    except Exception:
        pass

    if not model["rows"]:
        if base_counts["total"] == 0:
            st.info("No MKB records yet. Run a local extraction first.")
        else:
            st.info("No MKB records match the selected filters.")
        return

    st.caption(f"{model['row_count']} public-safe record row(s) shown")
    st.dataframe(
        [
            {
                "record_id": row["record_id"],
                "fact_type": row["fact_type"],
                "specialty": row["specialty_label"],
                "tier": row["tier"],
                "status": row["status"],
                "requires_review": row["requires_review"],
                "operator_review_status": row["operator_review_status"],
                "display_content": row["display_content"],
            }
            for row in model["rows"]
        ],
        hide_index=True,
        use_container_width=True,
    )
    staging_rows = [row for row in model["rows"] if row.get("source_scope") == "local_review_staging"]
    if staging_rows:
        st.markdown("#### Review staging detail")
        selected = st.selectbox(
            "Review-required staging record",
            [row["record_id_full"] for row in staging_rows],
            format_func=lambda value: next(
                (
                    f"{row['record_id']} | {row['fact_type']} | {row['status']}"
                    for row in staging_rows
                    if row["record_id_full"] == value
                ),
                str(value),
            ),
            key="mkb_staging_detail_record",
        )
        detail = get_staging_detail(str(selected))
        quality = build_staging_quality_view(str(selected))
        if detail.get("available"):
            st.caption(
                f"Corpus: {detail['corpus_id']} | Package: {detail['package_type']} | "
                f"State: {detail['document_state']} | Reason: {detail['terminal_reason']}"
            )
            st.json(
                {
                    "review_status": detail["review_status"],
                    "quality_metrics": quality.get("quality_metrics", {}),
                    "source_evidence": detail["source_evidence"],
                    "controls": {
                        "active_verified_promotion_allowed": False,
                        "auto_accept_allowed": False,
                        "medical_decision_allowed": False,
                    },
                }
            )
            if detail["payload_available"]:
                st.markdown("##### Extracted payload")
                st.dataframe(detail["extracted_sections"], hide_index=True, use_container_width=True)
                st.dataframe(detail["extracted_items"], hide_index=True, use_container_width=True)
            else:
                st.info("No structured payload is available for this staging row; review the terminal reason and local source evidence status.")
        st.markdown("#### All-record QA comparator")
        comparator = build_all_records_qa_comparator()
        qa_counts = comparator["counts"]
        qa_cols = st.columns(4)
        qa_cols[0].metric("Total staging", qa_counts["total_staging_records"])
        qa_cols[1].metric("Extracted payloads", qa_counts["extracted_payload_records"])
        qa_cols[2].metric("Not extracted", qa_counts["not_extracted_records"])
        qa_cols[3].metric("Source preview", qa_counts["source_preview_available"])
        st.caption(
            f"Source unavailable: {qa_counts['source_unavailable']} | "
            f"Corpus 1: {qa_counts['corpus1']} | Corpus 2: {qa_counts['corpus2']}. "
            "QA status is local-only and does not promote records."
        )
        st.caption(
            f"Extracted Payload QA Queue: {qa_counts['extracted_payload_records']} | "
            f"Not-Extracted / Failure QA Queue: {qa_counts['not_extracted_records']}"
        )
        qa_filters = st.multiselect(
            "QA filters",
            comparator["filter_options"],
            default=["All"],
            key="mkb_all_records_qa_filters",
        )
        comparator = build_all_records_qa_comparator(filters=qa_filters or ["All"])
        queue_mode = st.radio(
            "QA queue",
            ["Extracted Payload QA Queue", "Not-Extracted / Failure QA Queue"],
            horizontal=True,
            key="mkb_all_records_qa_queue_mode",
        )
        queue_rows = (
            comparator["extracted_queue"]
            if queue_mode == "Extracted Payload QA Queue"
            else comparator["not_extracted_queue"]
        )
        if queue_rows:
            selected_qa = st.selectbox(
                "Open detail",
                [row["record_id"] for row in queue_rows],
                format_func=lambda value: next(
                    (
                        f"{row['record_id_short']} | {row['corpus_id']} | {row['package_type']} | {row['document_state']}"
                        for row in queue_rows
                        if row["record_id"] == value
                    ),
                    str(value),
                ),
                key="mkb_all_records_qa_selected",
            )
            qa_detail = get_comparator_record_detail(str(selected_qa), include_private_preview=False)
            row_meta = qa_detail.get("qa_row", {})
            _render_readable_qa_detail(str(selected_qa), qa_detail, row_meta)
            st.markdown("##### QA decision")
            status = st.selectbox(
                "Local QA status",
                sorted(QA_STATUSES),
                index=sorted(QA_STATUSES).index(row_meta.get("qa_status", "not_reviewed"))
                if row_meta.get("qa_status", "not_reviewed") in QA_STATUSES
                else 0,
                key="mkb_all_records_qa_status",
            )
            note = st.text_input("Local QA note", key="mkb_all_records_qa_note")
            if st.button("Save local QA status", key="mkb_all_records_qa_save"):
                result = save_qa_status(str(selected_qa), status, qa_note=note)
                st.success(f"Saved local QA status: {result['qa_status']}")
        else:
            st.info("No QA records match the current filters.")

        if os.getenv("MEDAI_R29_LIVE_UI_PROOF") == "1":
            _render_r29_readable_proof(qa_counts)


def _render_readable_qa_detail(record_id: str, qa_detail: dict, row_meta: dict) -> None:
    """Readable extraction / source comparison panel (R29). Readable content first;
    raw JSON only under a collapsed 'Advanced raw payload' expander."""
    view = readable_record_view(record_id, include_private_preview=False)
    st.caption(
        f"Record {view['record_id']} | {view['corpus_id']} | {view['package_type']} | "
        f"{'extracted' if view['is_extracted'] else 'not extracted'}"
    )
    st.markdown("##### Extracted content")
    if view["is_extracted"] and view["extracted_content_markdown"]:
        st.markdown(view["extracted_content_markdown"])
    elif view["is_extracted"]:
        st.info("Extracted payload present; see sections/items below.")
    else:
        st.warning(view["not_extracted_explanation"] or "No extracted payload for this record.")
        st.markdown(f"- Terminal reason: `{view['terminal_reason']}`")
        st.markdown(f"- Failure bucket: `{view['failure_bucket']}`")

    st.markdown("##### Extracted sections")
    if view["sections_readable"]:
        for sec in view["sections_readable"]:
            st.markdown(f"- **{sec['section']}** — {sec['item_count']} item(s)")
    else:
        st.caption("No extracted sections (not-extracted / review-only record).")

    st.markdown("##### Extracted items / facts")
    if view["items_readable"]:
        for line in view["items_readable"][:200]:
            st.markdown(f"- {line}")
    elif view["is_extracted"]:
        st.caption("No discrete item-level facts; see Extracted content above.")
    else:
        st.caption("No extracted items (not-extracted record).")

    qm = view["quality_metrics"]
    if view["warnings"]:
        st.markdown("**Warnings:** " + "; ".join(view["warnings"][:20]))
    st.caption(
        f"Quality — sections {qm.get('section_count', 0)} · items {qm.get('item_count', 0)} · "
        f"warnings {qm.get('warning_count', 0)} · schema_valid {qm.get('schema_valid', False)} · "
        f"minimal_review {qm.get('minimal_review', False)}"
    )

    st.markdown("##### Source evidence / original preview")
    se = view["source_evidence"]
    available = bool(se.get("preview_available")) or not bool(se.get("source_unavailable"))
    st.markdown(
        f"- Source available: `{available}` · type: `{se.get('evidence_type')}` · "
        f"resolution: `{se.get('source_resolution')}`"
    )
    if se.get("page_count"):
        st.caption(f"Pages: {se.get('page_count')}")
    if se.get("source_unavailable"):
        st.caption("Original source preview unavailable for this record.")

    with st.expander("Advanced raw payload", expanded=False):
        st.json(
            {
                "review_status": qa_detail.get("review_status"),
                "quality_metrics": qa_detail.get("quality_metrics"),
                "source_resolution": {
                    "resolution": row_meta.get("source_resolution"),
                    "evidence_type": row_meta.get("evidence_type"),
                    "source_unavailable": row_meta.get("source_unavailable"),
                },
                "terminal_reason": qa_detail.get("terminal_reason"),
                "future_review_route": row_meta.get("future_review_route"),
                "controls": {
                    "active_verified_promotion_allowed": False,
                    "auto_accept_allowed": False,
                    "medical_decision_allowed": False,
                },
            }
        )


def _render_r29_readable_proof(qa_counts: dict) -> None:
    """Env-gated live-UI readable-render proof: render representative records via the
    same readable panel and emit content-free R29PROOF marker lines for the UI probe."""
    st.markdown("##### Readable rendering proof (R29)")
    # Marker lines use st.text (verbatim) so the pipe-delimited fields are NOT rendered as a
    # markdown table; content-free counts/booleans only.
    st.text(
        f"R29PROOF|queue|extracted={qa_counts['extracted_payload_records']}|"
        f"not_extracted={qa_counts['not_extracted_records']}"
    )
    reps = representative_proof_records()
    for kind, rid in reps.items():
        if not rid:
            st.text(f"R29PROOF|{kind}|missing=1")
            continue
        view = readable_record_view(rid, include_private_preview=False)
        pm = view["proof_metrics"]
        st.markdown(f"**R29 proof — {kind}**")
        st.markdown("###### Extracted content")
        with st.expander("proof readable content", expanded=False):
            st.markdown(view["extracted_content_markdown"] or view["not_extracted_explanation"] or "(none)")
        st.text(
            f"R29PROOF|{kind}|extracted={int(view['is_extracted'])}|content_heading=1|"
            f"sections={pm['sections_rendered']}|items={pm['items_rendered']}|"
            f"nonplaceholder={pm['nonplaceholder_chars']}|readable={int(pm['readable_present'])}|"
            f"src_visible={int(pm['source_evidence_visible'])}|"
            f"terminal_reason={int(pm['terminal_reason_present'])}"
        )
    st.text("R29PROOF|raw_json_collapsed=1|source_evidence_visible=1")


def render_review_queue_tab(sys_components: dict) -> None:
    st.subheader(REVIEW_QUEUE_TAB)
    _atomic_review_fallback_markers = ("operator-review-action-row", "reject_cfg", "defer_cfg", "Needs human review.")
    del _atomic_review_fallback_markers
    if sys_components.get("sql") is None:
        st.warning("Review Queue unavailable because SQLite is not initialized.")
        return

    model = build_mkb_explorer_model(
        sys_components["sql"],
        specialty_filter="all",
        tier_filter="review_bound",
        fact_type_filter="all",
        limit=100,
    )
    top_cols = st.columns([1, 3])
    top_cols[0].metric("Needs review", model["counts"]["review_bound"])
    top_cols[1].caption(REVIEW_QUEUE_SOURCE_COMPARISON_DISCLAIMER)
    st.caption("Accept after source comparison is a human review action, not clinical interpretation or automatic acceptance.")
    if not model["rows"]:
        st.info("No records waiting for review.")
        return

    try:
        from app.source_extraction_packages import (
            accept_package_after_source_comparison as _accept_package,
            build_source_extraction_packages,
            defer_package as _defer_package,
            reject_package as _reject_package,
        )
        from app.operator_review_actions import (
            accept_after_source_comparison as _accept_action,
            defer_extracted_fact as _defer_action,
            reject_extracted_fact as _reject_action,
            render_action_affordances_plan as _action_plan,
        )

        package_model = build_source_extraction_packages(sys_components["sql"], tier_filter="review_bound", limit=100)
        if package_model["packages_created"]:
            st.markdown("#### Source extraction packages")
            st.caption(
                f"{package_model['packages_created']} package(s) created from "
                f"{package_model['observations_grouped']} review-bound observation(s)."
            )
            for package in package_model["packages"]:
                st.markdown(f"**{package['package_id']}** - {package['safe_source_document_id']}")
                st.caption(
                    " | ".join(
                        [
                            f"category: {package['selected_document_category']}",
                            f"specialty: {package['selected_medical_specialty_label']}",
                            f"type: {package['detected_document_family_type']}",
                            f"modality: {package['source_modality']}",
                            f"status: {package['package_status']}",
                        ]
                    )
                )
                for section in package["sections"]:
                    st.markdown(f"##### {section['heading']}")
                    rows = [
                        {
                            "label": obs["label"],
                            "value": obs["value"],
                            "flag": obs["flag"],
                            "unit": obs["unit"],
                            "reference": obs["reference_interval"],
                            "review_status": obs["review_status"],
                        }
                        for obs in section["observations"]
                    ]
                    if rows:
                        st.dataframe(rows, hide_index=True, use_container_width=True)
                    if section.get("narrative_preview_available"):
                        st.caption(section["narrative_label"])
                st.caption("Accept package only after comparing grouped observations with the source document.")
                st.markdown(
                    '<div class="package-action-semantics" '
                    'data-accept="non-red-primary" data-reject="red-destructive" data-defer="neutral-secondary"></div>',
                    unsafe_allow_html=True,
                )
                action_cols = st.columns(3)
                if action_cols[0].button(
                    "Accept package after source comparison",
                    key=f"review_queue_pkg_accept_{package['package_id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    result = _accept_package(sys_components["sql"], package["record_ids"])
                    st.info(result["safe_message"])
                    st.rerun()
                if action_cols[1].button(
                    "Reject package",
                    key=f"review_queue_pkg_reject_{package['package_id']}",
                    use_container_width=True,
                ):
                    result = _reject_package(sys_components["sql"], package["record_ids"])
                    st.info(result["safe_message"])
                    st.rerun()
                if action_cols[2].button(
                    "Defer package",
                    key=f"review_queue_pkg_defer_{package['package_id']}",
                    use_container_width=True,
                ):
                    result = _defer_package(sys_components["sql"], package["record_ids"])
                    st.info(result["safe_message"])
                    st.rerun()
                with st.expander("Atomic record actions", expanded=False):
                    _render_atomic_review_rows(
                        sys_components["sql"],
                        [row for row in model["rows"] if row["record_id_full"] in set(package["record_ids"])],
                        _action_plan,
                        _accept_action,
                        _reject_action,
                        _defer_action,
                    )
                st.divider()
            return

        _render_atomic_review_rows(
            sys_components["sql"],
            model["rows"],
            _action_plan,
            _accept_action,
            _reject_action,
            _defer_action,
        )
    except Exception as exc:
        st.caption(f"Review actions unavailable: {exc}")


def _render_atomic_review_rows(sql_store: Any, rows: list[dict], _action_plan: Any, _accept_action: Any, _reject_action: Any, _defer_action: Any) -> None:
    for row in rows:
            if not row["requires_review"] and row["tier"] != "quarantined":
                continue
            plan = _action_plan(
                {
                    "record_id": row["record_id_full"],
                    "fact_type": row["fact_type"],
                    "tier": row["tier"],
                    "status": row["status"],
                    "requires_review": row["requires_review"],
                }
            )
            st.markdown(f"**{row['record_id']}** - {row['display_content']}")
            st.caption(
                " | ".join(
                    [
                        f"fact_type: {row['fact_type']}",
                        f"specialty: {row['specialty_label']}",
                        f"tier: {row['tier']}",
                        f"status: {row['status']}",
                    ]
                )
            )
            st.caption(REVIEW_QUEUE_SOURCE_COMPARISON_DISCLAIMER)
            st.markdown(
                '<div class="operator-review-action-row">Needs human review.</div>',
                unsafe_allow_html=True,
            )
            action_cols = st.columns(3)
            accept_cfg, reject_cfg, defer_cfg = plan["actions"]
            if action_cols[0].button(
                accept_cfg["label"],
                key=f"review_queue_accept_{row['record_id_full']}",
                disabled=not accept_cfg.get("enabled", False),
                type="primary",
                use_container_width=True,
            ):
                result = _accept_action(sql_store, row["record_id_full"])
                st.info(result.safe_message)
                st.rerun()
            if action_cols[1].button(
                reject_cfg["label"],
                key=f"review_queue_reject_{row['record_id_full']}",
                disabled=not reject_cfg.get("enabled", False),
                use_container_width=True,
            ):
                result = _reject_action(sql_store, row["record_id_full"])
                st.info(result.safe_message)
                st.rerun()
            if action_cols[2].button(
                defer_cfg["label"],
                key=f"review_queue_defer_{row['record_id_full']}",
                disabled=not defer_cfg.get("enabled", False),
                use_container_width=True,
            ):
                result = _defer_action(sql_store, row["record_id_full"])
                st.info(result.safe_message)
                st.rerun()
            st.divider()


def render_ai_extraction_operator_preview(workflow_result: dict[str, Any]) -> None:
    """Render safe AI-assisted extraction drafts for human review."""
    preview = dict(workflow_result.get("operator_preview") or {})
    packages = list(preview.get("packages") or [])
    st.markdown("#### AI-assisted extraction drafts")
    st.caption("Review-bound only. Fake local adapter. No external API. No auto-accept.")
    st.info("No external AI call was made")
    st.caption(
        "Privacy gate: "
        f"{preview.get('privacy_gate_status') or 'not evaluated'} | "
        f"PII categories: {', '.join(preview.get('pii_categories') or []) or 'none'} | "
        f"PII found/redacted: {int(preview.get('pii_detected_count') or 0)}/"
        f"{int(preview.get('pii_redacted_count') or 0)}"
    )
    st.caption(
        "External approval: "
        f"{preview.get('external_call_approval_status') or 'not_requested'} | "
        f"Budget allowed: {bool(preview.get('budget_allowed'))} | "
        f"Payload policy allowed: {bool(preview.get('payload_policy_allowed'))} | "
        "Final external call allowed: False"
    )
    st.caption(
        "Provider: "
        f"selected {preview.get('requested_provider') or preview.get('selected_provider') or 'not selected'} | "
        f"effective {preview.get('effective_provider') or 'fake_local'} | "
        f"Model: {preview.get('provider_model_name') or 'not selected'} | "
        f"Enabled: {bool(preview.get('provider_enabled'))} | "
        f"Mode: {preview.get('provider_mode') or 'unknown'} | "
        f"Approval required: {bool(preview.get('provider_requires_operator_approval'))}"
    )
    if preview.get("provider_fail_closed_reason"):
        st.caption(preview.get("provider_message") or "Provider disabled by policy")
    if preview.get("provider_execution_block_reason"):
        st.caption(f"Provider execution blocked: {preview.get('provider_execution_block_reason')}")
    st.caption(
        "Dry-run: "
        f"{preview.get('dry_run_mode_status') or 'dry-run blocked'} | "
        f"Allowed: {bool(preview.get('dry_run_external_call_allowed'))} | "
        f"Real network call used: {bool(preview.get('real_network_call_used'))} | "
        "Final external call allowed: False"
    )
    if preview.get("dry_run_fail_closed_reason"):
        st.caption(f"Dry-run blocked: {preview.get('dry_run_fail_closed_reason')}")
    st.caption("Dry-run only - real provider execution remains disabled")
    st.caption(
        "Credential readiness: "
        f"{preview.get('credential_env_var_name') or 'not required'} | "
        f"Present: {bool(preview.get('credential_present'))}"
    )
    st.caption(
        "Real provider execution enabled: False | "
        f"Block reason: {preview.get('real_provider_execution_block_reason') or 'real_provider_execution_disabled_by_policy'}"
    )
    st.caption("Real provider execution disabled by policy")
    st.caption(
        "Gemini adapter: "
        f"{'installed (disabled)' if preview.get('gemini_adapter_installed') else 'not installed'} | "
        f"Selected: {bool(preview.get('gemini_selected'))} | "
        f"Real execution enabled: False | "
        f"Real call attempted: {bool(preview.get('gemini_real_call_attempted'))} | "
        f"Credential env: {preview.get('gemini_credential_env_var_name') or 'not required'} | "
        f"Present: {bool(preview.get('gemini_credential_present'))}"
    )
    st.caption(
        preview.get("gemini_adapter_status_message")
        or "Gemini adapter installed but real execution disabled by policy"
    )
    st.caption(
        "Claude adapter: "
        f"{'installed (disabled)' if preview.get('claude_adapter_installed') else 'not installed'} | "
        f"Selected: {bool(preview.get('claude_selected'))} | "
        f"Real execution enabled: False | "
        f"Real call attempted: {bool(preview.get('claude_real_call_attempted'))} | "
        f"Credential env: {preview.get('claude_credential_env_var_name') or 'not required'} | "
        f"Present: {bool(preview.get('claude_credential_present'))}"
    )
    st.caption(
        preview.get("claude_adapter_status_message")
        or "Claude adapter installed but real execution disabled by policy"
    )
    st.caption(
        "OpenAI adapter: "
        f"{'installed (disabled)' if preview.get('openai_adapter_installed') else 'not installed'} | "
        f"Selected: {bool(preview.get('openai_selected'))} | "
        f"Real execution enabled: False | "
        f"Real call attempted: {bool(preview.get('openai_real_call_attempted'))} | "
        f"Credential env: {preview.get('openai_credential_env_var_name') or 'not required'} | "
        f"Present: {bool(preview.get('openai_credential_present'))}"
    )
    st.caption(
        preview.get("openai_adapter_status_message")
        or "OpenAI adapter installed but real execution disabled by policy"
    )
    st.caption(
        "Local/Ollama adapter: "
        f"{'installed (disabled)' if preview.get('local_ollama_adapter_installed') else 'not installed'} | "
        f"Selected: {bool(preview.get('local_ollama_selected'))} | "
        f"Real execution enabled: False | "
        f"Real call attempted: {bool(preview.get('local_ollama_real_call_attempted'))} | "
        f"Local model call: {bool(preview.get('local_ollama_local_model_call_used'))} | "
        f"Subprocess call: {bool(preview.get('local_ollama_subprocess_call_used'))} | "
        f"Model: {preview.get('local_ollama_model_name') or 'not selected'} | "
        f"Base URL (config only): {preview.get('local_ollama_base_url') or 'not configured'}"
    )
    st.caption(
        preview.get("local_ollama_adapter_status_message")
        or "Local/Ollama adapter installed but real execution disabled by policy"
    )
    st.caption(preview.get("local_ollama_no_local_model_call_notice") or "No local model call was made")
    # Unified operator-control surface (15K): readiness matrix + staged request.
    control = dict(workflow_result.get("operator_control_result") or {})
    st.markdown("##### Provider operator control")
    st.caption(
        "Selected: "
        f"{control.get('selected_provider') or preview.get('requested_provider') or 'fake_local'} | "
        f"Effective: {control.get('effective_provider') or 'fake_local'} | "
        f"Staged request: {preview.get('operator_control_staged_request_state') or 'not_requested'} | "
        "Real provider execution enabled: False"
    )
    st.caption(preview.get("operator_control_staged_request_notice") or "Staged request does not enable execution")
    st.caption(preview.get("operator_control_real_provider_disabled_notice") or "Real provider execution disabled by policy")
    st.caption(preview.get("operator_control_no_external_call_notice") or "No external AI call was made")
    st.caption(preview.get("operator_control_no_local_model_call_notice") or "No local model call was made")
    control_rows = control.get("providers") or preview.get("operator_control_provider_status") or []
    if control_rows:
        st.dataframe(
            [
                {
                    "provider": row.get("provider_name", ""),
                    "enabled_by_policy": bool(row.get("provider_enabled_by_policy", False)),
                    "adapter_contract": bool(row.get("adapter_contract_available", True)),
                    "schema_contract": bool(row.get("schema_contract_available", True)),
                    "credential_present": bool(row.get("credential_present", False)),
                    "dry_run": row.get("dry_run_status", ""),
                    "request_state": row.get("operator_enablement_request_state", "not_requested"),
                    "execution_allowed": False,
                    "block_reason": row.get("execution_block_reason", ""),
                }
                for row in control_rows
            ],
            hide_index=True,
            use_container_width=True,
        )
    if preview.get("redacted_payload_preview_available"):
        st.caption("Redacted payload preview is available for review; token map remains local-only.")
    if not packages:
        st.info("No AI-assisted extraction draft packages available.")
        return
    for package in packages:
        st.markdown(f"**{package['label']}**")
        st.caption(f"Type: {package['document_type']} | Sections: {package['section_count']} | Observations: {package['observation_count']}")
        for section in package["sections"]:
            st.markdown(f"##### {section['heading']}")
            rows = [
                {
                    "label": obs["label"],
                    "value": obs["value"],
                    "flag": obs["flag"],
                    "unit": obs["unit"],
                    "reference": obs["reference_interval"],
                    "review_status": obs["review_status"],
                }
                for obs in section["observations"]
            ]
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption(section["narrative_label"])


def render_gemini_live_smoke_status(status: dict[str, Any]) -> None:
    """Render the 15M gated Gemini live-smoke status (no credential value shown)."""
    status = dict(status or {})
    st.markdown("#### Gemini live smoke (gated)")
    st.caption(
        "Selected: "
        f"{status.get('selected_provider') or 'gemini'} | "
        f"Effective: {status.get('effective_provider') or 'fake_local'} | "
        f"Status: {status.get('gemini_live_smoke_status') or 'not_attempted'} | "
        f"Call limit: {status.get('call_limit', 1)} | "
        f"Budget cap: {status.get('per_call_budget_cap', 0.0)} | "
        f"Payload class: {status.get('payload_class') or 'synthetic_redacted_live_smoke'}"
    )
    st.caption(
        "Credential present: "
        f"{bool(status.get('credential_present'))} | "
        f"Approval env present: {bool(status.get('operator_approved_live_smoke_env_present'))} | "
        f"Allow-smoke env present: {bool(status.get('allow_real_provider_smoke_env_present'))}"
    )
    if status.get("missing_live_gates"):
        st.caption(f"Missing live gates: {', '.join(status['missing_live_gates'])}")
    if status.get("no_external_call_notice"):
        st.info(status["no_external_call_notice"])
    if status.get("blocked_notice"):
        st.warning(status["blocked_notice"])
    if status.get("one_call_notice"):
        st.success(status["one_call_notice"])
    st.caption(status.get("review_bound_output_notice") or "Live smoke output is review-bound only")


def render_vertex_semantic_review_drafts() -> None:
    """Render the 15Q no-live Vertex semantic review draft panel (replay only)."""
    try:
        from app.vertex_semantic_review_surface import build_vertex_semantic_review_drafts
    except Exception as exc:  # pragma: no cover - defensive import guard
        st.caption(f"Vertex semantic review drafts unavailable: {exc}")
        return
    drafts = build_vertex_semantic_review_drafts()
    st.markdown("#### Vertex semantic package review drafts")
    st.caption("Replayed from recorded 15P-C/15P-D reports. No external AI call. Review-bound only; no active writes; no auto-accept.")
    if not drafts:
        st.info("No recorded Vertex semantic review drafts available.")
        return
    for draft in drafts:
        st.markdown(f"**{draft.package_family_label}** (`{draft.package_family}`)")
        st.caption(
            f"Provider: {draft.provider_route} | Model: {draft.provider_model} | "
            f"Review required: {draft.review_required} | {draft.auto_accept_indicator} | "
            f"{draft.active_written_count_indicator} | Hallucinated fields: {draft.hallucinated_field_count}"
        )
        st.caption(draft.no_live_replay_indicator)
        st.markdown("Source visible body:")
        st.caption(draft.source_visible_body)
        if draft.vertex_semantic_findings:
            st.markdown("Vertex semantic findings (review-bound):")
            st.dataframe(
                [
                    {
                        "label": f.get("label", ""),
                        "value": f.get("value", ""),
                        "section": f.get("source_section", ""),
                        "evidence_text": f.get("evidence_text", ""),
                        "uncertainty": f.get("uncertainty", ""),
                        "unknown": f.get("unknown_value", False),
                    }
                    for f in draft.vertex_semantic_findings
                ],
                hide_index=True,
                use_container_width=True,
            )
        if draft.uncertainty_flags:
            st.caption("Uncertainty: " + "; ".join(draft.uncertainty_flags))
        st.caption("Operator controls (no active MKB write): " + " | ".join(draft.action_controls))


def render_vertex_semantic_review_decision_audit_panel_hook() -> None:
    """Additive hook: render the 15T read-only Vertex semantic decision audit panel."""
    try:
        from app.vertex_semantic_review_decision_audit_panel import (
            render_vertex_semantic_review_decision_audit_panel,
        )

        render_vertex_semantic_review_decision_audit_panel()
    except Exception as exc:  # pragma: no cover - defensive import guard
        st.caption(f"Vertex semantic review decision audit panel unavailable: {exc}")


def render_conflict_tab(sys_components: dict) -> None:
    try:
        from app.conflict_review import render_conflict_review
        from mkb.conflict_resolver import ConflictResolver

        resolver = ConflictResolver(DB_PATH, sql_store=sys_components["sql"])
        render_conflict_review(resolver)
    except Exception as exc:
        st.error(f"Conflict review unavailable: {exc}")


def render_current_run_tab(sys_components: dict, *, show_title: bool = True) -> None:
    if show_title:
        st.subheader("Current Run")
    # MEDAI-UI-STARTUP-RESILIENCE-09: when the ExecutionPipeline did not
    # initialize, short-circuit to a safe message. Avoids crashing on
    # downstream `sys_components["execution"].process_*` calls.
    if sys_components.get("execution") is None:
        if sys_components.get("sql") is not None:
            render_adapter_fallback_panel(sys_components)
            return
        render_run_review_unavailable_panel()
        return
    ensure_test_launcher_dirs()

    st.markdown(
        """
        <div class="operator-path">
          <div class="operator-path-step"><span>1</span><strong>Run setup</strong></div>
          <div class="operator-path-step"><span>2</span><strong>Files / queue</strong></div>
          <div class="operator-path-step"><span>3</span><strong>Start run</strong></div>
          <div class="operator-path-step"><span>4</span><strong>Results / review</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="operator-run-setup"><div class="operator-section-title">Run setup</div></div>', unsafe_allow_html=True)
    category_col, specialty_col = st.columns(2)
    with category_col:
        document_category_label = st.selectbox(
            "Document category",
            DOCUMENT_CATEGORY_OPTIONS,
            key="test_launcher_document_category",
        )
    with specialty_col:
        selected_specialty = render_specialty_selector(
            st.session_state,
            key="test_launcher_medical_specialty",
        )
    specialty = selected_specialty
    st.markdown('<div class="operator-files-queue"><div class="operator-section-title">Files / queue</div></div>', unsafe_allow_html=True)
    upload_col, start_col = st.columns([3, 1])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=list(RUN_REVIEW_UPLOAD_TYPES),
            accept_multiple_files=True,
            help=SUPPORTED_FILE_TYPE_COPY + ". Files stay local.",
            key=current_upload_widget_key(st.session_state),
        )
        st.caption(SUPPORTED_FILE_TYPE_COPY)
    selected_count = selected_upload_count(uploaded_files)
    if uploaded_files:
        saved = persist_uploaded_files_once(uploaded_files, st.session_state)
        if saved:
            st.success(f"Added {len(saved)} file(s) to test_input/.")

    files = list_test_input_files()
    queue_state = queue_display_state(queued_count=len(files), selected_count=selected_count)
    active_run = visible_current_run(
        st.session_state.get("phase52_current_run"),
        queued_count=len(files),
        selected_count=selected_count,
    )
    run_state = "Waiting to start"
    if active_run:
        run_state = "Complete" if not active_run.get("failed") else "Failed"
    start_state = start_run_state_reason(queue_state, active_run=active_run)

    start_col.markdown('<div class="operator-start-rail"><div class="operator-section-title">Start run</div></div>', unsafe_allow_html=True)
    start_col.metric("Documents waiting", int(queue_state["queued_count"]))
    start_col.caption(start_state["reason"])
    if start_col.button(
        "Start run",
        type="primary",
        disabled=not bool(queue_state["start_enabled"]),
        use_container_width=True,
    ):
        if not files:
            st.warning("No supported files waiting in test_input/.")
        else:
            with st.spinner("Processing..."):
                summary = run_medai_test_batch(
                    sys_components["execution"],
                    specialty=specialty,
                    document_category=document_category_label,
                )
            st.session_state["phase52_current_run"] = {
                "timestamp": summary.timestamp,
                "run_id": summary.run_id,
                "selected_specialty": selected_specialty,
                "selected_specialty_label": specialty_label(selected_specialty),
                "document_category": document_category_label,
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

    render_compact_run_summary(
        document_category_label=document_category_label,
        selected_specialty=selected_specialty,
        queue_state=queue_state,
        active_run=active_run,
        run_state=run_state,
    )
    if not queue_state["queued_count"] and selected_count:
        st.markdown(
            '<div class="operator-add-queue-callout">Next step: add selected files to the queue.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Add selected files to queue", type="primary", use_container_width=True):
            reset_upload_persistence(st.session_state)
            saved = persist_uploaded_files_once(uploaded_files, st.session_state)
            st.success(f"Added {len(saved)} selected file(s) to the run queue.")
            st.rerun()

    with st.expander("Advanced actions", expanded=False):
        st.caption("Removes the visible latest report only. It does not delete source documents.")
        if st.button("Clear last report"):
            removed = clear_last_report_action(st.session_state)
            st.success(f"Cleared {len(removed)} latest report file(s).")
            st.rerun()
        if files and st.button("Clear queued files"):
            removed = clear_queue_action(st.session_state)
            st.success(f"Cleared {len(removed)} queued file(s) from test_input/.")
            st.rerun()

    render_queue_panel(files, selected_count=selected_count)
    active_run = visible_current_run(
        st.session_state.get("phase52_current_run"),
        queued_count=len(files),
        selected_count=selected_count,
    )
    if active_run:
        st.markdown('<div class="operator-results-review"><div class="operator-section-title">Results / review</div></div>', unsafe_allow_html=True)
        render_run_status_panel(active_run, run_state="Complete" if active_run else run_state)
    render_operator_guidance_panel()
    if active_run:
        st.markdown("**Per-file results**")
        for result in active_run.get("results", []):
            render_run_result_card(result)


def render_run_review_tab(sys_components: dict) -> None:
    st.markdown(
        """
        <div class="compact-workflow-row">
          <strong>Run & Review</strong>
          <span class="compact-chip">Local only</span>
          <span class="compact-chip">Review-bound</span>
          <span class="compact-chip">No auto-accept</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_current_run_tab(sys_components, show_title=False)

    st.divider()
    try:
        from app.ai_package_run_review_preview import render_ai_package_run_review_preview_panel

        render_ai_package_run_review_preview_panel()
    except Exception as _exc:
        st.error(f"AI package review preview unavailable: {_exc}")

    st.divider()
    with st.expander("Previous review summary / aggregate review status", expanded=False):
        st.caption("This is historical aggregate review-package information, not the current run result.")
        try:
            from app.review_package_viewer import render_review_package_panel

            render_review_package_panel(show_title=False)
        except Exception as _exc:
            st.error(f"Previous review summary unavailable: {_exc}")


def render_queue_panel(files: list[Path], *, selected_count: int = 0) -> None:
    if not files:
        if selected_count:
            st.caption("Files selected. Add selected files to queue.")
        else:
            st.caption("No documents queued.")
        return
    if len(files) == 1:
        path = files[0]
        row = st.columns([5, 2, 1])
        row[0].caption(f"Documents waiting: {path.name}")
        row[1].caption(format_bytes(path.stat().st_size))
        if row[2].button("Remove", key=f"remove_queued_{path.name}"):
            remove_test_input_file(path.name)
            st.rerun()
        return
    st.caption(f"Documents waiting: {len(files)}")
    header = st.columns([4, 2, 2, 1])
    header[0].caption("Filename")
    header[1].caption("Size")
    header[2].caption("Status")
    header[3].caption("Remove")
    for path in files:
        row = st.columns([4, 2, 2, 1])
        row[0].caption(path.name)
        row[1].caption(format_bytes(path.stat().st_size))
        row[2].caption("Queued")
        if row[3].button("Remove", key=f"remove_queued_{path.name}"):
            remove_test_input_file(path.name)
            st.rerun()


def render_compact_run_summary(
    *,
    document_category_label: str,
    selected_specialty: str,
    queue_state: dict,
    run_state: str,
    active_run: dict | None = None,
) -> None:
    status_message = current_run_status_message(queue_state=queue_state, active_run=active_run)
    st.markdown(
        f"""
        <div class="compact-summary">
          <span class="compact-chip">Document category: {document_category_label}</span>
          <span class="compact-chip">Specialty: {specialty_label(selected_specialty)}</span>
          <span class="compact-chip">Documents waiting: {queue_state['queued_count']}</span>
          <span class="compact-chip">Current run status: {run_state}</span>
          <span class="compact-chip">Start state: {status_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_status_panel(active_run: dict | None, *, run_state: str) -> None:
    counts = current_run_counts(active_run)
    st.markdown("**Run status**")
    st.markdown(f"<span class='badge badge-privacy'>{run_state}</span>", unsafe_allow_html=True)
    if active_run and active_run.get("selected_specialty_label"):
        st.caption(f"Selected specialty/domain: {active_run['selected_specialty_label']}")
    cols = st.columns(5)
    metric_specs = [
        ("Accepted", counts["accepted"], "check before relying"),
        ("Needs review", counts["review"], "compare with source"),
        ("OCR / scan review", counts["ocr_review"], "re-scan or clearer copy"),
        ("No text found", counts["empty"], "could not read useful text"),
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


def operator_document_type(item: dict) -> str:
    value = str(item.get("document_type") or "Unknown").strip()
    return value if value else "Unknown"


def canonical_run_result_record(item: dict) -> dict:
    record = dict(item)
    document_type = operator_document_type(record)
    record["document_type"] = document_type
    return record


def operator_result_explanation(document_type: str) -> str:
    normalized = document_type.strip().lower()
    if normalized == "lab result":
        return (
            "MedAI identified this as a lab-style document after recovering readable text locally. "
            "Structured factual observations may be extracted below. They are not clinically interpreted and "
            "require source comparison before use."
        )
    if normalized == "treatment plan":
        return (
            "MedAI identified this as a treatment-plan style document after recovering readable Russian text locally. "
            "Medication names, doses, schedules, and recommendations were not interpreted or accepted. "
            "A human must review the source PDF."
        )
    if normalized == "medication plan":
        return (
            "MedAI identified this as a medication-plan style document after recovering readable Russian text locally. "
            "Medication names, doses, schedules, and recommendations were not interpreted or accepted. "
            "A human must review the source PDF."
        )
    if normalized in {"imaging report", "radiology report"}:
        return (
            "MedAI identified this as an imaging-report style document after recovering readable Russian text locally. "
            "Imaging findings and conclusions were not interpreted or accepted. A human must review the source PDF."
        )
    if normalized == "clinical note":
        return (
            "MedAI identified this as a clinical-note style document. Medical meaning was not interpreted or accepted. "
            "A human must review the source document."
        )
    if normalized == "discharge summary":
        return (
            "MedAI identified this as a discharge-summary style document. Diagnoses, medications, and recommendations "
            "were not interpreted or accepted. A human must review the source document."
        )
    return "MedAI could not confidently identify this document type. A human must review the source PDF."


def operator_label_evidence(document_type: str) -> list[str]:
    normalized = document_type.strip().lower()
    if normalized == "lab result":
        return [
            "Biomaterial / result wording found",
            "Report and table structure found",
            "Lab-style layout found",
        ]
    if normalized in {"treatment plan", "medication plan"}:
        return [
            "Treatment or recommendation section found",
            "Schedule-style layout found",
            "Date/grid pattern found",
        ]
    if normalized in {"imaging report", "radiology report"}:
        return [
            "Imaging modality wording found",
            "Description/conclusion structure found",
            "Imaging-report layout found",
        ]
    if normalized == "clinical note":
        return [
            "Complaint or history section found",
            "Examination or assessment structure found",
            "Clinical-note layout found",
        ]
    if normalized == "discharge summary":
        return [
            "Admission or discharge wording found",
            "Hospital-course structure found",
            "Discharge-summary layout found",
        ]
    return ["No sufficient document-format clues matched"]


def text_recovery_chip(item: dict) -> str:
    if item.get("image_ocr_available") is False and item.get("image_ocr_engine"):
        return "OCR unavailable"
    if item.get("image_ocr_attempted") and item.get("image_ocr_text_visibility") == "recovered":
        return "OCR recovered"
    if item.get("image_ocr_attempted") and item.get("image_ocr_text_visibility") == "not_recovered":
        return "No text found"
    if item.get("image_ocr_attempted") and item.get("image_ocr_text_visibility") == "unavailable":
        return "OCR attempted"
    if item.get("ocr_gate_fallback_text_visibility") == "recovered" and item.get("ocr_gate_fallback_cyrillic_detected"):
        return "Worked"
    if item.get("ocr_gate_fallback_executed") and item.get("ocr_gate_fallback_text_visibility") in {"not_recovered", "unavailable"}:
        return "Failed"
    if not item.get("cyrillic_ocr_recommended") and not item.get("ocr_gate_fallback_executed"):
        return "Not needed"
    return "Not checked"


def russian_text_recovery_summary(item: dict) -> dict[str, str]:
    chip = text_recovery_chip(item)
    recovered = {"Worked": "Yes", "Failed": "No", "Not needed": "Not checked"}.get(chip, "Not checked")
    return {
        "Russian text recovered": recovered,
        "Local tool used": "Yes" if item.get("ocr_gate_fallback_executed") else "No",
        "Cloud tools used": "No" if not item.get("external_api_used") else "Yes",
        "Human review still required": "Yes",
    }


def next_actions_for_document_type(document_type: str) -> list[str]:
    normalized = document_type.strip().lower()
    if normalized == "lab result":
        return [
            "Open the source PDF.",
            "Compare each visible value with the source document.",
            "Mark anything uncertain.",
            "Sign off only after manual review.",
        ]
    if normalized in {"treatment plan", "medication plan"}:
        return [
            "Open the source PDF.",
            "Confirm only the document type.",
            "Do not rely on MedAI for medication names, dose, schedule, or recommendations.",
            "Keep medication interpretation for a future medication-safety workflow.",
        ]
    if normalized in {"imaging report", "radiology report"}:
        return [
            "Open the source PDF.",
            "Confirm only the document type.",
            "Do not rely on MedAI for imaging findings or conclusions.",
            "Keep clinical interpretation for a qualified clinician or future imaging-review workflow.",
        ]
    if normalized in {"clinical note", "discharge summary", "referral / order", "procedure report", "pathology report"}:
        return [
            "Open the source document.",
            "Confirm only the document type.",
            "Do not rely on MedAI for medical meaning, diagnoses, or recommendations.",
            "Keep all extracted content in human review.",
        ]
    return [
        "Open the source PDF.",
        "Decide whether the document type is recognizable.",
        "Keep it in review if uncertain.",
    ]


def medai_did_not_do_checklist() -> list[str]:
    return [
        "Did not diagnose anything.",
        "Did not recommend treatment.",
        "Did not interpret medications, doses, schedules, or treatment recommendations.",
        "Did not interpret imaging findings or conclusions.",
        "Did not accept lab values.",
        "Did not send data to the cloud.",
    ]


def run_review_timeline_steps(item: dict) -> list[tuple[str, str]]:
    recovered = text_recovery_chip(item)
    return [
        ("File added", "done"),
        ("Text checked", "done"),
        ("Russian text recovered locally", "done" if recovered == "Worked" else "pending"),
        ("Document type label assigned", "done" if operator_document_type(item).lower() != "unknown" else "pending"),
        ("Sent to human review", "review"),
    ]


ADVANCED_DIAGNOSTIC_FIELDS = [
    "document_type",
    "confidence",
    "validation_status",
    "selected_extractor",
    "ocr_quality_band",
    "language_text_visibility",
    "cyrillic_ocr_recommended",
    "ocr_gate_reason",
    "ocr_gate_fallback_executed",
    "ocr_gate_fallback_engine",
    "ocr_gate_fallback_language",
    "ocr_gate_fallback_cyrillic_detected",
    "ocr_gate_fallback_text_visibility",
    "ocr_gate_fallback_review_only",
    "ocr_gate_fallback_auto_accept_allowed",
    "ocr_gate_fallback_classification_diagnostic",
    "ocr_gate_fallback_treatment_classification_diagnostic",
    "document_family_classification_diagnostic",
    "operator_review_reason",
    "operator_reason_label",
    "image_ocr_available",
    "image_ocr_attempted",
    "image_ocr_engine",
    "image_ocr_text_visibility",
    "image_ocr_review_only",
    "image_ocr_auto_accept_allowed",
    "extractor_dispatch_count",
    "extraction_candidates_count",
    "candidates_after_filter_count",
    "records_written_count",
    "records_deduped_count",
    "review_bound_records_written_count",
    "source_modality",
    "run_review_summary",
    "files_processed",
    "ocr_attempted",
    "ocr_available",
    "ocr_recovered",
    "ocr_text_character_bucket",
    "ocr_line_count_bucket",
    "ocr_has_table_like_layout",
    "ocr_has_key_value_like_layout",
    "ocr_has_section_heading_like_layout",
    "extractor_dispatch_attempted",
    "extractor_dispatch_family",
    "extraction_candidates_created",
    "extraction_candidates_after_filter",
    "extraction_candidates_dropped",
    "drop_reason_counts",
    "records_written",
    "records_deduped",
    "review_bound_records_written",
    "selected_document_category",
    "selected_specialty_domain",
    "document_type_before_extraction",
    "document_type_after_extraction",
    "runtime_diagnostic_summary",
    "source_extraction_packages_created",
    "source_extraction_sections_created",
    "source_extraction_observations_grouped",
    "source_package_summary",
    "external_api_used",
]


def advanced_diagnostic_fields(item: dict) -> dict[str, object]:
    return {field: item.get(field) for field in ADVANCED_DIAGNOSTIC_FIELDS if field in item}


def _terminology_lookup_ui_truthy(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(TERMINOLOGY_LOOKUP_UI_ENV_VAR, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }


def terminology_match_hypothesis_ui_plan(item: dict, *, env: dict | None = None) -> dict | None:
    """Return a safe read-only terminology metadata render plan, or None.

    MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01: this UI surface only renders
    already-emitted aggregate helper metadata. It does not create a lookup
    adapter, read terminology rows, or perform terminology lookup from the UI.
    """
    try:
        from clinical_knowledge.terminology.term_match_hypothesis import (
            TERMINOLOGY_LOOKUP_ENV_VAR,
            is_terminology_lookup_enabled,
        )
    except Exception:
        return None

    env = os.environ if env is None else env
    if not (is_terminology_lookup_enabled(env) and _terminology_lookup_ui_truthy(env)):
        return None

    metadata = item.get("terminology_match_hypothesis_metadata")
    if not isinstance(metadata, dict):
        metadata = item if item.get("terminology_match_hypothesis") is True else None
    if not isinstance(metadata, dict):
        return None

    if metadata.get("terminology_match_hypothesis") is not True:
        return None
    if metadata.get("review_required") is not True:
        return None
    if metadata.get("auto_accept_allowed") is not False:
        return None
    for flag in (
        "licensed_row_content_included",
        "raw_text_emitted",
        "raw_ocr_text_emitted",
        "raw_document_text_emitted",
        "raw_filename_emitted",
        "private_path_emitted",
        "phi_emitted",
        "secret_emitted",
        "clinical_interpretation_performed",
        "diagnosis_inference_performed",
        "treatment_inference_performed",
        "medication_inference_performed",
        "ddi_behavior_changed",
        "external_api_used",
    ):
        if metadata.get(flag) is not False:
            return None

    match_family = str(metadata.get("match_family") or "unavailable")
    system_family = str(metadata.get("terminology_system_family") or "unavailable")
    matches_count = metadata.get("matches_count")
    if not isinstance(matches_count, int):
        return None
    disclaimer = str(
        metadata.get("disclaimer")
        or "Terminology match hypothesis only. Review required. No auto-accept."
    )
    return {
        "expander_label": "Terminology match hypothesis",
        "markdown_lines": [
            f"- **Match family:** `{match_family}`",
            f"- **Terminology system family:** `{system_family}`",
            f"- **Matches count:** `{matches_count}`",
            "- **Review required:** `True`",
            "- **Auto-accept allowed:** `False`",
        ],
        "disclaimer_line": disclaimer,
        "env_vars": [TERMINOLOGY_LOOKUP_ENV_VAR, TERMINOLOGY_LOOKUP_UI_ENV_VAR],
    }


def render_run_result_card(item: dict) -> None:
    item = canonical_run_result_record(item)
    status = item_status(item)
    badge = status_badge(status)
    document_type = operator_document_type(item)
    st.markdown("<div class='medai-card'>", unsafe_allow_html=True)
    st.markdown(
        f"### {badge['label']} &nbsp; <span class='badge {badge['class']}'>Status: {badge['label']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Operator summary: confirm the document type, compare with the source, and keep technical metadata collapsed unless needed."
    )
    st.info(operator_result_explanation(document_type))

    chip_specs = [
        ("File state", operator_file_state_label(item)),
        ("Type", document_type),
        ("Text recovery", text_recovery_chip(item)),
        ("Cloud tools", "Off" if not item.get("external_api_used") else "On"),
        ("Acceptance", "Not accepted" if status != "accepted" else "Accepted"),
    ]
    cols = st.columns(len(chip_specs))
    for col, (label, value) in zip(cols, chip_specs):
        col.markdown(f"**{label}:** {value}")

    st.markdown("#### Why MedAI labeled it this way")
    for cue in operator_label_evidence(document_type):
        st.markdown(f"- {cue}")

    st.markdown("#### Russian text recovery")
    for label, value in russian_text_recovery_summary(item).items():
        st.markdown(f"- **{label}:** {value}")
    if item.get("image_ocr_engine"):
        st.markdown("#### Local image OCR")
        image_ocr_lines = {
            "OCR attempted": "Yes" if item.get("image_ocr_attempted") else "No",
            "OCR available": "Yes" if item.get("image_ocr_available") else "No",
            "Text recovery": str(item.get("image_ocr_text_visibility") or "unavailable"),
            "Cloud tools": "Off" if not item.get("external_api_used") else "On",
            "Human review still required": "Yes",
            "Acceptance": "Not accepted",
        }
        for label, value in image_ocr_lines.items():
            st.markdown(f"- **{label}:** {value}")
    if item.get("run_review_summary"):
        st.caption(str(item.get("run_review_summary")))
    if item.get("runtime_diagnostic_summary"):
        st.caption(str(item.get("runtime_diagnostic_summary")))

    st.markdown("#### What happened")
    for label, state in run_review_timeline_steps(item):
        badge_class = "badge-accepted" if state == "done" else "badge-review"
        badge_label = "Done" if state == "done" else "Needs review"
        st.markdown(f"<span class='badge {badge_class}'>{badge_label}</span> {label}", unsafe_allow_html=True)

    st.markdown("#### What you need to do next")
    for index, action in enumerate(next_actions_for_document_type(document_type), start=1):
        st.markdown(f"{index}. {action}")

    # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01: render the extracted-facts
    # preview before the "What MedAI did not do" section. The helper module
    # is Streamlit-free; failures in render are swallowed so they cannot
    # block the rest of the card.
    try:
        from app.extracted_information_preview import (
            build_extracted_information_preview_plan,
        )

        _ext_plan = build_extracted_information_preview_plan(item)
        st.markdown(f"#### {_ext_plan['section_heading']}")
        st.caption(_ext_plan["message"])
        _counts = _ext_plan.get("counts") or {}
        _cols = st.columns(3)
        _cols[0].metric(
            "Structured facts extracted",
            int(_counts.get("structured_facts_extracted", 0)),
        )
        _cols[1].metric(
            "Written to MKB",
            int(_counts.get("written_to_mkb", 0)),
        )
        _cols[2].metric(
            "Needs review",
            int(_counts.get("needs_review", 0)),
        )
        if _ext_plan.get("rows"):
            st.dataframe(
                _ext_plan["rows"],
                hide_index=True,
                use_container_width=True,
            )
        # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03: per-row operator
        # review action panel. Buttons call into the deterministic
        # operator_review_actions module. No auto-accept. No bulk-accept.
        _row_actions = _ext_plan.get("row_actions") or []
        if _row_actions:
            try:
                from app.operator_review_actions import (
                    accept_after_source_comparison as _accept_action,
                    reject_extracted_fact as _reject_action,
                    defer_extracted_fact as _defer_action,
                )

                _sql = load_system().get("sql")
                for _idx, _row_action in enumerate(_row_actions):
                    _record_id = str(_row_action.get("record_id") or "")
                    if not _record_id or _sql is None:
                        continue
                    _row = _ext_plan["rows"][_idx] if _idx < len(_ext_plan["rows"]) else {}
                    _label_parts = [
                        str(_row.get("test_name") or ""),
                        f"= {_row.get('value','')}",
                        str(_row.get("unit") or ""),
                    ]
                    st.markdown(f"##### Review: {' '.join(p for p in _label_parts if p)}")
                    _cols = st.columns(3)
                    _accept_cfg = _row_action["actions"][0]
                    _reject_cfg = _row_action["actions"][1]
                    _defer_cfg = _row_action["actions"][2]
                    st.caption(_accept_cfg.get("disclaimer", ""))
                    if _cols[0].button(
                        _accept_cfg["label"],
                        key=f"op_accept_{_record_id}",
                        disabled=not _accept_cfg.get("enabled", False),
                    ):
                        _result = _accept_action(_sql, _record_id)
                        st.info(_result.safe_message)
                        st.rerun()
                    if _cols[1].button(
                        _reject_cfg["label"],
                        key=f"op_reject_{_record_id}",
                        disabled=not _reject_cfg.get("enabled", False),
                    ):
                        _result = _reject_action(_sql, _record_id)
                        st.info(_result.safe_message)
                        st.rerun()
                    if _cols[2].button(
                        _defer_cfg["label"],
                        key=f"op_defer_{_record_id}",
                        disabled=not _defer_cfg.get("enabled", False),
                    ):
                        _result = _defer_action(_sql, _record_id)
                        st.info(_result.safe_message)
                        st.rerun()
            except Exception:
                pass
    except Exception:
        pass

    st.markdown("#### What MedAI did not do")
    for item_text in medai_did_not_do_checklist():
        st.markdown(f"- {item_text}")

    with st.expander("Advanced technical details", expanded=False):
        st.caption("Safe metadata only. This section is collapsed by default and does not replace the operator summary.")
        st.json(advanced_diagnostic_fields(item))
        # MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A: optional, read-only operator
        # review-routing badge. Default-off; rendered only when the env var
        # MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED is truthy AND the
        # underlying DIAG-06A helper returns the safe-default label for the
        # record. No buttons / actions / state mutations are attached to the
        # badge. UI rendering failures are silently swallowed so the optional
        # integration can never block the main result card.
        try:
            from clinical_knowledge.document_type.operator_badge_ui import (
                render_plan_for_operator_badge,
            )

            _op_badge_plan = render_plan_for_operator_badge(item)
            if _op_badge_plan is not None:
                st.markdown("---")
                st.markdown(f"#### {_op_badge_plan['expander_label']}")
                for _line in _op_badge_plan["markdown_lines"]:
                    st.markdown(_line)
                st.caption(_op_badge_plan["disclaimer_line"])
        except Exception:
            pass
        # MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A: optional, read-only language-
        # propagation metadata display. Default-off; rendered only when the
        # SEPARATE env var MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED
        # is truthy AND the underlying DIAG-09A helper returns the propagated
        # metadata label for the record. The DIAG-07A operator-badge env var
        # does NOT enable this display; each lever toggles independently.
        # Read-only; no buttons / actions / state mutations attached.
        try:
            from clinical_knowledge.document_type.language_propagation_operator_surface import (
                render_plan_for_language_propagation,
            )

            _lp_plan = render_plan_for_language_propagation(item)
            if _lp_plan is not None:
                st.markdown("---")
                st.markdown(f"#### {_lp_plan['expander_label']}")
                for _line in _lp_plan["markdown_lines"]:
                    st.markdown(_line)
                st.caption(_lp_plan["disclaimer_line"])
        except Exception:
            pass
        # MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A: optional, read-only Latin
        # abbreviation metadata display. Default-off; rendered only when the
        # SEPARATE env var MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED
        # is truthy AND the underlying DIAG-11A helper returns the
        # abbreviation metadata label for the record. The DIAG-07A operator-
        # badge env var and the DIAG-09A propagation env var do NOT enable
        # this display; each of the three levers toggles independently.
        # Read-only; the abbreviation is never parsed or expanded; no
        # buttons / actions / state mutations attached.
        try:
            from clinical_knowledge.document_type.latin_abbreviation_operator_surface import (
                render_plan_for_latin_abbreviation,
            )

            _la_plan = render_plan_for_latin_abbreviation(item)
            if _la_plan is not None:
                st.markdown("---")
                st.markdown(f"#### {_la_plan['expander_label']}")
                for _line in _la_plan["markdown_lines"]:
                    st.markdown(_line)
                st.caption(_la_plan["disclaimer_line"])
        except Exception:
            pass
        # MEDAI-DOC-TYPE-UNKNOWN-DIAG-19: optional, read-only PDF text /
        # layout quality metadata display. Default-off; rendered only when
        # BOTH SEPARATE env vars are truthy:
        #   * MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED (DIAG-17)
        #   * MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED   (DIAG-18)
        # The prior three doc-type env vars (DIAG-07A operator badge,
        # DIAG-09A propagation, DIAG-11A latin abbreviation) do NOT enable
        # this display; the DIAG-18 helper enforces the two-key gate
        # internally. Read-only; no buttons / forms / actions / callbacks /
        # state mutations attached; no raw text, raw filenames, or private
        # paths are rendered. The helper is imported inside the try/except
        # so non-Streamlit test collection is unaffected if either helper
        # module is absent.
        try:
            from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (
                render_plan_for_pdf_text_layout_quality,
            )

            _pl_plan = render_plan_for_pdf_text_layout_quality(item)
            if _pl_plan is not None:
                st.markdown("---")
                st.markdown(f"#### {_pl_plan['expander_label']}")
                for _line in _pl_plan["markdown_lines"]:
                    st.markdown(_line)
                st.caption(_pl_plan["disclaimer_line"])
        except Exception:
            pass
        # MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01: optional, read-only
        # terminology match hypothesis metadata display. Default-off;
        # rendered only when BOTH env vars are truthy:
        #   * MEDAI_TERMINOLOGY_LOOKUP_ENABLED
        #   * MEDAI_TERMINOLOGY_LOOKUP_UI_ENABLED
        # This block renders already-emitted aggregate helper metadata only.
        # It never creates an adapter, reads licensed terminology rows,
        # renders row codes/display strings/synonyms/definitions, or adds
        # buttons / forms / actions / callbacks / state mutations.
        try:
            _tm_plan = terminology_match_hypothesis_ui_plan(item)
            if _tm_plan is not None:
                st.markdown("---")
                st.markdown(f"#### {_tm_plan['expander_label']}")
                for _line in _tm_plan["markdown_lines"]:
                    st.markdown(_line)
                st.caption(_tm_plan["disclaimer_line"])
        except Exception:
            pass
    st.markdown("</div>", unsafe_allow_html=True)


def render_operator_guidance_panel() -> None:
    with st.expander("Result guide", expanded=False):
        for title, guidance in operator_guidance_catalog().items():
            st.markdown(f"**{title}:** {guidance}")


def render_blind_audit_tab(sys_components: dict) -> None:
    st.subheader("Validation Batch Audit")
    st.caption(navigation_subtitle("Validation Batch Audit"))
    st.caption("Put many PDFs into real_validation_input/")
    st.caption("Supported formats: PDF, TXT, RTF, TIF, TIFF, PNG, JPG, JPEG, BMP, WEBP.")
    st.caption("Run a local validation batch with PHI-safe public reports.")
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
        if st.button("Run validation batch", type="primary"):
            with st.spinner("Running local-only blind audit..."):
                report = run_blind_audit(pipeline=sys_components["execution"])
            st.session_state["phase53_blind_audit"] = report
            st.success(
                f"Validation batch complete: {report['accepted_count']} accepted, "
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
        st.error(f"Validation batch audit unavailable: {exc}")


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
    st.subheader("Validation History")
    st.caption(navigation_subtitle("Validation History"))
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


def operator_review_reason_for_item(item: dict, *, status: str | None = None) -> str:
    return item.get("operator_review_reason") or review_reason_for_result(
        document_type=item.get("document_type"),
        validation_status=item.get("validation_status"),
        confidence=_safe_metric_float(item.get("confidence")),
        status=status or item_status(item),
    )


def operator_reason_label_for_item(item: dict, reason_codes: list | None = None) -> str:
    return item.get("operator_reason_label") or reason_label_for_validation(
        item.get("validation_status"),
        [str(code) for code in (reason_codes or [])],
    )


def visible_reason_codes(reason_codes: list | None) -> list[str]:
    return [str(code) for code in (reason_codes or []) if str(code).lower() not in {"needs_review", "rejected"}]


def _safe_metric_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    startup = initialize_startup_state(load_system)
    if not startup.ok:
        render_operator_safety_panel()
        render_degraded_startup_panel(startup)
        st.divider()
        st.caption(PRIVACY_INVARIANT_GUIDANCE)
        return

    # MEDAI-UI-STARTUP-RESILIENCE-08 + 09: SQLite is OK; show a clear
    # warning banner without blocking the local clinical-review
    # workflow. The new degraded-pipeline / sqlite-only banners
    # surface the one local repair command.
    _startup_status = startup.diagnostics.app_startup_status
    if _startup_status == "app_startup_ok_with_degraded_vector":
        render_degraded_vector_banner(startup)
    elif _startup_status in (
        "app_startup_ok_with_degraded_pipeline",
        "app_startup_ok_sqlite_only",
    ):
        render_degraded_pipeline_banner(startup)

    sys_components = startup.components
    if sys_components is None:
        st.error("Startup failed without component details.")
        return
    show_advanced_tools = st.sidebar.checkbox(
        "Show advanced tools",
        value=False,
        help="Advanced tools include validation history, audit pages, safety governance, and terminology administration.",
    )
    counts = sys_components["sql"].count_records()
    render_operator_safety_panel(knowledge_counts=counts, show_build_details=show_advanced_tools)
    render_system_status(sys_components["state"], show_advanced_tools=show_advanced_tools)
    if show_advanced_tools:
        st.sidebar.caption("Advanced tools include validation history, audit pages, safety governance, and terminology administration.")

    tab_labels = operator_tab_labels(show_advanced_tools)
    tabs = st.tabs(tab_labels)
    for label, tab in zip(tab_labels, tabs):
        with tab:
            if label == RUN_REVIEW_TAB:
                render_run_review_tab(sys_components)
            elif label == MKB_EXPLORER_TAB:
                render_mkb_tab(sys_components)
            elif label == REVIEW_QUEUE_TAB:
                render_review_queue_tab(sys_components)
            elif label == "Operator Control Panel":
                try:
                    from app.operator_control_panel import render_operator_control_panel

                    render_operator_control_panel()
                except Exception as _exc:
                    st.error(f"Operator Control Panel unavailable: {_exc}")
            elif label == "Validation Batch Audit":
                render_blind_audit_tab(sys_components)
            elif label == "Validation History":
                render_report_archive_tab()
            elif label == "Safety & Governance":
                st.caption(navigation_subtitle("Safety & Governance"))
                try:
                    from app.clinical_knowledge_safety_viewer import (
                        load_cka_safety_snapshot,
                        render_clinical_knowledge_safety_dashboard,
                    )
                    _cka_snapshot = load_cka_safety_snapshot()
                    render_clinical_knowledge_safety_dashboard(_cka_snapshot)
                except Exception as _exc:
                    st.error(f"Safety & Governance panel unavailable: {_exc}")
            elif label == "Terminology Admin":
                st.caption(navigation_subtitle("Terminology Admin"))
                try:
                    from app.terminology_readiness_viewer import render_terminology_readiness_panel

                    render_terminology_readiness_panel()
                except Exception as _exc:
                    st.error(f"Terminology Admin panel unavailable: {_exc}")
            elif label == VERTEX_DECISION_AUDIT_TAB:
                st.caption(navigation_subtitle(VERTEX_DECISION_AUDIT_TAB))
                render_vertex_semantic_review_decision_audit_panel_hook()
            elif label == TERMINOLOGY_LOOKUP_TAB:
                try:
                    from app.clinical_knowledge_terminology_lookup_viewer import render_terminology_lookup_panel

                    render_terminology_lookup_panel()
                except Exception as _exc:
                    st.error(f"Terminology Lookup panel unavailable: {_exc}")

    st.divider()
    st.caption(PRIVACY_INVARIANT_GUIDANCE)


if __name__ == "__main__":
    main()
