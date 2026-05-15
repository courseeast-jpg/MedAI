"""CKA-B09 Clinical Knowledge Safety Panels for Operator UI.

Provides:
- load_cka_safety_snapshot()    — load CKA public reports into snapshot dict
- get_cka_block_status_summary() — aggregate block readiness flags
- get_cka_safety_flags()         — aggregate privacy/safety flags
- get_cka_operator_panels()      — summary of all panel states
- render_*_panel()               — Streamlit render helpers (one per block)
- render_clinical_knowledge_safety_dashboard() — master dashboard renderer

Rules:
- Reads only public report JSON files (never *_PRIVATE.json).
- Never loads replacement_map or source_response_raw.
- Never displays raw PHI, private filenames, or raw paths.
- Never displays clinical recommendations or prescription dosing advice.
- Never claims MedAI is a medical device, autonomous, or diagnostic/prescribing.
- All Streamlit imports are guarded so module can be imported without streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Public report paths — keyed by block_id
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

_CKA_REPORT_PATHS: Dict[str, Path] = {
    "CKA-B01": _PROJECT_ROOT / "reports" / "cka_block01_mkb_foundation" / "cka_block01_mkb_foundation_report.json",
    "CKA-B02": _PROJECT_ROOT / "reports" / "cka_block02_privacy_boundary" / "cka_block02_privacy_boundary_report.json",
    "CKA-B03": _PROJECT_ROOT / "reports" / "cka_block03_decision_engine" / "cka_block03_decision_engine_report.json",
    "CKA-B04": _PROJECT_ROOT / "reports" / "cka_block04_truth_resolution" / "cka_block04_truth_resolution_report.json",
    "CKA-B05": _PROJECT_ROOT / "reports" / "cka_block05_medication_safety" / "cka_block05_medication_safety_report.json",
    "CKA-B06": _PROJECT_ROOT / "reports" / "cka_block06_controlled_enrichment" / "cka_block06_controlled_enrichment_report.json",
    "CKA-B07": _PROJECT_ROOT / "reports" / "cka_block07_medical_coding" / "cka_block07_medical_coding_report.json",
    "CKA-B08": _PROJECT_ROOT / "reports" / "cka_block08_multi_connector_consensus" / "cka_block08_multi_connector_consensus_report.json",
    "CKA-B09": _PROJECT_ROOT / "reports" / "cka_block09_operator_ui" / "cka_block09_operator_ui_report.json",
    "CKA-B10": _PROJECT_ROOT / "reports" / "cka_block10_preflight_scaffold" / "cka_block10_preflight_scaffold_report.json",
}

_UNSAFE_KEYS = frozenset({
    "replacement_map",
    "source_response_raw",
    "raw_source_text",
    "private_payload",
    "private_text",
})


def _is_private_file(path: Path) -> bool:
    """Return True if the file looks like a private report (never load)."""
    name = path.name.lower()
    return "_private" in name or "private_" in name


def _strip_unsafe_keys(data: Any) -> Any:
    """Recursively remove unsafe keys from dicts."""
    if isinstance(data, dict):
        return {k: _strip_unsafe_keys(v) for k, v in data.items() if k not in _UNSAFE_KEYS}
    if isinstance(data, list):
        return [_strip_unsafe_keys(item) for item in data]
    return data


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single CKA public report JSON. Returns None if unavailable."""
    if _is_private_file(path):
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _strip_unsafe_keys(data)


# ---------------------------------------------------------------------------
# Public API: data loading helpers
# ---------------------------------------------------------------------------

def load_cka_safety_snapshot(
    report_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load CKA public reports into a unified snapshot dict.

    - Never reads *_PRIVATE.json files.
    - Never loads replacement_map or source_response_raw.
    - Returns graceful empty state if reports are missing.
    """
    paths_map = dict(_CKA_REPORT_PATHS)

    if report_paths:
        for raw_path in report_paths:
            p = Path(raw_path)
            if _is_private_file(p):
                continue   # silently skip private files
            data = _load_report(p)
            if data:
                block_id = data.get("block_id", p.stem)
                paths_map[block_id] = p

    reports: Dict[str, Any] = {}
    for block_id, path in paths_map.items():
        data = _load_report(path)
        if data is not None:
            reports[block_id] = data

    return {
        "reports": reports,
        "blocks_loaded": list(reports.keys()),
        "blocks_missing": [b for b in _CKA_REPORT_PATHS if b not in reports],
        "private_files_read": False,
        "replacement_map_loaded": False,
        "source_response_raw_loaded": False,
    }


def get_cka_block_status_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate block readiness flags from the snapshot."""
    reports = snapshot.get("reports", {})

    def _flag(block: str, key: str, default: bool = False) -> bool:
        return bool(reports.get(block, {}).get(key, default))

    # B05 medication-safety readiness is composed of the three actual flags
    # carried in the public B05 report. We keep this conjunctive so a single
    # missing layer does not falsely report "ready".
    b05 = reports.get("CKA-B05", {})
    medication_safety_ready = bool(
        b05.get("ddi_stub_ready", False)
        and b05.get("ddi_layer1_evidence_modifier_ready", False)
        and b05.get("ddi_layer2_write_gate_ready", False)
    ) if b05 else False

    return {
        "CKA-B01_ready": "CKA-B01" in reports,
        "CKA-B02_ready": "CKA-B02" in reports,
        "CKA-B03_ready": "CKA-B03" in reports,
        "CKA-B04_ready": "CKA-B04" in reports,
        "CKA-B05_ready": "CKA-B05" in reports,
        "CKA-B06_ready": "CKA-B06" in reports,
        "CKA-B07_ready": "CKA-B07" in reports,
        "CKA-B08_ready": "CKA-B08" in reports,
        "CKA-B09_ready": "CKA-B09" in reports,
        "CKA-B10_ready": "CKA-B10" in reports,
        "blocks_loaded_count": len(reports),
        "blocks_missing": snapshot.get("blocks_missing", []),
        # Selected readiness flags from individual reports.
        # NOTE: keys below match what each block's public report actually
        # carries; if a key is missing the flag stays False (no claim made).
        "safe_mode_ready": _flag("CKA-B03", "safe_mode_tested"),
        "truth_resolution_ready": _flag("CKA-B04", "truth_resolution_ready"),
        "quarantine_ready": _flag("CKA-B04", "quarantine_ready"),
        "medication_safety_ready": medication_safety_ready,
        "enrichment_ready": _flag("CKA-B06", "controlled_enrichment_ready"),
        "medical_coding_ready": _flag("CKA-B07", "medical_coding_interface_ready"),
        "consensus_ready": _flag("CKA-B08", "consensus_engine_ready"),
        "operator_ui_ready": _flag("CKA-B09", "streamlit_integration_ready"),
        "preflight_scaffold_ready": _flag("CKA-B10", "all_passed"),
    }


def get_cka_safety_flags(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate privacy/safety flags from all block reports."""
    reports = snapshot.get("reports", {})

    # Collect worst-case across all blocks
    raw_phi = any(r.get("raw_phi_logged_in_public_reports", False) for r in reports.values())
    path_leaks = sum(r.get("private_filename_path_leaks", 0) for r in reports.values())
    secret_leaks = sum(r.get("secret_leaks", 0) for r in reports.values())
    replacement_map = any(r.get("replacement_map_written_to_public_reports", False) for r in reports.values())
    ext_api = any(r.get("external_api_used", False) for r in reports.values())
    clinical_recs = any(r.get("clinical_recommendations_generated", False) for r in reports.values())
    dosing = any(r.get("prescription_dosing_advice_generated", False) for r in reports.values())
    hitl_reopened = any(r.get("frozen_hitl_release_reopened", False) for r in reports.values())
    ocr_changed = any(r.get("production_ocr_changed", False) for r in reports.values())
    extractor_changed = any(r.get("production_extractor_changed", False) for r in reports.values())
    gate_changed = any(r.get("safety_gate_changed", False) for r in reports.values())

    return {
        "raw_phi_logged_in_public_reports": raw_phi,
        "private_filename_path_leaks": path_leaks,
        "secret_leaks": secret_leaks,
        "replacement_map_written_to_public_reports": replacement_map,
        "external_api_used": ext_api,
        "clinical_recommendations_generated": clinical_recs,
        "prescription_dosing_advice_generated": dosing,
        "frozen_hitl_release_reopened": hitl_reopened,
        "production_ocr_changed": ocr_changed,
        "production_extractor_changed": extractor_changed,
        "safety_gate_changed": gate_changed,
        "private_files_read": snapshot.get("private_files_read", False),
        "replacement_map_loaded": snapshot.get("replacement_map_loaded", False),
        "source_response_raw_loaded": snapshot.get("source_response_raw_loaded", False),
        "all_clear": not any([
            raw_phi, path_leaks > 0, secret_leaks > 0, replacement_map,
            ext_api, clinical_recs, dosing, hitl_reopened,
            ocr_changed, extractor_changed, gate_changed,
        ]),
    }


def get_cka_operator_panels(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict of all panel readiness states for use in tests/reports."""
    status = get_cka_block_status_summary(snapshot)
    flags = get_cka_safety_flags(snapshot)
    return {
        "mkb_status_panel_ready": "CKA-B01" in snapshot.get("reports", {}),
        "decision_engine_panel_ready": "CKA-B03" in snapshot.get("reports", {}),
        "privacy_panel_ready": "CKA-B02" in snapshot.get("reports", {}),
        "truth_resolution_panel_ready": "CKA-B04" in snapshot.get("reports", {}),
        "medication_safety_panel_ready": "CKA-B05" in snapshot.get("reports", {}),
        "enrichment_panel_ready": "CKA-B06" in snapshot.get("reports", {}),
        "medical_coding_panel_ready": "CKA-B07" in snapshot.get("reports", {}),
        "consensus_panel_ready": "CKA-B08" in snapshot.get("reports", {}),
        "release_readiness_panel_ready": True,
        "panels_count": 9,
        "block_status": status,
        "safety_flags": flags,
    }


# ---------------------------------------------------------------------------
# Render helpers — each returns a string summary for non-Streamlit use
# and renders via Streamlit when called in a Streamlit context
# ---------------------------------------------------------------------------

def _st_ok() -> bool:
    """Return True if streamlit is importable and a session is active."""
    try:
        import streamlit as st
        return True
    except ImportError:
        return False


def _badge(ok: bool) -> str:
    """ASCII-safe badge for text output (emojis used in Streamlit path separately)."""
    return "[OK]" if ok else "[WARN]"


def render_mkb_status_panel(snapshot: Dict[str, Any]) -> str:
    """Render the MKB tier/status panel. Returns text summary."""
    report = snapshot.get("reports", {}).get("CKA-B01", {})
    lines = ["## MKB Tier / Status Panel"]
    if not report:
        lines.append("[WARN] CKA-B01 report unavailable.")
    else:
        lines.append(f"Active records (validation): {report.get('active_records_count', 'N/A')}")
        lines.append(f"Hypothesis records: {report.get('hypothesis_records_count', 'N/A')}")
        lines.append(f"Quarantined: {report.get('quarantined_records_count', 'N/A')}")
        lines.append(f"Superseded: {report.get('superseded_records_count', 'N/A')}")
        lines.append(f"Ledger events written: {report.get('ledger_events_written', 'N/A')}")
        lines.append(f"Encryption boundary ready: {_badge(report.get('encryption_boundary_ready', False))}")
    if _st_ok():
        import streamlit as st
        st.subheader("🗂 MKB Tier / Status")
        if not report:
            st.warning("CKA-B01 report unavailable.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Active (validation)", report.get("active_records_count", "N/A"))
            c2.metric("Hypothesis", report.get("hypothesis_records_count", "N/A"))
            c3.metric("Quarantined", report.get("quarantined_records_count", "N/A"))
            c4.metric("Superseded", report.get("superseded_records_count", "N/A"))
            st.caption(f"Ledger events: {report.get('ledger_events_written', 'N/A')} | "
                       f"Encryption boundary: {_badge(report.get('encryption_boundary_ready', False))}")
    return "\n".join(lines)


def render_decision_engine_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Safe Mode / Decision Engine panel."""
    report = snapshot.get("reports", {}).get("CKA-B03", {})
    lines = ["## Safe Mode / Decision Engine Panel"]
    if not report:
        lines.append("[WARN] CKA-B03 report unavailable.")
    else:
        lines.append(f"Safe mode ready: {_badge(report.get('safe_mode_ready', False))}")
        lines.append(f"Connector stubs local-only: {_badge(not report.get('external_api_used', True))}")
        lines.append(f"Response scoring ready: {_badge(report.get('response_scoring_ready', False))}")
        lines.append(f"MKB-only mode: available when all connectors fail")
        lines.append("Note: No clinical answer content displayed.")
    if _st_ok():
        import streamlit as st
        st.subheader("🛡 Safe Mode / Decision Engine")
        if not report:
            st.warning("CKA-B03 report unavailable.")
        else:
            st.success("Safe mode ready") if report.get("safe_mode_ready") else st.warning("Safe mode: check report")
            st.caption(f"External API used: {report.get('external_api_used', False)} | "
                       f"Response scoring: {_badge(report.get('response_scoring_ready', False))}")
            st.info("ℹ️ No clinical answer content is displayed by this panel.")
    return "\n".join(lines)


def render_privacy_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Privacy Audit panel."""
    report = snapshot.get("reports", {}).get("CKA-B02", {})
    flags = get_cka_safety_flags(snapshot)
    lines = ["## Privacy Audit Panel"]
    blocked = (
        flags.get("raw_phi_logged_in_public_reports")
        or flags.get("private_filename_path_leaks", 0) > 0
        or flags.get("secret_leaks", 0) > 0
        or flags.get("replacement_map_written_to_public_reports")
    )
    if blocked:
        lines.append("[BLOCKED / REVIEW REQUIRED] -- privacy flags raised.")
    else:
        lines.append("[OK] All privacy checks clear.")
    lines.append(f"raw_phi_logged: {flags.get('raw_phi_logged_in_public_reports')}")
    lines.append(f"private_filename_path_leaks: {flags.get('private_filename_path_leaks', 0)}")
    lines.append(f"secret_leaks: {flags.get('secret_leaks', 0)}")
    lines.append(f"private_mapping_written_to_reports: {flags.get('replacement_map_written_to_public_reports')}")
    lines.append(f"external_api_used: {flags.get('external_api_used')}")
    if _st_ok():
        import streamlit as st
        st.subheader("🔒 Privacy Audit")
        if blocked:
            st.error("🚨 BLOCKED / REVIEW REQUIRED — One or more privacy flags raised across CKA blocks.")
        else:
            st.success("✅ All privacy checks clear across CKA-B01–B08.")
        col1, col2 = st.columns(2)
        col1.metric("PHI in public reports", "NO" if not flags.get("raw_phi_logged_in_public_reports") else "YES ⚠️")
        col1.metric("Path leaks", flags.get("private_filename_path_leaks", 0))
        col2.metric("Secret leaks", flags.get("secret_leaks", 0))
        col2.metric("Private mapping written", "NO" if not flags.get("replacement_map_written_to_public_reports") else "YES ⚠️")
    return "\n".join(lines)


def render_truth_resolution_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Truth Resolution / Quarantine panel."""
    report = snapshot.get("reports", {}).get("CKA-B04", {})
    lines = ["## Truth Resolution / Quarantine Panel"]
    if not report:
        lines.append("[WARN] CKA-B04 report unavailable.")
    else:
        lines.append(f"Truth resolution ready: {_badge(report.get('truth_resolution_ready', False))}")
        lines.append(f"Quarantine ready: {_badge(report.get('quarantine_ready', False))}")
        lines.append(f"Ordered rules enforced: {_badge(report.get('ordered_rules_enforced', False))}")
        lines.append(f"Active retrieval excludes quarantined/superseded: {_badge(report.get('active_retrieval_excludes_quarantined', False))}")
        lines.append(f"Medication dose conflict quarantines only: {_badge(report.get('medication_dose_conflict_quarantines_only', False))}")
        lines.append("Note: No dose values or medication advice shown.")
    if _st_ok():
        import streamlit as st
        st.subheader("⚖️ Truth Resolution / Quarantine")
        if not report:
            st.warning("CKA-B04 report unavailable.")
        else:
            cols = st.columns(3)
            cols[0].metric("Truth resolution", "Ready" if report.get("truth_resolution_ready") else "Check")
            cols[1].metric("Quarantine", "Ready" if report.get("quarantine_ready") else "Check")
            cols[2].metric("Ordered rules", "Enforced" if report.get("ordered_rules_enforced") else "Check")
            st.caption("Medication dose conflicts → quarantine only. No DDI invocation from Truth Resolution.")
            st.info("ℹ️ No dose values or medication advice displayed.")
    return "\n".join(lines)


def render_medication_safety_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Medication Safety / DDI panel."""
    report = snapshot.get("reports", {}).get("CKA-B05", {})
    lines = ["## Medication Safety / DDI Panel"]
    if not report:
        lines.append("[WARN] CKA-B05 report unavailable.")
    else:
        lines.append(f"DDI stub ready: {_badge(report.get('ddi_stub_ready', False))}")
        lines.append(f"Layer 1 evidence modifier ready: {_badge(report.get('layer1_evidence_modifier_ready', False))}")
        lines.append(f"Layer 2 write gate ready: {_badge(report.get('layer2_write_gate_ready', False))}")
        lines.append(f"HIGH interaction blocks without confirmation: {_badge(report.get('high_interaction_blocks_without_confirmation', False))}")
        lines.append(f"MEDIUM requires acknowledgment: {_badge(report.get('medium_interaction_requires_acknowledgment', False))}")
        lines.append(f"DDI unavailable queues pending: {_badge(report.get('ddi_unavailable_queues_pending', False))}")
        lines.append(f"Real PatientNotes API used: {report.get('real_patientnotes_api_used', False)}")
        lines.append("Note: No medication recommendations or dose advice displayed.")
    if _st_ok():
        import streamlit as st
        st.subheader("💊 Medication Safety / DDI")
        if not report:
            st.warning("CKA-B05 report unavailable.")
        else:
            cols = st.columns(3)
            cols[0].metric("DDI stub", "Ready" if report.get("ddi_stub_ready") else "Check")
            cols[1].metric("Layer 1 modifier", "Ready" if report.get("layer1_evidence_modifier_ready") else "Check")
            cols[2].metric("Layer 2 gate", "Ready" if report.get("layer2_write_gate_ready") else "Check")
            st.caption(
                "HIGH → blocks | MEDIUM → acknowledgment | LOW → note | "
                "PatientNotes API: synthetic stub only"
            )
            st.info("ℹ️ No medication recommendations or dosing advice displayed by this panel.")
    return "\n".join(lines)


def render_enrichment_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Enrichment / Hypothesis panel."""
    report = snapshot.get("reports", {}).get("CKA-B06", {})
    lines = ["## Enrichment / Hypothesis Panel"]
    if not report:
        lines.append("[WARN] CKA-B06 report unavailable.")
    else:
        lines.append(f"All AI-derived facts hypothesis-only: {_badge(report.get('ai_facts_written_active', True) is False)}")
        lines.append(f"AI facts written active: {report.get('ai_facts_written_active', False)}")
        lines.append(f"ENRICH_PROMOTE default false: {_badge(not report.get('enrich_promote_enabled_by_default', True))}")
        lines.append(f"Auto-promotion blocked: {_badge(report.get('auto_promotion_blocked', False))}")
        lines.append(f"Medication candidates pass through DDI gate: {_badge(report.get('medication_candidates_pass_ddi_gate', False))}")
        lines.append(f"Safe mode disables enrichment: {_badge(report.get('safe_mode_disables_enrichment', False))}")
        lines.append("Note: No raw connector responses displayed.")
    if _st_ok():
        import streamlit as st
        st.subheader("🔬 Enrichment / Hypothesis")
        if not report:
            st.warning("CKA-B06 report unavailable.")
        else:
            ai_active = report.get("ai_facts_written_active", False)
            if ai_active:
                st.error("⚠️ AI facts written active — review required.")
            else:
                st.success("✅ All AI-derived facts remain hypothesis-only.")
            st.caption(
                f"ENRICH_PROMOTE default=false | Auto-promotion blocked | "
                f"Medication via DDI gate | Safe mode disables enrichment"
            )
            st.info("ℹ️ No raw source response text displayed.")
    return "\n".join(lines)


def render_medical_coding_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Medical Coding panel."""
    report = snapshot.get("reports", {}).get("CKA-B07", {})
    lines = ["## Medical Coding Panel"]
    if not report:
        lines.append("[WARN] CKA-B07 report unavailable.")
    else:
        lines.append(f"Medical coding interface ready: {_badge(report.get('medical_coding_interface_ready', False))}")
        lines.append(f"Synthetic mapper ready: {_badge(report.get('synthetic_mapper_ready', False))}")
        lines.append(f"Local lookup ready: {_badge(report.get('local_lookup_loader_ready', False))}")
        lines.append(f"Unknown entities remain unmapped: {_badge(report.get('unknown_entities_remain_unmapped', False))}")
        lines.append(f"No code hallucinated: {_badge(report.get('no_code_hallucinated', False))}")
        lines.append(f"Real UMLS API used: {report.get('real_umls_api_used', False)}")
        lines.append(f"Real SNOMED download: {report.get('real_snomed_download_used', False)}")
        lines.append(f"scispaCy required: {report.get('real_scispacy_linker_required', False)}")
        lines.append(f"Coding does not promote hypothesis: {_badge(report.get('coding_does_not_promote_hypothesis', False))}")
        lines.append(f"Coding does not clear DDI status: {_badge(report.get('coding_does_not_clear_ddi_status', False))}")
        lines.append("Note: Real clinical coding correctness not asserted for synthetic-only stubs.")
    if _st_ok():
        import streamlit as st
        st.subheader("🏷 Medical Coding")
        if not report:
            st.warning("CKA-B07 report unavailable.")
        else:
            cols = st.columns(3)
            cols[0].metric("Interface", "Ready" if report.get("medical_coding_interface_ready") else "Check")
            cols[1].metric("No hallucination", "✅" if report.get("no_code_hallucinated") else "⚠️")
            cols[2].metric("Unmapped unknowns", "✅" if report.get("unknown_entities_remain_unmapped") else "⚠️")
            st.caption(
                "No real UMLS/SNOMED/scispaCy required. "
                "Coding is metadata normalisation only — not clinical coding certification."
            )
            st.info("ℹ️ Real clinical coding correctness not asserted for synthetic-only stubs.")
    return "\n".join(lines)


def render_consensus_panel(snapshot: Dict[str, Any]) -> str:
    """Render the Multi-Connector Consensus panel."""
    report = snapshot.get("reports", {}).get("CKA-B08", {})
    lines = ["## Multi-Connector Consensus Panel"]
    if not report:
        lines.append("[WARN] CKA-B08 report unavailable.")
    else:
        lines.append(f"Connector registry ready: {_badge(report.get('connector_registry_ready', False))}")
        lines.append(f"Privacy-gated requests ready: {_badge(report.get('privacy_gated_requests_ready', False))}")
        lines.append(f"Connector stubs ready: {_badge(report.get('connector_stubs_ready', False))}")
        lines.append(f"Consensus engine ready: {_badge(report.get('consensus_engine_ready', False))}")
        lines.append(f"Agreement scoring ready: {_badge(report.get('agreement_scoring_ready', False))}")
        lines.append(f"Contradiction detection ready: {_badge(report.get('contradiction_detection_ready', False))}")
        lines.append(f"Truth Resolution handoff ready: {_badge(report.get('truth_resolution_handoff_ready', False))}")
        lines.append(f"Real external connectors implemented: {report.get('real_external_connectors_implemented', False)}")
        lines.append(f"Consensus does not synthesize over contradiction: {_badge(report.get('consensus_does_not_synthesize_over_contradiction', False))}")
        lines.append(f"Consensus does not auto-write active: {_badge(report.get('consensus_does_not_auto_write_active', False))}")
        lines.append(f"Consensus-to-enrichment remains hypothesis: {_badge(report.get('consensus_to_enrichment_remains_hypothesis', False))}")
    if _st_ok():
        import streamlit as st
        st.subheader("🔗 Multi-Connector Consensus")
        if not report:
            st.warning("CKA-B08 report unavailable.")
        else:
            cols = st.columns(4)
            cols[0].metric("Registry", "Ready" if report.get("connector_registry_ready") else "Check")
            cols[1].metric("Consensus engine", "Ready" if report.get("consensus_engine_ready") else "Check")
            cols[2].metric("No synthesis over contradiction", "✅" if report.get("consensus_does_not_synthesize_over_contradiction") else "⚠️")
            cols[3].metric("No auto-write active", "✅" if report.get("consensus_does_not_auto_write_active") else "⚠️")
            st.caption(
                "All connectors: local stubs only. "
                "Consensus → enrichment: hypothesis-only. "
                "Contradictions → Truth Resolution quarantine."
            )
    return "\n".join(lines)


def render_cka_release_readiness_panel(snapshot: Dict[str, Any]) -> str:
    """Render the overall CKA release readiness panel."""
    status = get_cka_block_status_summary(snapshot)
    flags = get_cka_safety_flags(snapshot)
    blocks_complete = status.get("blocks_loaded_count", 0)
    lines = [
        "## CKA Release Readiness",
        f"CKA architecture blocks completed: B01–B0{min(blocks_complete, 8)} ({blocks_complete}/8)",
        "UI block B09: in progress / ready",
        "Next: CKA-B10 Final CKA Validation / MVP Release Package",
        "",
        "⚠️  MedAI is NOT production-autonomous.",
        "⚠️  MedAI is NOT a medical device.",
        "⚠️  MedAI does NOT perform autonomous diagnosis.",
        "⚠️  MedAI does NOT issue medication orders.",
        "",
        f"All connectors: local stubs only — external_api_used={flags.get('external_api_used', False)}",
        f"Privacy clear: {_badge(flags.get('all_clear', False))}",
        f"HITL release frozen: {_badge(not flags.get('frozen_hitl_release_reopened', True))}",
    ]
    if _st_ok():
        import streamlit as st
        st.subheader("📋 CKA Release Readiness")
        st.info(
            f"**CKA blocks completed:** B01–B08 ({blocks_complete}/8)  \n"
            "**UI block B09:** In progress / ready  \n"
            "**Next:** CKA-B10 Final CKA Validation / MVP Release Package"
        )
        st.warning(
            "**⚠️ This system is NOT production-autonomous.  \n"
            "⚠️ This system is NOT a medical device.  \n"
            "⚠️ This system does NOT perform autonomous clinical diagnosis.  \n"
            "⚠️ This system does NOT prescribe medication.**"
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Blocks loaded", f"{blocks_complete}/8")
        col2.metric("Privacy", "✅ Clear" if flags.get("all_clear") else "⚠️ Review")
        col3.metric("HITL freeze", "🔒 Closed" if not flags.get("frozen_hitl_release_reopened") else "⚠️ Opened")
        st.caption("All connectors: local synthetic stubs only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master dashboard
# ---------------------------------------------------------------------------

def render_clinical_knowledge_safety_dashboard(snapshot: Optional[Dict[str, Any]] = None) -> None:
    """Render the full CKA Safety dashboard inside an active Streamlit session.

    Safe to call even without streamlit — silently degrades.
    """
    if snapshot is None:
        snapshot = load_cka_safety_snapshot()

    if not _st_ok():
        return

    import streamlit as st

    st.header("🏥 Clinical Knowledge Architecture — Safety State")
    st.caption(
        "Safety checks for privacy, knowledge state, and controlled clinical logic. "
        "Displays safe public report summaries only. No PHI, no private filenames, "
        "no private mappings, no clinical advice."
    )

    panels = [
        ("MKB Tier / Status", render_mkb_status_panel),
        ("Privacy Audit", render_privacy_panel),
        ("Decision Engine / Safe Mode", render_decision_engine_panel),
        ("Truth Resolution / Quarantine", render_truth_resolution_panel),
        ("Medication Safety / DDI", render_medication_safety_panel),
        ("Enrichment / Hypothesis", render_enrichment_panel),
        ("Medical Coding", render_medical_coding_panel),
        ("Multi-Connector Consensus", render_consensus_panel),
        ("Release Readiness", render_cka_release_readiness_panel),
    ]

    for panel_name, render_fn in panels:
        with st.expander(panel_name, expanded=False):
            try:
                render_fn(snapshot)
            except Exception as exc:
                st.error(f"Panel error ({panel_name}): {exc}")

    st.divider()
    st.caption(
        "CKA-B09 Operator UI — public reports only — "
        "no PHI, no private paths, no clinical advice."
    )
