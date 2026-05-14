"""CKA-TERM-07 operator-only terminology lookup viewer.

The panel is hidden unless explicitly enabled. It opens the TERM-02 local
terminology store read-only, displays only adapter-safe lookup metadata, and
never writes clinical annotations or integrates with B07.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.term02_controlled_import import TERM02_DB_RELATIVE
from clinical_knowledge.terminology.term05_read_only_adapter import (
    SyntheticReadOnlyTerminologyAdapter,
    Term05AdapterResult,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERM07_FEATURE_FLAG = "MEDAI_TERMINOLOGY_LOOKUP_UI"
SOURCE_FILTERS = ("all", "rxnorm", "loinc")


@dataclass(frozen=True)
class TerminologyLookupStoreStatus:
    feature_flag_enabled: bool
    local_only_mode: bool
    store_available: bool
    store_opened_read_only: bool
    private_path_displayed: bool = False
    external_api_used: bool = False

    def safe_public_summary(self) -> dict[str, Any]:
        return {
            "feature_flag_enabled": self.feature_flag_enabled,
            "local_only_mode": self.local_only_mode,
            "store_available": self.store_available,
            "store_opened_read_only": self.store_opened_read_only,
            "private_path_displayed": self.private_path_displayed,
            "external_api_used": self.external_api_used,
        }


def terminology_lookup_panel_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get(TERM07_FEATURE_FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


def local_only_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get("MEDAI_LOCAL_ONLY", "1")).strip().lower() not in {"0", "false", "no", "off"}


def normalize_source_filter(source_filter: str | None) -> tuple[str, ...]:
    value = (source_filter or "all").strip().lower()
    if value not in SOURCE_FILTERS:
        raise ValueError(f"unsupported_source_filter:{value}")
    return () if value == "all" else (value,)


def get_lookup_store_status(
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> TerminologyLookupStoreStatus:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    db_path = root / TERM02_DB_RELATIVE
    return TerminologyLookupStoreStatus(
        feature_flag_enabled=terminology_lookup_panel_enabled(env),
        local_only_mode=local_only_mode_enabled(env),
        store_available=db_path.exists(),
        store_opened_read_only=False,
    )


def run_local_lookup(
    query: str,
    *,
    source_filter: str = "all",
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Run one local read-only lookup and return UI-safe metadata.

    The returned value may include adapter-safe code/display fields for local
    operator visibility, but never includes private paths or raw source rows.
    Public validation reports should summarize counts/statuses only.
    """
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    db_path = root / TERM02_DB_RELATIVE
    if not terminology_lookup_panel_enabled(env):
        return _unavailable_result("feature_flag_disabled", source_filter=source_filter)
    if not local_only_mode_enabled(env):
        return _unavailable_result("local_only_mode_required", source_filter=source_filter)
    if not db_path.exists():
        return _unavailable_result("store_missing", source_filter=source_filter)

    systems = normalize_source_filter(source_filter)
    store = _open_read_only_store(db_path)
    adapter = SyntheticReadOnlyTerminologyAdapter(
        store=store,
        fixture_metadata={"synthetic_only": False, "private_store_read_only": True, "term07_ui_only": True},
    )
    result = adapter.lookup(query, source_filter=systems, max_results=max_results)
    safe = _safe_lookup_result(result)
    safe["store_available"] = True
    safe["store_opened_read_only"] = True
    safe["feature_flag_enabled"] = True
    safe["local_only_mode"] = True
    safe["private_path_displayed"] = False
    safe["clinical_write_performed"] = False
    safe["automatic_annotation_created"] = False
    return safe


def render_lookup_result_text(result: Mapping[str, Any]) -> str:
    status = str(result.get("status") or "unavailable")
    match_count = int(result.get("match_count") or 0)
    sources = ", ".join(result.get("source_systems") or []) or "none"
    reasons = ", ".join(result.get("reason_codes") or []) or "none"
    return "\n".join(
        [
            f"Status: {status}",
            f"Match count: {match_count}",
            f"Source systems: {sources}",
            f"Reason codes: {reasons}",
            f"Read-only: {bool(result.get('read_only', True))}",
            f"External API used: {bool(result.get('external_api_used', False))}",
        ]
    )


def render_terminology_lookup_panel() -> None:
    """Render Streamlit UI. Import Streamlit lazily for import-safe tests."""
    import streamlit as st

    enabled = terminology_lookup_panel_enabled()
    status = get_lookup_store_status()
    st.subheader("Terminology Lookup")
    st.caption("Operator-only local read-only lookup. No extraction, B07, MKB write, or clinical workflow is changed.")
    st.write(f"Feature flag `{TERM07_FEATURE_FLAG}`: {'ON' if enabled else 'OFF'}")
    st.write(f"Local-only mode: {'ON' if status.local_only_mode else 'OFF'}")
    st.write(f"Store available: {'YES' if status.store_available else 'NO'}")

    if not enabled:
        st.info(f"Terminology lookup panel is disabled. Set `{TERM07_FEATURE_FLAG}=1` to enable local operator lookup.")
        return
    if not status.local_only_mode:
        st.error("Local-only mode is required. Terminology lookup remains blocked.")
        return
    if not status.store_available:
        st.warning("Local terminology store is not available. The UI remains safe and no lookup is attempted.")
        return

    source_filter = st.selectbox("Source filter", SOURCE_FILTERS, index=0)
    query = st.text_input("Lookup query", value="", help="Local-only operator lookup. Results are not written anywhere.")
    if st.button("Run local read-only terminology lookup"):
        result = run_local_lookup(query, source_filter=source_filter)
        st.markdown(f"**Status:** `{result.get('status')}`")
        st.markdown(f"**Match count:** `{result.get('match_count')}`")
        st.markdown(f"**Read-only:** `{result.get('read_only')}`")
        st.markdown(f"**External API used:** `{result.get('external_api_used')}`")
        st.markdown(f"**Reason codes:** {', '.join(result.get('reason_codes') or []) or 'none'}")
        for match in result.get("matches", []):
            st.write(
                {
                    "system": match.get("system"),
                    "code": match.get("code"),
                    "display_normalized": match.get("display_normalized"),
                }
            )


def _open_read_only_store(db_path: Path) -> LocalTerminologyStore:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.execute("SELECT 1").fetchone()
    store = object.__new__(LocalTerminologyStore)
    store.db_path = uri
    store._mem_con = None
    store._uri = True
    return store


def _safe_lookup_result(result: Term05AdapterResult) -> dict[str, Any]:
    matches = [match.safe_public_summary() for match in result.matches]
    return {
        "status": result.status,
        "source_filter": list(result.source_filter),
        "match_count": len(matches),
        "source_systems": sorted({str(match.get("system")) for match in matches}),
        "matches": matches,
        "confidence_label": result.confidence_label,
        "reason_codes": list(result.reason_codes),
        "read_only": result.read_only,
        "external_api_used": result.external_api_used,
        "clinical_advice_generated": result.clinical_advice_generated,
        "dosing_advice_generated": result.dosing_advice_generated,
        "mkb_write_performed": result.mkb_write_performed,
        "b07_integrated": result.b07_integrated,
        "ddi_status_cleared": result.ddi_status_cleared,
        "hypothesis_promoted": result.hypothesis_promoted,
        "no_code_hallucinated": result.no_code_hallucinated,
    }


def _unavailable_result(reason_code: str, *, source_filter: str) -> dict[str, Any]:
    return {
        "status": "unavailable" if reason_code != "store_missing" else "unmapped",
        "source_filter": list(normalize_source_filter(source_filter)),
        "match_count": 0,
        "source_systems": [],
        "matches": [],
        "confidence_label": "none",
        "reason_codes": [reason_code],
        "read_only": True,
        "external_api_used": False,
        "clinical_advice_generated": False,
        "dosing_advice_generated": False,
        "mkb_write_performed": False,
        "automatic_annotation_created": False,
        "b07_integrated": False,
        "ddi_status_cleared": False,
        "hypothesis_promoted": False,
        "no_code_hallucinated": True,
        "store_available": False,
        "store_opened_read_only": False,
        "private_path_displayed": False,
        "clinical_write_performed": False,
    }
