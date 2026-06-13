"""No-live, read-only tests for 15S Vertex semantic decision audit/export."""
from __future__ import annotations

import csv
import hashlib
import io
import json

from execution.vertex_semantic_review_decision_audit_export import (
    CSV_COLUMNS,
    DEFAULT_DECISION_STORE_PATH,
    PROVIDER_MODEL,
    PROVIDER_ROUTE,
    build_audit_export,
    load_decision_records,
)

EXPECTED_FAMILIES = {
    "portal_result_cards",
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
}


def test_loads_records_read_only_without_mutation() -> None:
    before = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    records = load_decision_records()
    _ = build_audit_export()
    after = hashlib.sha256(DEFAULT_DECISION_STORE_PATH.read_bytes()).hexdigest()
    assert before == after  # export did not mutate the 15R store
    assert len(records) >= 1


def test_json_and_csv_export_counts_match() -> None:
    e = build_audit_export()
    s = e.summary
    assert s["decision_records_loaded"] == len(e.json_export)
    rows = list(csv.DictReader(io.StringIO(e.csv_export)))
    assert len(rows) == s["decision_records_loaded"]
    assert s["decision_records_exported_csv"] == len(rows)
    # csv header matches the declared columns
    assert e.csv_export.splitlines()[0] == ",".join(CSV_COLUMNS)


def test_family_and_action_breakdowns_present() -> None:
    s = build_audit_export().summary
    assert set(s["package_family_breakdown"].keys()) == EXPECTED_FAMILIES
    assert s["package_family_breakdown_present"] is True
    assert s["action_breakdown_present"] is True
    assert s["accepted_for_review_count"] >= 1
    assert s["rejected_count"] >= 1
    assert s["deferred_count"] >= 1
    assert sum(s["package_family_breakdown"].values()) == s["decision_records_loaded"]


def test_provenance_and_evidence_visible() -> None:
    e = build_audit_export()
    s = e.summary
    n = s["decision_records_loaded"]
    assert s["provider_provenance_visible_count"] == n
    assert s["evidence_anchor_visible_count"] == n
    assert s["uncertainty_flags_visible_count"] == n
    assert s["source_report_reference_visible_count"] == n
    assert s["audit_reason_visible_count"] == n
    for r in e.json_export:
        assert r["provider_route"] == PROVIDER_ROUTE
        assert r["provider_model"] == PROVIDER_MODEL


def test_read_only_and_no_active_write_indicators() -> None:
    s = build_audit_export().summary
    assert s["export_read_only"] is True
    assert s["active_mkb_record_created_count"] == 0
    assert s["active_written_count"] == 0
    assert s["auto_accept_true_count"] == 0
    assert s["review_required_true_count"] == s["decision_records_loaded"]
    assert s["hallucinated_field_count"] == 0
    assert s["all_records_review_bound"] is True
    assert s["live_call_made"] is False
    assert s["external_api_used"] is False
    assert s["billing_check_pending"] is True


def test_export_does_not_change_decision_status() -> None:
    records = load_decision_records()
    e = build_audit_export()
    # statuses in the export are identical to the loaded store (no mutation)
    assert [r["decision_status"] for r in records] == [r["decision_status"] for r in e.json_export]
    assert {r["decision_status"] for r in e.json_export} <= {"review_draft", "rejected", "deferred"}


def test_markdown_summary_shows_audit_content() -> None:
    md = build_audit_export().audit_summary_markdown
    assert "Operator Audit Summary" in md
    assert "Per-family decision counts" in md
    assert "Per-action decision counts" in md
    assert "gemini-2.5-flash-lite" in md
    assert "vertex" in md


def test_no_credentials_or_private_markers_in_exports() -> None:
    e = build_audit_export()
    blob = json.dumps(e.summary, default=str) + json.dumps(e.json_export, default=str) + e.csv_export + e.audit_summary_markdown + e.audit_matrix_markdown
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_missing_store_returns_empty_safely(tmp_path) -> None:
    records = load_decision_records(tmp_path / "does_not_exist.jsonl")
    assert records == []


def test_no_live_gates_or_provider_calls_in_source() -> None:
    import inspect
    import execution.vertex_semantic_review_decision_audit_export as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        assert marker not in source
