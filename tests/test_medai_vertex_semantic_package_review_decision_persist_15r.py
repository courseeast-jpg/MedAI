"""No-live tests for 15R Vertex semantic review-decision persistence."""
from __future__ import annotations

import json
from pathlib import Path

from app.vertex_semantic_review_surface import build_vertex_semantic_review_drafts
from execution.vertex_semantic_review_decision_store import (
    DETERMINISTIC_TIMESTAMP,
    VertexSemanticReviewDecisionStore,
    operator_status_message,
    persist_decisions_for_all_families,
)

EXPECTED_FAMILIES = {
    "portal_result_cards",
    "cytology_pathology_narrative",
    "urinalysis_table_like_lab",
    "mixed_narrative_numeric_result",
}
REQUIRED_FIELDS = {
    "decision_id", "package_id", "package_family", "finding_id", "candidate_fact_id",
    "action", "decision_status", "provider_route", "provider_model", "source_report_reference",
    "source_evidence_text", "evidence_anchor", "unknown_values", "uncertainty_flags",
    "hallucinated_field_count", "review_required", "auto_accept", "creates_active_mkb_record",
    "active_written_count_delta", "timestamp", "audit_reason", "operator_id", "local_only",
}


def _report(tmp_path: Path):
    return persist_decisions_for_all_families(store_path=tmp_path / "decision_store.jsonl")


def _first_draft_finding():
    draft = build_vertex_semantic_review_drafts()[0]
    return draft, draft.vertex_semantic_findings[0]


def test_accept_persists_review_bound_no_active_write(tmp_path) -> None:
    store = VertexSemanticReviewDecisionStore(store_path=tmp_path / "s.jsonl")
    draft, finding = _first_draft_finding()
    rec = store.record_decision(draft, finding, 0, "accept")
    assert rec is not None
    assert rec.action == "accept_for_review"
    assert rec.decision_status == "review_draft"
    assert rec.review_required is True
    assert rec.auto_accept is False
    assert rec.creates_active_mkb_record is False
    assert rec.active_written_count_delta == 0
    assert rec.evidence_anchor
    assert rec.provider_route == "vertex"
    assert rec.provider_model == "gemini-2.5-flash-lite"
    assert operator_status_message(rec) == "accepted for review queue only"


def test_reject_and_defer_persist_no_active_write(tmp_path) -> None:
    store = VertexSemanticReviewDecisionStore(store_path=tmp_path / "s.jsonl")
    draft, finding = _first_draft_finding()
    rej = store.record_decision(draft, finding, 0, "reject")
    dfr = store.record_decision(draft, finding, 1, "defer")
    assert rej.decision_status == "rejected"
    assert dfr.decision_status == "deferred"
    assert operator_status_message(rej) == "rejected - no active write"
    assert operator_status_message(dfr) == "deferred - no active write"
    for rec in (rej, dfr):
        assert rec.creates_active_mkb_record is False
        assert rec.active_written_count_delta == 0
        assert rec.auto_accept is False
        assert rec.audit_reason.strip()
        assert rec.local_only is True


def test_invalid_action_writes_no_record(tmp_path) -> None:
    store = VertexSemanticReviewDecisionStore(store_path=tmp_path / "s.jsonl")
    draft, finding = _first_draft_finding()
    result = store.record_decision(draft, finding, 0, "promote_to_active_mkb")
    assert result is None
    assert store.records() == []
    assert store.active_mkb_record_created_count() == 0
    assert store.active_written_count() == 0


def test_all_required_fields_present(tmp_path) -> None:
    report = _report(tmp_path)
    for rec in report["decision_records"]:
        assert REQUIRED_FIELDS.issubset(set(rec.keys()))


def test_all_families_and_actions_persisted(tmp_path) -> None:
    s = _report(tmp_path)["summary"]
    assert s["package_families_loaded"] == 4
    assert s["package_families_with_persisted_decisions"] == 4
    assert s["all_actions_exercised"] is True
    assert s["accept_decision_records_created"] >= 1
    assert s["reject_decision_records_created"] >= 1
    assert s["defer_decision_records_created"] >= 1
    assert s["invalid_action_rejected_count"] >= 1


def test_no_active_writes_or_auto_accept(tmp_path) -> None:
    s = _report(tmp_path)["summary"]
    assert s["active_mkb_record_created_count"] == 0
    assert s["active_written_count"] == 0
    assert s["auto_accept_true_count"] == 0
    assert s["review_required_true_count"] == s["decision_records_created"]
    assert s["review_bound_decision_count"] == s["decision_records_created"]
    assert s["local_only_decision_count"] == s["decision_records_created"]


def test_provenance_and_evidence_preserved(tmp_path) -> None:
    s = _report(tmp_path)["summary"]
    n = s["decision_records_created"]
    assert s["evidence_anchor_preserved_count"] == n
    assert s["provider_provenance_preserved_count"] == n
    assert s["uncertainty_flags_preserved_count"] == n
    assert s["audit_reason_present_count"] == n
    assert s["source_report_reference_present_count"] == n
    assert s["hallucinated_field_count"] == 0


def test_deterministic_timestamp_and_ids(tmp_path) -> None:
    a = _report(tmp_path / "a")
    b = _report(tmp_path / "b")
    assert [r["timestamp"] for r in a["decision_records"]] == [DETERMINISTIC_TIMESTAMP] * len(a["decision_records"])
    assert [r["decision_id"] for r in a["decision_records"]] == [r["decision_id"] for r in b["decision_records"]]


def test_store_persists_jsonl(tmp_path) -> None:
    path = tmp_path / "decision_store.jsonl"
    persist_decisions_for_all_families(store_path=path)
    assert path.exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    for ln in lines:
        rec = json.loads(ln)
        assert rec["creates_active_mkb_record"] is False
        assert rec["auto_accept"] is False
        assert rec["review_required"] is True


def test_no_credentials_or_private_markers(tmp_path) -> None:
    report = _report(tmp_path)
    blob = json.dumps(report, default=str)
    for token in ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "DOB", "MRN", "Accession"):
        assert token not in blob


def test_no_live_gates_in_source() -> None:
    import inspect
    import execution.vertex_semantic_review_decision_store as mod

    source = inspect.getsource(mod)
    for gate in (
        "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_LIVE_COMPARE_ALLOWED",
        "MEDAI_VERTEX_SEMANTIC_REMAINING_FAMILIES_LIVE_ALLOWED",
    ):
        assert gate not in source
    assert s_no_provider_call(source)


def s_no_provider_call(source: str) -> bool:
    for marker in ("requests.", "urllib.request", "httpx.", "generate_content", "acquire_google_cloud_access_token"):
        if marker in source:
            return False
    return True
