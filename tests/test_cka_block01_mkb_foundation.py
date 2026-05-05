"""Tests for CKA-B01 — MKB Foundation, Feature Flags, Ledger, Store.

Covers:
- default feature flags
- trust level → tier default mapping
- explicit tier override
- record creation validation
- ledger append on create/update
- active retrieval excludes quarantined/superseded
- hypothesis retrieval works
- safe ID hashing does not expose raw references
- public report privacy self-check
- validation script exits successfully
- no external API call required
- no real PHI required
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.ledger import (
    make_created_event,
    make_status_changed_event,
    make_tier_changed_event,
    make_updated_event,
)
from clinical_knowledge.models import (
    KnowledgeTier,
    LedgerEvent,
    LedgerEventType,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
    _default_tier_for_trust,
)
from clinical_knowledge.safe_ids import (
    hash_source_ref,
    make_safe_record_id,
    new_record_id,
)
from clinical_knowledge.store import (
    ENCRYPTION_BOUNDARY_READY,
    SQLCIPHER_ENCRYPTION_ACTIVE,
    MKBStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    trust_level: TrustLevel = TrustLevel.UNVERIFIED,
    tier: KnowledgeTier | None = None,
    status: RecordStatus = RecordStatus.PENDING,
    confidence: float = 0.5,
    fact_type: str = "test_fact",
) -> MKBRecord:
    rec_id = new_record_id()
    return MKBRecord(
        record_id=rec_id,
        safe_record_id=make_safe_record_id(rec_id),
        session_id="test_session",
        fact_type=fact_type,
        entity_text="Synthetic test entity",
        source_type=SourceType.SYNTHETIC,
        source_ref=hash_source_ref("synthetic_test_source"),
        trust_level=trust_level,
        tier=tier,
        status=status,
        confidence=confidence,
    )


def _make_store() -> MKBStore:
    return MKBStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# Milestone 1: branch / namespace guard
# ---------------------------------------------------------------------------


def test_cka_package_exists():
    import clinical_knowledge
    assert clinical_knowledge.__roadmap__ == "CKA"
    assert "3c0c869" in clinical_knowledge.__frozen_hitl_release__


# ---------------------------------------------------------------------------
# Milestone 3: Feature flags
# ---------------------------------------------------------------------------


def test_default_feature_flags():
    cfg = CKAConfig()
    assert cfg.ENABLE_GRAPH is False
    assert cfg.ENABLE_LOCAL_LLM is False
    assert cfg.ENRICH_PROMOTE is False
    assert cfg.ENABLE_WEB_INGESTION is False
    assert cfg.ENABLE_EPUB is False
    assert cfg.ENABLE_YOUTUBE is False
    assert cfg.MEDAI_LOCAL_ONLY is True
    assert cfg.EXTERNAL_APIS_ENABLED is False
    assert cfg.SAFE_MODE_THRESHOLD == pytest.approx(0.4)
    assert "dxgpt_stub" in cfg.ACTIVE_CONNECTORS


def test_feature_flags_as_dict():
    cfg = CKAConfig()
    d = cfg.as_dict()
    assert d["ENABLE_GRAPH"] is False
    assert d["MEDAI_LOCAL_ONLY"] is True
    assert isinstance(d["ACTIVE_CONNECTORS"], list)


def test_feature_flags_no_external_api():
    cfg = CKAConfig()
    assert cfg.EXTERNAL_APIS_ENABLED is False


# ---------------------------------------------------------------------------
# Milestone 4: Trust level → tier mapping
# ---------------------------------------------------------------------------


def test_trust1_defaults_to_active():
    assert _default_tier_for_trust(TrustLevel.EXPERT_VALIDATED) == KnowledgeTier.ACTIVE


def test_trust2_defaults_to_active():
    assert _default_tier_for_trust(TrustLevel.PEER_REVIEWED) == KnowledgeTier.ACTIVE


def test_trust3_defaults_to_hypothesis():
    assert _default_tier_for_trust(TrustLevel.OPERATOR_REVIEWED) == KnowledgeTier.HYPOTHESIS


def test_trust4_defaults_to_hypothesis():
    assert _default_tier_for_trust(TrustLevel.MODEL_SUGGESTED) == KnowledgeTier.HYPOTHESIS


def test_trust5_defaults_to_hypothesis():
    assert _default_tier_for_trust(TrustLevel.UNVERIFIED) == KnowledgeTier.HYPOTHESIS


def test_trust1_record_tier_is_active():
    r = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED)
    assert r.tier == KnowledgeTier.ACTIVE


def test_trust5_record_tier_is_hypothesis():
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    assert r.tier == KnowledgeTier.HYPOTHESIS


# ---------------------------------------------------------------------------
# Milestone 4: Explicit tier override
# ---------------------------------------------------------------------------


def test_explicit_tier_overrides_trust():
    # trust_level 1 would default to ACTIVE, but we force QUARANTINED
    r = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED, tier=KnowledgeTier.QUARANTINED)
    assert r.tier == KnowledgeTier.QUARANTINED


def test_quarantined_requires_review():
    r = _make_record(tier=KnowledgeTier.QUARANTINED)
    assert r.requires_review is True


def test_superseded_requires_review():
    r = _make_record(tier=KnowledgeTier.SUPERSEDED)
    assert r.requires_review is True


def test_active_does_not_force_requires_review():
    r = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED)
    assert r.tier == KnowledgeTier.ACTIVE
    assert r.requires_review is False


def test_hypothesis_does_not_force_requires_review():
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    assert r.tier == KnowledgeTier.HYPOTHESIS
    assert r.requires_review is False


# ---------------------------------------------------------------------------
# Milestone 4: Record creation validation
# ---------------------------------------------------------------------------


def test_record_confidence_range_valid():
    r = _make_record(confidence=0.0)
    assert r.confidence == 0.0
    r2 = _make_record(confidence=1.0)
    assert r2.confidence == 1.0


def test_record_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        _make_record(confidence=1.5)
    with pytest.raises(ValueError):
        _make_record(confidence=-0.1)


def test_record_active_retrievable():
    r = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED)
    assert r.is_retrievable_active() is True
    assert r.is_retrievable_hypothesis() is False


def test_record_hypothesis_retrievable():
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    assert r.is_retrievable_active() is False
    assert r.is_retrievable_hypothesis() is True


def test_quarantined_not_active_retrievable():
    r = _make_record(tier=KnowledgeTier.QUARANTINED)
    assert r.is_retrievable_active() is False
    assert r.is_retrievable_hypothesis() is False


def test_superseded_not_active_retrievable():
    r = _make_record(tier=KnowledgeTier.SUPERSEDED)
    assert r.is_retrievable_active() is False


# ---------------------------------------------------------------------------
# Milestone 6: SQLite store
# ---------------------------------------------------------------------------


def test_store_insert_and_fetch():
    store = _make_store()
    r = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED)
    event = make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value)
    store.insert_record(r, ledger_event=event)
    fetched = store.fetch_by_record_id(r.record_id)
    assert fetched is not None
    assert fetched["record_id"] == r.record_id
    assert fetched["tier"] == KnowledgeTier.ACTIVE.value


def test_store_list_active_excludes_quarantined():
    store = _make_store()
    r_active = _make_record(trust_level=TrustLevel.EXPERT_VALIDATED)
    r_quarantined = _make_record(tier=KnowledgeTier.QUARANTINED)
    store.insert_record(r_active, make_created_event(r_active.record_id, r_active.safe_record_id, r_active.tier.value, r_active.trust_level.value))
    store.insert_record(r_quarantined, make_created_event(r_quarantined.record_id, r_quarantined.safe_record_id, r_quarantined.tier.value, r_quarantined.trust_level.value))
    active = store.list_active()
    ids = [r["record_id"] for r in active]
    assert r_active.record_id in ids
    assert r_quarantined.record_id not in ids


def test_store_list_active_excludes_superseded():
    store = _make_store()
    r_active = _make_record(trust_level=TrustLevel.PEER_REVIEWED)
    r_superseded = _make_record(tier=KnowledgeTier.SUPERSEDED)
    store.insert_record(r_active, make_created_event(r_active.record_id, r_active.safe_record_id, r_active.tier.value, r_active.trust_level.value))
    store.insert_record(r_superseded, make_created_event(r_superseded.record_id, r_superseded.safe_record_id, r_superseded.tier.value, r_superseded.trust_level.value))
    active = store.list_active()
    ids = [r["record_id"] for r in active]
    assert r_superseded.record_id not in ids


def test_store_list_hypothesis():
    store = _make_store()
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    hyp = store.list_hypothesis()
    assert any(x["record_id"] == r.record_id for x in hyp)


def test_store_list_quarantined():
    store = _make_store()
    r = _make_record(tier=KnowledgeTier.QUARANTINED)
    store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    q = store.list_quarantined()
    assert any(x["record_id"] == r.record_id for x in q)


def test_store_update_tier():
    store = _make_store()
    from datetime import datetime, timezone
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    now = datetime.now(timezone.utc).isoformat()
    tier_event = make_tier_changed_event(r.record_id, r.safe_record_id, r.tier.value, KnowledgeTier.QUARANTINED.value)
    store.update_record_tier(r.record_id, KnowledgeTier.QUARANTINED, now, ledger_event=tier_event)
    fetched = store.fetch_by_record_id(r.record_id)
    assert fetched["tier"] == KnowledgeTier.QUARANTINED.value
    assert fetched["requires_review"] == 1


def test_store_update_status():
    store = _make_store()
    from datetime import datetime, timezone
    r = _make_record()
    store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    now = datetime.now(timezone.utc).isoformat()
    status_event = make_status_changed_event(r.record_id, r.safe_record_id, r.status.value, RecordStatus.CONFIRMED.value)
    store.update_record_status(r.record_id, RecordStatus.CONFIRMED, now, ledger_event=status_event)
    fetched = store.fetch_by_record_id(r.record_id)
    assert fetched["status"] == RecordStatus.CONFIRMED.value


# ---------------------------------------------------------------------------
# Milestone 7: Ledger
# ---------------------------------------------------------------------------


def test_ledger_event_appended_on_create():
    store = _make_store()
    r = _make_record()
    event = make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value)
    store.insert_record(r, ledger_event=event)
    events = store.read_ledger_events(record_id=r.record_id)
    assert len(events) == 1
    assert events[0]["event_type"] == LedgerEventType.MKB_RECORD_CREATED.value


def test_ledger_event_appended_on_update():
    store = _make_store()
    r = _make_record()
    create_event = make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value)
    store.insert_record(r, ledger_event=create_event)
    update_event = make_updated_event(r.record_id, r.safe_record_id, ["confidence"])
    store.append_ledger_event(update_event)
    events = store.read_ledger_events(record_id=r.record_id)
    assert len(events) == 2
    types = [e["event_type"] for e in events]
    assert LedgerEventType.MKB_RECORD_UPDATED.value in types


def test_ledger_event_appended_on_tier_change():
    store = _make_store()
    from datetime import datetime, timezone
    r = _make_record(trust_level=TrustLevel.UNVERIFIED)
    store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    now = datetime.now(timezone.utc).isoformat()
    tier_event = make_tier_changed_event(r.record_id, r.safe_record_id, r.tier.value, KnowledgeTier.QUARANTINED.value)
    store.update_record_tier(r.record_id, KnowledgeTier.QUARANTINED, now, ledger_event=tier_event)
    events = store.read_ledger_events(record_id=r.record_id)
    types = [e["event_type"] for e in events]
    assert LedgerEventType.TIER_CHANGED.value in types


def test_all_event_types_active_in_b06():
    # CKA-B06 activates ENRICHMENT_WRITE and HYPOTHESIS_PROMOTED.
    # _RESERVED_EVENT_TYPES is now empty — all types may be created.
    from clinical_knowledge.models import _RESERVED_EVENT_TYPES, LedgerEventType
    assert len(_RESERVED_EVENT_TYPES) == 0
    r = _make_record()
    # Should NOT raise any more
    evt = LedgerEvent(
        event_id="x",
        event_type=LedgerEventType.ENRICHMENT_WRITE,
        record_id=r.record_id,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert evt.event_type == LedgerEventType.ENRICHMENT_WRITE


def test_ledger_count():
    store = _make_store()
    for _ in range(5):
        r = _make_record()
        store.insert_record(r, make_created_event(r.record_id, r.safe_record_id, r.tier.value, r.trust_level.value))
    assert store.count_ledger_events() == 5


# ---------------------------------------------------------------------------
# Milestone 5: Safe IDs
# ---------------------------------------------------------------------------


def test_hash_source_ref_is_deterministic():
    raw = "real/path/to/patient_file_123.pdf"
    assert hash_source_ref(raw) == hash_source_ref(raw)


def test_hash_source_ref_does_not_expose_raw():
    raw = "patient_mrn_0001234_results.pdf"
    hashed = hash_source_ref(raw)
    assert raw not in hashed
    assert "patient" not in hashed
    assert "mrn" not in hashed.lower()


def test_safe_record_id_does_not_expose_raw_uuid():
    rec_id = "some-internal-uuid-1234"
    safe = make_safe_record_id(rec_id)
    assert rec_id not in safe
    assert safe.startswith("cka_rec_")


def test_source_ref_in_record_is_hashed():
    raw = "original_patient_report.pdf"
    safe_src = hash_source_ref(raw)
    r = MKBRecord(
        record_id=new_record_id(),
        safe_record_id=make_safe_record_id(new_record_id()),
        session_id="s",
        fact_type="f",
        entity_text="e",
        source_ref=safe_src,
    )
    public = r.to_public_dict()
    assert raw not in json.dumps(public)


def test_public_dict_no_phi():
    r = _make_record()
    public = r.to_public_dict()
    payload = json.dumps(public).lower()
    for forbidden in ("patient", "mrn", "dob", "ssn", "date_of_birth"):
        assert forbidden not in payload


# ---------------------------------------------------------------------------
# Milestone 8: Validation script
# ---------------------------------------------------------------------------


def test_validation_script_runs(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["conclusion"] == "cka_b01_mkb_foundation_ready"


def test_validation_records_inserted(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["records_inserted"] >= 10


def test_validation_ledger_events(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["ledger_events_written"] >= report["records_inserted"]


def test_validation_tier_counts(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["active_records_count"] >= 1
    assert report["hypothesis_records_count"] >= 1
    assert report["quarantined_records_count"] >= 1
    assert report["superseded_records_count"] >= 1


def test_validation_no_phi(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0


def test_validation_no_external_api(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["external_api_used"] is False


def test_validation_safety_flags(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["production_ocr_changed"] is False
    assert report["production_extractor_changed"] is False
    assert report["safety_gate_changed"] is False
    assert report["frozen_hitl_release_reopened"] is False


def test_validation_encryption_flags(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["sqlcipher_encryption_active"] is False
    assert report["encryption_boundary_ready"] is True


def test_validation_feature_flags_present(tmp_path: Path):
    from scripts.run_cka_block01_mkb_foundation_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    flags = report["feature_flags"]
    assert flags["ENABLE_GRAPH"] is False
    assert flags["MEDAI_LOCAL_ONLY"] is True
    assert flags["EXTERNAL_APIS_ENABLED"] is False


def test_store_sqlcipher_flag():
    assert SQLCIPHER_ENCRYPTION_ACTIVE is False
    assert ENCRYPTION_BOUNDARY_READY is True
