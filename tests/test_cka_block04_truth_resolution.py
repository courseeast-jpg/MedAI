"""Tests for CKA-B04 Truth Resolution + Quarantine Engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from clinical_knowledge.models import (
    KnowledgeTier, MKBRecord, RecordStatus, SourceType, TrustLevel,
)
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
from clinical_knowledge.store import MKBStore
from clinical_knowledge.truth_resolution.conflict_detection import detect_conflict
from clinical_knowledge.truth_resolution.engine import apply_truth_resolution, resolve_conflict
from clinical_knowledge.truth_resolution.integration import (
    active_context_is_clean,
    check_candidate_before_insert,
)
from clinical_knowledge.truth_resolution.models import (
    ConflictPair,
    ConflictType,
    ResolutionAction,
    ResolutionRule,
    TruthResolutionResult,
)
from clinical_knowledge.truth_resolution.rules import (
    ORDERED_RULES,
    rule_clinical_supremacy,
    rule_medication_dose_conflict,
    rule_peer_review_beats_ai,
    rule_recency_same_trust,
    rule_source_agreement,
    rule_unresolvable,
    rule_value_range_merge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _rec(
    fact_type="lab_value",
    entity_text="Hemoglobin A1c < 5.7%",
    specialty="general",
    trust=TrustLevel.UNVERIFIED,
    tier=KnowledgeTier.ACTIVE,
    status=RecordStatus.CONFIRMED,
    source_type=SourceType.SYNTHETIC,
    structured=None,
    created_at=None,
) -> MKBRecord:
    rid = new_record_id()
    return MKBRecord(
        record_id=rid,
        safe_record_id=make_safe_record_id(rid),
        session_id="test_b04",
        fact_type=fact_type,
        entity_text=entity_text,
        specialty=specialty,
        trust_level=trust,
        tier=tier,
        status=status,
        source_type=source_type,
        structured=structured or {},
        confidence=0.80,
        created_at=created_at or _ts(0),
    )


def _conflict_pair(
    cand, exist, conflict_type=ConflictType.VALUE_CONFLICT
) -> ConflictPair:
    return ConflictPair(
        candidate_fact=cand,
        existing_fact=exist,
        conflict_type=conflict_type,
        detected_reasons=["test"],
        safe_public_summary={},
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_conflict_type_values(self):
        assert ConflictType.VALUE_CONFLICT == "value_conflict"
        assert ConflictType.MEDICATION_DOSE_CONFLICT == "medication_dose_conflict"
        assert ConflictType.STATUS_CONFLICT == "status_conflict"
        assert ConflictType.DATE_CONFLICT == "date_conflict"
        assert ConflictType.SOURCE_CONFLICT == "source_conflict"
        assert ConflictType.UNKNOWN_CONFLICT == "unknown_conflict"

    def test_resolution_action_values(self):
        assert ResolutionAction.KEEP_EXISTING == "keep_existing"
        assert ResolutionAction.REPLACE_WITH_NEW == "replace_with_new"
        assert ResolutionAction.MERGE == "merge"
        assert ResolutionAction.QUARANTINE == "quarantine"

    def test_resolution_rule_values(self):
        assert ResolutionRule.CLINICAL_SUPREMACY == "clinical_supremacy"
        assert ResolutionRule.PEER_REVIEW_BEATS_AI == "peer_review_beats_ai"
        assert ResolutionRule.RECENCY_SAME_TRUST == "recency_same_trust"
        assert ResolutionRule.SOURCE_AGREEMENT == "source_agreement"
        assert ResolutionRule.VALUE_RANGE_MERGE == "value_range_merge"
        assert ResolutionRule.MEDICATION_DOSE_CONFLICT == "medication_dose_conflict"
        assert ResolutionRule.UNRESOLVABLE == "unresolvable"

    def test_ordered_rules_length(self):
        assert len(ORDERED_RULES) == 6  # rule_unresolvable is the fallback, not in list


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_value_conflict_detected(self):
        cand = _rec(structured={"value": 5.9})
        exist = _rec(structured={"value": 5.5})
        pair = detect_conflict(cand, exist)
        assert pair is not None
        assert pair.conflict_type == ConflictType.VALUE_CONFLICT

    def test_medication_dose_conflict_detected(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = detect_conflict(cand, exist)
        assert pair is not None
        assert pair.conflict_type == ConflictType.MEDICATION_DOSE_CONFLICT

    def test_status_conflict_detected(self):
        cand = _rec(structured={"status_value": "abnormal"})
        exist = _rec(structured={"status_value": "normal"})
        pair = detect_conflict(cand, exist)
        assert pair is not None
        assert pair.conflict_type == ConflictType.STATUS_CONFLICT

    def test_no_conflict_different_fact_types(self):
        cand = _rec(fact_type="lab_value", structured={"value": 5.9})
        exist = _rec(fact_type="medication_reference", structured={"value": 5.5})
        pair = detect_conflict(cand, exist)
        assert pair is None

    def test_no_conflict_different_entities(self):
        cand = _rec(entity_text="Hemoglobin A1c < 5.7%", structured={"value": 5.9})
        exist = _rec(entity_text="Blood Glucose fasting < 100mg/dL", structured={"value": 99})
        pair = detect_conflict(cand, exist)
        assert pair is None

    def test_source_conflict_detected(self):
        cand = _rec(source_type=SourceType.STUB_CONNECTOR, structured={"value": 5.9})
        exist = _rec(source_type=SourceType.OPERATOR_MANUAL, structured={"value": 5.5})
        pair = detect_conflict(cand, exist)
        assert pair is not None

    def test_conflict_pair_has_safe_public_summary(self):
        cand = _rec(structured={"value": 5.9})
        exist = _rec(structured={"value": 5.5})
        pair = detect_conflict(cand, exist)
        assert pair is not None
        assert "candidate_safe_id" in pair.safe_public_summary
        assert "existing_safe_id" in pair.safe_public_summary

    def test_safe_public_summary_no_raw_source_ref(self):
        cand = _rec(structured={"value": 5.9})
        exist = _rec(structured={"value": 5.5})
        pair = detect_conflict(cand, exist)
        assert pair is not None
        summary_str = str(pair.safe_public_summary)
        assert "source_ref" not in summary_str.lower() or "source_ref" not in pair.safe_public_summary


# ---------------------------------------------------------------------------
# Rule tests
# ---------------------------------------------------------------------------


class TestRuleClinicalSupremacy:
    def test_trust1_candidate_wins(self):
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.9})
        exist = _rec(trust=TrustLevel.OPERATOR_REVIEWED, structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = rule_clinical_supremacy(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.CLINICAL_SUPREMACY
        assert result.resolution == ResolutionAction.REPLACE_WITH_NEW
        assert result.confidence == 0.95
        assert not result.requires_review
        assert exist.record_id in result.superseded_record_ids

    def test_trust1_existing_wins(self):
        cand = _rec(trust=TrustLevel.MODEL_SUGGESTED, structured={"value": 5.9})
        exist = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = rule_clinical_supremacy(pair)
        assert result is not None
        assert result.resolution == ResolutionAction.KEEP_EXISTING
        assert cand.record_id in result.superseded_record_ids

    def test_same_trust1_returns_none(self):
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.9})
        exist = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = rule_clinical_supremacy(pair)
        assert result is None

    def test_trust2_vs_trust4_returns_none(self):
        cand = _rec(trust=TrustLevel.PEER_REVIEWED)
        exist = _rec(trust=TrustLevel.MODEL_SUGGESTED)
        pair = _conflict_pair(cand, exist)
        result = rule_clinical_supremacy(pair)
        assert result is None


class TestRulePeerReviewBeatsAI:
    def test_trust2_candidate_beats_trust3(self):
        cand = _rec(trust=TrustLevel.PEER_REVIEWED, structured={"value": 5.9})
        exist = _rec(trust=TrustLevel.OPERATOR_REVIEWED, structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = rule_peer_review_beats_ai(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.PEER_REVIEW_BEATS_AI
        assert result.confidence == 0.90

    def test_trust2_existing_beats_trust4_candidate(self):
        cand = _rec(trust=TrustLevel.MODEL_SUGGESTED)
        exist = _rec(trust=TrustLevel.PEER_REVIEWED)
        pair = _conflict_pair(cand, exist)
        result = rule_peer_review_beats_ai(pair)
        assert result is not None
        assert result.resolution == ResolutionAction.KEEP_EXISTING

    def test_trust1_vs_trust2_returns_none(self):
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED)
        exist = _rec(trust=TrustLevel.PEER_REVIEWED)
        pair = _conflict_pair(cand, exist)
        result = rule_peer_review_beats_ai(pair)
        assert result is None


class TestRuleRecencySameTrust:
    def test_newer_candidate_wins(self):
        cand = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.9}, created_at=_ts(0))
        exist = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.5}, created_at=_ts(120))
        pair = _conflict_pair(cand, exist)
        result = rule_recency_same_trust(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.RECENCY_SAME_TRUST
        assert result.resolution == ResolutionAction.REPLACE_WITH_NEW
        assert result.confidence == 0.80
        assert "supersede" in result.explanation.lower()

    def test_newer_existing_wins(self):
        cand = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.9}, created_at=_ts(120))
        exist = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.5}, created_at=_ts(0))
        pair = _conflict_pair(cand, exist)
        result = rule_recency_same_trust(pair)
        assert result is not None
        assert result.resolution == ResolutionAction.KEEP_EXISTING

    def test_same_day_returns_none(self):
        ts = _ts(0)
        cand = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.9}, created_at=ts)
        exist = _rec(trust=TrustLevel.UNVERIFIED, structured={"value": 5.5}, created_at=ts)
        pair = _conflict_pair(cand, exist)
        result = rule_recency_same_trust(pair)
        assert result is None

    def test_different_trust_returns_none(self):
        cand = _rec(trust=TrustLevel.PEER_REVIEWED, created_at=_ts(0))
        exist = _rec(trust=TrustLevel.UNVERIFIED, created_at=_ts(120))
        pair = _conflict_pair(cand, exist)
        result = rule_recency_same_trust(pair)
        assert result is None


class TestRuleSourceAgreement:
    def test_candidate_more_sources_wins(self):
        cand = _rec(structured={"value": 5.9, "source_count": 3})
        exist = _rec(structured={"value": 5.5, "source_count": 1})
        pair = _conflict_pair(cand, exist)
        result = rule_source_agreement(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.SOURCE_AGREEMENT
        assert result.resolution == ResolutionAction.REPLACE_WITH_NEW
        assert result.confidence == 0.75

    def test_candidate_less_sources_returns_none(self):
        cand = _rec(structured={"value": 5.9, "source_count": 1})
        exist = _rec(structured={"value": 5.5, "source_count": 2})
        pair = _conflict_pair(cand, exist)
        result = rule_source_agreement(pair)
        assert result is None

    def test_candidate_source_count_below_2_returns_none(self):
        cand = _rec(structured={"value": 5.9, "source_count": 1})
        exist = _rec(structured={"value": 5.5, "source_count": 1})
        pair = _conflict_pair(cand, exist)
        result = rule_source_agreement(pair)
        assert result is None


class TestRuleValueRangeMerge:
    def test_numeric_merge_same_day(self):
        ts = _ts(0)
        cand = _rec(structured={"value": 5.9}, created_at=ts)
        exist = _rec(structured={"value": 5.5}, created_at=ts)
        pair = _conflict_pair(cand, exist)
        result = rule_value_range_merge(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.VALUE_RANGE_MERGE
        assert result.resolution == ResolutionAction.MERGE
        assert result.merged_record is not None
        assert result.confidence == 0.70
        merged_struct = result.merged_record.structured
        assert "value_range" in merged_struct
        assert merged_struct["value_range"]["low"] == 5.5
        assert merged_struct["value_range"]["high"] == 5.9

    def test_merged_record_no_raw_source_ref(self):
        ts = _ts(0)
        cand = _rec(structured={"value": 5.9}, created_at=ts)
        exist = _rec(structured={"value": 5.5}, created_at=ts)
        pair = _conflict_pair(cand, exist)
        result = rule_value_range_merge(pair)
        assert result is not None
        assert "source_ref" not in result.merged_record.structured

    def test_non_numeric_returns_none(self):
        ts = _ts(0)
        cand = _rec(structured={"value": "high"}, created_at=ts)
        exist = _rec(structured={"value": "low"}, created_at=ts)
        pair = _conflict_pair(cand, exist)
        result = rule_value_range_merge(pair)
        assert result is None

    def test_same_week_threshold(self):
        cand = _rec(structured={"value": 5.9}, created_at=_ts(3))
        exist = _rec(structured={"value": 5.5}, created_at=_ts(0))
        pair = _conflict_pair(cand, exist)
        result = rule_value_range_merge(pair)
        assert result is not None  # within 7 days

    def test_old_records_returns_none(self):
        cand = _rec(structured={"value": 5.9}, created_at=_ts(30))
        exist = _rec(structured={"value": 5.5}, created_at=_ts(0))
        pair = _conflict_pair(cand, exist)
        result = rule_value_range_merge(pair)
        assert result is None  # >7 days apart


class TestRuleMedicationDoseConflict:
    def test_medication_dose_quarantines_both(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        assert result is not None
        assert result.rule_applied == ResolutionRule.MEDICATION_DOSE_CONFLICT
        assert result.resolution == ResolutionAction.QUARANTINE
        assert result.requires_review is True
        assert result.confidence == 0.0
        assert len(result.quarantined_record_ids) == 2
        assert cand.record_id in result.quarantined_record_ids
        assert exist.record_id in result.quarantined_record_ids

    def test_medication_dose_explanation_no_dose_advice(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        assert result is not None
        expl_lower = result.explanation.lower()
        assert "clinician" in expl_lower
        assert "ddi" not in expl_lower
        assert "prescribe" not in expl_lower
        assert "recommend" not in expl_lower

    def test_non_medication_conflict_returns_none(self):
        cand = _rec(structured={"value": 5.9})
        exist = _rec(structured={"value": 5.5})
        pair = _conflict_pair(cand, exist, ConflictType.VALUE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        assert result is None


class TestRuleUnresolvable:
    def test_unresolvable_quarantines_candidate(self):
        cand = _rec(structured={"value": 5.9})
        exist = _rec(structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = rule_unresolvable(pair)
        assert result.rule_applied == ResolutionRule.UNRESOLVABLE
        assert result.resolution == ResolutionAction.QUARANTINE
        assert result.requires_review is True
        assert cand.record_id in result.quarantined_record_ids
        assert exist.record_id not in result.quarantined_record_ids

    def test_unresolvable_confidence_zero(self):
        cand = _rec()
        exist = _rec()
        pair = _conflict_pair(cand, exist)
        result = rule_unresolvable(pair)
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# First matching rule wins
# ---------------------------------------------------------------------------


class TestOrderedRules:
    def test_clinical_supremacy_fires_first(self):
        # trust=1 vs trust=4 → clinical_supremacy should fire before peer_review
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.9})
        exist = _rec(trust=TrustLevel.MODEL_SUGGESTED, structured={"value": 5.5})
        pair = _conflict_pair(cand, exist)
        result = resolve_conflict(pair)
        assert result.rule_applied == ResolutionRule.CLINICAL_SUPREMACY

    def test_peer_review_fires_before_recency(self):
        # trust=2 vs trust=3 → peer_review fires, not recency
        cand = _rec(trust=TrustLevel.PEER_REVIEWED, structured={"value": 5.9}, created_at=_ts(0))
        exist = _rec(trust=TrustLevel.OPERATOR_REVIEWED, structured={"value": 5.5}, created_at=_ts(120))
        pair = _conflict_pair(cand, exist)
        result = resolve_conflict(pair)
        assert result.rule_applied == ResolutionRule.PEER_REVIEW_BEATS_AI

    def test_unresolvable_fires_as_fallback(self):
        # same trust, same day, no source_count, no numeric values
        ts = _ts(0)
        cand = _rec(trust=TrustLevel.UNVERIFIED, structured={"status_value": "x"}, created_at=ts)
        exist = _rec(trust=TrustLevel.UNVERIFIED, structured={"status_value": "y"}, created_at=ts)
        pair = _conflict_pair(cand, exist, ConflictType.STATUS_CONFLICT)
        result = resolve_conflict(pair)
        assert result.rule_applied == ResolutionRule.UNRESOLVABLE


# ---------------------------------------------------------------------------
# Quarantine behavior
# ---------------------------------------------------------------------------


class TestQuarantineBehavior:
    def test_quarantine_sets_requires_review(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        assert result.requires_review is True

    def test_quarantined_records_excluded_from_active(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
            tier=KnowledgeTier.ACTIVE,
        )
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        active_ids = {r["record_id"] for r in store.list_active()}
        assert exist.record_id not in active_ids

    def test_quarantined_records_retrievable_separately(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
            tier=KnowledgeTier.ACTIVE,
        )
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        quarantined = store.list_quarantined()
        quarantined_ids = {r["record_id"] for r in quarantined}
        assert exist.record_id in quarantined_ids

    def test_superseded_excluded_from_active(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(trust=TrustLevel.OPERATOR_REVIEWED, structured={"value": 5.5})
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.9})
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        active_ids = {r["record_id"] for r in store.list_active()}
        superseded_ids = {r["record_id"] for r in store.list_superseded()}
        assert exist.record_id in superseded_ids
        assert exist.record_id not in active_ids


# ---------------------------------------------------------------------------
# Ledger event tests
# ---------------------------------------------------------------------------


class TestLedgerEvents:
    def test_truth_resolution_event_written(self):
        from clinical_knowledge.models import LedgerEventType
        store = MKBStore(db_path=":memory:")
        exist = _rec(trust=TrustLevel.OPERATOR_REVIEWED, structured={"value": 5.5})
        cand = _rec(trust=TrustLevel.EXPERT_VALIDATED, structured={"value": 5.9})
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.TRUTH_RESOLUTION.value in types

    def test_quarantine_event_written(self):
        from clinical_knowledge.models import LedgerEventType
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
            tier=KnowledgeTier.ACTIVE,
        )
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.QUARANTINE.value in types

    def test_ledger_events_now_not_reserved(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES, LedgerEventType
        assert LedgerEventType.TRUTH_RESOLUTION not in _RESERVED_EVENT_TYPES
        assert LedgerEventType.QUARANTINE not in _RESERVED_EVENT_TYPES

    def test_truth_resolution_ledger_event_helper(self):
        from clinical_knowledge.ledger import make_truth_resolution_event
        from clinical_knowledge.models import LedgerEventType
        evt = make_truth_resolution_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            rule_applied="clinical_supremacy",
            resolution="replace_with_new",
            winner_safe_id="safe_win",
            loser_safe_id="safe_lose",
            confidence=0.95,
            requires_review=False,
            explanation="test explanation",
        )
        assert evt.event_type == LedgerEventType.TRUTH_RESOLUTION
        assert evt.safe_public_details["rule_applied"] == "clinical_supremacy"
        assert evt.safe_public_details["confidence"] == 0.95

    def test_quarantine_ledger_event_helper(self):
        from clinical_knowledge.ledger import make_quarantine_event
        from clinical_knowledge.models import LedgerEventType
        evt = make_quarantine_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            quarantined_safe_ids=["safe_a", "safe_b"],
            conflict_type="medication_dose_conflict",
            explanation="test quarantine",
        )
        assert evt.event_type == LedgerEventType.QUARANTINE
        assert evt.safe_public_details["requires_review"] is True
        assert "safe_a" in evt.safe_public_details["quarantined_safe_ids"]

    def test_no_raw_source_ref_in_ledger_event(self):
        from clinical_knowledge.ledger import make_truth_resolution_event
        evt = make_truth_resolution_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            rule_applied="clinical_supremacy",
            resolution="replace_with_new",
            winner_safe_id="safe_win",
            loser_safe_id="safe_lose",
            confidence=0.95,
            requires_review=False,
            explanation="resolution explanation",
        )
        details_str = str(evt.safe_public_details)
        assert "source_ref" not in details_str
        assert "C:\\" not in details_str
        assert "/home/" not in details_str


# ---------------------------------------------------------------------------
# Safety / no clinical recommendation tests
# ---------------------------------------------------------------------------


class TestSafetyBoundaries:
    def test_no_clinical_recommendation_in_explanation(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        expl = result.explanation.lower()
        assert "recommend" not in expl
        assert "should take" not in expl
        assert "prescribe" not in expl

    def test_no_prescription_dosing_advice(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        expl = result.explanation.lower()
        assert "mg" not in expl
        assert "1000" not in expl
        assert "500" not in expl

    def test_all_ledger_types_active_in_b06(self):
        # B05 activated DDI_BLOCK/DDI_WARNING; B06 activates ENRICHMENT_WRITE/HYPOTHESIS_PROMOTED
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES, LedgerEventType
        assert LedgerEventType.DDI_BLOCK not in _RESERVED_EVENT_TYPES
        assert LedgerEventType.DDI_WARNING not in _RESERVED_EVENT_TYPES
        assert LedgerEventType.ENRICHMENT_WRITE not in _RESERVED_EVENT_TYPES
        assert len(_RESERVED_EVENT_TYPES) == 0

    def test_public_summary_no_raw_refs(self):
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
        )
        pair = _conflict_pair(cand, exist, ConflictType.MEDICATION_DOSE_CONFLICT)
        result = rule_medication_dose_conflict(pair)
        summary_str = str(result.safe_public_summary)
        assert "source_ref" not in summary_str


# ---------------------------------------------------------------------------
# Integration helpers tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_active_context_clean_after_quarantine(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 500mg twice daily",
            tier=KnowledgeTier.ACTIVE,
        )
        cand = _rec(
            fact_type="medication_antiepileptic",
            entity_text="Levetiracetam 1000mg once daily",
        )
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)
        assert active_context_is_clean(store)

    def test_check_candidate_before_insert_finds_conflict(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            trust=TrustLevel.OPERATOR_REVIEWED,
            structured={"value": 5.5},
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
        )
        store.insert_record(exist)
        cand = _rec(
            trust=TrustLevel.UNVERIFIED,
            structured={"value": 5.9},
        )
        result = check_candidate_before_insert(cand, store)
        assert result is not None
        assert result["conflict_detected"] is True

    def test_check_candidate_no_conflict_returns_none(self):
        store = MKBStore(db_path=":memory:")
        exist = _rec(
            fact_type="different_fact",
            entity_text="completely different entity xyz",
            structured={"value": 5.5},
        )
        store.insert_record(exist)
        cand = _rec(structured={"value": 5.9})
        result = check_candidate_before_insert(cand, store)
        assert result is None


# ---------------------------------------------------------------------------
# Validation script smoke test
# ---------------------------------------------------------------------------


class TestValidationScript:
    def test_validation_script_succeeds(self):
        import importlib.util
        import sys
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "run_cka_block04_truth_resolution_validation.py"
        spec = importlib.util.spec_from_file_location("val_b04", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            report = mod.run_validation(report_dir=Path(tmpdir))
        assert report["all_cases_passed"] is True
        assert report["conclusion"] == "cka_b04_truth_resolution_ready"

    def test_final_report_no_private_strings(self):
        import tempfile
        import importlib.util
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "run_cka_block04_truth_resolution_validation.py"
        spec = importlib.util.spec_from_file_location("val_b04_priv", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = mod.run_validation(report_dir=Path(tmpdir))

        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        check = check_public_report_payload(report)
        assert check.passed
        assert not check.raw_phi_logged_in_public_reports
        assert check.private_filename_path_leaks == 0
