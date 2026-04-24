from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from app.schemas import MKBRecord
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from governance.decision_scoring import GovernanceDecisionScoring
from governance.hypothesis_tier import GovernanceHypothesisTier
from governance.truth_resolution import GovernanceTruthResolutionAdapter, GovernanceTruthResolutionEngine


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_record(
    *,
    name: str,
    fact_type: str = "diagnosis",
    trust_level: int = 3,
    source_count: int = 1,
    days_ago: int = 0,
    source_type: str = "document",
    extraction_method: str = "spacy",
    tier: str = "active",
    value=None,
    dose: str | None = None,
) -> MKBRecord:
    structured = {"name": name}
    if value is not None:
        structured["value"] = value
    if dose is not None:
        structured["dose"] = dose
    content = f"{fact_type.title()}: {name}"
    if dose is not None:
        content += f" {dose}"
    return MKBRecord(
        fact_type=fact_type,
        content=content,
        structured=structured,
        specialty="epilepsy",
        source_type=source_type,
        source_name="fixture",
        trust_level=trust_level,
        confidence=0.9,
        source_count=source_count,
        extraction_method=extraction_method,
        tier=tier,
        status="active" if tier == "active" else tier,
        first_recorded=datetime.utcnow() - timedelta(days=days_ago),
        last_confirmed=datetime.utcnow() - timedelta(days=days_ago),
    )


def test_trust_level_1_beats_ai_derived_fact():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="Epilepsy", trust_level=1)
    candidate = make_record(name="Epilepsy", trust_level=3, source_type="ai_response", extraction_method="gemini")

    result = engine.resolve(candidate, existing)

    assert result.action == "keep_existing"
    assert result.winner == existing


def test_peer_reviewed_trust_level_2_beats_ai_derived_fact():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="Migraine", trust_level=2)
    candidate = make_record(name="Migraine", trust_level=3, source_type="ai_response", extraction_method="phi3")

    result = engine.resolve(candidate, existing)

    assert result.action == "keep_existing"
    assert result.reason == "peer_review_beats_ai"


def test_same_trust_newer_fact_wins_when_over_90_days():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="Asthma", trust_level=2, days_ago=180)
    candidate = make_record(name="Asthma", trust_level=2, days_ago=1)

    result = engine.resolve(candidate, existing)

    assert result.action == "replace_with_new"
    assert result.winner == candidate


def test_multi_source_fact_can_replace_weaker_single_source_fact():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="IBS", trust_level=4, source_count=1)
    candidate = make_record(name="IBS", trust_level=3, source_count=2)

    result = engine.resolve(candidate, existing)

    assert result.action == "replace_with_new"
    assert result.reason == "multi_source_replaces_weaker_single_source"


def test_numeric_range_merge_works():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="HbA1c", fact_type="test_result", value=7.2, days_ago=10)
    candidate = make_record(name="HbA1c", fact_type="test_result", value=7.8, days_ago=1)

    result = engine.resolve(candidate, existing)

    assert result.action == "merge"
    assert result.winner is not None
    assert result.winner.structured["range_min"] == 7.2
    assert result.winner.structured["range_max"] == 7.8


def test_medication_dose_conflict_quarantines_both():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="Lamotrigine", fact_type="medication", dose="100mg")
    candidate = make_record(name="Lamotrigine", fact_type="medication", dose="200mg")

    result = engine.resolve(candidate, existing)

    assert result.action == "quarantine"
    assert len(result.quarantined_records) == 2
    assert all(record.tier == "quarantined" for record in result.quarantined_records)


def test_unresolvable_conflict_quarantines_candidate():
    engine = GovernanceTruthResolutionEngine()
    existing = make_record(name="Headache", fact_type="symptom", trust_level=4)
    candidate = make_record(name="Headache", fact_type="symptom", trust_level=4)
    existing.structured["severity"] = "mild"
    candidate.structured["severity"] = "severe"

    result = engine.resolve(candidate, existing)

    assert result.action == "quarantine"
    assert len(result.quarantined_records) == 1
    assert result.quarantined_records[0].id == candidate.id


def test_ai_derived_fact_enters_hypothesis_tier_only():
    tiering = GovernanceHypothesisTier(enabled=True)
    record = make_record(
        name="Possible MS",
        trust_level=3,
        source_type="ai_response",
        extraction_method="gemini",
    )

    result = tiering.classify_record(record)

    assert result.tier == "hypothesis"
    assert result.status == "hypothesis"
    assert result.requires_review is True


def test_hypothesis_is_excluded_from_active_context_retrieval():
    tiering = GovernanceHypothesisTier(enabled=True)
    active = make_record(name="Epilepsy", tier="active")
    hypothesis = tiering.classify_record(make_record(
        name="Possible autoimmune etiology",
        trust_level=3,
        source_type="web",
        extraction_method="gemini",
    ))

    result = tiering.active_context([active, hypothesis])

    assert result == [active]


def test_feature_flags_disabled_preserve_phase10_behavior(tmp_path: Path):
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "medication", "text": "Lamotrigine", "structured": {"dose": "100mg"}}],
            "confidence": 0.85,
            "latency_ms": 1,
            "notes": [],
        }, specialty="epilepsy"),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    result = pipeline.process_text("x" * 4500, specialty="epilepsy")
    scoring = GovernanceDecisionScoring(enabled=False).score(content="short note")
    disabled_resolution = GovernanceTruthResolutionAdapter(enabled=False).resolve_batch([
        make_record(name="Epilepsy"),
    ])

    assert result.outcome == "written"
    assert result.records[0].tier == "active"
    assert result.records[0].status == "active"
    assert scoring.enabled is False
    assert scoring.score_breakdown == {}
    assert disabled_resolution.records_to_write[0].resolution_action == "replace_with_new"
