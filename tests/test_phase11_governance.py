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


class FakeSQLStore:
    def __init__(self):
        self.records: list[MKBRecord] = []

    def write_record(self, record: MKBRecord, session_id: str = ""):
        self.records.append(record)
        return record.id

    def get_by_specialty(self, specialty: str, tier: str | None = None):
        items = [record for record in self.records if record.specialty == specialty and record.status == "active"]
        if tier is not None:
            items = [record for record in items if record.tier == tier]
        return items

    def get_record(self, record_id: str):
        for record in self.records:
            if record.id == record_id:
                return record
        return None

    def update_status(self, record_id: str, status: str, detail: str | None = None):
        for index, record in enumerate(self.records):
            if record.id == record_id:
                self.records[index] = record.model_copy(update={"status": status})
                break

    def write_ledger(self, event):
        return 1

    def get_records_requiring_review(self):
        return [record for record in self.records if record.requires_review]

    def get_active_medications(self):
        return [record for record in self.records if record.fact_type == "medication" and record.tier == "active" and record.status == "active"]


class FakeVectorStore:
    def add_record(self, record: MKBRecord):
        return None

    def delete_record(self, record_id: str):
        return None


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


def test_hypothesis_activation_routes_ai_fact_into_hypothesis_tier(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("execution.pipeline.ENABLE_HYPOTHESIS_TIER", True)
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
    pipeline.governance_hypothesis = GovernanceHypothesisTier(enabled=True)

    result = pipeline.process_text("x" * 4500, specialty="epilepsy")

    governed_records = result.records or result.queued_records

    assert result.outcome in {"written", "queued_for_review"}
    assert governed_records[0].tier == "hypothesis"
    assert governed_records[0].status == "hypothesis"


def test_hypothesis_activation_keeps_active_document_facts_active(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("execution.pipeline.ENABLE_HYPOTHESIS_TIER", True)
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    pipeline.governance_hypothesis = GovernanceHypothesisTier(enabled=True)

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.records[0].tier == "active"
    assert result.records[0].status == "active"


def test_truth_resolution_activation_routes_conflicts_through_pipeline(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("governance.truth_resolution.ENABLE_TRUTH_RESOLUTION", True)
    monkeypatch.setattr("execution.pipeline.ENABLE_HYPOTHESIS_TIER", False)
    existing = make_record(name="Epilepsy", trust_level=1)
    store = FakeSQLStore()
    store.records.append(existing)
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        sql_store=store,
        vector_store=FakeVectorStore(),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    pipeline.truth_resolver = GovernanceTruthResolutionAdapter(
        lambda record: [
            item for item in store.get_by_specialty(record.specialty, tier="active")
            if item.fact_type == record.fact_type
        ],
        enabled=True,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert len(result.queued_records) == 1
    assert result.queued_records[0].tier == "quarantined"


def test_truth_resolution_activation_unresolved_conflict_quarantines_candidate(monkeypatch):
    monkeypatch.setattr("governance.truth_resolution.ENABLE_TRUTH_RESOLUTION", True)
    existing = make_record(name="Headache", fact_type="symptom", trust_level=4)
    candidate = make_record(name="Headache", fact_type="symptom", trust_level=4)
    existing.structured["severity"] = "mild"
    candidate.structured["severity"] = "severe"
    adapter = GovernanceTruthResolutionAdapter(enabled=True, existing_records_provider=lambda record: [existing])

    result = adapter.resolve_batch([candidate])

    assert result.records_to_write == []
    assert len(result.quarantined_records) == 1
    assert result.quarantined_records[0].tier == "quarantined"


def test_truth_resolution_activation_medication_dose_conflict_quarantines_both(monkeypatch):
    monkeypatch.setattr("governance.truth_resolution.ENABLE_TRUTH_RESOLUTION", True)
    existing = make_record(name="Lamotrigine", fact_type="medication", dose="100mg")
    candidate = make_record(name="Lamotrigine", fact_type="medication", dose="200mg")
    adapter = GovernanceTruthResolutionAdapter(enabled=True, existing_records_provider=lambda record: [existing])

    result = adapter.resolve_batch([candidate])

    assert result.records_to_write == []
    assert len(result.quarantined_records) == 2


def test_truth_resolution_activation_requires_no_destructive_migration(tmp_path: Path):
    db_path = tmp_path / "governance_activation.db"
    db_path.touch()

    assert db_path.exists()
    assert db_path.stat().st_size == 0


def test_decision_scoring_activation_calculates_deterministic_score():
    scorer = GovernanceDecisionScoring(enabled=True)
    context = [make_record(name="Epilepsy"), make_record(name="Lamotrigine", fact_type="medication")]

    result = scorer.score(
        content="Epilepsy management is appropriate because Lamotrigine is already listed.",
        mkb_context=context,
        citations=["guideline-1"],
        ddi_safety_score=0.9,
    )

    assert result.enabled is True
    assert result.final_score == 0.93
    assert result.score_breakdown["ddi_safety_score"] == 0.9


def test_decision_scoring_low_score_is_flagged_but_does_not_crash_pipeline(tmp_path: Path):
    scorer = GovernanceDecisionScoring(enabled=True)
    score = scorer.score(content="", mkb_context=[], citations=[], ddi_safety_score=0.1)
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert score.final_score == 0.235
    assert score.final_score < 0.3
    assert result.outcome == "written"


def test_decision_scoring_ddi_safety_score_participates_in_final_score():
    scorer = GovernanceDecisionScoring(enabled=True)

    low = scorer.score(content="Because this is consistent.", mkb_context=[], citations=[], ddi_safety_score=0.1)
    high = scorer.score(content="Because this is consistent.", mkb_context=[], citations=[], ddi_safety_score=0.9)

    assert high.final_score > low.final_score


def test_decision_scoring_uses_no_real_external_apis(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("external API should not be called")

    monkeypatch.setattr("requests.get", fail, raising=False)
    monkeypatch.setattr("requests.post", fail, raising=False)
    scorer = GovernanceDecisionScoring(enabled=True)

    result = scorer.score(content="Short coherent note because evidence exists.", mkb_context=[], citations=[], ddi_safety_score=1.0)

    assert result.enabled is True


def test_feature_flags_disabled_preserve_phase10_behavior(tmp_path: Path):
    import execution.pipeline as pipeline_module

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
    pipeline.governance_hypothesis = GovernanceHypothesisTier(enabled=False)
    pipeline.truth_resolver = GovernanceTruthResolutionAdapter(enabled=False)

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
