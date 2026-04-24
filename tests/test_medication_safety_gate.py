from __future__ import annotations

import json
from pathlib import Path

from app.schemas import MKBRecord
from execution.logging import AuditLogger
from execution.medication_safety_gate import MedicationSafetyGate
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict):
        self.payload = payload

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_record(name: str) -> MKBRecord:
    return MKBRecord(
        fact_type="medication",
        content=f"Medication: {name}",
        structured={"name": name},
        specialty="epilepsy",
        source_type="document",
        source_name="existing.txt",
        trust_level=1,
        confidence=0.9,
    )


def build_pipeline(
    *,
    payload: dict,
    tmp_path: Path,
    medication_gate=None,
    active_medications_provider=None,
) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor(payload),
        medication_gate=medication_gate,
        active_medications_provider=active_medications_provider,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_non_medication_bypass(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.records[0].ddi_checked is False
    assert result.records[0].ddi_status is None


def test_medication_clear_allowed(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "medication", "text": "Lamotrigine", "structured": {"dose": "100mg"}}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        active_medications_provider=lambda: [],
    )

    result = pipeline.process_text("Lamotrigine 100mg daily.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.records[0].ddi_checked is True
    assert result.records[0].ddi_status == "clear"
    assert result.records[0].safety_action == "allow"


def test_low_interaction_allowed_with_note(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "medication", "text": "Sertraline"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        active_medications_provider=lambda: [make_record("Ibuprofen")],
    )

    result = pipeline.process_text("Sertraline.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.records[0].ddi_status == "low"
    assert result.records[0].safety_action == "allow_with_note"
    assert "Low-severity" in result.records[0].structured["ddi_note"]


def test_medium_interaction_routes_to_review(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "medication", "text": "Warfarin"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        active_medications_provider=lambda: [make_record("Ibuprofen")],
    )

    result = pipeline.process_text("Warfarin.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.queued_count == 1
    assert result.queued_records[0].ddi_status == "medium"
    assert result.queued_records[0].safety_action == "needs_review"
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["ddi_status"] == "medium"


def test_high_interaction_blocked(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "medication", "text": "Lamotrigine"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        active_medications_provider=lambda: [make_record("Valproate")],
    )

    result = pipeline.process_text("Lamotrigine.", specialty="epilepsy")

    assert result.outcome == "blocked_ddi"
    assert result.blocked_records[0].ddi_status == "high_blocked"
    assert result.blocked_records[0].safety_action == "block_write"
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["ddi_status"] == "high_blocked"


def test_unavailable_ddi_routes_pending_check(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "medication", "text": "Lamotrigine"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        medication_gate=MedicationSafetyGate(available=False),
    )

    result = pipeline.process_text("Lamotrigine.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.queued_count == 1
    assert result.queued_records[0].ddi_checked is False
    assert result.queued_records[0].ddi_status == "pending_ddi_check"
    assert result.queued_records[0].safety_action == "pending_ddi_check"
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["ddi_status"] == "pending_ddi_check"
