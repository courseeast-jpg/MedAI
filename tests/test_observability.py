from __future__ import annotations

import json
from pathlib import Path

from execution.audit import StageAuditLogger
from execution.enrichment import ControlledEnrichment
from execution.metrics import PipelineMetrics
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict):
        self.payload = payload

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


class DuplicateDiagnosisEnrichment:
    def enrich(self, records):
        if not records:
            return []
        base = records[0]
        return [base.model_copy(update={
            "id": "enrichment-diagnosis",
            "source_type": "enrichment",
            "source_name": "observability_enrichment",
            "tier": "hypothesis",
            "requires_review": True,
            "trust_level": 3,
            "confidence": 0.88,
            "enrichment_confidence": 0.88,
            "promotion_history": [],
            "source_count": 1,
        })]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_audit_records_and_metrics_cover_all_stages(tmp_path: Path):
    audit_path = tmp_path / "pipeline_stages.jsonl"
    metrics = PipelineMetrics()
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
        enrichment_engine=DuplicateDiagnosisEnrichment(),
        stage_audit_logger=StageAuditLogger(path=audit_path),
        pipeline_metrics=metrics,
        review_queue_path=tmp_path / "review_queue.jsonl",
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy", session_id="obs-session")

    assert result.outcome == "written"

    audit_events = read_jsonl(audit_path)
    stages = {event["stage"] for event in audit_events}
    assert stages == {
        "extraction",
        "consensus",
        "validation",
        "truth_resolution",
        "safety_gate",
        "enrichment",
        "promotion",
        "final_write",
    }
    assert {event["action"] for event in audit_events} >= {
        "extraction_started",
        "extraction_completed",
        "consensus_result",
        "validation_result",
        "truth_resolution_action",
        "safety_gate_action",
        "enrichment_write",
        "promotion_event",
        "final_write",
    }
    assert all(event["record_id"] for event in audit_events)
    assert all("decision_reason" in event for event in audit_events)

    assert metrics.snapshot() == {
        "total_records_processed": 1,
        "accepted_count": 1,
        "review_count": 0,
        "rejected_count": 0,
        "promoted_count": 1,
        "spacy_count": 1,
        "gemini_count": 0,
        "fallback_count": 0,
        "failure_count": 0,
        "avg_confidence": 0.9,
        "avg_agreement_score": 1.0,
        "avg_latency_per_connector": {
            "spacy": 1.0,
            "gemini": 120.0,
            "phi3": 25.0,
        },
        "avg_confidence_per_connector": {
            "spacy": 0.9,
            "gemini": 0.9,
            "phi3": 0.76,
        },
        "cost_estimate_per_connector": {
            "spacy": 0.0,
            "gemini": 0.02,
            "phi3": 0.005,
        },
        "success_rate_per_connector": {
            "spacy": 1.0,
            "gemini": 0.9,
            "phi3": 0.95,
        },
    }
