from __future__ import annotations

from pathlib import Path

from execution.consensus import consensus_merge
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_gemini_route_pipeline(*, spacy_payload: dict, gemini_payload: dict, tmp_path: Path) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor(spacy_payload),
        gemini_extractor=StaticExtractor(gemini_payload, specialty="epilepsy"),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def test_consensus_merge_full_agreement():
    merged = consensus_merge([
        {
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.8,
            "latency_ms": 1,
            "raw_text": "Diagnosis: Epilepsy.",
            "notes": [],
        },
        {
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 2,
            "raw_text": "Diagnosis: Epilepsy.",
            "notes": [],
        },
    ], extractor_route="gemini")

    assert merged["agreement_score"] == 1.0
    assert merged["confidence"] == 0.85
    assert merged["disagreement_flag"] is False
    assert merged["entities"][0]["consensus_support_count"] == 2


def test_consensus_partial_disagreement_routes_to_review(tmp_path: Path):
    pipeline = make_gemini_route_pipeline(
        spacy_payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [
                {"type": "diagnosis", "text": "Epilepsy"},
                {"type": "diagnosis", "text": "Migraine"},
            ],
            "confidence": 1.0,
            "latency_ms": 1,
            "notes": [],
        },
        gemini_payload={
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 1.0,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert result.audit["confidence"] == 0.5
    assert result.extractor_result["agreement_score"] == 0.5
    assert any(error["code"] == "agreement_below_accept_threshold" for error in result.validation_errors)
    assert any(error["code"] == "consensus_disagreement_flag" for error in result.validation_errors)


def test_consensus_conflicting_outputs_are_rejected(tmp_path: Path):
    pipeline = make_gemini_route_pipeline(
        spacy_payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        gemini_payload={
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert result.audit["confidence"] == 0.0
    assert result.extractor_result["agreement_score"] == 0.0
    assert any(error["code"] == "agreement_below_reject_threshold" for error in result.validation_errors)
