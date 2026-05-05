from __future__ import annotations

import json
from pathlib import Path

from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from execution.review_queue import read_review_queue
from execution.semantic_enrichment import enrich_semantics
from monitoring.observability import build_phase24_metrics, write_phase24_outputs


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_pipeline(*, extractor, tmp_path: Path, gemini_extractor=None) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=extractor,
        gemini_extractor=gemini_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def make_summary() -> dict:
    return {
        "generated_at": "2026-04-25T04:00:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 2,
        "documents_processed": 2,
        "written": 1,
        "queued_for_review": 1,
        "external_quota_blocked": 0,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "documents": [
            {
                "document": "accepted.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "confidence": 0.9,
                "confidence_band": "high",
                "semantic_enrichment": {
                    "applied": True,
                    "enrichment_source": "rules_based",
                    "entities": [
                        {
                            "entity_index": 0,
                            "entity_type": "diagnosis",
                            "entity_text": "Epilepsy",
                            "negation_flag": False,
                            "temporal_info": {"kind": "relative", "value": "today"},
                            "relationships": [],
                            "enrichment_confidence": 0.85,
                            "enrichment_source": "rules_based",
                        }
                    ],
                    "enriched_entity_count": 1,
                    "negation_detected_count": 0,
                    "temporal_detected_count": 1,
                    "relationships_detected_count": 0,
                },
                "enrichment_applied": True,
                "negation_detected_count": 0,
                "temporal_detected_count": 1,
                "relationships_detected_count": 0,
            },
            {
                "document": "review.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "confidence": 0.68,
                "confidence_band": "review",
                "semantic_enrichment": {
                    "applied": True,
                    "enrichment_source": "rules_based",
                    "entities": [
                        {
                            "entity_index": 0,
                            "entity_type": "diagnosis",
                            "entity_text": "Migraine",
                            "negation_flag": True,
                            "temporal_info": None,
                            "relationships": [],
                            "enrichment_confidence": 0.85,
                            "enrichment_source": "rules_based",
                        }
                    ],
                    "enriched_entity_count": 1,
                    "negation_detected_count": 1,
                    "temporal_detected_count": 0,
                    "relationships_detected_count": 0,
                },
                "enrichment_applied": True,
                "negation_detected_count": 1,
                "temporal_detected_count": 0,
                "relationships_detected_count": 0,
            },
        ],
    }


def test_enrichment_does_not_change_confidence_values(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.92,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Today patient has Epilepsy.", specialty="epilepsy")

    assert result.audit["confidence"] == 0.92
    assert result.audit["raw_confidence"] == 0.92
    assert result.audit["calibrated_confidence"] == 0.92


def test_enrichment_does_not_change_confidence_bands(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.6,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Patient denies Migraine today.", specialty="neurology")

    assert result.audit["confidence_band"] == "review"
    assert result.outcome == "queued_for_review"


def test_enrichment_does_not_change_routing_decisions(tmp_path: Path):
    gemini_extractor = StaticExtractor({
        "extractor": "gemini",
        "actual_extractor": "rules_based",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
        "confidence": 0.85,
        "latency_ms": 1,
        "notes": [],
    }, specialty="epilepsy")
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=gemini_extractor,
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.audit["requested_extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "rules_based"
    assert result.extractor_result["semantic_enrichment"]["applied"] is True


def test_enrichment_does_not_remove_review_band_items(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.64,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("History of Migraine.", specialty="neurology")
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert result.outcome == "queued_for_review"
    assert result.audit["review_recommendation"] == "operator_review"
    assert queued[0]["confidence_band"] == "review"


def test_enrichment_attaches_metadata_correctly(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [
                {"type": "diagnosis", "text": "Epilepsy"},
                {"type": "medication", "text": "Keppra"},
            ],
            "confidence": 0.88,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Today Epilepsy managed with Keppra.", specialty="epilepsy")
    enrichment = result.extractor_result["semantic_enrichment"]

    assert enrichment["applied"] is True
    assert enrichment["enrichment_source"] == "rules_based"
    assert enrichment["temporal_detected_count"] == 2
    assert enrichment["relationships_detected_count"] == 2
    assert enrichment["entities"][0]["entity_text"] == "Epilepsy"


def test_enrichment_is_deterministic_across_reruns():
    raw_text = "Patient denies migraine today and takes sumatriptan."
    entities = [
        {"type": "diagnosis", "text": "migraine"},
        {"type": "medication", "text": "sumatriptan"},
    ]

    first = enrich_semantics(raw_text=raw_text, entities=entities)
    second = enrich_semantics(raw_text=raw_text, entities=entities)

    assert first == second


def test_reject_band_outputs_are_not_enriched(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Unknown condition"}],
            "confidence": 0.4,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Unknown condition.", specialty="general")

    assert result.validation_status == "rejected"
    assert result.extractor_result["semantic_enrichment"] is None


def test_phase24_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase24" / "semantic_enrichment.json"
    report_path = tmp_path / "reports" / "phase24" / "semantic_enrichment_report.md"

    metrics = write_phase24_outputs(make_summary(), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 24 Semantic Enrichment Report" in report_path.read_text(encoding="utf-8")
    assert build_phase24_metrics(make_summary())["enrichment_applied_count"] == 2
