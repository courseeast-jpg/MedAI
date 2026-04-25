from __future__ import annotations

import json
from pathlib import Path

from execution.logging import AuditLogger
from execution.medical_coding import map_medical_codes
from execution.pipeline import ExecutionPipeline
from monitoring.observability import build_phase25_metrics, write_phase25_outputs


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_pipeline(*, extractor, tmp_path: Path, gemini_extractor=None, phi3_extractor=None) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=extractor,
        gemini_extractor=gemini_extractor,
        phi3_extractor=phi3_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def make_summary() -> dict:
    return {
        "generated_at": "2026-04-25T05:00:00+00:00",
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
                "coding_attempted_count": 2,
                "coding_success_count": 2,
                "coding_unmapped_count": 0,
                "coding_ambiguous_count": 0,
                "coding_skipped_count": 0,
                "medical_coding": {
                    "applied": True,
                    "coding_source": "rules_based_seed",
                    "entries": [
                        {
                            "original_entity_text": "type 2 diabetes",
                            "normalized_entity_text": "type 2 diabetes",
                            "entity_type": "diagnosis",
                            "coding_system": "SNOMED-CT-seed",
                            "code": "44054006",
                            "code_display": "Type 2 diabetes mellitus",
                            "coding_confidence": 0.95,
                            "coding_source": "rules_based_seed",
                            "coding_status": "coded",
                        },
                        {
                            "original_entity_text": "metformin",
                            "normalized_entity_text": "metformin",
                            "entity_type": "medication",
                            "coding_system": "UMLS-seed",
                            "code": "C0025598",
                            "code_display": "Metformin",
                            "coding_confidence": 0.95,
                            "coding_source": "rules_based_seed",
                            "coding_status": "coded",
                        },
                    ],
                },
            },
            {
                "document": "review.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "confidence": 0.68,
                "confidence_band": "review",
                "coding_attempted_count": 2,
                "coding_success_count": 0,
                "coding_unmapped_count": 1,
                "coding_ambiguous_count": 1,
                "coding_skipped_count": 0,
                "medical_coding": {
                    "applied": True,
                    "coding_source": "rules_based_seed",
                    "entries": [
                        {
                            "original_entity_text": "MI",
                            "normalized_entity_text": "mi",
                            "entity_type": "diagnosis",
                            "coding_system": "SNOMED-CT-seed",
                            "code": None,
                            "code_display": "Ambiguous abbreviation for myocardial infarction",
                            "coding_confidence": 0.5,
                            "coding_source": "rules_based_seed",
                            "coding_status": "ambiguous",
                        },
                        {
                            "original_entity_text": "unknown drug",
                            "normalized_entity_text": "unknown drug",
                            "entity_type": "medication",
                            "coding_system": None,
                            "code": None,
                            "code_display": None,
                            "coding_confidence": 0.0,
                            "coding_source": "rules_based_seed",
                            "coding_status": "unmapped",
                        },
                    ],
                },
            },
        ],
    }


def test_deterministic_synonym_mapping():
    result = map_medical_codes(
        entities=[
            {"type": "diagnosis", "text": "heart attack"},
            {"type": "diagnosis", "text": "myocardial infarction"},
            {"type": "diagnosis", "text": "high blood pressure"},
            {"type": "medication", "text": "aspirin"},
        ]
    )

    assert result.entries[0].code == "22298006"
    assert result.entries[1].code == "22298006"
    assert result.entries[2].code == "38341003"
    assert result.entries[3].code == "C0004057"


def test_unmapped_entities_remain_unmapped_not_guessed():
    result = map_medical_codes(
        entities=[{"type": "diagnosis", "text": "mystery syndrome"}]
    )

    assert result.entries[0].coding_status == "unmapped"
    assert result.entries[0].code is None
    assert result.coding_unmapped_count == 1


def test_ambiguous_mappings_are_not_auto_resolved():
    result = map_medical_codes(
        entities=[{"type": "diagnosis", "text": "MI"}]
    )

    assert result.entries[0].coding_status == "ambiguous"
    assert result.entries[0].code is None
    assert result.coding_ambiguous_count == 1


def test_coding_does_not_change_confidence_values(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "type 2 diabetes"}],
            "confidence": 0.92,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Type 2 diabetes noted.", specialty="general")

    assert result.audit["confidence"] == 0.92
    assert result.audit["raw_confidence"] == 0.92
    assert result.audit["calibrated_confidence"] == 0.92
    assert result.extractor_result["medical_coding"]["coding_success_count"] == 1


def test_coding_does_not_change_confidence_bands(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "high blood pressure"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("High blood pressure documented.", specialty="general")

    assert result.audit["confidence_band"] == "review"
    assert result.outcome == "queued_for_review"
    assert result.extractor_result["medical_coding"]["coding_success_count"] == 1


def test_coding_does_not_change_routing_decisions(tmp_path: Path):
    gemini_extractor = StaticExtractor({
        "extractor": "gemini",
        "actual_extractor": "rules_based",
        "entities": [{"type": "diagnosis", "text": "heart attack"}],
        "confidence": 0.85,
        "latency_ms": 1,
        "notes": [],
    }, specialty="cardiology")
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

    result = pipeline.process_text("x" * 4000, specialty="cardiology")

    assert result.audit["requested_extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "rules_based"
    assert result.extractor_result["medical_coding"]["coding_success_count"] == 1


def test_coding_does_not_change_review_write_decisions(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "MI"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("MI under review.", specialty="cardiology")

    assert result.outcome == "queued_for_review"
    assert result.audit["review_recommendation"] == "operator_review"
    assert result.extractor_result["medical_coding"]["coding_ambiguous_count"] == 1


def test_rejected_outputs_are_not_coded(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "unknown condition"}],
            "confidence": 0.4,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Unknown condition.", specialty="general")

    assert result.validation_status == "rejected"
    assert result.extractor_result["medical_coding"] is None


def test_phase25_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase25" / "medical_coding.json"
    report_path = tmp_path / "reports" / "phase25" / "medical_coding_report.md"

    metrics = write_phase25_outputs(make_summary(), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 25 Medical Coding Report" in report_path.read_text(encoding="utf-8")


def test_phase25_metrics_are_deterministic_across_reruns():
    first = build_phase25_metrics(make_summary())
    second = build_phase25_metrics(make_summary())

    assert first == second
