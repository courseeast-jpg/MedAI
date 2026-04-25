from __future__ import annotations

import json
from pathlib import Path

from execution.language_support import detect_language_support
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from execution.review_queue import read_review_queue
from monitoring.observability import build_phase26_metrics, write_phase26_outputs


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
        "generated_at": "2026-04-25T06:00:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 4,
        "documents_processed": 4,
        "written": 2,
        "queued_for_review": 2,
        "external_quota_blocked": 0,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "documents": [
            {
                "document": "english.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "language_support": {
                    "detected_language": "english",
                    "language_confidence": 1.0,
                    "script_detected": "latin",
                    "cyrillic_detected": False,
                    "requires_ocr": False,
                    "language_route_note": "metadata_only:english_latin_detected",
                    "translation_status": "not_required",
                    "language_support_status": "supported_metadata_only",
                },
                "detected_language": "english",
                "language_confidence": 1.0,
                "script_detected": "latin",
                "cyrillic_detected": False,
                "requires_ocr": False,
                "language_route_note": "metadata_only:english_latin_detected",
                "translation_status": "not_required",
                "language_support_status": "supported_metadata_only",
            },
            {
                "document": "russian.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "language_support": {
                    "detected_language": "russian",
                    "language_confidence": 1.0,
                    "script_detected": "cyrillic",
                    "cyrillic_detected": True,
                    "requires_ocr": False,
                    "language_route_note": "metadata_only:russian_cyrillic_detected",
                    "translation_status": "pending_translation",
                    "language_support_status": "supported_metadata_only",
                },
                "detected_language": "russian",
                "language_confidence": 1.0,
                "script_detected": "cyrillic",
                "cyrillic_detected": True,
                "requires_ocr": False,
                "language_route_note": "metadata_only:russian_cyrillic_detected",
                "translation_status": "pending_translation",
                "language_support_status": "supported_metadata_only",
            },
            {
                "document": "mixed.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "language_support": {
                    "detected_language": "mixed",
                    "language_confidence": 0.6,
                    "script_detected": "mixed",
                    "cyrillic_detected": True,
                    "requires_ocr": False,
                    "language_route_note": "metadata_only:mixed_language_detected",
                    "translation_status": "pending_translation",
                    "language_support_status": "supported_metadata_only",
                },
                "detected_language": "mixed",
                "language_confidence": 0.6,
                "script_detected": "mixed",
                "cyrillic_detected": True,
                "requires_ocr": False,
                "language_route_note": "metadata_only:mixed_language_detected",
                "translation_status": "pending_translation",
                "language_support_status": "supported_metadata_only",
            },
            {
                "document": "unknown.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "rejected",
                "language_support": {
                    "detected_language": "unknown",
                    "language_confidence": 0.0,
                    "script_detected": "unknown",
                    "cyrillic_detected": False,
                    "requires_ocr": True,
                    "language_route_note": "metadata_only:language_signal_insufficient",
                    "translation_status": "not_required",
                    "language_support_status": "unknown_metadata_only",
                },
                "detected_language": "unknown",
                "language_confidence": 0.0,
                "script_detected": "unknown",
                "cyrillic_detected": False,
                "requires_ocr": True,
                "language_route_note": "metadata_only:language_signal_insufficient",
                "translation_status": "not_required",
                "language_support_status": "unknown_metadata_only",
            },
        ],
    }


def test_russian_cyrillic_detection_is_deterministic():
    first = detect_language_support(text="Пациент принимает аспирин.")
    second = detect_language_support(text="Пациент принимает аспирин.")

    assert first == second
    assert first.detected_language == "russian"
    assert first.script_detected == "cyrillic"
    assert first.cyrillic_detected is True
    assert first.translation_status == "pending_translation"


def test_english_detection_is_deterministic():
    result = detect_language_support(text="Patient takes aspirin daily.")

    assert result.detected_language == "english"
    assert result.script_detected == "latin"
    assert result.cyrillic_detected is False
    assert result.translation_status == "not_required"


def test_mixed_language_detection_is_deterministic():
    result = detect_language_support(text="Patient принимает aspirin daily.")

    assert result.detected_language == "mixed"
    assert result.script_detected == "mixed"
    assert result.cyrillic_detected is True
    assert result.translation_status == "pending_translation"


def test_unknown_language_fallback_is_deterministic():
    result = detect_language_support(text="12345 !!!")

    assert result.detected_language == "unknown"
    assert result.script_detected == "unknown"
    assert result.language_confidence == 0.0
    assert result.language_support_status == "unknown_metadata_only"


def test_ocr_requirement_is_metadata_only(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("12345 !!!", specialty="general")

    assert result.outcome == "written"
    assert result.extractor_result["language_support"]["requires_ocr"] is True
    assert result.audit["confidence"] == 0.9


def test_translation_status_does_not_block_pipeline(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Пациент сообщает об эпилепсии.", specialty="general")

    assert result.outcome == "written"
    assert result.extractor_result["language_support"]["translation_status"] == "pending_translation"
    assert result.audit["validation_status"] == "accepted"


def test_language_metadata_does_not_change_confidence_values(tmp_path: Path):
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

    result = pipeline.process_text("Patient has epilepsy.", specialty="general")

    assert result.audit["confidence"] == 0.92
    assert result.audit["raw_confidence"] == 0.92
    assert result.audit["calibrated_confidence"] == 0.92


def test_language_metadata_does_not_change_confidence_bands(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Пациент сообщает о migraine.", specialty="general")

    assert result.audit["confidence_band"] == "review"
    assert result.outcome == "queued_for_review"


def test_language_metadata_does_not_change_routing_review_write_decisions(tmp_path: Path):
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

    result = pipeline.process_text("Пациент epilepsy " * 600, specialty="epilepsy")

    assert result.audit["requested_extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "rules_based"
    assert result.outcome == "written"
    assert result.extractor_result["language_support"]["detected_language"] == "mixed"


def test_language_metadata_does_not_remove_review_band_items(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Пациент сообщает о migraine.", specialty="neurology")
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert result.outcome == "queued_for_review"
    assert result.audit["review_recommendation"] == "operator_review"
    assert queued[0]["confidence_band"] == "review"


def test_phase25_medical_coding_output_structure_remains_valid(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [
                {"type": "diagnosis", "text": "type 2 diabetes"},
                {"type": "medication", "text": "metformin"},
            ],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Пациент с type 2 diabetes принимает metformin.", specialty="general")
    medical_coding = result.extractor_result["medical_coding"]

    assert set(medical_coding.keys()) == {
        "applied",
        "coding_source",
        "entries",
        "coding_attempted_count",
        "coding_success_count",
        "coding_unmapped_count",
        "coding_ambiguous_count",
        "coding_skipped_count",
    }
    assert medical_coding["coding_success_count"] == 2


def test_phase26_metrics_are_deterministic_across_reruns():
    first = build_phase26_metrics(make_summary())
    second = build_phase26_metrics(make_summary())

    assert first == second
    assert first["language_detected_counts"] == {
        "english": 1,
        "mixed": 1,
        "russian": 1,
        "unknown": 1,
    }
    assert first["cyrillic_detected_count"] == 2
    assert first["pending_translation_count"] == 2
    assert first["requires_ocr_count"] == 1


def test_phase26_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase26" / "language_support.json"
    report_path = tmp_path / "reports" / "phase26" / "language_support_report.md"

    metrics = write_phase26_outputs(make_summary(), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 26 Language Support Report" in report_path.read_text(encoding="utf-8")
