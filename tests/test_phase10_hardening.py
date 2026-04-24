from __future__ import annotations

import json
from pathlib import Path

from app.schemas import ExtractionOutput, MKBRecord
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from execution.validation import validate_extraction_result
from extraction.extractor import Extractor
from extraction.ocr_validator import OCRValidator
from ingestion.pdf_pipeline import PDFPipeline
from mkb.quality_gate import QualityGate


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


class RaisingExtractor:
    def __init__(self, exc: Exception, *, specialty: str = "general"):
        self.exc = exc
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        raise self.exc


class FakeSQLStore:
    def __init__(self):
        self.ledger = []

    def get_by_specialty(self, specialty: str, tier: str | None = None):
        return []

    def get_record(self, record_id: str):
        return None

    def update_status(self, record_id: str, status: str, detail: str | None = None):
        return None

    def write_ledger(self, event):
        self.ledger.append(event)
        return len(self.ledger)

    def get_records_requiring_review(self):
        return []


class FakeVectorStore:
    def check_duplicate(self, content: str):
        return None

    def delete_record(self, record_id: str):
        return None


class CorruptPDFPipeline(PDFPipeline):
    def _extract_text(self, pdf_path: Path) -> str:
        return ""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pipeline(
    *,
    tmp_path: Path,
    spacy_extractor,
    gemini_extractor=None,
    phi3_extractor=None,
) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=spacy_extractor,
        gemini_extractor=gemini_extractor,
        phi3_extractor=phi3_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def test_phase10_audit_fields_present_for_fallback_execution(tmp_path: Path):
    pipeline = build_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.74,
            "latency_ms": 2,
            "notes": [],
        }),
        gemini_extractor=RaisingExtractor(RuntimeError("gemini unavailable"), specialty="epilepsy"),
        phi3_extractor=RaisingExtractor(RuntimeError("phi3 unavailable"), specialty="epilepsy"),
    )

    result = pipeline.process_text("x" * 5000, specialty="epilepsy", source_name="phase10-fallback.txt", session_id="phase10-run")

    assert result.outcome == "queued_for_review"
    assert result.audit["run_id"] == "phase10-run"
    assert result.audit["document_id"] == "phase10-fallback.txt"
    assert result.audit["extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "spacy"
    assert result.audit["fallback_reason"] == "gemini unavailable"
    assert result.audit["confidence_band"] == "auto_accept"
    assert result.audit["quality_gate_decision"] == "review"
    assert result.audit["timestamp"]
    assert result.audit["error_category"] == "route_actual_mismatch"

    audit_events = read_jsonl(tmp_path / "audit.jsonl")
    assert len(audit_events) == 1
    assert set(audit_events[0]) >= {
        "run_id",
        "document_id",
        "extractor_route",
        "extractor_actual",
        "fallback_reason",
        "confidence",
        "confidence_band",
        "quality_gate_decision",
        "timestamp",
        "error_category",
    }


def test_phase10_empty_extraction_is_rejected(tmp_path: Path):
    pipeline = build_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.95,
            "latency_ms": 1,
            "notes": [],
        }),
    )

    result = pipeline.process_text("No extractable facts.", specialty="epilepsy", source_name="empty.txt", session_id="empty-run")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert any(error["code"] == "empty_extraction" for error in result.validation_errors)
    assert result.audit["quality_gate_decision"] == "rejected"
    assert result.audit["error_category"] == "empty_extraction"


def test_phase10_confidence_bands_cover_reject_review_and_auto_accept():
    rejected = validate_extraction_result({
        "extractor": "spacy",
        "actual_extractor": "spacy",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
        "confidence": 0.40,
        "agreement_score": 1.0,
        "disagreement_flag": False,
        "fallback_used": False,
    }, extractor_route="spacy")
    review = validate_extraction_result({
        "extractor": "spacy",
        "actual_extractor": "spacy",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
        "confidence": 0.60,
        "agreement_score": 1.0,
        "disagreement_flag": False,
        "fallback_used": False,
    }, extractor_route="spacy")
    accepted = validate_extraction_result({
        "extractor": "spacy",
        "actual_extractor": "spacy",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
        "confidence": 0.90,
        "agreement_score": 1.0,
        "disagreement_flag": False,
        "fallback_used": False,
    }, extractor_route="spacy")

    assert rejected.status == "rejected"
    assert any(error["code"] == "confidence_below_reject_threshold" for error in rejected.errors)
    assert review.status == "needs_review"
    assert any(error["code"] == "confidence_below_accept_threshold" for error in review.errors)
    assert accepted.status == "accepted"


def test_phase10_route_mismatch_is_rejected_and_audited(tmp_path: Path):
    pipeline = build_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.85,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "rules_based",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.85,
            "latency_ms": 1,
            "notes": ["gemini_route_legacy_fallback=rules_based"],
        }, specialty="epilepsy"),
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy", source_name="mismatch.txt", session_id="mismatch-run")

    assert result.validation_status == "rejected"
    assert any(error["code"] == "route_actual_mismatch" for error in result.validation_errors)
    assert result.audit["error_category"] == "route_actual_mismatch"


def test_phase10_claude_unavailable_legacy_flag_forces_rules_fallback(monkeypatch):
    extractor = Extractor()
    extractor.client = object()
    extractor._gemini_available = True

    def fail_gemini(text: str, specialty: str):
        raise AssertionError("gemini path should not run when claude is marked unavailable")

    def rules_result(text: str):
        return ExtractionOutput(extraction_method="rules_based", confidence=0.45)

    monkeypatch.setattr(extractor, "_extract_gemini", fail_gemini)
    monkeypatch.setattr(extractor, "_extract_rules", rules_result)

    extractor.mark_claude_unavailable()
    result = extractor.extract("Patient has epilepsy and migraine.", specialty="epilepsy")

    assert result.extraction_method == "rules_based"
    assert extractor.claude_available is False


def test_phase10_ocr_failure_is_detected():
    validator = OCRValidator()

    result = validator.validate_ocr_quality("Patient weight 50Omg prescribed metf0rmin l00mg", None)

    assert result["errors_detected"] is True
    assert result["confidence"] < 0.8


def test_phase10_corrupt_pdf_returns_no_records(tmp_path: Path):
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"not-a-real-pdf")

    pipeline = CorruptPDFPipeline(Extractor(), NoopPIIStripper())

    records = pipeline.process(pdf_path, specialty="epilepsy", session_id="corrupt-run")

    assert records == []


def test_phase10_quality_gate_certifies_threshold_behavior():
    sql_store = FakeSQLStore()
    vector_store = FakeVectorStore()
    gate = QualityGate(sql=sql_store, vec=vector_store)

    low_candidate = MKBRecord(
        fact_type="diagnosis",
        content="Diagnosis: Epilepsy",
        structured={"name": "Epilepsy"},
        specialty="epilepsy",
        source_type="document",
        source_name="quality-low.txt",
        confidence=0.20,
        extraction_method="rules_based",
    )
    high_candidate = MKBRecord(
        fact_type="diagnosis",
        content="Diagnosis: Epilepsy",
        structured={"name": "Epilepsy"},
        specialty="epilepsy",
        source_type="document",
        source_name="quality-high.txt",
        confidence=0.90,
        extraction_method="gemini",
    )

    low_approved, low_reason, _ = gate.check(low_candidate, session_id="quality-low")
    high_approved, high_reason, final = gate.check(high_candidate, session_id="quality-high")

    assert low_approved is False
    assert "below threshold" in low_reason.lower()
    assert high_approved is True
    assert high_reason == "approved"
    assert final is not None
