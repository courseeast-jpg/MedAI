from pathlib import Path

import pytest

from app.schemas import DDIFinding
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class MemoryAudit:
    def __init__(self):
        self.events = []

    def log(self, **kwargs):
        event = {**kwargs, "final_status": kwargs["outcome"]}
        self.events.append(event)
        return event


class DeterministicExtractor:
    def __init__(self, entities=None):
        self.entities = entities or [
            {"type": "diagnosis", "text": "Epilepsy"},
            {"type": "medication", "text": "Lamotrigine", "dose": "100mg"},
        ]

    def extract(self, text: str) -> dict:
        return {
            "extractor": "spacy",
            "entities": self.entities,
            "confidence": 0.9,
            "latency_ms": 1,
            "raw_text": text,
            "notes": [],
        }


class InvalidExtractor:
    def extract(self, text: str) -> dict:
        return {"extractor": "bad", "entities": []}


class RecordingWriter:
    def __init__(self):
        self.calls = []
        self.sql_store = None

    def write(self, records, session_id=""):
        self.calls.append([record.content for record in records])
        return records, []


class BlockingMedicationGate:
    def __init__(self):
        self.calls = []

    def gate_medication_write(self, candidate, session_id=""):
        self.calls.append(candidate.content)
        finding = DDIFinding(
            drug_a=candidate.structured["name"],
            drug_b="Valproate",
            severity="HIGH",
            management="Do not combine without clinician review.",
        )
        return "block", "blocked", [finding]


class LegacyRulesOutput:
    diagnoses = []
    medications = []
    test_results = []
    symptoms = []
    notes = []
    recommendations = []
    extraction_method = "rules_based"
    confidence = 0.45


class LegacyRulesExtractor:
    def extract(self, text: str, specialty: str = "general"):
        return LegacyRulesOutput()


def test_execution_pipeline_runs_end_to_end(tmp_path: Path):
    audit = MemoryAudit()
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=DeterministicExtractor(),
        audit_logger=audit,
    )

    result = pipeline.process_text(
        "Diagnosis: Epilepsy. Lamotrigine 100mg daily.",
        specialty="epilepsy",
        source_name="unit-test.txt",
        session_id="test-session",
    )

    assert result.outcome == "written"
    assert result.written_count == 2
    assert result.extractor_result["extractor"] == "spacy"
    assert {record.fact_type for record in result.records} == {"diagnosis", "medication"}
    assert result.audit["outcome"] == "written"
    assert result.audit["final_status"] == "written"
    assert audit.events[-1]["entity_count"] == 2


def test_ddi_gate_runs_before_write():
    gate = BlockingMedicationGate()
    writer = RecordingWriter()
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        medication_gate=gate,
        spacy_extractor=DeterministicExtractor(
            entities=[{"type": "medication", "text": "Lamotrigine", "dose": "100mg"}]
        ),
        audit_logger=MemoryAudit(),
    )
    pipeline.writer = writer

    result = pipeline.process_text("Lamotrigine 100mg daily.", specialty="epilepsy")

    assert result.outcome == "blocked_ddi"
    assert gate.calls == ["Medication: Lamotrigine 100mg"]
    assert writer.calls == []


def test_invalid_extractor_schema_is_rejected():
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=InvalidExtractor(),
        audit_logger=MemoryAudit(),
    )

    with pytest.raises(ValueError, match="missing keys"):
        pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")


def test_simulated_real_document_routes_and_writes():
    audit = MemoryAudit()
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=DeterministicExtractor(
            entities=[
                {"type": "diagnosis", "text": "Focal epilepsy"},
                {"type": "test_result", "text": "EEG", "value": "abnormal"},
                {"type": "medication", "text": "Levetiracetam", "dose": "500mg"},
            ]
        ),
        audit_logger=audit,
    )

    document_text = """
    Neurology clinic note.
    Diagnosis: Focal epilepsy.
    EEG: abnormal temporal sharp waves.
    Medication plan: Levetiracetam 500mg twice daily.
    """

    result = pipeline.process_text(document_text, specialty="epilepsy", source_name="simulated-clinic-note.txt")

    assert result.outcome == "written"
    assert result.written_count == 3
    assert result.audit["extractor"] == "spacy"
    assert result.audit["entity_count"] == 3
    assert result.audit["final_status"] == "written"


def test_gemini_adapter_reports_route_when_legacy_falls_back():
    from extractors.gemini_extractor import GeminiExtractor

    result = GeminiExtractor(legacy_extractor=LegacyRulesExtractor()).extract("x" * 2000)

    assert result["extractor"] == "gemini"
    assert result["notes"] == ["gemini_route_legacy_fallback=rules_based"]
