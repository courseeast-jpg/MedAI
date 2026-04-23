from pathlib import Path

from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class DeterministicExtractor:
    def extract(self, text: str) -> dict:
        return {
            "extractor": "spacy",
            "entities": [
                {"type": "diagnosis", "text": "Epilepsy"},
                {"type": "medication", "text": "Lamotrigine", "dose": "100mg"},
            ],
            "confidence": 0.9,
            "latency_ms": 1,
            "raw_text": text,
            "notes": [],
        }


def test_execution_pipeline_runs_end_to_end(tmp_path: Path):
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=DeterministicExtractor(),
        audit_logger=type("MemoryAudit", (), {
            "log": lambda self, **kwargs: kwargs,
        })(),
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
