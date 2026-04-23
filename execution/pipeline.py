"""Phase 1 deterministic execution pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from app.config import TIER_ACTIVE, TIER_QUARANTINED, TRUST_CLINICAL
from app.schemas import MKBRecord
from execution.jobs import ExecutionJob, ExecutionResult
from execution.logging import AuditLogger
from execution.mkb_writer import MKBWriter
from execution.safety import ExecutionSafety
from extractors.gemini_extractor import GeminiExtractor
from extractors.spacy_extractor import SpacyExtractor


OCR_ARTIFACT_RE = re.compile(r"(\ufffd|[|]{3,}|_{4,}|\b(?:l|I){8,}\b)")
EXTRACTOR_SCHEMA_KEYS = {"extractor", "entities", "confidence", "latency_ms", "raw_text", "notes"}


class ExecutionPipeline:
    """Routes extraction, checks DDI blocks, and writes to MKB."""

    def __init__(
        self,
        *,
        sql_store=None,
        vector_store=None,
        quality_gate=None,
        medication_gate=None,
        pii_stripper=None,
        audit_logger: AuditLogger | None = None,
        spacy_extractor: SpacyExtractor | None = None,
        gemini_extractor: GeminiExtractor | None = None,
    ):
        self.pii_stripper = pii_stripper or self._build_pii_stripper()
        self.audit_logger = audit_logger or AuditLogger()
        self.spacy_extractor = spacy_extractor or SpacyExtractor()
        self.gemini_extractor = gemini_extractor
        self.writer = MKBWriter(sql_store, vector_store, quality_gate)
        self.safety = ExecutionSafety(medication_gate)

    def run(self, job: ExecutionJob) -> ExecutionResult:
        source_text = job.text
        source_name = job.source_name
        session_id = job.session_id or str(uuid4())

        if job.pdf_path is not None:
            source_text = self.extract_pdf_text(job.pdf_path)
            source_name = job.source_name or job.pdf_path.name

        stripped_text, pii_method = self._strip_pii(source_text)
        extractor = self._select_extractor(stripped_text, job.specialty)
        extracted = extractor.extract(stripped_text)
        self._validate_extractor_output(extracted)
        extracted["notes"] = list(extracted.get("notes", [])) + [f"pii_method={pii_method}"]

        candidates = self._entities_to_records(
            extracted.get("entities", []),
            source_name=source_name,
            specialty=job.specialty,
            session_id=session_id,
            confidence=float(extracted.get("confidence", 0.0)),
            extraction_method=str(extracted.get("extractor", "")),
        )

        blocked_records, queued_records, ddi_findings = self._apply_safety(candidates, session_id)
        if blocked_records:
            audit = self._audit(extracted, "blocked_ddi")
            return ExecutionResult(
                outcome="blocked_ddi",
                blocked_records=blocked_records,
                ddi_findings=ddi_findings,
                extractor_result=extracted,
                audit=audit,
                notes=extracted.get("notes", []),
            )

        if queued_records:
            self._persist_review_queue(queued_records, session_id)
            queued_ids = {record.id for record in queued_records}
            safe_candidates = [record for record in candidates if record.id not in queued_ids]
            written, quality_queued = self.writer.write(safe_candidates, session_id=session_id)
            audit = self._audit(extracted, "queued_for_review")
            return ExecutionResult(
                outcome="queued_for_review",
                records=written,
                queued_records=queued_records + quality_queued,
                ddi_findings=ddi_findings,
                extractor_result=extracted,
                audit=audit,
                notes=extracted.get("notes", []),
            )

        written, queued = self.writer.write(candidates, session_id=session_id)
        outcome = "queued_for_review" if queued else "written"
        audit = self._audit(extracted, outcome)

        return ExecutionResult(
            outcome=outcome,
            records=written,
            queued_records=queued,
            extractor_result=extracted,
            audit=audit,
            notes=extracted.get("notes", []),
        )

    def process_text(
        self,
        text: str,
        *,
        specialty: str = "general",
        source_name: str = "manual",
        session_id: str = "",
    ) -> ExecutionResult:
        return self.run(ExecutionJob(text=text, specialty=specialty, source_name=source_name, session_id=session_id))

    def process_pdf(self, pdf_path: Path, *, specialty: str = "general", session_id: str = "") -> ExecutionResult:
        return self.run(ExecutionJob(pdf_path=pdf_path, specialty=specialty, source_name=pdf_path.name, session_id=session_id))

    def extract_pdf_text(self, pdf_path: Path) -> str:
        from extraction.extractor import Extractor
        from ingestion.pdf_pipeline import PDFPipeline

        pdf_pipeline = PDFPipeline(Extractor(), self.pii_stripper)
        return pdf_pipeline._extract_text(pdf_path)

    def _select_extractor(self, text: str, specialty: str):
        if len(text) < 1500 and not self._has_ocr_artifacts(text):
            return self.spacy_extractor
        if self.gemini_extractor is None or self.gemini_extractor.specialty != specialty:
            self.gemini_extractor = GeminiExtractor(specialty=specialty)
        return self.gemini_extractor

    def _strip_pii(self, text: str) -> tuple[str, str]:
        if not text:
            return "", "none"
        return self.pii_stripper.strip(text)

    def _has_ocr_artifacts(self, text: str) -> bool:
        if OCR_ARTIFACT_RE.search(text):
            return True
        if not text:
            return False
        non_alnum = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return (non_alnum / max(len(text), 1)) > 0.18

    def _validate_extractor_output(self, extracted: dict) -> None:
        missing = EXTRACTOR_SCHEMA_KEYS - set(extracted)
        if missing:
            raise ValueError(f"Extractor output missing keys: {sorted(missing)}")
        if not isinstance(extracted["extractor"], str):
            raise TypeError("Extractor output 'extractor' must be a string")
        if not isinstance(extracted["entities"], list):
            raise TypeError("Extractor output 'entities' must be a list")
        if not isinstance(extracted["confidence"], (int, float)):
            raise TypeError("Extractor output 'confidence' must be numeric")
        if not isinstance(extracted["latency_ms"], int):
            raise TypeError("Extractor output 'latency_ms' must be an int")
        if not isinstance(extracted["raw_text"], str):
            raise TypeError("Extractor output 'raw_text' must be a string")
        if not isinstance(extracted["notes"], list):
            raise TypeError("Extractor output 'notes' must be a list")

    def _entities_to_records(
        self,
        entities: list[dict],
        *,
        source_name: str,
        specialty: str,
        session_id: str,
        confidence: float,
        extraction_method: str,
    ) -> list[MKBRecord]:
        records: list[MKBRecord] = []
        for entity in entities:
            fact_type = self._normalize_fact_type(str(entity.get("type", "note")))
            text = str(entity.get("text", "")).strip()
            if not text:
                continue
            structured = dict(entity.get("structured") or entity)
            structured.pop("type", None)
            structured.pop("text", None)

            records.append(MKBRecord(
                fact_type=fact_type,
                content=self._content_for_entity(fact_type, text, structured),
                structured={"name": text, **structured} if fact_type in {"diagnosis", "medication"} else {"text": text, **structured},
                specialty=specialty,
                source_type="document",
                source_name=source_name,
                trust_level=TRUST_CLINICAL,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method or "unknown",
                ddi_checked=False,
                session_id=session_id,
                tags=[fact_type],
            ))
        return records

    def _apply_safety(self, records: list[MKBRecord], session_id: str) -> tuple[list[MKBRecord], list[MKBRecord], list[dict]]:
        blocked: list[MKBRecord] = []
        queued: list[MKBRecord] = []
        findings: list[dict] = []

        for record in records:
            decision, message, record_findings = self.safety.check_medication(record, session_id=session_id)
            finding_dicts = [self._finding_to_dict(item) for item in record_findings or []]
            findings.extend(finding_dicts)

            if decision == "block":
                record.ddi_status = "high_blocked"
                record.ddi_findings = finding_dicts
                record.structured["ddi_message"] = message
                blocked.append(record)
            elif decision == "queue":
                queued_record = record.model_copy(update={
                    "tier": TIER_QUARANTINED,
                    "status": "pending_ddi",
                    "requires_review": True,
                    "ddi_status": "pending",
                    "ddi_findings": finding_dicts,
                    "structured": {**record.structured, "ddi_message": message},
                })
                queued.append(queued_record)

        return blocked, queued, findings

    def _persist_review_queue(self, records: list[MKBRecord], session_id: str) -> None:
        if self.writer.sql_store is None:
            return
        for record in records:
            self.writer.sql_store.write_record(record, session_id=session_id)

    def _audit(self, extracted: dict, outcome: str) -> dict:
        return self.audit_logger.log(
            extractor=extracted.get("extractor", ""),
            entity_count=len(extracted.get("entities", [])),
            confidence=float(extracted.get("confidence", 0.0)),
            outcome=outcome,
        )

    def _finding_to_dict(self, finding) -> dict:
        if hasattr(finding, "model_dump"):
            return finding.model_dump()
        if isinstance(finding, dict):
            return finding
        return {"finding": str(finding)}

    def _content_for_entity(self, fact_type: str, text: str, structured: dict) -> str:
        if fact_type == "diagnosis":
            return f"Diagnosis: {text}"
        if fact_type == "medication":
            dose = structured.get("dose") or ""
            frequency = structured.get("frequency") or ""
            suffix = " ".join(part for part in (dose, frequency) if part)
            return f"Medication: {text}" + (f" {suffix}" if suffix else "")
        if fact_type == "test_result":
            value = structured.get("value")
            unit = structured.get("unit") or ""
            return f"Test: {text}" + (f": {value} {unit}".rstrip() if value else "")
        if fact_type == "symptom":
            return f"Symptom: {text}"
        if fact_type == "recommendation":
            return f"Recommendation: {text}"
        return text[:500]

    def _normalize_fact_type(self, value: str) -> str:
        if value in {"diagnosis", "medication", "test_result", "symptom", "note", "recommendation"}:
            return value
        return "note"

    def _build_pii_stripper(self):
        try:
            from ingestion.pii_handler import PIIHandler

            return PIIHandler()
        except Exception:
            from extraction.pii_stripper import PIIStripper

            return PIIStripper()
