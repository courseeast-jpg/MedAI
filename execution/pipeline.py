"""Phase 1 deterministic execution pipeline."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.config import REVIEW_QUEUE_PATH, TIER_ACTIVE, TIER_QUARANTINED, TRUST_CLINICAL
from app.schemas import MKBRecord
from execution.consensus import consensus_merge
from execution.jobs import ExecutionJob, ExecutionResult
from execution.logging import AuditLogger
from execution.mkb_writer import MKBWriter
from execution.safety import ExecutionSafety
from execution.truth_resolution import ResolutionBatch, TruthResolutionResolver
from execution.validation import ValidationDecision, validate_extraction_result
from extractors.gemini_extractor import GeminiExtractor
from extractors.spacy_extractor import SpacyExtractor


OCR_ARTIFACT_RE = re.compile(r"(\ufffd|[|]{3,}|_{4,}|\b(?:l|I){8,}\b)")
EXTRACTOR_SCHEMA_KEYS = {"extractor", "entities", "confidence", "latency_ms", "raw_text", "notes"}
SPACY_FAST_PATH_CHAR_LIMIT = 3000


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
        existing_records_provider=None,
        review_queue_path: Path | str = REVIEW_QUEUE_PATH,
    ):
        self.pii_stripper = pii_stripper or self._build_pii_stripper()
        self.audit_logger = audit_logger or AuditLogger()
        self.spacy_extractor = spacy_extractor or SpacyExtractor()
        self.gemini_extractor = gemini_extractor
        self.review_queue_path = Path(review_queue_path)
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = MKBWriter(sql_store, vector_store, quality_gate)
        self.safety = ExecutionSafety(medication_gate)
        self.truth_resolver = TruthResolutionResolver(
            existing_records_provider or self._build_existing_records_provider(sql_store)
        )

    def run(self, job: ExecutionJob) -> ExecutionResult:
        source_text = job.text
        source_name = job.source_name
        session_id = job.session_id or str(uuid4())

        if job.pdf_path is not None:
            source_text = self.extract_pdf_text(job.pdf_path)
            source_name = job.source_name or job.pdf_path.name

        stripped_text, pii_method = self._strip_pii(source_text)
        extractor_route = self._select_extractor_route(stripped_text)
        extraction_results = self._collect_extraction_results(stripped_text, job.specialty, extractor_route)
        extracted = consensus_merge(extraction_results, extractor_route=extractor_route)
        self._validate_extractor_output(extracted)
        extracted.setdefault("actual_extractor", extracted.get("actual_extractor", extracted.get("extractor", "unknown")))
        extracted["notes"] = list(extracted.get("notes", [])) + [f"pii_method={pii_method}"]
        validation = validate_extraction_result(extracted, extractor_route=extractor_route)
        extracted["validation_status"] = validation.status
        extracted["validation_errors"] = validation.errors

        if validation.status != "accepted":
            self._append_review_queue_item(
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
                validation=validation,
            )
            queued_records: list[MKBRecord] = []
            if validation.status == "needs_review":
                queued_records = self._mark_records_for_validation_review(
                    self._entities_to_records(
                        extracted.get("entities", []),
                        source_name=source_name,
                        specialty=job.specialty,
                        session_id=session_id,
                        confidence=float(extracted.get("confidence", 0.0)),
                        extraction_method=str(extracted.get("extractor", "")),
                    ),
                    validation,
                )
                self._persist_review_queue(queued_records, session_id)

            audit = self._audit(extracted, "queued_for_review", extractor_route, validation)
            return ExecutionResult(
                outcome="queued_for_review",
                validation_status=validation.status,
                validation_errors=validation.errors,
                queued_records=queued_records,
                extractor_result=extracted,
                audit=audit,
                notes=extracted.get("notes", []),
            )

        candidates = self._entities_to_records(
            extracted.get("entities", []),
            source_name=source_name,
            specialty=job.specialty,
            session_id=session_id,
            confidence=float(extracted.get("confidence", 0.0)),
            extraction_method=str(extracted.get("extractor", "")),
        )
        resolution = self.truth_resolver.resolve_batch(candidates)
        if resolution.quarantined_records:
            self._persist_review_queue(resolution.quarantined_records, session_id)
            self._append_resolution_review_queue_items(
                resolution=resolution,
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
        candidates = resolution.records_to_write

        blocked_records, queued_records, ddi_findings = self._apply_safety(candidates, session_id)
        if blocked_records:
            audit = self._audit(extracted, "blocked_ddi", extractor_route, validation)
            return ExecutionResult(
                outcome="blocked_ddi",
                validation_status=validation.status,
                validation_errors=validation.errors,
                blocked_records=blocked_records,
                ddi_findings=ddi_findings,
                extractor_result=extracted,
                audit=audit,
                notes=extracted.get("notes", []),
            )

        if queued_records:
            self._persist_review_queue(queued_records, session_id)
            combined_review_records = resolution.quarantined_records + queued_records
            queued_ids = {record.id for record in combined_review_records}
            safe_candidates = [record for record in candidates if record.id not in queued_ids]
            written, quality_queued = self.writer.write(safe_candidates, session_id=session_id)
            audit = self._audit(extracted, "queued_for_review", extractor_route, validation)
            return ExecutionResult(
                outcome="queued_for_review",
                validation_status=validation.status,
                validation_errors=validation.errors,
                records=written,
                queued_records=combined_review_records + quality_queued,
                ddi_findings=ddi_findings,
                extractor_result=extracted,
                audit=audit,
                notes=extracted.get("notes", []),
            )

        written, queued = self.writer.write(candidates, session_id=session_id)
        combined_queued = resolution.quarantined_records + queued
        outcome = "queued_for_review" if combined_queued else "written"
        audit = self._audit(extracted, outcome, extractor_route, validation)

        return ExecutionResult(
            outcome=outcome,
            validation_status=validation.status,
            validation_errors=validation.errors,
            records=written,
            queued_records=combined_queued,
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

    def _select_extractor_route(self, text: str) -> str:
        if len(text) < SPACY_FAST_PATH_CHAR_LIMIT and not self._has_ocr_artifacts(text):
            return "spacy"
        return "gemini"

    def _collect_extraction_results(self, text: str, specialty: str, extractor_route: str) -> list[dict]:
        results: list[dict] = []
        spacy_result = self.spacy_extractor.extract(text)
        self._validate_extractor_output(spacy_result)
        spacy_result.setdefault("actual_extractor", spacy_result.get("extractor", "unknown"))
        include_spacy = extractor_route == "spacy" or bool(spacy_result.get("entities"))
        if include_spacy:
            results.append(spacy_result)

        if extractor_route == "gemini":
            gemini_extractor = self._get_gemini_extractor(specialty)
            gemini_result = gemini_extractor.extract(text)
            self._validate_extractor_output(gemini_result)
            gemini_result.setdefault("actual_extractor", gemini_result.get("extractor", "unknown"))
            results.append(gemini_result)

        return results

    def _get_gemini_extractor(self, specialty: str):
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

    def _append_review_queue_item(
        self,
        *,
        source_name: str,
        specialty: str,
        session_id: str,
        extractor_route: str,
        extracted: dict,
        validation: ValidationDecision,
    ) -> None:
        item = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_name": source_name,
            "specialty": specialty,
            "session_id": session_id,
            "extractor_route": extractor_route,
            "extractor_actual": str(extracted.get("actual_extractor", extracted.get("extractor", ""))),
            "extractor": str(extracted.get("extractor", "")),
            "validation_status": validation.status,
            "validation_errors": validation.errors,
            "reasons": [error["code"] for error in validation.errors],
            "confidence": float(extracted.get("confidence", 0.0)),
            "entity_count": len(extracted.get("entities", [])),
            "notes": list(extracted.get("notes", [])),
            "raw_text": str(extracted.get("raw_text", "")),
        }
        with self.review_queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    def _append_resolution_review_queue_items(
        self,
        *,
        resolution: ResolutionBatch,
        source_name: str,
        specialty: str,
        session_id: str,
        extractor_route: str,
        extracted: dict,
    ) -> None:
        for decision in resolution.decisions:
            if not decision.requires_review or decision.record is None:
                continue
            item = {
                "timestamp": datetime.utcnow().isoformat(),
                "source_name": source_name,
                "specialty": specialty,
                "session_id": session_id,
                "extractor_route": extractor_route,
                "extractor_actual": str(extracted.get("actual_extractor", extracted.get("extractor", ""))),
                "extractor": str(extracted.get("extractor", "")),
                "validation_status": extracted.get("validation_status", "accepted"),
                "resolution_action": decision.action,
                "resolution_confidence": decision.confidence,
                "requires_review": True,
                "reasons": [decision.reason],
                "record_id": decision.record.id,
                "fact_type": decision.record.fact_type,
                "content": decision.record.content,
                "confidence": float(decision.record.confidence),
            }
            with self.review_queue_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    def _mark_records_for_validation_review(
        self,
        records: list[MKBRecord],
        validation: ValidationDecision,
    ) -> list[MKBRecord]:
        review_codes = [error["code"] for error in validation.errors]
        return [
            record.model_copy(update={
                "tier": TIER_QUARANTINED,
                "status": "pending_validation_review",
                "requires_review": True,
                "structured": {
                    **record.structured,
                    "validation_status": validation.status,
                    "validation_errors": validation.errors,
                    "review_reasons": review_codes,
                },
            })
            for record in records
        ]

    def _audit(self, extracted: dict, outcome: str, extractor_route: str, validation: ValidationDecision) -> dict:
        return self.audit_logger.log(
            extractor=extracted.get("extractor", ""),
            extractor_route=extractor_route,
            extractor_actual=str(extracted.get("actual_extractor", extracted.get("extractor", ""))),
            entity_count=len(extracted.get("entities", [])),
            confidence=float(extracted.get("confidence", 0.0)),
            validation_status=validation.status,
            validation_error_count=validation.error_count,
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

    def _build_existing_records_provider(self, sql_store):
        if sql_store is None:
            return lambda record: []

        def provider(record: MKBRecord) -> list[MKBRecord]:
            if hasattr(sql_store, "get_by_specialty"):
                return [
                    item
                    for item in sql_store.get_by_specialty(record.specialty, tier="active")
                    if item.fact_type == record.fact_type
                ]
            return []

        return provider
