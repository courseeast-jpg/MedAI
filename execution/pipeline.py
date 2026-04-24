"""Phase 1 deterministic execution pipeline."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.config import (
    ENABLE_HYPOTHESIS_TIER,
    EXTRACTION_ACCEPT_THRESHOLD,
    EXTRACTION_REVIEW_THRESHOLD,
    REVIEW_QUEUE_PATH,
    TIER_ACTIVE,
    TIER_QUARANTINED,
    TRUST_CLINICAL,
)
from app.schemas import MKBRecord
from execution.audit import StageAuditLogger
from execution.consensus import consensus_merge
from execution.connectors.gemini_connector import GeminiConnector
from execution.connectors.phi3_connector import Phi3Connector
from execution.connectors.spacy_connector import SpacyConnector
from execution.enrichment import ControlledEnrichment
from execution.jobs import ExecutionJob, ExecutionResult
from execution.logging import AuditLogger
from execution.metrics import PipelineMetrics
from execution.mkb_writer import MKBWriter
from execution.promotion import HypothesisPromotion
from execution.router import ExecutionRouter
from execution.safety import ExecutionSafety
from execution.truth_resolution import ResolutionBatch, TruthResolutionResolver
from execution.validation import ValidationDecision, validate_extraction_result
from extractors.gemini_extractor import GeminiExtractor
from extractors.spacy_extractor import SpacyExtractor
from governance.hypothesis_tier import GovernanceHypothesisTier
from governance.truth_resolution import GovernanceTruthResolutionAdapter


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
        phi3_extractor=None,
        existing_records_provider=None,
        active_medications_provider=None,
        enrichment_engine: ControlledEnrichment | None = None,
        promotion_engine: HypothesisPromotion | None = None,
        stage_audit_logger: StageAuditLogger | None = None,
        pipeline_metrics: PipelineMetrics | None = None,
        review_queue_path: Path | str = REVIEW_QUEUE_PATH,
        router: ExecutionRouter | None = None,
    ):
        self.pii_stripper = pii_stripper or self._build_pii_stripper()
        self.audit_logger = audit_logger or AuditLogger()
        self.stage_audit = stage_audit_logger or StageAuditLogger()
        self.metrics = pipeline_metrics or PipelineMetrics()
        self.spacy_extractor = spacy_extractor or SpacyExtractor()
        self.gemini_extractor = gemini_extractor
        self.phi3_extractor = phi3_extractor
        self.review_queue_path = Path(review_queue_path)
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = MKBWriter(sql_store, vector_store, quality_gate)
        self.safety = ExecutionSafety(
            medication_gate,
            active_medications_provider or self._build_active_medications_provider(sql_store),
        )
        self.existing_records_provider = existing_records_provider or self._build_existing_records_provider(sql_store)
        self.truth_resolver = GovernanceTruthResolutionAdapter(self.existing_records_provider)
        self.governance_hypothesis = GovernanceHypothesisTier(enabled=ENABLE_HYPOTHESIS_TIER)
        self.enrichment = enrichment_engine
        self.promoter = promotion_engine or HypothesisPromotion(self.existing_records_provider)
        self.router = router or self._build_router()

    def run(self, job: ExecutionJob) -> ExecutionResult:
        source_text = job.text
        source_name = job.source_name
        session_id = job.session_id or str(uuid4())

        if job.pdf_path is not None:
            source_text = self.extract_pdf_text(job.pdf_path)
            source_name = job.source_name or job.pdf_path.name

        self._stage_log(
            record_id=session_id,
            stage="extraction",
            action="extraction_started",
            confidence=0.0,
            decision_reason=f"source={source_name}",
        )
        stripped_text, pii_method = self._strip_pii(source_text)
        routed = self.router.execute(stripped_text, specialty=job.specialty)
        extractor_route = routed.extractor_route
        extraction_results = routed.results
        for extraction_result in extraction_results:
            self._validate_extractor_output(extraction_result)
        self._stage_log(
            record_id=session_id,
            stage="extraction",
            action="extraction_completed",
            confidence=max(float(item.get("confidence", 0.0)) for item in extraction_results) if extraction_results else 0.0,
            decision_reason=f"collectors={','.join(str(item.get('extractor', 'unknown')) for item in extraction_results)}",
            extra={
                "extractor_route": extractor_route,
                "extractor_actual": routed.extractor_actual,
                "fallback_used": routed.fallback_used,
                "failure_count": routed.failure_count,
                "routing_decision_reason": routed.decision_reason,
                "route_score": routed.route_score,
                "routing_events": routed.events,
            },
        )
        extracted = consensus_merge(extraction_results, extractor_route=extractor_route)
        extracted["actual_extractor"] = routed.extractor_actual
        extracted["fallback_used"] = routed.fallback_used
        extracted["routing_failure_count"] = routed.failure_count
        extracted["routing_events"] = routed.events
        extracted["routing_decision_reason"] = routed.decision_reason
        extracted["routing_route_score"] = routed.route_score
        self._validate_extractor_output(extracted)
        extracted.setdefault("actual_extractor", extracted.get("actual_extractor", extracted.get("extractor", "unknown")))
        extracted["notes"] = list(extracted.get("notes", [])) + [f"pii_method={pii_method}"]
        extracted["notes"].append(f"routing_decision={routed.decision_reason}")
        self.metrics.record_routing(
            extractor_actual=str(extracted.get("actual_extractor", "")),
            fallback_used=routed.fallback_used,
            failure_count=routed.failure_count,
        )
        for extraction_result in extraction_results:
            self.metrics.record_connector_result(
                connector=str(extraction_result.get("actual_extractor", extraction_result.get("extractor", "unknown"))),
                latency_ms=float(extraction_result.get("latency_ms", 0.0)),
                confidence=float(extraction_result.get("confidence", 0.0)),
                success=True,
            )
        self._stage_log(
            record_id=session_id,
            stage="consensus",
            action="consensus_result",
            confidence=float(extracted.get("confidence", 0.0)),
            decision_reason=(
                f"agreement_score={float(extracted.get('agreement_score', 1.0))} "
                f"routing={routed.decision_reason}"
            ),
        )
        validation = validate_extraction_result(extracted, extractor_route=extractor_route)
        extracted["validation_status"] = validation.status
        extracted["validation_errors"] = validation.errors
        self.metrics.record_validation(
            record_count=len(extracted.get("entities", [])),
            validation_status=validation.status,
            confidence=float(extracted.get("confidence", 0.0)),
            agreement_score=float(extracted.get("agreement_score", 1.0)),
        )
        self._stage_log(
            record_id=session_id,
            stage="validation",
            action="validation_result",
            confidence=float(extracted.get("confidence", 0.0)),
            decision_reason=validation.status if not validation.errors else ",".join(error["code"] for error in validation.errors),
        )

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
                for record in queued_records:
                    self._stage_log(
                        record_id=record.id,
                        stage="final_write",
                        action="review_queue_write",
                        confidence=float(record.confidence),
                        decision_reason="validation_needs_review",
                    )

            audit = self._audit(
                extracted,
                "queued_for_review",
                extractor_route,
                validation,
                run_id=session_id,
                document_id=source_name,
            )
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
        for decision in resolution.decisions:
            self._stage_log(
                record_id=decision.record.id if decision.record is not None else session_id,
                stage="truth_resolution",
                action="truth_resolution_action",
                confidence=float(decision.confidence),
                decision_reason=decision.reason or decision.action,
            )
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
            for record in resolution.quarantined_records:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason="truth_resolution_quarantine",
                )
        candidates = resolution.records_to_write

        blocked_records, queued_records, ddi_findings = self._apply_safety(candidates, session_id, audit_context=extracted)
        if blocked_records:
            self._append_medication_review_queue_items(
                records=blocked_records,
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
            for record in blocked_records:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason="medication_high_block",
                )
            audit = self._audit(
                extracted,
                "blocked_ddi",
                extractor_route,
                validation,
                run_id=session_id,
                document_id=source_name,
            )
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
            self._append_medication_review_queue_items(
                records=[record for record in queued_records if record.fact_type == "medication"],
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
            self.metrics.record_review(review_count=len(queued_records))
            combined_review_records = resolution.quarantined_records + queued_records
            queued_ids = {record.id for record in combined_review_records}
            safe_candidates = [record for record in candidates if record.id not in queued_ids]
            written, quality_queued = self.writer.write(safe_candidates, session_id=session_id)
            for record in queued_records + quality_queued:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason=record.status or "review_required",
                )
            for record in written:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="final_write",
                    confidence=float(record.confidence),
                    decision_reason=record.resolution_action or "written",
                )
            audit = self._audit(
                extracted,
                "queued_for_review",
                extractor_route,
                validation,
                run_id=session_id,
                document_id=source_name,
            )
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

        enrichment_records: list[MKBRecord] = []
        if self.enrichment is not None:
            enrichment_records = self.enrichment.enrich(candidates)
        for record in enrichment_records:
            self._stage_log(
                record_id=record.id,
                stage="enrichment",
                action="enrichment_write",
                confidence=float(record.enrichment_confidence or record.confidence),
                decision_reason=record.content,
            )
        enrichment_blocked, enrichment_queued, _ = self._apply_safety(enrichment_records, session_id, audit_context=extracted)
        enrichment_review_records = enrichment_blocked + enrichment_queued
        if enrichment_review_records:
            self._persist_review_queue(enrichment_review_records, session_id)
            self._append_medication_review_queue_items(
                records=[record for record in enrichment_review_records if record.fact_type == "medication"],
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
            self.metrics.record_review(review_count=len(enrichment_review_records))
            for record in enrichment_review_records:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason=record.ddi_status or "enrichment_review",
                )

        enrichment_review_ids = {record.id for record in enrichment_review_records}
        safe_enrichment = [record for record in enrichment_records if record.id not in enrichment_review_ids]

        promotion_batch = self.promoter.promote(safe_enrichment, corroborating_records=candidates)
        self.metrics.record_promotion(promoted_count=len(promotion_batch.promoted_records))
        for record in promotion_batch.promoted_records:
            self._stage_log(
                record_id=record.id,
                stage="promotion",
                action="promotion_event",
                confidence=float(record.confidence),
                decision_reason="promoted_to_active",
            )
        promotion_resolver = TruthResolutionResolver(
            lambda record: list(self.existing_records_provider(record)) + candidates
        )
        promoted_resolution = promotion_resolver.resolve_batch(promotion_batch.promoted_records)
        for decision in promoted_resolution.decisions:
            self._stage_log(
                record_id=decision.record.id if decision.record is not None else session_id,
                stage="truth_resolution",
                action="truth_resolution_action",
                confidence=float(decision.confidence),
                decision_reason=decision.reason or decision.action,
            )
        if promoted_resolution.quarantined_records:
            self._append_resolution_review_queue_items(
                resolution=promoted_resolution,
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
            self.metrics.record_review(review_count=len(promoted_resolution.quarantined_records))
            for record in promoted_resolution.quarantined_records:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason="promotion_conflict",
                )

        promoted_blocked, promoted_queued, promoted_findings = self._apply_safety(
            promoted_resolution.records_to_write,
            session_id,
            audit_context=extracted,
        )
        ddi_findings.extend(promoted_findings)
        promoted_review_records = promoted_resolution.quarantined_records + promoted_blocked + promoted_queued
        if promoted_review_records:
            self._persist_review_queue(promoted_review_records, session_id)
            self._append_medication_review_queue_items(
                records=[record for record in promoted_review_records if record.fact_type == "medication"],
                source_name=source_name,
                specialty=job.specialty,
                session_id=session_id,
                extractor_route=extractor_route,
                extracted=extracted,
            )
            self.metrics.record_review(review_count=len(promoted_review_records))
            for record in promoted_review_records:
                self._stage_log(
                    record_id=record.id,
                    stage="final_write",
                    action="review_queue_write",
                    confidence=float(record.confidence),
                    decision_reason=record.ddi_status or "promotion_review",
                )

        promoted_review_ids = {record.id for record in promoted_review_records}
        safe_promoted = [record for record in promoted_resolution.records_to_write if record.id not in promoted_review_ids]
        writable_records = self._dedupe_records_by_id(candidates + safe_promoted + promotion_batch.remaining_hypotheses)
        written, queued = self.writer.write(writable_records, session_id=session_id)
        combined_queued = resolution.quarantined_records + enrichment_review_records + promoted_review_records + queued
        self.metrics.record_review(review_count=len(queued))
        for record in queued:
            self._stage_log(
                record_id=record.id,
                stage="final_write",
                action="review_queue_write",
                confidence=float(record.confidence),
                decision_reason=record.status or "writer_queue",
            )
        for record in written:
            self._stage_log(
                record_id=record.id,
                stage="final_write",
                action="final_write",
                confidence=float(record.confidence),
                decision_reason=record.resolution_action or record.source_type,
            )
        outcome = "queued_for_review" if combined_queued else "written"
        audit = self._audit(
            extracted,
            outcome,
            extractor_route,
            validation,
            run_id=session_id,
            document_id=source_name,
        )

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
        return self.router.select_route(text)

    def _collect_extraction_results(self, text: str, specialty: str, extractor_route: str) -> list[dict]:
        routed = self.router.execute(text, specialty=specialty)
        if routed.extractor_route != extractor_route:
            raise ValueError(f"Router route mismatch: expected {extractor_route}, got {routed.extractor_route}")
        return routed.results

    def _get_gemini_extractor(self, specialty: str):
        if self.gemini_extractor is None or self.gemini_extractor.specialty != specialty:
            self.gemini_extractor = GeminiExtractor(specialty=specialty)
        return self.gemini_extractor

    def _get_gemini_connector(self, specialty: str) -> GeminiConnector:
        return GeminiConnector(self._get_gemini_extractor(specialty))

    def _build_router(self) -> ExecutionRouter:
        return ExecutionRouter(
            spacy_connector=SpacyConnector(self.spacy_extractor),
            gemini_connector_factory=self._get_gemini_connector,
            phi3_connector=Phi3Connector(self.phi3_extractor),
            metrics=self.metrics,
            spacy_fast_path_char_limit=SPACY_FAST_PATH_CHAR_LIMIT,
        )

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

            record = MKBRecord(
                fact_type=fact_type,
                content=self._content_for_entity(fact_type, text, structured),
                structured={"name": text, **structured} if fact_type in {"diagnosis", "medication"} else {"text": text, **structured},
                specialty=specialty,
                source_type="extraction",
                source_name=source_name,
                trust_level=TRUST_CLINICAL,
                confidence=confidence,
                tier=TIER_ACTIVE,
                extraction_method=extraction_method or "unknown",
                ddi_checked=False,
                session_id=session_id,
                tags=[fact_type],
            )
            records.append(self.governance_hypothesis.classify_record(record))
        return records

    def _apply_safety(
        self,
        records: list[MKBRecord],
        session_id: str,
        *,
        audit_context: dict | None = None,
    ) -> tuple[list[MKBRecord], list[MKBRecord], list[dict]]:
        blocked: list[MKBRecord] = []
        queued: list[MKBRecord] = []
        findings: list[dict] = []

        for record in records:
            decision, message, record_findings = self.safety.check_medication(record, session_id=session_id)
            finding_dicts = [self._finding_to_dict(item) for item in record_findings or []]
            findings.extend(finding_dicts)
            self._stage_log(
                record_id=record.id,
                stage="safety_gate",
                action="safety_gate_action",
                confidence=float(record.confidence),
                decision_reason=message or decision,
                extra={
                    "ddi_status": record.ddi_status,
                    "safety_action": record.safety_action,
                    "fact_type": record.fact_type,
                    "extractor": str((audit_context or {}).get("extractor", "")),
                },
            )

            if decision == "block":
                record.ddi_status = "high_blocked"
                record.ddi_findings = finding_dicts
                record.structured["ddi_message"] = message
                blocked.append(record)
            elif decision in {"queue", "review", "pending"}:
                status = "pending_ddi"
                if decision == "review":
                    status = "pending_medication_review"
                elif decision == "pending":
                    status = "pending_ddi_check"
                queued_record = record.model_copy(update={
                    "tier": TIER_QUARANTINED,
                    "status": status,
                    "requires_review": True,
                    "ddi_status": record.ddi_status or ("pending_ddi_check" if decision == "pending" else "medium"),
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

    def _append_medication_review_queue_items(
        self,
        *,
        records: list[MKBRecord],
        source_name: str,
        specialty: str,
        session_id: str,
        extractor_route: str,
        extracted: dict,
    ) -> None:
        for record in records:
            if record.fact_type != "medication":
                continue
            reasons = []
            ddi_note = record.structured.get("ddi_note") or record.structured.get("ddi_message")
            if ddi_note:
                reasons.append(str(ddi_note))
            if not reasons:
                reasons.append(record.ddi_status or "medication_review")
            item = {
                "timestamp": datetime.utcnow().isoformat(),
                "source_name": source_name,
                "specialty": specialty,
                "session_id": session_id,
                "extractor_route": extractor_route,
                "extractor_actual": str(extracted.get("actual_extractor", extracted.get("extractor", ""))),
                "extractor": str(extracted.get("extractor", "")),
                "validation_status": extracted.get("validation_status", "accepted"),
                "record_id": record.id,
                "fact_type": record.fact_type,
                "content": record.content,
                "ddi_status": record.ddi_status,
                "ddi_findings": record.ddi_findings,
                "safety_action": record.safety_action,
                "requires_review": record.requires_review,
                "reasons": reasons,
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

    def _audit(
        self,
        extracted: dict,
        outcome: str,
        extractor_route: str,
        validation: ValidationDecision,
        *,
        run_id: str,
        document_id: str,
    ) -> dict:
        return self.audit_logger.log(
            extractor=extracted.get("extractor", ""),
            extractor_route=extractor_route,
            extractor_actual=str(extracted.get("actual_extractor", extracted.get("extractor", ""))),
            entity_count=len(extracted.get("entities", [])),
            confidence=float(extracted.get("confidence", 0.0)),
            validation_status=validation.status,
            validation_error_count=validation.error_count,
            fallback_used=bool(extracted.get("fallback_used", False)),
            failure_count=int(extracted.get("routing_failure_count", 0)),
            outcome=outcome,
            run_id=run_id,
            document_id=document_id,
            fallback_reason=self._fallback_reason(extracted),
            confidence_band=self._confidence_band(float(extracted.get("confidence", 0.0))),
            quality_gate_decision=self._quality_gate_decision(outcome, validation),
            error_category=self._error_category(outcome, validation, extracted),
        )

    def _fallback_reason(self, extracted: dict) -> str | None:
        for event in extracted.get("routing_events", []):
            if event.get("action") == "fallback_invoked":
                return str(event.get("reason", "fallback_invoked"))
        return None

    def _confidence_band(self, confidence: float) -> str:
        if confidence < EXTRACTION_REVIEW_THRESHOLD:
            return "reject"
        if confidence < EXTRACTION_ACCEPT_THRESHOLD:
            return "review"
        return "auto_accept"

    def _quality_gate_decision(self, outcome: str, validation: ValidationDecision) -> str:
        if outcome == "written":
            return "accepted"
        if outcome == "blocked_ddi":
            return "blocked"
        if validation.status == "needs_review":
            return "review"
        if validation.status == "rejected":
            return "rejected"
        return outcome

    def _error_category(self, outcome: str, validation: ValidationDecision, extracted: dict) -> str | None:
        if validation.errors:
            return str(validation.errors[0].get("code", "validation_error"))
        if outcome == "blocked_ddi":
            return "safety_gate_block"
        if int(extracted.get("routing_failure_count", 0)) > 0:
            return "connector_failure"
        return None

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

    def _build_active_medications_provider(self, sql_store):
        if sql_store is None or not hasattr(sql_store, "get_active_medications"):
            return lambda: []
        return lambda: list(sql_store.get_active_medications())

    def _dedupe_records_by_id(self, records: list[MKBRecord]) -> list[MKBRecord]:
        deduped: dict[str, MKBRecord] = {}
        for record in records:
            deduped[record.id] = record
        return list(deduped.values())

    def _stage_log(
        self,
        *,
        record_id: str,
        stage: str,
        action: str,
        confidence: float,
        decision_reason: str,
        extra: dict | None = None,
    ) -> None:
        self.stage_audit.log(
            record_id=record_id,
            stage=stage,
            action=action,
            confidence=confidence,
            decision_reason=decision_reason,
            extra=extra,
        )
