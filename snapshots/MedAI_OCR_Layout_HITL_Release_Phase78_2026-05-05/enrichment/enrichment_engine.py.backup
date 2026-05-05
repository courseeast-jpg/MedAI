"""
MedAI v1.1 — Enrichment Engine (Track B)
Extracts new facts from AI responses → writes as HYPOTHESIS tier.
All AI-derived facts are hypothesis until promoted.
Auto-promotion DISABLED in MVP (ENRICH_PROMOTE=False).
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List
from loguru import logger

from app.config import (
    ENABLE_ENRICHMENT, ENRICH_PROMOTE, PENDING_QUEUE_PATH,
    TIER_HYPOTHESIS, TRUST_AI, TRUST_PEER_REVIEW
)
from app.schemas import MKBRecord, ScoredResponse, LedgerEvent, UnifiedResponse
from extraction.extractor import Extractor
from mkb.sqlite_store import SQLiteStore
from mkb.vector_store import VectorStore
from mkb.quality_gate import QualityGate
from decision.medication_safety import MedicationSafetyGate


class EnrichmentEngine:
    def __init__(
        self,
        extractor: Extractor,
        sql: SQLiteStore,
        vec: VectorStore,
        quality_gate: QualityGate,
        med_gate: MedicationSafetyGate,
    ):
        self.extractor = extractor
        self.sql = sql
        self.vec = vec
        self.gate = quality_gate
        self.med_gate = med_gate
        PENDING_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def enrich_from_response(
        self, response: UnifiedResponse, session_id: str = ""
    ) -> List[MKBRecord]:
        """
        Main enrichment entry point.
        Returns list of records actually written to MKB.
        """
        if not ENABLE_ENRICHMENT:
            logger.debug("Enrichment disabled (ENABLE_ENRICHMENT=False)")
            return []

        if not self.extractor.claude_available:
            logger.warning("Claude unavailable — queuing enrichment for later")
            self._queue_for_later(response.synthesis, response.specialty, session_id)
            return []

        # Extract candidate facts from synthesis content
        extraction = self.extractor.extract(response.synthesis, response.specialty)
        candidates = self._extraction_to_records(
            extraction, response.specialty, response.session_id or session_id
        )

        written = []
        for candidate in candidates:
            result = self._process_candidate(candidate, session_id)
            if result:
                written.append(result)

        if written:
            logger.info(f"Enrichment: {len(written)} hypothesis facts written from AI response")
            self.sql.write_ledger(LedgerEvent(
                event_type="enrichment_batch",
                details={"count": len(written), "session_id": session_id},
                session_id=session_id,
            ))

        return written

    def process_pending_queue(self) -> int:
        """Reprocess queued enrichment items when Claude becomes available."""
        if not PENDING_QUEUE_PATH.exists():
            return 0
        if not self.extractor.claude_available:
            return 0

        processed = 0
        remaining = []

        with open(PENDING_QUEUE_PATH) as f:
            lines = f.readlines()

        for line in lines:
            try:
                item = json.loads(line.strip())
                extraction = self.extractor.extract(item["content"], item.get("specialty", "general"))
                candidates = self._extraction_to_records(extraction, item.get("specialty", "general"), item.get("session_id", ""))
                for c in candidates:
                    if self._process_candidate(c, item.get("session_id", "")):
                        processed += 1
            except Exception as e:
                logger.warning(f"Failed to process queued item: {e}")
                remaining.append(line)

        # Rewrite only unprocessed items
        with open(PENDING_QUEUE_PATH, "w") as f:
            f.writelines(remaining)

        logger.info(f"Pending queue: {processed} items processed, {len(remaining)} remaining")
        return processed

    def _process_candidate(self, candidate: MKBRecord, session_id: str) -> MKBRecord | None:
        """Run candidate through medication gate (if needed) then quality gate."""
        # Medication facts: Layer 2 gate first
        if candidate.fact_type == "medication":
            decision, msg, findings = self.med_gate.gate_medication_write(candidate, session_id)
            if decision == "block":
                logger.warning(f"Enrichment blocked by DDI gate: {msg}")
                # Write as blocked — visible in UI for user review
                candidate.ddi_status = "high_blocked"
                candidate.tier = "quarantined"
                candidate.requires_review = True
                candidate.status = "blocked_ddi"
                self.sql.write_record(candidate, session_id)
                self.sql.write_ledger(LedgerEvent(
                    event_type="ddi_block",
                    record_id=candidate.id,
                    source_type="enrichment",
                    details={"message": msg, "findings": [f.model_dump() for f in findings]},
                    session_id=session_id,
                ))
                return None
            elif decision == "queue":
                self._queue_for_later(candidate.content, candidate.specialty, session_id)
                return None

        # Quality gate
        approved, reason, final_record = self.gate.check(candidate, session_id)
        if not approved:
            logger.debug(f"Enrichment rejected by quality gate: {reason}")
            return None

        # Write to MKB
        self.sql.write_record(final_record, session_id)
        self.vec.add_record(final_record)
        self.sql.write_ledger(LedgerEvent(
            event_type="enrichment_write",
            record_id=final_record.id,
            source_type="ai_response",
            details={"tier": final_record.tier, "fact_type": final_record.fact_type},
            session_id=session_id,
        ))
        return final_record

    def _extraction_to_records(self, extraction, specialty: str, session_id: str) -> List[MKBRecord]:
        records = []
        for diag in extraction.diagnoses:
            records.append(MKBRecord(
                fact_type="diagnosis",
                content=f"AI-suggested diagnosis: {diag.name}",
                structured=diag.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="ai_response",
                source_name="enrichment",
                trust_level=TRUST_AI,
                confidence=extraction.confidence * 0.9,
                tier=TIER_HYPOTHESIS,
                extraction_method=extraction.extraction_method,
                session_id=session_id,
            ))
        for med in extraction.medications:
            records.append(MKBRecord(
                fact_type="medication",
                content=f"AI-suggested medication: {med.name}" + (f" {med.dose}" if med.dose else ""),
                structured=med.model_dump(exclude_none=True),
                specialty=specialty,
                source_type="ai_response",
                source_name="enrichment",
                trust_level=TRUST_AI,
                confidence=extraction.confidence * 0.85,
                tier=TIER_HYPOTHESIS,
                extraction_method=extraction.extraction_method,
                ddi_checked=False,
                session_id=session_id,
                tags=["medication", "hypothesis"],
            ))
        for rec in extraction.recommendations:
            records.append(MKBRecord(
                fact_type="recommendation",
                content=f"AI recommendation: {rec[:300]}",
                structured={"text": rec},
                specialty=specialty,
                source_type="ai_response",
                source_name="enrichment",
                trust_level=TRUST_AI,
                confidence=extraction.confidence * 0.8,
                tier=TIER_HYPOTHESIS,
                extraction_method=extraction.extraction_method,
                session_id=session_id,
            ))
        return records

    def _queue_for_later(self, content: str, specialty: str, session_id: str):
        """Write to pending queue for reprocessing when Claude recovers."""
        entry = json.dumps({
            "content": content[:2000],
            "specialty": specialty,
            "session_id": session_id,
            "queued_at": datetime.utcnow().isoformat(),
        })
        with open(PENDING_QUEUE_PATH, "a") as f:
            f.write(entry + "\n")
        logger.info("Enrichment item queued for Claude recovery")
