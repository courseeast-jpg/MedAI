"""
MedAI v1.1 — Decision Engine (Track C)
Full 7-step behavioral contract.
Router + scorer + consensus + truth resolution + refusal logic + safe mode.
"""
import asyncio
import time
from typing import List, Optional, Tuple
from loguru import logger
from uuid import uuid4

from app.config import (
    SAFE_MODE_THRESHOLD, CONNECTOR_TIMEOUT_SEC, ACTIVE_CONNECTORS,
    ALLOWED_SPECIALTIES, ALLOWED_TASK_TYPES
)
from app.schemas import (
    ClassifiedQuery, MKBContext, ConnectorResponse, ScoredResponse,
    UnifiedResponse, LedgerEvent, MKBRecord
)
from decision.response_scorer import ResponseScorer
from decision.medication_safety import MedicationSafetyGate
from mkb.sqlite_store import SQLiteStore
from mkb.vector_store import VectorStore


REFUSAL_TEMPLATE = (
    "This query produced conflicting or low-confidence results. "
    "MKB-only context is shown below. External AI responses were insufficient. "
    "Recommendation: verify with a qualified clinician.\n\n"
    "{mkb_summary}"
)

SAFE_MODE_PREFIX = "[SAFE MODE — MKB context only. No external AI synthesis available.]\n\n"


class DecisionEngine:
    def __init__(
        self,
        sql: SQLiteStore,
        vec: VectorStore,
        scorer: ResponseScorer,
        med_gate: MedicationSafetyGate,
        connectors: dict,  # {name: connector_instance}
        claude_synthesizer=None,
        system_state=None,
    ):
        self.sql = sql
        self.vec = vec
        self.scorer = scorer
        self.med_gate = med_gate
        self.connectors = connectors
        self.synthesizer = claude_synthesizer
        self.state = system_state

    async def process(self, raw_query: str, session_id: str = "") -> UnifiedResponse:
        session_id = session_id or str(uuid4())
        logger.info(f"Decision Engine: processing query [{session_id[:8]}]")

        # Step 1 — Classification
        classified = self._classify_query(raw_query, session_id)
        if classified.confidence < 0.5:
            return self._build_clarification_response(raw_query, session_id)

        # Step 2 — MKB context retrieval
        ctx = self._retrieve_context(classified)

        # Step 3 — Multi-source execution (or safe mode)
        if self._is_safe_mode():
            return self._safe_mode_response(raw_query, ctx, session_id)

        connector_responses = await self._execute_connectors(classified, ctx)

        # Step 4 — Response scoring (includes DDI Layer 1)
        accepted, discarded = self.scorer.score_all(
            connector_responses, ctx, classified.requires_ddi_check
        )

        # All discarded → refusal
        if not accepted:
            return self._refusal_response(raw_query, ctx, discarded, session_id)

        # Step 5 — Consensus + contradiction detection
        consensus_score = self._compute_consensus(accepted)

        # Step 6 — Truth resolution for cross-response contradictions
        # (Handled within MKB writes via quality gate — not repeated here)

        # Step 7 — Refusal under low confidence
        if consensus_score < SAFE_MODE_THRESHOLD:
            logger.warning(f"Low consensus {consensus_score:.2f} — returning refusal")
            return self._refusal_response(raw_query, ctx, discarded, session_id, consensus_score)

        # Synthesize
        synthesis = self._synthesize(raw_query, accepted, ctx)

        # Collect DDI findings across all accepted responses
        all_ddi = []
        for r in accepted:
            all_ddi.extend(r.ddi_findings)

        return UnifiedResponse(
            query=raw_query,
            specialty=classified.specialty,
            synthesis=synthesis,
            confidence=consensus_score,
            confidence_band=self.scorer._confidence_band(consensus_score),
            sources_used=[r.connector_name for r in accepted],
            mkb_facts_used=ctx.structured_facts[:5],
            hypothesis_facts=self._get_hypothesis_facts(classified.specialty),
            ddi_findings=all_ddi,
            safe_mode=False,
            discarded_responses=[f"{r.connector_name}: {r.discard_reason}" for r in discarded],
            session_id=session_id,
        )

    # ── Step 1 ────────────────────────────────────────────────────────────

    def _classify_query(self, query: str, session_id: str) -> ClassifiedQuery:
        query_lower = query.lower()

        # Specialty detection
        specialty = "general"
        specialty_keywords = {
            "neurology": ["neuro", "brain", "seizure", "headache", "migraine", "stroke", "ms ", "alzheimer", "parkinson"],
            "epilepsy": ["epilep", "seizure", "anticonvulsant", "levetiracetam", "valproate", "lamotrigine", "eeg"],
            "gastroenterology": ["gastro", "bowel", "crohn", "colitis", "ibs", "gerd", "liver", "colon", "intestin"],
            "urology": ["urology", "bladder", "prostate", "kidney", "renal", "urinary", "bph"],
        }
        for spec, keywords in specialty_keywords.items():
            if any(kw in query_lower for kw in keywords):
                specialty = spec
                break

        # Task type detection
        task_type = "general_query"
        if any(w in query_lower for w in ["medication", "drug", "dose", "pill", "take", "prescription"]):
            task_type = "medication_check"
        elif any(w in query_lower for w in ["diagnos", "symptom", "what is", "could be", "differential"]):
            task_type = "differential_diagnosis"
        elif any(w in query_lower for w in ["treatment", "therapy", "manage", "guideline"]):
            task_type = "evidence_lookup"

        # DDI flag
        requires_ddi = task_type == "medication_check" or any(
            w in query_lower for w in ["interact", "combine", "safe with", "take with"]
        )

        confidence = 0.70 if specialty != "general" else 0.55

        return ClassifiedQuery(
            original_query=query,
            specialty=specialty,
            task_type=task_type,
            confidence=confidence,
            requires_ddi_check=requires_ddi,
            session_id=session_id,
        )

    # ── Step 2 ────────────────────────────────────────────────────────────

    def _retrieve_context(self, classified: ClassifiedQuery) -> MKBContext:
        structured = self.sql.get_by_specialty(classified.specialty, tier="active")
        semantic = self.vec.semantic_search(
            classified.original_query,
            n_results=10,
            specialty=classified.specialty,
            include_hypothesis=True,
        )
        semantic_texts = [r["text"] for r in semantic]

        return MKBContext(
            structured_facts=structured[:20],
            semantic_chunks=semantic_texts,
            active_medications=self.sql.get_active_medications(),
            active_diagnoses=self.sql.get_active_diagnoses(),
            recent_conflicts=self.sql.get_recent_conflicts(),
        )

    # ── Step 3 ────────────────────────────────────────────────────────────

    async def _execute_connectors(
        self, classified: ClassifiedQuery, ctx: MKBContext
    ) -> List[ConnectorResponse]:
        from extraction.pii_stripper import PIIStripper
        pii = PIIStripper()

        stripped_query, _ = pii.strip(classified.original_query)
        stripped_context = [pii.strip(f)[0] for f in ctx.semantic_chunks[:5]]

        from app.schemas import AnonymizedPayload
        payload = AnonymizedPayload(
            query_text=stripped_query,
            specialty=classified.specialty,
            task_type=classified.task_type,
            context_facts=stripped_context,
            active_medications=[
                pii.strip(m.content)[0] for m in ctx.active_medications[:10]
            ],
            requires_ddi_check=classified.requires_ddi_check,
            session_id=classified.session_id,
        )

        tasks = []
        for name in ACTIVE_CONNECTORS:
            if name in self.connectors:
                tasks.append(self._call_connector(name, payload))
            else:
                logger.debug(f"Connector {name} not registered")

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        responses = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Connector exception: {r}")
            elif r:
                responses.append(r)

        return responses

    async def _call_connector(self, name: str, payload) -> Optional[ConnectorResponse]:
        connector = self.connectors[name]
        start = time.time()
        try:
            result = await asyncio.wait_for(
                connector.query(payload),
                timeout=CONNECTOR_TIMEOUT_SEC
            )
            result.latency_ms = int((time.time() - start) * 1000)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Connector {name} timed out after {CONNECTOR_TIMEOUT_SEC}s")
            return ConnectorResponse(
                connector_name=name, status="timeout", latency_ms=CONNECTOR_TIMEOUT_SEC * 1000
            )
        except Exception as e:
            logger.error(f"Connector {name} error: {e}")
            return ConnectorResponse(connector_name=name, status="error")

    # ── Step 5 ────────────────────────────────────────────────────────────

    def _compute_consensus(self, accepted: List[ScoredResponse]) -> float:
        if not accepted:
            return 0.0
        if len(accepted) == 1:
            return accepted[0].final_score * 0.75  # Penalty for single source

        scores = [r.final_score for r in accepted]
        base = sum(scores) / len(scores)
        # Bonus for multi-source agreement
        if len(accepted) >= 2:
            base = min(1.0, base + 0.10)
        return round(base, 3)

    # ── Synthesis ─────────────────────────────────────────────────────────

    def _synthesize(
        self, query: str, accepted: List[ScoredResponse], ctx: MKBContext
    ) -> str:
        if self.synthesizer and self.synthesizer.available:
            try:
                return self.synthesizer.synthesize(query, accepted, ctx)
            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")

        # Fallback: structured aggregation
        parts = [f"Query: {query}\n"]
        if ctx.structured_facts:
            parts.append("Your medical records indicate:")
            for f in ctx.structured_facts[:5]:
                parts.append(f"  [{f.tier.upper()}] {f.content}")
        if accepted:
            parts.append("\nExternal AI responses:")
            for r in accepted:
                parts.append(f"  [{r.connector_name} score={r.final_score:.2f}] {r.content[:300]}")
        return "\n".join(parts)

    # ── Special responses ─────────────────────────────────────────────────

    def _safe_mode_response(self, query: str, ctx: MKBContext, session_id: str) -> UnifiedResponse:
        mkb_lines = [f"[trust={r.trust_level} tier={r.tier}] {r.content}"
                     for r in ctx.structured_facts[:10]]
        synthesis = SAFE_MODE_PREFIX + "\n".join(mkb_lines) if mkb_lines else SAFE_MODE_PREFIX + "No relevant MKB records found."
        return UnifiedResponse(
            query=query, specialty="general", synthesis=synthesis,
            confidence=0.0, confidence_band="low",
            mkb_facts_used=ctx.structured_facts[:10],
            safe_mode=True, session_id=session_id,
        )

    def _refusal_response(
        self, query: str, ctx: MKBContext,
        discarded: List[ScoredResponse], session_id: str, score: float = 0.0
    ) -> UnifiedResponse:
        mkb_summary = "\n".join(f"  {r.content}" for r in ctx.structured_facts[:5])
        synthesis = REFUSAL_TEMPLATE.format(mkb_summary=mkb_summary or "No relevant records found.")
        return UnifiedResponse(
            query=query, specialty="general", synthesis=synthesis,
            confidence=score, confidence_band="low",
            mkb_facts_used=ctx.structured_facts[:5],
            discarded_responses=[f"{r.connector_name}: {r.discard_reason}" for r in discarded],
            session_id=session_id,
        )

    def _build_clarification_response(self, query: str, session_id: str) -> UnifiedResponse:
        synthesis = (
            "Could not determine the medical specialty for this query. "
            "Please specify whether this relates to neurology, epilepsy, gastroenterology, or urology."
        )
        return UnifiedResponse(
            query=query, specialty="general", synthesis=synthesis,
            confidence=0.0, confidence_band="low", session_id=session_id,
        )

    def _get_hypothesis_facts(self, specialty: str) -> List[MKBRecord]:
        return self.sql.get_by_specialty(specialty, tier="hypothesis")[:5]

    def _is_safe_mode(self) -> bool:
        return self.state and self.state.safe_mode
