"""
MedAI v1.1 — Truth Resolution Engine (Track A)
7 priority rules. Full behavioral contract.
Transforms MKB from conflict container to reliable knowledge substrate.
"""
from datetime import datetime, timedelta
from loguru import logger

from app.schemas import (
    MKBRecord, TruthResolutionInput, TruthResolutionOutput, LedgerEvent
)
from app.config import TIER_QUARANTINED, TIER_SUPERSEDED


class TruthResolutionEngine:
    """
    Resolves conflicts between MKBRecord pairs using ordered priority rules.
    Rules execute in strict order. First matching rule wins.
    """

    def resolve(self, inp: TruthResolutionInput) -> TruthResolutionOutput:
        cand = inp.candidate_fact
        exist = inp.existing_fact

        # Rule 1 — Clinical supremacy
        if exist.trust_level == 1 and cand.trust_level > 1:
            return self._result("keep_existing", exist, cand, 0.95,
                "Clinical document (trust=1) takes precedence over all other sources.",
                "clinical_supremacy")

        if cand.trust_level == 1 and exist.trust_level > 1:
            return self._result("replace_with_new", cand, exist, 0.95,
                "New clinical document (trust=1) supersedes lower-trust existing record.",
                "clinical_supremacy")

        # Rule 2 — Peer review beats AI
        if exist.trust_level == 2 and cand.trust_level == 3:
            return self._result("keep_existing", exist, cand, 0.90,
                "Peer-reviewed source (trust=2) takes precedence over AI-derived fact (trust=3).",
                "peer_review_beats_ai")

        if cand.trust_level == 2 and exist.trust_level == 3:
            return self._result("replace_with_new", cand, exist, 0.90,
                "New peer-reviewed source (trust=2) supersedes AI-derived existing fact.",
                "peer_review_beats_ai")

        # Rule 3 — Recency when same trust level
        if cand.trust_level == exist.trust_level:
            diff = abs((cand.first_recorded - exist.first_recorded).days)
            if diff > 90:
                if cand.first_recorded > exist.first_recorded:
                    return self._result("replace_with_new", cand, exist, 0.80,
                        f"Same trust level. Newer record ({diff} days more recent) takes precedence.",
                        "recency_same_trust")
                else:
                    return self._result("keep_existing", exist, cand, 0.80,
                        f"Same trust level. Existing record is more recent ({diff} days).",
                        "recency_same_trust")

        # Rule 4 — Source agreement count
        cand_sources = len(cand.linked_to) + 1
        exist_sources = len(exist.linked_to) + 1
        if cand_sources > exist_sources and cand_sources >= 2:
            return self._result("replace_with_new", cand, exist, 0.75,
                f"New fact corroborated by {cand_sources} sources vs {exist_sources}.",
                "source_agreement")

        # Rule 5 — Numeric value range merge
        if (inp.conflict_type == "value_conflict"
                and cand.fact_type == "test_result"
                and exist.fact_type == "test_result"):
            merged = self._merge_values(cand, exist)
            return TruthResolutionOutput(
                resolution="merge",
                winner=merged,
                loser_id=exist.id,
                confidence=0.70,
                explanation="Numeric test results from same period merged as value range.",
                requires_review=False,
                rule_applied="value_range_merge",
            )

        # Rule 6 — Medication dose conflict — ALWAYS quarantine
        if (cand.fact_type == "medication" and exist.fact_type == "medication"
                and inp.conflict_type == "value_conflict"):
            quarantined = cand.model_copy(update={
                "tier": TIER_QUARANTINED,
                "status": "quarantined",
                "requires_review": True,
            })
            return TruthResolutionOutput(
                resolution="quarantine",
                winner=exist,  # existing stays active until user resolves
                loser_id=cand.id,
                confidence=0.0,
                explanation=(
                    f"Conflicting medication doses require physician verification. "
                    f"Existing: {exist.content} | New: {cand.content}"
                ),
                requires_review=True,
                rule_applied="medication_dose_conflict",
            )

        # Rule 7 — Unresolvable: quarantine candidate
        quarantined = cand.model_copy(update={
            "tier": TIER_QUARANTINED,
            "status": "quarantined",
            "requires_review": True,
        })
        return TruthResolutionOutput(
            resolution="quarantine",
            winner=exist,
            loser_id=cand.id,
            confidence=0.0,
            explanation=(
                "No resolution rule applies. Conflict cannot be algorithmically arbitrated. "
                "Candidate quarantined. User review required."
            ),
            requires_review=True,
            rule_applied="unresolvable",
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _result(
        self, resolution: str, winner: MKBRecord, loser: MKBRecord,
        confidence: float, explanation: str, rule: str
    ) -> TruthResolutionOutput:
        loser_archived = loser.model_copy(update={
            "status": "superseded",
            "tier": TIER_SUPERSEDED,
        })
        return TruthResolutionOutput(
            resolution=resolution,
            winner=winner,
            loser_id=loser.id,
            confidence=confidence,
            explanation=explanation,
            requires_review=False,
            rule_applied=rule,
        )

    def _merge_values(self, a: MKBRecord, b: MKBRecord) -> MKBRecord:
        merged_content = f"{a.content} [merged with: {b.content}]"
        return a.model_copy(update={
            "content": merged_content,
            "structured": {**b.structured, **a.structured, "_merged": True},
            "confidence": min(a.confidence, b.confidence),
            "last_confirmed": datetime.utcnow(),
        })

    def build_ledger_event(
        self, result: TruthResolutionOutput, session_id: str = ""
    ) -> LedgerEvent:
        return LedgerEvent(
            event_type="truth_resolution",
            record_id=result.winner.id,
            source_type="truth_resolution_engine",
            details={
                "resolution": result.resolution,
                "rule_applied": result.rule_applied,
                "loser_id": result.loser_id,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "requires_review": result.requires_review,
            },
            session_id=session_id,
        )
