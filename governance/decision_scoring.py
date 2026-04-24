from __future__ import annotations

from dataclasses import dataclass

from app.config import ENABLE_DECISION_SCORING
from app.schemas import MKBRecord


@dataclass(frozen=True)
class GovernanceDecisionScore:
    enabled: bool
    final_score: float
    score_breakdown: dict[str, float]


class GovernanceDecisionScoring:
    """Deterministic governance scoring wrapper for decision support."""

    def __init__(self, *, enabled: bool = ENABLE_DECISION_SCORING):
        self.enabled = enabled

    def score(
        self,
        *,
        content: str,
        mkb_context: list[MKBRecord] | None = None,
        citations: list[str] | None = None,
        ddi_safety_score: float = 1.0,
    ) -> GovernanceDecisionScore:
        if not self.enabled:
            return GovernanceDecisionScore(
                enabled=False,
                final_score=0.0,
                score_breakdown={},
            )

        mkb_context = mkb_context or []
        citations = citations or []
        mkb_score = self._mkb_consistency(content, mkb_context)
        coherence_score = self._internal_coherence(content)
        citation_score = 1.0 if citations else 0.2
        ddi_score = max(0.0, min(ddi_safety_score, 1.0))
        final = round(
            (mkb_score * 0.35)
            + (coherence_score * 0.25)
            + (citation_score * 0.20)
            + (ddi_score * 0.20),
            3,
        )
        return GovernanceDecisionScore(
            enabled=True,
            final_score=final,
            score_breakdown={
                "mkb_consistency": round(mkb_score, 3),
                "internal_coherence": round(coherence_score, 3),
                "citation_presence": round(citation_score, 3),
                "ddi_safety_score": round(ddi_score, 3),
            },
        )

    def _mkb_consistency(self, content: str, mkb_context: list[MKBRecord]) -> float:
        if not mkb_context:
            return 0.5
        lowered = content.lower()
        matches = sum(
            1 for record in mkb_context
            if (record.structured.get("name") or record.structured.get("text") or record.content).lower() in lowered
        )
        return min(1.0, 0.4 + (matches / max(len(mkb_context), 1)))

    def _internal_coherence(self, content: str) -> float:
        if not content:
            return 0.0
        lowered = content.lower()
        score = 0.6
        if any(token in lowered for token in ("because", "therefore", "given")):
            score += 0.2
        if len(content.split()) > 12:
            score += 0.1
        if "contradict" in lowered:
            score -= 0.4
        return max(0.0, min(score, 1.0))

