"""
MedAI v1.1 — Response Scoring Engine (Track C)
4-dimension weighted scoring formula.
DDI modifier integrated as evidence (Layer 1).
Discard threshold: < 0.30
"""
from typing import List, Tuple
from loguru import logger

from app.schemas import ConnectorResponse, ScoredResponse, DDIFinding, MKBContext
from app.config import (
    RESPONSE_DISCARD_THRESHOLD, ANTHROPIC_API_KEY, CLAUDE_MODEL
)


class ResponseScorer:
    def __init__(self, vector_store=None, medication_gate=None, claude_client=None):
        self.vec = vector_store
        self.med_gate = medication_gate
        self.claude = claude_client

    def score(
        self,
        response: ConnectorResponse,
        mkb_context: MKBContext,
        requires_ddi_check: bool = False,
    ) -> ScoredResponse:
        """Score a connector response on 4 dimensions. Returns ScoredResponse."""
        if not response.content or response.status in ("timeout", "error"):
            return ScoredResponse(
                connector_name=response.connector_name,
                content=response.content,
                final_score=0.0,
                discarded=True,
                discard_reason=f"Connector status: {response.status}",
                confidence_band="discarded",
            )

        # Dim 1: MKB consistency (0.35)
        mkb_score = self._score_mkb_consistency(response.content, mkb_context)

        # Dim 2: Internal coherence (0.25)
        coherence_score = self._score_coherence(response.content)

        # Dim 3: Citation presence (0.20)
        citation_score = self._score_citations(response)

        # Dim 4: DDI safety modifier (0.20)
        ddi_score = 1.0
        ddi_findings = []
        if requires_ddi_check and self.med_gate:
            ddi_score, ddi_findings = self.med_gate.compute_ddi_score_modifier(
                response.content,
                mkb_context.active_medications,
            )

        # Weighted sum
        final = (
            mkb_score    * 0.35 +
            coherence_score * 0.25 +
            citation_score  * 0.20 +
            ddi_score       * 0.20
        )
        final = round(min(1.0, max(0.0, final)), 3)

        discarded = final < RESPONSE_DISCARD_THRESHOLD
        band = self._confidence_band(final)

        if discarded:
            reason = f"Score {final:.3f} below discard threshold {RESPONSE_DISCARD_THRESHOLD}"
            logger.info(f"Discarded {response.connector_name}: {reason}")
        else:
            logger.info(f"Scored {response.connector_name}: {final:.3f} ({band})")

        return ScoredResponse(
            connector_name=response.connector_name,
            content=response.content,
            final_score=final,
            score_breakdown={
                "mkb_consistency": round(mkb_score, 3),
                "internal_coherence": round(coherence_score, 3),
                "citation_presence": round(citation_score, 3),
                "ddi_safety_score": round(ddi_score, 3),
            },
            ddi_findings=ddi_findings,
            discarded=discarded,
            discard_reason=reason if discarded else None,
            confidence_band=band,
        )

    def score_all(
        self,
        responses: List[ConnectorResponse],
        mkb_context: MKBContext,
        requires_ddi_check: bool = False,
    ) -> Tuple[List[ScoredResponse], List[ScoredResponse]]:
        """Score all responses. Returns (accepted, discarded)."""
        scored = [self.score(r, mkb_context, requires_ddi_check) for r in responses]
        accepted = [s for s in scored if not s.discarded]
        discarded = [s for s in scored if s.discarded]
        logger.info(f"Scoring: {len(accepted)} accepted, {len(discarded)} discarded of {len(responses)}")
        return accepted, discarded

    # ── Scoring dimensions ─────────────────────────────────────────────────

    def _score_mkb_consistency(self, content: str, ctx: MKBContext) -> float:
        """Semantic similarity between response and MKB context chunks."""
        if not self.vec or not ctx.semantic_chunks:
            return 0.5  # Neutral when no context available

        try:
            results = self.vec.semantic_search(content, n_results=5)
            if not results:
                return 0.4
            similarities = [r["similarity"] for r in results[:5]]
            return sum(similarities) / len(similarities)
        except Exception as e:
            logger.warning(f"MKB consistency scoring failed: {e}")
            return 0.5

    def _score_coherence(self, content: str) -> float:
        """
        Internal coherence check.
        Uses simple heuristics; upgrades to Claude check when available.
        """
        if not content:
            return 0.0

        # Heuristic checks
        content_lower = content.lower()

        # Contradiction indicators (lower score)
        contradiction_terms = [
            "however, this is not", "on the other hand, it is",
            "this contradicts", "inconsistent with itself"
        ]
        for term in contradiction_terms:
            if term in content_lower:
                return 0.4

        # Completeness indicators (higher score)
        has_reasoning = any(w in content_lower for w in ["because", "therefore", "due to", "given that"])
        has_specifics = len([w for w in content.split() if len(w) > 6]) > 10

        base = 0.65
        if has_reasoning:
            base += 0.15
        if has_specifics:
            base += 0.10
        if len(content) < 50:
            base -= 0.20

        return min(1.0, base)

    def _score_citations(self, response: ConnectorResponse) -> float:
        """Citation presence scoring."""
        if response.citations and len(response.citations) > 0:
            return 0.85
        content = response.content or ""
        # Look for URL patterns or named sources in content
        import re
        has_url = bool(re.search(r'https?://', content))
        has_named_source = any(
            src in content.lower()
            for src in ["pubmed", "cochrane", "nejm", "jama", "guidelines", "according to"]
        )
        if has_url:
            return 0.80
        if has_named_source:
            return 0.65
        return 0.20  # No citations

    def _confidence_band(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.50:
            return "acceptable"
        elif score >= 0.30:
            return "low"
        return "discarded"
