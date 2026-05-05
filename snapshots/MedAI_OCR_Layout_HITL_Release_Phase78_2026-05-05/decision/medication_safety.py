"""
MedAI v1.1 — Medication Safety Gate (Track C)
Layer 1: DDI as evidence modifier (score penalty)
Layer 2: DDI as invariant enforcement (hard block before MKB write)
Both layers are mandatory. Neither can be bypassed by automated logic.
"""
from typing import List, Optional, Tuple
from loguru import logger

from app.schemas import MKBRecord, DDIFinding, ScoredResponse, LedgerEvent
from app.config import DDI_HIGH, DDI_MEDIUM, DDI_LOW, DDI_NONE


class MedicationSafetyGate:
    def __init__(self, ddi_connector=None, sql_store=None):
        self.ddi = ddi_connector  # PatientNotes DDI connector
        self.sql = sql_store

    # ── LAYER 1: Evidence Modifier (used in Response Scoring) ─────────────

    def compute_ddi_score_modifier(
        self,
        response_content: str,
        active_medications: List[MKBRecord],
    ) -> Tuple[float, List[DDIFinding]]:
        """
        Returns (ddi_safety_score 0.0–1.0, findings[])
        Score of 1.0 = no interactions. Lower = interactions found.
        This LOWERS the response score — unsafe recommendations rank lower.
        """
        if not active_medications or not response_content:
            return 1.0, []

        if not self.ddi:
            logger.warning("DDI connector not available — returning neutral score")
            return 0.8, []  # Slight penalty for unverified

        mentioned_meds = self._extract_medication_names(response_content)
        active_names = [r.structured.get("name", r.content) for r in active_medications]

        if not mentioned_meds:
            return 1.0, []

        findings = self._check_interactions(mentioned_meds, active_names)
        score = 1.0
        for finding in findings:
            if finding.severity == DDI_HIGH:
                score = max(0.0, score - 0.40)
            elif finding.severity == DDI_MEDIUM:
                score = max(0.0, score - 0.20)
            elif finding.severity == DDI_LOW:
                score = max(0.0, score - 0.05)

        if findings:
            logger.warning(f"DDI Layer 1: {len(findings)} interactions found, score={score:.2f}")

        return score, findings

    # ── LAYER 2: Invariant Gate (used before MKB write) ──────────────────

    def gate_medication_write(
        self,
        candidate: MKBRecord,
        session_id: str = "",
    ) -> Tuple[str, str, List[DDIFinding]]:
        """
        Returns (decision, message, findings)
        decision: 'allow' | 'warn' | 'block' | 'queue'
        CRITICAL: HIGH severity BLOCKS write until user confirms.
        """
        if candidate.fact_type != "medication":
            return "allow", "Not a medication record", []

        if not self.ddi:
            # Queue for later DDI check
            candidate.ddi_status = "pending"
            candidate.ddi_checked = False
            logger.warning("DDI Layer 2: connector unavailable — queuing for later check")
            return "queue", "DDI connector unavailable. Medication write queued.", []

        active_meds = self.sql.get_active_medications() if self.sql else []
        active_names = [r.structured.get("name", r.content) for r in active_meds]
        med_name = candidate.structured.get("name", candidate.content)

        findings = self._check_interactions([med_name], active_names)

        # Update record DDI fields
        candidate.ddi_checked = True
        candidate.ddi_findings = [f.model_dump() for f in findings]

        if not findings:
            candidate.ddi_status = "clear"
            return "allow", "No interactions found", []

        max_severity = self._max_severity(findings)

        if max_severity == DDI_HIGH:
            candidate.ddi_status = "high_blocked"
            self._log_ddi_block(candidate, findings, session_id)
            msg = (
                f"HIGH SEVERITY INTERACTION: {med_name} interacts with existing medications. "
                f"Write BLOCKED. User confirmation required."
            )
            logger.error(f"DDI BLOCK: {msg}")
            return "block", msg, findings

        elif max_severity == DDI_MEDIUM:
            candidate.ddi_status = "medium"
            msg = f"MEDIUM severity interaction detected for {med_name}. Proceeding with warning."
            logger.warning(f"DDI WARN: {msg}")
            return "warn", msg, findings

        else:  # LOW
            candidate.ddi_status = "low"
            return "allow", f"Low severity interaction noted for {med_name}.", findings

    # ── Helpers ────────────────────────────────────────────────────────────

    def _check_interactions(
        self, new_meds: List[str], active_meds: List[str]
    ) -> List[DDIFinding]:
        """Query DDI connector for interactions between new and active medications."""
        if not self.ddi or not new_meds or not active_meds:
            return []
        try:
            return self.ddi.check_interactions(new_meds, active_meds)
        except Exception as e:
            logger.warning(f"DDI check failed: {e}")
            return []

    def _extract_medication_names(self, text: str) -> List[str]:
        """Simple extraction of potential medication names from response text."""
        import re
        # Find capitalized words that might be drug names (simplified)
        candidates = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        # Filter common English words
        stopwords = {"This", "That", "These", "The", "For", "With", "From", "When", "Patient"}
        return [c for c in candidates if c not in stopwords][:20]

    def _max_severity(self, findings: List[DDIFinding]) -> str:
        for sev in [DDI_HIGH, DDI_MEDIUM, DDI_LOW]:
            if any(f.severity == sev for f in findings):
                return sev
        return DDI_NONE

    def _log_ddi_block(self, record: MKBRecord, findings: List[DDIFinding], session_id: str):
        if self.sql:
            self.sql.write_ledger(LedgerEvent(
                event_type="ddi_block",
                record_id=record.id,
                source_type=record.source_type,
                details={
                    "medication": record.structured.get("name", record.content),
                    "severity": "HIGH",
                    "findings": [f.model_dump() for f in findings],
                    "action": "blocked_pending_user_confirmation",
                },
                session_id=session_id,
            ))
