"""CKA-B08 Consensus Engine package.

Provides:
- Consensus models and enums
- Fact extraction from normalized connector responses
- Agreement scoring
- Contradiction detection
- Truth Resolution handoff for contradictions
- Integration helpers

Rules:
- No synthesis over contradictions.
- No auto-write of active facts.
- Consensus-to-enrichment path remains hypothesis-only.
- Medication dose contradictions: quarantine-only via Truth Resolution.
- Truth Resolution in B08 does NOT invoke DDI check.
"""
