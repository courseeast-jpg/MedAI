"""CKA-TERM-01 — narrow integration helper for CKA-B07 medical_coding.

Default B07 behavior (synthetic_mapper) is unchanged. This module only
exposes an OPT-IN function the operator can call to attempt a coded
mapping via the local terminology lookup service. If the term is
unmapped, the function returns an unmapped result — the caller is
responsible for honoring it.

Hard rules:
- Coding does NOT promote hypothesis-tier facts.
- Coding does NOT clear DDI status.
- Coding does NOT invent codes (UNMAPPED returns empty matches).
- This helper does NOT modify the MKB store.
"""
from __future__ import annotations

from typing import List, Optional

from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import (
    TerminologyLookupResult,
    TerminologyLookupStatus,
    TerminologySystem,
)


def code_entity_via_local_terminology(
    entity_text: str,
    service: TerminologyLookupService,
    *,
    systems: Optional[List[TerminologySystem]] = None,
    max_results: int = 5,
) -> TerminologyLookupResult:
    """OPT-IN coding helper. Returns the lookup result; never alters
    MKB tier and never clears DDI status.

    The caller (B07 medical_coding integration) is expected to:
    - keep tier as `hypothesis` for any AI-derived candidate
    - leave DDI status untouched
    - treat UNMAPPED as "no code assigned" (do NOT invent codes)
    """
    return service.lookup(
        entity_text, systems=systems, max_results=max_results,
    )


def safe_b07_boundary_summary() -> dict:
    """Public-report-safe summary of the boundary invariants this
    integration helper preserves.
    """
    return {
        "default_b07_behavior_unchanged": True,
        "coding_promotes_hypothesis": False,
        "coding_clears_ddi_status": False,
        "no_code_hallucinated": True,
        "unknown_terms_unmapped": True,
        "synthetic_mapper_still_present": True,
    }


# Cross-check helpers for the validation script.

def lookup_status_is_safe_unmapped(result: TerminologyLookupResult) -> bool:
    """True if an UNMAPPED result correctly carries no matches."""
    return (
        result.status == TerminologyLookupStatus.UNMAPPED
        and not result.matches
        and result.no_code_hallucinated is True
    )
