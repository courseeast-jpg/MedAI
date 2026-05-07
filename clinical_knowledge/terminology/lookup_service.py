"""CKA-TERM-01 — terminology lookup service.

Wraps a `LocalTerminologyStore` and returns deterministic
`TerminologyLookupResult` values:

- exact normalized display match → status=EXACT
- synonym normalized match → status=SYNONYM
- multiple distinct (system, code) results → status=AMBIGUOUS
- nothing found → status=UNMAPPED, matches=[]

Hard rules:
- No external API.
- No code hallucination (UNMAPPED returns empty matches).
- No clinical interpretation.
- No DDI status modification.
"""
from __future__ import annotations

from typing import List, Optional

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologyLookupResult,
    TerminologyLookupStatus,
    TerminologySystem,
    normalize_query,
)


class TerminologyLookupService:
    """Lookup facade. Always safe — never invents a code."""

    def __init__(self, store: LocalTerminologyStore) -> None:
        self.store = store

    def lookup(
        self,
        term: str,
        *,
        systems: Optional[List[TerminologySystem]] = None,
        max_results: int = 10,
    ) -> TerminologyLookupResult:
        result = TerminologyLookupResult.for_query(term, systems=systems)
        norm = result.normalized_query
        if not norm:
            # Empty query is unmapped.
            result.status = TerminologyLookupStatus.UNMAPPED
            return result

        matches = self.store.fetch_concepts_by_norm(
            norm=norm, systems=systems, max_results=max_results,
        )
        result.matches = matches

        if not matches:
            result.status = TerminologyLookupStatus.UNMAPPED
            result.exact_match = False
            result.ambiguous = False
            return result

        # Distinguish exact vs synonym vs ambiguous.
        exact_hits = [c for c in matches if normalize_query(c.display) == norm]
        # Group by (system, code) tuple to detect cross-system / cross-code
        # ambiguity.
        unique_codes = {(c.system.value, c.code) for c in matches}

        if len(unique_codes) > 1:
            result.status = TerminologyLookupStatus.AMBIGUOUS
            result.ambiguous = True
            result.exact_match = bool(exact_hits)
            return result

        # Single (system, code) — exact display match wins; otherwise it
        # matched via a synonym.
        if exact_hits:
            result.status = TerminologyLookupStatus.EXACT
            result.exact_match = True
        else:
            result.status = TerminologyLookupStatus.SYNONYM
            result.exact_match = False
        result.ambiguous = False
        return result
