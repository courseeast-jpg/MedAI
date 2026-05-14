"""CKA-TERM-08 hypothesis-only terminology annotation pilot.

This module is intentionally isolated from B07 and the runtime clinical
pipeline. When the feature flag is disabled, it produces no annotation. When
enabled, it uses the read-only TERM-05 adapter interface to create review-only
metadata; it never writes accepted facts, clears DDI status, or promotes a
hypothesis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from clinical_knowledge.terminology.term05_read_only_adapter import (
    SyntheticReadOnlyTerminologyAdapter,
    Term05AdapterMatch,
)


TERM08_FEATURE_FLAG = "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION"


@dataclass(frozen=True)
class Term08CandidateCode:
    source_system: str
    code: str
    display_normalized: str

    def safe_public_summary(self) -> dict:
        return {
            "source_system": self.source_system,
            "code": self.code,
            "display_normalized": self.display_normalized,
        }


@dataclass(frozen=True)
class Term08HypothesisAnnotation:
    input_term: str
    normalized_term: str
    terminology_status: str
    source_system: str | None
    candidate_codes: tuple[Term08CandidateCode, ...] = field(default_factory=tuple)
    annotation_tier: str = "hypothesis"
    requires_review: bool = True
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    read_only_lookup: bool = True
    writes_active_fact: bool = False
    clears_ddi_status: bool = False
    promotes_hypothesis: bool = False
    external_api_used: bool = False
    dosing_advice_generated: bool = False
    prescribing_advice_generated: bool = False
    clinical_recommendation_generated: bool = False
    b07_integrated: bool = False
    no_code_hallucinated: bool = True

    def safe_public_summary(self) -> dict:
        return {
            "input_term": self.input_term,
            "normalized_term": self.normalized_term,
            "terminology_status": self.terminology_status,
            "source_system": self.source_system,
            "candidate_codes": [candidate.safe_public_summary() for candidate in self.candidate_codes],
            "annotation_tier": self.annotation_tier,
            "requires_review": self.requires_review,
            "reason_codes": list(self.reason_codes),
            "read_only_lookup": self.read_only_lookup,
            "writes_active_fact": self.writes_active_fact,
            "clears_ddi_status": self.clears_ddi_status,
            "promotes_hypothesis": self.promotes_hypothesis,
            "external_api_used": self.external_api_used,
            "dosing_advice_generated": self.dosing_advice_generated,
            "prescribing_advice_generated": self.prescribing_advice_generated,
            "clinical_recommendation_generated": self.clinical_recommendation_generated,
            "b07_integrated": self.b07_integrated,
            "no_code_hallucinated": self.no_code_hallucinated,
        }


def term08_annotation_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get(TERM08_FEATURE_FLAG, "")).strip().lower() in {"1", "true", "yes", "on"}


def annotate_candidate_term(
    input_term: str,
    *,
    adapter: SyntheticReadOnlyTerminologyAdapter | None = None,
    source_filter: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
    max_results: int = 10,
) -> Term08HypothesisAnnotation | None:
    """Return hypothesis-only terminology metadata, or None when disabled."""
    if not term08_annotation_enabled(env):
        return None

    lookup_adapter = adapter or SyntheticReadOnlyTerminologyAdapter()
    result = lookup_adapter.lookup(input_term, source_filter=source_filter, max_results=max_results)
    candidates = tuple(_candidate_from_match(match) for match in result.matches)
    source_systems = sorted({candidate.source_system for candidate in candidates})
    status = result.status if result.status in {"exact", "ambiguous", "unmapped"} else "unmapped"
    reason_codes = list(result.reason_codes)
    if status == "exact":
        reason_codes.append("hypothesis_only_exact_lookup")
    elif status == "ambiguous":
        reason_codes.append("ambiguity_requires_manual_review")
    else:
        reason_codes.append("unmapped_no_code_hallucinated")

    return Term08HypothesisAnnotation(
        input_term=input_term,
        normalized_term=result.normalized_query,
        terminology_status=status,
        source_system=source_systems[0] if len(source_systems) == 1 else None,
        candidate_codes=candidates,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        no_code_hallucinated=result.no_code_hallucinated and not (status == "unmapped" and bool(candidates)),
    )


def summarize_annotation_for_public_report(annotation: Term08HypothesisAnnotation | None) -> dict:
    if annotation is None:
        return {
            "annotation_produced": False,
            "terminology_status": "disabled",
            "candidate_code_count": 0,
            "annotation_tier": None,
            "requires_review": None,
            "reason_codes": ["feature_flag_disabled"],
            "read_only_lookup": True,
            "writes_active_fact": False,
            "clears_ddi_status": False,
            "promotes_hypothesis": False,
            "external_api_used": False,
        }
    return {
        "annotation_produced": True,
        "terminology_status": annotation.terminology_status,
        "candidate_code_count": len(annotation.candidate_codes),
        "source_system": annotation.source_system,
        "annotation_tier": annotation.annotation_tier,
        "requires_review": annotation.requires_review,
        "reason_codes": list(annotation.reason_codes),
        "read_only_lookup": annotation.read_only_lookup,
        "writes_active_fact": annotation.writes_active_fact,
        "clears_ddi_status": annotation.clears_ddi_status,
        "promotes_hypothesis": annotation.promotes_hypothesis,
        "external_api_used": annotation.external_api_used,
        "no_code_hallucinated": annotation.no_code_hallucinated,
        "b07_integrated": annotation.b07_integrated,
        "dosing_advice_generated": annotation.dosing_advice_generated,
        "prescribing_advice_generated": annotation.prescribing_advice_generated,
        "clinical_recommendation_generated": annotation.clinical_recommendation_generated,
    }


def _candidate_from_match(match: Term05AdapterMatch) -> Term08CandidateCode:
    return Term08CandidateCode(
        source_system=match.system,
        code=match.code,
        display_normalized=match.display_normalized,
    )
