"""B07-TERM-01 opt-in terminology metadata shim.

This module is not wired into the runtime pipeline by default. It provides a
small fail-closed helper that B07 can call only when all terminology opt-in
flags are explicitly enabled. Output is hypothesis-only review metadata.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from clinical_knowledge.terminology.term05_read_only_adapter import SyntheticReadOnlyTerminologyAdapter
from clinical_knowledge.terminology.term08_hypothesis_annotation_pilot import (
    TERM08_FEATURE_FLAG,
    Term08HypothesisAnnotation,
    annotate_candidate_term,
)


B07_TERM_OPT_IN_FLAG = "MEDAI_B07_TERMINOLOGY_OPT_IN"
TERM_LOOKUP_ENABLED_FLAG = "MEDAI_TERMINOLOGY_LOOKUP_ENABLED"
TERM_READ_ONLY_FLAG = "MEDAI_TERMINOLOGY_READ_ONLY"
TERM_ALLOW_WRITES_FLAG = "MEDAI_TERMINOLOGY_ALLOW_WRITES"
REQUIRED_FLAGS = (
    B07_TERM_OPT_IN_FLAG,
    TERM_LOOKUP_ENABLED_FLAG,
    TERM08_FEATURE_FLAG,
    TERM_READ_ONLY_FLAG,
    TERM_ALLOW_WRITES_FLAG,
)


@dataclass(frozen=True)
class B07TerminologyFlagState:
    b07_opt_in: bool = False
    lookup_enabled: bool = False
    hypothesis_annotation_enabled: bool = False
    read_only: bool = True
    allow_writes: bool = False
    raw_values: dict[str, str | None] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return (
            self.b07_opt_in
            and self.lookup_enabled
            and self.hypothesis_annotation_enabled
            and self.read_only
            and not self.allow_writes
        )

    def reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.b07_opt_in:
            reasons.append("b07_terminology_opt_in_disabled")
        if not self.lookup_enabled:
            reasons.append("terminology_lookup_disabled")
        if not self.hypothesis_annotation_enabled:
            reasons.append("terminology_hypothesis_annotation_disabled")
        if not self.read_only:
            reasons.append("terminology_read_only_required")
        if self.allow_writes:
            reasons.append("terminology_writes_forbidden")
        return tuple(reasons)

    def safe_public_summary(self) -> dict:
        return {
            "MEDAI_B07_TERMINOLOGY_OPT_IN": self.b07_opt_in,
            "MEDAI_TERMINOLOGY_LOOKUP_ENABLED": self.lookup_enabled,
            "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION": self.hypothesis_annotation_enabled,
            "MEDAI_TERMINOLOGY_READ_ONLY": self.read_only,
            "MEDAI_TERMINOLOGY_ALLOW_WRITES": self.allow_writes,
            "enabled": self.enabled,
            "reason_codes": list(self.reason_codes()),
        }


@dataclass(frozen=True)
class B07TerminologyMetadata:
    enabled: bool
    flags_state: B07TerminologyFlagState
    input_term: str | None = None
    normalized_term: str | None = None
    terminology_status: str = "disabled"
    source_system: str | None = None
    candidate_code_count: int = 0
    candidate_codes: tuple[dict, ...] = field(default_factory=tuple)
    annotation_tier: str | None = None
    requires_review: bool = True
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    read_only_lookup: bool = True
    writes_active_fact: bool = False
    clears_ddi_status: bool = False
    promotes_hypothesis: bool = False
    external_api_used: bool = False
    b07_authority_source: bool = False
    clinical_advice_generated: bool = False
    dosing_advice_generated: bool = False
    prescribing_advice_generated: bool = False
    no_code_hallucinated: bool = True

    def safe_public_summary(self) -> dict:
        return {
            "enabled": self.enabled,
            "flags_state": self.flags_state.safe_public_summary(),
            "input_term": self.input_term,
            "normalized_term": self.normalized_term,
            "terminology_status": self.terminology_status,
            "source_system": self.source_system,
            "candidate_code_count": self.candidate_code_count,
            "candidate_codes": list(self.candidate_codes),
            "annotation_tier": self.annotation_tier,
            "requires_review": self.requires_review,
            "reason_codes": list(self.reason_codes),
            "read_only_lookup": self.read_only_lookup,
            "writes_active_fact": self.writes_active_fact,
            "clears_ddi_status": self.clears_ddi_status,
            "promotes_hypothesis": self.promotes_hypothesis,
            "external_api_used": self.external_api_used,
            "b07_authority_source": self.b07_authority_source,
            "clinical_advice_generated": self.clinical_advice_generated,
            "dosing_advice_generated": self.dosing_advice_generated,
            "prescribing_advice_generated": self.prescribing_advice_generated,
            "no_code_hallucinated": self.no_code_hallucinated,
        }


def read_b07_term_flag_state(env: Mapping[str, str] | None = None) -> B07TerminologyFlagState:
    values = env if env is not None else os.environ
    raw = {name: values.get(name) for name in REQUIRED_FLAGS}
    return B07TerminologyFlagState(
        b07_opt_in=_truthy(raw[B07_TERM_OPT_IN_FLAG]),
        lookup_enabled=_truthy(raw[TERM_LOOKUP_ENABLED_FLAG]),
        hypothesis_annotation_enabled=_truthy(raw[TERM08_FEATURE_FLAG]),
        read_only=_truthy(raw[TERM_READ_ONLY_FLAG], default=True),
        allow_writes=_truthy(raw[TERM_ALLOW_WRITES_FLAG], default=False),
        raw_values=raw,
    )


def build_b07_terminology_metadata(
    input_term: str,
    *,
    adapter: SyntheticReadOnlyTerminologyAdapter | None = None,
    source_filter: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> B07TerminologyMetadata:
    flags = read_b07_term_flag_state(env)
    if not flags.enabled:
        return B07TerminologyMetadata(
            enabled=False,
            flags_state=flags,
            input_term=None,
            normalized_term=None,
            terminology_status="disabled",
            candidate_code_count=0,
            annotation_tier=None,
            reason_codes=flags.reason_codes() or ("terminology_flags_disabled",),
        )

    annotation = annotate_candidate_term(
        input_term,
        adapter=adapter,
        source_filter=source_filter,
        env={TERM08_FEATURE_FLAG: "1"},
    )
    if annotation is None:
        return B07TerminologyMetadata(
            enabled=False,
            flags_state=flags,
            input_term=None,
            terminology_status="disabled",
            reason_codes=("term08_annotation_unavailable",),
        )
    return _metadata_from_annotation(annotation, flags)


def _metadata_from_annotation(
    annotation: Term08HypothesisAnnotation,
    flags: B07TerminologyFlagState,
) -> B07TerminologyMetadata:
    candidate_codes = tuple(
        {
            "source_system": candidate.source_system,
            "code": candidate.code,
            "display_normalized": candidate.display_normalized,
        }
        for candidate in annotation.candidate_codes
    )
    return B07TerminologyMetadata(
        enabled=True,
        flags_state=flags,
        input_term=annotation.input_term,
        normalized_term=annotation.normalized_term,
        terminology_status=annotation.terminology_status,
        source_system=annotation.source_system,
        candidate_code_count=len(candidate_codes),
        candidate_codes=candidate_codes,
        annotation_tier="hypothesis",
        requires_review=True,
        reason_codes=tuple(annotation.reason_codes),
        read_only_lookup=True,
        writes_active_fact=False,
        clears_ddi_status=False,
        promotes_hypothesis=False,
        external_api_used=False,
        b07_authority_source=False,
        clinical_advice_generated=False,
        dosing_advice_generated=False,
        prescribing_advice_generated=False,
        no_code_hallucinated=annotation.no_code_hallucinated,
    )


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
