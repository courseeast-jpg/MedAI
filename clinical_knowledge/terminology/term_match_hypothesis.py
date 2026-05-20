"""MEDAI-CKA-TERM-INTEGRATION-NEXT-01 — Default-off terminology match
hypothesis helper.

Implements the first concrete terminology / coding integration helper
under the CKA-TERM-INTEGRATION-PLAN-01 SPEC. It is a pure function:

    derive_terminology_match_hypothesis(
        record,
        *,
        enabled=None,
        env=None,
        lookup_adapter=None,
    ) -> dict | None

Default-off semantics (any one suffices to return ``None``):

    * ``enabled`` is passed as ``False`` — overrides a truthy env.
    * ``enabled`` is ``None`` AND the env var
      ``MEDAI_TERMINOLOGY_LOOKUP_ENABLED`` is unset or falsy.
    * ``lookup_adapter`` is ``None`` — fail closed; the helper never
      defaults to an adapter (synthetic test fixtures must be injected
      explicitly by the test suite).
    * The record does not match the strict positive signature.
    * The adapter's lookup returns no candidate text.

When the helper does emit a metadata dict, the output is **purely
aggregate / controlled-vocabulary** and contains:

    * No licensed terminology row content (no codes, no display strings,
      no system identifiers beyond a controlled-vocabulary family tag).
    * No raw extracted text, raw OCR text, raw document text, raw
      filenames, private paths, PHI, or secrets.
    * Explicit ``review_required=True`` / ``auto_accept_allowed=False``
      flags.
    * Explicit ``clinical_*_performed=False`` /
      ``ddi_behavior_changed=False`` refusal flags.

This module does NOT (under any combination of kwargs / env):

    * Open licensed terminology files.
    * Read ``LICENSE_ACK_PRIVATE.json``.
    * Inspect the private RxNorm / LOINC store directly — only an
      injected adapter may.
    * Modify ``app/main.py`` or any launcher / preflight / config file.
    * Add Streamlit wiring or an operator UI surface.
    * Change OCR routing, extraction, layout/table extraction, classifier
      behavior, thresholds, scoring, cue packs, lab-value parsing,
      medication / DDI / diagnosis / treatment inference, or
      abbreviation expansion.
    * Enable any external API.
    * Touch PARK-20..23 or FREEZE tags.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Protocol


# ── Env-var name and default-off semantics ────────────────────────────────

TERMINOLOGY_LOOKUP_ENV_VAR = "MEDAI_TERMINOLOGY_LOOKUP_ENABLED"

_TRUTHY = {"1", "true", "yes", "on", "enabled"}


def is_terminology_lookup_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Pure env-gating predicate. ``env`` defaults to ``os.environ``."""
    if env is None:
        env = os.environ
    raw = env.get(TERMINOLOGY_LOOKUP_ENV_VAR, "")
    val = str(raw).strip().lower()
    return val in _TRUTHY


def is_terminology_lookup_default_disabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    return not is_terminology_lookup_enabled(env)


# ── Controlled-vocabulary output values ───────────────────────────────────

SOURCE_PHASE = "MEDAI-CKA-TERM-INTEGRATION-NEXT-01"

MATCH_FAMILY_VALUES = (
    "exact_terminology_match",
    "ambiguous_terminology_match",
    "unmapped_terminology_candidate",
)

TERMINOLOGY_SYSTEM_FAMILY_VALUES = (
    "rxnorm",
    "loinc",
    "internal_public_reference",
)

# License class for the OUTPUT of this helper, not for the underlying
# terminology source. The helper never emits row content, so the output
# itself is safe to embed in aggregate public reports.
LICENSE_CLASS_OF_OUTPUT = "aggregate_public_report_only"

DISCLAIMER = (
    "Terminology match hypothesis only. Review required. Not a final "
    "code. Not clinical interpretation. No auto-accept. No DDI status "
    "change. No diagnosis, medication, or treatment inference."
)


# ── Lookup adapter protocol ───────────────────────────────────────────────


class _AdapterMatchLike(Protocol):
    system: str


class _AdapterResultLike(Protocol):
    status: str
    matches: tuple[_AdapterMatchLike, ...]


class TerminologyLookupAdapter(Protocol):
    """Minimal duck-typed protocol the helper requires from any adapter.

    A real adapter (or the existing
    ``SyntheticReadOnlyTerminologyAdapter`` from
    ``clinical_knowledge.terminology.term05_read_only_adapter``)
    satisfies this protocol when its ``lookup`` returns an object with
    ``status`` and ``matches``.
    """

    def lookup(
        self,
        query: str,
        *,
        source_filter: Optional[list[str]] = ...,
        max_results: int = ...,
    ) -> _AdapterResultLike: ...


# ── Positive signature ────────────────────────────────────────────────────


def _extract_candidate_text(record: Mapping[str, Any]) -> str | None:
    """Pull a candidate-text string from the record.

    The helper accepts a small set of explicit, non-PHI candidate-text
    fields. It does NOT scan arbitrary record fields; that protects
    against accidentally feeding raw document text or OCR text into the
    lookup. The accepted fields are typed at the caller level and must
    contain only short, operator-curated candidate strings.
    """
    for key in (
        "terminology_candidate_text",
        "candidate_text",
        "candidate_query",
    ):
        value = record.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def matches_terminology_lookup_signature(record: Mapping[str, Any]) -> bool:
    """Strict positive signature.

    Returns True only when the record carries a non-empty,
    operator-curated candidate-text field. The signature deliberately
    excludes raw document fields (``ocr_text``, ``native_text``,
    ``raw_text``, ``filename``, etc.) so the helper cannot be misused as
    a free-text terminology lookup over private content.
    """
    return _extract_candidate_text(record) is not None


# ── System-family classification ──────────────────────────────────────────


def _classify_system_family(systems_seen: set[str]) -> str:
    """Map an adapter's observed systems to a controlled-vocab family."""
    normalized = {s.lower() for s in systems_seen if isinstance(s, str)}
    if "rxnorm" in normalized:
        return "rxnorm"
    if "loinc" in normalized:
        return "loinc"
    return "internal_public_reference"


def _classify_match_family(adapter_status: str) -> str:
    """Map an adapter status string to a controlled-vocab match family."""
    status_lower = str(adapter_status or "").lower()
    if status_lower == "exact":
        return "exact_terminology_match"
    if status_lower == "ambiguous":
        return "ambiguous_terminology_match"
    return "unmapped_terminology_candidate"


# ── Public derive function ────────────────────────────────────────────────


def derive_terminology_match_hypothesis(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
    lookup_adapter: Optional[TerminologyLookupAdapter] = None,
) -> Optional[dict[str, Any]]:
    """Return a privacy-safe controlled-vocabulary metadata dict, or
    ``None``.

    The function is pure: ``record`` is never mutated. The return value
    is a fresh dict that contains only controlled-vocabulary tokens,
    explicit invariant flags, and aggregate counts — never any row
    content from the underlying terminology source.

    Default-off paths (any one suffices):

        * ``enabled=False``
        * ``enabled is None`` and the env var is unset or falsy
        * ``lookup_adapter is None`` (fail closed)
        * The record does not match the positive signature
    """
    if enabled is False:
        return None
    if enabled is None:
        if not is_terminology_lookup_enabled(env):
            return None
    # ``enabled`` is True or env says enabled

    if lookup_adapter is None:
        # Fail closed: never default to any adapter. Production callers
        # must explicitly inject an audited adapter.
        return None

    if not matches_terminology_lookup_signature(record):
        return None

    candidate_text = _extract_candidate_text(record)
    if candidate_text is None:
        return None

    # Optional caller-supplied controlled-vocabulary system filter. Only
    # known family tokens are forwarded; unknown values are dropped so
    # we never pass arbitrary operator strings into the adapter.
    source_filter: Optional[list[str]] = None
    raw_filter = record.get("terminology_system_filter")
    if isinstance(raw_filter, (list, tuple)):
        allowed = [
            str(s).lower()
            for s in raw_filter
            if isinstance(s, str) and s.lower() in {"rxnorm", "loinc"}
        ]
        if allowed:
            source_filter = allowed

    # Run the lookup. The adapter is duck-typed; any exception from a
    # caller-supplied adapter must NOT leak terminology row content into
    # the return path — we re-raise so the caller can decide, but we do
    # not embed exception text in the helper's output.
    if source_filter is not None:
        result = lookup_adapter.lookup(
            candidate_text, source_filter=source_filter
        )
    else:
        result = lookup_adapter.lookup(candidate_text)

    matches = tuple(getattr(result, "matches", ()) or ())
    systems_seen: set[str] = set()
    for match in matches:
        system_val = getattr(match, "system", None)
        if isinstance(system_val, str):
            systems_seen.add(system_val)

    adapter_status = str(getattr(result, "status", "") or "")
    match_family = _classify_match_family(adapter_status)
    terminology_system_family = _classify_system_family(systems_seen)

    # Aggregate-only counts; NO row content escapes this helper.
    matches_count = len(matches)

    out: dict[str, Any] = {
        "terminology_match_hypothesis": True,
        "source_phase": SOURCE_PHASE,
        "enabled": True,
        "env_var": TERMINOLOGY_LOOKUP_ENV_VAR,

        # Controlled-vocabulary family labels (no row content)
        "match_family": match_family,
        "terminology_system_family": terminology_system_family,

        # Aggregate counts only
        "matches_count": matches_count,

        # License class of THIS helper's output (the output is aggregate-
        # only; underlying source license class is tracked separately by
        # the PLAN-01 license-class table).
        "license_class": LICENSE_CLASS_OF_OUTPUT,

        # Review-bound invariants
        "review_required": True,
        "auto_accept_allowed": False,

        # Refusal flags (explicit, audit-visible)
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "treatment_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_behavior_changed": False,
        "abbreviation_expanded": False,
        "lab_value_parsed": False,

        # Privacy / non-emission invariants
        "licensed_row_content_included": False,
        "raw_text_emitted": False,
        "raw_ocr_text_emitted": False,
        "raw_document_text_emitted": False,
        "raw_filename_emitted": False,
        "private_path_emitted": False,
        "phi_emitted": False,
        "secret_emitted": False,
        "public_report_safe": True,

        # Behavior-non-change invariants
        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_or_scoring_changed": False,
        "cue_packs_added": False,
        "external_api_used": False,

        # Frozen-artifact invariants
        "frozen_operator_release_preserved": True,
        "freeze_tags_touched": False,
        "park_20_tags_touched": False,
        "park_21_tags_touched": False,
        "park_22_tags_touched": False,
        "park_23_tags_touched": False,

        "disclaimer": DISCLAIMER,
    }
    return out


__all__ = [
    "DISCLAIMER",
    "LICENSE_CLASS_OF_OUTPUT",
    "MATCH_FAMILY_VALUES",
    "SOURCE_PHASE",
    "TERMINOLOGY_LOOKUP_ENV_VAR",
    "TERMINOLOGY_SYSTEM_FAMILY_VALUES",
    "TerminologyLookupAdapter",
    "derive_terminology_match_hypothesis",
    "is_terminology_lookup_default_disabled",
    "is_terminology_lookup_enabled",
    "matches_terminology_lookup_signature",
]
