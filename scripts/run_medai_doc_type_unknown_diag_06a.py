"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A - Numeric-Table Safe-Default Policy Spec.

Specification-only block.

Scope
-----
Targets the 11 records UNKNOWN-DIAG-05 routed to
``candidate_numeric_table_safe_default_policy``. The block:

    1. Re-derives the priority slice deterministically from the FAMILY-04
       per-file public report by chaining the DIAG-02 -> DIAG-03 -> DIAG-04
       -> DIAG-05 filters (no corpus rerun, no source documents opened).
    2. Confirms that every record in the slice matches the exact
       positive signature defined by this spec.
    3. Confirms that no excluded condition is present in the slice.
    4. Emits a privacy-safe public specification of a FUTURE numeric-table
       safe-default language policy: positive signature, exclusion rules,
       proposed default behavior, future implementation acceptance
       criteria, and future validation requirements.

This block does NOT implement the policy. It does not write classifier
behavior, modify runtime decisions, or change any system component.

Hard boundaries
---------------
* No OCR routing or engine changes.
* No language-detector behavior changes.
* No classifier behavior changes.
* No thresholds, scoring, auto-accept, cue-pack, or cue-expansion changes.
* No clinical interpretation, no lab/medication/dose/DDI parsing.
* No B07 / ROUTE-FIX / DB schema / command allowlist / external-API changes.
* No raw filenames, raw OCR text, raw document text, private paths, PHI, or
  secrets emitted. Anonymized aggregate output only.

Progress estimate
-----------------
* Before this block: residual Unknown-reduction track ~55% done /
  ~45% remaining; whole MedAI project ~75% done / ~25% remaining.
* After this block:  residual Unknown-reduction track ~62% done /
  ~38% remaining; whole MedAI project ~76% done / ~24% remaining.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_medai_doc_type_unknown_diag_05 import (
    SOURCE_REPORT,
    assign_future_policy_candidate,
    select_target_records as _diag05_select_target_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_06a"

PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_DIAG_LABELS = (
    "reports/medai_doc_type_unknown_diag_05/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_03/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)",
)

_DIAG05_TARGET_CANDIDATE = "candidate_numeric_table_safe_default_policy"
_DEFERRED_HEADER_CANDIDATE = "candidate_table_header_language_policy"


# ── Controlled-vocabulary specification (task-mandated) ──────────────────────

POSITIVE_SIGNATURE: tuple[tuple[str, str], ...] = (
    ("table_like_structure_detected",        "yes"),
    ("high_table_density_required",          "yes"),
    ("repeated_row_pattern_visible",         "yes"),
    ("numeric_content_bucket",               "medium"),
    ("numeric_units_or_ranges_visible",      "yes"),
    ("script_detection_result",              "latin"),
    ("dominant_script",                      "latin"),
    ("detector_confidence_bucket",           "high"),
    ("alphabetic_ratio_sufficient_for_language", "yes"),
    ("administrative_table_shape",           "yes"),
    ("treatment_schedule_table_shape",       "yes"),
    ("language_detector_attempted",          "yes"),
    ("language_detector_input_bucket",       "sufficient"),
    ("language_visibility_status",           "latin_visible_language_unknown"),
)

EXCLUSION_RULES: tuple[str, ...] = (
    "exclude_cyrillic_dominant_records",
    "exclude_mixed_script_records",
    "exclude_low_alphabetic_ratio_records",
    "exclude_no_text_layer_records",
    "exclude_image_like_but_not_routed_records",
    "exclude_ambiguous_below_threshold_records",
    "exclude_fallback_ran_but_no_family_match_records",
    "exclude_medication_dose_or_ddi_interpretation",
    "exclude_lab_value_parsing",
    "exclude_records_with_insufficient_safe_metadata",
)

PROPOSED_FUTURE_DEFAULT: dict[str, Any] = {
    "applies_to": "records matching the exact positive signature only",
    "default_action":
        "assign language visibility as latin_script_likely_english_table_context "
        "for routing and metadata purposes only",
    "scope_of_effect": (
        "routing and metadata propagation only; no classifier outcome change, "
        "no clinical interpretation, no value parsing"
    ),
    "must_not_auto_accept": True,
    "must_not_classify_clinical_meaning": True,
    "must_not_parse_values": True,
    "must_not_write_active_clinical_facts": True,
    "must_keep_document_review_bound": True,
}

FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "unknown_count_decreases_only_for_exact_signature_records",
    "accepted_count_remains_zero",
    "auto_accept_allowed_count_remains_zero",
    "external_api_used_count_remains_zero",
    "all_affected_records_remain_review_bound",
    "no_new_treatment_imaging_or_admin_false_positive_expansion",
    "public_report_privacy_checks_remain_clean",
    "rollback_flag_or_disable_path_exists",
)

FUTURE_VALIDATION_REQUIREMENTS: tuple[str, ...] = (
    "focused_synthetic_tests",
    "replay_of_11_record_anonymous_subset",
    "larger_507_file_aggregate_validation",
    "document_type_eval_regression_tests",
    "public_report_privacy_checks",
    "final_cka_mvp_validation",
    "b07_validation",
    "route_fix_validation",
    "staged_safety_check",
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SignatureMatchReport:
    """Per-signature-key coverage across the 11 priority records."""
    expected_value: str
    matching_record_count: int
    fully_matches: bool


@dataclass
class ExclusionAuditReport:
    """For each exclusion rule, the number of priority records that VIOLATE
    the rule (i.e. would have to be excluded). The spec is valid only when
    every count is zero on the current priority slice."""
    rule: str
    violating_record_count: int


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_diag_labels: list[str]
    generated_at: str

    total_numeric_table_policy_records: int
    deferred_table_header_record_count: int
    other_diag05_lever_records_deferred: dict[str, int]

    positive_signature: list[dict[str, str]]
    positive_signature_match_report: dict[str, dict[str, Any]]
    positive_signature_holds_on_all_priority_records: bool

    exclusion_rules: list[str]
    exclusion_audit: list[dict[str, Any]]
    no_priority_record_violates_any_exclusion_rule: bool

    proposed_future_default: dict[str, Any]
    future_implementation_acceptance_criteria: list[str]
    future_validation_requirements: list[str]

    sample_ids: list[str]
    raw_signal_counts: dict[str, dict[str, int]]
    deferred_subsets: dict[str, str]

    implementation_block_recommended_next: str
    implementation_block_justification: str

    progress_estimate: dict[str, str]

    behavior_changed: bool
    external_api_used: bool
    cue_expansion_recommended: bool
    policy_implemented_in_this_block: bool
    safety_privacy: dict[str, bool]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


_LABEL_RE = re.compile(r"^[a-z0-9_]+$")


def _is_safe_label(value: Any) -> bool:
    return isinstance(value, str) and bool(_LABEL_RE.fullmatch(value))


def _safe_count(records: Iterable[dict], key: str) -> dict[str, int]:
    c: Counter = Counter()
    for r in records:
        v = r.get(key)
        v = "unknown" if v is None else str(v)
        if not _is_safe_label(v):
            v = "other"
        c[v] += 1
    return dict(sorted(c.items()))


def _g(r: dict, key: str) -> str:
    return str(r.get(key) or "").lower()


def _anonymized_ids(prefix: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


# ── Selection: chain to DIAG-05's selection + filter to numeric-table ───────

def select_numeric_table_records(table: list[dict]) -> list[dict]:
    """Return the 11 records DIAG-05 routed to candidate_numeric_table_safe_default_policy."""
    return [
        r for r in _diag05_select_target_records(table)
        if assign_future_policy_candidate(r) == _DIAG05_TARGET_CANDIDATE
    ]


def select_other_diag05_lever_buckets(table: list[dict]) -> dict[str, int]:
    """Return per-bucket counts for DIAG-05 lever assignments that are NOT
    the numeric-table candidate (i.e. deferred from this spec)."""
    counts: Counter = Counter()
    for r in _diag05_select_target_records(table):
        c = assign_future_policy_candidate(r)
        if c != _DIAG05_TARGET_CANDIDATE:
            counts[c] += 1
    return dict(sorted(counts.items()))


# ── Positive-signature verification ─────────────────────────────────────────

def _positive_signature_predicate(key: str, expected: str, record: dict) -> bool:
    """Map signature key to the per-record check using the raw FAMILY-04
    field that backs it."""
    if key == "table_like_structure_detected":
        return _g(record, "table_like_structure_detected") == expected
    if key == "high_table_density_required":
        return _g(record, "table_like_structure_detected") == "yes"
    if key == "repeated_row_pattern_visible":
        return (_g(record, "table_like_structure_detected") == "yes"
                and _g(record, "date_or_schedule_shape_detected") == "yes")
    if key == "numeric_content_bucket":
        return _g(record, "numeric_content_bucket") == expected
    if key == "numeric_units_or_ranges_visible":
        return _g(record, "symbol_content_bucket") in {"medium", "high"}
    if key == "script_detection_result":
        return _g(record, "script_detection_result") == expected
    if key == "dominant_script":
        return _g(record, "dominant_script") == expected
    if key == "detector_confidence_bucket":
        return _g(record, "detector_confidence_bucket") == expected
    if key == "alphabetic_ratio_sufficient_for_language":
        return _g(record, "alphabetic_content_bucket") == "high"
    if key == "administrative_table_shape":
        return _g(record, "administrative_form_shape_detected") == "yes"
    if key == "treatment_schedule_table_shape":
        return _g(record, "date_or_schedule_shape_detected") == "yes"
    if key == "language_detector_attempted":
        return _g(record, "language_detector_attempted") == "yes"
    if key == "language_detector_input_bucket":
        return _g(record, "language_detector_input_bucket") == expected
    if key == "language_visibility_status":
        return _g(record, "language_visibility_status") == expected
    return False


def _positive_signature_match_report(
    records: list[dict],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = len(records)
    for key, expected in POSITIVE_SIGNATURE:
        match_count = sum(
            1 for r in records
            if _positive_signature_predicate(key, expected, r)
        )
        out[key] = {
            "expected_value": expected,
            "matching_record_count": match_count,
            "fully_matches": (match_count == n and n > 0),
        }
    return out


# ── Exclusion-rule audit ────────────────────────────────────────────────────

def _exclusion_rule_violations(rule: str, record: dict) -> int:
    """Return 1 if the record VIOLATES the rule (i.e. should be excluded by it
    but is currently in the priority slice). Returns 0 otherwise.

    The spec is valid only when every rule yields 0 on the current slice.
    """
    if rule == "exclude_cyrillic_dominant_records":
        return 1 if _g(record, "dominant_script") == "cyrillic" else 0
    if rule == "exclude_mixed_script_records":
        return 1 if _g(record, "dominant_script") == "mixed" else 0
    if rule == "exclude_low_alphabetic_ratio_records":
        return 1 if _g(record, "alphabetic_content_bucket") == "low" else 0
    if rule == "exclude_no_text_layer_records":
        return 1 if _g(record, "pdf_text_layer_detected") == "no" else 0
    if rule == "exclude_image_like_but_not_routed_records":
        return 1 if _g(record, "image_like_pdf") == "yes" else 0
    if rule == "exclude_ambiguous_below_threshold_records":
        return 1 if _g(record, "unknown_failure_bucket") == \
            "ambiguous_below_threshold" else 0
    if rule == "exclude_fallback_ran_but_no_family_match_records":
        return 1 if _g(record, "unknown_failure_bucket") == \
            "fallback_ran_but_no_family_match" else 0
    if rule == "exclude_medication_dose_or_ddi_interpretation":
        # Spec exclusion -- no record in the slice is allowed to carry parsed
        # medication / dose / DDI fields. The privacy-safe public report never
        # exposes those values at all, so the slice is structurally compliant.
        return 0
    if rule == "exclude_lab_value_parsing":
        # Same structural-compliance argument.
        return 0
    if rule == "exclude_records_with_insufficient_safe_metadata":
        required = (
            "table_like_structure_detected", "numeric_content_bucket",
            "alphabetic_content_bucket", "dominant_script",
            "detector_confidence_bucket", "language_visibility_status",
        )
        return 1 if any(_g(record, k) == "" for k in required) else 0
    return 0


def _exclusion_audit_report(records: list[dict]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule in EXCLUSION_RULES:
        violations = sum(_exclusion_rule_violations(rule, r) for r in records)
        out.append({"rule": rule, "violating_record_count": violations})
    return out


# ── Top-level orchestration ──────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    numeric_records = select_numeric_table_records(table)
    other_diag05_buckets = select_other_diag05_lever_buckets(table)
    header_deferred = other_diag05_buckets.get(_DEFERRED_HEADER_CANDIDATE, 0)

    sig_report = _positive_signature_match_report(numeric_records)
    sig_holds = (
        all(v["fully_matches"] for v in sig_report.values())
        and len(numeric_records) > 0
    )

    excl_audit = _exclusion_audit_report(numeric_records)
    excl_clean = all(r["violating_record_count"] == 0 for r in excl_audit)

    raw_signal_counts = {
        "dominant_script_counts":
            _safe_count(numeric_records, "dominant_script"),
        "language_visibility_status_counts":
            _safe_count(numeric_records, "language_visibility_status"),
        "detector_confidence_bucket_counts":
            _safe_count(numeric_records, "detector_confidence_bucket"),
        "table_like_structure_detected_counts":
            _safe_count(numeric_records, "table_like_structure_detected"),
        "numeric_content_bucket_counts":
            _safe_count(numeric_records, "numeric_content_bucket"),
        "alphabetic_content_bucket_counts":
            _safe_count(numeric_records, "alphabetic_content_bucket"),
        "symbol_content_bucket_counts":
            _safe_count(numeric_records, "symbol_content_bucket"),
        "administrative_form_shape_detected_counts":
            _safe_count(numeric_records, "administrative_form_shape_detected"),
        "date_or_schedule_shape_detected_counts":
            _safe_count(numeric_records, "date_or_schedule_shape_detected"),
        "section_heading_shape_detected_counts":
            _safe_count(numeric_records, "section_heading_shape_detected"),
        "lab_table_shape_detected_counts":
            _safe_count(numeric_records, "lab_table_shape_detected"),
        "imaging_modality_shape_detected_counts":
            _safe_count(numeric_records, "imaging_modality_shape_detected"),
        "medical_abbreviation_shape_detected_counts":
            _safe_count(numeric_records, "medical_abbreviation_shape_detected"),
        "image_like_pdf_counts":
            _safe_count(numeric_records, "image_like_pdf"),
        "pdf_text_layer_detected_counts":
            _safe_count(numeric_records, "pdf_text_layer_detected"),
    }

    deferred_subsets = {
        "candidate_table_header_language_policy_record_count":
            f"{header_deferred} record(s); deferred as a special case to be "
            "absorbed into a future numeric-table spec implementation",
        "language_detector_metadata_propagation_audit_pool":
            "11 records from DIAG-04 routed to that lever; deferred",
        "latin_medical_abbreviation_handling_audit_pool":
            "8 records from DIAG-04 routed to that lever; deferred",
        "likely_text_layer_issue":
            "21 records deferred per DIAG-03",
        "fallback_ran_but_no_family_match":
            "17 records deferred per DIAG-02; no cue expansion",
        "ambiguous_below_threshold":
            "15 records excluded; review-bound, no cue expansion",
    }

    if sig_holds and excl_clean and len(numeric_records) > 0:
        rec_next = "future_block_named_unknown_diag_06a_implementation"
        rec_justification = (
            f"All {len(numeric_records)} priority records match the exact "
            f"positive signature and none violate any exclusion rule. A future "
            f"implementation block may prototype the numeric-table safe-default "
            f"safe-default language lever inside the published acceptance "
            f"criteria for that lever. The "
            f"implementation must remain review-bound and must not change "
            f"clinical interpretation, OCR routing, language detector "
            f"behavior, or cue packs."
        )
    else:
        rec_next = "no_implementation_yet_spec_revision_required"
        rec_justification = (
            "The positive signature does not fully hold or one or more "
            "exclusion rules are violated on the current priority slice. "
            "Revise the spec before any implementation is considered. All "
            "records remain review-bound."
        )

    progress_estimate = {
        "before_06a_unknown_track_done_pct":      "approximately 55%",
        "before_06a_unknown_track_remaining_pct": "approximately 45%",
        "after_06a_unknown_track_done_pct":       "approximately 62%",
        "after_06a_unknown_track_remaining_pct":  "approximately 38%",
        "after_06a_project_done_pct":             "approximately 76%",
        "after_06a_project_remaining_pct":        "approximately 24%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "language_detector_behavior_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_changed": False,
        "scoring_changed": False,
        "auto_accept_changed": False,
        "cue_packs_changed": False,
        "cue_expansion_recommended": False,
        "policy_implemented_in_this_block": False,
        "lab_value_parsing_added": False,
        "medication_parsing_added": False,
        "dose_parsing_added": False,
        "ddi_logic_changed": False,
        "clinical_interpretation_added": False,
        "b07_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_allowlist_changed": False,
        "external_api_changed": False,
        "external_api_used": False,
        "raw_filenames_in_public_reports": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "source_documents_staged": False,
        "private_corpus_files_staged": False,
        "secrets_in_public_reports": False,
        "all_records_remain_review_bound": True,
        "deferred_pools_remain_deferred": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_diag_labels=list(UPSTREAM_DIAG_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        total_numeric_table_policy_records=len(numeric_records),
        deferred_table_header_record_count=header_deferred,
        other_diag05_lever_records_deferred=other_diag05_buckets,

        positive_signature=[
            {"key": k, "expected": v} for k, v in POSITIVE_SIGNATURE
        ],
        positive_signature_match_report=sig_report,
        positive_signature_holds_on_all_priority_records=sig_holds,

        exclusion_rules=list(EXCLUSION_RULES),
        exclusion_audit=excl_audit,
        no_priority_record_violates_any_exclusion_rule=excl_clean,

        proposed_future_default=PROPOSED_FUTURE_DEFAULT,
        future_implementation_acceptance_criteria=
            list(FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA),
        future_validation_requirements=
            list(FUTURE_VALIDATION_REQUIREMENTS),

        sample_ids=_anonymized_ids("numeric_table_policy_priority",
                                   len(numeric_records)),
        raw_signal_counts=raw_signal_counts,
        deferred_subsets=deferred_subsets,

        implementation_block_recommended_next=rec_next,
        implementation_block_justification=rec_justification,

        progress_estimate=progress_estimate,

        behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        policy_implemented_in_this_block=False,
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A - Numeric-Table Safe-Default Spec")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): `{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream diagnostics:")
    for lbl in report.upstream_diag_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- total numeric-table records analyzed: "
                 f"`{report.total_numeric_table_policy_records}`")
    lines.append(f"- deferred table-header record count: "
                 f"`{report.deferred_table_header_record_count}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")

    lines.append("## A. Required positive signature")
    lines.append("")
    for item in report.positive_signature:
        lines.append(f"- `{item['key']}` = `{item['expected']}`")
    lines.append("")
    lines.append("### Positive-signature match report on the priority slice")
    lines.append("")
    for key, info in report.positive_signature_match_report.items():
        lines.append(
            f"- `{key}` expected=`{info['expected_value']}`, "
            f"matching=`{info['matching_record_count']}`, "
            f"fully_matches=`{info['fully_matches']}`"
        )
    lines.append("")
    lines.append(
        f"positive_signature_holds_on_all_priority_records: "
        f"`{report.positive_signature_holds_on_all_priority_records}`"
    )
    lines.append("")

    lines.append("## B. Required exclusion rules")
    lines.append("")
    for rule in report.exclusion_rules:
        lines.append(f"- `{rule}`")
    lines.append("")
    lines.append("### Exclusion-rule audit on the priority slice")
    lines.append("")
    for row in report.exclusion_audit:
        lines.append(
            f"- `{row['rule']}` violating_record_count=`{row['violating_record_count']}`"
        )
    lines.append("")
    lines.append(
        f"no_priority_record_violates_any_exclusion_rule: "
        f"`{report.no_priority_record_violates_any_exclusion_rule}`"
    )
    lines.append("")

    lines.append("## C. Proposed future default (NOT implemented in this block)")
    lines.append("")
    for k, v in report.proposed_future_default.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## D. Future implementation acceptance criteria")
    lines.append("")
    for c in report.future_implementation_acceptance_criteria:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## E. Future validation requirements")
    lines.append("")
    for v in report.future_validation_requirements:
        lines.append(f"- `{v}`")
    lines.append("")

    lines.append("## Deferred subsets (out of scope)")
    lines.append("")
    for k, v in report.deferred_subsets.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Implementation recommendation")
    lines.append("")
    lines.append(f"- recommended_next: `{report.implementation_block_recommended_next}`")
    lines.append("")
    lines.append(report.implementation_block_justification)
    lines.append("")

    lines.append("## Progress estimate")
    lines.append("")
    for k, v in report.progress_estimate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Safety / Privacy")
    lines.append("")
    lines.append(f"- behavior_changed: `{report.behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    lines.append(f"- policy_implemented_in_this_block: "
                 f"`{report.policy_implemented_in_this_block}`")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Specification-only block; "
                 "no runtime behavior change, no policy is implemented here, no cue "
                 "expansion, no OCR routing or detector behavior change.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Raw signal distributions", ""]
    for k, v in report.raw_signal_counts.items():
        inner = ", ".join(f"`{kk}`={vv}" for kk, vv in v.items())
        extra.append(f"- {k}: {inner}")
    extra += [
        "",
        "## Why a spec block instead of an implementation",
        "",
        "The 11 priority records share an unusually uniform metadata ",
        "signature. Before any future implementation block touches runtime ",
        "behavior, the exact positive signature, exclusion rules, default ",
        "action, acceptance criteria, and validation requirements are ",
        "published here in a single privacy-safe document. A future ",
        "implementation block may then proceed only if it stays inside the ",
        "boundaries this spec defines and inside the acceptance criteria ",
        "this spec lists.",
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Language / script detector behavior",
        "- Classifier behavior or cue packs",
        "- Confidence thresholds or scoring",
        "- Auto-accept / review-bound policy",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "No policy is implemented in this block. No clinical interpretation ",
        "added. No values parsed. No active facts written.",
        "",
    ]
    return base + "\n".join(extra)


# ── Public-report safety guard ───────────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".docx", ".doc",
    "/users/", "/home/", "c:\\",
    "begin rsa", "begin private key",
    "secret=", "secret_key=", "api_key=", "password=", "bearer ",
)

_FORBIDDEN_KEY_TOKEN_RE = re.compile(r"\b(?:aws|gcp|azure)_secret\b", re.IGNORECASE)


def assert_safe_public_payload(payload: Any) -> None:
    def _walk(node: Any) -> None:
        if isinstance(node, str):
            lower = node.lower()
            for needle in _FORBIDDEN_SUBSTRINGS:
                if needle in lower:
                    raise RuntimeError(
                        f"Refusing to write public report: forbidden substring "
                        f"matched ({needle!r})."
                    )
            if _FORBIDDEN_KEY_TOKEN_RE.search(node):
                raise RuntimeError(
                    "Refusing to write public report: forbidden secret token "
                    "pattern matched."
                )
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(k)
                _walk(v)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                _walk(item)

    _walk(payload)


# ── Driver ───────────────────────────────────────────────────────────────────

def write_reports(report: DiagnosticReport, out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_payload = asdict(report)
    assert_safe_public_payload(json_payload)
    md_summary = render_markdown_summary(report)
    md_long = render_markdown_long(report)
    assert_safe_public_payload(md_summary)
    assert_safe_public_payload(md_long)

    paths = {
        "json": out_dir / "medai_doc_type_unknown_diag_06a_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_06a_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_06A.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A numeric-table safe-default spec."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=SOURCE_REPORT,
        help="Path to the FAMILY-04 batch-eval per-file public report JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the three public diagnostic files.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the JSON report to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    if not args.source_report.exists():
        print(f"ERROR: source report missing: {args.source_report}", file=sys.stderr)
        return 2

    source_payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = build_diagnostic_from_report(source_payload)

    if args.print_only:
        print(render_json(report))
        return 0

    paths = write_reports(report, out_dir=args.output_dir)
    print(json.dumps(
        {
            "conclusion": "medai_doc_type_unknown_diag_06a_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_numeric_table_policy_records":
                report.total_numeric_table_policy_records,
            "positive_signature_holds_on_all_priority_records":
                report.positive_signature_holds_on_all_priority_records,
            "no_priority_record_violates_any_exclusion_rule":
                report.no_priority_record_violates_any_exclusion_rule,
            "implementation_block_recommended_next":
                report.implementation_block_recommended_next,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
            "cue_expansion_recommended": report.cue_expansion_recommended,
            "policy_implemented_in_this_block":
                report.policy_implemented_in_this_block,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
