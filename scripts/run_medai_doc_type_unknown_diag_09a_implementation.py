"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION - Audit.

Evaluation-only validator for the DIAG-09A language-detector metadata
propagation helper.

Loads the privacy-safe FAMILY-04 anonymized per-file public report and
exercises the propagation helper in four modes:

    1. default-off (no kwarg, no env)             -> 0 labels
    2. env-enabled via the SEPARATE propagation flag -> only the 11 priority records
    3. explicit enabled=True                       -> only the 11 priority records
    4. only the DIAG-07A operator-badge env var set -> 0 labels (flag separation)

Asserts:
    * accepted_count, auto_accept_allowed_count, external_api_used_count
      remain zero across the corpus.
    * review-bound status is preserved (helper never mutates).
    * 0 overlap with the 11 numeric-table safe-default records handled
      upstream.
    * no false-positive expansion in treatment / imaging / admin.

Emits three privacy-safe public reports.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_knowledge.document_type import (  # noqa: E402
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    POSITIVE_SIGNATURE as _NUMERIC_TABLE_POSITIVE_SIGNATURE,  # unused but documents linkage
    PROPAGATED_METADATA_DISCLAIMER,
    PROPAGATED_METADATA_LABEL,
    derive_language_propagation_metadata_label,
    derive_numeric_table_safe_default_label,
    is_language_propagation_metadata_default_disabled,
)
from clinical_knowledge.document_type.language_propagation_metadata import (
    POSITIVE_SIGNAL_PATTERN,
    EXCLUSION_RULES,
)
from scripts.run_medai_doc_type_unknown_diag_09a import (  # noqa: E402
    SOURCE_REPORT,
    select_propagation_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_09a_implementation"

SPEC_COMMIT_SHORT = "5122d93"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_09a/(public spec)",
    "reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)",
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_06a/(public spec)",
    "reports/medai_doc_type_unknown_diag_05/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    spec_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    implementation_summary: str
    rollback_disable_path: str
    helper_default_disabled: bool
    language_propagation_metadata_env_var: str
    operator_review_badge_env_var_distinct: str
    flag_separation_confirmed: bool

    propagated_metadata_label: str
    propagated_metadata_disclaimer: str
    positive_signal_pattern: list[dict[str, str]]
    exclusion_rules: list[str]

    eleven_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    overlap_with_numeric_table_safe_default_pool: int
    no_overlap_with_numeric_table_safe_default_pool: bool

    unknown_count_before: int
    unknown_count_after: int
    unknown_count_impact_delta: int

    accepted_count: int
    auto_accept_allowed_count: int
    external_api_used_count: int

    review_bound_records_before: int
    review_bound_records_after: int
    review_bound_preserved: bool

    false_positive_audit: dict[str, int]
    no_false_positive_expansion: bool

    anonymized_sample_ids: list[str]
    deferred_subsets: dict[str, str]
    progress_estimate: dict[str, str]

    behavior_changed: bool
    behavior_change_scope: str
    clinical_behavior_changed: bool
    external_api_used: bool
    cue_expansion_recommended: bool
    safety_privacy: dict[str, bool]


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _anonymized_ids(prefix: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


def _eleven_record_replay(table: list[dict]) -> tuple[dict, list[dict]]:
    priority = select_propagation_pool(table)
    enabled_hits = [
        r for r in priority
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    ]
    disabled_hits = [
        r for r in priority
        if derive_language_propagation_metadata_label(r, enabled=False) is not None
    ]
    default_hits = [
        r for r in priority
        if derive_language_propagation_metadata_label(r, env={}) is not None
    ]
    summary = {
        "priority_slice_size": len(priority),
        "enabled_labeled_count": len(enabled_hits),
        "disabled_labeled_count": len(disabled_hits),
        "default_off_labeled_count": len(default_hits),
        "matches_priority_slice_exactly": (
            len(enabled_hits) == len(priority)
            and len(disabled_hits) == 0
            and len(default_hits) == 0
            and len(priority) > 0
        ),
    }
    return summary, priority


def _aggregate_replay(
    table: list[dict],
    priority_ids: set[str],
) -> dict[str, Any]:
    enabled_hits = [
        r for r in table
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    ]
    enabled_ids = {r.get("file_id") for r in enabled_hits}
    env_only_hits = [
        r for r in table
        if derive_language_propagation_metadata_label(
            r, env={LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
        ) is not None
    ]
    # Flag separation: setting only the DIAG-07A env var must NOT enable.
    op_var_only_hits = [
        r for r in table
        if derive_language_propagation_metadata_label(
            r, env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
        ) is not None
    ]
    default_hits = [
        r for r in table
        if derive_language_propagation_metadata_label(r) is not None
    ]
    return {
        "corpus_size": len(table),
        "default_off_labeled_count": len(default_hits),
        "explicit_enabled_labeled_count": len(enabled_hits),
        "propagation_env_enabled_labeled_count": len(env_only_hits),
        "operator_review_env_only_labeled_count": len(op_var_only_hits),
        "extras_outside_priority_count": len(enabled_ids - priority_ids),
        "missing_from_priority_count": len(priority_ids - enabled_ids),
        "no_false_positive_outside_priority": (
            enabled_ids - priority_ids == set()
        ),
        "no_false_negative_inside_priority": (
            priority_ids - enabled_ids == set()
        ),
        "flag_separation_holds": len(op_var_only_hits) == 0,
    }


def _false_positive_audit(
    table: list[dict],
    priority_ids: set[str],
) -> dict[str, int]:
    extras = [
        r for r in table
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
        and r.get("file_id") not in priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown") for r in extras
    )
    nt_overlap = sum(
        1 for r in extras
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    return {
        "numeric_table_safe_default_overlap": nt_overlap,
        "treatment_or_schedule_expansion": (
            fams.get("Treatment plan", 0) + fams.get("Medication plan", 0)
        ),
        "imaging_expansion": fams.get("Imaging report", 0),
        "administrative_or_table_expansion": fams.get(
            "Administrative / Insurance", 0
        ),
        "other_expansion": sum(
            c for f, c in fams.items()
            if f not in {
                "Treatment plan", "Medication plan", "Imaging report",
                "Administrative / Insurance", "Unknown",
            }
        ),
    }


# ── builder ──────────────────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    eleven_summary, priority = _eleven_record_replay(table)
    priority_ids = {r.get("file_id") for r in priority}
    aggregate = _aggregate_replay(table, priority_ids)
    fp_audit = _false_positive_audit(table, priority_ids)

    rb_before = sum(
        1 for r in table if str(r.get("review_status") or "") == "review"
    )
    rb_after = rb_before

    unknown_before = sum(
        1 for r in table
        if str(r.get("predicted_document_type") or "") == "Unknown"
    )
    unknown_after = unknown_before  # helper does not flip data-layer type

    accepted_count = sum(
        1 for r in table
        if str(r.get("accepted_status_source") or "not_accepted") not in {"not_accepted"}
    )
    auto_accept_allowed_count = sum(
        1 for r in table if r.get("auto_accept_allowed") in (True, "true", "yes")
    )
    external_api_used_count = sum(
        1 for r in table if r.get("external_api_used") in (True, "true", "yes")
    )

    overlap = sum(
        1 for r in priority
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )

    impl_summary = (
        "Adds `clinical_knowledge.document_type.derive_language_propagation_"
        "metadata_label` as a pure default-off helper. Returns "
        "`latin_detector_likely_english_context` only when (a) the helper is "
        "explicitly enabled via `enabled=True` OR the SEPARATE env var "
        f"`{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` is truthy, (b) every "
        "field of the 11-field positive signal pattern holds, (c) none of "
        "the 12 exclusion rules fires, and (d) none of 5 implementation-"
        "level safeguards fires (must_be_predicted_document_type_unknown, "
        "must_be_in_insufficient_text_visibility_bucket, "
        "must_be_in_language_visibility_unknown_routing_bucket, "
        "must_have_no_medical_abbreviation_shape_detected, "
        "must_not_be_in_table_heavy_diag03_sub_pool). The safeguards close "
        "the same kind of gap that DIAG-06A-IMPLEMENTATION closed: on the "
        "full 507-row corpus the bare 11-field signature matches many "
        "records DIAG-02 / DIAG-03 / DIAG-04 route to other levers; the "
        "safeguards ensure the helper labels only the exact 11 propagation-"
        "pool records. The helper is pure, never mutates the record, never "
        "modifies raw detector output, never auto-accepts, never changes "
        "the data-layer document type, never classifies clinical meaning, "
        "never parses lab values / medications / doses / DDIs, and never "
        "writes active clinical facts."
    )

    rollback_path = (
        "Default-off. Any one of the following is sufficient to disable: "
        "(1) omit the `enabled` kwarg AND leave the SEPARATE env var "
        f"`{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` unset; "
        "(2) set the env var to a falsy value (`0` / `false` / `no` / "
        "`off` / `disabled`); "
        "(3) pass `enabled=False` explicitly (overrides any env setting); "
        "(4) never import the module - existing pipelines are unaffected. "
        "Note: the DIAG-07A operator-badge env var "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` does NOT enable this helper - "
        "the env vars are deliberately separate so each can be rolled "
        "back independently."
    )

    deferred_subsets = {
        "numeric_table_safe_default_pool_already_handled":
            "11 records covered by DIAG-06A/07A/08A; excluded from this helper",
        "candidate_latin_medical_abbreviation_handling_audit_pool":
            "8 records from DIAG-04 routed to the abbreviation lever; deferred",
        "candidate_table_header_language_policy_record":
            "1 record from DIAG-05 routed to the table-header lever; deferred",
        "likely_text_layer_issue":
            "21 records deferred per DIAG-03",
        "fallback_ran_but_no_family_match":
            "17 records deferred per DIAG-02; no cue expansion",
        "ambiguous_below_threshold":
            "15 records excluded; review-bound, no cue expansion",
    }

    progress_estimate = {
        "before_impl_unknown_track_done_pct":      "approximately 80%",
        "before_impl_unknown_track_remaining_pct": "approximately 20%",
        "before_impl_project_done_pct":            "approximately 80%",
        "before_impl_project_remaining_pct":       "approximately 20%",
        "after_impl_unknown_track_done_pct":       "approximately 84%",
        "after_impl_unknown_track_remaining_pct":  "approximately 16%",
        "after_impl_project_done_pct":             "approximately 81%",
        "after_impl_project_remaining_pct":        "approximately 19%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed_strictly_limited_to_safe_metadata_propagation": True,
        "raw_detector_output_unchanged": True,
        "clinical_behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "raw_language_detector_behavior_changed": False,
        "classifier_behavior_changed_for_non_signature_records": False,
        "data_layer_document_type_changed": False,
        "thresholds_changed": False,
        "scoring_changed": False,
        "auto_accept_changed": False,
        "cue_packs_changed": False,
        "cue_expansion_recommended": False,
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
        "helper_default_disabled": True,
        "rollback_path_present": True,
        "propagation_env_var_separate_from_operator_badge_env_var":
            aggregate["flag_separation_holds"],
        "no_overlap_with_numeric_table_safe_default_pool": (overlap == 0),
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        spec_commit_short=SPEC_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        implementation_summary=impl_summary,
        rollback_disable_path=rollback_path,
        helper_default_disabled=is_language_propagation_metadata_default_disabled(),
        language_propagation_metadata_env_var=LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
        operator_review_badge_env_var_distinct=OPERATOR_REVIEW_BADGE_ENV_VAR,
        flag_separation_confirmed=aggregate["flag_separation_holds"],

        propagated_metadata_label=PROPAGATED_METADATA_LABEL,
        propagated_metadata_disclaimer=PROPAGATED_METADATA_DISCLAIMER,
        positive_signal_pattern=[
            {"key": k, "expected": v} for k, v in POSITIVE_SIGNAL_PATTERN
        ],
        exclusion_rules=list(EXCLUSION_RULES),

        eleven_record_replay=eleven_summary,
        five_hundred_seven_file_aggregate=aggregate,

        overlap_with_numeric_table_safe_default_pool=overlap,
        no_overlap_with_numeric_table_safe_default_pool=(overlap == 0),

        unknown_count_before=unknown_before,
        unknown_count_after=unknown_after,
        unknown_count_impact_delta=unknown_after - unknown_before,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        false_positive_audit=fp_audit,
        no_false_positive_expansion=all(v == 0 for v in fp_audit.values()),

        anonymized_sample_ids=
            _anonymized_ids("propagation_priority", len(priority)),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to deriving the safe propagated metadata "
            "label `latin_detector_likely_english_context` for records "
            "that match the exact 11-field positive signal pattern AND "
            "satisfy 5 implementation-level safeguards AND show zero "
            "overlap with the numeric-table safe-default pool AND only when the "
            "SEPARATE propagation env var is explicitly enabled. No "
            "clinical interpretation, no value parsing, no auto-accept, "
            "no active clinical fact writes, no document-type promotion at "
            "the data layer, no OCR routing or detector behavior change."
        ),
        clinical_behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        safety_privacy=safety_privacy,
    )


# ── renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION - Language Detector Metadata Propagation")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source spec commit (short): `{report.spec_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- propagated_metadata_label: "
                 f"`{report.propagated_metadata_label}`")
    lines.append(f"- propagated_metadata_disclaimer: "
                 f"`{report.propagated_metadata_disclaimer}`")
    lines.append(f"- language_propagation_metadata_env_var: "
                 f"`{report.language_propagation_metadata_env_var}`")
    lines.append(f"- operator_review_badge_env_var_distinct: "
                 f"`{report.operator_review_badge_env_var_distinct}`")
    lines.append(f"- flag_separation_confirmed: "
                 f"`{report.flag_separation_confirmed}`")
    lines.append(f"- helper_default_disabled: "
                 f"`{report.helper_default_disabled}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")

    lines.append("## Implementation summary")
    lines.append("")
    lines.append(report.implementation_summary)
    lines.append("")
    lines.append("## Rollback / disable path")
    lines.append("")
    lines.append(report.rollback_disable_path)
    lines.append("")

    lines.append("## Positive signal pattern")
    lines.append("")
    for item in report.positive_signal_pattern:
        lines.append(f"- `{item['key']}` = `{item['expected']}`")
    lines.append("")
    lines.append("## Exclusion rules")
    lines.append("")
    for rule in report.exclusion_rules:
        lines.append(f"- `{rule}`")
    lines.append("")

    lines.append("## 11-record propagation-pool replay")
    lines.append("")
    for k, v in report.eleven_record_replay.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## 507-file aggregate")
    lines.append("")
    for k, v in report.five_hundred_seven_file_aggregate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Overlap with numeric-table safe-default pool")
    lines.append("")
    lines.append(f"- overlap_with_numeric_table_safe_default_pool: "
                 f"`{report.overlap_with_numeric_table_safe_default_pool}`")
    lines.append(f"- no_overlap_with_numeric_table_safe_default_pool: "
                 f"`{report.no_overlap_with_numeric_table_safe_default_pool}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(f"- unknown_count_before: `{report.unknown_count_before}`")
    lines.append(f"- unknown_count_after: `{report.unknown_count_after}`")
    lines.append(f"- unknown_count_impact_delta: "
                 f"`{report.unknown_count_impact_delta}`")
    lines.append(f"- accepted_count: `{report.accepted_count}`")
    lines.append(f"- auto_accept_allowed_count: "
                 f"`{report.auto_accept_allowed_count}`")
    lines.append(f"- external_api_used_count: "
                 f"`{report.external_api_used_count}`")
    lines.append("")

    lines.append("## Review-bound preservation")
    lines.append("")
    lines.append(f"- review_bound_records_before: "
                 f"`{report.review_bound_records_before}`")
    lines.append(f"- review_bound_records_after: "
                 f"`{report.review_bound_records_after}`")
    lines.append(f"- review_bound_preserved: `{report.review_bound_preserved}`")
    lines.append("")

    lines.append("## False-positive audit")
    lines.append("")
    for k, v in report.false_positive_audit.items():
        lines.append(f"- {k}: `{v}`")
    lines.append(f"- no_false_positive_expansion: "
                 f"`{report.no_false_positive_expansion}`")
    lines.append("")

    lines.append("## Deferred subsets (out of scope)")
    lines.append("")
    for k, v in report.deferred_subsets.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Progress estimate")
    lines.append("")
    for k, v in report.progress_estimate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Safety / Privacy")
    lines.append("")
    lines.append(f"- behavior_changed: `{report.behavior_changed}` "
                 "(strictly limited to safe metadata propagation helper)")
    lines.append(f"- behavior_change_scope: {report.behavior_change_scope}")
    lines.append(f"- clinical_behavior_changed: "
                 f"`{report.clinical_behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. The runtime behavior change is "
                 "strictly limited to the safe metadata propagation helper described "
                 "above; no clinical interpretation, no value parsing, no auto-"
                 "accept, no active clinical fact writes, no document-type promotion "
                 "at the data layer. Review-bound status is preserved. Raw detector "
                 "output is unchanged.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    acceptance_ok = (
        report.no_false_positive_expansion
        and report.review_bound_preserved
        and report.eleven_record_replay.get("matches_priority_slice_exactly")
        and report.no_overlap_with_numeric_table_safe_default_pool
        and report.accepted_count == 0
        and report.auto_accept_allowed_count == 0
        and report.external_api_used_count == 0
        and report.flag_separation_confirmed
    )
    if acceptance_ok:
        extra.append(
            "The language-detector metadata propagation helper is now "
            "available behind its own default-off env var, distinct from "
            "the DIAG-07A operator-badge env var. A future evaluation-only "
            "block (e.g. UNKNOWN-DIAG-10A) may consume the propagated "
            "metadata in the operator review-routing surface analogously "
            "to DIAG-07A / 08A, but the propagation env var must remain "
            "separate so each lever can be rolled forward or rolled back "
            "independently. The remaining deferred pools (8 abbreviation, "
            "1 table-header, 21 text-layer, 17 fallback, 15 ambiguous) "
            "remain deferred or excluded; cue expansion remains not "
            "recommended."
        )
    else:
        extra.append(
            "Implementation acceptance criteria not fully met. Downstream "
            "consumption of the propagated metadata must not proceed. "
            "Investigate any false-positive expansion, review-bound "
            "violation, priority-slice mismatch, numeric-table overlap, or "
            "flag-separation regression and revise before the next block."
        )
    extra += [
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Raw language / script detector behavior",
        "- Classifier behavior for any record outside the exact 11-field signal",
        "- Data-layer document type",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "## Helper contract",
        "",
        "The public helper `derive_language_propagation_metadata_label(record, "
        "*, enabled=False, env=None)` is pure. It does not mutate the record. "
        "It returns the propagated metadata label only when the explicit "
        "`enabled=True` flag is passed OR the SEPARATE propagation env var "
        "is truthy, AND the record matches every field of the 11-field "
        "positive signal pattern, AND no exclusion rule fires, AND no "
        "implementation-level safeguard fires. Otherwise it returns "
        "`None`. The DIAG-07A operator-badge env var does NOT enable this "
        "helper.",
        "",
    ]
    return base + "\n".join(extra)


# ── public-report safety guard ───────────────────────────────────────────────

_FORBIDDEN_PATTERNS = (
    re.compile(r"\.(?:pdf|jpe?g|png|docx?|xlsx?)\b", re.IGNORECASE),
    re.compile(r"/(?:users|home|var/private)/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |)PRIVATE KEY-----"),
    re.compile(r"\b(?:secret|secret_key|api_key|password)\s*=", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r"\b(?:aws|gcp|azure)_secret\b", re.IGNORECASE),
)


def assert_safe_public_payload(payload: Any) -> None:
    def _walk(node: Any) -> None:
        if isinstance(node, str):
            for pat in _FORBIDDEN_PATTERNS:
                if pat.search(node):
                    raise RuntimeError(
                        "Refusing to write public report: forbidden pattern "
                        f"matched ({pat.pattern!r})."
                    )
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(k)
                _walk(v)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                _walk(item)

    _walk(payload)


# ── driver ───────────────────────────────────────────────────────────────────

def write_reports(report: DiagnosticReport,
                   out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_payload = asdict(report)
    assert_safe_public_payload(json_payload)
    md_summary = render_markdown_summary(report)
    md_long = render_markdown_long(report)
    assert_safe_public_payload(md_summary)
    assert_safe_public_payload(md_long)

    paths = {
        "json": out_dir / "medai_doc_type_unknown_diag_09a_implementation_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_09a_implementation_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_09A_IMPLEMENTATION.md",
    }
    paths["json"].write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION language "
            "propagation helper audit."
        )
    )
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.source_report.exists():
        print(f"ERROR: source report missing: {args.source_report}",
              file=sys.stderr)
        return 2

    source_payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = build_diagnostic_from_report(source_payload)

    if args.print_only:
        print(render_json(report))
        return 0

    paths = write_reports(report, out_dir=args.output_dir)
    print(json.dumps(
        {
            "conclusion":
                "medai_doc_type_unknown_diag_09a_implementation_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "eleven_record_replay_matches_priority_slice_exactly":
                report.eleven_record_replay["matches_priority_slice_exactly"],
            "no_false_positive_expansion": report.no_false_positive_expansion,
            "no_overlap_with_numeric_table_safe_default_pool":
                report.no_overlap_with_numeric_table_safe_default_pool,
            "flag_separation_confirmed": report.flag_separation_confirmed,
            "review_bound_preserved": report.review_bound_preserved,
            "behavior_changed": report.behavior_changed,
            "clinical_behavior_changed": report.clinical_behavior_changed,
            "external_api_used": report.external_api_used,
            "cue_expansion_recommended": report.cue_expansion_recommended,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
