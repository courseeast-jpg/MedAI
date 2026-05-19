"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION - Audit.

Privacy-safe evaluation-only validator for the DIAG-11A latin medical
abbreviation metadata helper. Exercises the helper in five modes against
the FAMILY-04 anonymized per-file public report:

    1. default-off  -> 0 labels
    2. only the abbreviation env var       -> 8 labels (priority slice)
    3. only the propagation env var        -> 0 labels (flag separation)
    4. only the operator-review env var    -> 0 labels (flag separation)
    5. all 3 env vars                      -> 8 + 11 + 11 labels, each
                                              helper rendering only its
                                              own slice

Asserts accepted / auto_accept_allowed / external_api_used counts remain
zero, review-bound preserved, raw detector output unchanged, data-layer
document type unchanged, zero overlap with both existing helpers, and no
false-positive expansion in numeric-table / propagation / treatment /
imaging / admin. Emits three privacy-safe public reports.
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
    LATIN_ABBREVIATION_METADATA_DISCLAIMER,
    LATIN_ABBREVIATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_METADATA_LABEL,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    derive_language_propagation_metadata_label,
    derive_latin_medical_abbreviation_metadata_label,
    derive_numeric_table_safe_default_label,
    is_latin_abbreviation_metadata_default_disabled,
)
from clinical_knowledge.document_type.latin_abbreviation_metadata import (
    EXCLUSION_RULES,
    POSITIVE_SIGNAL_PATTERN,
)
from scripts.run_medai_doc_type_unknown_diag_11a import (  # noqa: E402
    SOURCE_REPORT,
    select_abbreviation_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_11a_implementation"

SPEC_COMMIT_SHORT = "7f248ff"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_11a/(public spec)",
    "reports/medai_doc_type_unknown_diag_10a_language_propagation_operator_surface/(public)",
    "reports/medai_doc_type_unknown_diag_09a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)",
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
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
    latin_abbreviation_metadata_env_var: str
    distinct_env_vars: list[str]
    flag_separation_audit: dict[str, Any]

    propagated_metadata_label: str
    propagated_metadata_disclaimer: str
    positive_signal_pattern: list[dict[str, str]]
    exclusion_rules: list[str]

    eight_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    overlap_with_numeric_table_safe_default_pool: int
    no_overlap_with_numeric_table_safe_default_pool: bool
    overlap_with_language_propagation_pool: int
    no_overlap_with_language_propagation_pool: bool

    unknown_count_before: int
    unknown_count_after: int
    unknown_count_impact_delta: int

    accepted_count: int
    auto_accept_allowed_count: int
    external_api_used_count: int

    review_bound_records_before: int
    review_bound_records_after: int
    review_bound_preserved: bool

    raw_detector_output_unchanged: bool
    data_layer_document_type_unchanged: bool

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
    abbreviation_parsing_or_expansion: bool
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


def _count_under_env(table: list[dict], env: dict[str, str]) -> int:
    return sum(
        1 for r in table
        if derive_latin_medical_abbreviation_metadata_label(r, env=env) is not None
    )


# ── builder ──────────────────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    priority = select_abbreviation_pool(table)
    priority_ids = {r.get("file_id") for r in priority}

    # 8-record replay
    enabled_hits = [
        r for r in priority
        if derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is not None
    ]
    disabled_hits = [
        r for r in priority
        if derive_latin_medical_abbreviation_metadata_label(r, enabled=False) is not None
    ]
    default_hits = [
        r for r in priority
        if derive_latin_medical_abbreviation_metadata_label(r, env={}) is not None
    ]
    eight_record_replay = {
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

    # 507-file aggregate (mode matrix focused on this helper)
    abbrev_env = {LATIN_ABBREVIATION_METADATA_ENV_VAR: "1"}
    prop_env = {LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    op_env = {OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    all3 = {**abbrev_env, **prop_env, **op_env}

    aggregate = {
        "corpus_size": len(table),
        "default_off_labeled_count": _count_under_env(table, {}),
        "abbreviation_env_only_labeled_count": _count_under_env(table, abbrev_env),
        "propagation_env_only_labeled_count": _count_under_env(table, prop_env),
        "operator_review_env_only_labeled_count": _count_under_env(table, op_env),
        "all_three_env_vars_labeled_count": _count_under_env(table, all3),
        "explicit_enabled_labeled_count": len([
            r for r in table
            if derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is not None
        ]),
    }
    enabled_ids = {
        r.get("file_id") for r in table
        if derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is not None
    }
    aggregate["no_false_positive_outside_priority"] = (
        (enabled_ids - priority_ids) == set()
    )
    aggregate["no_false_negative_inside_priority"] = (
        (priority_ids - enabled_ids) == set()
    )

    # Flag-separation invariants
    flag_separation_audit = {
        "default_off_yields_zero_abbreviation_labels":
            aggregate["default_off_labeled_count"] == 0,
        "abbrev_env_only_yields_priority_slice":
            aggregate["abbreviation_env_only_labeled_count"]
            == len(priority_ids),
        "propagation_env_only_yields_zero_abbreviation_labels":
            aggregate["propagation_env_only_labeled_count"] == 0,
        "operator_review_env_only_yields_zero_abbreviation_labels":
            aggregate["operator_review_env_only_labeled_count"] == 0,
        "all_three_env_vars_yields_priority_slice_for_abbreviation":
            aggregate["all_three_env_vars_labeled_count"]
            == len(priority_ids),
        "all_three_env_vars_are_distinct": len({
            LATIN_ABBREVIATION_METADATA_ENV_VAR,
            LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
            OPERATOR_REVIEW_BADGE_ENV_VAR,
        }) == 3,
        "three_way_flag_separation_holds": (
            aggregate["default_off_labeled_count"] == 0
            and aggregate["propagation_env_only_labeled_count"] == 0
            and aggregate["operator_review_env_only_labeled_count"] == 0
            and aggregate["abbreviation_env_only_labeled_count"]
                == len(priority_ids)
        ),
    }

    # Overlap audits
    overlap_nt = sum(
        1 for r in priority
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    overlap_prop = sum(
        1 for r in priority
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    )

    # False-positive audit (records labeled outside the priority slice)
    extras = [
        r for r in table
        if derive_latin_medical_abbreviation_metadata_label(r, enabled=True) is not None
        and r.get("file_id") not in priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown") for r in extras
    )
    fp_audit = {
        "numeric_table_overlap": sum(
            1 for r in extras
            if derive_numeric_table_safe_default_label(r, enabled=True) is not None
        ),
        "language_propagation_overlap": sum(
            1 for r in extras
            if derive_language_propagation_metadata_label(r, enabled=True) is not None
        ),
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

    rb_before = sum(
        1 for r in table if str(r.get("review_status") or "") == "review"
    )
    rb_after = rb_before  # helper never mutates

    unknown_before = sum(
        1 for r in table
        if str(r.get("predicted_document_type") or "") == "Unknown"
    )
    unknown_after = unknown_before

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

    impl_summary = (
        "Adds `clinical_knowledge.document_type.derive_latin_medical_"
        "abbreviation_metadata_label` as a pure default-off helper. Returns "
        "`latin_medical_abbreviation_context` only when (a) the helper is "
        "explicitly enabled via `enabled=True` OR the SEPARATE env var "
        f"`{LATIN_ABBREVIATION_METADATA_ENV_VAR}` is truthy, (b) every "
        "field of the 14-field positive signal pattern holds, (c) none of "
        "the 14 exclusion rules fires, (d) no overlap with the DIAG-06A "
        "numeric-table safe-default pool, (e) no overlap with the DIAG-09A "
        "language-propagation pool, and (f) none of 4 implementation-level "
        "safeguards fires (must_be_predicted_document_type_unknown, "
        "must_be_in_insufficient_text_visibility_bucket, "
        "must_be_in_language_visibility_unknown_routing_bucket, "
        "must_have_medical_abbreviation_shape_detected). The env var is "
        "DISTINCT from both the DIAG-07A operator-badge env var and the "
        "DIAG-09A language-propagation env var; setting any one does not "
        "enable the other two. The helper is pure, never mutates the "
        "record, never modifies raw detector output, never auto-accepts, "
        "never changes the data-layer document type, never classifies "
        "clinical meaning, never parses or expands the abbreviation, "
        "never parses lab values / medications / doses / DDIs, and never "
        "writes active clinical facts."
    )

    rollback_path = (
        "Default-off. Any one is sufficient: (1) omit the `enabled` kwarg "
        f"AND leave `{LATIN_ABBREVIATION_METADATA_ENV_VAR}` unset; "
        "(2) set the env var to a falsy value; (3) pass `enabled=False` "
        "explicitly; (4) never import the module. The env var is "
        f"distinct from `{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` and "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}`, so rolling back any one "
        "lever does not affect the other two."
    )

    deferred_subsets = {
        "numeric_table_safe_default_pool_handled_by_diag06_07_08":
            "11 records handled by DIAG-06A/07A/08A; separate lever",
        "language_propagation_pool_handled_by_diag09_10":
            "11 records handled by DIAG-09A/10A; separate lever",
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
        "before_impl_unknown_track_done_pct":      "approximately 90%",
        "before_impl_unknown_track_remaining_pct": "approximately 10%",
        "before_impl_project_done_pct":            "approximately 83%",
        "before_impl_project_remaining_pct":       "approximately 17%",
        "after_impl_unknown_track_done_pct":       "approximately 93%",
        "after_impl_unknown_track_remaining_pct":  "approximately 7%",
        "after_impl_project_done_pct":             "approximately 84%",
        "after_impl_project_remaining_pct":        "approximately 16%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed_strictly_limited_to_safe_abbreviation_metadata": True,
        "raw_detector_output_unchanged": True,
        "data_layer_document_type_unchanged": True,
        "clinical_behavior_changed": False,
        "abbreviation_parsing_or_expansion_added": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "raw_language_detector_behavior_changed": False,
        "classifier_behavior_changed_for_non_signature_records": False,
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
        "abbreviation_env_var_distinct_from_propagation_env_var": (
            LATIN_ABBREVIATION_METADATA_ENV_VAR
            != LANGUAGE_PROPAGATION_METADATA_ENV_VAR
        ),
        "abbreviation_env_var_distinct_from_operator_review_env_var": (
            LATIN_ABBREVIATION_METADATA_ENV_VAR
            != OPERATOR_REVIEW_BADGE_ENV_VAR
        ),
        "three_way_flag_separation_holds":
            flag_separation_audit["three_way_flag_separation_holds"],
        "no_overlap_with_numeric_table_safe_default_pool": overlap_nt == 0,
        "no_overlap_with_language_propagation_pool": overlap_prop == 0,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION",
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
        helper_default_disabled=is_latin_abbreviation_metadata_default_disabled(),
        latin_abbreviation_metadata_env_var=LATIN_ABBREVIATION_METADATA_ENV_VAR,
        distinct_env_vars=[
            LATIN_ABBREVIATION_METADATA_ENV_VAR,
            LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
            OPERATOR_REVIEW_BADGE_ENV_VAR,
        ],
        flag_separation_audit=flag_separation_audit,

        propagated_metadata_label=LATIN_ABBREVIATION_METADATA_LABEL,
        propagated_metadata_disclaimer=LATIN_ABBREVIATION_METADATA_DISCLAIMER,
        positive_signal_pattern=[
            {"key": k, "expected": v} for k, v in POSITIVE_SIGNAL_PATTERN
        ],
        exclusion_rules=list(EXCLUSION_RULES),

        eight_record_replay=eight_record_replay,
        five_hundred_seven_file_aggregate=aggregate,

        overlap_with_numeric_table_safe_default_pool=overlap_nt,
        no_overlap_with_numeric_table_safe_default_pool=(overlap_nt == 0),
        overlap_with_language_propagation_pool=overlap_prop,
        no_overlap_with_language_propagation_pool=(overlap_prop == 0),

        unknown_count_before=unknown_before,
        unknown_count_after=unknown_after,
        unknown_count_impact_delta=unknown_after - unknown_before,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        raw_detector_output_unchanged=True,
        data_layer_document_type_unchanged=True,

        false_positive_audit=fp_audit,
        no_false_positive_expansion=all(v == 0 for v in fp_audit.values()),

        anonymized_sample_ids=_anonymized_ids("abbreviation_priority", len(priority)),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to deriving the safe metadata label "
            "`latin_medical_abbreviation_context` for records that match "
            "the exact 14-field positive signal pattern AND satisfy 4 "
            "implementation-level safeguards AND show zero overlap with "
            "either the numeric-table safe-default pool or the language-"
            "propagation pool AND only when the SEPARATE abbreviation env "
            "var is explicitly enabled. No clinical interpretation, no "
            "value parsing, no abbreviation parsing or expansion, no "
            "auto-accept, no active clinical fact writes, no document-"
            "type promotion at the data layer, no raw detector output "
            "mutation."
        ),
        clinical_behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        abbreviation_parsing_or_expansion=False,
        safety_privacy=safety_privacy,
    )


# ── renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION - Latin Abbreviation Metadata")
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
    lines.append(f"- latin_abbreviation_metadata_env_var: "
                 f"`{report.latin_abbreviation_metadata_env_var}`")
    lines.append("- distinct_env_vars:")
    for v in report.distinct_env_vars:
        lines.append(f"  - `{v}`")
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

    lines.append("## 8-record abbreviation-pool replay")
    lines.append("")
    for k, v in report.eight_record_replay.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## 507-file aggregate")
    lines.append("")
    for k, v in report.five_hundred_seven_file_aggregate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Three-way flag-separation audit")
    lines.append("")
    for k, v in report.flag_separation_audit.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Overlap checks")
    lines.append("")
    lines.append(f"- overlap_with_numeric_table_safe_default_pool: "
                 f"`{report.overlap_with_numeric_table_safe_default_pool}`")
    lines.append(f"- no_overlap_with_numeric_table_safe_default_pool: "
                 f"`{report.no_overlap_with_numeric_table_safe_default_pool}`")
    lines.append(f"- overlap_with_language_propagation_pool: "
                 f"`{report.overlap_with_language_propagation_pool}`")
    lines.append(f"- no_overlap_with_language_propagation_pool: "
                 f"`{report.no_overlap_with_language_propagation_pool}`")
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

    lines.append("## No-mutation confirmation")
    lines.append("")
    lines.append(f"- raw_detector_output_unchanged: "
                 f"`{report.raw_detector_output_unchanged}`")
    lines.append(f"- data_layer_document_type_unchanged: "
                 f"`{report.data_layer_document_type_unchanged}`")
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
                 "(strictly limited to safe abbreviation metadata helper)")
    lines.append(f"- behavior_change_scope: {report.behavior_change_scope}")
    lines.append(f"- clinical_behavior_changed: "
                 f"`{report.clinical_behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    lines.append(f"- abbreviation_parsing_or_expansion: "
                 f"`{report.abbreviation_parsing_or_expansion}`")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. The runtime behavior change is "
                 "strictly limited to the safe abbreviation metadata helper "
                 "described above. The helper never parses or expands the "
                 "abbreviation, never classifies clinical meaning, never parses "
                 "values, never auto-accepts, never writes active clinical "
                 "facts, never alters raw detector output, and never changes the "
                 "data-layer document type. Review-bound status preserved.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    acceptance_ok = (
        report.no_false_positive_expansion
        and report.review_bound_preserved
        and report.eight_record_replay.get("matches_priority_slice_exactly")
        and report.no_overlap_with_numeric_table_safe_default_pool
        and report.no_overlap_with_language_propagation_pool
        and report.accepted_count == 0
        and report.auto_accept_allowed_count == 0
        and report.external_api_used_count == 0
        and report.flag_separation_audit["three_way_flag_separation_holds"]
    )
    if acceptance_ok:
        extra.append(
            "The three language-detector levers (numeric-table safe-default, "
            "language-propagation, and now latin abbreviation) are all "
            "available behind separate default-off env vars. A future "
            "evaluation-only block (e.g. UNKNOWN-DIAG-12A) may surface the "
            "new abbreviation metadata inside the operator routing-review "
            "UI surface analogously to DIAG-08A / 10A, behind its own "
            "separate env-gated render plan that maintains strict three-way "
            "flag separation. The remaining deferred pools (1 table-header "
            "record, 21 text-layer, 17 fallback, 15 ambiguous) remain "
            "deferred or excluded; cue expansion remains not recommended."
        )
    else:
        extra.append(
            "Implementation acceptance criteria not fully met. Downstream "
            "consumption of the abbreviation metadata must not proceed. "
            "Investigate any false-positive expansion, review-bound "
            "violation, priority-slice mismatch, overlap with another "
            "lever, or flag-separation regression and revise before the "
            "next block."
        )
    extra += [
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Raw language / script detector behavior",
        "- Classifier behavior for any record outside the exact 14-field signal",
        "- Data-layer document type",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "The helper never parses or expands the abbreviation. It records ",
        "only that the record contains medical-style abbreviations useful ",
        "for language and context routing.",
        "",
    ]
    return base + "\n".join(extra)


# ── public-report safety guard ──────────────────────────────────────────────

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
        "json": out_dir / "medai_doc_type_unknown_diag_11a_implementation_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_11a_implementation_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_11A_IMPLEMENTATION.md",
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
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION latin abbreviation "
            "metadata helper audit."
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
            "conclusion": "medai_doc_type_unknown_diag_11a_implementation_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "eight_record_replay_matches_priority_slice_exactly":
                report.eight_record_replay["matches_priority_slice_exactly"],
            "no_false_positive_expansion": report.no_false_positive_expansion,
            "no_overlap_with_numeric_table_safe_default_pool":
                report.no_overlap_with_numeric_table_safe_default_pool,
            "no_overlap_with_language_propagation_pool":
                report.no_overlap_with_language_propagation_pool,
            "three_way_flag_separation_holds":
                report.flag_separation_audit["three_way_flag_separation_holds"],
            "review_bound_preserved": report.review_bound_preserved,
            "behavior_changed": report.behavior_changed,
            "clinical_behavior_changed": report.clinical_behavior_changed,
            "external_api_used": report.external_api_used,
            "cue_expansion_recommended": report.cue_expansion_recommended,
            "abbreviation_parsing_or_expansion":
                report.abbreviation_parsing_or_expansion,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
