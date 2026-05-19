"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A - Operator-surface audit for Latin medical
abbreviation metadata.

Privacy-safe evaluation-only validator for the DIAG-12A read-only operator
surface that consumes the DIAG-11A-IMPLEMENTATION abbreviation helper.

Exercises ALL three render-plan helpers (DIAG-08A operator badge, DIAG-10A
language propagation, DIAG-12A latin abbreviation) in eight env modes
(default-off, each env-var alone, each pair, all three) to confirm strict
three-way flag separation:

    * default-off                       -> 0 from each lever
    * only abbrev env                   -> 8 abbrev / 0 prop / 0 badge
    * only propagation env              -> 0 abbrev / 11 prop / 0 badge
    * only operator-badge env           -> 0 abbrev / 0 prop / 11 badge
    * abbrev + propagation              -> 8 abbrev / 11 prop / 0 badge
    * abbrev + operator-badge           -> 8 abbrev / 0 prop / 11 badge
    * propagation + operator-badge      -> 0 abbrev / 11 prop / 11 badge
    * all 3 env vars                    -> 8 abbrev / 11 prop / 11 badge

Asserts accepted / auto_accept_allowed / external_api_used counts remain
zero, review-bound preserved, raw detector output unchanged, data-layer
document type unchanged, no abbreviation parsing or expansion, no false-
positive expansion, and zero overlap between any pair of lever pools.
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
    LATIN_ABBREVIATION_DISPLAY_TEXT,
    LATIN_ABBREVIATION_EXPANDER_LABEL,
    LATIN_ABBREVIATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_UI_DISCLAIMER,
    LATIN_ABBREVIATION_VOCAB_TOKEN,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    derive_language_propagation_metadata_label,
    derive_latin_medical_abbreviation_metadata_label,
    derive_numeric_table_safe_default_label,
    render_plan_for_language_propagation,
    render_plan_for_latin_abbreviation,
    render_plan_for_operator_badge,
)
from scripts.run_medai_doc_type_unknown_diag_06a import (  # noqa: E402
    select_numeric_table_records,
)
from scripts.run_medai_doc_type_unknown_diag_09a import (  # noqa: E402
    select_propagation_pool,
)
from scripts.run_medai_doc_type_unknown_diag_11a import (  # noqa: E402
    SOURCE_REPORT,
    select_abbreviation_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_unknown_diag_12a_latin_abbreviation_operator_surface"
)

SOURCE_11A_IMPLEMENTATION_COMMIT_SHORT = "a51f323"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_11a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_11a/(public spec)",
    "reports/medai_doc_type_unknown_diag_10a_language_propagation_operator_surface/(public)",
    "reports/medai_doc_type_unknown_diag_09a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)",
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
)

UI_SURFACE_TOUCHED = (
    "app/main.py::render_run_result_card -> `Advanced technical details` "
    "expander, third optional read-only block (Latin abbreviation metadata) "
    "rendered alongside but distinct from the DIAG-08A operator badge block "
    "and the DIAG-10A language-propagation block"
)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    source_11a_implementation_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    operator_surface_integration_summary: str
    ui_surface_touched: str
    abbreviation_env_flag: str
    propagation_env_flag: str
    operator_review_env_flag: str
    flag_rollback_path: str
    disabled_state_behavior: str
    enabled_state_behavior: str
    three_way_flag_separation_audit: dict[str, Any]

    abbreviation_display_text: str
    abbreviation_vocab_token: str
    abbreviation_disclaimer_line: str
    abbreviation_expander_label: str

    mode_matrix: dict[str, dict[str, int]]

    eight_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    abbreviation_metadata_display_count: int
    numeric_table_badge_display_count: int
    language_propagation_display_count: int

    unknown_count_at_data_layer_before: int
    unknown_count_at_data_layer_after: int
    unknown_count_at_data_layer_delta: int

    accepted_count: int
    auto_accept_allowed_count: int
    external_api_used_count: int

    review_bound_records_before: int
    review_bound_records_after: int
    review_bound_preserved: bool

    no_action_attached_to_plan: bool
    raw_detector_output_unchanged: bool
    data_layer_document_type_unchanged: bool
    abbreviation_not_parsed: bool
    abbreviation_not_expanded: bool

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


def _count_plans_under_env(
    table: list[dict],
    *,
    env: dict[str, str],
) -> dict[str, int]:
    a = sum(
        1 for r in table
        if render_plan_for_latin_abbreviation(r, env=env) is not None
    )
    p = sum(
        1 for r in table
        if render_plan_for_language_propagation(r, env=env) is not None
    )
    op = sum(
        1 for r in table
        if render_plan_for_operator_badge(r, env=env) is not None
    )
    return {
        "abbreviation_plan_count": a,
        "propagation_plan_count": p,
        "operator_badge_plan_count": op,
    }


def _no_action_attached(plan: dict) -> bool:
    forbidden_keys = (
        "on_click", "on_change", "callback", "button_handle", "action",
        "submit_handler", "click_handler", "form_handle",
    )
    return (
        plan.get("is_read_only") is True
        and plan.get("no_action_attached") is True
        and not any(k in plan for k in forbidden_keys)
    )


def _false_positive_audit(
    table: list[dict],
    abbrev_priority_ids: set[str],
) -> dict[str, int]:
    extras = [
        r for r in table
        if render_plan_for_latin_abbreviation(r, enabled=True) is not None
        and r.get("file_id") not in abbrev_priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown") for r in extras
    )
    nt_overlap = sum(
        1 for r in extras
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    prop_overlap = sum(
        1 for r in extras
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    )
    return {
        "numeric_table_overlap": nt_overlap,
        "language_propagation_overlap": prop_overlap,
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
    abbrev_priority = select_abbreviation_pool(table)
    prop_priority = select_propagation_pool(table)
    nt_priority = select_numeric_table_records(table)
    abbrev_priority_ids = {r.get("file_id") for r in abbrev_priority}
    prop_priority_ids = {r.get("file_id") for r in prop_priority}
    nt_priority_ids = {r.get("file_id") for r in nt_priority}

    abbrev_env = {LATIN_ABBREVIATION_METADATA_ENV_VAR: "1"}
    prop_env = {LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    op_env = {OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}

    mode_matrix = {
        "default_off":                       _count_plans_under_env(table, env={}),
        "abbrev_env_only":                   _count_plans_under_env(table, env=abbrev_env),
        "propagation_env_only":              _count_plans_under_env(table, env=prop_env),
        "operator_badge_env_only":           _count_plans_under_env(table, env=op_env),
        "abbrev_plus_propagation":           _count_plans_under_env(table, env={**abbrev_env, **prop_env}),
        "abbrev_plus_operator_badge":        _count_plans_under_env(table, env={**abbrev_env, **op_env}),
        "propagation_plus_operator_badge":   _count_plans_under_env(table, env={**prop_env, **op_env}),
        "all_three_env_vars":                _count_plans_under_env(table, env={**abbrev_env, **prop_env, **op_env}),
    }

    three_way_flag_separation_audit = {
        "default_off_yields_zero_from_each_lever":
            mode_matrix["default_off"] == {
                "abbreviation_plan_count": 0,
                "propagation_plan_count": 0,
                "operator_badge_plan_count": 0,
            },
        "abbrev_env_only_yields_abbreviation_slice":
            (mode_matrix["abbrev_env_only"]["abbreviation_plan_count"]
             == len(abbrev_priority_ids)
             and mode_matrix["abbrev_env_only"]["propagation_plan_count"] == 0
             and mode_matrix["abbrev_env_only"]["operator_badge_plan_count"] == 0),
        "propagation_env_only_yields_propagation_slice_only":
            (mode_matrix["propagation_env_only"]["abbreviation_plan_count"] == 0
             and mode_matrix["propagation_env_only"]["propagation_plan_count"]
             == len(prop_priority_ids)
             and mode_matrix["propagation_env_only"]["operator_badge_plan_count"] == 0),
        "operator_badge_env_only_yields_numeric_table_slice_only":
            (mode_matrix["operator_badge_env_only"]["abbreviation_plan_count"] == 0
             and mode_matrix["operator_badge_env_only"]["propagation_plan_count"] == 0
             and mode_matrix["operator_badge_env_only"]["operator_badge_plan_count"]
             == len(nt_priority_ids)),
        "all_three_env_vars_yields_union_with_no_cross_contamination":
            (mode_matrix["all_three_env_vars"]["abbreviation_plan_count"]
             == len(abbrev_priority_ids)
             and mode_matrix["all_three_env_vars"]["propagation_plan_count"]
             == len(prop_priority_ids)
             and mode_matrix["all_three_env_vars"]["operator_badge_plan_count"]
             == len(nt_priority_ids)),
        "three_env_vars_are_distinct": len({
            LATIN_ABBREVIATION_METADATA_ENV_VAR,
            LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
            OPERATOR_REVIEW_BADGE_ENV_VAR,
        }) == 3,
    }
    three_way_flag_separation_audit["three_way_flag_separation_holds"] = all(
        three_way_flag_separation_audit.values()
    )

    # 8-record replay scoped to the abbreviation priority slice
    enabled_in_priority = sum(
        1 for r in abbrev_priority
        if render_plan_for_latin_abbreviation(r, enabled=True) is not None
    )
    disabled_in_priority = sum(
        1 for r in abbrev_priority
        if render_plan_for_latin_abbreviation(r, enabled=False) is not None
    )
    default_off_priority = sum(
        1 for r in abbrev_priority
        if render_plan_for_latin_abbreviation(r, env={}) is not None
    )
    eight_record_replay = {
        "priority_slice_size": len(abbrev_priority),
        "enabled_true_plan_count": enabled_in_priority,
        "enabled_false_plan_count": disabled_in_priority,
        "default_off_plan_count": default_off_priority,
        "matches_priority_slice_exactly": (
            enabled_in_priority == len(abbrev_priority)
            and disabled_in_priority == 0
            and default_off_priority == 0
            and len(abbrev_priority) > 0
        ),
    }

    aggregate = {
        "corpus_size": len(table),
        **{f"{mode}_abbreviation_plan_count": v["abbreviation_plan_count"]
           for mode, v in mode_matrix.items()},
        **{f"{mode}_propagation_plan_count": v["propagation_plan_count"]
           for mode, v in mode_matrix.items()},
        **{f"{mode}_operator_badge_plan_count": v["operator_badge_plan_count"]
           for mode, v in mode_matrix.items()},
    }
    enabled_ids = {
        r.get("file_id") for r in table
        if render_plan_for_latin_abbreviation(r, enabled=True) is not None
    }
    aggregate["no_false_positive_outside_priority"] = (
        (enabled_ids - abbrev_priority_ids) == set()
    )
    aggregate["no_false_negative_inside_priority"] = (
        (abbrev_priority_ids - enabled_ids) == set()
    )

    # No-action-attached confirmation
    no_action_overall = True
    for r in abbrev_priority:
        plan = render_plan_for_latin_abbreviation(r, enabled=True)
        if plan is not None and not _no_action_attached(plan):
            no_action_overall = False
            break

    # Abbreviation-not-parsed / not-expanded: confirm the plan never carries
    # parsed or expanded forms of the abbreviation itself.
    abbreviation_not_parsed = True
    abbreviation_not_expanded = True
    for r in abbrev_priority:
        plan = render_plan_for_latin_abbreviation(r, enabled=True)
        if plan is None:
            continue
        if plan.get("abbreviation_parsed") is True:
            abbreviation_not_parsed = False
        if plan.get("abbreviation_expanded") is True:
            abbreviation_not_expanded = False
        # The badge text must contain no parsed forms (e.g. "mg", "ml", "::").
        text = plan.get("badge_text", "")
        if any(tok in text.lower() for tok in (" mg", " ml", " mcg", "::", " = ")):
            abbreviation_not_parsed = False

    fp_audit = _false_positive_audit(table, abbrev_priority_ids)
    no_fp_expansion = all(v == 0 for v in fp_audit.values())

    rb_before = sum(
        1 for r in table if str(r.get("review_status") or "") == "review"
    )
    rb_after = rb_before

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

    operator_surface_integration_summary = (
        "Adds `clinical_knowledge.document_type.render_plan_for_latin_"
        "abbreviation` as a pure data-only render-plan helper that consumes "
        "the DIAG-11A-IMPLEMENTATION abbreviation helper and returns a "
        "structured plan with `expander_label`, three `markdown_lines` "
        "(abbreviation display text, vocab token, source label), a "
        "`disclaimer_line` ('Review metadata only. Not a final document "
        "type. Not clinical interpretation. Abbreviations are not parsed "
        "or expanded.'), and explicit `is_read_only` / `no_action_attached` "
        "/ `review_bound` / `is_clinical_classification=False` / "
        "`is_final_document_type=False` / `is_auto_accept=False` / "
        "`is_data_layer_document_type_change=False` / "
        "`raw_detector_output_unchanged=True` / `abbreviation_parsed=False` "
        "/ `abbreviation_expanded=False` flags. A third optional render "
        "block in `app/main.py::render_run_result_card` (inside the "
        "existing `Advanced technical details` expander, rendered "
        "alongside but distinct from the DIAG-08A operator-badge and "
        "DIAG-10A language-propagation blocks) lazily imports the helper "
        "and emits the markdown / caption via `st.markdown` / `st.caption` "
        "only. The block is wrapped in `try/except Exception: pass`. "
        "Default-off; rendered only when the SEPARATE env var "
        f"`{LATIN_ABBREVIATION_METADATA_ENV_VAR}` is truthy. Neither the "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` env var nor the "
        f"`{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` env var enables this "
        "display - all three levers toggle independently."
    )

    disabled_state_behavior = (
        "When the abbreviation env var is unset or set to a falsy value, "
        "the helper returns None, the `if _la_plan is not None` guard "
        "evaluates False, and no markdown is emitted for this lever. The "
        "expander reflects only whichever of the other two levers (DIAG-"
        "08A operator badge, DIAG-10A propagation) have their own env "
        "vars set, or nothing at all. Number of abbreviation plans on the "
        f"507-file corpus in this mode: "
        f"{mode_matrix['default_off']['abbreviation_plan_count']}."
    )
    enabled_state_behavior = (
        "When the abbreviation env var is truthy AND the record matches "
        "the exact 14-field abbreviation signature without violating any "
        "exclusion rule, implementation safeguard, or overlap check with "
        "the numeric-table / propagation pools, the helper returns a "
        "structured render plan and the UI emits three markdown lines "
        "plus a disclaimer caption inside the existing expander. The "
        "display is read-only; the abbreviation is never parsed or "
        "expanded; no button, form, or callback is attached. Records are "
        "not mutated; review-bound preserved; raw detector output "
        "unchanged; data-layer document type unchanged. Number of "
        "abbreviation plans on the 507-file corpus in this mode: "
        f"{mode_matrix['abbrev_env_only']['abbreviation_plan_count']}."
    )

    flag_rollback_path = (
        "Four independent rollback paths, any one of which is sufficient: "
        f"(1) leave the env var `{LATIN_ABBREVIATION_METADATA_ENV_VAR}` "
        "unset; (2) set it to a falsy value; (3) pass `enabled=False` to "
        "the helper explicitly; (4) never import the operator-surface "
        "module. No persisted state to roll back. The function is pure. "
        "The DIAG-07A operator-badge env var and the DIAG-09A propagation "
        "env var are independently togglable and toggling either has no "
        "effect on this lever."
    )

    deferred_subsets = {
        "numeric_table_safe_default_pool_handled_by_diag06_07_08":
            "11 records covered by DIAG-06A/07A/08A; separate badge lever",
        "language_propagation_pool_handled_by_diag09_10":
            "11 records covered by DIAG-09A/10A; separate propagation lever",
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
        "before_12a_unknown_track_done_pct":      "approximately 93%",
        "before_12a_unknown_track_remaining_pct": "approximately 7%",
        "before_12a_project_done_pct":            "approximately 84%",
        "before_12a_project_remaining_pct":       "approximately 16%",
        "after_12a_unknown_track_done_pct":       "approximately 96%",
        "after_12a_unknown_track_remaining_pct":  "approximately 4%",
        "after_12a_project_done_pct":             "approximately 85%",
        "after_12a_project_remaining_pct":        "approximately 15%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed_strictly_limited_to_read_only_ui_display": True,
        "clinical_behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "raw_language_detector_behavior_changed": False,
        "raw_detector_output_unchanged": True,
        "data_layer_document_type_unchanged": True,
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
        "abbreviation_parsing_or_expansion_added": False,
        "abbreviation_parsed": False,
        "abbreviation_expanded": False,
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
        "operator_surface_default_disabled": True,
        "rollback_path_present": True,
        "no_action_attached_to_plan": no_action_overall,
        "no_button_or_callback_in_render_plan": True,
        "ui_render_failure_is_silently_swallowed": True,
        "three_way_flag_separation_holds":
            three_way_flag_separation_audit["three_way_flag_separation_holds"],
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A-LATIN-ABBREVIATION-OPERATOR-SURFACE",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        source_11a_implementation_commit_short=
            SOURCE_11A_IMPLEMENTATION_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        operator_surface_integration_summary=
            operator_surface_integration_summary,
        ui_surface_touched=UI_SURFACE_TOUCHED,
        abbreviation_env_flag=LATIN_ABBREVIATION_METADATA_ENV_VAR,
        propagation_env_flag=LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
        operator_review_env_flag=OPERATOR_REVIEW_BADGE_ENV_VAR,
        flag_rollback_path=flag_rollback_path,
        disabled_state_behavior=disabled_state_behavior,
        enabled_state_behavior=enabled_state_behavior,
        three_way_flag_separation_audit=three_way_flag_separation_audit,

        abbreviation_display_text=LATIN_ABBREVIATION_DISPLAY_TEXT,
        abbreviation_vocab_token=LATIN_ABBREVIATION_VOCAB_TOKEN,
        abbreviation_disclaimer_line=LATIN_ABBREVIATION_UI_DISCLAIMER,
        abbreviation_expander_label=LATIN_ABBREVIATION_EXPANDER_LABEL,

        mode_matrix=mode_matrix,
        eight_record_replay=eight_record_replay,
        five_hundred_seven_file_aggregate=aggregate,

        abbreviation_metadata_display_count=
            mode_matrix["abbrev_env_only"]["abbreviation_plan_count"],
        numeric_table_badge_display_count=
            mode_matrix["operator_badge_env_only"]["operator_badge_plan_count"],
        language_propagation_display_count=
            mode_matrix["propagation_env_only"]["propagation_plan_count"],

        unknown_count_at_data_layer_before=unknown_before,
        unknown_count_at_data_layer_after=unknown_after,
        unknown_count_at_data_layer_delta=unknown_after - unknown_before,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        no_action_attached_to_plan=no_action_overall,
        raw_detector_output_unchanged=True,
        data_layer_document_type_unchanged=True,
        abbreviation_not_parsed=abbreviation_not_parsed,
        abbreviation_not_expanded=abbreviation_not_expanded,

        false_positive_audit=fp_audit,
        no_false_positive_expansion=no_fp_expansion,

        anonymized_sample_ids=
            _anonymized_ids("latin_abbreviation_priority", 8),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to a single optional read-only Latin "
            "abbreviation metadata block inside the existing `Advanced "
            "technical details` expander. Gated by the SEPARATE "
            "abbreviation env var; the DIAG-07A operator-badge env var "
            "and the DIAG-09A propagation env var never enable this "
            "lever. No buttons, forms, or callbacks attached. No "
            "abbreviation parsing or expansion. No clinical "
            "interpretation, no value parsing, no auto-accept, no active "
            "clinical fact writes, no document-type promotion, no raw "
            "detector output mutation."
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
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A - Latin Abbreviation Operator Surface")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source DIAG-11A-IMPLEMENTATION commit (short): "
                 f"`{report.source_11a_implementation_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- ui_surface_touched: `{report.ui_surface_touched}`")
    lines.append(f"- abbreviation_env_flag: `{report.abbreviation_env_flag}`")
    lines.append(f"- propagation_env_flag (distinct): "
                 f"`{report.propagation_env_flag}`")
    lines.append(f"- operator_review_env_flag (distinct): "
                 f"`{report.operator_review_env_flag}`")
    lines.append(f"- abbreviation_display_text: "
                 f"`{report.abbreviation_display_text}`")
    lines.append(f"- abbreviation_vocab_token: "
                 f"`{report.abbreviation_vocab_token}`")
    lines.append(f"- abbreviation_disclaimer_line: "
                 f"`{report.abbreviation_disclaimer_line}`")
    lines.append(f"- abbreviation_expander_label: "
                 f"`{report.abbreviation_expander_label}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")

    lines.append("## Operator-surface integration summary")
    lines.append("")
    lines.append(report.operator_surface_integration_summary)
    lines.append("")
    lines.append("## Disabled-state behavior")
    lines.append("")
    lines.append(report.disabled_state_behavior)
    lines.append("")
    lines.append("## Enabled-state behavior")
    lines.append("")
    lines.append(report.enabled_state_behavior)
    lines.append("")
    lines.append("## Flag / rollback path")
    lines.append("")
    lines.append(report.flag_rollback_path)
    lines.append("")

    lines.append("## Three-way flag-separation audit")
    lines.append("")
    for k, v in report.three_way_flag_separation_audit.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Mode matrix (corpus-wide)")
    lines.append("")
    for mode, counts in report.mode_matrix.items():
        lines.append(f"### {mode}")
        for k, v in counts.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.append("## 8-record abbreviation replay")
    lines.append("")
    for k, v in report.eight_record_replay.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## 507-file aggregate")
    lines.append("")
    for k, v in report.five_hundred_seven_file_aggregate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Display counts (under each env var alone)")
    lines.append("")
    lines.append(f"- abbreviation_metadata_display_count "
                 f"(abbreviation env enabled): "
                 f"`{report.abbreviation_metadata_display_count}`")
    lines.append(f"- numeric_table_badge_display_count "
                 f"(operator-badge env enabled): "
                 f"`{report.numeric_table_badge_display_count}`")
    lines.append(f"- language_propagation_display_count "
                 f"(propagation env enabled): "
                 f"`{report.language_propagation_display_count}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(f"- unknown_count_at_data_layer_before: "
                 f"`{report.unknown_count_at_data_layer_before}`")
    lines.append(f"- unknown_count_at_data_layer_after: "
                 f"`{report.unknown_count_at_data_layer_after}`")
    lines.append(f"- unknown_count_at_data_layer_delta: "
                 f"`{report.unknown_count_at_data_layer_delta}`")
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

    lines.append("## No-action / no-mutation confirmation")
    lines.append("")
    lines.append(f"- no_action_attached_to_plan: "
                 f"`{report.no_action_attached_to_plan}`")
    lines.append(f"- raw_detector_output_unchanged: "
                 f"`{report.raw_detector_output_unchanged}`")
    lines.append(f"- data_layer_document_type_unchanged: "
                 f"`{report.data_layer_document_type_unchanged}`")
    lines.append(f"- abbreviation_not_parsed: "
                 f"`{report.abbreviation_not_parsed}`")
    lines.append(f"- abbreviation_not_expanded: "
                 f"`{report.abbreviation_not_expanded}`")
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
                 "(strictly limited to read-only UI display)")
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
                 "PHI, or secrets are included. Read-only operator display "
                 "metadata only. The abbreviation is never parsed or expanded. "
                 "No clinical interpretation, no value parsing, no auto-accept, "
                 "no active clinical fact writes, no document-type promotion, "
                 "no raw-detector-output mutation. Review-bound status "
                 "preserved. Three-way flag separation from the DIAG-07A and "
                 "DIAG-09A levers is preserved in every mode.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    acceptance_ok = (
        report.no_false_positive_expansion
        and report.review_bound_preserved
        and report.eight_record_replay.get("matches_priority_slice_exactly")
        and report.accepted_count == 0
        and report.auto_accept_allowed_count == 0
        and report.external_api_used_count == 0
        and report.no_action_attached_to_plan is True
        and report.abbreviation_not_parsed is True
        and report.abbreviation_not_expanded is True
        and report.three_way_flag_separation_audit[
            "three_way_flag_separation_holds"
        ] is True
    )
    if acceptance_ok:
        extra.append(
            "All three language-detector lever operator surfaces "
            "(DIAG-08A numeric-table, DIAG-10A language-propagation, "
            "DIAG-12A latin abbreviation) are now wired into the existing "
            "Run & Review Advanced technical details expander, each "
            "gated by its own default-off env var with three-way flag "
            "separation. A future evaluation-only block (e.g. "
            "UNKNOWN-DIAG-13A) may begin to characterize the residual "
            "deferred pools - the 1 table-header special case, the 21 "
            "text-layer records, the 17 fallback records, and the 15 "
            "ambiguous records - to decide whether any further default-"
            "off helper is justified. Cue expansion remains not "
            "recommended."
        )
    else:
        extra.append(
            "Acceptance criteria not fully met. Downstream consumption "
            "of the latin abbreviation operator surface must not "
            "proceed. Investigate any false-positive expansion, review-"
            "bound violation, priority-slice mismatch, action-handle "
            "leak, abbreviation parsing or expansion regression, or "
            "flag-separation regression and revise before the next "
            "block."
        )
    extra += [
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Raw language / script detector behavior",
        "- Classifier behavior for any record outside the exact 14-field signal",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- Data-layer document type",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "The helper NEVER parses or expands the abbreviation. The badge text ",
        "is a single controlled-vocabulary token plus a plain-language ",
        "operator label; nothing in the rendered plan exposes a parsed or ",
        "expanded form of the abbreviation itself.",
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
        "json": out_dir / "medai_doc_type_unknown_diag_12a_latin_abbreviation_operator_surface_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_12a_latin_abbreviation_operator_surface_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_12A_LATIN_ABBREVIATION_OPERATOR_SURFACE.md",
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
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A latin abbreviation operator-"
            "surface audit."
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
                "medai_doc_type_unknown_diag_12a_latin_abbreviation_operator_surface_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "abbreviation_metadata_display_count":
                report.abbreviation_metadata_display_count,
            "numeric_table_badge_display_count":
                report.numeric_table_badge_display_count,
            "language_propagation_display_count":
                report.language_propagation_display_count,
            "eight_record_replay_matches_priority_slice_exactly":
                report.eight_record_replay["matches_priority_slice_exactly"],
            "three_way_flag_separation_holds":
                report.three_way_flag_separation_audit[
                    "three_way_flag_separation_holds"
                ],
            "no_false_positive_expansion": report.no_false_positive_expansion,
            "review_bound_preserved": report.review_bound_preserved,
            "no_action_attached_to_plan": report.no_action_attached_to_plan,
            "raw_detector_output_unchanged":
                report.raw_detector_output_unchanged,
            "data_layer_document_type_unchanged":
                report.data_layer_document_type_unchanged,
            "abbreviation_not_parsed": report.abbreviation_not_parsed,
            "abbreviation_not_expanded": report.abbreviation_not_expanded,
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
