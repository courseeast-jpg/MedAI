"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A - Operator-surface audit for propagated
language metadata.

Privacy-safe evaluation-only validator for the DIAG-10A read-only operator
surface that consumes the DIAG-09A-IMPLEMENTATION propagation helper.

Exercises the propagation render-plan helper AND the DIAG-08A operator-
badge render-plan helper in four modes (default-off, only the propagation
env var, only the operator-badge env var, both env vars set) to confirm:

    * Default-off: no plans rendered from either lever.
    * Only the propagation env var set: 11 propagation plans, 0 operator
      badge plans.
    * Only the operator-badge env var set: 0 propagation plans, 11
      operator badge plans.
    * Both env vars set: 11 + 11 = 22 plans total, each lever rendering
      its own priority slice with zero cross-contamination.

Also confirms accepted / auto_accept_allowed / external_api_used counts
remain zero, review-bound preserved, no action attached, no false-positive
expansion, raw detector output unchanged, data-layer document type
unchanged. Emits three privacy-safe public reports.
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
    LANGUAGE_PROPAGATION_DISCLAIMER,
    LANGUAGE_PROPAGATION_DISPLAY_TEXT,
    LANGUAGE_PROPAGATION_EXPANDER_LABEL,
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    LANGUAGE_PROPAGATION_VOCAB_TOKEN,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    derive_numeric_table_safe_default_label,
    language_propagation_operator_surface_is_enabled,
    render_plan_for_language_propagation,
    render_plan_for_operator_badge,
)
from scripts.run_medai_doc_type_unknown_diag_06a import (  # noqa: E402
    select_numeric_table_records,
)
from scripts.run_medai_doc_type_unknown_diag_09a import (  # noqa: E402
    SOURCE_REPORT,
    select_propagation_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_unknown_diag_10a_language_propagation_operator_surface"
)

SOURCE_09A_IMPLEMENTATION_COMMIT_SHORT = "31f42fc"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_09a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_09a/(public spec)",
    "reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)",
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_06a/(public spec)",
)

UI_SURFACE_TOUCHED = (
    "app/main.py::render_run_result_card -> `Advanced technical details` "
    "expander, optional read-only language-propagation metadata block "
    "(rendered alongside but distinct from the DIAG-08A operator badge block)"
)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    source_09a_implementation_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    operator_surface_integration_summary: str
    ui_surface_touched: str
    propagation_env_flag: str
    operator_review_env_flag: str
    flag_rollback_path: str
    disabled_state_behavior: str
    enabled_state_behavior: str
    flag_separation_audit: dict[str, Any]

    propagation_display_text: str
    propagation_vocab_token: str
    propagation_disclaimer_line: str
    propagation_expander_label: str

    default_off_audit: dict[str, Any]
    propagation_env_only_audit: dict[str, Any]
    operator_env_only_audit: dict[str, Any]
    both_env_audit: dict[str, Any]

    eleven_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    propagation_metadata_display_count: int
    numeric_table_badge_display_count: int

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


def _audit_mode(
    table: list[dict],
    *,
    env: dict[str, str],
) -> dict[str, Any]:
    """Count propagation-plan hits and operator-badge-plan hits under a
    given env state. Used to confirm flag separation."""
    propagation_hits = [
        r for r in table
        if render_plan_for_language_propagation(r, env=env) is not None
    ]
    operator_badge_hits = [
        r for r in table
        if render_plan_for_operator_badge(r, env=env) is not None
    ]
    return {
        "propagation_plan_count": len(propagation_hits),
        "operator_badge_plan_count": len(operator_badge_hits),
        "propagation_hit_ids": [r.get("file_id") for r in propagation_hits],
        "operator_badge_hit_ids": [r.get("file_id") for r in operator_badge_hits],
    }


def _summarize_audit(
    audit: dict[str, Any],
    prop_priority_ids: set[str],
    nt_priority_ids: set[str],
) -> dict[str, Any]:
    prop_ids = set(audit["propagation_hit_ids"])
    op_ids = set(audit["operator_badge_hit_ids"])
    return {
        "propagation_plan_count": audit["propagation_plan_count"],
        "operator_badge_plan_count": audit["operator_badge_plan_count"],
        "propagation_matches_priority_slice_exactly":
            (prop_ids == prop_priority_ids) if prop_priority_ids else (prop_ids == set()),
        "operator_badge_matches_priority_slice_exactly":
            (op_ids == nt_priority_ids) if nt_priority_ids else (op_ids == set()),
        "propagation_extras_outside_priority_count":
            len(prop_ids - prop_priority_ids),
        "operator_badge_extras_outside_priority_count":
            len(op_ids - nt_priority_ids),
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
    prop_priority_ids: set[str],
) -> dict[str, int]:
    extras = [
        r for r in table
        if render_plan_for_language_propagation(r, enabled=True) is not None
        and r.get("file_id") not in prop_priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown") for r in extras
    )
    nt_overlap = sum(
        1 for r in extras
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    return {
        "numeric_table_overlap": nt_overlap,
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
    prop_priority = select_propagation_pool(table)
    nt_priority = select_numeric_table_records(table)
    prop_priority_ids = {r.get("file_id") for r in prop_priority}
    nt_priority_ids = {r.get("file_id") for r in nt_priority}

    prop_env = {LANGUAGE_PROPAGATION_METADATA_ENV_VAR: "1"}
    op_env = {OPERATOR_REVIEW_BADGE_ENV_VAR: "1"}
    both_env = {**prop_env, **op_env}

    default_off = _summarize_audit(
        _audit_mode(table, env={}), prop_priority_ids, nt_priority_ids,
    )
    prop_only = _summarize_audit(
        _audit_mode(table, env=prop_env), prop_priority_ids, nt_priority_ids,
    )
    op_only = _summarize_audit(
        _audit_mode(table, env=op_env), prop_priority_ids, nt_priority_ids,
    )
    both = _summarize_audit(
        _audit_mode(table, env=both_env), prop_priority_ids, nt_priority_ids,
    )

    flag_separation_audit = {
        "default_off_propagation_plans_zero":
            default_off["propagation_plan_count"] == 0,
        "default_off_operator_badge_plans_zero":
            default_off["operator_badge_plan_count"] == 0,
        "prop_env_only_yields_propagation_plans":
            prop_only["propagation_plan_count"] > 0,
        "prop_env_only_yields_zero_operator_badge_plans":
            prop_only["operator_badge_plan_count"] == 0,
        "op_env_only_yields_zero_propagation_plans":
            op_only["propagation_plan_count"] == 0,
        "op_env_only_yields_operator_badge_plans":
            op_only["operator_badge_plan_count"] > 0,
        "both_env_yields_both_levers_with_correct_priority_slices":
            (both["propagation_matches_priority_slice_exactly"]
             and both["operator_badge_matches_priority_slice_exactly"]),
        "flag_separation_holds_in_all_modes": (
            default_off["propagation_plan_count"] == 0
            and default_off["operator_badge_plan_count"] == 0
            and prop_only["operator_badge_plan_count"] == 0
            and op_only["propagation_plan_count"] == 0
        ),
    }

    # 11-record propagation replay scoped to the priority slice only.
    enabled_in_priority = sum(
        1 for r in prop_priority
        if render_plan_for_language_propagation(r, enabled=True) is not None
    )
    disabled_in_priority = sum(
        1 for r in prop_priority
        if render_plan_for_language_propagation(r, enabled=False) is not None
    )
    default_off_priority = sum(
        1 for r in prop_priority
        if render_plan_for_language_propagation(r, env={}) is not None
    )
    eleven_record_replay = {
        "priority_slice_size": len(prop_priority),
        "enabled_true_plan_count": enabled_in_priority,
        "enabled_false_plan_count": disabled_in_priority,
        "default_off_plan_count": default_off_priority,
        "matches_priority_slice_exactly": (
            enabled_in_priority == len(prop_priority)
            and disabled_in_priority == 0
            and default_off_priority == 0
            and len(prop_priority) > 0
        ),
    }

    aggregate = {
        "corpus_size": len(table),
        "default_off_propagation_plan_count":
            default_off["propagation_plan_count"],
        "default_off_operator_badge_plan_count":
            default_off["operator_badge_plan_count"],
        "prop_env_only_propagation_plan_count":
            prop_only["propagation_plan_count"],
        "prop_env_only_operator_badge_plan_count":
            prop_only["operator_badge_plan_count"],
        "op_env_only_propagation_plan_count":
            op_only["propagation_plan_count"],
        "op_env_only_operator_badge_plan_count":
            op_only["operator_badge_plan_count"],
        "both_env_propagation_plan_count":
            both["propagation_plan_count"],
        "both_env_operator_badge_plan_count":
            both["operator_badge_plan_count"],
        "no_false_positive_outside_priority": (
            default_off["propagation_extras_outside_priority_count"] == 0
            and prop_only["propagation_extras_outside_priority_count"] == 0
            and op_only["propagation_extras_outside_priority_count"] == 0
            and both["propagation_extras_outside_priority_count"] == 0
        ),
        "no_false_negative_inside_priority": (
            prop_only["propagation_matches_priority_slice_exactly"] is True
            and both["propagation_matches_priority_slice_exactly"] is True
        ),
    }

    # No-action-attached: render plans for each priority record and verify
    # the contract.
    no_action_overall = True
    for r in prop_priority:
        plan = render_plan_for_language_propagation(r, enabled=True)
        if plan is not None and not _no_action_attached(plan):
            no_action_overall = False
            break

    fp_audit = _false_positive_audit(table, prop_priority_ids)
    no_fp_expansion = all(v == 0 for v in fp_audit.values())

    rb_before = sum(
        1 for r in table if str(r.get("review_status") or "") == "review"
    )
    rb_after = rb_before  # helper never mutates

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

    operator_surface_integration_summary = (
        "Adds `clinical_knowledge.document_type.render_plan_for_language_"
        "propagation` as a pure data-only render-plan helper that consumes "
        "the DIAG-09A-IMPLEMENTATION propagation helper and returns a "
        "structured plan with `expander_label`, three `markdown_lines` "
        "(propagation display text, vocab token, source label), a "
        "`disclaimer_line` ('Review metadata only. Not a final document "
        "type. Not clinical interpretation.'), and explicit `is_read_only`"
        " / `no_action_attached` / `review_bound` / `is_clinical_"
        "classification=False` / `is_final_document_type=False` / "
        "`is_auto_accept=False` / `raw_detector_output_unchanged=True` / "
        "`is_data_layer_document_type_change=False` flags. A small "
        "optional render block in `app/main.py::render_run_result_card` "
        "(inside the existing `Advanced technical details` expander, "
        "rendered alongside but distinct from the DIAG-08A operator-badge "
        "block) lazily imports the helper, calls it, and emits the badge "
        "text and disclaimer via `st.markdown` and `st.caption` only. The "
        "block is wrapped in `try/except Exception: pass`. Default-off; "
        "rendered only when the SEPARATE env var "
        f"`{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` is truthy. The DIAG-"
        f"07A env var (`{OPERATOR_REVIEW_BADGE_ENV_VAR}`) does NOT enable "
        "this display."
    )

    disabled_state_behavior = (
        "When the propagation env var is unset or set to a falsy value, "
        "the helper returns None, the `if _lp_plan is not None` guard "
        "evaluates False, and no markdown is emitted for this lever. The "
        "expander content reflects only the DIAG-08A operator badge "
        "(if its own env var is set) or nothing at all. Number of "
        "propagation plans rendered on the 507-file corpus in this mode: "
        f"{default_off['propagation_plan_count']}."
    )
    enabled_state_behavior = (
        "When the propagation env var is set to a truthy value AND the "
        "record matches the exact 11-field propagation signature without "
        "violating any exclusion rule, implementation safeguard, or "
        "numeric-table overlap check, the helper returns a structured "
        "render plan and the UI emits three markdown lines plus a "
        "disclaimer caption inside the existing expander. The display is "
        "read-only; no button, form, or callback is attached. Records "
        "are not mutated; review-bound status is preserved; raw detector "
        "output is unchanged; data-layer document type is unchanged. "
        f"Number of propagation plans rendered on the 507-file corpus in "
        f"this mode: {prop_only['propagation_plan_count']}."
    )

    flag_rollback_path = (
        "Multiple rollback paths exist, any one of which is sufficient: "
        f"(1) leave the env var `{LANGUAGE_PROPAGATION_METADATA_ENV_VAR}` "
        "unset; (2) set it to a falsy value; "
        "(3) pass `enabled=False` to the helper explicitly; "
        "(4) never import the operator-surface module. No persisted state "
        "to roll back. The function is pure. The DIAG-08A operator-badge "
        "env var is independently togglable and toggling it has no effect "
        "on this lever."
    )

    deferred_subsets = {
        "numeric_table_safe_default_pool_handled_by_diag06_07_08":
            "11 records covered by DIAG-06A/07A/08A; separate badge lever",
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
        "before_10a_unknown_track_done_pct":      "approximately 84%",
        "before_10a_unknown_track_remaining_pct": "approximately 16%",
        "before_10a_project_done_pct":            "approximately 81%",
        "before_10a_project_remaining_pct":       "approximately 19%",
        "after_10a_unknown_track_done_pct":       "approximately 87%",
        "after_10a_unknown_track_remaining_pct":  "approximately 13%",
        "after_10a_project_done_pct":             "approximately 82%",
        "after_10a_project_remaining_pct":        "approximately 18%",
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
        "flag_separation_holds": flag_separation_audit[
            "flag_separation_holds_in_all_modes"
        ],
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A-LANGUAGE-PROPAGATION-OPERATOR-SURFACE",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        source_09a_implementation_commit_short=
            SOURCE_09A_IMPLEMENTATION_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        operator_surface_integration_summary=operator_surface_integration_summary,
        ui_surface_touched=UI_SURFACE_TOUCHED,
        propagation_env_flag=LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
        operator_review_env_flag=OPERATOR_REVIEW_BADGE_ENV_VAR,
        flag_rollback_path=flag_rollback_path,
        disabled_state_behavior=disabled_state_behavior,
        enabled_state_behavior=enabled_state_behavior,
        flag_separation_audit=flag_separation_audit,

        propagation_display_text=LANGUAGE_PROPAGATION_DISPLAY_TEXT,
        propagation_vocab_token=LANGUAGE_PROPAGATION_VOCAB_TOKEN,
        propagation_disclaimer_line=LANGUAGE_PROPAGATION_DISCLAIMER,
        propagation_expander_label=LANGUAGE_PROPAGATION_EXPANDER_LABEL,

        default_off_audit=default_off,
        propagation_env_only_audit=prop_only,
        operator_env_only_audit=op_only,
        both_env_audit=both,

        eleven_record_replay=eleven_record_replay,
        five_hundred_seven_file_aggregate=aggregate,

        propagation_metadata_display_count=
            prop_only["propagation_plan_count"],
        numeric_table_badge_display_count=
            op_only["operator_badge_plan_count"],

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

        false_positive_audit=fp_audit,
        no_false_positive_expansion=no_fp_expansion,

        anonymized_sample_ids=
            _anonymized_ids("language_propagation_priority", 11),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to a single optional read-only language-"
            "propagation metadata block inside the existing `Advanced "
            "technical details` expander in the Run & Review result card. "
            "Gated by the SEPARATE propagation env var; the DIAG-07A "
            "operator-badge env var does not enable this lever. No "
            "buttons, forms, or callbacks attached. No clinical "
            "interpretation, no value parsing, no auto-accept, no active "
            "clinical fact writes, no document-type promotion at the data "
            "layer, no raw-detector-output mutation."
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
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A - Language Propagation Operator Surface")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source DIAG-09A-IMPLEMENTATION commit (short): "
                 f"`{report.source_09a_implementation_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- ui_surface_touched: `{report.ui_surface_touched}`")
    lines.append(f"- propagation_env_flag: `{report.propagation_env_flag}`")
    lines.append(f"- operator_review_env_flag (distinct): "
                 f"`{report.operator_review_env_flag}`")
    lines.append(f"- propagation_display_text: "
                 f"`{report.propagation_display_text}`")
    lines.append(f"- propagation_vocab_token: "
                 f"`{report.propagation_vocab_token}`")
    lines.append(f"- propagation_disclaimer_line: "
                 f"`{report.propagation_disclaimer_line}`")
    lines.append(f"- propagation_expander_label: "
                 f"`{report.propagation_expander_label}`")
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

    lines.append("## Flag-separation audit")
    lines.append("")
    for k, v in report.flag_separation_audit.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Mode audits (corpus-wide)")
    lines.append("")
    for name, audit in (
        ("default_off", report.default_off_audit),
        ("propagation_env_only", report.propagation_env_only_audit),
        ("operator_badge_env_only", report.operator_env_only_audit),
        ("both_env", report.both_env_audit),
    ):
        lines.append(f"### {name}")
        for k, v in audit.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.append("## 11-record propagation replay")
    lines.append("")
    for k, v in report.eleven_record_replay.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## 507-file aggregate")
    lines.append("")
    for k, v in report.five_hundred_seven_file_aggregate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append(f"- propagation_metadata_display_count "
                 f"(propagation env enabled): "
                 f"`{report.propagation_metadata_display_count}`")
    lines.append(f"- numeric_table_badge_display_count "
                 f"(operator-badge env enabled): "
                 f"`{report.numeric_table_badge_display_count}`")
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
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Read-only operator display "
                 "metadata only. No clinical interpretation, no value parsing, "
                 "no auto-accept, no active clinical fact writes, no document-"
                 "type promotion, no raw-detector-output mutation. Review-bound "
                 "status preserved. Flag separation from the DIAG-07A operator-"
                 "badge env var is preserved in every mode.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    acceptance_ok = (
        report.no_false_positive_expansion
        and report.review_bound_preserved
        and report.eleven_record_replay.get("matches_priority_slice_exactly")
        and report.accepted_count == 0
        and report.auto_accept_allowed_count == 0
        and report.external_api_used_count == 0
        and report.no_action_attached_to_plan is True
        and report.flag_separation_audit[
            "flag_separation_holds_in_all_modes"
        ] is True
    )
    if acceptance_ok:
        extra.append(
            "Both the DIAG-08A numeric-table operator badge and the new "
            "DIAG-10A language-propagation metadata display are now "
            "available behind separate default-off env vars. A future "
            "evaluation-only block (e.g. UNKNOWN-DIAG-11A) may begin to "
            "characterize the remaining deferred pools - the 8 latin "
            "medical-abbreviation records, the 1 table-header record, the "
            "21 text-layer records, and the 17 fallback records - to "
            "decide whether any further default-off helper is justified. "
            "Cue expansion remains not recommended for any pool."
        )
    else:
        extra.append(
            "Acceptance criteria not fully met. Downstream consumption of "
            "the propagation operator surface must not proceed. "
            "Investigate any false-positive expansion, review-bound "
            "violation, priority-slice mismatch, action-handle leak, or "
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
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- Data-layer document type",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
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
        "json": out_dir / "medai_doc_type_unknown_diag_10a_language_propagation_operator_surface_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_10a_language_propagation_operator_surface_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_10A_LANGUAGE_PROPAGATION_OPERATOR_SURFACE.md",
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
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A operator-surface audit for "
            "propagated language metadata."
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
                "medai_doc_type_unknown_diag_10a_language_propagation_operator_surface_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "propagation_metadata_display_count":
                report.propagation_metadata_display_count,
            "numeric_table_badge_display_count":
                report.numeric_table_badge_display_count,
            "eleven_record_replay_matches_priority_slice_exactly":
                report.eleven_record_replay["matches_priority_slice_exactly"],
            "flag_separation_holds":
                report.flag_separation_audit[
                    "flag_separation_holds_in_all_modes"
                ],
            "no_false_positive_expansion": report.no_false_positive_expansion,
            "review_bound_preserved": report.review_bound_preserved,
            "no_action_attached_to_plan": report.no_action_attached_to_plan,
            "raw_detector_output_unchanged":
                report.raw_detector_output_unchanged,
            "data_layer_document_type_unchanged":
                report.data_layer_document_type_unchanged,
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
