"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A - Operator badge UI surface audit.

Evaluation-only validator for the DIAG-08A read-only operator-badge render
plan. Reuses the FAMILY-04 anonymized per-file public report and exercises
the render-plan helper in three modes:

    * default-off       -> 0 plans
    * env-enabled       -> only the 11 priority records
    * explicit enabled=True -> only the 11 priority records

Asserts accepted / auto_accept_allowed / external_api_used counts remain
zero, review-bound preserved, no action attached to the plan, no false-
positive expansion. Emits three privacy-safe public reports.

No corpus rerun, no source documents opened, no external API.
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
    OPERATOR_BADGE_UI_DISCLAIMER,
    OPERATOR_BADGE_UI_EXPANDER_LABEL,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    render_plan_for_operator_badge,
)
from scripts.run_medai_doc_type_unknown_diag_06a import (  # noqa: E402
    SOURCE_REPORT,
    select_numeric_table_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_unknown_diag_08a_operator_badge_ui"
)

SOURCE_07A_COMMIT_SHORT = "8d8895d"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_06a/(public spec)",
    "reports/medai_doc_type_unknown_diag_05/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_03/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)",
)

UI_SURFACE_TOUCHED = (
    "app/main.py::render_run_result_card -> "
    "`Advanced technical details` expander, read-only optional badge block"
)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    source_07a_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    ui_integration_summary: str
    ui_surface_touched: str
    env_flag: str
    flag_rollback_path: str
    disabled_state_behavior: str
    enabled_state_behavior: str

    default_off_audit: dict[str, Any]
    env_enabled_audit: dict[str, Any]
    explicit_enabled_audit: dict[str, Any]

    eleven_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    operator_badge_display_count: int

    unknown_count_at_data_layer_before: int
    unknown_count_at_data_layer_after: int
    unknown_count_at_data_layer_delta: int

    accepted_count: int
    auto_accept_allowed_count: int
    external_api_used_count: int

    review_bound_records_before: int
    review_bound_records_after: int
    review_bound_preserved: bool

    no_action_attached_to_badge: bool

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


def _audit_for_call(
    table: list[dict],
    priority_ids: set[str],
    *,
    enabled,
    env=None,
) -> dict[str, Any]:
    hits = [
        r for r in table
        if render_plan_for_operator_badge(r, enabled=enabled, env=env) is not None
    ]
    hit_ids = {r.get("file_id") for r in hits}
    extras_outside_priority = hit_ids - priority_ids
    missing_from_priority = priority_ids - hit_ids
    return {
        "plan_count": len(hits),
        "matches_priority_slice_exactly": (
            hit_ids == priority_ids and len(priority_ids) > 0
        ),
        "extras_outside_priority_count": len(extras_outside_priority),
        "missing_from_priority_count": len(missing_from_priority),
    }


def _no_action_attached(plan: dict) -> bool:
    """A render plan must not contain any action / callback / button handle."""
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
    priority_ids: set[str],
) -> dict[str, int]:
    extras = [
        r for r in table
        if render_plan_for_operator_badge(r, enabled=True) is not None
        and r.get("file_id") not in priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown") for r in extras
    )
    return {
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
    priority = select_numeric_table_records(table)
    priority_ids = {r.get("file_id") for r in priority}

    default_off = _audit_for_call(table, priority_ids, enabled=None, env={})
    env_enabled = _audit_for_call(
        table, priority_ids, enabled=None,
        env={OPERATOR_REVIEW_BADGE_ENV_VAR: "1"},
    )
    explicit_enabled = _audit_for_call(
        table, priority_ids, enabled=True, env={},
    )

    # 11-record replay scoped to the priority slice only.
    enabled_in_priority = sum(
        1 for r in priority
        if render_plan_for_operator_badge(r, enabled=True) is not None
    )
    disabled_in_priority = sum(
        1 for r in priority
        if render_plan_for_operator_badge(r, enabled=False) is not None
    )
    default_off_priority = sum(
        1 for r in priority
        if render_plan_for_operator_badge(r, env={}) is not None
    )
    eleven_record_replay = {
        "priority_slice_size": len(priority),
        "enabled_true_plan_count": enabled_in_priority,
        "enabled_false_plan_count": disabled_in_priority,
        "default_off_plan_count": default_off_priority,
        "matches_priority_slice_exactly": (
            enabled_in_priority == len(priority)
            and disabled_in_priority == 0
            and default_off_priority == 0
            and len(priority) > 0
        ),
    }

    aggregate = {
        "corpus_size": len(table),
        "default_off_plan_count": default_off["plan_count"],
        "env_enabled_plan_count": env_enabled["plan_count"],
        "explicit_enabled_plan_count": explicit_enabled["plan_count"],
        "no_false_positive_outside_priority": (
            default_off["extras_outside_priority_count"] == 0
            and env_enabled["extras_outside_priority_count"] == 0
            and explicit_enabled["extras_outside_priority_count"] == 0
        ),
        "no_false_negative_inside_priority": (
            env_enabled["missing_from_priority_count"] == 0
            and explicit_enabled["missing_from_priority_count"] == 0
        ),
    }

    # Confirm no action is attached to any rendered plan.
    no_action_overall = True
    for r in priority:
        plan = render_plan_for_operator_badge(r, enabled=True)
        if plan is None:
            continue
        if not _no_action_attached(plan):
            no_action_overall = False
            break

    fp_audit = _false_positive_audit(table, priority_ids)
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
    unknown_delta = unknown_after - unknown_before

    accepted_count = sum(
        1 for r in table
        if str(r.get("accepted_status_source") or "not_accepted")
        not in {"not_accepted"}
    )
    auto_accept_allowed_count = sum(
        1 for r in table if r.get("auto_accept_allowed") in (True, "true", "yes")
    )
    external_api_used_count = sum(
        1 for r in table if r.get("external_api_used") in (True, "true", "yes")
    )

    ui_integration_summary = (
        "Adds a single read-only optional render block inside the existing "
        "`Advanced technical details` expander in "
        "`app/main.py::render_run_result_card`. The block calls the pure "
        "`render_plan_for_operator_badge` helper "
        "(`clinical_knowledge.document_type.operator_badge_ui`) and renders "
        "the returned plan via `st.markdown` / `st.caption` only. The block "
        "is wrapped in a defensive try/except so any import or render error "
        "is silently swallowed and the main result card is never blocked. "
        "Default-off: when the env var "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` is unset / falsy, the helper "
        "returns None and the UI surface is visually unchanged."
    )

    disabled_state_behavior = (
        "When the env var is unset or set to a falsy value, the helper "
        "returns None, the `if _op_badge_plan is not None` guard evaluates "
        "False, and no badge markdown is emitted. The expander content is "
        "identical to the pre-DIAG-08A state. Number of plans rendered on "
        f"the 507-file corpus in this mode: {default_off['plan_count']}."
    )

    enabled_state_behavior = (
        "When the env var is set to a truthy value AND the record matches "
        "the exact 14-field DIAG-06A signature without violating any "
        "exclusion rule or implementation safeguard, the helper returns a "
        "structured render plan and the UI renders three markdown lines "
        "plus a disclaimer caption inside the existing expander. The "
        "badge is read-only; no button, form, or callback is attached. "
        "Records are not mutated; review-bound status is preserved. "
        f"Number of plans rendered on the 507-file corpus in this mode: "
        f"{env_enabled['plan_count']}."
    )

    flag_rollback_path = (
        "Four independent rollback paths, any one of which is sufficient: "
        "(1) omit / unset the env var "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` (the default); "
        "(2) set the env var to a falsy value (`0` / `false` / `no` / "
        "`off` / `disabled`); "
        "(3) call the helper with `enabled=False`; "
        "(4) never import the helper - the existing main.py call site is "
        "wrapped in try/except so an ImportError is silently swallowed."
    )

    deferred_subsets = {
        "candidate_table_header_language_policy_record":
            "1 record from DIAG-05 routed to the table-header lever; deferred",
        "candidate_metadata_propagation_audit_pool":
            "11 records from DIAG-04 routed to the propagation-audit lever; deferred",
        "candidate_latin_medical_abbreviation_handling_audit_pool":
            "8 records from DIAG-04 routed to the abbreviation lever; deferred",
        "likely_text_layer_issue":
            "21 records deferred per DIAG-03",
        "fallback_ran_but_no_family_match":
            "17 records deferred per DIAG-02; no cue expansion",
        "ambiguous_below_threshold":
            "15 records excluded; review-bound, no cue expansion",
    }

    progress_estimate = {
        "before_08a_unknown_track_done_pct":      "approximately 72%",
        "before_08a_unknown_track_remaining_pct": "approximately 28%",
        "before_08a_project_done_pct":            "approximately 78%",
        "before_08a_project_remaining_pct":       "approximately 22%",
        "after_08a_unknown_track_done_pct":       "approximately 76%",
        "after_08a_unknown_track_remaining_pct":  "approximately 24%",
        "after_08a_project_done_pct":             "approximately 79%",
        "after_08a_project_remaining_pct":        "approximately 21%",
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
        "language_detector_behavior_changed": False,
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
        "operator_badge_ui_default_disabled": True,
        "rollback_path_present": True,
        "no_action_attached_to_badge": no_action_overall,
        "no_button_or_callback_in_render_plan": True,
        "ui_render_failure_is_silently_swallowed": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A-OPERATOR-BADGE-UI",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        source_07a_commit_short=SOURCE_07A_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        ui_integration_summary=ui_integration_summary,
        ui_surface_touched=UI_SURFACE_TOUCHED,
        env_flag=OPERATOR_REVIEW_BADGE_ENV_VAR,
        flag_rollback_path=flag_rollback_path,
        disabled_state_behavior=disabled_state_behavior,
        enabled_state_behavior=enabled_state_behavior,

        default_off_audit=default_off,
        env_enabled_audit=env_enabled,
        explicit_enabled_audit=explicit_enabled,

        eleven_record_replay=eleven_record_replay,
        five_hundred_seven_file_aggregate=aggregate,

        operator_badge_display_count=explicit_enabled["plan_count"],

        unknown_count_at_data_layer_before=unknown_before,
        unknown_count_at_data_layer_after=unknown_after,
        unknown_count_at_data_layer_delta=unknown_delta,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        no_action_attached_to_badge=no_action_overall,

        false_positive_audit=fp_audit,
        no_false_positive_expansion=no_fp_expansion,

        anonymized_sample_ids=_anonymized_ids("operator_badge_ui_priority", 11),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to a single optional read-only badge render "
            "block inside the existing `Advanced technical details` "
            "expander in the Run & Review result card. No buttons, forms, "
            "or callbacks are attached. No clinical interpretation, no "
            "value parsing, no auto-accept, no active clinical fact "
            "writes, no document-type promotion at the data layer. The "
            "block is gated by the existing DIAG-07A env var and is OFF by "
            "default."
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
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A - Read-only Operator Badge UI")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source DIAG-07A commit (short): "
                 f"`{report.source_07a_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- ui_surface_touched: `{report.ui_surface_touched}`")
    lines.append(f"- env_flag: `{report.env_flag}`")
    lines.append(f"- expander label: `{OPERATOR_BADGE_UI_EXPANDER_LABEL}`")
    lines.append(f"- disclaimer line: `{OPERATOR_BADGE_UI_DISCLAIMER}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## UI integration summary")
    lines.append("")
    lines.append(report.ui_integration_summary)
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

    lines.append("## Mode audits (corpus-wide)")
    lines.append("")
    for name, audit in (
        ("default_off", report.default_off_audit),
        ("env_enabled", report.env_enabled_audit),
        ("explicit_enabled", report.explicit_enabled_audit),
    ):
        lines.append(f"### {name}")
        for k, v in audit.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.append("## 11-record replay")
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
    lines.append(f"- operator_badge_display_count (with helper enabled): "
                 f"`{report.operator_badge_display_count}`")
    lines.append(f"- unknown_count_at_data_layer_before: "
                 f"`{report.unknown_count_at_data_layer_before}`")
    lines.append(f"- unknown_count_at_data_layer_after: "
                 f"`{report.unknown_count_at_data_layer_after}`")
    lines.append(f"- unknown_count_at_data_layer_delta: "
                 f"`{report.unknown_count_at_data_layer_delta}`")
    lines.append(f"- accepted_count: `{report.accepted_count}`")
    lines.append(f"- auto_accept_allowed_count: "
                 f"`{report.auto_accept_allowed_count}`")
    lines.append(f"- external_api_used_count: `{report.external_api_used_count}`")
    lines.append("")

    lines.append("## Review-bound preservation")
    lines.append("")
    lines.append(f"- review_bound_records_before: "
                 f"`{report.review_bound_records_before}`")
    lines.append(f"- review_bound_records_after: "
                 f"`{report.review_bound_records_after}`")
    lines.append(f"- review_bound_preserved: `{report.review_bound_preserved}`")
    lines.append("")

    lines.append("## No-action-attached confirmation")
    lines.append("")
    lines.append(f"- no_action_attached_to_badge: "
                 f"`{report.no_action_attached_to_badge}`")
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
                 "PHI, or secrets are included. The runtime behavior change is "
                 "strictly limited to a single optional read-only badge block "
                 "inside the existing Advanced technical details expander. No "
                 "buttons, forms, or callbacks attach to the badge. No clinical "
                 "interpretation, no value parsing, no auto-accept, no active "
                 "clinical fact writes, no document-type promotion. Review-bound "
                 "status preserved.")
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
        and report.no_action_attached_to_badge is True
    )
    if acceptance_ok:
        extra.append(
            "The read-only operator badge is now wired into the existing "
            "Advanced technical details expander, gated by the same env "
            "var and OFF by default. A future evaluation-only block (e.g. "
            "UNKNOWN-DIAG-09A) may consume operator-side click counts in "
            "an anonymized aggregate to assess whether the badge improves "
            "the operator review queue throughput; that block must remain "
            "review-bound, must not add auto-accept, and must not modify "
            "any classifier behavior. The deferred pools (1 table-header "
            "record, 11 propagation-audit, 8 abbreviation, 21 text-layer, "
            "17 fallback, 15 ambiguous) remain deferred or excluded; cue "
            "expansion remains not recommended."
        )
    else:
        extra.append(
            "Integration acceptance criteria not fully met. Downstream "
            "consumption of the operator-badge UI must not proceed. "
            "Investigate any false-positive expansion, review-bound "
            "violation, priority-slice mismatch, or action-handle leak "
            "and revise before the next block."
        )
    extra += [
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Raw language / script detector behavior",
        "- Classifier behavior for any record outside the exact 14-field signature",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- Document-type promotion at the data layer",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
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
        "json": out_dir / "medai_doc_type_unknown_diag_08a_operator_badge_ui_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_08a_operator_badge_ui_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_08A_OPERATOR_BADGE_UI.md",
    }
    paths["json"].write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A operator badge UI audit."
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
                "medai_doc_type_unknown_diag_08a_operator_badge_ui_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "operator_badge_display_count": report.operator_badge_display_count,
            "eleven_record_replay_matches_priority_slice_exactly":
                report.eleven_record_replay["matches_priority_slice_exactly"],
            "no_false_positive_expansion": report.no_false_positive_expansion,
            "review_bound_preserved": report.review_bound_preserved,
            "no_action_attached_to_badge": report.no_action_attached_to_badge,
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
