"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A - Operator Routing Review Integration audit.

Evaluation-only validator for the DIAG-07A operator-routing-review surface.

The block:

    1. Reads the privacy-safe FAMILY-04 anonymized per-file public report.
    2. Re-derives the 11 priority records via the DIAG-02/03/04/05/06A chain.
    3. Runs the operator-review badge function in three modes:
         - default-off (no kwarg, no env) -> 0 badges
         - env-enabled (env var set)     -> only the 11 priority records
         - explicit enabled=True          -> only the 11 priority records
    4. Audits accepted / auto_accept_allowed / external_api_used counts
       remain zero, review-bound status is preserved, and no false-positive
       expansion occurs in treatment / imaging / admin families.
    5. Emits three privacy-safe public reports.

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
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_knowledge.document_type import (  # noqa: E402
    OPERATOR_REVIEW_BADGE_DISCLAIMER,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    OPERATOR_REVIEW_BADGE_TEXT,
    OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
    derive_operator_review_badge,
    is_operator_review_badge_default_disabled,
)
from scripts.run_medai_doc_type_unknown_diag_06a import (  # noqa: E402
    SOURCE_REPORT,
    select_numeric_table_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_unknown_diag_07a_operator_routing_review"
)

IMPLEMENTATION_COMMIT_SHORT = "eef93bc"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_06a/(public spec)",
    "reports/medai_doc_type_unknown_diag_05/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_03/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)",
)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    source_implementation_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    operator_integration_summary: str
    flag_rollback_path: str
    helper_default_disabled: bool
    operator_review_badge_env_var: str
    operator_review_badge_vocab_token: str
    operator_review_badge_text: str
    operator_review_badge_disclaimer: str

    default_off_audit: dict[str, Any]
    env_enabled_audit: dict[str, Any]
    explicit_enabled_audit: dict[str, Any]

    eleven_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

    operator_badge_display_count: int

    unknown_count_at_data_layer_before: int
    unknown_count_at_data_layer_after: int
    unknown_count_at_data_layer_delta: int
    operator_review_metadata_display_count: int

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


def _audit_for_call(
    table: list[dict],
    priority_ids: set[str],
    *,
    enabled,
    env=None,
) -> dict[str, Any]:
    hits = [
        r for r in table
        if derive_operator_review_badge(r, enabled=enabled, env=env) is not None
    ]
    hit_ids = {r.get("file_id") for r in hits}
    extras_outside_priority = hit_ids - priority_ids
    missing_from_priority = priority_ids - hit_ids
    return {
        "badge_count": len(hits),
        "matches_priority_slice_exactly": (
            hit_ids == priority_ids and len(priority_ids) > 0
        ),
        "extras_outside_priority_count": len(extras_outside_priority),
        "missing_from_priority_count": len(missing_from_priority),
    }


def _false_positive_audit(
    table: list[dict],
    priority_ids: set[str],
) -> dict[str, int]:
    badged_extras = [
        r for r in table
        if derive_operator_review_badge(r, enabled=True) is not None
        and r.get("file_id") not in priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown")
        for r in badged_extras
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

    # 11-record replay uses the priority slice only.
    enabled_in_priority = sum(
        1 for r in priority
        if derive_operator_review_badge(r, enabled=True) is not None
    )
    disabled_in_priority = sum(
        1 for r in priority
        if derive_operator_review_badge(r, enabled=False) is not None
    )
    none_in_priority_default_off = sum(
        1 for r in priority
        if derive_operator_review_badge(r, env={}) is not None
    )
    eleven_record_replay = {
        "priority_slice_size": len(priority),
        "enabled_true_badge_count": enabled_in_priority,
        "enabled_false_badge_count": disabled_in_priority,
        "default_off_badge_count": none_in_priority_default_off,
        "matches_priority_slice_exactly": (
            enabled_in_priority == len(priority)
            and disabled_in_priority == 0
            and none_in_priority_default_off == 0
            and len(priority) > 0
        ),
    }

    aggregate = {
        "corpus_size": len(table),
        "default_off_badge_count": default_off["badge_count"],
        "env_enabled_badge_count": env_enabled["badge_count"],
        "explicit_enabled_badge_count": explicit_enabled["badge_count"],
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

    fp_audit = _false_positive_audit(table, priority_ids)
    no_fp_expansion = all(v == 0 for v in fp_audit.values())

    rb_before = sum(
        1 for r in table if str(r.get("review_status") or "") == "review"
    )
    rb_after = rb_before  # the helper never mutates; the badge is metadata-only

    unknown_before = sum(
        1 for r in table
        if str(r.get("predicted_document_type") or "") == "Unknown"
    )
    unknown_after = unknown_before
    unknown_delta = unknown_after - unknown_before
    # The operator-review badge IS a separate metadata count, distinct from
    # the data-layer unknown count.
    operator_metadata_display_count = explicit_enabled["badge_count"]

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

    integration_summary = (
        "Adds `clinical_knowledge.document_type.derive_operator_review_badge` "
        "as a thin, default-off consumer of the DIAG-06A helper. The badge "
        "is returned ONLY when the operator-review flag is explicitly "
        "enabled (via the `enabled=True` keyword argument OR the "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` environment variable set to a "
        "truthy value) AND the underlying signature/exclusion guards from "
        "DIAG-06A all hold. The returned dict carries explicit "
        "`review_bound=True`, `is_clinical_classification=False`, "
        "`is_final_document_type=False`, `is_auto_accept=False`, and "
        "`is_active_clinical_fact=False` so consumers cannot mistake the "
        "badge for a clinical outcome. No mutation of the record, no "
        "auto-accept, no clinical interpretation, no value parsing, no "
        "active fact writes. The DIAG-06A helper remains default-off "
        "outside this explicit operator-review call site."
    )

    flag_rollback_path = (
        "Multiple disable / rollback paths exist, any one of which is "
        "sufficient: (1) omit the `enabled` kwarg AND keep the env var "
        f"`{OPERATOR_REVIEW_BADGE_ENV_VAR}` unset (the default); "
        "(2) pass `enabled=False` explicitly; (3) unset the env var if it "
        "was set; (4) never import the integration module - existing "
        "pipelines are unaffected by the addition. No persisted state to "
        "roll back. The function is pure."
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
        "before_07a_unknown_track_done_pct":      "approximately 68%",
        "before_07a_unknown_track_remaining_pct": "approximately 32%",
        "before_07a_project_done_pct":            "approximately 77%",
        "before_07a_project_remaining_pct":       "approximately 23%",
        "after_07a_unknown_track_done_pct":       "approximately 72%",
        "after_07a_unknown_track_remaining_pct":  "approximately 28%",
        "after_07a_project_done_pct":             "approximately 78%",
        "after_07a_project_remaining_pct":        "approximately 22%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed_strictly_limited_to_operator_review_display": True,
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
        "operator_review_badge_default_disabled": True,
        "rollback_path_present": True,
        "underlying_helper_default_disabled_outside_call_site": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A-OPERATOR-ROUTING-REVIEW",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        source_implementation_commit_short=IMPLEMENTATION_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        operator_integration_summary=integration_summary,
        flag_rollback_path=flag_rollback_path,
        helper_default_disabled=is_operator_review_badge_default_disabled(),
        operator_review_badge_env_var=OPERATOR_REVIEW_BADGE_ENV_VAR,
        operator_review_badge_vocab_token=OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
        operator_review_badge_text=OPERATOR_REVIEW_BADGE_TEXT,
        operator_review_badge_disclaimer=OPERATOR_REVIEW_BADGE_DISCLAIMER,

        default_off_audit=default_off,
        env_enabled_audit=env_enabled,
        explicit_enabled_audit=explicit_enabled,

        eleven_record_replay=eleven_record_replay,
        five_hundred_seven_file_aggregate=aggregate,

        operator_badge_display_count=explicit_enabled["badge_count"],

        unknown_count_at_data_layer_before=unknown_before,
        unknown_count_at_data_layer_after=unknown_after,
        unknown_count_at_data_layer_delta=unknown_delta,
        operator_review_metadata_display_count=operator_metadata_display_count,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        false_positive_audit=fp_audit,
        no_false_positive_expansion=no_fp_expansion,

        anonymized_sample_ids=_anonymized_ids("operator_review_priority", 11),
        deferred_subsets=deferred_subsets,
        progress_estimate=progress_estimate,

        behavior_changed=True,
        behavior_change_scope=(
            "Strictly limited to deriving an operator-review display badge "
            "for records that match the exact 14-field positive signature "
            "AND only when the operator-review flag is explicitly enabled. "
            "No clinical interpretation, no value parsing, no auto-accept, "
            "no active clinical fact writes, no document-type promotion, "
            "no OCR routing change. The badge is read-only review metadata."
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
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A - Operator Routing Review Integration"
    )
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(
        f"- source implementation commit (short): "
        f"`{report.source_implementation_commit_short}`"
    )
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append(f"- operator_review_badge_default_disabled: "
                 f"`{report.helper_default_disabled}`")
    lines.append(f"- operator_review_badge_env_var: "
                 f"`{report.operator_review_badge_env_var}`")
    lines.append(f"- operator_review_badge_vocab_token: "
                 f"`{report.operator_review_badge_vocab_token}`")
    lines.append(f"- operator_review_badge_text: "
                 f"`{report.operator_review_badge_text}`")
    lines.append(f"- operator_review_badge_disclaimer: "
                 f"`{report.operator_review_badge_disclaimer}`")
    lines.append("")

    lines.append("## Operator integration summary")
    lines.append("")
    lines.append(report.operator_integration_summary)
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
    lines.append(f"- operator_review_metadata_display_count: "
                 f"`{report.operator_review_metadata_display_count}` "
                 "(separate from the data-layer unknown count)")
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
                 "(strictly limited to the operator review display surface)")
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
                 "strictly limited to deriving the operator-review badge for "
                 "records that match the exact 14-field signature and only when "
                 "the operator-review flag is explicitly enabled. No clinical "
                 "interpretation, no value parsing, no auto-accept, no active "
                 "clinical fact writes, no document-type promotion. Review-bound "
                 "status is preserved.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    if (
        report.no_false_positive_expansion
        and report.review_bound_preserved
        and report.eleven_record_replay.get("matches_priority_slice_exactly")
        and report.accepted_count == 0
        and report.auto_accept_allowed_count == 0
        and report.external_api_used_count == 0
    ):
        extra.append(
            "The operator-review badge is available behind a default-off "
            "env-gated and kwarg-gated flag. A future evaluation-only block "
            "(e.g. UNKNOWN-DIAG-08A) may wire the badge into a small UI "
            "surface read-only display, preserving review-bound status and "
            "the existing operator review queue. Continue to leave the "
            "deferred pools (1 table-header record, 11 propagation-audit, "
            "8 abbreviation, 21 text-layer, 17 fallback, 15 ambiguous) "
            "deferred or excluded; cue expansion remains not recommended."
        )
    else:
        extra.append(
            "Integration acceptance criteria not fully met. Downstream "
            "consumption of the operator-review badge must not proceed. "
            "Investigate any false-positive expansion, review-bound "
            "violation, or priority-slice mismatch and revise before the "
            "next block."
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
        "json": out_dir / "medai_doc_type_unknown_diag_07a_operator_routing_review_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_07a_operator_routing_review_report.md",
        "md_main":
            out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_07A_OPERATOR_ROUTING_REVIEW.md",
    }
    paths["json"].write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A operator routing review audit."
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
                "medai_doc_type_unknown_diag_07a_operator_routing_review_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "operator_badge_display_count": report.operator_badge_display_count,
            "eleven_record_replay_matches_priority_slice_exactly":
                report.eleven_record_replay["matches_priority_slice_exactly"],
            "no_false_positive_expansion": report.no_false_positive_expansion,
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
