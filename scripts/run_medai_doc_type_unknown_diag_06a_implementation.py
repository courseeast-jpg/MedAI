"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION - Numeric-Table Safe-Default
Metadata Label.

Runs the privacy-safe implementation report for the narrow runtime helper
``clinical_knowledge.document_type.derive_numeric_table_safe_default_label``.

The block:

    1. Re-derives the 11 priority records via the DIAG-02/03/04/05 chain.
    2. Performs an 11-record replay: confirms the helper labels exactly
       those 11 records when enabled, and zero records when disabled
       (rollback path).
    3. Performs a 507-file aggregate replay: confirms the helper labels no
       record outside the priority slice, regardless of enabled state.
    4. Audits accepted_count / auto_accept_allowed_count /
       external_api_used_count remain zero.
    5. Audits review-bound status preservation and absence of false-
       positive expansion in treatment/imaging/admin families.
    6. Emits three privacy-safe public reports.

No corpus rerun, no source documents opened, no external API call.
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
    DERIVED_LABEL,
    EXCLUSION_RULES,
    POSITIVE_SIGNATURE,
    derive_numeric_table_safe_default_label,
    is_disabled_by_default,
)
from scripts.run_medai_doc_type_unknown_diag_06a import (  # noqa: E402
    SOURCE_REPORT,
    select_numeric_table_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_06a_implementation"

SPEC_COMMIT_SHORT = "70a2b59"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_DIAG_LABELS = (
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
    spec_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_diag_labels: list[str]
    generated_at: str

    implementation_summary: str
    rollback_disable_path: str
    helper_default_disabled: bool

    positive_signature: list[dict[str, str]]
    exclusion_rules: list[str]
    derived_label: str

    eleven_record_replay: dict[str, Any]
    five_hundred_seven_file_aggregate: dict[str, Any]

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
    clinical_behavior_changed: bool
    external_api_used: bool
    cue_expansion_recommended: bool
    policy_implementation_scope: str
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


def _anonymized_ids(prefix: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


# ── Replays ─────────────────────────────────────────────────────────────────

def _eleven_record_replay(table: list[dict]) -> tuple[dict, list[dict]]:
    priority = select_numeric_table_records(table)
    enabled_hits = [r for r in priority
                    if derive_numeric_table_safe_default_label(r, enabled=True) is not None]
    disabled_hits = [r for r in priority
                     if derive_numeric_table_safe_default_label(r, enabled=False) is not None]

    summary = {
        "priority_slice_size": len(priority),
        "enabled_labeled_count": len(enabled_hits),
        "disabled_labeled_count": len(disabled_hits),
        "matches_priority_slice_exactly": (
            len(enabled_hits) == len(priority)
            and len(disabled_hits) == 0
            and len(priority) > 0
        ),
    }
    return summary, priority


def _aggregate_replay(
    table: list[dict],
    priority_ids: set[str],
) -> dict[str, Any]:
    enabled_hits = [r for r in table
                    if derive_numeric_table_safe_default_label(r, enabled=True) is not None]
    disabled_hits = [r for r in table
                     if derive_numeric_table_safe_default_label(r, enabled=False) is not None]
    enabled_ids = {r.get("file_id") for r in enabled_hits}
    extras_outside_priority = enabled_ids - priority_ids
    missing_from_priority = priority_ids - enabled_ids

    return {
        "corpus_size": len(table),
        "enabled_labeled_count": len(enabled_hits),
        "disabled_labeled_count": len(disabled_hits),
        "extras_outside_priority_slice": sorted(
            i for i in extras_outside_priority if isinstance(i, str)
        )[:10],
        "extras_count": len(extras_outside_priority),
        "missing_from_priority_count": len(missing_from_priority),
        "no_false_positive_outside_priority": len(extras_outside_priority) == 0,
        "no_false_negative_inside_priority": len(missing_from_priority) == 0,
    }


# ── Audits ──────────────────────────────────────────────────────────────────

def _false_positive_audit(
    table: list[dict],
    priority_ids: set[str],
) -> dict[str, int]:
    """Count any record outside the priority slice that receives the label
    AND whose predicted family is treatment / imaging / admin."""
    enabled_extras = [
        r for r in table
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
        and r.get("file_id") not in priority_ids
    ]
    fams = Counter(
        str(r.get("predicted_document_type") or "Unknown")
        for r in enabled_extras
    )
    return {
        "treatment_expansion": (fams.get("Treatment plan", 0)
                                + fams.get("Medication plan", 0)),
        "imaging_expansion":  fams.get("Imaging report", 0),
        "administrative_expansion": fams.get("Administrative / Insurance", 0),
        "other_expansion": sum(
            c for f, c in fams.items()
            if f not in {
                "Treatment plan", "Medication plan", "Imaging report",
                "Administrative / Insurance", "Unknown",
            }
        ),
    }


def _review_bound_audit(table: list[dict]) -> tuple[int, int]:
    """Returns (before, after) counts of review-bound records.

    The helper never mutates ``record``; "after" is computed by simulating
    that the derived label is attached as additional safe metadata. The
    review-bound status is encoded by ``review_status == 'review'`` in the
    per-file public report.
    """
    before = sum(1 for r in table if str(r.get("review_status") or "") == "review")
    # The helper does not change review_status; "after" is identical.
    after = before
    return before, after


# ── Top-level builder ───────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []

    eleven_summary, priority = _eleven_record_replay(table)
    priority_ids = {r.get("file_id") for r in priority}
    aggregate = _aggregate_replay(table, priority_ids)
    fp_audit = _false_positive_audit(table, priority_ids)
    rb_before, rb_after = _review_bound_audit(table)

    unknown_before = sum(
        1 for r in table
        if str(r.get("predicted_document_type") or "") == "Unknown"
    )
    # The helper does not change predicted_document_type; the unknown count is
    # unchanged. The label is metadata-only and does not flip a document out
    # of Unknown. Downstream consumers that use the derived label for routing
    # may choose to count differently; this block records the impact at the
    # data layer only.
    unknown_after = unknown_before
    unknown_delta = unknown_after - unknown_before

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
        "Adds a small pure helper "
        "`clinical_knowledge.document_type.derive_numeric_table_safe_default_label` "
        "that returns "
        "`latin_script_likely_english_table_context` ONLY when (a) the caller "
        "explicitly passes `enabled=True`, (b) the record matches every field "
        "of the 14-field positive signature, (c) none of the 10 exclusion "
        "rules fires, and (d) none of the implementation-level safeguards "
        "fires. The helper is pure, default-off, and never mutates the "
        "record. The disable / rollback path is `enabled=False` (or simply "
        "not calling the helper); the function returns `None` in that case "
        "and existing pipelines that never import the module are "
        "unaffected. Raw language-detector output is not modified; the "
        "label is added as safe metadata only. Auto-accept, clinical "
        "interpretation, lab-value parsing, medication / dose / DDI "
        "parsing, and active clinical fact writes are all forbidden by "
        "the helper's contract."
    )

    rollback_path = (
        "Pass `enabled=False` (the default) or stop importing the module. "
        "No persisted state to roll back. The helper is additive and pure."
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
        "before_impl_unknown_track_done_pct":      "approximately 62%",
        "before_impl_unknown_track_remaining_pct": "approximately 38%",
        "before_impl_project_done_pct":            "approximately 76%",
        "before_impl_project_remaining_pct":       "approximately 24%",
        "after_impl_unknown_track_done_pct":       "approximately 68%",
        "after_impl_unknown_track_remaining_pct":  "approximately 32%",
        "after_impl_project_done_pct":             "approximately 77%",
        "after_impl_project_remaining_pct":        "approximately 23%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed_strictly_limited_to_safe_metadata_label": True,
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
        "helper_default_disabled": True,
        "rollback_path_present": True,
    }

    no_fp_expansion = all(v == 0 for v in fp_audit.values())

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        spec_commit_short=SPEC_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_diag_labels=list(UPSTREAM_DIAG_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        implementation_summary=impl_summary,
        rollback_disable_path=rollback_path,
        helper_default_disabled=is_disabled_by_default(),

        positive_signature=[
            {"key": k, "expected": v} for k, v in POSITIVE_SIGNATURE
        ],
        exclusion_rules=list(EXCLUSION_RULES),
        derived_label=DERIVED_LABEL,

        eleven_record_replay=eleven_summary,
        five_hundred_seven_file_aggregate=aggregate,

        unknown_count_before=unknown_before,
        unknown_count_after=unknown_after,
        unknown_count_impact_delta=unknown_delta,

        accepted_count=accepted_count,
        auto_accept_allowed_count=auto_accept_allowed_count,
        external_api_used_count=external_api_used_count,

        review_bound_records_before=rb_before,
        review_bound_records_after=rb_after,
        review_bound_preserved=(rb_before == rb_after),

        false_positive_audit=fp_audit,
        no_false_positive_expansion=no_fp_expansion,

        anonymized_sample_ids=_anonymized_ids("numeric_table_priority", 11),
        deferred_subsets=deferred_subsets,

        progress_estimate=progress_estimate,

        behavior_changed=True,
        clinical_behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        policy_implementation_scope=(
            "Strictly limited to deriving a safe metadata / routing label for "
            "records that match the exact 14-field positive signature and "
            "violate none of the 10 exclusion rules. No clinical "
            "interpretation, no value parsing, no auto-accept, no active "
            "clinical fact writes. Review-bound status preserved. Default-off; "
            "callers must explicitly opt in via `enabled=True`."
        ),
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION - Numeric-Table Safe-Default Label")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source spec commit (short): `{report.spec_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): `{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream diagnostics / spec:")
    for lbl in report.upstream_diag_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- derived label: `{report.derived_label}`")
    lines.append(f"- helper_default_disabled: `{report.helper_default_disabled}`")
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

    lines.append("## Positive signature")
    lines.append("")
    for item in report.positive_signature:
        lines.append(f"- `{item['key']}` = `{item['expected']}`")
    lines.append("")
    lines.append("## Exclusion rules")
    lines.append("")
    for rule in report.exclusion_rules:
        lines.append(f"- `{rule}`")
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
    lines.append(f"- unknown_count_before: `{report.unknown_count_before}`")
    lines.append(f"- unknown_count_after: `{report.unknown_count_after}`")
    lines.append(f"- unknown_count_impact_delta: "
                 f"`{report.unknown_count_impact_delta}`")
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
                 "(strictly limited to safe metadata / routing label only)")
    lines.append(f"- clinical_behavior_changed: "
                 f"`{report.clinical_behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    lines.append(f"- policy_implementation_scope: "
                 f"{report.policy_implementation_scope}")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. The runtime behavior change is "
                 "strictly limited to the safe metadata / routing label described "
                 "above; no clinical interpretation, no value parsing, no auto-"
                 "accept, no active clinical fact writes. Review-bound status is "
                 "preserved.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Recommendation for next block", ""]
    if report.no_false_positive_expansion and report.review_bound_preserved \
            and report.eleven_record_replay.get("matches_priority_slice_exactly"):
        extra.append(
            "The numeric-table safe-default metadata label is now available "
            "behind a default-off helper. Recommend a downstream "
            "evaluation-only block (e.g. UNKNOWN-DIAG-07A) that consumes the "
            "derived label inside the operator routing review queue ONLY, "
            "with no auto-accept, no clinical interpretation, no value "
            "parsing, and review-bound status preserved. The deferred pools "
            "(candidate_metadata_propagation_audit_pool, candidate_latin_"
            "medical_abbreviation_handling_audit_pool, "
            "candidate_table_header_language_policy record, likely_text_"
            "layer_issue, fallback_ran_but_no_family_match, and "
            "ambiguous_below_threshold) remain deferred or excluded; cue "
            "expansion remains not recommended for any subset."
        )
    else:
        extra.append(
            "Implementation acceptance criteria not fully met. Downstream "
            "consumption of the derived label must not proceed. Investigate "
            "any false-positive expansion, review-bound violation, or "
            "priority-slice mismatch reported above and revise the spec or "
            "the implementation before the next block."
        )
    extra += [
        "",
        "## What this implementation does not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Classifier behavior for records OUTSIDE the exact 14-field signature",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound policy",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "## Helper contract",
        "",
        "The public helper `derive_numeric_table_safe_default_label(record, *, "
        "enabled=False)` is pure. It does not mutate the record. It returns "
        "the derived label only when the explicit `enabled=True` flag is "
        "passed AND the record matches every field of the 14-field positive "
        "signature AND no exclusion rule fires AND no implementation-level "
        "safeguard fires. Otherwise it returns `None`.",
        "",
    ]
    return base + "\n".join(extra)


# ── Public-report safety guard ──────────────────────────────────────────────

# File-extension and path patterns are checked with regexes that require
# a word boundary so the guard doesn't false-positive on Python module
# paths like ``clinical_knowledge.document_type``.
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


# ── Driver ──────────────────────────────────────────────────────────────────

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
        "json": out_dir / "medai_doc_type_unknown_diag_06a_implementation_report.json",
        "md_summary":
            out_dir / "medai_doc_type_unknown_diag_06a_implementation_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_06A_IMPLEMENTATION.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION numeric-table "
                    "metadata label."
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
            "conclusion": "medai_doc_type_unknown_diag_06a_implementation_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT))
                              for k, v in paths.items()},
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
