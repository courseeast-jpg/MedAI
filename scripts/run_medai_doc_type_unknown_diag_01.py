"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-01 - Unknown Residual Diagnostic.

Read-only, diagnostic-only block.

What this script does
---------------------
* Reads the existing FAMILY-04 public report from
  ``reports/medai_doc_type_family_04_larger_slice_validation/``.
* Produces aggregate, anonymized diagnostics for the three Unknown buckets:
    - insufficient_text_visibility
    - fallback_ran_but_no_family_match
    - ambiguous_below_threshold
* Writes three public reports under
  ``reports/medai_doc_type_unknown_diag_01/``.

What this script does NOT do
----------------------------
* Does NOT change OCR routing.
* Does NOT change the OCR engine.
* Does NOT change classifier behavior.
* Does NOT change thresholds, scoring, auto-accept, or review-bound safety.
* Does NOT parse lab values, medications, doses, frequencies, or DDIs.
* Does NOT call any external API.
* Does NOT emit raw filenames, raw OCR text, raw document text, private
  paths, PHI, or secrets. Anonymized sample IDs only (file_001, file_002...).

Privacy invariant: every public report row is one of:
    - an aggregate count
    - a label from a pre-vetted controlled vocabulary
    - an anonymized sample identifier of the form ``file_NNN``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_REPORT = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_family_04_larger_slice_validation"
    / "medai_doc_type_family_04_larger_slice_validation_report.json"
)
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_01"

PARK_19_COMMIT_FULL = "ac466e0f9ab84d3be371b6f2aec5fc4d35c86970"
PARK_19_COMMIT_SHORT = PARK_19_COMMIT_FULL[:12]
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized larger-slice report)"
)

# Controlled vocabulary for diagnostic labels — never includes raw content.
_VOCAB = {
    "image_like_pdf_but_not_routed_to_ocr",
    "no_text_layer",
    "text_layer_present_but_too_short",
    "language_visibility_unknown",
    "routing_not_eligible",
    "extraction_errors",
    "fallback_eligible_but_not_triggered",
    "ambiguous_candidate_set",
    "fallback_ran_no_family_match_aggregate_only",
}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class BucketDiagnostic:
    name: str
    count: int
    sub_breakdown: dict[str, int] = field(default_factory=dict)
    sample_ids: list[str] = field(default_factory=list)
    likely_next_action: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    generated_at: str
    total_unknown_analyzed: int
    bucket_counts: dict[str, int]
    buckets: dict[str, dict]
    conclusion: str
    recommendation_next_block: str
    behavior_changed: bool
    external_api_used: bool
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
    """Return placeholder anonymized IDs only — never raw filenames."""
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


_VOCAB_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_safe_label(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_VOCAB_LABEL_RE.fullmatch(value))


# ── Aggregation ──────────────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    """Aggregate Unknown bucket diagnostics from the FAMILY-04 public report."""
    unknown_buckets = source_payload.get("unknown_failure_buckets", {})
    routing = source_payload.get("unknown_ocr_routing_diagnostics", {})
    ambig_sets = source_payload.get("false_positive_risk_audit", {}).get(
        "unknown_ambiguous_candidate_sets", {}
    )

    insufficient_count = int(unknown_buckets.get("insufficient_text_visibility", 0))
    fallback_count = int(unknown_buckets.get("fallback_ran_but_no_family_match", 0))
    ambiguous_count = int(unknown_buckets.get("ambiguous_below_threshold", 0))
    total_unknown = insufficient_count + fallback_count + ambiguous_count

    insufficient_sub = routing.get("fallback_false_unknown_bucket_counts", {}) or {}
    insufficient_sub = {
        k: int(v) for k, v in insufficient_sub.items() if _is_safe_label(k)
    }

    insufficient_bucket = BucketDiagnostic(
        name="insufficient_text_visibility",
        count=insufficient_count,
        sub_breakdown=insufficient_sub,
        sample_ids=_anonymized_ids("insufficient_text_visibility", insufficient_count),
        likely_next_action=(
            "improve OCR or text-visibility coverage; must not expand cue packs"
        ),
        notes=[
            "image_like_pdf_but_not_routed_to_ocr indicates a candidate for OCR "
            "routing review in a later, dedicated block.",
            "language_visibility_unknown is the largest sub-bucket; primary lever is "
            "text-layer or OCR coverage, not classifier cue addition.",
            "Sub-buckets may overlap (a single file can match more than one "
            "reason); the bucket count is authoritative, the sub-breakdown is "
            "informational.",
            "Aggregate-only. No raw filenames, no raw text, no private paths.",
        ],
    )

    fallback_bucket = BucketDiagnostic(
        name="fallback_ran_but_no_family_match",
        count=fallback_count,
        sub_breakdown={"fallback_ran_no_family_match_aggregate_only": fallback_count},
        sample_ids=_anonymized_ids("fallback_ran_but_no_family_match", fallback_count),
        likely_next_action=(
            "audit shape cues (lab-like / imaging-like / treatment-like / generic "
            "admin / no-known-shapes) in a follow-up evaluation-only block"
        ),
        notes=[
            "Aggregate count from the FAMILY-04 public report. Per-shape "
            "breakdown for this bucket is not present in the public report; "
            "EVAL-05 introduced the shape-audit framework that could be invoked "
            "in a follow-up evaluation-only block if operational coverage matters.",
            "Failure mode is most plausibly missing family cues on degraded text "
            "rather than OCR routing -- the bucket name implies fallback ran.",
            "No cue expansion in this block.",
        ],
    )

    ambiguous_breakdown: dict[str, int] = {}
    for raw_key, raw_val in ambig_sets.items():
        normalized_key = re.sub(r"[^a-z0-9_]+", "_", str(raw_key).lower()).strip("_")
        if not normalized_key:
            continue
        ambiguous_breakdown[normalized_key] = int(raw_val)

    ambiguous_bucket = BucketDiagnostic(
        name="ambiguous_below_threshold",
        count=ambiguous_count,
        sub_breakdown=ambiguous_breakdown,
        sample_ids=_anonymized_ids("ambiguous_below_threshold", ambiguous_count),
        likely_next_action=(
            "leave review-bound; must not drive cue expansion from this bucket yet"
        ),
        notes=[
            "Higher false-positive risk. Reported as summary only per the "
            "UNKNOWN-DIAG-01 scope.",
            "All ambiguities remain review-bound; no auto-accept allowance.",
        ],
    )

    if (
        insufficient_count >= fallback_count
        and insufficient_count >= ambiguous_count
        and insufficient_count > 0
    ):
        primary_focus = "ocr_or_text_visibility"
    elif fallback_count > 0:
        primary_focus = "family_cue_coverage_under_degraded_text"
    else:
        primary_focus = "leave_manual_review"

    conclusion = (
        f"Primary diagnostic focus: {primary_focus}. The dominant Unknown bucket "
        f"is insufficient_text_visibility ({insufficient_count} of {total_unknown} "
        f"records). The fallback_ran_but_no_family_match bucket "
        f"({fallback_count}) is a secondary, evaluation-only candidate for a "
        f"shape-cue audit. ambiguous_below_threshold ({ambiguous_count}) remains "
        f"review-bound and is not optimized in this block."
    )

    recommendation = (
        "If operational coverage requires it, a follow-up evaluation-only block "
        "could (1) audit OCR/text-visibility eligibility for the "
        "insufficient_text_visibility sub-buckets, prioritizing "
        "language_visibility_unknown and text_layer_present_but_too_short, "
        "and (2) re-run the EVAL-05 shape audit specifically over the "
        "fallback_ran_but_no_family_match bucket to characterize cue gaps. "
        "Otherwise leave all 107 Unknown records review-bound."
    )

    safety_privacy = {
        "behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_changed": False,
        "scoring_changed": False,
        "auto_accept_changed": False,
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
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-01",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_unknown_analyzed=total_unknown,
        bucket_counts={
            "insufficient_text_visibility": insufficient_count,
            "fallback_ran_but_no_family_match": fallback_count,
            "ambiguous_below_threshold": ambiguous_count,
        },
        buckets={
            "insufficient_text_visibility": asdict(insufficient_bucket),
            "fallback_ran_but_no_family_match": asdict(fallback_bucket),
            "ambiguous_below_threshold": asdict(ambiguous_bucket),
        },
        conclusion=conclusion,
        recommendation_next_block=recommendation,
        behavior_changed=False,
        external_api_used=False,
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-01 - Unknown Residual Diagnostic")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): `{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- total Unknown records analyzed: `{report.total_unknown_analyzed}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## Bucket counts")
    lines.append("")
    for k, v in report.bucket_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    for name, payload in report.buckets.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- count: `{payload['count']}`")
        if payload.get("sub_breakdown"):
            lines.append("- sub-breakdown:")
            for k, v in payload["sub_breakdown"].items():
                lines.append(f"  - {k}: `{v}`")
        lines.append(f"- likely_next_action: `{payload['likely_next_action']}`")
        if payload.get("sample_ids"):
            lines.append(f"- anonymized sample IDs (synthetic): "
                         f"{', '.join('`' + s + '`' for s in payload['sample_ids'])}")
        if payload.get("notes"):
            lines.append("- notes:")
            for note in payload["notes"]:
                lines.append(f"  - {note}")
        lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(report.conclusion)
    lines.append("")
    lines.append("## Recommendation for next block (optional)")
    lines.append("")
    lines.append(report.recommendation_next_block)
    lines.append("")
    lines.append("## Safety / Privacy")
    lines.append("")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included in this report. Diagnostic-only, "
                 "evaluation/reporting changes only.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra = [
        "",
        "## Data source",
        "",
        "This block reads exclusively from the existing public report:",
        "",
        f"- `{report.source_report_label}`",
        "",
        "The source is the FAMILY-04 anonymized larger-slice batch evaluation "
        "(507 supported files). No corpus was re-read, no source documents were "
        "opened, and no external API was invoked.",
        "",
        "## Why insufficient_text_visibility is the primary focus",
        "",
        "Its dominant sub-buckets - language_visibility_unknown and "
        "text_layer_present_but_too_short - are upstream of the classifier. "
        "Adding cues would not materially recover these records because the "
        "evaluator never had enough text-shape signal to score against any "
        "family. The correct lever is OCR routing coverage or improved native "
        "text-layer extraction, both of which are out of scope for this block.",
        "",
        "## Why ambiguous_below_threshold is summary-only",
        "",
        "Each candidate set involves at least one family with higher false-",
        "positive risk if cues are loosened. The block deliberately reports the "
        "count and candidate-set distribution but does NOT propose cue "
        "expansion. All 15 records remain review-bound.",
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Classifier behavior or cue packs",
        "- Confidence thresholds or scoring",
        "- Auto-accept / review-bound policy",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
    ]
    return base + "\n".join(extra)


# ── Public-report safety guard ───────────────────────────────────────────────

_FORBIDDEN_SUBSTRINGS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".docx",
    ".doc",
    "/users/",
    "/home/",
    "c:\\",
    "begin rsa",
    "begin private key",
    "secret=",
    "secret_key=",
    "api_key=",
    "password=",
    "bearer ",
)


_FORBIDDEN_KEY_TOKEN_RE = re.compile(r"\b(?:aws|gcp|azure)_secret\b", re.IGNORECASE)


def assert_safe_public_payload(payload: Any) -> None:
    """Cheap belt-and-suspenders pre-write check.

    The richer ``check_public_report_payload`` runs in the test/validation
    step; this is a fast guard so we never produce a public file containing
    an obvious filename, path, or secret.
    """

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
        "json": out_dir / "medai_doc_type_unknown_diag_01_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_01_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_01.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-01 residual Unknown diagnostic."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=SOURCE_REPORT,
        help="Path to the FAMILY-04 public report JSON.",
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
        help="Print the JSON report to stdout and do not write files.",
    )
    args = parser.parse_args(argv)

    if not args.source_report.exists():
        print(
            f"ERROR: source report missing: {args.source_report}", file=sys.stderr
        )
        return 2

    source_payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = build_diagnostic_from_report(source_payload)

    if args.print_only:
        print(render_json(report))
        return 0

    paths = write_reports(report, out_dir=args.output_dir)
    print(json.dumps(
        {
            "conclusion": "medai_doc_type_unknown_diag_01_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_unknown_analyzed": report.total_unknown_analyzed,
            "bucket_counts": report.bucket_counts,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
