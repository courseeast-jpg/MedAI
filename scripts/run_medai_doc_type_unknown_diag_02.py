"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-02 - OCR/Text Visibility + Fallback Shape Audit.

Evaluation-only, read-only block.

What this script does
---------------------
* Reads the existing safe FAMILY-04 batch-eval public report and projects
  the 507-row anonymized per-file table into three failure-bucket views.
* Produces aggregate-only counts for:
    Section A - OCR / text-visibility eligibility (75 records)
    Section B - fallback shape-audit (17 records)
    Section C - ambiguous bucket handling (15 records, count only)
* Writes three public diagnostic files under
  ``reports/medai_doc_type_unknown_diag_02/``.

What this script does NOT do
----------------------------
* Does NOT rerun the corpus, does NOT open source documents.
* Does NOT change OCR routing or the OCR engine.
* Does NOT change classifier behavior, thresholds, scoring, or auto-accept.
* Does NOT add cue packs or modify the cue-audit framework.
* Does NOT parse lab values, medications, doses, frequencies, or DDIs.
* Does NOT call any external API.
* Does NOT emit raw filenames, raw OCR text, raw document text, private
  paths, PHI, or secrets. Anonymized aggregate output only.

Source data
-----------
The per-file ``anonymous_per_file_table`` of the FAMILY-04 batch-eval
public report carries privacy-safe ``file_id`` placeholders plus the
``unknown_failure_bucket`` and EVAL-05 cue-audit fields. Both sections
are derived directly from that public report; no other input is read.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_REPORT = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_family_04_larger_slice_validation"
    / "medai_doc_type_eval_01_report.json"
)
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_02"

PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_DIAG_01_LABEL = "reports/medai_doc_type_unknown_diag_01/(public diagnostic)"

# Controlled vocabulary for shape-audit verdicts (EVAL-05).
_SHAPE_AUDIT_VERDICTS = (
    "possible_lab_shape_without_language_cues",
    "possible_imaging_shape_without_language_cues",
    "possible_treatment_shape_without_language_cues",
    "generic_form_shapes_only",
    "likely_nonmedical_or_header_noise",
    "no_known_family_shapes",
    "needs_manual_review",
)

# Diagnostic labels for OCR/text-visibility eligibility (Section A).
_OCR_VISIBILITY_LABELS = (
    "likely_text_layer_issue",
    "likely_ocr_eligibility_issue",
    "image_like_but_ocr_not_routed",
    "no_text_layer",
    "language_script_visible_detector_unresolved",
    "fallback_eligible_but_not_triggered",
    "extraction_error",
    "non_actionable_leave_manual_review",
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SectionAReport:
    """OCR / text-visibility eligibility for insufficient_text_visibility (75)."""
    name: str = "insufficient_text_visibility"
    count: int = 0
    ocr_visibility_breakdown: dict[str, int] = field(default_factory=dict)
    raw_routing_bucket_counts: dict[str, int] = field(default_factory=dict)
    raw_visibility_status_counts: dict[str, int] = field(default_factory=dict)
    pdf_text_layer_detected_counts: dict[str, int] = field(default_factory=dict)
    image_like_pdf_counts: dict[str, int] = field(default_factory=dict)
    ocr_fallback_eligible_counts: dict[str, int] = field(default_factory=dict)
    primary_signals: list[str] = field(default_factory=list)
    block_justified: bool = False
    block_justification: str = ""


@dataclass
class SectionBReport:
    """Shape-audit for fallback_ran_but_no_family_match (17)."""
    name: str = "fallback_ran_but_no_family_match"
    count: int = 0
    shape_audit_counts: dict[str, int] = field(default_factory=dict)
    raw_shape_counts: dict[str, int] = field(default_factory=dict)
    block_justified: bool = False
    block_justification: str = ""


@dataclass
class SectionCReport:
    """Summary-only handling for ambiguous_below_threshold (15)."""
    name: str = "ambiguous_below_threshold"
    count: int = 0
    note: str = ""
    cue_expansion_recommended: bool = False


@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    upstream_diag_01_label: str
    source_report_label: str
    generated_at: str
    total_unknown_analyzed: int
    bucket_counts: dict[str, int]
    section_a: dict
    section_b: dict
    section_c: dict
    overall_recommendation: str   # "A" | "B" | "C" | mixed
    overall_explanation: str
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


_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_safe_label(value: Any) -> bool:
    return isinstance(value, str) and bool(_LABEL_RE.fullmatch(value))


def _safe_count(records: Iterable[dict], key: str) -> dict[str, int]:
    """Counter that drops any value that is not a safe label string."""
    c: Counter = Counter()
    for r in records:
        v = r.get(key)
        if v is None:
            v = "unknown"
        if not _is_safe_label(str(v)):
            v = "other"
        c[str(v)] += 1
    return dict(sorted(c.items()))


# ── Section A: OCR / text-visibility eligibility ─────────────────────────────

def _map_ocr_visibility_label(record: dict) -> str:
    """Project a row into one of the controlled-vocab OCR/visibility labels."""
    routing = str(record.get("unknown_ocr_routing_bucket") or "").lower()
    visibility = str(record.get("language_visibility_status") or "").lower()
    text_layer = str(record.get("pdf_text_layer_detected") or "").lower()
    image_like = str(record.get("image_like_pdf") or "").lower()
    fallback_eligible = str(record.get("ocr_fallback_eligible") or "").lower()
    fallback_reason = str(record.get("ocr_fallback_not_triggered_reason") or "").lower()

    if routing == "extraction_error":
        return "extraction_error"
    if routing == "image_like_pdf_but_not_routed_to_ocr" or image_like == "yes":
        return "image_like_but_ocr_not_routed"
    if routing == "no_text_layer" or text_layer == "no":
        return "no_text_layer"
    if routing == "text_layer_present_but_too_short":
        return "likely_text_layer_issue"
    if routing == "routing_not_eligible":
        return "non_actionable_leave_manual_review"
    if (
        fallback_eligible == "yes"
        and fallback_reason not in {"fallback_executed", "fallback_triggered"}
        and routing not in {"fallback_executed", "fallback_triggered"}
    ):
        return "fallback_eligible_but_not_triggered"
    if visibility in {
        "latin_visible_language_unknown",
        "cyrillic_visible_language_unknown",
        "mixed_visible_language_unknown",
    }:
        return "language_script_visible_detector_unresolved"
    if routing == "language_visibility_unknown":
        return "likely_ocr_eligibility_issue"
    return "non_actionable_leave_manual_review"


def build_section_a(records: list[dict]) -> SectionAReport:
    selected = [r for r in records
                if r.get("unknown_failure_bucket") == "insufficient_text_visibility"]
    visibility_counts: Counter = Counter()
    for r in selected:
        visibility_counts[_map_ocr_visibility_label(r)] += 1

    # Ensure controlled-vocab keys all present, sorted.
    breakdown: dict[str, int] = {label: 0 for label in _OCR_VISIBILITY_LABELS}
    for label, count in visibility_counts.items():
        if label in breakdown:
            breakdown[label] += count
        else:  # defensive
            breakdown["non_actionable_leave_manual_review"] += count

    raw_routing = _safe_count(selected, "unknown_ocr_routing_bucket")
    raw_visibility = _safe_count(selected, "language_visibility_status")
    text_layer_counts = _safe_count(selected, "pdf_text_layer_detected")
    image_like_counts = _safe_count(selected, "image_like_pdf")
    fallback_counts = _safe_count(selected, "ocr_fallback_eligible")

    actionable = (
        breakdown["likely_text_layer_issue"]
        + breakdown["likely_ocr_eligibility_issue"]
        + breakdown["image_like_but_ocr_not_routed"]
        + breakdown["no_text_layer"]
        + breakdown["language_script_visible_detector_unresolved"]
        + breakdown["fallback_eligible_but_not_triggered"]
    )
    block_justified = actionable >= 20

    signals: list[str] = []
    if breakdown["language_script_visible_detector_unresolved"] > 0:
        signals.append("language_script_visible_detector_unresolved")
    if breakdown["likely_text_layer_issue"] > 0:
        signals.append("likely_text_layer_issue")
    if breakdown["image_like_but_ocr_not_routed"] > 0:
        signals.append("image_like_but_ocr_not_routed")
    if breakdown["no_text_layer"] > 0:
        signals.append("no_text_layer")

    if block_justified:
        justification = (
            f"{actionable} of {len(selected)} records show actionable text-"
            f"visibility issues. A future OCR / text-visibility evaluation-"
            f"only block is justified, prioritizing "
            f"language_script_visible_detector_unresolved and "
            f"likely_text_layer_issue. No runtime change required by this "
            f"block."
        )
    else:
        justification = (
            f"Only {actionable} of {len(selected)} records are individually "
            f"actionable for OCR / text-visibility work; recommend leaving "
            f"the bucket review-bound."
        )

    return SectionAReport(
        count=len(selected),
        ocr_visibility_breakdown=breakdown,
        raw_routing_bucket_counts=raw_routing,
        raw_visibility_status_counts=raw_visibility,
        pdf_text_layer_detected_counts=text_layer_counts,
        image_like_pdf_counts=image_like_counts,
        ocr_fallback_eligible_counts=fallback_counts,
        primary_signals=signals,
        block_justified=block_justified,
        block_justification=justification,
    )


# ── Section B: shape audit on fallback_ran_but_no_family_match ───────────────

def build_section_b(records: list[dict]) -> SectionBReport:
    selected = [r for r in records
                if r.get("unknown_failure_bucket") == "fallback_ran_but_no_family_match"]

    shape_counts: Counter = Counter()
    for r in selected:
        verdict = str(r.get("cue_audit_result") or "needs_manual_review")
        if not _is_safe_label(verdict):
            verdict = "other"
        shape_counts[verdict] += 1

    breakdown: dict[str, int] = {v: 0 for v in _SHAPE_AUDIT_VERDICTS}
    other = 0
    for verdict, count in shape_counts.items():
        if verdict in breakdown:
            breakdown[verdict] = count
        else:
            other += count
    if other:
        breakdown["needs_manual_review"] += other

    raw_shape_counts = {
        "table_like_structure_detected_counts": _safe_count(
            selected, "table_like_structure_detected"
        ),
        "imaging_modality_shape_detected_counts": _safe_count(
            selected, "imaging_modality_shape_detected"
        ),
        "lab_table_shape_detected_counts": _safe_count(
            selected, "lab_table_shape_detected"
        ),
        "date_or_schedule_shape_detected_counts": _safe_count(
            selected, "date_or_schedule_shape_detected"
        ),
        "administrative_form_shape_detected_counts": _safe_count(
            selected, "administrative_form_shape_detected"
        ),
        "section_heading_shape_detected_counts": _safe_count(
            selected, "section_heading_shape_detected"
        ),
        "medical_abbreviation_shape_detected_counts": _safe_count(
            selected, "medical_abbreviation_shape_detected"
        ),
    }

    actionable_shape_total = sum(
        breakdown[k] for k in (
            "possible_lab_shape_without_language_cues",
            "possible_imaging_shape_without_language_cues",
            "possible_treatment_shape_without_language_cues",
        )
    )
    block_justified = actionable_shape_total >= 3

    if block_justified:
        justification = (
            f"{actionable_shape_total} of {len(selected)} fallback-bucket records "
            f"surface a recognizable medical shape without language cues. A "
            f"narrow cue-audit / cue-coverage review block (evaluation-only, "
            f"no cue expansion) is justified. The largest sub-category should "
            f"steer the audit focus."
        )
    else:
        justification = (
            f"Only {actionable_shape_total} of {len(selected)} fallback-bucket "
            f"records surface a recognizable medical shape; recommend leaving "
            f"this bucket review-bound."
        )

    return SectionBReport(
        count=len(selected),
        shape_audit_counts=breakdown,
        raw_shape_counts=raw_shape_counts,
        block_justified=block_justified,
        block_justification=justification,
    )


# ── Section C: ambiguous bucket (summary only) ───────────────────────────────

def build_section_c(records: list[dict]) -> SectionCReport:
    selected = [r for r in records
                if r.get("unknown_failure_bucket") == "ambiguous_below_threshold"]
    return SectionCReport(
        count=len(selected),
        note=(
            "Higher false-positive risk. Reported as a summary count only per "
            "UNKNOWN-DIAG-02 scope. All records remain review-bound; cue "
            "expansion must not be driven from this bucket."
        ),
        cue_expansion_recommended=False,
    )


# ── Top-level builder ────────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    # Only look at Unknown rows; the failure-bucket field is only meaningful there.
    unknown_rows = [
        r for r in table
        if (r.get("predicted_document_type") or r.get("unknown_failure_bucket"))
        and str(r.get("predicted_document_type") or "").strip() == "Unknown"
    ]

    section_a = build_section_a(unknown_rows)
    section_b = build_section_b(unknown_rows)
    section_c = build_section_c(unknown_rows)

    bucket_counts = {
        "insufficient_text_visibility": section_a.count,
        "fallback_ran_but_no_family_match": section_b.count,
        "ambiguous_below_threshold": section_c.count,
    }
    total = sum(bucket_counts.values())

    if section_a.block_justified and section_b.block_justified:
        overall = "A_then_B"
        explanation = (
            "Both an OCR / text-visibility evaluation-only block (Section A) "
            "and a narrow cue-coverage audit block (Section B) are justified. "
            "Recommend sequencing Section A first (larger pool, upstream "
            "cause) and then Section B (smaller pool, downstream)."
        )
    elif section_a.block_justified:
        overall = "A"
        explanation = (
            "Recommend an OCR / text-visibility evaluation-only block as the "
            "next action. Section B does not currently meet the threshold "
            "for a follow-up cue-coverage audit."
        )
    elif section_b.block_justified:
        overall = "B"
        explanation = (
            "Recommend a narrow cue-coverage audit block (evaluation-only). "
            "Section A does not currently meet the threshold for an OCR / "
            "text-visibility block."
        )
    else:
        overall = "C"
        explanation = (
            "Stop. Neither section meets the threshold for a follow-up "
            "evaluation block. Leave all Unknown records under manual review."
        )

    safety_privacy = {
        "behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_changed": False,
        "scoring_changed": False,
        "auto_accept_changed": False,
        "cue_packs_changed": False,
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
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-02",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        upstream_diag_01_label=UPSTREAM_DIAG_01_LABEL,
        source_report_label=SOURCE_REPORT_LABEL,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_unknown_analyzed=total,
        bucket_counts=bucket_counts,
        section_a=asdict(section_a),
        section_b=asdict(section_b),
        section_c=asdict(section_c),
        overall_recommendation=overall,
        overall_explanation=explanation,
        behavior_changed=False,
        external_api_used=False,
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-02 - OCR/Text Visibility & Fallback Shape Audit")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): `{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append(f"- upstream diagnostic: `{report.upstream_diag_01_label}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- total Unknown analyzed: `{report.total_unknown_analyzed}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## Bucket counts")
    lines.append("")
    for k, v in report.bucket_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Section A - OCR / text-visibility eligibility")
    lines.append("")
    sa = report.section_a
    lines.append(f"- count: `{sa['count']}`")
    lines.append("- ocr_visibility_breakdown:")
    for k, v in sa["ocr_visibility_breakdown"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append(f"- block_justified: `{sa['block_justified']}`")
    lines.append(f"- justification: {sa['block_justification']}")
    lines.append("")
    lines.append("## Section B - fallback shape audit")
    lines.append("")
    sb = report.section_b
    lines.append(f"- count: `{sb['count']}`")
    lines.append("- shape_audit_counts:")
    for k, v in sb["shape_audit_counts"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.append(f"- block_justified: `{sb['block_justified']}`")
    lines.append(f"- justification: {sb['block_justification']}")
    lines.append("")
    lines.append("## Section C - ambiguous_below_threshold (summary only)")
    lines.append("")
    sc = report.section_c
    lines.append(f"- count: `{sc['count']}`")
    lines.append(f"- cue_expansion_recommended: `{sc['cue_expansion_recommended']}`")
    lines.append(f"- note: {sc['note']}")
    lines.append("")
    lines.append("## Overall recommendation")
    lines.append("")
    lines.append(f"- recommendation: `{report.overall_recommendation}`")
    lines.append("")
    lines.append(report.overall_explanation)
    lines.append("")
    lines.append("## Safety / Privacy")
    lines.append("")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Diagnostic-only, evaluation/reporting "
                 "changes only.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra = [
        "",
        "## Data source",
        "",
        "This block derives all counts from the privacy-safe anonymized per-file "
        "table inside the existing FAMILY-04 batch-evaluation public report. The "
        "table carries:",
        "",
        "- anonymized `file_id` placeholders only (e.g. `file_001`, `file_002`)",
        "- the `unknown_failure_bucket` label per Unknown row",
        "- the EVAL-05 shape-audit verdict (`cue_audit_result`) per row",
        "- the OCR-routing labels (`unknown_ocr_routing_bucket`, "
        "`language_visibility_status`, `pdf_text_layer_detected`, "
        "`image_like_pdf`, `ocr_fallback_eligible`)",
        "",
        "No source documents were opened, no raw text was inspected, no external "
        "API was invoked. UNKNOWN-DIAG-02 is purely an aggregation of the "
        "already-published public-report fields.",
        "",
        "## Why Section A is computed per failure bucket (not the global view)",
        "",
        "UNKNOWN-DIAG-01 used the global `unknown_ocr_routing_diagnostics` "
        "fallback-false bucket counts, which span all Unknown records "
        "regardless of failure bucket. UNKNOWN-DIAG-02 filters the per-file "
        "table to the `insufficient_text_visibility` bucket first, so the "
        "ocr-visibility breakdown reflects exactly the 75 records the bucket "
        "represents. This avoids the overlap that made the sub-bucket sums "
        "exceed the parent count in UNKNOWN-DIAG-01.",
        "",
        "## Why Section B does not propose cue expansion",
        "",
        "The shape-audit verdict identifies records whose shape looks like a "
        "known family without matching language cues. A follow-up block could "
        "audit cue coverage, but cue addition itself carries false-positive "
        "risk. This block produces evidence and a recommendation only.",
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
        "json": out_dir / "medai_doc_type_unknown_diag_02_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_02_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_02.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-02 OCR-visibility / shape-audit diagnostic."
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
            "conclusion": "medai_doc_type_unknown_diag_02_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_unknown_analyzed": report.total_unknown_analyzed,
            "bucket_counts": report.bucket_counts,
            "overall_recommendation": report.overall_recommendation,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
