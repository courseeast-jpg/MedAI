"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-05 - Table-Heavy Language Detection Lever Audit.

Evaluation-only, read-only block.

Scope
-----
Targets the 12 records that UNKNOWN-DIAG-04 routed to the proposed future
lever ``table_heavy_language_detection_policy_audit``. The block:

    1. Loads the privacy-safe ``anonymous_per_file_table`` from the
       FAMILY-04 batch-eval public report.
    2. Re-applies the DIAG-02 -> DIAG-03 -> DIAG-04 filters to isolate the
       12 priority records (no corpus rerun, no source documents opened).
    3. Projects each record to four parallel controlled-vocabulary
       evidence views (table-structure, numeric-distribution, alphabetic/
       script, section/shape).
    4. Assigns each record to exactly one ``candidate_*`` future-policy
       bucket via a priority order.
    5. Recommends at most one A/B/C/D/E follow-up evaluation block.

Hard boundaries
---------------
* No OCR routing or engine changes.
* No language-detector behavior changes.
* No classifier behavior changes.
* No thresholds, scoring, auto-accept, cue-pack, or cue-expansion changes.
* No clinical interpretation, no lab/medication/dose/DDI parsing.
* No B07 / ROUTE-FIX / DB schema / command allowlist / external-API changes.
* No raw filenames, raw OCR text, raw document text, private paths, PHI, or
  secrets. Anonymized aggregate output only.

This block recommends a follow-up diagnostic only; it never implements a
policy is implemented here; this block recommends only.
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

# Ensure the repository root is on sys.path so ``from scripts...`` resolves
# whether the file is invoked as ``python scripts/run_medai_doc_type_unknown_diag_05.py``
# directly or imported as ``scripts.run_medai_doc_type_unknown_diag_05`` from
# pytest. ``conftest.py`` does the same trick at test time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the DIAG-04 selection helpers so we never drift from the upstream
# filter chain. Importing from the sibling script keeps the priority logic
# defined in exactly one place.
from scripts.run_medai_doc_type_unknown_diag_04 import (
    SOURCE_REPORT,
    _select_priority_records,
    evidence_flags_for_latin_lang,
    evidence_flags_for_table_heavy,
    next_lever_for_latin_lang,
    next_lever_for_table_heavy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_05"

PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_DIAG_LABELS = (
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_03/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)",
)

_DIAG04_TARGET_LEVER = "table_heavy_language_detection_policy_audit"


# ── Controlled vocabularies (task-mandated) ──────────────────────────────────

_TABLE_STRUCTURE_VOCAB = (
    "high_table_density",
    "medium_table_density",
    "low_table_density",
    "table_headers_visible",
    "repeated_row_pattern_visible",
    "insufficient_table_metadata",
)

_NUMERIC_DISTRIBUTION_VOCAB = (
    "high_numeric_ratio",
    "medium_numeric_ratio",
    "low_numeric_ratio",
    "numeric_units_or_ranges_visible",
    "sparse_alpha_dense_numeric",
    "insufficient_numeric_metadata",
)

_ALPHA_SCRIPT_VOCAB = (
    "latin_script_high_confidence",
    "latin_script_medium_confidence",
    "alphabetic_ratio_sufficient_for_language",
    "alphabetic_ratio_too_low_for_language",
    "dominant_script_confidence_missing",
    "insufficient_script_metadata",
)

_SECTION_SHAPE_VOCAB = (
    "lab_or_result_section_shape",
    "administrative_table_shape",
    "treatment_schedule_table_shape",
    "generic_table_shape",
    "no_section_shape_available",
)

_POLICY_CANDIDATE_VOCAB = (
    "candidate_table_heavy_latin_policy",
    "candidate_table_header_language_policy",
    "candidate_numeric_table_safe_default_policy",
    "candidate_metadata_propagation_audit",
    "leave_manual_review",
    "insufficient_metadata_for_next_action",
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

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
    total_table_heavy_policy_records: int
    target_lever_label: str
    table_structure_evidence_counts: dict[str, int]
    numeric_distribution_evidence_counts: dict[str, int]
    alphabetic_script_evidence_counts: dict[str, int]
    section_shape_evidence_counts: dict[str, int]
    future_policy_candidate_counts: dict[str, int]
    sample_ids: list[str]
    raw_signal_counts: dict[str, dict[str, int]]
    deferred_subsets: dict[str, str]
    implementation_block_justified: bool
    implementation_block_choice: str
    implementation_block_explanation: str
    behavior_changed: bool
    external_api_used: bool
    cue_expansion_recommended: bool
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
    c: Counter = Counter()
    for r in records:
        v = r.get(key)
        v = "unknown" if v is None else str(v)
        if not _is_safe_label(v):
            v = "other"
        c[v] += 1
    return dict(sorted(c.items()))


def _anonymized_ids(prefix: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


def _empty(vocab: Iterable[str]) -> dict[str, int]:
    return {label: 0 for label in vocab}


def _g(r: dict, key: str) -> str:
    return str(r.get(key) or "").lower()


# ── Priority record selection (re-uses DIAG-04 chain) ────────────────────────

def select_target_records(table: list[dict]) -> list[dict]:
    """Return the records DIAG-04 assigned to ``table_heavy_language_detection_policy_audit``."""
    th_records, latin_records = _select_priority_records(table)
    target: list[dict] = []
    for r in th_records:
        flags = evidence_flags_for_table_heavy(r)
        if next_lever_for_table_heavy(r, flags) == _DIAG04_TARGET_LEVER:
            target.append(r)
    for r in latin_records:
        flags = evidence_flags_for_latin_lang(r)
        if next_lever_for_latin_lang(r, flags) == _DIAG04_TARGET_LEVER:
            target.append(r)
    return target


# ── Evidence flag projection (per record, multi-label per view) ──────────────

def table_structure_flags(record: dict) -> list[str]:
    table_like = _g(record, "table_like_structure_detected")
    section_heading = _g(record, "section_heading_shape_detected")
    date_or_schedule = _g(record, "date_or_schedule_shape_detected")
    flags: list[str] = []
    has_any_metadata = bool(table_like or section_heading)

    if not has_any_metadata:
        flags.append("insufficient_table_metadata")
        return flags

    if table_like == "yes":
        flags.append("high_table_density")
    if section_heading == "yes":
        flags.append("table_headers_visible")
    if table_like == "yes" and date_or_schedule == "yes":
        flags.append("repeated_row_pattern_visible")
    if not flags:
        flags.append("insufficient_table_metadata")
    return flags


def numeric_distribution_flags(record: dict) -> list[str]:
    numeric = _g(record, "numeric_content_bucket")
    alphabetic = _g(record, "alphabetic_content_bucket")
    symbol_content = _g(record, "symbol_content_bucket")
    if not numeric:
        return ["insufficient_numeric_metadata"]
    flags: list[str] = []
    if numeric == "high":
        flags.append("high_numeric_ratio")
    elif numeric == "medium":
        flags.append("medium_numeric_ratio")
    elif numeric == "low":
        flags.append("low_numeric_ratio")
    else:
        flags.append("insufficient_numeric_metadata")
    if symbol_content in {"medium", "high"} and numeric in {"medium", "high"}:
        flags.append("numeric_units_or_ranges_visible")
    if alphabetic == "low" and numeric in {"medium", "high"}:
        flags.append("sparse_alpha_dense_numeric")
    return flags


def alphabetic_script_flags(record: dict) -> list[str]:
    dominant = _g(record, "dominant_script")
    confidence = _g(record, "detector_confidence_bucket")
    alphabetic = _g(record, "alphabetic_content_bucket")
    flags: list[str] = []
    if not dominant and not alphabetic and not confidence:
        return ["insufficient_script_metadata"]

    if dominant == "latin" and confidence == "high":
        flags.append("latin_script_high_confidence")
    elif dominant == "latin" and confidence == "medium":
        flags.append("latin_script_medium_confidence")
    if alphabetic == "high":
        flags.append("alphabetic_ratio_sufficient_for_language")
    elif alphabetic == "low":
        flags.append("alphabetic_ratio_too_low_for_language")
    if not confidence:
        flags.append("dominant_script_confidence_missing")
    if not flags:
        flags.append("insufficient_script_metadata")
    return flags


def section_shape_flags(record: dict) -> list[str]:
    lab_table = _g(record, "lab_table_shape_detected")
    administrative = _g(record, "administrative_form_shape_detected")
    schedule = _g(record, "date_or_schedule_shape_detected")
    table_like = _g(record, "table_like_structure_detected")
    flags: list[str] = []
    if lab_table == "yes":
        flags.append("lab_or_result_section_shape")
    if administrative == "yes":
        flags.append("administrative_table_shape")
    if schedule == "yes":
        flags.append("treatment_schedule_table_shape")
    if table_like == "yes" and not flags:
        flags.append("generic_table_shape")
    if not flags:
        flags.append("no_section_shape_available")
    return flags


# ── Future-policy candidate assignment (single bucket per record) ────────────

def assign_future_policy_candidate(record: dict) -> str:
    """Return exactly one of ``_POLICY_CANDIDATE_VOCAB``."""
    table_flags = set(table_structure_flags(record))
    numeric_flags = set(numeric_distribution_flags(record))
    alpha_flags = set(alphabetic_script_flags(record))

    all_views_insufficient = (
        "insufficient_table_metadata" in table_flags
        and "insufficient_numeric_metadata" in numeric_flags
        and "insufficient_script_metadata" in alpha_flags
    )
    if all_views_insufficient:
        return "insufficient_metadata_for_next_action"

    if "table_headers_visible" in table_flags:
        return "candidate_table_header_language_policy"

    numeric_table_signal = (
        ({"medium_numeric_ratio", "high_numeric_ratio"} & numeric_flags)
        and "high_table_density" in table_flags
        and "table_headers_visible" not in table_flags
    )
    if numeric_table_signal:
        return "candidate_numeric_table_safe_default_policy"

    latin_table_signal = (
        "high_table_density" in table_flags
        and {"latin_script_high_confidence", "latin_script_medium_confidence"}
            & alpha_flags
        and "alphabetic_ratio_sufficient_for_language" in alpha_flags
    )
    if latin_table_signal:
        return "candidate_table_heavy_latin_policy"

    propagation_signal = (
        _g(record, "detector_confidence_bucket") in {"medium", "high"}
        and _g(record, "language_detector_input_bucket") == "sufficient"
        and _g(record, "language_detector_attempted") == "yes"
        and "latin_visible" in _g(record, "language_visibility_status")
    )
    if propagation_signal:
        return "candidate_metadata_propagation_audit"

    return "leave_manual_review"


# ── Aggregation ─────────────────────────────────────────────────────────────

def _aggregate_flags(records: list[dict], flagger, vocab) -> dict[str, int]:
    counts = _empty(vocab)
    for r in records:
        for f in flagger(r):
            if f in counts:
                counts[f] += 1
    return counts


def _aggregate_candidates(records: list[dict]) -> dict[str, int]:
    counts = _empty(_POLICY_CANDIDATE_VOCAB)
    for r in records:
        c = assign_future_policy_candidate(r)
        if c not in counts:
            c = "leave_manual_review"
        counts[c] += 1
    return counts


# ── Top-level orchestration ──────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    target_records = select_target_records(table)

    table_struct = _aggregate_flags(
        target_records, table_structure_flags, _TABLE_STRUCTURE_VOCAB,
    )
    numeric_dist = _aggregate_flags(
        target_records, numeric_distribution_flags, _NUMERIC_DISTRIBUTION_VOCAB,
    )
    alpha_script = _aggregate_flags(
        target_records, alphabetic_script_flags, _ALPHA_SCRIPT_VOCAB,
    )
    section_shape = _aggregate_flags(
        target_records, section_shape_flags, _SECTION_SHAPE_VOCAB,
    )
    candidates = _aggregate_candidates(target_records)

    raw_signal_counts = {
        "table_like_structure_detected_counts":
            _safe_count(target_records, "table_like_structure_detected"),
        "numeric_content_bucket_counts":
            _safe_count(target_records, "numeric_content_bucket"),
        "alphabetic_content_bucket_counts":
            _safe_count(target_records, "alphabetic_content_bucket"),
        "section_heading_shape_detected_counts":
            _safe_count(target_records, "section_heading_shape_detected"),
        "administrative_form_shape_detected_counts":
            _safe_count(target_records, "administrative_form_shape_detected"),
        "date_or_schedule_shape_detected_counts":
            _safe_count(target_records, "date_or_schedule_shape_detected"),
        "lab_table_shape_detected_counts":
            _safe_count(target_records, "lab_table_shape_detected"),
        "imaging_modality_shape_detected_counts":
            _safe_count(target_records, "imaging_modality_shape_detected"),
        "medical_abbreviation_shape_detected_counts":
            _safe_count(target_records, "medical_abbreviation_shape_detected"),
        "dominant_script_counts":
            _safe_count(target_records, "dominant_script"),
        "detector_confidence_bucket_counts":
            _safe_count(target_records, "detector_confidence_bucket"),
        "language_detector_input_bucket_counts":
            _safe_count(target_records, "language_detector_input_bucket"),
        "language_visibility_status_counts":
            _safe_count(target_records, "language_visibility_status"),
        "symbol_content_bucket_counts":
            _safe_count(target_records, "symbol_content_bucket"),
        "page_count_bucket_counts":
            _safe_count(target_records, "page_count_bucket"),
        "size_bucket_counts":
            _safe_count(target_records, "size_bucket"),
    }

    THRESH = 5
    pri_ordered = [
        ("A", candidates["candidate_table_heavy_latin_policy"],
         "candidate_table_heavy_latin_policy"),
        ("B", candidates["candidate_table_header_language_policy"],
         "candidate_table_header_language_policy"),
        ("C", candidates["candidate_numeric_table_safe_default_policy"],
         "candidate_numeric_table_safe_default_policy"),
        ("D", candidates["candidate_metadata_propagation_audit"],
         "candidate_metadata_propagation_audit"),
    ]
    pri_ordered.sort(key=lambda x: x[1], reverse=True)
    best_choice, best_count, best_name = pri_ordered[0]

    if best_count >= THRESH:
        justified = True
        choice = best_choice
        runners = [(c, n, lbl) for c, n, lbl in pri_ordered[1:] if n >= THRESH]
        if runners:
            extras = "; ".join(f"{c}=`{lbl}` ({n})" for c, n, lbl in runners)
            explanation = (
                f"`{best_name}` is the largest pool at {best_count} records. "
                f"Other levers above the threshold: {extras}. Recommend "
                f"{best_choice} first as the most-supported lever."
            )
        else:
            explanation = (
                f"`{best_name}` is the only lever above the threshold "
                f"({best_count} records). Recommend {best_choice}."
            )
    else:
        justified = False
        choice = "E"
        explanation = (
            f"No diagnostic lever clears the threshold (largest is "
            f"`{best_name}` at {best_count}). Recommend E (leave manual "
            f"review). All priority records remain review-bound."
        )

    deferred = {
        "language_detector_metadata_propagation_audit_pool":
            "11 records routed by DIAG-04 to this lever; deferred behind the "
            "table-heavy lever audit per the recommendation order",
        "latin_medical_abbreviation_handling_audit_pool":
            "8 records routed by DIAG-04 to this lever; deferred behind the "
            "table-heavy lever audit",
        "likely_text_layer_issue":
            "21 records deferred to the option-B follow-up per UNKNOWN-DIAG-03",
        "fallback_ran_but_no_family_match":
            "17 records deferred to a future cue-coverage audit per "
            "UNKNOWN-DIAG-02; no cue expansion",
        "ambiguous_below_threshold":
            "15 records remain excluded from optimization; review-bound, "
            "no cue expansion",
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
        "metadata_propagation_audit_pool_deferred": True,
        "latin_medical_abbreviation_pool_deferred": True,
        "likely_text_layer_issue_deferred": True,
        "fallback_ran_but_no_family_match_deferred": True,
        "ambiguous_below_threshold_excluded": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-05",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_diag_labels=list(UPSTREAM_DIAG_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_table_heavy_policy_records=len(target_records),
        target_lever_label=_DIAG04_TARGET_LEVER,
        table_structure_evidence_counts=table_struct,
        numeric_distribution_evidence_counts=numeric_dist,
        alphabetic_script_evidence_counts=alpha_script,
        section_shape_evidence_counts=section_shape,
        future_policy_candidate_counts=candidates,
        sample_ids=_anonymized_ids("table_heavy_policy_priority",
                                   len(target_records)),
        raw_signal_counts=raw_signal_counts,
        deferred_subsets=deferred,
        implementation_block_justified=justified,
        implementation_block_choice=choice,
        implementation_block_explanation=explanation,
        behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-05 - Table-Heavy Language Detection Audit")
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
    lines.append(f"- target lever: `{report.target_lever_label}`")
    lines.append(f"- total table-heavy lever records analyzed: "
                 f"`{report.total_table_heavy_policy_records}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## Table-structure evidence")
    lines.append("")
    for k, v in report.table_structure_evidence_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Numeric-distribution evidence")
    lines.append("")
    for k, v in report.numeric_distribution_evidence_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Alphabetic / script evidence")
    lines.append("")
    for k, v in report.alphabetic_script_evidence_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Section / shape evidence")
    lines.append("")
    for k, v in report.section_shape_evidence_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Future-lever candidate counts (per-record single assignment)")
    lines.append("")
    for k, v in report.future_policy_candidate_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Implementation-block recommendation")
    lines.append("")
    lines.append(
        f"- implementation_block_justified: "
        f"`{report.implementation_block_justified}`"
    )
    lines.append(f"- choice: `{report.implementation_block_choice}` "
                 f"(A=`candidate_table_heavy_latin_policy` prototype, "
                 f"B=`candidate_table_header_language_policy` prototype, "
                 f"C=`candidate_numeric_table_safe_default_policy` prototype, "
                 f"D=`candidate_metadata_propagation_audit` instead, "
                 f"E=`leave_manual_review`)")
    lines.append("")
    lines.append(report.implementation_block_explanation)
    lines.append("")
    lines.append("## Deferred subsets (out of scope)")
    lines.append("")
    for k, v in report.deferred_subsets.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Safety / Privacy")
    lines.append("")
    lines.append(f"- behavior_changed: `{report.behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Diagnostic-only, evaluation/reporting "
                 "changes only. This block recommends a follow-up diagnostic only; "
                 "no policy is implemented here. Cue expansion is not recommended. "
                 "No OCR routing or detector behavior change is proposed.")
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
        "## Why a multi-view evidence projection",
        "",
        "Each priority record is projected to four parallel controlled-",
        "vocabulary views (table-structure, numeric-distribution, alphabetic/",
        "script, section/shape). A record may light up several flags inside ",
        "each view; the counts here are independent per-flag totals. The ",
        "per-record future-lever candidate is a single-bucket priority-",
        "ordered assignment that names the most operationally distinct ",
        "follow-up lever for that record.",
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
        "Even if the recommendation indicates a future lever prototype, this ",
        "block must only recommend it. No policy is implemented here.",
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
        "json": out_dir / "medai_doc_type_unknown_diag_05_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_05_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_05.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-05 table-heavy language lever audit."
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
            "conclusion": "medai_doc_type_unknown_diag_05_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_table_heavy_policy_records": report.total_table_heavy_policy_records,
            "future_policy_candidate_counts": report.future_policy_candidate_counts,
            "implementation_block_choice": report.implementation_block_choice,
            "implementation_block_justified": report.implementation_block_justified,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
            "cue_expansion_recommended": report.cue_expansion_recommended,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
