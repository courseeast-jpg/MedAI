"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-04 - Language/Script Detector Diagnostic.

Evaluation-only, read-only block.

Scope
-----
Targets the 31 priority records routed by UNKNOWN-DIAG-03 to
``candidate_language_detector_diagnostic``. The split inside that pool is:

    * 16 ``numeric_or_table_heavy_language_detector_gap``
    * 15 ``latin_visible_language_unknown``

For each priority record, this script projects the privacy-safe metadata in
the FAMILY-04 batch-eval per-file table to two controlled-vocabulary views:

    * evidence buckets - per-signal flags. A single record may light up
      several evidence buckets at once (e.g. table-heavy + medical-abbrev).
      The buckets enumerate WHY the detector did not resolve language.
    * proposed future diagnostic levers - per-record, single-bucket
      assignment naming the operational diagnostic to consider next.

Hard boundaries
---------------
* No OCR routing / OCR engine changes.
* No language-detector or classifier behavior changes.
* No thresholds, scoring, auto-accept, cue-pack, or cue-expansion changes.
* No clinical interpretation, no lab/medication/dose/DDI parsing.
* No B07 / ROUTE-FIX / DB schema / command allowlist / external API changes.
* No raw filenames, raw OCR text, raw document text, private paths, PHI, or
  secrets emitted. Anonymized aggregate output only.

Data source
-----------
Only the privacy-safe ``anonymous_per_file_table`` of the FAMILY-04 batch-
eval public report (already anonymized to ``file_NNN`` IDs). No corpus
rerun, no source documents opened, no external API.
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
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_04"

PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_DIAG_LABELS = (
    "reports/medai_doc_type_unknown_diag_03/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)",
)


# ── Controlled vocabularies (task-mandated) ──────────────────────────────────

_EVIDENCE_TABLE_HEAVY = (
    "table_heavy_latin_visible",
    "numeric_heavy_latin_visible",
    "lab_table_shape_latin_visible",
    "sparse_words_many_numbers",
    "detector_input_too_structural",
    "insufficient_safe_metadata",
)

_EVIDENCE_LATIN_LANG_UNKNOWN = (
    "latin_words_visible_detector_unknown",
    "latin_medical_abbrev_visible",
    "latin_table_headers_visible",
    "latin_script_detected_language_missing",
    "detector_output_not_propagated",
    "insufficient_safe_metadata",
)

_FUTURE_LEVERS = (
    "language_detector_metadata_propagation_audit",
    "table_heavy_language_detection_policy_audit",
    "latin_medical_abbreviation_handling_audit",
    "leave_manual_review",
    "insufficient_metadata_for_next_action",
)

# DIAG-03 root-cause labels that drive each priority sub-pool.
_RC_TABLE_HEAVY = "numeric_or_table_heavy_language_detector_gap"
_RC_LATIN       = "latin_visible_language_unknown"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SubPoolReport:
    name: str
    diag03_root_cause: str
    count: int
    evidence_counts: dict[str, int] = field(default_factory=dict)
    next_action_counts: dict[str, int] = field(default_factory=dict)
    sample_ids: list[str] = field(default_factory=list)
    raw_signal_counts: dict[str, dict[str, int]] = field(default_factory=dict)


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
    total_language_detector_records: int
    sub_pool_counts: dict[str, int]
    evidence_bucket_counts: dict[str, dict[str, int]]
    future_lever_counts: dict[str, int]
    deferred_subsets: dict[str, str]
    implementation_block_justified: bool
    implementation_block_choice: str
    implementation_block_explanation: str
    sub_pools: dict[str, dict]
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


# ── Filtering (re-derived from DIAG-02 / DIAG-03 mappings) ───────────────────

def _diag02_visibility_label(record: dict) -> str:
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


def _diag03_root_cause(record: dict) -> str:
    """Re-derive DIAG-03 root-cause label for a language_script_visible_*
    record. Matches scripts.run_medai_doc_type_unknown_diag_03's
    classify_language_detector_root_cause."""
    visibility = str(record.get("language_visibility_status") or "").lower()
    dominant = str(record.get("dominant_script") or "").lower()
    table_like = str(record.get("table_like_structure_detected") or "").lower()
    numeric = str(record.get("numeric_content_bucket") or "").lower()
    alphabetic = str(record.get("alphabetic_content_bucket") or "").lower()

    if not (visibility or dominant):
        return "insufficient_safe_metadata"
    if table_like == "yes" and (numeric in {"medium", "high"}
                                or alphabetic in {"low", "medium"}):
        return _RC_TABLE_HEAVY
    if dominant == "mixed" or visibility == "mixed_visible_language_unknown":
        return "mixed_script_detector_gap"
    if dominant == "cyrillic" or visibility == "cyrillic_visible_language_unknown":
        return "cyrillic_visible_language_unknown"
    if dominant == "latin" or visibility == "latin_visible_language_unknown":
        return _RC_LATIN
    return "script_visible_language_detector_gap"


def _select_priority_records(table: list[dict]) -> tuple[list[dict], list[dict]]:
    table_heavy: list[dict] = []
    latin_lang: list[dict] = []
    for r in table:
        if r.get("unknown_failure_bucket") != "insufficient_text_visibility":
            continue
        if str(r.get("predicted_document_type") or "") != "Unknown":
            continue
        if _diag02_visibility_label(r) != "language_script_visible_detector_unresolved":
            continue
        rc = _diag03_root_cause(r)
        if rc == _RC_TABLE_HEAVY:
            table_heavy.append(r)
        elif rc == _RC_LATIN:
            latin_lang.append(r)
    return table_heavy, latin_lang


# ── Evidence-bucket flags (per record, multi-label allowed) ──────────────────

def evidence_flags_for_table_heavy(record: dict) -> list[str]:
    """Return all evidence labels that apply to a single table-heavy record."""
    flags: list[str] = []
    table_like = str(record.get("table_like_structure_detected") or "").lower()
    numeric = str(record.get("numeric_content_bucket") or "").lower()
    alphabetic = str(record.get("alphabetic_content_bucket") or "").lower()
    lab_table = str(record.get("lab_table_shape_detected") or "").lower()
    visibility = str(record.get("language_visibility_status") or "").lower()
    detector_unknown_bucket = str(
        record.get("language_script_detector_unknown_bucket") or ""
    ).lower()
    symbol_content = str(record.get("symbol_content_bucket") or "").lower()
    has_metadata = bool(visibility or table_like or numeric)

    if not has_metadata:
        flags.append("insufficient_safe_metadata")
        return flags

    if table_like == "yes" and visibility.startswith("latin_visible"):
        flags.append("table_heavy_latin_visible")
    if numeric in {"medium", "high"} and visibility.startswith("latin_visible"):
        flags.append("numeric_heavy_latin_visible")
    if lab_table == "yes":
        flags.append("lab_table_shape_latin_visible")
    if alphabetic in {"low"} and numeric in {"medium", "high"}:
        flags.append("sparse_words_many_numbers")
    if (
        detector_unknown_bucket
        in {"detector_input_garbled_or_mojibake", "detector_input_symbol_heavy"}
        or symbol_content == "high"
    ):
        flags.append("detector_input_too_structural")
    if not flags:
        flags.append("insufficient_safe_metadata")
    return flags


def evidence_flags_for_latin_lang(record: dict) -> list[str]:
    flags: list[str] = []
    visibility = str(record.get("language_visibility_status") or "").lower()
    table_like = str(record.get("table_like_structure_detected") or "").lower()
    medical_abbrev = str(record.get("medical_abbreviation_shape_detected") or "").lower()
    detector_attempted = str(record.get("language_detector_attempted") or "").lower()
    detector_input = str(record.get("language_detector_input_bucket") or "").lower()
    detector_confidence = str(record.get("detector_confidence_bucket") or "").lower()
    script_result = str(record.get("script_detection_result") or "").lower()
    alphabetic = str(record.get("alphabetic_content_bucket") or "").lower()
    has_metadata = bool(visibility or script_result or alphabetic)

    if not has_metadata:
        flags.append("insufficient_safe_metadata")
        return flags

    if visibility.startswith("latin_visible") and alphabetic == "high":
        flags.append("latin_words_visible_detector_unknown")
    if medical_abbrev == "yes":
        flags.append("latin_medical_abbrev_visible")
    if table_like == "yes" and visibility.startswith("latin_visible"):
        flags.append("latin_table_headers_visible")
    if script_result == "latin" and visibility.startswith("latin_visible"):
        flags.append("latin_script_detected_language_missing")
    if (
        detector_attempted == "yes"
        and detector_input == "sufficient"
        and detector_confidence in {"medium", "high"}
        and visibility.startswith("latin_visible")
    ):
        flags.append("detector_output_not_propagated")
    if not flags:
        flags.append("insufficient_safe_metadata")
    return flags


# ── Per-record next-action assignment (single bucket, priority-ordered) ──────

def next_lever_for_table_heavy(record: dict, flags: list[str]) -> str:
    if "insufficient_safe_metadata" in flags and len(flags) == 1:
        return "insufficient_metadata_for_next_action"
    # A table-heavy record that ALSO carries a medical-abbreviation shape
    # signal is operationally more interesting for the abbreviation handling
    # lever; consult the raw record signal (the flag is only enumerated in
    # the latin_visible_language_unknown evidence vocabulary).
    if str(record.get("medical_abbreviation_shape_detected") or "").lower() == "yes":
        return "latin_medical_abbreviation_handling_audit"
    # Table-heavy detection-policy lever is the most specific operational
    # diagnostic for this sub-pool.
    if ("table_heavy_latin_visible" in flags
            or "numeric_heavy_latin_visible" in flags
            or "lab_table_shape_latin_visible" in flags
            or "sparse_words_many_numbers" in flags
            or "detector_input_too_structural" in flags):
        return "table_heavy_language_detection_policy_audit"
    return "leave_manual_review"


def next_lever_for_latin_lang(record: dict, flags: list[str]) -> str:
    if "insufficient_safe_metadata" in flags and len(flags) == 1:
        return "insufficient_metadata_for_next_action"
    if "latin_medical_abbrev_visible" in flags:
        return "latin_medical_abbreviation_handling_audit"
    if "detector_output_not_propagated" in flags:
        return "language_detector_metadata_propagation_audit"
    if "latin_table_headers_visible" in flags:
        return "table_heavy_language_detection_policy_audit"
    return "leave_manual_review"


# ── Builders ────────────────────────────────────────────────────────────────

def _empty_counts(vocab: Iterable[str]) -> dict[str, int]:
    return {label: 0 for label in vocab}


def _aggregate_evidence(records: list[dict], flagger, vocab) -> dict[str, int]:
    counts = _empty_counts(vocab)
    for r in records:
        for f in flagger(r):
            if f in counts:
                counts[f] += 1
    return counts


def _aggregate_next_actions(records: list[dict], flagger, lever_chooser) -> dict[str, int]:
    counts = _empty_counts(_FUTURE_LEVERS)
    for r in records:
        flags = flagger(r)
        lever = lever_chooser(r, flags)
        if lever in counts:
            counts[lever] += 1
        else:
            counts["leave_manual_review"] += 1
    return counts


def _build_sub_pool(
    name: str, root_cause: str, records: list[dict],
    flagger, lever_chooser, evidence_vocab,
) -> SubPoolReport:
    evidence = _aggregate_evidence(records, flagger, evidence_vocab)
    actions = _aggregate_next_actions(records, flagger, lever_chooser)
    raw = {
        "table_like_structure_detected_counts":
            _safe_count(records, "table_like_structure_detected"),
        "numeric_content_bucket_counts":
            _safe_count(records, "numeric_content_bucket"),
        "alphabetic_content_bucket_counts":
            _safe_count(records, "alphabetic_content_bucket"),
        "lab_table_shape_detected_counts":
            _safe_count(records, "lab_table_shape_detected"),
        "medical_abbreviation_shape_detected_counts":
            _safe_count(records, "medical_abbreviation_shape_detected"),
        "language_detector_attempted_counts":
            _safe_count(records, "language_detector_attempted"),
        "language_detector_input_bucket_counts":
            _safe_count(records, "language_detector_input_bucket"),
        "language_script_detector_unknown_bucket_counts":
            _safe_count(records, "language_script_detector_unknown_bucket"),
        "detector_confidence_bucket_counts":
            _safe_count(records, "detector_confidence_bucket"),
        "script_detection_result_counts":
            _safe_count(records, "script_detection_result"),
        "symbol_content_bucket_counts":
            _safe_count(records, "symbol_content_bucket"),
    }
    return SubPoolReport(
        name=name,
        diag03_root_cause=root_cause,
        count=len(records),
        evidence_counts=evidence,
        next_action_counts=actions,
        sample_ids=_anonymized_ids(name, len(records)),
        raw_signal_counts=raw,
    )


# ── Top-level orchestration ──────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    table_heavy_records, latin_records = _select_priority_records(table)

    pool_a = _build_sub_pool(
        name="numeric_or_table_heavy_language_detector_gap",
        root_cause=_RC_TABLE_HEAVY,
        records=table_heavy_records,
        flagger=evidence_flags_for_table_heavy,
        lever_chooser=next_lever_for_table_heavy,
        evidence_vocab=_EVIDENCE_TABLE_HEAVY,
    )
    pool_b = _build_sub_pool(
        name="latin_visible_language_unknown",
        root_cause=_RC_LATIN,
        records=latin_records,
        flagger=evidence_flags_for_latin_lang,
        lever_chooser=next_lever_for_latin_lang,
        evidence_vocab=_EVIDENCE_LATIN_LANG_UNKNOWN,
    )

    # Aggregate the future-lever counts across both sub-pools.
    future_lever_totals = _empty_counts(_FUTURE_LEVERS)
    for k, v in pool_a.next_action_counts.items():
        future_lever_totals[k] += v
    for k, v in pool_b.next_action_counts.items():
        future_lever_totals[k] += v

    total = pool_a.count + pool_b.count
    sub_pool_counts = {pool_a.name: pool_a.count, pool_b.name: pool_b.count}

    THRESH = 5
    propagation = future_lever_totals["language_detector_metadata_propagation_audit"]
    table_policy = future_lever_totals["table_heavy_language_detection_policy_audit"]
    abbrev = future_lever_totals["latin_medical_abbreviation_handling_audit"]
    leave = future_lever_totals["leave_manual_review"]
    insuff = future_lever_totals["insufficient_metadata_for_next_action"]

    # Pick the single most-supported diagnostic lever above the threshold.
    candidates = [
        ("A", propagation, "language_detector_metadata_propagation_audit"),
        ("B", table_policy, "table_heavy_language_detection_policy_audit"),
        ("C", abbrev, "latin_medical_abbreviation_handling_audit"),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_choice, best_count, best_name = candidates[0]
    second_choice, second_count, second_name = candidates[1]

    if best_count >= THRESH:
        justified = True
        choice = best_choice
        if second_count >= THRESH:
            explanation = (
                f"Both `{best_name}` ({best_count}) and `{second_name}` "
                f"({second_count}) meet the threshold. Recommend {best_choice} "
                f"first (larger pool, more direct lever), then {second_choice}."
            )
        else:
            explanation = (
                f"`{best_name}` is the only lever above the threshold "
                f"({best_count} records). Recommend {best_choice}."
            )
    else:
        justified = False
        choice = "D"
        explanation = (
            f"No diagnostic lever clears the threshold (`{best_name}` has the "
            f"largest pool at {best_count}). Recommend D (leave manual review). "
            f"All priority records remain review-bound."
        )

    deferred = {
        "likely_text_layer_issue":
            "21 records deferred to the option B follow-up per UNKNOWN-DIAG-03",
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
        "likely_text_layer_issue_deferred": True,
        "fallback_ran_but_no_family_match_deferred": True,
        "ambiguous_below_threshold_excluded": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-04",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_diag_labels=list(UPSTREAM_DIAG_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_language_detector_records=total,
        sub_pool_counts=sub_pool_counts,
        evidence_bucket_counts={
            pool_a.name: pool_a.evidence_counts,
            pool_b.name: pool_b.evidence_counts,
        },
        future_lever_counts=future_lever_totals,
        deferred_subsets=deferred,
        implementation_block_justified=justified,
        implementation_block_choice=choice,
        implementation_block_explanation=explanation,
        sub_pools={pool_a.name: asdict(pool_a), pool_b.name: asdict(pool_b)},
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
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-04 - Language/Script Detector Diagnostic")
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
    lines.append(f"- total language-detector records analyzed: "
                 f"`{report.total_language_detector_records}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## Sub-pool counts")
    lines.append("")
    for k, v in report.sub_pool_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Evidence bucket counts")
    lines.append("")
    for pool, counts in report.evidence_bucket_counts.items():
        lines.append(f"### {pool}")
        lines.append("")
        for k, v in counts.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.append("## Proposed future diagnostic lever counts")
    lines.append("")
    for k, v in report.future_lever_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Implementation-block recommendation")
    lines.append("")
    lines.append(
        f"- implementation_block_justified: "
        f"`{report.implementation_block_justified}`"
    )
    lines.append(f"- choice: `{report.implementation_block_choice}` "
                 f"(A=`language_detector_metadata_propagation_audit`, "
                 f"B=`table_heavy_language_detection_policy_audit`, "
                 f"C=`latin_medical_abbreviation_handling_audit`, "
                 f"D=`leave_manual_review`)")
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
    lines.append(
        f"- cue_expansion_recommended: `{report.cue_expansion_recommended}`"
    )
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Diagnostic-only, evaluation/reporting "
                 "changes only. Cue expansion is not recommended for any bucket "
                 "in this block. No OCR routing or detector behavior change is "
                 "proposed in this block.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra = [
        "",
        "## Raw signal distributions (per sub-pool)",
        "",
    ]
    for pool_name, pool in report.sub_pools.items():
        extra.append(f"### {pool_name}")
        extra.append("")
        for k, v in pool["raw_signal_counts"].items():
            inner = ", ".join(f"`{kk}`={vv}" for kk, vv in v.items())
            extra.append(f"- {k}: {inner}")
        extra.append("")
    extra += [
        "## Why an evidence + future-lever projection and not an implementation",
        "",
        "Each priority record is projected to multiple evidence flags using "
        "only privacy-safe detector-side metadata already published in the "
        "FAMILY-04 per-file table (detector attempted, detector input bucket, "
        "detector confidence, script detection result, language script "
        "detector unknown bucket, plus shape signals). Each record is then "
        "assigned to exactly one proposed future diagnostic lever using a "
        "priority order. The output names the operational diagnostic a "
        "follow-up block should consider, not an implementation change. "
        "Runtime behavior, detector behavior, and OCR routing are unchanged.",
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
        "json": out_dir / "medai_doc_type_unknown_diag_04_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_04_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_04.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-04 language/script detector diagnostic."
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
            "conclusion": "medai_doc_type_unknown_diag_04_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_language_detector_records": report.total_language_detector_records,
            "sub_pool_counts": report.sub_pool_counts,
            "future_lever_counts": report.future_lever_counts,
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
