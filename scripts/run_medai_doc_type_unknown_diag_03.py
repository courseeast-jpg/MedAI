"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-03 - Text Visibility Root-Cause Classifier.

Evaluation-only, read-only block.

Scope
-----
Targets the 52 priority records identified by UNKNOWN-DIAG-02 inside the
``insufficient_text_visibility`` bucket:

    * 31 records with the DIAG-02 label ``language_script_visible_detector_unresolved``
    * 21 records with the DIAG-02 label ``likely_text_layer_issue``

For each priority record, the script projects the privacy-safe metadata in
the FAMILY-04 batch-eval per-file table to:

    1. A fine-grained root-cause candidate label (controlled vocabulary).
    2. A coarse next-action bucket (controlled vocabulary) that names the
       operational diagnostic lever, NOT an implementation change.

Hard boundaries
---------------
* No OCR routing / OCR engine changes.
* No language-detector or classifier behavior changes.
* No thresholds, scoring, auto-accept, or cue-pack changes.
* No clinical interpretation, no lab/medication/dose/DDI parsing.
* No B07 / ROUTE-FIX / DB schema / command allowlist / external API changes.
* No raw filenames, raw OCR text, raw document text, private paths, PHI, or
  secrets emitted. Anonymized aggregate output only.

Data source
-----------
The privacy-safe ``anonymous_per_file_table`` of the FAMILY-04 batch-eval
public report carries:
    file_id (anonymized), unknown_failure_bucket, language_visibility_status,
    dominant_script, pdf_text_layer_detected, image_like_pdf,
    ocr_fallback_eligible, native_text_length_bucket, table_like_structure_detected,
    alphabetic_content_bucket, numeric_content_bucket, visibility_unknown_reason,
    unknown_ocr_routing_bucket, and other per-row controlled-vocabulary fields.
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
DIAG_02_REPORT = (
    REPO_ROOT
    / "reports"
    / "medai_doc_type_unknown_diag_02"
    / "medai_doc_type_unknown_diag_02_report.json"
)
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_03"

PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
DIAG_02_REPORT_LABEL = (
    "reports/medai_doc_type_unknown_diag_02/(public diagnostic)"
)
DIAG_01_REPORT_LABEL = (
    "reports/medai_doc_type_unknown_diag_01/(public diagnostic)"
)


# ── Controlled vocabularies (task-mandated) ──────────────────────────────────

_LANG_DETECTOR_LABELS = (
    "script_visible_language_detector_gap",
    "latin_visible_language_unknown",
    "cyrillic_visible_language_unknown",
    "mixed_script_detector_gap",
    "numeric_or_table_heavy_language_detector_gap",
    "insufficient_safe_metadata",
    "leave_manual_review",
)

_TEXT_LAYER_LABELS = (
    "text_layer_too_short",
    "text_layer_present_but_low_signal",
    "table_structure_visible_but_text_insufficient",
    "image_like_with_partial_text",
    "no_safe_text_visibility_metadata",
    "leave_manual_review",
)

_NEXT_ACTION_BUCKETS = (
    "candidate_language_detector_diagnostic",
    "candidate_text_layer_extraction_diagnostic",
    "candidate_ocr_routing_diagnostic",
    "leave_manual_review",
    "insufficient_metadata_for_next_action",
)

# DIAG-02 labels that drive each priority subset.
_DIAG02_LABEL_LANG_DETECTOR = "language_script_visible_detector_unresolved"
_DIAG02_LABEL_TEXT_LAYER    = "likely_text_layer_issue"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SubsetReport:
    name: str
    diag02_label: str
    count: int
    root_cause_counts: dict[str, int] = field(default_factory=dict)
    secondary_signal_counts: dict[str, int] = field(default_factory=dict)
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
    upstream_diag_02_label: str
    upstream_diag_01_label: str
    generated_at: str
    total_priority_analyzed: int
    target_subset_counts: dict[str, int]
    root_cause_counts: dict[str, dict[str, int]]
    next_action_bucket_counts: dict[str, int]
    deferred_subsets: dict[str, str]
    implementation_block_justified: bool
    implementation_block_choice: str
    implementation_block_explanation: str
    subsets: dict[str, dict]
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


# ── DIAG-02 label projection (re-derived locally for stability) ──────────────

def _diag02_visibility_label(record: dict) -> str:
    """Return the DIAG-02 ocr-visibility label for an insufficient-text record.

    This mirrors the DIAG-02 mapping; we re-derive locally so DIAG-03 does
    not depend on the DIAG-02 JSON for per-row decisions.
    """
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


# ── Root-cause classifiers ──────────────────────────────────────────────────

def classify_language_detector_root_cause(record: dict) -> str:
    """Project a language-detector-unresolved record to a root-cause label."""
    visibility = str(record.get("language_visibility_status") or "").lower()
    dominant = str(record.get("dominant_script") or "").lower()
    table_like = str(record.get("table_like_structure_detected") or "").lower()
    numeric = str(record.get("numeric_content_bucket") or "").lower()
    alphabetic = str(record.get("alphabetic_content_bucket") or "").lower()

    has_metadata = bool(visibility or dominant)
    if not has_metadata:
        return "insufficient_safe_metadata"

    # Numeric/table-heavy take priority when present, since they are the
    # most operationally distinct sub-class (detector struggles on tables).
    if table_like == "yes" and (numeric in {"medium", "high"}
                                or alphabetic in {"low", "medium"}):
        return "numeric_or_table_heavy_language_detector_gap"

    if dominant == "mixed" or visibility == "mixed_visible_language_unknown":
        return "mixed_script_detector_gap"
    if dominant == "cyrillic" or visibility == "cyrillic_visible_language_unknown":
        return "cyrillic_visible_language_unknown"
    if dominant == "latin" or visibility == "latin_visible_language_unknown":
        return "latin_visible_language_unknown"

    # Script-visible but detector tagged it as a generic gap.
    return "script_visible_language_detector_gap"


def classify_text_layer_root_cause(record: dict) -> str:
    """Project a likely_text_layer_issue record to a root-cause label."""
    length = str(record.get("native_text_length_bucket") or "").lower()
    text_layer = str(record.get("pdf_text_layer_detected") or "").lower()
    image_like = str(record.get("image_like_pdf") or "").lower()
    table_like = str(record.get("table_like_structure_detected") or "").lower()
    alphabetic = str(record.get("alphabetic_content_bucket") or "").lower()

    if not text_layer and not length:
        return "no_safe_text_visibility_metadata"

    if image_like == "yes":
        return "image_like_with_partial_text"

    if length in {"none", "tiny"}:
        if table_like == "yes":
            return "table_structure_visible_but_text_insufficient"
        return "text_layer_too_short"

    if length == "short":
        if table_like == "yes":
            return "table_structure_visible_but_text_insufficient"
        return "text_layer_too_short"

    if alphabetic in {"low", "medium"} and length not in {"long"}:
        return "text_layer_present_but_low_signal"

    return "leave_manual_review"


def next_action_for_language_root_cause(label: str) -> str:
    if label in {
        "script_visible_language_detector_gap",
        "latin_visible_language_unknown",
        "cyrillic_visible_language_unknown",
        "mixed_script_detector_gap",
        "numeric_or_table_heavy_language_detector_gap",
    }:
        return "candidate_language_detector_diagnostic"
    if label == "insufficient_safe_metadata":
        return "insufficient_metadata_for_next_action"
    return "leave_manual_review"


def next_action_for_text_layer_root_cause(label: str) -> str:
    if label in {
        "text_layer_too_short",
        "text_layer_present_but_low_signal",
        "table_structure_visible_but_text_insufficient",
    }:
        return "candidate_text_layer_extraction_diagnostic"
    if label == "image_like_with_partial_text":
        return "candidate_ocr_routing_diagnostic"
    if label == "no_safe_text_visibility_metadata":
        return "insufficient_metadata_for_next_action"
    return "leave_manual_review"


# ── Section builders ─────────────────────────────────────────────────────────

def _select_priority_records(table: list[dict]) -> tuple[list[dict], list[dict]]:
    lang_subset: list[dict] = []
    text_subset: list[dict] = []
    for r in table:
        if r.get("unknown_failure_bucket") != "insufficient_text_visibility":
            continue
        if str(r.get("predicted_document_type") or "") != "Unknown":
            continue
        label = _diag02_visibility_label(r)
        if label == _DIAG02_LABEL_LANG_DETECTOR:
            lang_subset.append(r)
        elif label == _DIAG02_LABEL_TEXT_LAYER:
            text_subset.append(r)
    return lang_subset, text_subset


def _build_language_subset_report(records: list[dict]) -> SubsetReport:
    root_cause_counts: dict[str, int] = {label: 0 for label in _LANG_DETECTOR_LABELS}
    for r in records:
        label = classify_language_detector_root_cause(r)
        if label not in root_cause_counts:
            label = "leave_manual_review"
        root_cause_counts[label] += 1

    secondary = {
        "table_like_structure_yes": sum(
            1 for r in records
            if str(r.get("table_like_structure_detected") or "").lower() == "yes"
        ),
        "numeric_content_medium_or_high": sum(
            1 for r in records
            if str(r.get("numeric_content_bucket") or "").lower() in {"medium", "high"}
        ),
        "section_heading_yes": sum(
            1 for r in records
            if str(r.get("section_heading_shape_detected") or "").lower() == "yes"
        ),
        "medical_abbreviation_yes": sum(
            1 for r in records
            if str(r.get("medical_abbreviation_shape_detected") or "").lower() == "yes"
        ),
    }

    raw = {
        "dominant_script_counts": _safe_count(records, "dominant_script"),
        "language_visibility_status_counts":
            _safe_count(records, "language_visibility_status"),
        "native_text_length_bucket_counts":
            _safe_count(records, "native_text_length_bucket"),
        "numeric_content_bucket_counts":
            _safe_count(records, "numeric_content_bucket"),
        "table_like_structure_detected_counts":
            _safe_count(records, "table_like_structure_detected"),
    }

    return SubsetReport(
        name="language_script_visible_detector_unresolved",
        diag02_label=_DIAG02_LABEL_LANG_DETECTOR,
        count=len(records),
        root_cause_counts=root_cause_counts,
        secondary_signal_counts=secondary,
        sample_ids=_anonymized_ids("language_detector_priority", len(records)),
        raw_signal_counts=raw,
    )


def _build_text_layer_subset_report(records: list[dict]) -> SubsetReport:
    root_cause_counts: dict[str, int] = {label: 0 for label in _TEXT_LAYER_LABELS}
    for r in records:
        label = classify_text_layer_root_cause(r)
        if label not in root_cause_counts:
            label = "leave_manual_review"
        root_cause_counts[label] += 1

    secondary = {
        "table_like_structure_yes": sum(
            1 for r in records
            if str(r.get("table_like_structure_detected") or "").lower() == "yes"
        ),
        "native_text_length_none": sum(
            1 for r in records
            if str(r.get("native_text_length_bucket") or "").lower() == "none"
        ),
        "native_text_length_tiny": sum(
            1 for r in records
            if str(r.get("native_text_length_bucket") or "").lower() == "tiny"
        ),
        "native_text_length_short": sum(
            1 for r in records
            if str(r.get("native_text_length_bucket") or "").lower() == "short"
        ),
        "alphabetic_low_or_medium": sum(
            1 for r in records
            if str(r.get("alphabetic_content_bucket") or "").lower() in {"low", "medium"}
        ),
    }

    raw = {
        "pdf_text_layer_detected_counts":
            _safe_count(records, "pdf_text_layer_detected"),
        "image_like_pdf_counts": _safe_count(records, "image_like_pdf"),
        "native_text_length_bucket_counts":
            _safe_count(records, "native_text_length_bucket"),
        "table_like_structure_detected_counts":
            _safe_count(records, "table_like_structure_detected"),
        "alphabetic_content_bucket_counts":
            _safe_count(records, "alphabetic_content_bucket"),
    }

    return SubsetReport(
        name="likely_text_layer_issue",
        diag02_label=_DIAG02_LABEL_TEXT_LAYER,
        count=len(records),
        root_cause_counts=root_cause_counts,
        secondary_signal_counts=secondary,
        sample_ids=_anonymized_ids("text_layer_priority", len(records)),
        raw_signal_counts=raw,
    )


def _aggregate_next_actions(
    lang_records: list[dict],
    text_records: list[dict],
) -> dict[str, int]:
    counts: dict[str, int] = {b: 0 for b in _NEXT_ACTION_BUCKETS}
    for r in lang_records:
        action = next_action_for_language_root_cause(
            classify_language_detector_root_cause(r)
        )
        if action not in counts:
            action = "leave_manual_review"
        counts[action] += 1
    for r in text_records:
        action = next_action_for_text_layer_root_cause(
            classify_text_layer_root_cause(r)
        )
        if action not in counts:
            action = "leave_manual_review"
        counts[action] += 1
    return counts


# ── Top-level builder ────────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    lang_records, text_records = _select_priority_records(table)

    lang_subset = _build_language_subset_report(lang_records)
    text_subset = _build_text_layer_subset_report(text_records)
    next_actions = _aggregate_next_actions(lang_records, text_records)

    total = lang_subset.count + text_subset.count
    target_subset_counts = {
        _DIAG02_LABEL_LANG_DETECTOR: lang_subset.count,
        _DIAG02_LABEL_TEXT_LAYER: text_subset.count,
    }
    root_cause_counts = {
        _DIAG02_LABEL_LANG_DETECTOR: lang_subset.root_cause_counts,
        _DIAG02_LABEL_TEXT_LAYER: text_subset.root_cause_counts,
    }

    # Decision logic for the recommended follow-up implementation block.
    cand_lang = next_actions["candidate_language_detector_diagnostic"]
    cand_text = next_actions["candidate_text_layer_extraction_diagnostic"]
    cand_routing = next_actions["candidate_ocr_routing_diagnostic"]

    THRESH = 5
    if cand_lang >= THRESH and cand_text >= THRESH:
        # Larger pool first; tie -> language detector (upstream).
        if cand_lang >= cand_text:
            choice = "A"
            justified = True
            explanation = (
                f"Both diagnostics meet the threshold. {cand_lang} records "
                f"point to candidate_language_detector_diagnostic and "
                f"{cand_text} to candidate_text_layer_extraction_diagnostic. "
                f"Recommend A first (larger pool, upstream cause), then B."
            )
        else:
            choice = "B"
            justified = True
            explanation = (
                f"Both diagnostics meet the threshold. {cand_text} records "
                f"point to candidate_text_layer_extraction_diagnostic versus "
                f"{cand_lang} for the language detector. Recommend B first "
                f"(larger pool), then A."
            )
    elif cand_lang >= THRESH:
        choice = "A"
        justified = True
        explanation = (
            f"{cand_lang} records justify a language/script detector "
            f"diagnostic block (A). The text-layer pool ({cand_text}) is "
            f"below the threshold."
        )
    elif cand_text >= THRESH:
        choice = "B"
        justified = True
        explanation = (
            f"{cand_text} records justify a text-layer extraction "
            f"diagnostic block (B). The language-detector pool ({cand_lang}) "
            f"is below the threshold."
        )
    elif cand_routing >= THRESH:
        choice = "C"
        justified = True
        explanation = (
            f"{cand_routing} records suggest an OCR routing eligibility "
            f"diagnostic (C). Neither A nor B reaches the threshold."
        )
    else:
        choice = "D"
        justified = False
        explanation = (
            "No implementation lever clears the threshold. Recommend D "
            "(leave manual review). All priority records remain "
            "review-bound."
        )

    deferred = {
        "fallback_ran_but_no_family_match":
            "deferred to a future cue-coverage audit block per UNKNOWN-DIAG-02",
        "ambiguous_below_threshold":
            "remains excluded from optimization; review-bound, no cue expansion",
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
        "ambiguous_below_threshold_excluded": True,
        "fallback_ran_but_no_family_match_deferred": True,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-03",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_diag_02_label=DIAG_02_REPORT_LABEL,
        upstream_diag_01_label=DIAG_01_REPORT_LABEL,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_priority_analyzed=total,
        target_subset_counts=target_subset_counts,
        root_cause_counts=root_cause_counts,
        next_action_bucket_counts=next_actions,
        deferred_subsets=deferred,
        implementation_block_justified=justified,
        implementation_block_choice=choice,
        implementation_block_explanation=explanation,
        subsets={
            "language_script_visible_detector_unresolved": asdict(lang_subset),
            "likely_text_layer_issue": asdict(text_subset),
        },
        behavior_changed=False,
        external_api_used=False,
        safety_privacy=safety_privacy,
    )


# ── Renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-03 - Text Visibility Root-Cause Classifier")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): `{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append(f"- upstream diagnostics: `{report.upstream_diag_01_label}`, "
                 f"`{report.upstream_diag_02_label}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- total priority records analyzed: "
                 f"`{report.total_priority_analyzed}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")
    lines.append("## Target subset counts")
    lines.append("")
    for k, v in report.target_subset_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Root-cause candidate counts")
    lines.append("")
    for subset, counts in report.root_cause_counts.items():
        lines.append(f"### {subset}")
        lines.append("")
        for k, v in counts.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.append("## Next-action bucket counts")
    lines.append("")
    for k, v in report.next_action_bucket_counts.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Implementation-block recommendation")
    lines.append("")
    lines.append(
        f"- implementation_block_justified: "
        f"`{report.implementation_block_justified}`"
    )
    lines.append(f"- choice: `{report.implementation_block_choice}` "
                 f"(A=language-detector, B=text-layer, C=OCR-routing, "
                 f"D=leave manual review)")
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
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Diagnostic-only, evaluation/reporting "
                 "changes only. Cue expansion is not recommended for any bucket "
                 "in this block.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra = [
        "",
        "## Secondary signal counts (per subset)",
        "",
    ]
    for subset_name, subset in report.subsets.items():
        extra.append(f"### {subset_name}")
        extra.append("")
        for k, v in subset["secondary_signal_counts"].items():
            extra.append(f"- {k}: `{v}`")
        extra.append("")
        extra.append("Raw signal distributions (controlled vocabulary):")
        extra.append("")
        for k, v in subset["raw_signal_counts"].items():
            inner = ", ".join(f"`{kk}`={vv}" for kk, vv in v.items())
            extra.append(f"- {k}: {inner}")
        extra.append("")
    extra += [
        "## Why a root-cause classifier and not an implementation",
        "",
        "Each priority record is projected to a controlled-vocabulary "
        "root-cause label using only privacy-safe metadata already published "
        "in the FAMILY-04 per-file table. The next-action bucket names the "
        "operational diagnostic lever a follow-up block should investigate, "
        "not an implementation change. Runtime behavior is unchanged.",
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
        "json": out_dir / "medai_doc_type_unknown_diag_03_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_03_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_03.md",
    }
    paths["json"].write_text(json.dumps(json_payload, indent=2, sort_keys=True),
                              encoding="utf-8")
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-03 text-visibility root-cause classifier."
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
            "conclusion": "medai_doc_type_unknown_diag_03_ready",
            "files_written": {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_priority_analyzed": report.total_priority_analyzed,
            "target_subset_counts": report.target_subset_counts,
            "next_action_bucket_counts": report.next_action_bucket_counts,
            "implementation_block_choice": report.implementation_block_choice,
            "implementation_block_justified": report.implementation_block_justified,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
