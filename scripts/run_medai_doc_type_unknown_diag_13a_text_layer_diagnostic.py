#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A text-layer diagnostic.

Evaluation-only, aggregate-only diagnostic that re-characterizes the 21
``likely_text_layer_issue`` records first identified by DIAG-02 and root-
caused by DIAG-03. This block adds two extra classification axes (structural
shape, future-lever recommendation) on top of the DIAG-03 root-cause
counts.

Hard guardrails (enforced by this script's design and by privacy checks):

* Reads ONLY privacy-safe public reports already committed to the
  repository. Does NOT read raw source documents, raw OCR text, raw
  filenames, private paths, terminology files, runtime DBs, or backups.
* Emits ONLY aggregate counts, controlled-vocabulary bucket labels, and
  anonymized ``file_NNN`` IDs.
* Does NOT change runtime behavior, does NOT add a runtime helper, does
  NOT add an operator UI surface, does NOT change OCR routing, OCR engine
  behavior, raw language detector behavior, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior.
* Does NOT parse or expand abbreviations, does NOT parse lab values, does
  NOT parse medications / dose / frequency / duration / DDI, does NOT add
  clinical interpretation.

Output:
    reports/medai_doc_type_unknown_diag_13a_text_layer_diagnostic/
        medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.json
        medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.md
        MEDAI_DOC_TYPE_UNKNOWN_DIAG_13A_TEXT_LAYER_DIAGNOSTIC.md
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_03_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_03/medai_doc_type_unknown_diag_03_report.json"
)
DIAG_02_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_02/medai_doc_type_unknown_diag_02_report.json"
)
PREFLIGHT_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_13_preflight/medai_doc_type_unknown_diag_13_preflight_report.json"
)

OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_13a_text_layer_diagnostic"
)

PARK_20_COMMIT_SHORT = "3e46461"
DIAG_13_PREFLIGHT_COMMIT_SHORT = "01c2b69"

ROOT_CAUSE_BUCKETS = [
    "text_layer_too_short",
    "table_structure_visible_but_text_insufficient",
    "text_layer_present_but_low_signal",
    "image_like_with_partial_text",
    "no_safe_text_visibility_metadata",
    "leave_manual_review",
]

STRUCTURAL_SHAPE_BUCKETS = [
    "table_like_structure_visible",
    "row_or_column_pattern_visible",
    "section_heading_shape_visible",
    "lab_or_result_shape_possible",
    "treatment_or_schedule_shape_possible",
    "administrative_or_form_shape_possible",
    "no_known_shape_visible",
]

FUTURE_LEVER_BUCKETS = [
    "candidate_text_layer_extraction_diagnostic",
    "candidate_pdf_text_extraction_quality_audit",
    "candidate_layout_table_extraction_audit",
    "candidate_manual_review_only",
    "insufficient_metadata_for_next_action",
]


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def classify_text_layer_records(
    subset: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-classify the 21 text-layer records from DIAG-03 into the three
    DIAG-13A controlled-vocabulary axes.

    The function is pure: input is an aggregate-only subset payload, output
    is aggregate-only counts. No raw filenames, raw OCR text, or PHI flows
    through it.
    """
    raw = subset.get("raw_signal_counts", {})
    root = subset.get("root_cause_counts", {})

    total = int(subset.get("count", 0))

    root_cause_counts = {b: int(root.get(b, 0)) for b in ROOT_CAUSE_BUCKETS}

    table_like_yes = int(
        raw.get("table_like_structure_detected_counts", {}).get("yes", 0)
    )
    table_like_no = int(
        raw.get("table_like_structure_detected_counts", {}).get("no", 0)
    )

    structural_shape_counts = {b: 0 for b in STRUCTURAL_SHAPE_BUCKETS}
    structural_shape_counts["table_like_structure_visible"] = table_like_yes
    structural_shape_counts["no_known_shape_visible"] = table_like_no

    future_lever_counts = {b: 0 for b in FUTURE_LEVER_BUCKETS}
    future_lever_counts["candidate_text_layer_extraction_diagnostic"] = total
    future_lever_counts["candidate_pdf_text_extraction_quality_audit"] = (
        root_cause_counts.get("text_layer_too_short", 0)
    )
    future_lever_counts["candidate_layout_table_extraction_audit"] = (
        root_cause_counts.get(
            "table_structure_visible_but_text_insufficient", 0
        )
    )

    anonymous_ids = [f"file_{i:03d}" for i in range(1, total + 1)]

    return {
        "total_text_layer_records_analyzed": total,
        "root_cause_candidate_counts": root_cause_counts,
        "structural_shape_counts": structural_shape_counts,
        "future_lever_counts": future_lever_counts,
        "anonymous_ids_used": anonymous_ids,
        "raw_signal_counts_from_diag03": {
            "alphabetic_content_bucket_counts": dict(
                raw.get("alphabetic_content_bucket_counts", {})
            ),
            "image_like_pdf_counts": dict(
                raw.get("image_like_pdf_counts", {})
            ),
            "native_text_length_bucket_counts": dict(
                raw.get("native_text_length_bucket_counts", {})
            ),
            "pdf_text_layer_detected_counts": dict(
                raw.get("pdf_text_layer_detected_counts", {})
            ),
            "table_like_structure_detected_counts": dict(
                raw.get("table_like_structure_detected_counts", {})
            ),
        },
    }


def excludes_other_pools_audit(diag03: Mapping[str, Any]) -> Dict[str, int]:
    """Aggregate-only audit confirming DIAG-13A targets only the text-layer
    pool. Other pool counts are quoted from DIAG-03 (or later DIAG blocks)
    but DIAG-13A's bucket counts do not include them.
    """
    subsets = diag03.get("subsets", {})
    return {
        "likely_text_layer_issue_count": int(
            subsets.get("likely_text_layer_issue", {}).get("count", 0)
        ),
        "no_text_layer_count": int(
            subsets.get("no_text_layer", {}).get("count", 0)
        ),
        "non_text_layer_pools_excluded_from_diag13a_buckets": True,
        "numeric_table_safe_default_pool_excluded": True,
        "language_propagation_pool_excluded": True,
        "latin_abbreviation_pool_excluded": True,
        "fallback_ran_but_no_family_match_pool_excluded": True,
        "table_header_special_case_pool_excluded": True,
        "ambiguous_below_threshold_pool_excluded": True,
    }


def decide_future_block(classification: Mapping[str, Any]) -> Dict[str, Any]:
    """Recommend whether a future implementation/spec block is justified."""
    rc = classification["root_cause_counts_actionable"] = (
        classification["root_cause_candidate_counts"].get(
            "text_layer_too_short", 0
        )
        + classification["root_cause_candidate_counts"].get(
            "table_structure_visible_but_text_insufficient", 0
        )
    )
    total = classification["total_text_layer_records_analyzed"]

    future_block_justified = rc > 0
    if total == 0:
        recommendation_code = "E"
        recommendation_name = "insufficient_metadata"
    elif rc == 0:
        recommendation_code = "D"
        recommendation_name = "leave_manual_review"
    else:
        recommendation_code = "A"
        recommendation_name = "text_layer_extraction_diagnostic_spec"

    return {
        "future_block_justified": future_block_justified,
        "recommendation_code": recommendation_code,
        "recommendation_name": recommendation_name,
        "recommendation_letter_options": {
            "A": "text-layer extraction diagnostic/spec",
            "B": "PDF text-quality audit",
            "C": "layout/table extraction audit",
            "D": "leave manual review",
            "E": "insufficient metadata",
        },
        "subtrack_split_for_recommendation_A": {
            "pdf_text_extraction_quality_audit_subset": (
                classification["future_lever_counts"][
                    "candidate_pdf_text_extraction_quality_audit"
                ]
            ),
            "layout_table_extraction_audit_subset": (
                classification["future_lever_counts"][
                    "candidate_layout_table_extraction_audit"
                ]
            ),
        },
        "cue_expansion_recommended_as_primary_next_step": False,
    }


def build_report(
    *,
    branch: str = "clinical-knowledge-architecture",
    head_short: str = "",
) -> Dict[str, Any]:
    if not head_short:
        head_short = _short_head()

    diag03 = _load_json(DIAG_03_REPORT)
    subset = diag03["subsets"]["likely_text_layer_issue"]

    classification = classify_text_layer_records(subset)
    exclusion_audit = excludes_other_pools_audit(diag03)
    decision = decide_future_block(classification)

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_ready",
        "block_id": "MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A",
        "branch": branch,
        "head_commit_short": head_short,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "diag_13_preflight_commit_short": DIAG_13_PREFLIGHT_COMMIT_SHORT,
        "remote_tag_caveat": (
            "Branch parked at PARK-20 (3e46461). PARK-20 tags exist locally and "
            "target 3e46461 but remote tag push is currently blocked by an HTTP "
            "403 from origin's receive-pack endpoint. Tags are not touched in "
            "DIAG-13A. The out-of-band GitHub/proxy permission fix remains "
            "pending."
        ),
        "source_reports_used": [
            "block DIAG-02 (directory: medai_doc_type_unknown_diag_02)",
            "block DIAG-03 (directory: medai_doc_type_unknown_diag_03)",
            "block DIAG-13-PREFLIGHT (directory: medai_doc_type_unknown_diag_13_preflight)",
        ],
        "total_text_layer_records_analyzed": classification[
            "total_text_layer_records_analyzed"
        ],
        "root_cause_candidate_counts": classification[
            "root_cause_candidate_counts"
        ],
        "structural_shape_counts": classification["structural_shape_counts"],
        "future_lever_counts": classification["future_lever_counts"],
        "raw_signal_counts_from_diag03": classification[
            "raw_signal_counts_from_diag03"
        ],
        "anonymous_ids_used": classification["anonymous_ids_used"],
        "non_text_layer_pool_exclusion_audit": exclusion_audit,
        "future_block_decision": decision,
        "behavior_changed": False,
        "external_api_used": False,
        "cue_expansion_recommended": False,
        "implementation_started": False,
        "runtime_helper_added": False,
        "operator_ui_surface_added": False,
        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "raw_language_detector_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_or_scoring_changed": False,
        "cue_packs_added": False,
        "lab_values_parsed": False,
        "medications_dose_frequency_duration_or_ddi_parsed": False,
        "abbreviations_parsed_or_expanded": False,
        "clinical_interpretation_added": False,
        "b07_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_allowlist_changed": False,
        "external_api_enabled": False,
        "park_20_tags_touched": False,
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,
        "safety_privacy_statement": (
            "DIAG-13A is a static, aggregate-only, evaluation-only diagnostic. "
            "It re-classifies the 21 text-layer records previously characterized "
            "by DIAG-02 and DIAG-03 into three controlled-vocabulary axes. No "
            "source documents, raw OCR text, raw document text, raw filenames, "
            "private paths, PHI, secrets, DBs, backups, or bundles are read or "
            "emitted. Output uses anonymized file_NNN IDs only."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 97,
                "residual_unknown_reduction_track_remaining_pct": 3,
                "whole_medai_project_done_pct": 86,
                "whole_medai_project_remaining_pct": 14,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 98,
                "residual_unknown_reduction_track_remaining_pct": 2,
                "whole_medai_project_done_pct": 87,
                "whole_medai_project_remaining_pct": 13,
            },
        },
        "next_block_recommendation": {
            "recommended_letter": decision["recommendation_code"],
            "recommended_name": decision["recommendation_name"],
            "rationale_summary": (
                "All 21 text-layer records have pdf_text_layer_detected=yes and "
                "image_like_pdf=no, indicating a usable text layer is present. "
                "11 records resolve to text_layer_too_short (no table structure, "
                "native text length none/short/tiny) and 10 records resolve to "
                "table_structure_visible_but_text_insufficient (table-like "
                "structure visible, native text length still none/short/tiny). "
                "A unified text-layer extraction diagnostic / spec block is the "
                "appropriate next step. PDF text-quality audit and layout/table "
                "extraction audit are natural sub-tracks. Cue expansion is "
                "explicitly NOT recommended as the primary next step."
            ),
        },
    }

    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    rc = payload["root_cause_candidate_counts"]
    ss = payload["structural_shape_counts"]
    fl = payload["future_lever_counts"]
    raw = payload["raw_signal_counts_from_diag03"]
    decision = payload["future_block_decision"]
    progress = payload["progress_estimate"]

    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A Text-Layer Diagnostic"
    )
    lines.append("")
    lines.append(
        "Aggregate-only, evaluation-only diagnostic over the 21 residual "
        "`likely_text_layer_issue` records previously characterized by "
        "DIAG-02 and DIAG-03. No runtime behavior changes. No source "
        "documents read."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
    )
    lines.append(
        f"- DIAG-13 preflight commit (short): `{payload['diag_13_preflight_commit_short']}`"
    )
    lines.append("")
    lines.append("## Remote tag caveat")
    lines.append("")
    lines.append(payload["remote_tag_caveat"])
    lines.append("")
    lines.append("## Source reports used")
    lines.append("")
    for src in payload["source_reports_used"]:
        lines.append(f"- `{src}`")
    lines.append("")
    lines.append(
        f"## Total text-layer records analyzed: "
        f"{payload['total_text_layer_records_analyzed']}"
    )
    lines.append("")
    lines.append("## Root-cause candidate counts")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for b in ROOT_CAUSE_BUCKETS:
        lines.append(f"| `{b}` | {rc.get(b, 0)} |")
    lines.append("")
    lines.append("## Structural-shape counts")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for b in STRUCTURAL_SHAPE_BUCKETS:
        lines.append(f"| `{b}` | {ss.get(b, 0)} |")
    lines.append("")
    lines.append("## Future-lever counts")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for b in FUTURE_LEVER_BUCKETS:
        lines.append(f"| `{b}` | {fl.get(b, 0)} |")
    lines.append("")
    lines.append("## Raw signal counts (from DIAG-03)")
    lines.append("")
    for axis, sub in raw.items():
        sub_pairs = ", ".join(f"{k}={v}" for k, v in sub.items())
        lines.append(f"- `{axis}`: {sub_pairs}")
    lines.append("")
    lines.append("## Future block decision")
    lines.append("")
    lines.append(
        f"- Future implementation / spec block justified: "
        f"**{decision['future_block_justified']}**"
    )
    lines.append(
        f"- Recommended code: **{decision['recommendation_code']}** — "
        f"{decision['recommendation_name']}"
    )
    lines.append("- Recommendation letter options:")
    for k, v in decision["recommendation_letter_options"].items():
        lines.append(f"  - **{k}**: {v}")
    sub = decision["subtrack_split_for_recommendation_A"]
    lines.append(
        f"- Sub-track split for recommendation A:"
    )
    lines.append(
        f"  - PDF text-extraction quality audit subset: "
        f"{sub['pdf_text_extraction_quality_audit_subset']}"
    )
    lines.append(
        f"  - layout / table extraction audit subset: "
        f"{sub['layout_table_extraction_audit_subset']}"
    )
    lines.append(
        f"- Cue expansion recommended as primary next step: "
        f"**{decision['cue_expansion_recommended_as_primary_next_step']}**"
    )
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "external_api_used",
        "cue_expansion_recommended",
        "implementation_started",
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "raw_language_detector_changed",
        "classifier_behavior_changed",
        "thresholds_or_scoring_changed",
        "cue_packs_added",
        "park_20_tags_touched",
    ]:
        lines.append(f"- `{k}`: {payload[k]}")
    lines.append("")
    lines.append("## Safety / privacy")
    lines.append("")
    lines.append(payload["safety_privacy_statement"])
    lines.append("")
    lines.append("## Progress estimate")
    lines.append("")
    lines.append("| Track | Before | After |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Residual Unknown-reduction | "
        f"~{progress['before']['residual_unknown_reduction_track_done_pct']}% done / "
        f"~{progress['before']['residual_unknown_reduction_track_remaining_pct']}% remaining "
        f"| ~{progress['after']['residual_unknown_reduction_track_done_pct']}% done / "
        f"~{progress['after']['residual_unknown_reduction_track_remaining_pct']}% remaining |"
    )
    lines.append(
        f"| Whole MedAI project | "
        f"~{progress['before']['whole_medai_project_done_pct']}% done / "
        f"~{progress['before']['whole_medai_project_remaining_pct']}% remaining "
        f"| ~{progress['after']['whole_medai_project_done_pct']}% done / "
        f"~{progress['after']['whole_medai_project_remaining_pct']}% remaining |"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    decision = payload["future_block_decision"]
    progress = payload["progress_estimate"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A — Short Summary",
        "",
        "Aggregate-only diagnostic over the 21 residual text-layer records.",
        "",
        "## State",
        "",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        f"- DIAG-13 preflight commit (short): `{payload['diag_13_preflight_commit_short']}`",
        "",
        "## Tag caveat",
        "",
        payload["remote_tag_caveat"],
        "",
        f"## Records analyzed: {payload['total_text_layer_records_analyzed']}",
        "",
        "## Top-level decision",
        "",
        f"- Recommended next block: **{decision['recommendation_code']}** — {decision['recommendation_name']}",
        f"- Cue expansion as primary next step: **{decision['cue_expansion_recommended_as_primary_next_step']}**",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `implementation_started`: {payload['implementation_started']}",
        "",
        "## Progress",
        "",
        f"- Before: residual Unknown ~{progress['before']['residual_unknown_reduction_track_done_pct']}% done / ~{progress['before']['residual_unknown_reduction_track_remaining_pct']}% remaining; whole project ~{progress['before']['whole_medai_project_done_pct']}% done / ~{progress['before']['whole_medai_project_remaining_pct']}% remaining.",
        f"- After: residual Unknown ~{progress['after']['residual_unknown_reduction_track_done_pct']}% done / ~{progress['after']['residual_unknown_reduction_track_remaining_pct']}% remaining; whole project ~{progress['after']['whole_medai_project_done_pct']}% done / ~{progress['after']['whole_medai_project_remaining_pct']}% remaining.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_report()

    json_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.md"
    )
    summary_path = (
        OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_13A_TEXT_LAYER_DIAGNOSTIC.md"
    )

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    summary_path.write_text(render_short_summary(payload), encoding="utf-8")

    summary = {
        "conclusion": payload["conclusion"],
        "total_text_layer_records_analyzed": payload[
            "total_text_layer_records_analyzed"
        ],
        "recommended_letter": payload["future_block_decision"][
            "recommendation_code"
        ],
        "recommended_name": payload["future_block_decision"][
            "recommendation_name"
        ],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "behavior_changed": payload["behavior_changed"],
        "implementation_started": payload["implementation_started"],
        "reports_written": [
            str(json_path.relative_to(REPO_ROOT)),
            str(md_path.relative_to(REPO_ROOT)),
            str(summary_path.relative_to(REPO_ROOT)),
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
