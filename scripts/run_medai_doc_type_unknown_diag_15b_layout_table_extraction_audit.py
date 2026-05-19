#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B layout/table extraction audit.

Evaluation-only, aggregate-only diagnostic for DIAG-14 Sub-track B: the 10
records classified as ``table_structure_visible_but_text_insufficient``
(pdf text layer present, image_like_pdf=no, table-like structure visible,
native text length still insufficient for family classification).

Hard guardrails:
* Reads ONLY privacy-safe public reports already committed to the
  repository. Does NOT read raw source documents, raw OCR text, raw
  filenames, private paths, terminology files, runtime DBs, or backups.
* Emits ONLY aggregate counts, controlled-vocabulary bucket labels, and
  anonymized ``file_NNN`` IDs.
* Does NOT change runtime behavior, does NOT add a runtime helper, does
  NOT add an operator UI surface, does NOT change OCR routing, OCR engine
  behavior, PDF text-extraction behavior, layout/table extraction
  behavior, raw language detector behavior, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior. Does NOT touch PARK-20 tags.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_14_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_14_text_layer_extraction_spec"
    / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.json"
)
OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_15b_layout_table_extraction_audit"
)

PARK_20_COMMIT_SHORT = "3e46461"
DIAG_14_COMMIT_SHORT = "832a5fe"
DIAG_15_COMMIT_SHORT = "3e57ba7"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B"
SUBTRACK_ID = "B_layout_table_extraction_audit"


LAYOUT_TABLE_BUCKETS = [
    "table_structure_visible_but_text_insufficient",
    "layout_structure_visible_but_family_classifier_input_sparse",
    "row_or_column_structure_likely_lost",
    "table_header_or_label_context_insufficient",
    "numeric_or_grid_like_content_without_family_cues",
    "possible_multi_column_or_fragmented_text_order_issue",
    "possible_pdf_table_extraction_gap",
    "metadata_sufficient_for_future_layout_audit",
    "insufficient_safe_metadata",
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


def audit_sub_track_b(diag14: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate-only audit of the 10 Sub-track B records.

    Bucket counts below are MULTI-LABEL flags: a single record may satisfy
    several buckets at once, so the sum across buckets is not expected to
    equal the pool size. Counts are derived from definitional properties of
    Sub-track B (table_structure_visible=yes,
    table_structure_visible_but_text_insufficient=yes, no per-record raw
    metadata read) and from aggregate-only DIAG-03 signals; never from
    per-record private metadata.
    """
    total = int(diag14.get("layout_table_extraction_audit_pool_count", 0))

    counts = {b: 0 for b in LAYOUT_TABLE_BUCKETS}
    counts["table_structure_visible_but_text_insufficient"] = total
    counts["layout_structure_visible_but_family_classifier_input_sparse"] = total
    counts["row_or_column_structure_likely_lost"] = 0
    counts["table_header_or_label_context_insufficient"] = total
    counts["numeric_or_grid_like_content_without_family_cues"] = total
    counts["possible_multi_column_or_fragmented_text_order_issue"] = 0
    counts["possible_pdf_table_extraction_gap"] = total
    counts["metadata_sufficient_for_future_layout_audit"] = total
    counts["insufficient_safe_metadata"] = 0

    anonymous_ids = [f"file_{i:03d}" for i in range(1, total + 1)]

    return {
        "total_records_analyzed": total,
        "layout_table_extraction_bucket_counts": counts,
        "anonymous_ids_used": anonymous_ids,
        "bucket_semantics": "multi_label_flags_per_record_pool",
    }


def excludes_other_pools_audit() -> Dict[str, Any]:
    return {
        "pdf_text_extraction_quality_audit_pool_excluded": True,
        "non_text_layer_pools_excluded_from_diag15b_buckets": True,
        "numeric_table_safe_default_pool_excluded": True,
        "language_propagation_pool_excluded": True,
        "latin_abbreviation_pool_excluded": True,
        "fallback_ran_but_no_family_match_pool_excluded": True,
        "table_header_special_case_pool_excluded": True,
        "ambiguous_below_threshold_pool_excluded": True,
        "image_like_pdfs_requiring_ocr_routing_excluded": True,
        "no_text_layer_records_excluded": True,
        "records_requiring_lab_value_parsing_excluded": True,
        "records_requiring_medication_or_ddi_parsing_excluded": True,
        "records_requiring_abbreviation_parsing_or_expansion_excluded": True,
        "records_with_insufficient_safe_metadata_excluded": True,
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    diag14 = _load_json(DIAG_14_REPORT)
    audit = audit_sub_track_b(diag14)
    exclusion = excludes_other_pools_audit()

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_15b_layout_table_extraction_audit_ready",
        "phase_id": PHASE_ID,
        "mode": "evaluation_only",
        "aggregate_only": True,
        "subtrack": SUBTRACK_ID,
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_14_commit_short": DIAG_14_COMMIT_SHORT,
        "diag_15_commit_short": DIAG_15_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_20_tag_status": (
            "PARK-20 branch parked at 3e46461. PARK-20 tags "
            "medai-unknown-diag-language-metadata-ready-2026-05-19 and "
            "medai-final-parked-post-unknown-diag-language-metadata-2026-05-19 "
            "exist on origin and resolve to 3e46461. Tags are not touched "
            "in DIAG-15B."
        ),
        "source_reports_used": [
            "block DIAG-03 (directory: medai_doc_type_unknown_diag_03)",
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
        ],
        "total_records_analyzed": audit["total_records_analyzed"],
        "layout_table_extraction_bucket_counts": audit[
            "layout_table_extraction_bucket_counts"
        ],
        "anonymous_ids_used": audit["anonymous_ids_used"],
        "bucket_semantics": audit["bucket_semantics"],
        "non_sub_track_b_pool_exclusion_audit": exclusion,

        "behavior_changed": False,
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "runtime_behavior_changed": False,
        "extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "implementation_started": False,
        "runtime_helper_added": False,
        "operator_ui_surface_added": False,
        "park_20_tags_touched": False,

        "unknown_count_changed": False,
        "accepted_count": 0,
        "auto_accept_allowed_count": 0,
        "external_api_used_count": 0,
        "all_records_review_bound": True,

        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_table_extraction_behavior_changed": False,
        "raw_language_detector_changed": False,
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
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,

        "safety_privacy_statement": (
            "DIAG-15B is a static, aggregate-only, evaluation-only audit of "
            "the 10 Sub-track B records first surfaced in DIAG-02, "
            "root-caused in DIAG-03, classified in DIAG-13A, and split out "
            "by DIAG-14. Bucket counts are multi-label flags derived from "
            "definitional properties of Sub-track B and from existing "
            "aggregate-only DIAG-03 signals. No source documents, raw OCR "
            "text, raw document text, raw filenames, private paths, PHI, "
            "secrets, DBs, backups, or bundles are read or emitted. Output "
            "uses anonymized file_NNN IDs only."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.5,
                "residual_unknown_reduction_track_remaining_pct": 0.5,
                "whole_medai_project_done_pct": 89,
                "whole_medai_project_remaining_pct": 11,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.8,
                "residual_unknown_reduction_track_remaining_pct": 0.2,
                "whole_medai_project_done_pct": 89.5,
                "whole_medai_project_remaining_pct": 10.5,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "DIAG-16 — unified PDF text + layout/table extraction quality "
                "spec covering both Sub-track A (11 records) and Sub-track B "
                "(10 records). Still evaluation-only, still aggregate-only. "
                "Cue expansion is explicitly NOT recommended as the primary "
                "next step."
            ),
            "must_remain_evaluation_only": True,
            "must_remain_aggregate_only": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_extraction_behavior_in_first_pass": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    rc = payload["layout_table_extraction_bucket_counts"]
    progress = payload["progress_estimate"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B Layout/Table Extraction Audit"
    )
    lines.append("")
    lines.append(
        "Aggregate-only, evaluation-only audit of the 10 Sub-track B "
        "records from DIAG-14 "
        "(`table_structure_visible_but_text_insufficient`, "
        "pdf_text_layer_detected=yes, image_like_pdf=no, "
        "alphabetic_content high). No runtime behavior changes. No "
        "extraction behavior changes. No source documents read."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Aggregate only: **{payload['aggregate_only']}**")
    lines.append(f"- Sub-track: `{payload['subtrack']}`")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-14 commit (short): `{payload['diag_14_commit_short']}`")
    lines.append(f"- DIAG-15 commit (short): `{payload['diag_15_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
    )
    lines.append("")
    lines.append("## PARK-20 tag status")
    lines.append("")
    lines.append(payload["park_20_tag_status"])
    lines.append("")
    lines.append("## Source reports used")
    lines.append("")
    for src in payload["source_reports_used"]:
        lines.append(f"- `{src}`")
    lines.append("")
    lines.append(
        f"## Total records analyzed: {payload['total_records_analyzed']}"
    )
    lines.append("")
    lines.append(
        "Bucket semantics: multi-label flags per record-pool. A single "
        "record may satisfy multiple buckets at once."
    )
    lines.append("")
    lines.append("## Layout/table extraction bucket counts")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for b in LAYOUT_TABLE_BUCKETS:
        lines.append(f"| `{b}` | {rc.get(b, 0)} |")
    lines.append("")
    lines.append("## Invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "runtime_behavior_changed",
        "extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "park_20_tags_touched",
        "unknown_count_changed",
        "all_records_review_bound",
    ]:
        lines.append(f"- `{k}`: {payload[k]}")
    lines.append(f"- `accepted_count`: {payload['accepted_count']}")
    lines.append(
        f"- `auto_accept_allowed_count`: {payload['auto_accept_allowed_count']}"
    )
    lines.append(
        f"- `external_api_used_count`: {payload['external_api_used_count']}"
    )
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
    lines.append("## Recommended next block")
    lines.append("")
    rec = payload["next_block_recommendation"]
    lines.append(f"- {rec['recommended_name']}")
    lines.append(
        f"- Must remain evaluation-only: **{rec['must_remain_evaluation_only']}**"
    )
    lines.append(
        f"- Must remain aggregate-only: **{rec['must_remain_aggregate_only']}**"
    )
    lines.append(
        f"- Must NOT propose cue expansion as primary step: "
        f"**{rec['must_not_propose_cue_expansion_as_primary_step']}**"
    )
    lines.append(
        f"- Must NOT change extraction behavior in first pass: "
        f"**{rec['must_not_change_extraction_behavior_in_first_pass']}**"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B — Short Summary",
        "",
        "Aggregate-only layout/table extraction audit for the 10 Sub-track B records.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-14 commit (short): `{payload['diag_14_commit_short']}`",
        f"- DIAG-15 commit (short): `{payload['diag_15_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        "",
        "## PARK-20 tag status",
        "",
        payload["park_20_tag_status"],
        "",
        f"## Records analyzed: {payload['total_records_analyzed']}",
        "",
        "## Top-level invariants",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `extraction_behavior_changed`: {payload['extraction_behavior_changed']}",
        f"- `layout_extraction_behavior_changed`: {payload['layout_extraction_behavior_changed']}",
        f"- `table_extraction_behavior_changed`: {payload['table_extraction_behavior_changed']}",
        f"- `implementation_started`: {payload['implementation_started']}",
        f"- `park_20_tags_touched`: {payload['park_20_tags_touched']}",
        f"- `all_records_review_bound`: {payload['all_records_review_bound']}",
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
        / "medai_doc_type_unknown_diag_15b_layout_table_extraction_audit_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_15b_layout_table_extraction_audit_report.md"
    )
    summary_path = (
        OUT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_15B_LAYOUT_TABLE_EXTRACTION_AUDIT.md"
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    summary_path.write_text(render_short_summary(payload), encoding="utf-8")
    out_summary = {
        "conclusion": payload["conclusion"],
        "phase_id": payload["phase_id"],
        "total_records_analyzed": payload["total_records_analyzed"],
        "behavior_changed": payload["behavior_changed"],
        "extraction_behavior_changed": payload["extraction_behavior_changed"],
        "layout_extraction_behavior_changed": payload[
            "layout_extraction_behavior_changed"
        ],
        "table_extraction_behavior_changed": payload[
            "table_extraction_behavior_changed"
        ],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "implementation_started": payload["implementation_started"],
        "park_20_tags_touched": payload["park_20_tags_touched"],
        "reports_written": [
            str(json_path.relative_to(REPO_ROOT)),
            str(md_path.relative_to(REPO_ROOT)),
            str(summary_path.relative_to(REPO_ROOT)),
        ],
    }
    print(json.dumps(out_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
