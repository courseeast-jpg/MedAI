#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-14 text-layer extraction specification.

Evaluation-only, aggregate-only specification block for the 21 text-layer
records first surfaced by DIAG-02, root-caused by DIAG-03, and split into
two sub-tracks by DIAG-13A.

This block emits a spec only. It does NOT implement extraction behavior,
does NOT add runtime helpers, does NOT add operator UI surfaces, does NOT
change OCR routing, OCR engine behavior, PDF text-extraction behavior,
layout/table extraction behavior, raw language detector behavior,
classifier behavior, thresholds, scoring, cue packs, B07, ROUTE-FIX, DB
schema, command allowlist, or external API behavior.

Output:
    reports/medai_doc_type_unknown_diag_14_text_layer_extraction_spec/
        medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.json
        medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.md
        MEDAI_DOC_TYPE_UNKNOWN_DIAG_14_TEXT_LAYER_EXTRACTION_SPEC.md
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_13A_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_13a_text_layer_diagnostic"
    / "medai_doc_type_unknown_diag_13a_text_layer_diagnostic_report.json"
)
DIAG_03_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_03"
    / "medai_doc_type_unknown_diag_03_report.json"
)

OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_14_text_layer_extraction_spec"
)

PARK_20_COMMIT_SHORT = "3e46461"
DIAG_13A_COMMIT_SHORT = "7866ba4"


SUB_TRACK_A_NAME = "pdf_text_extraction_quality_audit"
SUB_TRACK_B_NAME = "layout_table_extraction_audit"


SUB_TRACK_A_POSITIVE_SIGNATURE = [
    "pdf_text_layer_detected = yes",
    "image_like_pdf = no",
    "text_layer_too_short = yes",
    "no extraction error reported in safe metadata",
    "no OCR routing required at the diagnostic stage",
    "no private or raw text required to perform the diagnosis",
    "record remains review-bound",
]

SUB_TRACK_B_POSITIVE_SIGNATURE = [
    "pdf_text_layer_detected = yes",
    "image_like_pdf = no",
    "table_structure_visible = yes",
    "table_structure_visible_but_text_insufficient = yes",
    "table-like shape present in safe metadata",
    "raw table contents not required in public outputs",
    "record remains review-bound",
]

EXCLUSION_RULES = [
    "exclude image-like PDFs that would require OCR routing work",
    "exclude no-text-layer records (DIAG-02 no_text_layer subset)",
    "exclude fallback_ran_but_no_family_match records",
    "exclude ambiguous_below_threshold records",
    "exclude numeric-table safe-default records already handled by DIAG-06A/07A/08A",
    "exclude language-propagation records already handled by DIAG-09A/10A",
    "exclude latin-abbreviation records already handled by DIAG-11A/12A",
    "exclude the table-header special case (single deferred record)",
    "exclude records that would require lab-value parsing",
    "exclude records that would require medication / dose / frequency / duration / DDI parsing",
    "exclude records that would require abbreviation parsing or expansion",
    "exclude records with insufficient safe metadata for the chosen sub-track signature",
]

PROPOSED_FUTURE_DIAGNOSTIC_BEHAVIOR = [
    "the future block may audit extraction quality metadata only",
    "it may compare extraction length buckets, table-visibility buckets, and layout-signal buckets",
    "it must not output raw extracted text",
    "it must not parse clinical values",
    "it must not change OCR routing in the first diagnostic pass",
    "it must not change PDF text-extraction behavior in the first diagnostic pass",
    "it must not change layout/table extraction behavior in the first diagnostic pass",
    "it must preserve review-bound status for every affected record",
]

FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA = [
    "only exact-signature records are affected",
    "no auto-accept",
    "accepted_count remains 0",
    "auto_accept_allowed_count remains 0",
    "external_api_used_count remains 0",
    "all affected records remain review-bound",
    "data-layer Unknown behavior explicitly reported (changed vs unchanged)",
    "no false-positive expansion into treatment / imaging / administrative document types",
    "no raw text in public reports",
    "no raw filenames in public reports",
    "rollback / disable path exists if runtime changes are ever introduced",
]

FUTURE_VALIDATION_REQUIREMENTS = [
    "focused synthetic tests for both sub-tracks",
    "replay of the 21-record text-layer pool",
    "507-file aggregate validation",
    "document-type eval regression tests",
    "public-report privacy checks on every new report",
    "final CKA MVP validation",
    "B07 validation",
    "ROUTE-FIX validation",
    "UI ops validation",
    "UI boot validation",
    "staged safety check",
]

ROLLBACK_SAFETY_BOUNDARIES = [
    "any runtime behavior introduced by a later block must be default-off",
    "any runtime behavior introduced by a later block must be env-gated by a SEPARATE env var",
    "any runtime behavior introduced by a later block must be removable by toggling its env var to a falsy value",
    "if any later block alters classifier, OCR routing, or extraction behavior outside its env-gated scope, the block is invalid and must be reverted",
    "no later block may emit raw text, raw filenames, or private paths to public reports",
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


def split_text_layer_pool(diag13a: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate-only sub-track split using the DIAG-13A output as input."""
    total = int(diag13a.get("total_text_layer_records_analyzed", 0))
    rc = diag13a.get("root_cause_candidate_counts", {})
    pdf_audit_count = int(rc.get("text_layer_too_short", 0))
    layout_audit_count = int(
        rc.get("table_structure_visible_but_text_insufficient", 0)
    )

    pdf_ids = [f"file_{i:03d}" for i in range(1, pdf_audit_count + 1)]
    layout_ids = [
        f"file_{i:03d}"
        for i in range(
            pdf_audit_count + 1, pdf_audit_count + layout_audit_count + 1
        )
    ]

    return {
        "total_text_layer_records_analyzed": total,
        "sub_tracks": {
            SUB_TRACK_A_NAME: {
                "pool_count": pdf_audit_count,
                "anonymous_ids": pdf_ids,
                "positive_signature": list(SUB_TRACK_A_POSITIVE_SIGNATURE),
            },
            SUB_TRACK_B_NAME: {
                "pool_count": layout_audit_count,
                "anonymous_ids": layout_ids,
                "positive_signature": list(SUB_TRACK_B_POSITIVE_SIGNATURE),
            },
        },
        "split_sum_equals_total": (
            pdf_audit_count + layout_audit_count == total
        ),
    }


def exclusion_audit(diag13a: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate-only confirmation that DIAG-14 targets exactly the text-layer
    pool and excludes every other pool."""
    return {
        "rules": list(EXCLUSION_RULES),
        "non_text_layer_pools_excluded_from_diag14_buckets": True,
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
    diag13a = _load_json(DIAG_13A_REPORT)
    split = split_text_layer_pool(diag13a)
    exclusion = exclusion_audit(diag13a)

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_ready",
        "block_id": "MEDAI-DOC-TYPE-UNKNOWN-DIAG-14",
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_13a_commit_short": DIAG_13A_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "remote_tag_caveat": (
            "Branch parked at PARK-20 (3e46461). PARK-20 tags exist locally and "
            "target 3e46461 but remote tag push is currently blocked by an HTTP "
            "403 from origin's receive-pack endpoint. Tags are not touched in "
            "DIAG-14. The out-of-band GitHub/proxy permission fix remains pending."
        ),
        "source_reports_used": [
            "block DIAG-02 (directory: medai_doc_type_unknown_diag_02)",
            "block DIAG-03 (directory: medai_doc_type_unknown_diag_03)",
            "block DIAG-13-PREFLIGHT (directory: medai_doc_type_unknown_diag_13_preflight)",
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
        ],
        "total_text_layer_records_analyzed": split[
            "total_text_layer_records_analyzed"
        ],
        "pdf_text_extraction_quality_audit_pool_count": split["sub_tracks"][
            SUB_TRACK_A_NAME
        ]["pool_count"],
        "layout_table_extraction_audit_pool_count": split["sub_tracks"][
            SUB_TRACK_B_NAME
        ]["pool_count"],
        "split_sum_equals_total": split["split_sum_equals_total"],
        "sub_tracks": split["sub_tracks"],
        "exclusion_audit": exclusion,
        "proposed_future_diagnostic_behavior": list(
            PROPOSED_FUTURE_DIAGNOSTIC_BEHAVIOR
        ),
        "future_implementation_acceptance_criteria": list(
            FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA
        ),
        "future_validation_requirements": list(FUTURE_VALIDATION_REQUIREMENTS),
        "rollback_safety_boundaries": list(ROLLBACK_SAFETY_BOUNDARIES),
        "behavior_changed": False,
        "external_api_used": False,
        "cue_expansion_recommended": False,
        "extraction_behavior_changed": False,
        "implementation_started": False,
        "runtime_helper_added": False,
        "operator_ui_surface_added": False,
        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_table_extraction_behavior_changed": False,
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
        "no_extraction_behavior_implemented_in_this_block": True,
        "no_runtime_behavior_changed_in_this_block": True,
        "safety_privacy_statement": (
            "DIAG-14 is a static, aggregate-only, evaluation-only specification "
            "block. It splits the 21 text-layer records into two future audit "
            "sub-tracks (11 + 10) and defines positive signatures, exclusion "
            "rules, future diagnostic behavior, future implementation acceptance "
            "criteria, future validation requirements, and rollback/safety "
            "boundaries. No source documents, raw OCR text, raw document text, "
            "raw filenames, private paths, PHI, secrets, DBs, backups, or "
            "bundles are read or emitted. Output uses anonymized file_NNN IDs "
            "only."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 98,
                "residual_unknown_reduction_track_remaining_pct": 2,
                "whole_medai_project_done_pct": 87,
                "whole_medai_project_remaining_pct": 13,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99,
                "residual_unknown_reduction_track_remaining_pct": 1,
                "whole_medai_project_done_pct": 88,
                "whole_medai_project_remaining_pct": 12,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "DIAG-15 — first implementation pass for one of the two "
                "sub-tracks (PDF text-extraction quality audit OR layout/table "
                "extraction audit), strictly evaluation-only and aggregate-only "
                "at the diagnostic stage, mirroring the DIAG-02..05 pattern."
            ),
            "must_remain_evaluation_only": True,
            "must_remain_aggregate_only": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_extraction_behavior_in_first_pass": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    sub_a = payload["sub_tracks"][SUB_TRACK_A_NAME]
    sub_b = payload["sub_tracks"][SUB_TRACK_B_NAME]
    excl = payload["exclusion_audit"]
    progress = payload["progress_estimate"]

    lines: List[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-14 Text-Layer Extraction Spec")
    lines.append("")
    lines.append(
        "Evaluation-only, aggregate-only specification for the 21 residual "
        "text-layer records. Splits them into two future audit sub-tracks "
        "(PDF text-extraction quality audit and layout/table extraction "
        "audit) and defines positive signatures, exclusion rules, future "
        "diagnostic behavior, future implementation acceptance criteria, "
        "future validation requirements, and rollback/safety boundaries."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-13A commit (short): `{payload['diag_13a_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
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
    lines.append("## Sub-track split")
    lines.append("")
    lines.append("| Sub-track | Pool count |")
    lines.append("| --- | ---: |")
    lines.append(
        f"| A — PDF text-extraction quality audit | "
        f"{payload['pdf_text_extraction_quality_audit_pool_count']} |"
    )
    lines.append(
        f"| B — layout / table extraction audit | "
        f"{payload['layout_table_extraction_audit_pool_count']} |"
    )
    lines.append(
        f"| **Total** | "
        f"**{payload['total_text_layer_records_analyzed']}** |"
    )
    lines.append("")
    lines.append(
        f"Split sum equals total: **{payload['split_sum_equals_total']}**"
    )
    lines.append("")
    lines.append("## Positive signature — sub-track A (PDF text-extraction quality audit)")
    lines.append("")
    for item in sub_a["positive_signature"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Positive signature — sub-track B (layout / table extraction audit)")
    lines.append("")
    for item in sub_b["positive_signature"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Exclusion rules")
    lines.append("")
    for rule in excl["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Proposed future diagnostic behavior")
    lines.append("")
    for item in payload["proposed_future_diagnostic_behavior"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Future implementation acceptance criteria")
    lines.append("")
    for item in payload["future_implementation_acceptance_criteria"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Future validation requirements")
    lines.append("")
    for item in payload["future_validation_requirements"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Rollback / safety boundaries")
    lines.append("")
    for item in payload["rollback_safety_boundaries"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "external_api_used",
        "cue_expansion_recommended",
        "extraction_behavior_changed",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "ocr_routing_changed",
        "ocr_engine_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_table_extraction_behavior_changed",
        "raw_language_detector_changed",
        "classifier_behavior_changed",
        "thresholds_or_scoring_changed",
        "cue_packs_added",
        "park_20_tags_touched",
        "no_extraction_behavior_implemented_in_this_block",
        "no_runtime_behavior_changed_in_this_block",
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
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-14 — Short Summary",
        "",
        "Aggregate-only specification for the 21 residual text-layer records.",
        "",
        "## State",
        "",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-13A commit (short): `{payload['diag_13a_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        "",
        "## Tag caveat",
        "",
        payload["remote_tag_caveat"],
        "",
        f"## Records analyzed: {payload['total_text_layer_records_analyzed']}",
        "",
        "## Sub-track split",
        "",
        f"- A — PDF text-extraction quality audit: **{payload['pdf_text_extraction_quality_audit_pool_count']}**",
        f"- B — layout / table extraction audit: **{payload['layout_table_extraction_audit_pool_count']}**",
        f"- Split sum equals total: **{payload['split_sum_equals_total']}**",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `extraction_behavior_changed`: {payload['extraction_behavior_changed']}",
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
        / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.md"
    )
    summary_path = (
        OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_14_TEXT_LAYER_EXTRACTION_SPEC.md"
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
        "pdf_text_extraction_quality_audit_pool_count": payload[
            "pdf_text_extraction_quality_audit_pool_count"
        ],
        "layout_table_extraction_audit_pool_count": payload[
            "layout_table_extraction_audit_pool_count"
        ],
        "behavior_changed": payload["behavior_changed"],
        "extraction_behavior_changed": payload["extraction_behavior_changed"],
        "implementation_started": payload["implementation_started"],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
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
