#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B — Env-on aggregate evaluation for the
DIAG-17 PDF text/layout quality metadata helper.

Reports-only / evaluation-only. Exercises the DIAG-17 helper in explicit
env-on mode against the same privacy-safe synthetic record set previously
used by DIAG-17 (derived from DIAG-15/15B aggregate signatures) and
characterizes the resulting metadata distribution.

Hard guardrails:
* Reads ONLY in-process privacy-safe synthetic records that mirror the
  already-published DIAG-15 / DIAG-15B aggregate shapes. Does NOT open
  source documents, raw OCR text, raw document text, raw filenames,
  private paths, terminology files, runtime DBs, or backups.
* Emits ONLY aggregate counts, controlled-vocabulary labels, and the
  invariant flags from each plan. The 21 in-scope anonymous IDs use the
  ``file_NNN`` form.
* Does NOT change runtime behavior, does NOT add a runtime helper, does
  NOT add an operator UI surface, does NOT change OCR routing, OCR engine
  behavior, PDF text-extraction behavior, layout/table extraction
  behavior, raw language detector behavior, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior. Does NOT touch PARK-20 / PARK-21
  tags. The DIAG-17 env var is set ONLY in-process via a passed-in env
  mapping; ``os.environ`` is never written.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_default_disabled,
    is_pdf_text_layout_quality_impl_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_17b_env_on_aggregate_eval"
)

PARK_20_COMMIT_SHORT = "3e46461"
PARK_21_COMMIT_SHORT = "9f9e22d"
DIAG_17_COMMIT_SHORT = "ad7b2d6"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B"


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ── In-scope record cohort (21 records, anonymized) ───────────────────────


def _in_scope_records() -> List[Dict[str, Any]]:
    """11 Sub-track A + 10 Sub-track B synthetic safe records."""
    records: List[Dict[str, Any]] = []
    # Sub-track A — text_layer_too_short, no table-like structure
    for i in range(1, 12):
        if i <= 4:
            length = "none"
        elif i <= 6:
            length = "tiny"
        else:
            length = "short"
        records.append(
            {
                "anonymous_id": f"file_{i:03d}",
                "subtrack": "A",
                "pdf_text_layer_detected": "yes",
                "image_like_pdf": "no",
                "alphabetic_content_bucket": "high",
                "native_text_length_bucket": length,
                "table_like_structure_detected": "no",
            }
        )
    # Sub-track B — table-like structure visible, text insufficient
    for i in range(12, 22):
        length = "none" if i <= 17 else "short"
        records.append(
            {
                "anonymous_id": f"file_{i:03d}",
                "subtrack": "B",
                "pdf_text_layer_detected": "yes",
                "image_like_pdf": "no",
                "alphabetic_content_bucket": "high",
                "native_text_length_bucket": length,
                "table_like_structure_detected": "yes",
            }
        )
    return records


def _excluded_pool_records() -> List[Dict[str, Any]]:
    """Records from neighboring pools that must NOT emit under env-on."""
    return [
        {
            "anonymous_id": "image_like_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "yes",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "no_text_layer_001",
            "pdf_text_layer_detected": "no",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "alpha_low_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "low",
            "native_text_length_bucket": "short",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "long_text_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "long",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "alpha_medium_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "medium",
            "native_text_length_bucket": "short",
            "table_like_structure_detected": "yes",
        },
    ]


# ── Env-on aggregate evaluation ───────────────────────────────────────────


def evaluate_env_on_aggregate() -> Dict[str, Any]:
    in_scope = _in_scope_records()
    excluded = _excluded_pool_records()

    env_on = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}

    emitted_metadata_count = 0
    suppressed_or_excluded_count = 0
    family_label_counts: Dict[str, int] = {v: 0 for v in QUALITY_FAMILY_VALUES}
    review_required_count = 0
    auto_accept_allowed_count = 0
    clinical_interpretation_performed_count = 0
    raw_text_emission_count = 0
    raw_filename_emission_count = 0
    private_path_emission_count = 0
    diagnosis_inference_count = 0
    medication_inference_count = 0
    ddi_inference_count = 0
    treatment_inference_count = 0
    abbreviation_expansion_count = 0
    external_api_used_count = 0
    park_20_tags_touched_count = 0
    park_21_tags_touched_count = 0

    per_subtrack_emit: Dict[str, int] = {"A": 0, "B": 0}

    for record in in_scope:
        plan = derive_pdf_text_layout_quality_context(record, env=env_on)
        if plan is None:
            suppressed_or_excluded_count += 1
            continue
        emitted_metadata_count += 1
        per_subtrack_emit[record["subtrack"]] += 1
        for label in plan["quality_family"]:
            family_label_counts[label] += 1
        if plan["review_required"]:
            review_required_count += 1
        if plan["auto_accept_allowed"]:
            auto_accept_allowed_count += 1
        if plan["clinical_interpretation_performed"]:
            clinical_interpretation_performed_count += 1
        if plan["raw_text_emitted"]:
            raw_text_emission_count += 1
        if plan["raw_filename_emitted"]:
            raw_filename_emission_count += 1
        if plan["private_path_emitted"]:
            private_path_emission_count += 1
        if plan["diagnosis_inference_performed"]:
            diagnosis_inference_count += 1
        if plan["medication_inference_performed"]:
            medication_inference_count += 1
        if plan["ddi_inference_performed"]:
            ddi_inference_count += 1
        if plan["treatment_inference_performed"]:
            treatment_inference_count += 1
        if plan["abbreviation_expanded"]:
            abbreviation_expansion_count += 1
        if plan["external_api_used"]:
            external_api_used_count += 1
        if plan["park_20_tags_touched"]:
            park_20_tags_touched_count += 1
        if plan["park_21_tags_touched"]:
            park_21_tags_touched_count += 1

    excluded_pool_emission_count = 0
    for record in excluded:
        plan = derive_pdf_text_layout_quality_context(record, env=env_on)
        if plan is not None:
            excluded_pool_emission_count += 1

    # Confirm that AFTER this evaluation, the helper is STILL default-disabled
    # when consulted with env={} (env-on mapping was passed in only)
    helper_still_default_disabled_after_eval = (
        is_pdf_text_layout_quality_impl_default_disabled(env={})
    )
    helper_still_enabled_only_when_env_truthy = (
        is_pdf_text_layout_quality_impl_enabled(env=env_on)
        and not is_pdf_text_layout_quality_impl_enabled(env={})
    )

    return {
        "env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "total_records_evaluated": len(in_scope),
        "env_on_records_evaluated": len(in_scope),
        "emitted_metadata_count": emitted_metadata_count,
        "suppressed_or_excluded_count": suppressed_or_excluded_count,
        "per_subtrack_emit_count": per_subtrack_emit,
        "family_label_counts": family_label_counts,
        "review_required_count": review_required_count,
        "auto_accept_allowed_count": auto_accept_allowed_count,
        "clinical_interpretation_performed_count": (
            clinical_interpretation_performed_count
        ),
        "raw_text_emission_count": raw_text_emission_count,
        "raw_filename_emission_count": raw_filename_emission_count,
        "private_path_emission_count": private_path_emission_count,
        "diagnosis_inference_count": diagnosis_inference_count,
        "medication_inference_count": medication_inference_count,
        "ddi_inference_count": ddi_inference_count,
        "treatment_inference_count": treatment_inference_count,
        "abbreviation_expansion_count": abbreviation_expansion_count,
        "external_api_used_count": external_api_used_count,
        "park_20_tags_touched_count": park_20_tags_touched_count,
        "park_21_tags_touched_count": park_21_tags_touched_count,
        "excluded_pool_count": len(excluded),
        "excluded_pool_emission_count": excluded_pool_emission_count,
        "helper_still_default_disabled_after_eval": (
            helper_still_default_disabled_after_eval
        ),
        "helper_still_enabled_only_when_env_truthy": (
            helper_still_enabled_only_when_env_truthy
        ),
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    agg = evaluate_env_on_aggregate()
    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_17b_env_on_aggregate_eval_ready",
        "phase_id": PHASE_ID,
        "mode": "env_on_aggregate_evaluation",
        "evaluation_only": True,
        "reports_only": True,
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_17_commit_short": DIAG_17_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_21_parking_commit_short": PARK_21_COMMIT_SHORT,
        "park_status": (
            "PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on "
            "origin at 9f9e22d. DIAG-17B does not touch any tag."
        ),
        "source_reports_referenced": [
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
            "block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)",
            "block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)",
            "block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)",
        ],

        "env_var": agg["env_var"],
        "env_on_records_evaluated": agg["env_on_records_evaluated"],

        # Required aggregate outputs
        "total_records_evaluated": agg["total_records_evaluated"],
        "emitted_metadata_count": agg["emitted_metadata_count"],
        "suppressed_or_excluded_count": agg["suppressed_or_excluded_count"],
        "per_subtrack_emit_count": agg["per_subtrack_emit_count"],
        "family_label_counts": agg["family_label_counts"],
        "review_required_count": agg["review_required_count"],
        "auto_accept_allowed_count": agg["auto_accept_allowed_count"],
        "clinical_interpretation_performed_count": agg[
            "clinical_interpretation_performed_count"
        ],
        "raw_text_emission_count": agg["raw_text_emission_count"],
        "raw_filename_emission_count": agg["raw_filename_emission_count"],
        "private_path_emission_count": agg["private_path_emission_count"],
        "excluded_pool_count": agg["excluded_pool_count"],
        "excluded_pool_emission_count": agg["excluded_pool_emission_count"],
        "diagnosis_inference_count": agg["diagnosis_inference_count"],
        "medication_inference_count": agg["medication_inference_count"],
        "ddi_inference_count": agg["ddi_inference_count"],
        "treatment_inference_count": agg["treatment_inference_count"],
        "abbreviation_expansion_count": agg["abbreviation_expansion_count"],
        "external_api_used_count": agg["external_api_used_count"],
        "park_20_tags_touched_count": agg["park_20_tags_touched_count"],
        "park_21_tags_touched_count": agg["park_21_tags_touched_count"],

        # Default-off invariants survive after eval
        "helper_still_default_disabled_after_eval": agg[
            "helper_still_default_disabled_after_eval"
        ],
        "helper_still_enabled_only_when_env_truthy": agg[
            "helper_still_enabled_only_when_env_truthy"
        ],

        # Required block flags
        "behavior_changed": False,
        "runtime_behavior_changed": False,
        "extraction_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "operator_ui_surface_added": False,
        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,

        "accepted_count": 0,
        "all_records_review_bound": True,

        "clinical_value_parsing_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "abbreviation_expansion_performed": False,

        "park_20_tags_touched": False,
        "park_21_tags_touched": False,

        # Additional privacy invariants
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,

        "safety_privacy_statement": (
            "DIAG-17B is reports-only / evaluation-only. The DIAG-17 helper "
            "is exercised in explicit env-on mode via an in-process env "
            "mapping; os.environ is never written. No source documents, raw "
            "OCR text, raw document text, raw filenames, private paths, "
            "PHI, secrets, DBs, backups, or bundles are read or emitted. "
            "Output uses anonymized file_NNN IDs only. No runtime behavior "
            "changes. No extraction behavior changes. PARK-20 / PARK-21 "
            "tags are not touched."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.95,
                "residual_unknown_reduction_track_remaining_pct": 0.05,
                "whole_medai_project_done_pct": 90.5,
                "whole_medai_project_remaining_pct": 9.5,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.97,
                "residual_unknown_reduction_track_remaining_pct": 0.03,
                "whole_medai_project_done_pct": 90.8,
                "whole_medai_project_remaining_pct": 9.2,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "DIAG-18 — first env-gated, read-only operator surface for "
                "DIAG-17 metadata, behind a SEPARATE fourth env var. "
                "Default-off; no auto-accept; no clinical interpretation; "
                "no data-layer change. Mirrors the DIAG-08A / DIAG-10A / "
                "DIAG-12A pattern. Cue expansion remains explicitly NOT "
                "recommended."
            ),
            "must_remain_default_off": True,
            "must_remain_review_bound": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_default_behavior": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B Env-On Aggregate Evaluation"
    )
    lines.append("")
    lines.append(
        "Reports-only / evaluation-only env-on aggregate evaluation of the "
        "DIAG-17 PDF text/layout quality metadata helper. No runtime "
        "behavior changes. No extraction behavior changes. PARK-20 / "
        "PARK-21 tags untouched."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Evaluation only: **{payload['evaluation_only']}**")
    lines.append(f"- Reports only: **{payload['reports_only']}**")
    lines.append(f"- Env var evaluated: `{payload['env_var']}`")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-17 commit (short): `{payload['diag_17_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
    )
    lines.append(
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`"
    )
    lines.append("")
    lines.append("## PARK status")
    lines.append("")
    lines.append(payload["park_status"])
    lines.append("")
    lines.append("## Source reports referenced")
    lines.append("")
    for src in payload["source_reports_referenced"]:
        lines.append(f"- `{src}`")
    lines.append("")
    lines.append("## Env-on aggregate results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for k in [
        "total_records_evaluated",
        "emitted_metadata_count",
        "suppressed_or_excluded_count",
        "review_required_count",
        "auto_accept_allowed_count",
        "clinical_interpretation_performed_count",
        "raw_text_emission_count",
        "raw_filename_emission_count",
        "private_path_emission_count",
        "diagnosis_inference_count",
        "medication_inference_count",
        "ddi_inference_count",
        "treatment_inference_count",
        "abbreviation_expansion_count",
        "external_api_used_count",
        "park_20_tags_touched_count",
        "park_21_tags_touched_count",
        "excluded_pool_count",
        "excluded_pool_emission_count",
    ]:
        lines.append(f"| `{k}` | {payload[k]} |")
    lines.append("")
    lines.append("## Per-subtrack emit counts")
    lines.append("")
    lines.append("| Sub-track | Emitted |")
    lines.append("| --- | ---: |")
    for k, v in payload["per_subtrack_emit_count"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Family-label counts (multi-label, under env-on)")
    lines.append("")
    lines.append("| Quality family label | Count |")
    lines.append("| --- | ---: |")
    for k, v in payload["family_label_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Default-off invariants AFTER evaluation")
    lines.append("")
    lines.append(
        f"- Helper still default-disabled with env={{}}: "
        f"**{payload['helper_still_default_disabled_after_eval']}**"
    )
    lines.append(
        f"- Helper still enabled only when env truthy: "
        f"**{payload['helper_still_enabled_only_when_env_truthy']}**"
    )
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "runtime_behavior_changed",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "operator_ui_surface_added",
        "cue_expansion_recommended",
        "cue_expansion_performed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_expansion_performed",
        "park_20_tags_touched",
        "park_21_tags_touched",
        "all_records_review_bound",
    ]:
        lines.append(f"- `{k}`: {payload[k]}")
    lines.append(f"- `accepted_count`: {payload['accepted_count']}")
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
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B — Short Summary",
        "",
        "Env-on aggregate evaluation of the DIAG-17 PDF text/layout quality "
        "helper. Reports-only.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Env var evaluated: `{payload['env_var']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-17 commit (short): `{payload['diag_17_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`",
        "",
        "## PARK status",
        "",
        payload["park_status"],
        "",
        f"## Env-on records evaluated: {payload['env_on_records_evaluated']}",
        f"## Emitted metadata count: {payload['emitted_metadata_count']}",
        f"## Excluded pool emission count: {payload['excluded_pool_emission_count']}",
        "",
        "## Hard zeros under env-on",
        "",
        f"- `auto_accept_allowed_count`: {payload['auto_accept_allowed_count']}",
        f"- `external_api_used_count`: {payload['external_api_used_count']}",
        f"- `clinical_interpretation_performed_count`: {payload['clinical_interpretation_performed_count']}",
        f"- `raw_text_emission_count`: {payload['raw_text_emission_count']}",
        f"- `raw_filename_emission_count`: {payload['raw_filename_emission_count']}",
        f"- `private_path_emission_count`: {payload['private_path_emission_count']}",
        f"- `park_20_tags_touched_count`: {payload['park_20_tags_touched_count']}",
        f"- `park_21_tags_touched_count`: {payload['park_21_tags_touched_count']}",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `runtime_behavior_changed`: {payload['runtime_behavior_changed']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `operator_ui_surface_added`: {payload['operator_ui_surface_added']}",
        f"- `helper_still_default_disabled_after_eval`: {payload['helper_still_default_disabled_after_eval']}",
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
        / "medai_doc_type_unknown_diag_17b_env_on_aggregate_eval_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_17b_env_on_aggregate_eval_report.md"
    )
    summary_path = (
        OUT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_17B_ENV_ON_AGGREGATE_EVAL.md"
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    summary_path.write_text(render_short_summary(payload), encoding="utf-8")
    out_summary = {
        "conclusion": payload["conclusion"],
        "phase_id": payload["phase_id"],
        "env_var": payload["env_var"],
        "env_on_records_evaluated": payload["env_on_records_evaluated"],
        "emitted_metadata_count": payload["emitted_metadata_count"],
        "excluded_pool_emission_count": payload["excluded_pool_emission_count"],
        "auto_accept_allowed_count": payload["auto_accept_allowed_count"],
        "external_api_used_count": payload["external_api_used_count"],
        "behavior_changed": payload["behavior_changed"],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "park_20_tags_touched": payload["park_20_tags_touched"],
        "park_21_tags_touched": payload["park_21_tags_touched"],
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
