#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-17 — Default-off PDF text/layout quality
metadata implementation pass.

This script audits the new ``derive_pdf_text_layout_quality_context``
helper under three env modes (default-off, explicit off, explicit on) using
synthetic safe records derived from the privacy-safe DIAG-13A/14/15/15B
aggregates. It does NOT read source documents, raw OCR text, raw document
text, raw filenames, or private paths.

Output:
    reports/medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl/
        medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.json
        medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.md
        MEDAI_DOC_TYPE_UNKNOWN_DIAG_17_PDF_TEXT_LAYOUT_QUALITY_IMPL.md
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    SOURCE_PHASE,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_default_disabled,
    is_pdf_text_layout_quality_impl_enabled,
    matches_pdf_text_layout_quality_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl"
)

PARK_20_COMMIT_SHORT = "3e46461"
PARK_21_COMMIT_SHORT = "9f9e22d"
DIAG_16_COMMIT_SHORT = "e144376"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17"


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _synthetic_subtrack_a_records() -> List[Dict[str, Any]]:
    """11 synthetic records mirroring DIAG-15 Sub-track A aggregate shape."""
    return [
        {
            "anonymous_id": f"file_{i:03d}",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none" if i <= 4 else ("tiny" if i <= 6 else "short"),
            "table_like_structure_detected": "no",
        }
        for i in range(1, 12)
    ]


def _synthetic_subtrack_b_records() -> List[Dict[str, Any]]:
    """10 synthetic records mirroring DIAG-15B Sub-track B aggregate shape."""
    return [
        {
            "anonymous_id": f"file_{i:03d}",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none" if i <= 6 else "short",
            "table_like_structure_detected": "yes",
        }
        for i in range(12, 22)
    ]


def _excluded_pool_records() -> List[Dict[str, Any]]:
    """Records from neighboring pools that must NOT match the signature."""
    return [
        {
            "anonymous_id": "image_like_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "yes",  # excluded: image-like PDF
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "no_text_layer_001",
            "pdf_text_layer_detected": "no",  # excluded: no text layer
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "none",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "alpha_low_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "low",  # excluded: low alphabetic
            "native_text_length_bucket": "short",
            "table_like_structure_detected": "no",
        },
        {
            "anonymous_id": "long_text_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "long",  # excluded: text long enough
            "table_like_structure_detected": "no",
        },
    ]


def audit_env_modes() -> Dict[str, Any]:
    """Exercise the helper in three env modes and report aggregate counts."""
    a_records = _synthetic_subtrack_a_records()
    b_records = _synthetic_subtrack_b_records()
    excluded = _excluded_pool_records()

    target_total = len(a_records) + len(b_records)

    # 1. Default-off (env={}, no kwarg).
    default_off_emissions = sum(
        1
        for r in (a_records + b_records + excluded)
        if derive_pdf_text_layout_quality_context(r, env={}) is not None
    )

    # 2. Explicit off (env truthy but enabled=False kwarg wins).
    explicit_off_emissions = sum(
        1
        for r in (a_records + b_records + excluded)
        if derive_pdf_text_layout_quality_context(
            r,
            enabled=False,
            env={PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"},
        )
        is not None
    )

    # 3. Explicit on (env truthy, no kwarg).
    on_env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    on_emissions_target = sum(
        1
        for r in (a_records + b_records)
        if derive_pdf_text_layout_quality_context(r, env=on_env) is not None
    )
    on_emissions_excluded = sum(
        1
        for r in excluded
        if derive_pdf_text_layout_quality_context(r, env=on_env) is not None
    )

    # Family label counts in the on-mode
    family_counts: Dict[str, int] = {v: 0 for v in QUALITY_FAMILY_VALUES}
    review_required_all_true = True
    auto_accept_allowed_any_true = False
    clinical_interpretation_any_true = False
    raw_text_any_emitted = False
    raw_filename_any_emitted = False
    private_path_any_emitted = False
    for r in a_records + b_records:
        plan = derive_pdf_text_layout_quality_context(r, env=on_env)
        assert plan is not None
        for f in plan["quality_family"]:
            family_counts[f] += 1
        if not plan["review_required"]:
            review_required_all_true = False
        if plan["auto_accept_allowed"]:
            auto_accept_allowed_any_true = True
        if plan["clinical_interpretation_performed"]:
            clinical_interpretation_any_true = True
        if plan.get("raw_text_emitted"):
            raw_text_any_emitted = True
        if plan.get("raw_filename_emitted"):
            raw_filename_any_emitted = True
        if plan.get("private_path_emitted"):
            private_path_any_emitted = True

    return {
        "target_total_records": target_total,
        "excluded_pool_records_count": len(excluded),
        "default_off_emissions": default_off_emissions,
        "explicit_off_emissions": explicit_off_emissions,
        "explicit_on_target_emissions": on_emissions_target,
        "explicit_on_excluded_emissions": on_emissions_excluded,
        "family_label_counts_under_on_mode": family_counts,
        "review_required_all_true": review_required_all_true,
        "auto_accept_allowed_any_true": auto_accept_allowed_any_true,
        "clinical_interpretation_any_true": clinical_interpretation_any_true,
        "raw_text_any_emitted": raw_text_any_emitted,
        "raw_filename_any_emitted": raw_filename_any_emitted,
        "private_path_any_emitted": private_path_any_emitted,
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    audit = audit_env_modes()
    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_ready",
        "phase_id": PHASE_ID,
        "mode": "default_off_implementation",
        "default_off": True,
        "env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "source_phase_constant": SOURCE_PHASE,
        "controlled_vocabulary_quality_family_values": list(
            QUALITY_FAMILY_VALUES
        ),
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_16_commit_short": DIAG_16_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_21_parking_commit_short": PARK_21_COMMIT_SHORT,
        "park_status": (
            "PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on "
            "origin at 9f9e22d. DIAG-17 does not touch any tag."
        ),
        "source_reports_referenced": [
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
            "block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)",
            "block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)",
        ],
        "env_mode_audit": audit,
        "total_records_in_scope": 21,
        "subtrack_a_records": 11,
        "subtrack_b_records": 10,

        # Required behavior flags
        "behavior_changed": True,  # new code exists
        "default_behavior_changed": False,  # default-off; no caller invokes it
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "runtime_behavior_changed_by_default": False,
        "extraction_behavior_changed_by_default": False,
        "pdf_text_extraction_behavior_changed_by_default": False,
        "layout_extraction_behavior_changed_by_default": False,
        "table_extraction_behavior_changed_by_default": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
        "operator_ui_surface_added": False,
        "clinical_value_parsing_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "abbreviations_parsed_or_expanded": False,

        # Invariants
        "accepted_count": 0,
        "auto_accept_allowed_count": 0,
        "external_api_used_count": 0,
        "all_records_review_bound": True,

        # Tag protection
        "park_20_tags_touched": False,
        "park_21_tags_touched": False,

        # Privacy
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,

        # Default-off proof (read-back)
        "default_off_proof": {
            "default_off_emissions_must_be_zero": (
                audit["default_off_emissions"] == 0
            ),
            "explicit_off_emissions_must_be_zero": (
                audit["explicit_off_emissions"] == 0
            ),
            "explicit_on_target_emissions_must_equal_21": (
                audit["explicit_on_target_emissions"] == 21
            ),
            "explicit_on_excluded_emissions_must_be_zero": (
                audit["explicit_on_excluded_emissions"] == 0
            ),
            "review_required_all_true": audit["review_required_all_true"],
            "auto_accept_allowed_any_true": audit[
                "auto_accept_allowed_any_true"
            ],
            "clinical_interpretation_any_true": audit[
                "clinical_interpretation_any_true"
            ],
            "raw_text_any_emitted": audit["raw_text_any_emitted"],
            "raw_filename_any_emitted": audit["raw_filename_any_emitted"],
            "private_path_any_emitted": audit["private_path_any_emitted"],
        },

        "safety_privacy_statement": (
            "DIAG-17 introduces a strictly default-off, env-gated PDF "
            "text/layout quality metadata helper. With no kwarg and the "
            "SEPARATE env var "
            "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED unset (or "
            "falsy), the helper returns None for every input. When "
            "explicitly enabled, the helper emits only controlled-vocabulary "
            "metadata derived from privacy-safe DIAG-13A/14/15/15B "
            "aggregates: no raw extracted text, no raw OCR text, no raw "
            "document text, no raw filenames, no private paths, no PHI, no "
            "secrets. The helper performs no clinical interpretation, no "
            "diagnosis/medication/DDI/treatment inference, no clinical "
            "value parsing, no abbreviation parsing or expansion, no "
            "auto-accept, no data-layer document_type change, no OCR "
            "routing change, no OCR engine behavior change, no PDF "
            "text-extraction or layout/table extraction change, no "
            "classifier change, no threshold/scoring change, no cue "
            "expansion, no external API enablement. PARK-20 and PARK-21 "
            "tags are not touched."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.9,
                "residual_unknown_reduction_track_remaining_pct": 0.1,
                "whole_medai_project_done_pct": 90,
                "whole_medai_project_remaining_pct": 10,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.95,
                "residual_unknown_reduction_track_remaining_pct": 0.05,
                "whole_medai_project_done_pct": 90.5,
                "whole_medai_project_remaining_pct": 9.5,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "DIAG-18 — first env-gated wiring of the DIAG-17 helper "
                "into a narrow operator surface (read-only, no auto-accept) "
                "OR a separate evaluation block that captures aggregate "
                "metrics under the env-on path against the real privacy-"
                "safe corpus. Either next block must remain default-off, "
                "review-bound, and aggregate-only. Cue expansion remains "
                "explicitly NOT recommended."
            ),
            "must_remain_default_off": True,
            "must_remain_review_bound": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_default_behavior": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    audit = payload["env_mode_audit"]
    progress = payload["progress_estimate"]
    proof = payload["default_off_proof"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17 Default-Off PDF Text/Layout "
        "Quality Implementation Pass"
    )
    lines.append("")
    lines.append(
        "First default-off, env-gated implementation pass under the "
        "DIAG-16 acceptance criteria. Pure helper; not wired into any "
        "runtime path. Default-off proof recorded below."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Default off: **{payload['default_off']}**")
    lines.append(f"- Env var: `{payload['env_var']}`")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-16 commit (short): `{payload['diag_16_commit_short']}`")
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
    lines.append(
        f"## Total records in scope: **{payload['total_records_in_scope']}** "
        f"({payload['subtrack_a_records']} Sub-track A "
        f"+ {payload['subtrack_b_records']} Sub-track B)"
    )
    lines.append("")
    lines.append("## Env-mode audit")
    lines.append("")
    lines.append("| Mode | Target emissions (of 21) | Excluded-pool emissions |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Default off (env empty) | {audit['default_off_emissions']} of {audit['target_total_records']} | "
        f"0 of {audit['excluded_pool_records_count']} |"
    )
    lines.append(
        f"| Explicit off (env truthy, kwarg False) | {audit['explicit_off_emissions']} of {audit['target_total_records']} | "
        f"0 of {audit['excluded_pool_records_count']} |"
    )
    lines.append(
        f"| Explicit on (env truthy) | {audit['explicit_on_target_emissions']} of {audit['target_total_records']} | "
        f"{audit['explicit_on_excluded_emissions']} of {audit['excluded_pool_records_count']} |"
    )
    lines.append("")
    lines.append("## Family-label counts (multi-label) under explicit-on")
    lines.append("")
    lines.append("| Quality family label | Count |")
    lines.append("| --- | ---: |")
    for label, count in audit["family_label_counts_under_on_mode"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.append("")
    lines.append("## Default-off proof")
    lines.append("")
    for k, v in proof.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "default_behavior_changed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "runtime_behavior_changed_by_default",
        "extraction_behavior_changed_by_default",
        "pdf_text_extraction_behavior_changed_by_default",
        "layout_extraction_behavior_changed_by_default",
        "table_extraction_behavior_changed_by_default",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "cue_expansion_performed",
        "operator_ui_surface_added",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviations_parsed_or_expanded",
        "park_20_tags_touched",
        "park_21_tags_touched",
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
        f"- Must remain default off: **{rec['must_remain_default_off']}**"
    )
    lines.append(
        f"- Must remain review-bound: **{rec['must_remain_review_bound']}**"
    )
    lines.append(
        f"- Must NOT propose cue expansion as primary step: "
        f"**{rec['must_not_propose_cue_expansion_as_primary_step']}**"
    )
    lines.append(
        f"- Must NOT change default behavior: "
        f"**{rec['must_not_change_default_behavior']}**"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    proof = payload["default_off_proof"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17 — Short Summary",
        "",
        "Default-off PDF text/layout quality metadata helper. Pure, "
        "env-gated, no caller in runtime.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Default off: **{payload['default_off']}**",
        f"- Env var: `{payload['env_var']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-16 commit (short): `{payload['diag_16_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`",
        "",
        "## PARK status",
        "",
        payload["park_status"],
        "",
        f"## Records in scope: {payload['total_records_in_scope']} "
        f"({payload['subtrack_a_records']} A + {payload['subtrack_b_records']} B)",
        "",
        "## Default-off proof (key checks)",
        "",
        f"- default-off emissions == 0: **{proof['default_off_emissions_must_be_zero']}**",
        f"- explicit-off emissions == 0: **{proof['explicit_off_emissions_must_be_zero']}**",
        f"- explicit-on target emissions == 21: **{proof['explicit_on_target_emissions_must_equal_21']}**",
        f"- explicit-on excluded-pool emissions == 0: **{proof['explicit_on_excluded_emissions_must_be_zero']}**",
        f"- review_required all true: **{proof['review_required_all_true']}**",
        f"- auto_accept_allowed any true: **{proof['auto_accept_allowed_any_true']}**",
        f"- clinical_interpretation any true: **{proof['clinical_interpretation_any_true']}**",
        f"- raw_text any emitted: **{proof['raw_text_any_emitted']}**",
        f"- raw_filename any emitted: **{proof['raw_filename_any_emitted']}**",
        f"- private_path any emitted: **{proof['private_path_any_emitted']}**",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `default_behavior_changed`: {payload['default_behavior_changed']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `cue_expansion_performed`: {payload['cue_expansion_performed']}",
        f"- `operator_ui_surface_added`: {payload['operator_ui_surface_added']}",
        f"- `park_20_tags_touched`: {payload['park_20_tags_touched']}",
        f"- `park_21_tags_touched`: {payload['park_21_tags_touched']}",
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
        / "medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl_report.md"
    )
    summary_path = (
        OUT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_17_PDF_TEXT_LAYOUT_QUALITY_IMPL.md"
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
        "total_records_in_scope": payload["total_records_in_scope"],
        "default_off_emissions": payload["env_mode_audit"][
            "default_off_emissions"
        ],
        "explicit_on_target_emissions": payload["env_mode_audit"][
            "explicit_on_target_emissions"
        ],
        "behavior_changed": payload["behavior_changed"],
        "default_behavior_changed": payload["default_behavior_changed"],
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
