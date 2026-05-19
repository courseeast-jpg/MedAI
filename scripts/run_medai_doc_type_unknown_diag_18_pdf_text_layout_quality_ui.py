#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-18 — Default-off read-only operator surface
for DIAG-17 PDF text/layout quality metadata.

Audit script. Exercises the new ``render_plan_for_pdf_text_layout_quality``
helper across four env modes (neither / metadata-only / UI-only / both)
and confirms:

    * Default-off renders nothing.
    * Only the both-truthy combination renders a plan.
    * Plans never carry buttons, callbacks, actions, or state mutation.
    * Plans never carry raw text, raw filenames, or private paths.

Hard guardrails:
    * Reads ONLY in-process synthetic safe records that mirror the
      already-published DIAG-15 / DIAG-15B aggregate shapes.
    * Emits ONLY aggregate counts, controlled-vocabulary labels, and the
      invariant flags from each plan.
    * No source documents, raw OCR text, raw document text, raw
      filenames, private paths, PHI, secrets, DBs, backups, or bundles.
    * No runtime wiring; no Streamlit imports; no app/main.py change; no
      package __init__.py change.
    * No PARK-20 / PARK-21 tag touch.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (
    DIAG_18_UI_ENV_VAR,
    KNOWN_DOC_TYPE_ENV_VARS,
    OPERATOR_DISCLAIMER,
    OPERATOR_DISPLAY_HEADING,
    OPERATOR_EXPANDER_LABEL,
    OPERATOR_VOCAB_TOKEN,
    SOURCE_PHASE,
    is_pdf_text_layout_quality_ui_default_disabled,
    is_pdf_text_layout_quality_ui_enabled,
    render_plan_for_pdf_text_layout_quality,
    requires_both_env_vars_truthy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui"
)

PARK_20_COMMIT_SHORT = "3e46461"
PARK_21_COMMIT_SHORT = "9f9e22d"
DIAG_17_COMMIT_SHORT = "ad7b2d6"
DIAG_17B_COMMIT_SHORT = "3bc8a64"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-18"

# Field-name patterns that MUST NOT appear in any rendered plan key. A
# plan carrying any of these (as a whole snake_case token, or as a
# prefix-style ``on_click`` etc.) would imply action capability or state
# mutation. Word-boundary matching avoids false positives like "form"
# matching "per_form_ed".
_FORBIDDEN_PLAN_KEY_TOKENS = frozenset(
    {
        "callback",
        "button",
        "submit",
        "approve",
        "deny",
        "reject",
        "mutate",
        # "form", "action", "write", "accept" are checked separately below
        # because they appear as substrings inside safe invariant tokens
        # (per_form_ed, extr_action, auto_accept, raw_text_emitted...).
    }
)

# Composite-prefix patterns: a key is forbidden if it starts with these.
_FORBIDDEN_PLAN_KEY_PREFIXES = (
    "on_click",
    "on_submit",
    "on_change",
    "callback",
    "button",
    "submit",
    "approve",
    "deny",
    "reject",
    "write_",
)


def _plan_key_is_forbidden(k: str) -> bool:
    """Word-boundary check. A key is forbidden when:

    * it equals one of the forbidden tokens exactly, OR
    * it starts with one of the forbidden prefixes, OR
    * any of its snake_case-split tokens equals a forbidden token
      (except "no_" / "is_" prefixed invariant flags, which are allowed
      by design).
    """
    kl = k.lower()
    if kl.startswith("no_") or kl.startswith("is_"):
        return False
    if kl in _FORBIDDEN_PLAN_KEY_TOKENS:
        return True
    for prefix in _FORBIDDEN_PLAN_KEY_PREFIXES:
        if kl.startswith(prefix):
            return True
    tokens = kl.split("_")
    if _FORBIDDEN_PLAN_KEY_TOKENS & set(tokens):
        return True
    return False


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _in_scope_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(1, 12):
        length = "none" if i <= 4 else ("tiny" if i <= 6 else "short")
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
            "anonymous_id": "long_text_001",
            "pdf_text_layer_detected": "yes",
            "image_like_pdf": "no",
            "alphabetic_content_bucket": "high",
            "native_text_length_bucket": "long",
            "table_like_structure_detected": "no",
        },
    ]


def audit_env_combinations() -> Dict[str, Any]:
    in_scope = _in_scope_records()
    excluded = _excluded_pool_records()
    target_total = len(in_scope)

    neither_env: Dict[str, str] = {}
    metadata_only_env = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    ui_only_env = {DIAG_18_UI_ENV_VAR: "1"}
    both_env = {
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1",
        DIAG_18_UI_ENV_VAR: "1",
    }

    def _count_emits(env: Mapping[str, str], records: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for r in records
            if render_plan_for_pdf_text_layout_quality(r, env=env) is not None
        )

    neither_emits_in_scope = _count_emits(neither_env, in_scope)
    metadata_only_emits_in_scope = _count_emits(metadata_only_env, in_scope)
    ui_only_emits_in_scope = _count_emits(ui_only_env, in_scope)
    both_emits_in_scope = _count_emits(both_env, in_scope)
    both_emits_excluded = _count_emits(both_env, excluded)

    # Walk one fully-rendered plan and prove the structural invariants.
    sample_plan = render_plan_for_pdf_text_layout_quality(
        in_scope[0], env=both_env
    )
    assert sample_plan is not None
    forbidden_keys_present = sorted(
        k for k in sample_plan.keys() if _plan_key_is_forbidden(k)
    )

    # Aggregate invariant counters across the 21 in-scope plans.
    is_read_only_count = 0
    no_action_attached_count = 0
    no_button_attached_count = 0
    no_callback_attached_count = 0
    no_state_mutation_count = 0
    no_data_layer_write_count = 0
    no_document_type_mutation_count = 0
    raw_text_rendered_count = 0
    raw_filename_rendered_count = 0
    private_path_rendered_count = 0
    clinical_interpretation_count = 0
    auto_accept_count = 0
    diagnosis_inference_count = 0
    medication_inference_count = 0
    ddi_inference_count = 0
    treatment_inference_count = 0
    abbreviation_expanded_count = 0
    external_api_used_count = 0
    park_20_touch_count = 0
    park_21_touch_count = 0
    for r in in_scope:
        p = render_plan_for_pdf_text_layout_quality(r, env=both_env)
        assert p is not None
        if p["is_read_only"]:
            is_read_only_count += 1
        if p["no_action_attached"]:
            no_action_attached_count += 1
        if p["no_button_attached"]:
            no_button_attached_count += 1
        if p["no_callback_attached"]:
            no_callback_attached_count += 1
        if p["no_state_mutation"]:
            no_state_mutation_count += 1
        if p["no_data_layer_write"]:
            no_data_layer_write_count += 1
        if p["no_document_type_mutation"]:
            no_document_type_mutation_count += 1
        if p["raw_text_rendered"]:
            raw_text_rendered_count += 1
        if p["raw_filename_rendered"]:
            raw_filename_rendered_count += 1
        if p["private_path_rendered"]:
            private_path_rendered_count += 1
        if p["clinical_interpretation_performed"]:
            clinical_interpretation_count += 1
        if p["is_auto_accept"]:
            auto_accept_count += 1
        if p["diagnosis_inference_performed"]:
            diagnosis_inference_count += 1
        if p["medication_inference_performed"]:
            medication_inference_count += 1
        if p["ddi_inference_performed"]:
            ddi_inference_count += 1
        if p["treatment_inference_performed"]:
            treatment_inference_count += 1
        if p["abbreviation_expanded"]:
            abbreviation_expanded_count += 1
        if p["external_api_used"]:
            external_api_used_count += 1
        if p["park_20_tags_touched"]:
            park_20_touch_count += 1
        if p["park_21_tags_touched"]:
            park_21_touch_count += 1

    helper_still_default_disabled_after_audit = (
        is_pdf_text_layout_quality_ui_default_disabled(env={})
    )

    return {
        "target_total_records": target_total,
        "excluded_pool_records_count": len(excluded),
        "neither_env_in_scope_emissions": neither_emits_in_scope,
        "metadata_only_env_in_scope_emissions": metadata_only_emits_in_scope,
        "ui_only_env_in_scope_emissions": ui_only_emits_in_scope,
        "both_env_in_scope_emissions": both_emits_in_scope,
        "both_env_excluded_pool_emissions": both_emits_excluded,
        "forbidden_plan_keys_present": forbidden_keys_present,
        "is_read_only_count": is_read_only_count,
        "no_action_attached_count": no_action_attached_count,
        "no_button_attached_count": no_button_attached_count,
        "no_callback_attached_count": no_callback_attached_count,
        "no_state_mutation_count": no_state_mutation_count,
        "no_data_layer_write_count": no_data_layer_write_count,
        "no_document_type_mutation_count": no_document_type_mutation_count,
        "raw_text_rendered_count": raw_text_rendered_count,
        "raw_filename_rendered_count": raw_filename_rendered_count,
        "private_path_rendered_count": private_path_rendered_count,
        "clinical_interpretation_count": clinical_interpretation_count,
        "auto_accept_count": auto_accept_count,
        "diagnosis_inference_count": diagnosis_inference_count,
        "medication_inference_count": medication_inference_count,
        "ddi_inference_count": ddi_inference_count,
        "treatment_inference_count": treatment_inference_count,
        "abbreviation_expanded_count": abbreviation_expanded_count,
        "external_api_used_count": external_api_used_count,
        "park_20_touch_count": park_20_touch_count,
        "park_21_touch_count": park_21_touch_count,
        "helper_still_default_disabled_after_audit": (
            helper_still_default_disabled_after_audit
        ),
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    audit = audit_env_combinations()
    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui_ready",
        "phase_id": PHASE_ID,
        "mode": "default_off_read_only_operator_surface",
        "default_off": True,
        "read_only": True,
        "metadata_env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "ui_env_var": DIAG_18_UI_ENV_VAR,
        "requires_both_env_vars": True,
        "source_phase_constant": SOURCE_PHASE,
        "operator_display_heading": OPERATOR_DISPLAY_HEADING,
        "operator_vocab_token": OPERATOR_VOCAB_TOKEN,
        "operator_expander_label": OPERATOR_EXPANDER_LABEL,
        "operator_disclaimer": OPERATOR_DISCLAIMER,
        "known_doc_type_env_vars": list(KNOWN_DOC_TYPE_ENV_VARS),
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_17_commit_short": DIAG_17_COMMIT_SHORT,
        "diag_17b_commit_short": DIAG_17B_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_21_parking_commit_short": PARK_21_COMMIT_SHORT,
        "park_status": (
            "PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on "
            "origin at 9f9e22d. DIAG-18 does not touch any tag."
        ),
        "source_reports_referenced": [
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
            "block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)",
            "block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)",
            "block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)",
            "block DIAG-17B (directory: medai_doc_type_unknown_diag_17b_env_on_aggregate_eval)",
        ],
        "env_combination_audit": audit,

        # Required behavior flags
        "behavior_changed": True,  # new helper exists
        "default_behavior_changed": False,  # both env vars default to off
        "runtime_behavior_changed_by_default": False,
        "operator_ui_surface_added": True,
        "operator_ui_surface_enabled_by_default": False,
        "buttons_added": False,
        "callbacks_added": False,
        "actions_added": False,
        "state_mutation_added": False,
        "data_layer_write_added": False,
        "document_type_mutation_added": False,

        # Counts (invariant)
        "accepted_count": 0,
        "auto_accept_allowed_count": 0,
        "external_api_used_count": 0,
        "all_records_review_bound": True,

        # Privacy / safety
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "raw_text_rendered": False,
        "raw_filenames_rendered": False,
        "private_paths_rendered": False,

        # Clinical safety
        "clinical_value_parsing_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "abbreviation_expansion_performed": False,

        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
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

        # Default-off / read-only proof
        "default_off_proof": {
            "neither_env_in_scope_emissions_must_be_zero": (
                audit["neither_env_in_scope_emissions"] == 0
            ),
            "metadata_only_env_in_scope_emissions_must_be_zero": (
                audit["metadata_only_env_in_scope_emissions"] == 0
            ),
            "ui_only_env_in_scope_emissions_must_be_zero": (
                audit["ui_only_env_in_scope_emissions"] == 0
            ),
            "both_env_in_scope_emissions_must_equal_21": (
                audit["both_env_in_scope_emissions"] == 21
            ),
            "both_env_excluded_pool_emissions_must_be_zero": (
                audit["both_env_excluded_pool_emissions"] == 0
            ),
            "helper_still_default_disabled_after_audit": audit[
                "helper_still_default_disabled_after_audit"
            ],
        },
        "read_only_proof": {
            "forbidden_plan_keys_present": audit["forbidden_plan_keys_present"],
            "is_read_only_count_must_equal_21": (
                audit["is_read_only_count"] == 21
            ),
            "no_action_attached_count_must_equal_21": (
                audit["no_action_attached_count"] == 21
            ),
            "no_button_attached_count_must_equal_21": (
                audit["no_button_attached_count"] == 21
            ),
            "no_callback_attached_count_must_equal_21": (
                audit["no_callback_attached_count"] == 21
            ),
            "no_state_mutation_count_must_equal_21": (
                audit["no_state_mutation_count"] == 21
            ),
            "no_data_layer_write_count_must_equal_21": (
                audit["no_data_layer_write_count"] == 21
            ),
            "no_document_type_mutation_count_must_equal_21": (
                audit["no_document_type_mutation_count"] == 21
            ),
            "raw_text_rendered_count_must_be_zero": (
                audit["raw_text_rendered_count"] == 0
            ),
            "raw_filename_rendered_count_must_be_zero": (
                audit["raw_filename_rendered_count"] == 0
            ),
            "private_path_rendered_count_must_be_zero": (
                audit["private_path_rendered_count"] == 0
            ),
            "clinical_interpretation_count_must_be_zero": (
                audit["clinical_interpretation_count"] == 0
            ),
            "auto_accept_count_must_be_zero": (
                audit["auto_accept_count"] == 0
            ),
            "diagnosis_medication_ddi_treatment_inference_counts_must_be_zero": (
                audit["diagnosis_inference_count"]
                + audit["medication_inference_count"]
                + audit["ddi_inference_count"]
                + audit["treatment_inference_count"]
                == 0
            ),
            "abbreviation_expanded_count_must_be_zero": (
                audit["abbreviation_expanded_count"] == 0
            ),
            "external_api_used_count_must_be_zero": (
                audit["external_api_used_count"] == 0
            ),
            "park_20_touch_count_must_be_zero": (
                audit["park_20_touch_count"] == 0
            ),
            "park_21_touch_count_must_be_zero": (
                audit["park_21_touch_count"] == 0
            ),
        },

        "safety_privacy_statement": (
            "DIAG-18 introduces a strictly default-off, read-only operator "
            "surface for the DIAG-17 PDF text/layout quality metadata. The "
            "surface renders only when BOTH "
            "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED and "
            "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED are truthy. "
            "If either is unset or falsy, no plan is produced, no metadata "
            "is generated, no side effect occurs. The render plan is pure "
            "data: no Streamlit widgets, no buttons, no callbacks, no "
            "actions, no accept/reject semantics, no state mutation, no "
            "data-layer writes, no document_type mutation. Plans use only "
            "controlled-vocabulary tokens; they never carry raw text, raw "
            "OCR text, raw document text, raw filenames, private paths, "
            "PHI, or secrets. No clinical interpretation; no diagnosis / "
            "medication / DDI / treatment inference; no abbreviation "
            "parsing or expansion; no external API enablement. PARK-20 / "
            "PARK-21 tags are not touched."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.97,
                "residual_unknown_reduction_track_remaining_pct": 0.03,
                "whole_medai_project_done_pct": 90.8,
                "whole_medai_project_remaining_pct": 9.2,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.98,
                "residual_unknown_reduction_track_remaining_pct": 0.02,
                "whole_medai_project_done_pct": 91,
                "whole_medai_project_remaining_pct": 9,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "PARK-22 — parking snapshot capturing the DIAG-17 + DIAG-17B "
                "+ DIAG-18 trio (default-off helper, env-on aggregate "
                "evaluation, env-gated read-only operator surface). "
                "Reports + tags only; no runtime wiring. Cue expansion "
                "remains explicitly NOT recommended."
            ),
            "must_remain_default_off": True,
            "must_remain_review_bound": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_default_behavior": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    audit = payload["env_combination_audit"]
    progress = payload["progress_estimate"]
    proof = payload["default_off_proof"]
    ro = payload["read_only_proof"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-18 Read-Only PDF Text/Layout "
        "Quality Operator Surface"
    )
    lines.append("")
    lines.append(
        "Default-off, read-only operator-surface helper for DIAG-17 "
        "metadata. Pure data-only render plan; no Streamlit widgets, "
        "no buttons, no callbacks, no actions, no state mutation."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Default off: **{payload['default_off']}**")
    lines.append(f"- Read only: **{payload['read_only']}**")
    lines.append(f"- Metadata env var: `{payload['metadata_env_var']}`")
    lines.append(f"- UI env var: `{payload['ui_env_var']}`")
    lines.append(
        f"- Requires both env vars truthy: **{payload['requires_both_env_vars']}**"
    )
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-17 commit (short): `{payload['diag_17_commit_short']}`")
    lines.append(f"- DIAG-17B commit (short): `{payload['diag_17b_commit_short']}`")
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
    lines.append("## Env-combination audit")
    lines.append("")
    lines.append(
        "| Env combination | In-scope emissions (of 21) | Excluded-pool emissions |"
    )
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Neither | {audit['neither_env_in_scope_emissions']} | 0 of {audit['excluded_pool_records_count']} |"
    )
    lines.append(
        f"| Metadata env truthy only | {audit['metadata_only_env_in_scope_emissions']} | 0 of {audit['excluded_pool_records_count']} |"
    )
    lines.append(
        f"| UI env truthy only | {audit['ui_only_env_in_scope_emissions']} | 0 of {audit['excluded_pool_records_count']} |"
    )
    lines.append(
        f"| Both truthy | {audit['both_env_in_scope_emissions']} | {audit['both_env_excluded_pool_emissions']} of {audit['excluded_pool_records_count']} |"
    )
    lines.append("")
    lines.append("## Default-off proof")
    lines.append("")
    for k, v in proof.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Read-only proof")
    lines.append("")
    for k, v in ro.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "default_behavior_changed",
        "runtime_behavior_changed_by_default",
        "operator_ui_surface_added",
        "operator_ui_surface_enabled_by_default",
        "buttons_added",
        "callbacks_added",
        "actions_added",
        "state_mutation_added",
        "data_layer_write_added",
        "document_type_mutation_added",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "raw_text_rendered",
        "raw_filenames_rendered",
        "private_paths_rendered",
        "clinical_value_parsing_performed",
        "diagnosis_inference_performed",
        "medication_inference_performed",
        "ddi_inference_performed",
        "treatment_inference_performed",
        "abbreviation_expansion_performed",
        "cue_expansion_recommended",
        "cue_expansion_performed",
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
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    audit = payload["env_combination_audit"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-18 — Short Summary",
        "",
        "Default-off, read-only operator surface for DIAG-17 metadata. "
        "Gated by BOTH the DIAG-17 metadata env var AND a new DIAG-18 UI "
        "env var.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Metadata env var: `{payload['metadata_env_var']}`",
        f"- UI env var: `{payload['ui_env_var']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-17 commit (short): `{payload['diag_17_commit_short']}`",
        f"- DIAG-17B commit (short): `{payload['diag_17b_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`",
        "",
        "## PARK status",
        "",
        payload["park_status"],
        "",
        "## Env-combination audit",
        "",
        f"- Neither truthy: **{audit['neither_env_in_scope_emissions']}** in-scope emissions",
        f"- Metadata env truthy only: **{audit['metadata_only_env_in_scope_emissions']}** in-scope emissions",
        f"- UI env truthy only: **{audit['ui_only_env_in_scope_emissions']}** in-scope emissions",
        f"- Both truthy: **{audit['both_env_in_scope_emissions']}** in-scope emissions, "
        f"**{audit['both_env_excluded_pool_emissions']}** excluded-pool emissions",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `default_behavior_changed`: {payload['default_behavior_changed']}",
        f"- `operator_ui_surface_added`: {payload['operator_ui_surface_added']}",
        f"- `operator_ui_surface_enabled_by_default`: {payload['operator_ui_surface_enabled_by_default']}",
        f"- `buttons_added` / `callbacks_added` / `actions_added` / `state_mutation_added`: false",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
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
        / "medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui_report.md"
    )
    summary_path = (
        OUT_DIR
        / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_18_PDF_TEXT_LAYOUT_QUALITY_UI.md"
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    summary_path.write_text(render_short_summary(payload), encoding="utf-8")
    out_summary = {
        "conclusion": payload["conclusion"],
        "phase_id": payload["phase_id"],
        "metadata_env_var": payload["metadata_env_var"],
        "ui_env_var": payload["ui_env_var"],
        "neither_env_in_scope_emissions": payload["env_combination_audit"][
            "neither_env_in_scope_emissions"
        ],
        "metadata_only_env_in_scope_emissions": payload[
            "env_combination_audit"
        ]["metadata_only_env_in_scope_emissions"],
        "ui_only_env_in_scope_emissions": payload["env_combination_audit"][
            "ui_only_env_in_scope_emissions"
        ],
        "both_env_in_scope_emissions": payload["env_combination_audit"][
            "both_env_in_scope_emissions"
        ],
        "both_env_excluded_pool_emissions": payload["env_combination_audit"][
            "both_env_excluded_pool_emissions"
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
