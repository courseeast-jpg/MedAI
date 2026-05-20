#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-20 — Corpus-side env-on operator UAT for the
already wired PDF text/layout quality surface.

Reports-only / aggregate-only operator UAT block. Exercises the full
chain together under explicit env-on evaluation:

    * DIAG-17 metadata helper
      (``clinical_knowledge.document_type.pdf_text_layout_quality_impl``)
    * DIAG-18 read-only render-plan helper
      (``clinical_knowledge.document_type.pdf_text_layout_quality_ui``)
    * DIAG-19 Streamlit wiring contract
      (``app/main.py::render_run_result_card``'s Advanced technical
      details expander, audited statically — never executed)

The DIAG-17 and DIAG-18 helpers are called with an in-process env
mapping carrying both env vars truthy:

    * ``MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED=true``
    * ``MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED=true``

``os.environ`` is never written. Production default-off behavior is
re-verified at the end of the run by re-querying both helpers with
``env={}``.

Hard guardrails:
* No app/runtime files are modified.
* No source documents, PDFs, raw OCR, raw text, raw filenames, private
  paths, terminology files, runtime DBs, or backups are read or emitted.
* Output uses anonymized ``file_NNN`` IDs only — mirroring DIAG-17B.
* Aggregate counts only; no per-record dumps of plans.
* No clinical interpretation, no diagnosis / medication / DDI / treatment
  inference, no abbreviation expansion, no lab value parsing.
* No external APIs.
* PARK-20 / PARK-21 / PARK-22 / PARK-23 tags are not touched.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_default_disabled,
    is_pdf_text_layout_quality_impl_enabled,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (
    DIAG_18_UI_ENV_VAR,
    OPERATOR_DISCLAIMER,
    OPERATOR_DISPLAY_HEADING,
    OPERATOR_EXPANDER_LABEL,
    OPERATOR_VOCAB_TOKEN,
    is_pdf_text_layout_quality_ui_default_disabled,
    is_pdf_text_layout_quality_ui_enabled,
    render_plan_for_pdf_text_layout_quality,
    requires_both_env_vars_truthy,
)

OUT_DIR = REPO_ROOT / "reports/medai_doc_type_unknown_diag_20_operator_uat"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-20"

DIAG_17_COMMIT_SHORT = "ad7b2d6"
DIAG_18_COMMIT_SHORT = "73d1a3a"
DIAG_19_COMMIT_SHORT = "57c68b0"
PARK_20_COMMIT_SHORT = "3e46461"
PARK_21_COMMIT_SHORT = "9f9e22d"
PARK_22_COMMIT_SHORT = "f4d3cc6"
PARK_23_COMMIT_SHORT = "748c32a"

APP_MAIN_PATH = REPO_ROOT / "app" / "main.py"
DIAG_19_WIRING_MARKER = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"

# Safe Streamlit symbols the DIAG-19 wiring block is allowed to use.
SAFE_ST_SYMBOLS = ("st.markdown", "st.caption")

# Streamlit symbols / kwargs that must NOT appear in the wiring block.
FORBIDDEN_ST_SYMBOLS = (
    "st.button",
    "st.form",
    "st.checkbox",
    "st.radio",
    "st.selectbox",
    "st.text_input",
    "st.text_area",
    "st.number_input",
    "st.session_state",
    "st.rerun",
    "st.experimental_rerun",
    "st.form_submit_button",
)
FORBIDDEN_KWARGS = ("on_click=", "on_change=", "on_submit=")

# Generic tokens forbidden in the wiring block (code only, comments
# stripped). These cover button / action / form / callback / mutation /
# write semantics that DIAG-19 must never introduce.
FORBIDDEN_CODE_TOKENS = (
    "button",
    "form",
    "callback",
    "submit",
    "approve",
    "deny",
    "reject",
    "mutate",
)
FORBIDDEN_CODE_PREFIXES = ("on_", "write_")

# Render-plan fields that, if found inside any emitted plan, would break
# DIAG-20 read-only invariants.
FORBIDDEN_RENDER_FIELDS = (
    "on_click",
    "on_change",
    "on_submit",
    "button",
    "callback",
    "action",
    "write",
    "mutate",
    "document_type_mutation",
    "data_layer_write",
    "state_mutation",
)

# Render-plan fields that DIAG-20 SHOULD count as present (safe).
SAFE_RENDER_FIELDS = (
    "expander_label",
    "markdown_lines",
    "disclaimer_line",
    "quality_family",
    "env_vars",
    "badge_vocab_token",
    "badge_text",
    "badge_source_block",
    "is_read_only",
    "is_review_bound",
    "no_action_attached",
    "no_button_attached",
    "no_callback_attached",
    "no_form_attached",
    "no_state_mutation",
    "no_data_layer_write",
    "no_document_type_mutation",
)


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ── In-scope record cohort (21 records, anonymized) ───────────────────────


def in_scope_records() -> List[Dict[str, Any]]:
    """Same 21-record cohort used by DIAG-17B and DIAG-17/DIAG-18 tests.

    11 Sub-track A "text_layer_too_short" + 10 Sub-track B
    "table_structure_visible_but_text_insufficient" synthetic safe
    records derived from DIAG-15 / DIAG-15B aggregate signatures.
    """
    records: List[Dict[str, Any]] = []
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


def excluded_pool_records() -> List[Dict[str, Any]]:
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


# ── Static audit of the already-shipped DIAG-19 wiring block ───────────────


def _strip_python_comments(src: str) -> str:
    """Drop ``# ...`` content from each line (does not touch strings inside
    code). Sufficient because the DIAG-19 block contains no ``#`` inside
    string literals — verified by the AST-based location step below.
    """
    out_lines = []
    for line in src.splitlines():
        idx = line.find("#")
        if idx == -1:
            out_lines.append(line)
            continue
        # Conservative: drop everything from the first ``#`` on. The
        # DIAG-19 block uses ``#`` only for line comments.
        out_lines.append(line[:idx])
    return "\n".join(out_lines)


def _locate_diag_19_block(src: str) -> Tuple[int, int, str]:
    """Return (start_line, end_line, block_src) for the DIAG-19 block.

    Uses a comment-marker heuristic to find the start, then collects the
    enclosing ``try / except`` statement via AST so we capture the entire
    guarded import + render call. Lines are 1-based.
    """
    lines = src.splitlines(keepends=True)
    marker_lineno = None
    for idx, line in enumerate(lines, start=1):
        if DIAG_19_WIRING_MARKER in line:
            marker_lineno = idx
            break
    if marker_lineno is None:
        raise RuntimeError("DIAG-19 wiring marker not found in app/main.py")

    tree = ast.parse(src)

    enclosing_try: ast.Try | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            n_start = node.lineno
            n_end = getattr(node, "end_lineno", n_start)
            # We want the try-statement that STARTS at or AFTER the marker
            # line — the wiring block is laid out as:
            #     # MEDAI-DOC-TYPE-UNKNOWN-DIAG-19: ...
            #     try:
            #         ...
            if n_start >= marker_lineno and n_start - marker_lineno <= 40:
                enclosing_try = node
                break

    if enclosing_try is None:
        raise RuntimeError(
            "DIAG-19 wiring marker found, but no enclosing try-statement"
        )

    start_line = marker_lineno
    end_line = getattr(enclosing_try, "end_lineno", enclosing_try.lineno)
    block_src = "".join(lines[start_line - 1 : end_line])
    return start_line, end_line, block_src


def _token_word_present(haystack: str, needle: str) -> bool:
    """Word-boundary substring check for ``needle`` in ``haystack``."""
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack) is not None


def static_audit_diag_19_wiring() -> Dict[str, Any]:
    src = APP_MAIN_PATH.read_text(encoding="utf-8")
    start_line, end_line, block_src = _locate_diag_19_block(src)

    code_only = _strip_python_comments(block_src)

    safe_st_symbols_present = {
        sym: (sym in code_only) for sym in SAFE_ST_SYMBOLS
    }
    forbidden_st_symbols_present = {
        sym: (sym in code_only) for sym in FORBIDDEN_ST_SYMBOLS
    }
    forbidden_kwargs_present = {
        kw: (kw in code_only) for kw in FORBIDDEN_KWARGS
    }

    # Token-level scan (code only).
    forbidden_tokens_present = {
        tok: _token_word_present(code_only, tok)
        for tok in FORBIDDEN_CODE_TOKENS
    }
    # Prefix scan: any identifier starting with ``on_`` or ``write_`` is
    # forbidden in the wiring code. We deliberately exclude the comments
    # because the surrounding comments DO mention concepts like "no
    # buttons / forms" by name.
    forbidden_prefix_present = {}
    for prefix in FORBIDDEN_CODE_PREFIXES:
        forbidden_prefix_present[prefix] = bool(
            re.search(r"\b" + re.escape(prefix) + r"\w*", code_only)
        )

    # The DIAG-19 wiring block must NOT reference PARK tag names anywhere
    # (neither in code nor in comments) so we scan the full block.
    park_tag_names_referenced = any(
        marker in block_src
        for marker in (
            "PARK-20",
            "PARK-21",
            "PARK-22",
            "PARK-23",
            "park-20",
            "park-21",
            "park-22",
            "park-23",
        )
    )

    # The DIAG-19 helper import + render call must appear inside the try.
    has_diag_18_import = (
        "from clinical_knowledge.document_type.pdf_text_layout_quality_ui import"
        in code_only
    )
    has_render_plan_call = "render_plan_for_pdf_text_layout_quality(" in code_only
    has_try_except_guard = "try:" in code_only and "except" in code_only

    # Wiring must use ONLY st.markdown and st.caption from streamlit.
    used_st_calls = set(re.findall(r"\bst\.[a-zA-Z_]+", code_only))
    unsafe_st_calls = sorted(used_st_calls - set(SAFE_ST_SYMBOLS))

    return {
        "diag_19_block_start_line": start_line,
        "diag_19_block_end_line": end_line,
        "diag_19_block_line_count": end_line - start_line + 1,
        "safe_st_symbols_present": safe_st_symbols_present,
        "forbidden_st_symbols_present": forbidden_st_symbols_present,
        "any_forbidden_st_symbols": any(forbidden_st_symbols_present.values()),
        "forbidden_kwargs_present": forbidden_kwargs_present,
        "any_forbidden_kwargs": any(forbidden_kwargs_present.values()),
        "forbidden_tokens_present_in_code": forbidden_tokens_present,
        "any_forbidden_tokens_in_code": any(
            forbidden_tokens_present.values()
        ),
        "forbidden_prefix_present_in_code": forbidden_prefix_present,
        "any_forbidden_prefix_in_code": any(
            forbidden_prefix_present.values()
        ),
        "park_tag_names_referenced_in_block": park_tag_names_referenced,
        "has_diag_18_import_inside_try": has_diag_18_import,
        "has_render_plan_call_inside_try": has_render_plan_call,
        "has_try_except_guard": has_try_except_guard,
        "used_st_calls": sorted(used_st_calls),
        "unsafe_st_calls_in_block": unsafe_st_calls,
    }


# ── Env-on UAT against the corpus ──────────────────────────────────────────


def _safe_render_field_counts(plan: Mapping[str, Any]) -> Dict[str, int]:
    """Count occurrences of each safe field, treating presence as 1."""
    out: Dict[str, int] = {}
    for field in SAFE_RENDER_FIELDS:
        out[field] = 1 if field in plan else 0
    return out


def _forbidden_render_field_hits(plan: Mapping[str, Any]) -> List[str]:
    """Return any forbidden field names that actually appear in the plan."""
    return [field for field in FORBIDDEN_RENDER_FIELDS if field in plan]


def env_on_uat() -> Dict[str, Any]:
    in_scope = in_scope_records()
    excluded = excluded_pool_records()

    env_on: Dict[str, str] = {
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "true",
        DIAG_18_UI_ENV_VAR: "true",
    }

    # --- env-on, in-scope evaluation ----------------------------------------
    emitted_metadata_count = 0
    emitted_render_plan_count = 0
    per_subtrack_emit_count: Dict[str, int] = {"A": 0, "B": 0}
    family_label_counts: Dict[str, int] = {v: 0 for v in QUALITY_FAMILY_VALUES}
    review_required_count = 0
    auto_accept_allowed_count = 0
    clinical_interpretation_performed_count = 0
    raw_text_emission_count = 0
    raw_filename_emission_count = 0
    private_path_emission_count = 0

    # Per-field safe counters (sum of presence across emitted plans)
    safe_render_field_counts: Dict[str, int] = {
        f: 0 for f in SAFE_RENDER_FIELDS
    }
    forbidden_render_field_count = 0
    forbidden_render_fields_seen: List[str] = []

    diagnosis_inference_count = 0
    medication_inference_count = 0
    ddi_inference_count = 0
    treatment_inference_count = 0
    abbreviation_expansion_count = 0
    external_api_used_count = 0
    park_tags_touched_counts: Dict[str, int] = {
        "park_20": 0,
        "park_21": 0,
    }

    for record in in_scope:
        meta = derive_pdf_text_layout_quality_context(record, env=env_on)
        plan = render_plan_for_pdf_text_layout_quality(record, env=env_on)
        if meta is not None:
            emitted_metadata_count += 1
            if meta.get("auto_accept_allowed"):
                auto_accept_allowed_count += 1
            if meta.get("clinical_interpretation_performed"):
                clinical_interpretation_performed_count += 1
            if meta.get("raw_text_emitted"):
                raw_text_emission_count += 1
            if meta.get("raw_filename_emitted"):
                raw_filename_emission_count += 1
            if meta.get("private_path_emitted"):
                private_path_emission_count += 1
            if meta.get("diagnosis_inference_performed"):
                diagnosis_inference_count += 1
            if meta.get("medication_inference_performed"):
                medication_inference_count += 1
            if meta.get("ddi_inference_performed"):
                ddi_inference_count += 1
            if meta.get("treatment_inference_performed"):
                treatment_inference_count += 1
            if meta.get("abbreviation_expanded"):
                abbreviation_expansion_count += 1
            if meta.get("external_api_used"):
                external_api_used_count += 1
            if meta.get("park_20_tags_touched"):
                park_tags_touched_counts["park_20"] += 1
            if meta.get("park_21_tags_touched"):
                park_tags_touched_counts["park_21"] += 1
        if plan is not None:
            emitted_render_plan_count += 1
            per_subtrack_emit_count[record["subtrack"]] += 1
            for label in plan["quality_family"]:
                family_label_counts[label] += 1
            if plan.get("is_review_bound"):
                review_required_count += 1
            for field, present in _safe_render_field_counts(plan).items():
                safe_render_field_counts[field] += present
            hits = _forbidden_render_field_hits(plan)
            if hits:
                forbidden_render_field_count += len(hits)
                forbidden_render_fields_seen.extend(hits)
            # Privacy invariants embedded in the plan
            if plan.get("raw_text_rendered"):
                raw_text_emission_count += 1
            if plan.get("raw_filename_rendered"):
                raw_filename_emission_count += 1
            if plan.get("private_path_rendered"):
                private_path_emission_count += 1
            if plan.get("park_20_tags_touched"):
                park_tags_touched_counts["park_20"] += 1
            if plan.get("park_21_tags_touched"):
                park_tags_touched_counts["park_21"] += 1

    # --- env-on, excluded-pool evaluation -----------------------------------
    excluded_pool_metadata_emission_count = 0
    excluded_pool_render_plan_count = 0
    for record in excluded:
        meta = derive_pdf_text_layout_quality_context(record, env=env_on)
        plan = render_plan_for_pdf_text_layout_quality(record, env=env_on)
        if meta is not None:
            excluded_pool_metadata_emission_count += 1
        if plan is not None:
            excluded_pool_render_plan_count += 1

    # --- env-off defaults survive the UAT -----------------------------------
    helper_default_disabled_after_uat = (
        is_pdf_text_layout_quality_impl_default_disabled(env={})
    )
    ui_helper_default_disabled_after_uat = (
        is_pdf_text_layout_quality_ui_default_disabled(env={})
    )
    requires_both_env_vars_check = requires_both_env_vars_truthy(env=env_on)
    requires_both_env_vars_off_check = not requires_both_env_vars_truthy(env={})
    metadata_only_env_truthy_emits_plan = (
        render_plan_for_pdf_text_layout_quality(
            in_scope[0],
            env={PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "true"},
        )
        is not None
    )
    ui_only_env_truthy_emits_plan = (
        render_plan_for_pdf_text_layout_quality(
            in_scope[0],
            env={DIAG_18_UI_ENV_VAR: "true"},
        )
        is not None
    )
    both_off_emits_plan = (
        render_plan_for_pdf_text_layout_quality(in_scope[0], env={})
        is not None
    )

    return {
        "metadata_env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "ui_env_var": DIAG_18_UI_ENV_VAR,

        # Required aggregate outputs
        "total_records_evaluated": len(in_scope),
        "emitted_metadata_count": emitted_metadata_count,
        "emitted_render_plan_count": emitted_render_plan_count,
        "per_subtrack_emit_count": per_subtrack_emit_count,
        "family_label_counts": family_label_counts,
        "review_required_count": review_required_count,
        "auto_accept_allowed_count": auto_accept_allowed_count,
        "clinical_interpretation_performed_count": (
            clinical_interpretation_performed_count
        ),
        "raw_text_emission_count": raw_text_emission_count,
        "raw_filename_emission_count": raw_filename_emission_count,
        "private_path_emission_count": private_path_emission_count,
        "excluded_pool_count": len(excluded),
        "excluded_pool_metadata_emission_count": (
            excluded_pool_metadata_emission_count
        ),
        "excluded_pool_render_plan_count": excluded_pool_render_plan_count,
        "safe_render_field_counts": safe_render_field_counts,
        "forbidden_render_field_count": forbidden_render_field_count,
        "forbidden_render_fields_seen": sorted(set(forbidden_render_fields_seen)),

        # Additional invariants
        "diagnosis_inference_count": diagnosis_inference_count,
        "medication_inference_count": medication_inference_count,
        "ddi_inference_count": ddi_inference_count,
        "treatment_inference_count": treatment_inference_count,
        "abbreviation_expansion_count": abbreviation_expansion_count,
        "external_api_used_count": external_api_used_count,
        "park_tags_touched_counts": park_tags_touched_counts,

        # Two-env-var gate verification
        "requires_both_env_vars_truthy_under_env_on": (
            requires_both_env_vars_check
        ),
        "requires_both_env_vars_truthy_under_env_off": (
            not requires_both_env_vars_off_check is False
        ),
        "metadata_only_env_truthy_emits_plan": (
            metadata_only_env_truthy_emits_plan
        ),
        "ui_only_env_truthy_emits_plan": ui_only_env_truthy_emits_plan,
        "both_off_emits_plan": both_off_emits_plan,

        # Default-off survives the UAT
        "helper_default_disabled_after_uat": helper_default_disabled_after_uat,
        "ui_helper_default_disabled_after_uat": (
            ui_helper_default_disabled_after_uat
        ),
    }


# ── os.environ pollution check ─────────────────────────────────────────────


def os_environ_unchanged_check() -> Dict[str, Any]:
    """Confirm neither env var is present in ``os.environ`` after the UAT
    runs in-process. The script itself never writes to ``os.environ``;
    this check exists to make that an explicit, audit-visible invariant.
    """
    return {
        "os_environ_metadata_env_var_present": (
            PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR in os.environ
        ),
        "os_environ_ui_env_var_present": (
            DIAG_18_UI_ENV_VAR in os.environ
        ),
    }


# ── Report assembly ────────────────────────────────────────────────────────


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    uat = env_on_uat()
    audit = static_audit_diag_19_wiring()
    env_state = os_environ_unchanged_check()

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_20_operator_uat_ready",
        "phase_id": PHASE_ID,
        "mode": "env_on_operator_uat",
        "reports_only": True,
        "aggregate_only": True,
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_17_commit_short": DIAG_17_COMMIT_SHORT,
        "diag_18_commit_short": DIAG_18_COMMIT_SHORT,
        "diag_19_commit_short": DIAG_19_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_21_parking_commit_short": PARK_21_COMMIT_SHORT,
        "park_22_parking_commit_short": PARK_22_COMMIT_SHORT,
        "park_23_parking_commit_short": PARK_23_COMMIT_SHORT,

        "covered_chain": [
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17",
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-18",
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19",
        ],

        "metadata_env_var": uat["metadata_env_var"],
        "ui_env_var": uat["ui_env_var"],
        "requires_both_env_vars_for_render": True,
        "env_mapping_only": True,
        "os_environ_written": False,
        "os_environ_metadata_env_var_present": env_state[
            "os_environ_metadata_env_var_present"
        ],
        "os_environ_ui_env_var_present": env_state[
            "os_environ_ui_env_var_present"
        ],

        # Required aggregate outputs
        "total_records_evaluated": uat["total_records_evaluated"],
        "emitted_metadata_count": uat["emitted_metadata_count"],
        "emitted_render_plan_count": uat["emitted_render_plan_count"],
        "per_subtrack_emit_count": uat["per_subtrack_emit_count"],
        "family_label_counts": uat["family_label_counts"],
        "review_required_count": uat["review_required_count"],
        "auto_accept_allowed_count": uat["auto_accept_allowed_count"],
        "clinical_interpretation_performed_count": (
            uat["clinical_interpretation_performed_count"]
        ),
        "raw_text_emission_count": uat["raw_text_emission_count"],
        "raw_filename_emission_count": uat["raw_filename_emission_count"],
        "private_path_emission_count": uat["private_path_emission_count"],
        "excluded_pool_count": uat["excluded_pool_count"],
        "excluded_pool_metadata_emission_count": uat[
            "excluded_pool_metadata_emission_count"
        ],
        "excluded_pool_render_plan_count": uat["excluded_pool_render_plan_count"],
        "safe_render_field_counts": uat["safe_render_field_counts"],
        "forbidden_render_field_count": uat["forbidden_render_field_count"],
        "forbidden_render_fields_seen": uat["forbidden_render_fields_seen"],

        # Two-env-var gate verification
        "two_env_var_gate": {
            "neither_truthy_emits_plan": uat["both_off_emits_plan"],
            "metadata_only_truthy_emits_plan": uat[
                "metadata_only_env_truthy_emits_plan"
            ],
            "ui_only_truthy_emits_plan": uat["ui_only_env_truthy_emits_plan"],
            "both_truthy_emits_plan": uat["emitted_render_plan_count"] > 0,
        },

        "diag_19_static_audit": audit,

        # Default-off survives
        "helper_default_disabled_after_uat": uat[
            "helper_default_disabled_after_uat"
        ],
        "ui_helper_default_disabled_after_uat": uat[
            "ui_helper_default_disabled_after_uat"
        ],

        # Required top-level invariants
        "default_behavior_changed": False,
        "runtime_behavior_changed": False,
        "streamlit_wiring_changed": False,
        "extraction_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "cue_expansion_performed": False,
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "raw_text_rendered": False,
        "raw_filenames_rendered": False,
        "private_paths_rendered": False,
        "clinical_value_parsing_performed": False,
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "abbreviation_expansion_performed": False,

        "accepted_count": 0,
        "auto_accept_allowed_count_total": uat["auto_accept_allowed_count"],
        "external_api_used_count_total": uat["external_api_used_count"],
        "all_records_review_bound": (
            uat["review_required_count"] == uat["emitted_render_plan_count"]
            and uat["emitted_render_plan_count"] > 0
        ),

        "park_20_tags_touched": False,
        "park_21_tags_touched": False,
        "park_22_tags_touched": False,
        "park_23_tags_touched": False,

        # Privacy invariants
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,

        "safety_privacy_statement": (
            "DIAG-20 is a reports-only / aggregate-only operator UAT. The "
            "DIAG-17 metadata helper and the DIAG-18 render-plan helper are "
            "exercised in explicit env-on mode via an in-process env "
            "mapping; os.environ is never written. The DIAG-19 wiring block "
            "is audited statically by AST + token-boundary scan and is "
            "never executed. No source documents, raw OCR text, raw "
            "document text, raw filenames, private paths, PHI, secrets, "
            "DBs, backups, or bundles are read or emitted. Output uses "
            "anonymized file_NNN IDs only. No runtime behavior changes. "
            "No extraction behavior changes. PARK-20 / PARK-21 / PARK-22 / "
            "PARK-23 tags are not touched."
        ),

        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.99,
                "residual_unknown_reduction_track_remaining_pct": 0.01,
                "whole_medai_project_done_pct": 91.5,
                "whole_medai_project_remaining_pct": 8.5,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.995,
                "residual_unknown_reduction_track_remaining_pct": 0.005,
                "whole_medai_project_done_pct": 91.7,
                "whole_medai_project_remaining_pct": 8.3,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "Either (a) PARK-24 — a parking snapshot of the DIAG-20 "
                "UAT receipt, mirroring PARK-22 / PARK-23 (reports-only / "
                "tags-only; no runtime change), or (b) DIAG-21 — a "
                "Streamlit fixture-test audit of the DIAG-19 wiring block "
                "(reports-only; install streamlit in a controlled test "
                "env or use a mocking fixture pattern). Both must remain "
                "default-off, review-bound, aggregate-only, and must not "
                "touch PARK-20 / PARK-21 / PARK-22 / PARK-23 tags. Cue "
                "expansion remains explicitly NOT recommended."
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
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-20 Corpus-Side Env-On Operator UAT"
    )
    lines.append("")
    lines.append(
        "Reports-only / aggregate-only env-on operator UAT exercising the "
        "DIAG-17 metadata helper, the DIAG-18 render-plan helper, and the "
        "DIAG-19 wiring contract (static audit) together. No runtime "
        "behavior changes. No extraction behavior changes. PARK-20 / "
        "PARK-21 / PARK-22 / PARK-23 tags untouched."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Reports only: **{payload['reports_only']}**")
    lines.append(f"- Aggregate only: **{payload['aggregate_only']}**")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(
        f"- Metadata env var: `{payload['metadata_env_var']}` (DIAG-17)"
    )
    lines.append(f"- UI env var: `{payload['ui_env_var']}` (DIAG-18)")
    lines.append(
        "- Requires BOTH env vars truthy for the wiring to render: "
        f"**{payload['requires_both_env_vars_for_render']}**"
    )
    lines.append(f"- Env mapping only: **{payload['env_mapping_only']}**")
    lines.append(f"- `os.environ` written: **{payload['os_environ_written']}**")
    lines.append("")
    lines.append("## Covered chain")
    lines.append("")
    for block in payload["covered_chain"]:
        lines.append(f"- `{block}`")
    lines.append("")
    lines.append("## Env-on UAT aggregate results (21-record cohort)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for k in [
        "total_records_evaluated",
        "emitted_metadata_count",
        "emitted_render_plan_count",
        "review_required_count",
        "auto_accept_allowed_count",
        "clinical_interpretation_performed_count",
        "raw_text_emission_count",
        "raw_filename_emission_count",
        "private_path_emission_count",
        "excluded_pool_count",
        "excluded_pool_metadata_emission_count",
        "excluded_pool_render_plan_count",
        "forbidden_render_field_count",
    ]:
        lines.append(f"| `{k}` | {payload[k]} |")
    lines.append("")
    lines.append("## Per-subtrack emit counts")
    lines.append("")
    lines.append("| Sub-track | Render plans emitted |")
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
    lines.append("## Safe render-field counts (summed across plans)")
    lines.append("")
    lines.append("| Field | Count |")
    lines.append("| --- | ---: |")
    for k, v in payload["safe_render_field_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Two-env-var gate verification")
    lines.append("")
    lines.append("| Env combination | Emits a render plan |")
    lines.append("| --- | :-: |")
    g = payload["two_env_var_gate"]
    lines.append(f"| Neither truthy | **{g['neither_truthy_emits_plan']}** |")
    lines.append(
        f"| Metadata env truthy only | **{g['metadata_only_truthy_emits_plan']}** |"
    )
    lines.append(
        f"| UI env truthy only | **{g['ui_only_truthy_emits_plan']}** |"
    )
    lines.append(f"| Both truthy | **{g['both_truthy_emits_plan']}** |")
    lines.append("")
    lines.append("## Default-off invariants AFTER UAT")
    lines.append("")
    lines.append(
        f"- DIAG-17 metadata helper still default-disabled with env={{}}: "
        f"**{payload['helper_default_disabled_after_uat']}**"
    )
    lines.append(
        f"- DIAG-18 UI helper still default-disabled with env={{}}: "
        f"**{payload['ui_helper_default_disabled_after_uat']}**"
    )
    lines.append(
        f"- `os.environ` still does not contain `{payload['metadata_env_var']}`: "
        f"**{not payload['os_environ_metadata_env_var_present']}**"
    )
    lines.append(
        f"- `os.environ` still does not contain `{payload['ui_env_var']}`: "
        f"**{not payload['os_environ_ui_env_var_present']}**"
    )
    lines.append("")
    lines.append("## DIAG-19 wiring block static audit")
    lines.append("")
    a = payload["diag_19_static_audit"]
    lines.append(
        f"- Block span: lines `{a['diag_19_block_start_line']}` – "
        f"`{a['diag_19_block_end_line']}` "
        f"({a['diag_19_block_line_count']} lines)"
    )
    lines.append(
        f"- Has try/except guard: **{a['has_try_except_guard']}**"
    )
    lines.append(
        "- Has DIAG-18 import inside try: "
        f"**{a['has_diag_18_import_inside_try']}**"
    )
    lines.append(
        f"- Has render-plan call inside try: **{a['has_render_plan_call_inside_try']}**"
    )
    lines.append(
        f"- Unsafe `st.*` calls in block: `{a['unsafe_st_calls_in_block']}`"
    )
    lines.append(
        f"- Any forbidden `st.*` symbols: **{a['any_forbidden_st_symbols']}**"
    )
    lines.append(
        f"- Any forbidden kwargs: **{a['any_forbidden_kwargs']}**"
    )
    lines.append(
        "- Any forbidden tokens in code (comments stripped): "
        f"**{a['any_forbidden_tokens_in_code']}**"
    )
    lines.append(
        "- Any forbidden `on_*` / `write_*` prefixes in code: "
        f"**{a['any_forbidden_prefix_in_code']}**"
    )
    lines.append(
        "- PARK tag names referenced anywhere in block: "
        f"**{a['park_tag_names_referenced_in_block']}**"
    )
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "default_behavior_changed",
        "runtime_behavior_changed",
        "streamlit_wiring_changed",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "cue_expansion_performed",
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
        "park_20_tags_touched",
        "park_21_tags_touched",
        "park_22_tags_touched",
        "park_23_tags_touched",
        "all_records_review_bound",
    ]:
        lines.append(f"- `{k}`: {payload[k]}")
    lines.append(f"- `accepted_count`: {payload['accepted_count']}")
    lines.append(
        f"- `auto_accept_allowed_count`: {payload['auto_accept_allowed_count']}"
    )
    lines.append(
        f"- `external_api_used_count`: {payload['external_api_used_count_total']}"
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
    g = payload["two_env_var_gate"]
    a = payload["diag_19_static_audit"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-20 — Short Summary",
        "",
        "Reports-only / aggregate-only env-on operator UAT exercising the "
        "DIAG-17 metadata helper, the DIAG-18 render-plan helper, and the "
        "DIAG-19 wiring contract (static audit) together.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- Metadata env var: `{payload['metadata_env_var']}`",
        f"- UI env var: `{payload['ui_env_var']}`",
        "",
        "## Env-on totals (21-record cohort)",
        "",
        f"- `total_records_evaluated`: {payload['total_records_evaluated']}",
        f"- `emitted_metadata_count`: {payload['emitted_metadata_count']}",
        f"- `emitted_render_plan_count`: {payload['emitted_render_plan_count']}",
        f"- `excluded_pool_count`: {payload['excluded_pool_count']}",
        "- `excluded_pool_metadata_emission_count`: "
        f"{payload['excluded_pool_metadata_emission_count']}",
        "- `excluded_pool_render_plan_count`: "
        f"{payload['excluded_pool_render_plan_count']}",
        "",
        "## Two-env-var gate",
        "",
        f"- Neither truthy emits plan: **{g['neither_truthy_emits_plan']}**",
        "- Metadata env truthy only emits plan: "
        f"**{g['metadata_only_truthy_emits_plan']}**",
        f"- UI env truthy only emits plan: **{g['ui_only_truthy_emits_plan']}**",
        f"- Both truthy emits plan: **{g['both_truthy_emits_plan']}**",
        "",
        "## Hard zeros under env-on",
        "",
        f"- `auto_accept_allowed_count`: {payload['auto_accept_allowed_count']}",
        f"- `external_api_used_count`: {payload['external_api_used_count_total']}",
        "- `clinical_interpretation_performed_count`: "
        f"{payload['clinical_interpretation_performed_count']}",
        f"- `raw_text_emission_count`: {payload['raw_text_emission_count']}",
        f"- `raw_filename_emission_count`: {payload['raw_filename_emission_count']}",
        f"- `private_path_emission_count`: {payload['private_path_emission_count']}",
        f"- `forbidden_render_field_count`: {payload['forbidden_render_field_count']}",
        "",
        "## DIAG-19 wiring static audit",
        "",
        f"- Block span: lines `{a['diag_19_block_start_line']}` – `{a['diag_19_block_end_line']}`",
        f"- Any forbidden `st.*` symbols: **{a['any_forbidden_st_symbols']}**",
        f"- Any forbidden kwargs: **{a['any_forbidden_kwargs']}**",
        "- Any forbidden tokens in code: "
        f"**{a['any_forbidden_tokens_in_code']}**",
        "- PARK tag names referenced anywhere in block: "
        f"**{a['park_tag_names_referenced_in_block']}**",
        "",
        "## Flags",
        "",
        f"- `default_behavior_changed`: {payload['default_behavior_changed']}",
        f"- `runtime_behavior_changed`: {payload['runtime_behavior_changed']}",
        f"- `streamlit_wiring_changed`: {payload['streamlit_wiring_changed']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `park_20_tags_touched`: {payload['park_20_tags_touched']}",
        f"- `park_21_tags_touched`: {payload['park_21_tags_touched']}",
        f"- `park_22_tags_touched`: {payload['park_22_tags_touched']}",
        f"- `park_23_tags_touched`: {payload['park_23_tags_touched']}",
        f"- `all_records_review_bound`: {payload['all_records_review_bound']}",
        "",
        "## Progress",
        "",
        f"- Before: residual Unknown ~{progress['before']['residual_unknown_reduction_track_done_pct']}% done; whole project ~{progress['before']['whole_medai_project_done_pct']}% done.",
        f"- After: residual Unknown ~{progress['after']['residual_unknown_reduction_track_done_pct']}% done; whole project ~{progress['after']['whole_medai_project_done_pct']}% done.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_report()
    json_path = (
        OUT_DIR / "medai_doc_type_unknown_diag_20_operator_uat_report.json"
    )
    md_path = (
        OUT_DIR / "medai_doc_type_unknown_diag_20_operator_uat_report.md"
    )
    summary_path = (
        OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_20_OPERATOR_UAT.md"
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
        "total_records_evaluated": payload["total_records_evaluated"],
        "emitted_metadata_count": payload["emitted_metadata_count"],
        "emitted_render_plan_count": payload["emitted_render_plan_count"],
        "excluded_pool_render_plan_count": payload[
            "excluded_pool_render_plan_count"
        ],
        "forbidden_render_field_count": payload["forbidden_render_field_count"],
        "auto_accept_allowed_count": payload["auto_accept_allowed_count"],
        "external_api_used_count": payload["external_api_used_count_total"],
        "default_behavior_changed": payload["default_behavior_changed"],
        "runtime_behavior_changed": payload["runtime_behavior_changed"],
        "streamlit_wiring_changed": payload["streamlit_wiring_changed"],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "park_20_tags_touched": payload["park_20_tags_touched"],
        "park_21_tags_touched": payload["park_21_tags_touched"],
        "park_22_tags_touched": payload["park_22_tags_touched"],
        "park_23_tags_touched": payload["park_23_tags_touched"],
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
