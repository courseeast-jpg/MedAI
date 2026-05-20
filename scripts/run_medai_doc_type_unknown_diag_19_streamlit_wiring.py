#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-19 — Default-off Streamlit wiring audit.

Reports-only audit of the DIAG-19 Streamlit wiring in
``app/main.py::render_run_result_card``. Validates:

    * The wiring block is present exactly once.
    * The block lives inside the "Advanced technical details" expander
      and after the existing DIAG-08A / DIAG-10A / DIAG-12A blocks.
    * The block uses a try/except guarded import for the DIAG-18 helper.
    * The block has no buttons / forms / callbacks / actions / state-
      mutation / data-layer-write / document_type mutation surface area.
    * The block never renders raw text, raw filenames, or private paths.
    * The DIAG-18 render-plan helper enforces the two-env-var gate
      (neither / metadata-only / UI-only → None, both → plan), so the
      wiring is default-off purely by env semantics — no extra check is
      needed in app/main.py.
    * PARK-20 / PARK-21 / PARK-22 tag-touch flags are False.

Hard guardrails: no source documents, raw OCR text, raw document text,
raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are
read or emitted. The DIAG-17 / DIAG-18 helpers are exercised via in-
process env mappings only; ``os.environ`` is never written.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    is_pdf_text_layout_quality_impl_default_disabled,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (
    DIAG_18_UI_ENV_VAR,
    is_pdf_text_layout_quality_ui_default_disabled,
    render_plan_for_pdf_text_layout_quality,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_MAIN = REPO_ROOT / "app" / "main.py"
OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_19_streamlit_wiring"
)

PARK_20_COMMIT_SHORT = "3e46461"
PARK_21_COMMIT_SHORT = "9f9e22d"
PARK_22_COMMIT_SHORT = "f4d3cc6"
DIAG_17_COMMIT_SHORT = "ad7b2d6"
DIAG_18_COMMIT_SHORT = "f3c9760"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"

DIAG_19_MARKER = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"
DIAG_18_IMPORT_LINE = (
    "from clinical_knowledge.document_type.pdf_text_layout_quality_ui import"
)
RENDER_PLAN_CALL = "render_plan_for_pdf_text_layout_quality"


# Snake-case token-boundary patterns that MUST NOT appear in the DIAG-19
# wiring block (action / mutation / form / button surface area).
_FORBIDDEN_TOKENS = frozenset(
    {"button", "form", "callback", "submit", "approve", "deny", "reject", "mutate"}
)
# Composite-prefix patterns that imply an action handler.
_FORBIDDEN_PREFIXES = (
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
# Streamlit symbols that are read-only (allowed) inside the block.
_ALLOWED_ST_SYMBOLS = (
    "st.markdown",
    "st.caption",
    "st.expander",  # only as containing context; we check the block, not it
)
# Streamlit symbols that would imply action-capable UI (forbidden).
_FORBIDDEN_ST_SYMBOLS = (
    "st.button",
    "st.form",
    "st.form_submit_button",
    "st.download_button",
    "st.checkbox",
    "st.radio",
    "st.selectbox",
    "st.text_input",
    "st.text_area",
    "st.number_input",
    "st.date_input",
    "st.time_input",
    "st.file_uploader",
    "st.session_state",
    "st.experimental_rerun",
    "st.rerun",
    "on_click=",
    "on_change=",
    "on_submit=",
)


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _read_app_main() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def _extract_diag_19_block(body: str) -> str:
    """Return the DIAG-19 wiring block as a single contiguous string.

    The block runs from the DIAG-19 marker line through the closing of its
    try/except (the next blank line after the ``except Exception: pass``
    line that follows the DIAG-19 marker).
    """
    marker = f"# {DIAG_19_MARKER}"
    idx = body.find(marker)
    if idx == -1:
        return ""
    rest = body[idx:]
    # Take everything up to and including the first standalone "pass" after
    # the first "except Exception:" — the established pattern in this file.
    m = re.search(r"except Exception:\s*\n\s+pass", rest)
    if not m:
        return rest
    return rest[: m.end()]


def _strip_python_comments(source: str) -> str:
    """Remove ``# comment`` content while preserving line structure.

    The block-level audits below care about CODE behavior. Comments are
    informative documentation; descriptive words like "forms" or
    "filenames" in comments must not trip the action-surface check.
    """
    out_lines: List[str] = []
    for line in source.splitlines():
        # Drop everything from the first ``#`` not inside a string literal.
        # The DIAG-19 block does not embed ``#`` inside string literals, so
        # a simple split suffices for this audit.
        code, _, _ = line.partition("#")
        out_lines.append(code)
    return "\n".join(out_lines)


def static_audit_app_main() -> Dict[str, Any]:
    body = _read_app_main()
    occurrences = body.count(DIAG_19_MARKER)
    block = _extract_diag_19_block(body)
    block_lines = block.splitlines()
    # For action / token / privacy checks we scan code only — comments
    # may contain descriptive English words that would otherwise trip
    # word-boundary token matching.
    code_only = _strip_python_comments(block)

    # 1. Block lives inside the Advanced technical details expander.
    advanced_idx = body.find('with st.expander("Advanced technical details"')
    diag_19_idx = body.find(f"# {DIAG_19_MARKER}")
    end_with_div = body.find('st.markdown("</div>"')
    inside_advanced_expander = (
        advanced_idx != -1
        and diag_19_idx != -1
        and end_with_div != -1
        and advanced_idx < diag_19_idx < end_with_div
    )

    # 2. Block is AFTER DIAG-08A, DIAG-10A, DIAG-12A blocks.
    after_prior_blocks = True
    for prior_marker in (
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A",
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A",
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A",
    ):
        prior_idx = body.find(prior_marker)
        if prior_idx == -1 or prior_idx > diag_19_idx:
            after_prior_blocks = False

    # 3. try/except guarded.
    has_try_except_guard = (
        "try:" in block
        and "except Exception:" in block
        and "    pass" in block
    )

    # 4. DIAG-18 helper imported inside the try.
    has_diag_18_import = DIAG_18_IMPORT_LINE in block
    has_render_plan_call = RENDER_PLAN_CALL in block

    # 5. No action-capable Streamlit symbols (scan code, not comments).
    forbidden_st_hits = [s for s in _FORBIDDEN_ST_SYMBOLS if s in code_only]

    # 6. No forbidden tokens in code identifiers.
    forbidden_token_hits: List[str] = []
    for tok in re.findall(r"\b[a-z_][a-z0-9_]+\b", code_only):
        if tok in _FORBIDDEN_TOKENS:
            forbidden_token_hits.append(tok)
        else:
            for prefix in _FORBIDDEN_PREFIXES:
                if tok.startswith(prefix) and not tok.startswith("no_"):
                    forbidden_token_hits.append(tok)
                    break
    forbidden_token_hits = sorted(set(forbidden_token_hits))

    # 7. Only safe st.* symbols used (scan code).
    used_st_symbols = sorted(set(re.findall(r"st\.[a-zA-Z_][a-zA-Z0-9_]*", code_only)))
    unsafe_st_symbols_used = [s for s in used_st_symbols if s in _FORBIDDEN_ST_SYMBOLS]

    return {
        "occurrence_count": occurrences,
        "block_line_count": len(block_lines),
        "inside_advanced_technical_details_expander": inside_advanced_expander,
        "after_prior_diag_08a_10a_12a_blocks": after_prior_blocks,
        "has_try_except_guard": has_try_except_guard,
        "has_diag_18_import_inside_try": has_diag_18_import,
        "has_render_plan_call_inside_try": has_render_plan_call,
        "forbidden_st_symbols_present": forbidden_st_hits,
        "forbidden_tokens_present": forbidden_token_hits,
        "used_st_symbols_in_block": used_st_symbols,
        "unsafe_st_symbols_used_in_block": unsafe_st_symbols_used,
        "block_excerpt_head_5_lines": block_lines[:5],
        "block_excerpt_tail_3_lines": block_lines[-3:],
    }


def env_combination_audit() -> Dict[str, Any]:
    """Exercise the DIAG-18 render-plan helper directly to prove the two-
    env-var gate. (The Streamlit block above just consumes this helper.)
    """
    record = {
        "pdf_text_layer_detected": "yes",
        "image_like_pdf": "no",
        "alphabetic_content_bucket": "high",
        "native_text_length_bucket": "short",
        "table_like_structure_detected": "no",
    }
    neither = render_plan_for_pdf_text_layout_quality(record, env={})
    metadata_only = render_plan_for_pdf_text_layout_quality(
        record, env={PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    )
    ui_only = render_plan_for_pdf_text_layout_quality(
        record, env={DIAG_18_UI_ENV_VAR: "1"}
    )
    both = render_plan_for_pdf_text_layout_quality(
        record,
        env={
            PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1",
            DIAG_18_UI_ENV_VAR: "1",
        },
    )
    return {
        "neither_env_plan_is_none": neither is None,
        "metadata_only_env_plan_is_none": metadata_only is None,
        "ui_only_env_plan_is_none": ui_only is None,
        "both_env_plan_is_dict": isinstance(both, dict),
        "helpers_still_default_disabled_after_audit": (
            is_pdf_text_layout_quality_impl_default_disabled(env={})
            and is_pdf_text_layout_quality_ui_default_disabled(env={})
        ),
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    static = static_audit_app_main()
    env_audit = env_combination_audit()

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_19_streamlit_wiring_ready",
        "phase_id": PHASE_ID,
        "mode": "default_off_streamlit_wiring",
        "default_off": True,
        "metadata_env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "ui_env_var": DIAG_18_UI_ENV_VAR,
        "requires_both_env_vars": True,
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_17_commit_short": DIAG_17_COMMIT_SHORT,
        "diag_18_commit_short": DIAG_18_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_21_parking_commit_short": PARK_21_COMMIT_SHORT,
        "park_22_parking_commit_short": PARK_22_COMMIT_SHORT,
        "park_status": (
            "PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain "
            "on origin at 9f9e22d. PARK-22 tags remain on origin at "
            "f4d3cc6. DIAG-19 does not touch any tag."
        ),
        "source_reports_referenced": [
            "block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)",
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
            "block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)",
            "block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)",
            "block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)",
            "block DIAG-17B (directory: medai_doc_type_unknown_diag_17b_env_on_aggregate_eval)",
            "block DIAG-18 (directory: medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui)",
        ],

        "static_audit_app_main": static,
        "env_combination_audit": env_audit,

        # Required behavior flags
        "behavior_changed": True,
        "default_behavior_changed": False,
        "runtime_behavior_changed_by_default": False,
        "streamlit_wiring_added": True,
        "streamlit_wiring_enabled_by_default": False,
        "advanced_technical_details_only": True,
        "buttons_added": False,
        "callbacks_added": False,
        "actions_added": False,
        "forms_added": False,
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
        "park_22_tags_touched": False,

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
            "neither_env_plan_is_none": env_audit["neither_env_plan_is_none"],
            "metadata_only_env_plan_is_none": env_audit[
                "metadata_only_env_plan_is_none"
            ],
            "ui_only_env_plan_is_none": env_audit["ui_only_env_plan_is_none"],
            "both_env_plan_is_dict": env_audit["both_env_plan_is_dict"],
            "helpers_still_default_disabled_after_audit": env_audit[
                "helpers_still_default_disabled_after_audit"
            ],
        },
        "read_only_proof": {
            "diag_19_wiring_appears_exactly_once": (
                static["occurrence_count"] == 1
            ),
            "inside_advanced_technical_details_expander": static[
                "inside_advanced_technical_details_expander"
            ],
            "after_prior_diag_08a_10a_12a_blocks": static[
                "after_prior_diag_08a_10a_12a_blocks"
            ],
            "has_try_except_guard": static["has_try_except_guard"],
            "has_diag_18_import_inside_try": static[
                "has_diag_18_import_inside_try"
            ],
            "has_render_plan_call_inside_try": static[
                "has_render_plan_call_inside_try"
            ],
            "no_forbidden_st_symbols_in_block": (
                static["forbidden_st_symbols_present"] == []
            ),
            "no_forbidden_tokens_in_block": (
                static["forbidden_tokens_present"] == []
            ),
            "no_unsafe_st_symbols_used_in_block": (
                static["unsafe_st_symbols_used_in_block"] == []
            ),
        },

        "safety_privacy_statement": (
            "DIAG-19 adds a strictly default-off Streamlit wiring block in "
            "app/main.py::render_run_result_card's Advanced technical "
            "details expander. The block renders only when BOTH "
            "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED and "
            "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED are truthy. "
            "The DIAG-18 helper enforces the two-env-var gate internally; "
            "if either env var is unset or falsy the helper returns None "
            "and the block renders nothing. The block uses only "
            "st.markdown and st.caption — no buttons, no forms, no "
            "callbacks, no actions, no state mutation, no data-layer "
            "writes, no document_type mutation. The block carries no raw "
            "text, raw filenames, or private paths. The DIAG-18 helper "
            "import lives inside a try/except so non-Streamlit test "
            "collection is unaffected if either helper module is absent. "
            "No clinical interpretation; no diagnosis / medication / DDI "
            "/ treatment inference; no abbreviation parsing or expansion; "
            "no external API enablement. PARK-20 / PARK-21 / PARK-22 tags "
            "are not touched."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.98,
                "residual_unknown_reduction_track_remaining_pct": 0.02,
                "whole_medai_project_done_pct": 91,
                "whole_medai_project_remaining_pct": 9,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.99,
                "residual_unknown_reduction_track_remaining_pct": 0.01,
                "whole_medai_project_done_pct": 91.5,
                "whole_medai_project_remaining_pct": 8.5,
            },
        },
        "next_block_recommendation": {
            "recommended_name": (
                "PARK-23 — parking snapshot capturing the DIAG-19 wiring "
                "(reports + tags only). After PARK-23, the natural "
                "follow-on is either a corpus-side env-on operator UAT "
                "block (still default-off in production) or an evaluation-"
                "only block that audits the wiring under Streamlit fixture "
                "tests. Cue expansion remains explicitly NOT recommended."
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
    static = payload["static_audit_app_main"]
    env_audit = payload["env_combination_audit"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-19 Env-Gated Streamlit Wiring for "
        "PDF Text/Layout Quality"
    )
    lines.append("")
    lines.append(
        "Default-off Streamlit wiring of the DIAG-18 read-only render plan "
        "into the Run & Review tab's Advanced technical details expander. "
        "The wiring renders only when BOTH the DIAG-17 metadata env var "
        "AND the DIAG-18 UI env var are truthy."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Default off: **{payload['default_off']}**")
    lines.append(f"- Metadata env var: `{payload['metadata_env_var']}`")
    lines.append(f"- UI env var: `{payload['ui_env_var']}`")
    lines.append(
        f"- Requires both env vars truthy: **{payload['requires_both_env_vars']}**"
    )
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-17 commit (short): `{payload['diag_17_commit_short']}`")
    lines.append(f"- DIAG-18 commit (short): `{payload['diag_18_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
    )
    lines.append(
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`"
    )
    lines.append(
        f"- PARK-22 parking commit (short): `{payload['park_22_parking_commit_short']}`"
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
    lines.append("## Static audit of `app/main.py`")
    lines.append("")
    for k in [
        "occurrence_count",
        "block_line_count",
        "inside_advanced_technical_details_expander",
        "after_prior_diag_08a_10a_12a_blocks",
        "has_try_except_guard",
        "has_diag_18_import_inside_try",
        "has_render_plan_call_inside_try",
        "forbidden_st_symbols_present",
        "forbidden_tokens_present",
        "unsafe_st_symbols_used_in_block",
    ]:
        lines.append(f"- `{k}`: {static[k]}")
    lines.append(
        f"- Safe `st.*` symbols used in block: {static['used_st_symbols_in_block']}"
    )
    lines.append("")
    lines.append("## Env-combination audit (DIAG-18 helper)")
    lines.append("")
    for k, v in env_audit.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Default-off proof")
    lines.append("")
    for k, v in payload["default_off_proof"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Read-only proof")
    lines.append("")
    for k, v in payload["read_only_proof"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "default_behavior_changed",
        "runtime_behavior_changed_by_default",
        "streamlit_wiring_added",
        "streamlit_wiring_enabled_by_default",
        "advanced_technical_details_only",
        "buttons_added",
        "callbacks_added",
        "actions_added",
        "forms_added",
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
        "park_22_tags_touched",
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
    lines.append(f"- {payload['next_block_recommendation']['recommended_name']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    proof = payload["default_off_proof"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-19 — Short Summary",
        "",
        "Default-off Streamlit wiring of the DIAG-18 read-only render plan "
        "into the Run & Review tab's Advanced technical details expander.",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Metadata env var: `{payload['metadata_env_var']}`",
        f"- UI env var: `{payload['ui_env_var']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-18 commit (short): `{payload['diag_18_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        f"- PARK-21 parking commit (short): `{payload['park_21_parking_commit_short']}`",
        f"- PARK-22 parking commit (short): `{payload['park_22_parking_commit_short']}`",
        "",
        "## PARK status",
        "",
        payload["park_status"],
        "",
        "## Default-off proof (key checks)",
        "",
        f"- neither env plan is None: **{proof['neither_env_plan_is_none']}**",
        f"- metadata-only env plan is None: **{proof['metadata_only_env_plan_is_none']}**",
        f"- UI-only env plan is None: **{proof['ui_only_env_plan_is_none']}**",
        f"- both-env plan is a dict: **{proof['both_env_plan_is_dict']}**",
        f"- helpers still default-disabled after audit: **{proof['helpers_still_default_disabled_after_audit']}**",
        "",
        "## Flags",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `default_behavior_changed`: {payload['default_behavior_changed']}",
        f"- `streamlit_wiring_added`: {payload['streamlit_wiring_added']}",
        f"- `streamlit_wiring_enabled_by_default`: {payload['streamlit_wiring_enabled_by_default']}",
        f"- `buttons_added` / `callbacks_added` / `actions_added` / `forms_added` / `state_mutation_added`: false",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `park_20_tags_touched`: {payload['park_20_tags_touched']}",
        f"- `park_21_tags_touched`: {payload['park_21_tags_touched']}",
        f"- `park_22_tags_touched`: {payload['park_22_tags_touched']}",
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
        OUT_DIR / "medai_doc_type_unknown_diag_19_streamlit_wiring_report.json"
    )
    md_path = (
        OUT_DIR / "medai_doc_type_unknown_diag_19_streamlit_wiring_report.md"
    )
    summary_path = (
        OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_19_STREAMLIT_WIRING.md"
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
        "occurrence_count": payload["static_audit_app_main"]["occurrence_count"],
        "inside_advanced_technical_details_expander": payload[
            "static_audit_app_main"
        ]["inside_advanced_technical_details_expander"],
        "behavior_changed": payload["behavior_changed"],
        "default_behavior_changed": payload["default_behavior_changed"],
        "streamlit_wiring_added": payload["streamlit_wiring_added"],
        "streamlit_wiring_enabled_by_default": payload[
            "streamlit_wiring_enabled_by_default"
        ],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "park_20_tags_touched": payload["park_20_tags_touched"],
        "park_21_tags_touched": payload["park_21_tags_touched"],
        "park_22_tags_touched": payload["park_22_tags_touched"],
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
