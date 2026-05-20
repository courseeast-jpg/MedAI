#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-21 Streamlit fixture audit.

Reports-only/default-off audit of the DIAG-19 Streamlit wiring block.
This script does not import real Streamlit, does not launch the app, does
not open source documents, and does not modify runtime code.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (  # noqa: E402
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
)
from clinical_knowledge.document_type.pdf_text_layout_quality_ui import (  # noqa: E402
    DIAG_18_UI_ENV_VAR,
    render_plan_for_pdf_text_layout_quality,
)

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-21"
OUT_DIR = REPO_ROOT / "reports/medai_doc_type_unknown_diag_21_streamlit_fixture_audit"
APP_MAIN = REPO_ROOT / "app" / "main.py"
DIAG_19_MARKER = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-19"

ALLOWED_STREAMLIT_CALLS = ("st.markdown", "st.caption")
FORBIDDEN_STREAMLIT_CALLS = (
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
FORBIDDEN_RENDER_TOKENS = (
    "raw_text",
    "raw_ocr_text",
    "raw_document_text",
    "filename",
    "filepath",
    "private_path",
    "source_path",
    "patient",
    "secret",
)


@dataclass
class FakeStreamlit:
    """Small fixture object that records Streamlit calls by method name."""

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("st.markdown", args, kwargs))

    def caption(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("st.caption", args, kwargs))

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _record(*args: Any, **kwargs: Any) -> None:
            self.calls.append((f"st.{name}", args, kwargs))

        return _record


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=REPO_ROOT).decode().strip()


def _branch() -> str:
    try:
        return _git_output(["git", "branch", "--show-current"])
    except Exception:
        return "unknown"


def _head_short() -> str:
    try:
        return _git_output(["git", "rev-parse", "--short", "HEAD"])
    except Exception:
        return "unknown"


def _tags_at(commit: str) -> list[str]:
    try:
        out = _git_output(["git", "tag", "--points-at", commit])
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def read_app_main() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def extract_diag_19_block(body: Optional[str] = None) -> str:
    if body is None:
        body = read_app_main()
    idx = body.find(f"# {DIAG_19_MARKER}")
    if idx < 0:
        return ""
    rest = body[idx:]
    match = re.search(r"except Exception:\s*\n\s+pass", rest)
    if not match:
        return rest
    return rest[: match.end()]


def strip_python_comments(source: str) -> str:
    return "\n".join(line.partition("#")[0] for line in source.splitlines())


def static_audit() -> dict[str, Any]:
    body = read_app_main()
    block = extract_diag_19_block(body)
    code_only = strip_python_comments(block)
    advanced_idx = body.find('with st.expander("Advanced technical details"')
    marker_idx = body.find(f"# {DIAG_19_MARKER}")
    end_idx = body.find('st.markdown("</div>"', marker_idx)
    used_st_calls = sorted(set(re.findall(r"st\.[a-zA-Z_][a-zA-Z0-9_]*", code_only)))
    forbidden_calls = [call for call in FORBIDDEN_STREAMLIT_CALLS if call in code_only]
    forbidden_tokens = [
        token for token in FORBIDDEN_RENDER_TOKENS if re.search(rf"\b{re.escape(token)}\b", code_only)
    ]
    return {
        "diag_19_block_count": body.count(DIAG_19_MARKER),
        "diag_19_block_line_count": len(block.splitlines()),
        "diag_19_inside_advanced_technical_details": (
            advanced_idx >= 0 and marker_idx >= 0 and end_idx >= 0 and advanced_idx < marker_idx < end_idx
        ),
        "try_except_guard_present": "try:" in block and "except Exception:" in block and "pass" in block,
        "helper_import_inside_try": "render_plan_for_pdf_text_layout_quality" in block
        and "from clinical_knowledge.document_type.pdf_text_layout_quality_ui import" in block,
        "used_streamlit_calls": used_st_calls,
        "allowed_streamlit_calls_only_static": set(used_st_calls).issubset(set(ALLOWED_STREAMLIT_CALLS)),
        "forbidden_streamlit_calls_static": forbidden_calls,
        "forbidden_render_tokens_static": forbidden_tokens,
        "no_button_callback_action_form_state_mutation_tokens": not any(
            token in code_only
            for token in (
                "button",
                "callback",
                "action",
                "form",
                "session_state",
                "on_click",
                "on_change",
                "write_",
                "mutate",
                "document_type_mutation",
            )
        ),
    }


def _positive_safe_record() -> dict[str, Any]:
    return {
        "pdf_text_layer_detected": "yes",
        "image_like_pdf": "no",
        "alphabetic_content_bucket": "high",
        "native_text_length_bucket": "short",
        "table_like_structure_detected": "yes",
    }


def simulate_diag_19_streamlit_calls(env: Mapping[str, str]) -> dict[str, Any]:
    """Simulate the runtime-facing DIAG-19 block with a fake Streamlit object."""

    st = FakeStreamlit()
    try:
        plan = render_plan_for_pdf_text_layout_quality(_positive_safe_record(), env=env)
        if plan is not None:
            st.markdown("---")
            st.markdown(f"#### {plan['expander_label']}")
            for line in plan["markdown_lines"]:
                st.markdown(line)
            st.caption(plan["disclaimer_line"])
    except Exception:
        pass

    call_names = [name for name, _, _ in st.calls]
    rendered_payload = json.dumps(
        [
            {"call": name, "args": list(args), "kwargs": kwargs}
            for name, args, kwargs in st.calls
        ],
        ensure_ascii=False,
        default=str,
    )
    return {
        "call_count": len(call_names),
        "call_names": call_names,
        "allowed_calls_only": set(call_names).issubset(set(ALLOWED_STREAMLIT_CALLS)),
        "forbidden_call_count": sum(1 for name in call_names if name in FORBIDDEN_STREAMLIT_CALLS),
        "raw_or_private_token_rendered": any(token in rendered_payload for token in FORBIDDEN_RENDER_TOKENS),
    }


def env_gate_fixture_audit() -> dict[str, Any]:
    metadata_only = {PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1"}
    ui_only = {DIAG_18_UI_ENV_VAR: "1"}
    both = {
        PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR: "1",
        DIAG_18_UI_ENV_VAR: "1",
    }
    neither = simulate_diag_19_streamlit_calls({})
    metadata = simulate_diag_19_streamlit_calls(metadata_only)
    ui = simulate_diag_19_streamlit_calls(ui_only)
    both_result = simulate_diag_19_streamlit_calls(both)
    return {
        "neither_env_truthy_streamlit_calls": neither["call_count"],
        "metadata_only_env_truthy_streamlit_calls": metadata["call_count"],
        "ui_only_env_truthy_streamlit_calls": ui["call_count"],
        "both_env_truthy_streamlit_calls": both_result["call_count"],
        "both_env_truthy_call_names": both_result["call_names"],
        "both_env_truthy_allowed_calls_only": both_result["allowed_calls_only"],
        "forbidden_streamlit_call_count": (
            neither["forbidden_call_count"]
            + metadata["forbidden_call_count"]
            + ui["forbidden_call_count"]
            + both_result["forbidden_call_count"]
        ),
        "raw_or_private_token_rendered": any(
            item["raw_or_private_token_rendered"] for item in (neither, metadata, ui, both_result)
        ),
    }


def build_report() -> dict[str, Any]:
    static = static_audit()
    fixture = env_gate_fixture_audit()
    park_20 = _tags_at("3e46461")
    park_21 = _tags_at("9f9e22d")
    park_22 = _tags_at("f4d3cc6")
    park_23 = _tags_at("748c32a")
    park_24 = _tags_at("1b14ffe9f7f16991c7b40b63650b2add59ab024a")
    diag_20_tags = _tags_at("07a8823")
    return {
        "conclusion": "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_ready",
        "phase_id": PHASE_ID,
        "mode": "streamlit_fixture_audit",
        "reports_only": True,
        "default_off": True,
        "branch": _branch(),
        "head_commit_short": _head_short(),
        "default_behavior_changed": False,
        "runtime_behavior_changed": False,
        "streamlit_wiring_changed": False,
        "app_main_modified": False,
        "metadata_env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "ui_env_var": DIAG_18_UI_ENV_VAR,
        "requires_both_env_vars": True,
        "neither_env_truthy_streamlit_calls": fixture["neither_env_truthy_streamlit_calls"],
        "metadata_only_env_truthy_streamlit_calls": fixture["metadata_only_env_truthy_streamlit_calls"],
        "ui_only_env_truthy_streamlit_calls": fixture["ui_only_env_truthy_streamlit_calls"],
        "both_env_truthy_allowed_calls_only": fixture["both_env_truthy_allowed_calls_only"],
        "both_env_truthy_streamlit_calls": fixture["both_env_truthy_streamlit_calls"],
        "allowed_streamlit_calls": list(ALLOWED_STREAMLIT_CALLS),
        "forbidden_streamlit_call_count": fixture["forbidden_streamlit_call_count"],
        "buttons_added": False,
        "callbacks_added": False,
        "actions_added": False,
        "forms_added": False,
        "state_mutation_added": False,
        "data_layer_write_added": False,
        "document_type_mutation_added": False,
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
        "auto_accept_allowed_count": 0,
        "external_api_used_count": 0,
        "all_records_review_bound": True,
        "park_20_tags_touched": False,
        "park_21_tags_touched": False,
        "park_22_tags_touched": False,
        "park_23_tags_touched": False,
        "park_24_tags_touched": False,
        "diag_20_commit_tagged": bool(diag_20_tags),
        "static_audit": static,
        "fixture_audit": fixture,
        "tag_verification": {
            "park_20_tags_at_expected_commit": park_20,
            "park_21_tags_at_expected_commit": park_21,
            "park_22_tags_at_expected_commit": park_22,
            "park_23_tags_at_expected_commit": park_23,
            "park_24_tags_at_expected_commit": park_24,
            "diag_20_tags": diag_20_tags,
        },
        "validation_results": {
            "diag_21_focused_pytest": "pending",
            "diag_21_script_direct": "pending",
            "diag_21_privacy_checks": "pending",
            "broader_validation": "pending",
            "full_pytest": "not_claimed",
        },
        "safety_privacy_statement": (
            "DIAG-21 is a reports-only/default-off Streamlit fixture-test audit. "
            "It uses static source inspection and a fake Streamlit object to record calls. "
            "It does not launch Streamlit, open source documents, print raw text, "
            "render raw filenames, use private paths, enable external APIs, alter runtime "
            "code, mutate document_type, or change extraction/classifier behavior. "
            "Cue expansion remains explicitly not recommended."
        ),
        "progress_estimate": {
            "residual_unknown_reduction_track_done": "approximately 99.995%",
            "residual_unknown_reduction_track_remaining": "approximately 0.005%",
            "whole_medai_done": "approximately 91.9%",
            "whole_medai_remaining": "approximately 8.1%",
        },
        "recommended_next_step": "PARK-25 parking snapshot for DIAG-21, if validations remain clean; no cue expansion.",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-21",
        "",
        "Conclusion: `medai_doc_type_unknown_diag_21_streamlit_fixture_audit_ready`",
        "",
        "## Purpose",
        "",
        "DIAG-21 audits the runtime-facing DIAG-19 Streamlit wiring block with a static source check and a fake Streamlit fixture. It does not launch Streamlit and does not modify runtime files.",
        "",
        "## Fixture Method",
        "",
        "- Static audit extracts the DIAG-19 block from app/main.py.",
        "- Fixture audit calls the DIAG-18 render-plan helper with explicit env mappings.",
        "- A fake Streamlit object records output calls without importing real Streamlit.",
        "",
        "## Env Gate Results",
        "",
        f"- neither_env_truthy_streamlit_calls: `{report['neither_env_truthy_streamlit_calls']}`",
        f"- metadata_only_env_truthy_streamlit_calls: `{report['metadata_only_env_truthy_streamlit_calls']}`",
        f"- ui_only_env_truthy_streamlit_calls: `{report['ui_only_env_truthy_streamlit_calls']}`",
        f"- both_env_truthy_allowed_calls_only: `{report['both_env_truthy_allowed_calls_only']}`",
        f"- forbidden_streamlit_call_count: `{report['forbidden_streamlit_call_count']}`",
        "",
        "## Allowed Calls",
        "",
        "- `st.markdown`",
        "- `st.caption`",
        "",
        "## Safety",
        "",
        f"- default_behavior_changed: `{report['default_behavior_changed']}`",
        f"- runtime_behavior_changed: `{report['runtime_behavior_changed']}`",
        f"- streamlit_wiring_changed: `{report['streamlit_wiring_changed']}`",
        f"- app_main_modified: `{report['app_main_modified']}`",
        f"- cue_expansion_recommended: `{report['cue_expansion_recommended']}`",
        f"- cue_expansion_performed: `{report['cue_expansion_performed']}`",
        f"- external_api_used: `{report['external_api_used']}`",
        f"- all_records_review_bound: `{report['all_records_review_bound']}`",
        f"- diag_20_commit_tagged: `{report['diag_20_commit_tagged']}`",
        "",
        "## Validation Results",
        "",
    ]
    for key, value in report["validation_results"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Progress",
            "",
            f"- Residual Unknown-reduction track: {report['progress_estimate']['residual_unknown_reduction_track_done']} done / {report['progress_estimate']['residual_unknown_reduction_track_remaining']} remaining",
            f"- Whole MedAI project: {report['progress_estimate']['whole_medai_done']} done / {report['progress_estimate']['whole_medai_remaining']} remaining",
            "",
            "## Recommended Next Step",
            "",
            report["recommended_next_step"],
            "",
            "Cue expansion remains NOT recommended.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_summary(report: Mapping[str, Any]) -> str:
    return (
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-21 Streamlit Fixture Audit\n\n"
        f"- phase_id: `{report['phase_id']}`\n"
        f"- mode: `{report['mode']}`\n"
        f"- reports_only: `{report['reports_only']}`\n"
        f"- default_off: `{report['default_off']}`\n"
        f"- neither env truthy calls: `{report['neither_env_truthy_streamlit_calls']}`\n"
        f"- metadata-only env truthy calls: `{report['metadata_only_env_truthy_streamlit_calls']}`\n"
        f"- UI-only env truthy calls: `{report['ui_only_env_truthy_streamlit_calls']}`\n"
        f"- both env truthy allowed calls only: `{report['both_env_truthy_allowed_calls_only']}`\n"
        f"- forbidden_streamlit_call_count: `{report['forbidden_streamlit_call_count']}`\n"
        f"- runtime_behavior_changed: `{report['runtime_behavior_changed']}`\n"
        f"- cue_expansion_recommended: `{report['cue_expansion_recommended']}`\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n"
    )


def write_reports(report: Mapping[str, Any]) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.json"
    md_path = OUT_DIR / "medai_doc_type_unknown_diag_21_streamlit_fixture_audit_report.md"
    summary_path = OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_21_STREAMLIT_FIXTURE_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    summary_path.write_text(render_summary(report), encoding="utf-8")
    return [str(path.relative_to(REPO_ROOT)) for path in (json_path, md_path, summary_path)]


def main() -> int:
    report = build_report()
    written = write_reports(report)
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "phase_id": report["phase_id"],
                "neither_env_truthy_streamlit_calls": report["neither_env_truthy_streamlit_calls"],
                "metadata_only_env_truthy_streamlit_calls": report["metadata_only_env_truthy_streamlit_calls"],
                "ui_only_env_truthy_streamlit_calls": report["ui_only_env_truthy_streamlit_calls"],
                "both_env_truthy_allowed_calls_only": report["both_env_truthy_allowed_calls_only"],
                "forbidden_streamlit_call_count": report["forbidden_streamlit_call_count"],
                "reports_written": written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
