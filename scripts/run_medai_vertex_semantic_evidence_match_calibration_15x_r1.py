#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-EVIDENCE-MATCH-CALIBRATION-15X-R1 (no-live).

Hardens evidence anchoring for the Vertex semantic contract so future live
calibration requires verbatim source evidence (whitespace/Unicode normalization
only), rejecting paraphrase / inferred / out-of-section evidence. Makes no
provider call, sets no live gate, and uses no embeddings / LLM judge.

It validates the strict matcher against synthetic cases and the recorded 15X
``cal_uncertainty`` failure shape, confirms the prompt contract is hardened, and
emits a retry recommendation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.vertex_semantic_package_contract import (
    PROMPT_REQUIRED_PHRASES,
    evidence_text_is_source_verbatim,
    normalize_evidence_text,
)
from execution.vertex_semantic_calibration_batch import build_calibration_fixtures

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_evidence_match_calibration_15x_r1"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "evidence_match_cases.json"
MATRIX_MD = REPORT_DIR / "evidence_match_matrix.md"
RETRY_MD = REPORT_DIR / "retry_recommendation.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession")

# Synthetic evidence-match calibration cases. Source is a synthetic snippet; the
# candidate evidence_text is classified expect_pass True only when it is a
# verbatim (normalized) source span.
SYNTHETIC_SOURCE = "SYNTHETIC note: 'possible mild finding, uncertain' in section Results; value 5 mg/dL within range."
SECTION_RESULTS = "value 5 mg/dL within range"
SECTION_OTHER = "header metadata only"

CASES = [
    {"case": "exact_evidence_match", "evidence": "possible mild finding, uncertain", "section": None, "expect_pass": True, "kind": "exact"},
    {"case": "normalized_whitespace_match", "evidence": "possible   mild   finding,   uncertain", "section": None, "expect_pass": True, "kind": "normalized_whitespace"},
    {"case": "unicode_quote_normalized_match", "evidence": "‘possible mild finding, uncertain’", "section": None, "expect_pass": True, "kind": "substring"},
    {"case": "substring_of_source", "evidence": "value 5 mg/dL within range", "section": None, "expect_pass": True, "kind": "substring"},
    {"case": "paraphrase_dropped_comma", "evidence": "possible mild finding uncertain", "section": None, "expect_pass": False, "kind": "paraphrase"},
    {"case": "paraphrase_synonym", "evidence": "a slight finding that is unclear", "section": None, "expect_pass": False, "kind": "paraphrase"},
    {"case": "inferred_evidence_not_in_source", "evidence": "patient is clinically stable", "section": None, "expect_pass": False, "kind": "inferred"},
    {"case": "wrong_section_constraint", "evidence": "possible mild finding, uncertain", "section": SECTION_OTHER, "expect_pass": False, "kind": "wrong_section"},
    {"case": "right_section_constraint", "evidence": "value 5 mg/dL within range", "section": SECTION_RESULTS, "expect_pass": True, "kind": "substring_section"},
    {"case": "null_evidence", "evidence": None, "section": None, "expect_pass": False, "kind": "null"},
    {"case": "empty_evidence", "evidence": "", "section": None, "expect_pass": False, "kind": "empty"},
]


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str, ensure_ascii=True)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _run_cases() -> list[dict[str, Any]]:
    out = []
    for c in CASES:
        observed = evidence_text_is_source_verbatim(c["evidence"], SYNTHETIC_SOURCE, section_text=c["section"])
        out.append({**c, "observed_pass": observed, "correct": observed == c["expect_pass"]})
    return out


def _recorded_15x_failure_case() -> dict[str, Any]:
    """The recorded cal_uncertainty failure: model evidence was not a verbatim
    source span. Confirm the strict matcher keeps it a failure, and that a true
    verbatim source span WOULD pass (so the rule is verbatim, not blanket-reject)."""
    fixture = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_uncertainty")
    source = fixture.source_visible_body
    # Representative paraphrase Vertex returned (comma dropped / reworded) — must stay FAIL.
    paraphrase_evidence = "possible mild finding uncertain"
    verbatim_span = "possible mild finding, uncertain"
    return {
        "fixture_id": fixture.fixture_id,
        "recorded_marker": "finding_0_not_source_anchored",
        "paraphrase_rejected": evidence_text_is_source_verbatim(paraphrase_evidence, source) is False,
        "verbatim_span_would_pass": evidence_text_is_source_verbatim(verbatim_span, source) is True,
    }


def build_report() -> dict[str, Any]:
    case_results = _run_cases()
    recorded = _recorded_15x_failure_case()
    prompt_hardened = "evidence_verbatim" in PROMPT_REQUIRED_PHRASES and "verbatim" in PROMPT_REQUIRED_PHRASES["evidence_verbatim"].lower()
    evidence_verbatim_required = (
        "do not paraphrase" in PROMPT_REQUIRED_PHRASES.get("evidence_verbatim", "").lower()
        and "set evidence_text to null" in PROMPT_REQUIRED_PHRASES.get("evidence_verbatim", "").lower()
    )

    summary = {
        "block": "MEDAI-VERTEX-SEMANTIC-EVIDENCE-MATCH-CALIBRATION-15X-R1",
        "exact_evidence_match_pass_count": sum(1 for c in case_results if c["kind"] == "exact" and c["observed_pass"]),
        "normalized_whitespace_match_pass_count": sum(1 for c in case_results if c["kind"] == "normalized_whitespace" and c["observed_pass"]),
        "substring_evidence_match_pass_count": sum(1 for c in case_results if c["kind"] in {"substring", "substring_section"} and c["observed_pass"]),
        "paraphrase_rejected_count": sum(1 for c in case_results if c["kind"] == "paraphrase" and not c["observed_pass"]),
        "inferred_evidence_rejected_count": sum(1 for c in case_results if c["kind"] == "inferred" and not c["observed_pass"]),
        "wrong_section_rejected_count": sum(1 for c in case_results if c["kind"] == "wrong_section" and not c["observed_pass"]),
        "null_or_empty_rejected_count": sum(1 for c in case_results if c["kind"] in {"null", "empty"} and not c["observed_pass"]),
        "all_cases_correct": all(c["correct"] for c in case_results),
        "recorded_15x_failure_preserved": recorded["paraphrase_rejected"] and recorded["verbatim_span_would_pass"],
        "recorded_case": recorded,
        "prompt_contract_hardened": prompt_hardened,
        "evidence_text_verbatim_required": evidence_verbatim_required,
        "embedding_or_llm_judge_used": False,
        "hallucinated_field_count": 0,
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "review_required": True,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }
    summary["evidence_match_calibration_passed"] = (
        summary["all_cases_correct"]
        and summary["recorded_15x_failure_preserved"]
        and summary["prompt_contract_hardened"]
        and summary["evidence_text_verbatim_required"]
        and summary["paraphrase_rejected_count"] >= 1
        and summary["inferred_evidence_rejected_count"] >= 1
        and summary["wrong_section_rejected_count"] >= 1
        and summary["exact_evidence_match_pass_count"] >= 1
        and summary["normalized_whitespace_match_pass_count"] >= 1
        and summary["substring_evidence_match_pass_count"] >= 1
    )
    return {"summary": summary, "cases": case_results}


def _matrix_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "evidence_match_calibration_passed", "exact_evidence_match_pass_count",
        "normalized_whitespace_match_pass_count", "substring_evidence_match_pass_count",
        "paraphrase_rejected_count", "inferred_evidence_rejected_count", "wrong_section_rejected_count",
        "null_or_empty_rejected_count", "recorded_15x_failure_preserved", "prompt_contract_hardened",
        "evidence_text_verbatim_required", "embedding_or_llm_judge_used", "hallucinated_field_count",
        "live_call_made", "external_api_used", "active_written_count", "active_mkb_record_created_count",
        "auto_accept_true_count", "review_required", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        ["# 15X-R1 evidence-match calibration matrix", "", "| Metric | Value |", "| --- | --- |",
         *[f"| {k} | `{summary[k]}` |" for k in keys], "",
         "No embeddings / no LLM judge; verbatim-substring matching only; no live call.", ""]
    )


def _retry_markdown(summary: dict[str, Any]) -> str:
    ready = summary["evidence_match_calibration_passed"]
    decision = "READY_TO_RERUN_15X_ONCE" if ready else "DO_NOT_RERUN_15X"
    return "\n".join(
        [
            "# 15X-R1 Retry Recommendation",
            "",
            f"## Decision: **{decision}**",
            "",
            "Rationale:",
            "",
            f"- Strict verbatim matcher behaves correctly on all calibration cases: `{summary['all_cases_correct']}`.",
            f"- Recorded 15X cal_uncertainty paraphrase stays a failure; a verbatim source span would pass: `{summary['recorded_15x_failure_preserved']}`.",
            f"- Prompt contract hardened (verbatim required, null+uncertainty fallback): `{summary['prompt_contract_hardened']}` / `{summary['evidence_text_verbatim_required']}`.",
            f"- No embeddings / LLM judge used; paraphrase still rejected: `{not summary['embedding_or_llm_judge_used']}`.",
            "",
            (
                "The prompt now instructs Vertex to copy evidence_text verbatim (or null + uncertainty). "
                "A single bounded 15X live re-run is appropriate; the validator remains strict and will "
                "stop on the first non-verbatim/paraphrased evidence."
                if ready
                else "Evidence validator or prompt hardening is not yet conclusive; do not re-run live."
            ),
            "",
            "Note: re-running 15X live is a separate, explicitly-gated operator action; this block does not run it.",
            "",
        ]
    )


def _implementation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-EVIDENCE-MATCH-CALIBRATION-15X-R1",
            "",
            f"- evidence_match_calibration_passed: `{summary['evidence_match_calibration_passed']}`",
            f"- exact / whitespace / substring pass: `{summary['exact_evidence_match_pass_count']}` / "
            f"`{summary['normalized_whitespace_match_pass_count']}` / `{summary['substring_evidence_match_pass_count']}`",
            f"- paraphrase / inferred / wrong-section / null-empty rejected: "
            f"`{summary['paraphrase_rejected_count']}` / `{summary['inferred_evidence_rejected_count']}` / "
            f"`{summary['wrong_section_rejected_count']}` / `{summary['null_or_empty_rejected_count']}`",
            f"- recorded_15x_failure_preserved: `{summary['recorded_15x_failure_preserved']}`",
            f"- prompt_contract_hardened: `{summary['prompt_contract_hardened']}` | evidence_text_verbatim_required: `{summary['evidence_text_verbatim_required']}`",
            f"- embedding_or_llm_judge_used: `{summary['embedding_or_llm_judge_used']}`",
            f"- live_call_made: `{summary['live_call_made']}` | external_api_used: `{summary['external_api_used']}` | "
            f"active_written_count: `{summary['active_written_count']}` | auto_accept_true_count: `{summary['auto_accept_true_count']}`",
            f"- privacy_result: `{summary['privacy_result']}` | billing_check_pending: `{summary['billing_check_pending']}`",
            "",
            "## Decision",
            "",
            "- Evidence anchoring stays STRICT: only a whitespace/Unicode-normalized verbatim substring of the",
            "  source body (and section, when constrained) is accepted. Paraphrase/synonym/inferred evidence is rejected.",
            "- The Vertex semantic prompt contract now requires evidence_text copied verbatim, or null + uncertainty.",
            "- No embeddings, no LLM judge, no fuzzy 'close enough' matching.",
            "",
        ]
    )


def main() -> int:
    report = build_report()
    summary = report["summary"]
    cases = report["cases"]

    privacy_ok = _privacy_scan(summary) and _privacy_scan(cases) and all(
        "evidence_verbatim" in PROMPT_REQUIRED_PHRASES for _ in [0]
    )
    summary["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"cases": cases, "synthetic_source": SYNTHETIC_SOURCE}, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(summary), encoding="utf-8")
    RETRY_MD.write_text(_retry_markdown(summary), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(summary), encoding="utf-8")

    ready = all(
        [
            summary["evidence_match_calibration_passed"] is True,
            summary["all_cases_correct"] is True,
            summary["recorded_15x_failure_preserved"] is True,
            summary["prompt_contract_hardened"] is True,
            summary["evidence_text_verbatim_required"] is True,
            summary["embedding_or_llm_judge_used"] is False,
            summary["hallucinated_field_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_semantic_evidence_match_calibration_15x_r1_ready" if ready else "medai_vertex_semantic_evidence_match_calibration_15x_r1_not_ready")
    print(
        json.dumps(
            {
                "evidence_match_calibration_passed": summary["evidence_match_calibration_passed"],
                "recorded_15x_failure_preserved": summary["recorded_15x_failure_preserved"],
                "prompt_contract_hardened": summary["prompt_contract_hardened"],
                "evidence_text_verbatim_required": summary["evidence_text_verbatim_required"],
                "embedding_or_llm_judge_used": summary["embedding_or_llm_judge_used"],
                "paraphrase_rejected_count": summary["paraphrase_rejected_count"],
                "inferred_evidence_rejected_count": summary["inferred_evidence_rejected_count"],
                "wrong_section_rejected_count": summary["wrong_section_rejected_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "privacy_result": summary["privacy_result"],
                "retry_recommendation": "READY_TO_RERUN_15X_ONCE" if ready else "DO_NOT_RERUN_15X",
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
