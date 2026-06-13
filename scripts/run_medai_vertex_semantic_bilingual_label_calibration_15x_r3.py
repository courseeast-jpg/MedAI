#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-BILINGUAL-LABEL-CALIBRATION-15X-R3 (no-live).

Makes candidate-label matching robust to deterministic, explicitly-declared
source-derived label aliases (e.g. bilingual ``pH`` for canonical
``pH (bilingual)``) while keeping evidence anchoring strictly verbatim and
hallucination blocking intact. No provider call, no live gate, no embeddings,
no LLM judge, no fuzzy similarity.

Validates the label matcher against synthetic cases and the recorded 15X-R2
``cal_cyrillic`` failure shape, confirms ``cal_uncertainty`` (15X-R1) protection
is preserved, and emits a retry recommendation.
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
    evidence_text_is_source_verbatim,
    label_matches_candidate,
)
from execution.vertex_semantic_calibration_batch import (
    build_calibration_fixtures,
    validate_calibration_response,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_bilingual_label_calibration_15x_r3"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "label_match_cases.json"
MATRIX_MD = REPORT_DIR / "label_match_matrix.md"
RETRY_MD = REPORT_DIR / "retry_recommendation.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession")

CANONICAL = "pH (bilingual)"
ALIASES = ["pH", "рН"]

# Deterministic label-match calibration cases.
CASES = [
    {"case": "exact_canonical_label", "label": "pH (bilingual)", "expect_pass": True, "kind": "exact"},
    {"case": "declared_alias_latin_pH", "label": "pH", "expect_pass": True, "kind": "declared_alias"},
    {"case": "declared_alias_cyrillic", "label": "рН", "expect_pass": True, "kind": "declared_alias"},
    {"case": "declared_alias_whitespace_normalized", "label": "  pH  ", "expect_pass": True, "kind": "unicode_alias"},
    {"case": "undeclared_alias_ph_level", "label": "ph level", "expect_pass": False, "kind": "undeclared"},
    {"case": "undeclared_alias_lowercase_ph_word", "label": "acidity", "expect_pass": False, "kind": "undeclared"},
    {"case": "unrelated_label_glucose", "label": "Glucose", "expect_pass": False, "kind": "unrelated"},
]


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str, ensure_ascii=True)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _run_label_cases() -> list[dict[str, Any]]:
    out = []
    for c in CASES:
        observed = label_matches_candidate(c["label"], CANONICAL, ALIASES)
        out.append({**c, "observed_pass": observed, "correct": observed == c["expect_pass"]})
    return out


def _section_cases() -> dict[str, Any]:
    """Section must still match even when the label alias matches."""
    fx = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_cyrillic")

    def resp(label: str, section: str, evidence: str) -> dict[str, Any]:
        return {
            "package_family": fx.package_family,
            "review_required": True,
            "auto_accept": False,
            "semantic_findings": [{
                "label": label, "value": "6.0", "source_section": section,
                "evidence_text": evidence, "uncertainty": "x", "unknown_value": False, "source_faithful": True,
            }],
        }

    src_evidence = "Анализ мочи pH 6.0"
    return {
        "declared_alias_right_section": validate_calibration_response(resp("pH", "Urinalysis", src_evidence), fx),
        "declared_alias_wrong_section": validate_calibration_response(resp("pH", "Other", src_evidence), fx),
        "canonical_right_section": validate_calibration_response(resp("pH (bilingual)", "Urinalysis", src_evidence), fx),
        "undeclared_label_right_section": validate_calibration_response(resp("acidity", "Urinalysis", src_evidence), fx),
        "alias_but_paraphrased_evidence": validate_calibration_response(resp("pH", "Urinalysis", "urine pH six"), fx),
    }


def _cal_uncertainty_protection() -> dict[str, Any]:
    fx = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_uncertainty")
    src = fx.source_visible_body
    return {
        "paraphrase_still_rejected": evidence_text_is_source_verbatim("possible mild finding uncertain", src) is False,
        "verbatim_still_passes": evidence_text_is_source_verbatim("possible mild finding, uncertain", src) is True,
    }


def build_report() -> dict[str, Any]:
    label_cases = _run_label_cases()
    section_cases = _section_cases()
    uncertainty = _cal_uncertainty_protection()

    cyrillic_fixture = next(f for f in build_calibration_fixtures() if f.fixture_id == "cal_cyrillic")
    cyrillic_aliases = cyrillic_fixture.candidate_facts[0].get("accepted_label_aliases", [])

    summary = {
        "block": "MEDAI-VERTEX-SEMANTIC-BILINGUAL-LABEL-CALIBRATION-15X-R3",
        "exact_label_match_pass_count": sum(1 for c in label_cases if c["kind"] == "exact" and c["observed_pass"]),
        "declared_alias_match_pass_count": sum(1 for c in label_cases if c["kind"] == "declared_alias" and c["observed_pass"]),
        "unicode_alias_match_pass_count": sum(1 for c in label_cases if c["kind"] == "unicode_alias" and c["observed_pass"]),
        "undeclared_alias_rejected_count": sum(1 for c in label_cases if c["kind"] == "undeclared" and not c["observed_pass"]),
        "unrelated_label_rejected_count": sum(1 for c in label_cases if c["kind"] == "unrelated" and not c["observed_pass"]),
        "wrong_section_rejected_count": int(
            section_cases["declared_alias_wrong_section"]["verbatim_evidence_anchor_pass"] is False
        ),
        "all_label_cases_correct": all(c["correct"] for c in label_cases),
        "cal_cyrillic_natural_label_supported": (
            section_cases["declared_alias_right_section"]["verbatim_evidence_anchor_pass"] is True
            and "pH" in cyrillic_aliases
        ),
        "cal_cyrillic_undeclared_drift_rejected": section_cases["undeclared_label_right_section"]["verbatim_evidence_anchor_pass"] is False,
        "evidence_anchor_strictness_preserved": (
            section_cases["alias_but_paraphrased_evidence"]["verbatim_evidence_anchor_pass"] is False
            and uncertainty["paraphrase_still_rejected"]
            and uncertainty["verbatim_still_passes"]
        ),
        "cal_uncertainty_protection_preserved": uncertainty["paraphrase_still_rejected"] and uncertainty["verbatim_still_passes"],
        "fuzzy_or_embedding_or_llm_label_judge_used": False,
        "live_call_made": False,
        "external_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "review_required": True,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }
    summary["bilingual_label_calibration_passed"] = (
        summary["all_label_cases_correct"]
        and summary["exact_label_match_pass_count"] >= 1
        and summary["declared_alias_match_pass_count"] >= 1
        and summary["unicode_alias_match_pass_count"] >= 1
        and summary["undeclared_alias_rejected_count"] >= 1
        and summary["unrelated_label_rejected_count"] >= 1
        and summary["wrong_section_rejected_count"] >= 1
        and summary["cal_cyrillic_natural_label_supported"]
        and summary["cal_cyrillic_undeclared_drift_rejected"]
        and summary["evidence_anchor_strictness_preserved"]
        and summary["cal_uncertainty_protection_preserved"]
    )
    return {"summary": summary, "label_cases": label_cases, "uncertainty_protection": uncertainty, "cyrillic_aliases": cyrillic_aliases}


def _matrix_markdown(s: dict[str, Any]) -> str:
    keys = [
        "bilingual_label_calibration_passed", "exact_label_match_pass_count", "declared_alias_match_pass_count",
        "unicode_alias_match_pass_count", "undeclared_alias_rejected_count", "unrelated_label_rejected_count",
        "wrong_section_rejected_count", "evidence_anchor_strictness_preserved", "cal_uncertainty_protection_preserved",
        "cal_cyrillic_natural_label_supported", "cal_cyrillic_undeclared_drift_rejected",
        "fuzzy_or_embedding_or_llm_label_judge_used", "live_call_made", "external_api_used",
        "active_written_count", "active_mkb_record_created_count", "auto_accept_true_count",
        "review_required", "privacy_result", "billing_check_pending",
    ]
    return "\n".join(
        ["# 15X-R3 bilingual label calibration matrix", "", "| Metric | Value |", "| --- | --- |",
         *[f"| {k} | `{s[k]}` |" for k in keys], "",
         "Deterministic explicit aliases only; verbatim evidence anchoring preserved; no fuzzy/embeddings/LLM judge.", ""]
    )


def _retry_markdown(s: dict[str, Any]) -> str:
    ready = s["bilingual_label_calibration_passed"]
    decision = "READY_TO_RERUN_15X_R4_ONCE" if ready else "DO_NOT_RERUN_15X"
    return "\n".join(
        [
            "# 15X-R3 Retry Recommendation",
            "",
            f"## Decision: **{decision}**",
            "",
            "Rationale:",
            "",
            f"- Declared bilingual aliases (`pH`, `рН`) for canonical `pH (bilingual)` accepted; undeclared/unrelated drift rejected: `{s['all_label_cases_correct']}`.",
            f"- cal_cyrillic natural source-derived label (`pH`) now supported; undeclared drift still fails: "
            f"`{s['cal_cyrillic_natural_label_supported']}` / `{s['cal_cyrillic_undeclared_drift_rejected']}`.",
            f"- Verbatim evidence anchoring preserved (paraphrase still fails, incl. cal_uncertainty): `{s['evidence_anchor_strictness_preserved']}`.",
            f"- No fuzzy / embeddings / LLM label judge: `{not s['fuzzy_or_embedding_or_llm_label_judge_used']}`.",
            "",
            (
                "Label matching now accepts only explicit, locally-declared, deterministic source-derived "
                "aliases while evidence remains strictly verbatim. A single bounded 15X re-run (R4) is "
                "appropriate; the validator still stops on undeclared label drift or non-verbatim evidence."
                if ready
                else "Label matching remains ambiguous; do not re-run live."
            ),
            "",
            "Note: re-running 15X live is a separate, explicitly-gated operator action; this block does not run it.",
            "",
        ]
    )


def _implementation_markdown(s: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-BILINGUAL-LABEL-CALIBRATION-15X-R3",
            "",
            f"- bilingual_label_calibration_passed: `{s['bilingual_label_calibration_passed']}`",
            f"- exact / declared-alias / unicode-alias pass: `{s['exact_label_match_pass_count']}` / "
            f"`{s['declared_alias_match_pass_count']}` / `{s['unicode_alias_match_pass_count']}`",
            f"- undeclared / unrelated / wrong-section rejected: `{s['undeclared_alias_rejected_count']}` / "
            f"`{s['unrelated_label_rejected_count']}` / `{s['wrong_section_rejected_count']}`",
            f"- cal_cyrillic natural label supported / undeclared drift rejected: "
            f"`{s['cal_cyrillic_natural_label_supported']}` / `{s['cal_cyrillic_undeclared_drift_rejected']}`",
            f"- evidence_anchor_strictness_preserved: `{s['evidence_anchor_strictness_preserved']}` | "
            f"cal_uncertainty_protection_preserved: `{s['cal_uncertainty_protection_preserved']}`",
            f"- fuzzy_or_embedding_or_llm_label_judge_used: `{s['fuzzy_or_embedding_or_llm_label_judge_used']}`",
            f"- live_call_made: `{s['live_call_made']}` | external_api_used: `{s['external_api_used']}` | "
            f"active_written_count: `{s['active_written_count']}` | auto_accept_true_count: `{s['auto_accept_true_count']}`",
            f"- privacy_result: `{s['privacy_result']}` | billing_check_pending: `{s['billing_check_pending']}`",
            "",
            "## Decision",
            "",
            "- Candidate-label matching now accepts the canonical label or an explicit, locally-declared,",
            "  deterministic alias (whitespace/Unicode-normalized). Undeclared/inferred/fuzzy labels are rejected.",
            "- cal_cyrillic declares aliases `pH` and `рН` for canonical `pH (bilingual)`.",
            "- Evidence anchoring stays strictly verbatim (15X-R1); paraphrase and cal_uncertainty protection intact.",
            "- No embeddings, LLM judge, or fuzzy similarity.",
            "",
        ]
    )


def main() -> int:
    report = build_report()
    summary = report["summary"]
    privacy_ok = _privacy_scan(summary) and _privacy_scan(report["label_cases"])
    summary["privacy_result"] = "passed" if privacy_ok else "failed"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    CASES_JSON.write_text(json.dumps({"label_cases": report["label_cases"], "canonical": CANONICAL, "aliases": ALIASES}, indent=2, ensure_ascii=True), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(summary), encoding="utf-8")
    RETRY_MD.write_text(_retry_markdown(summary), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(summary), encoding="utf-8")

    ready = all(
        [
            summary["bilingual_label_calibration_passed"] is True,
            summary["all_label_cases_correct"] is True,
            summary["cal_cyrillic_natural_label_supported"] is True,
            summary["cal_cyrillic_undeclared_drift_rejected"] is True,
            summary["evidence_anchor_strictness_preserved"] is True,
            summary["cal_uncertainty_protection_preserved"] is True,
            summary["fuzzy_or_embedding_or_llm_label_judge_used"] is False,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["privacy_result"] == "passed",
        ]
    )
    print("medai_vertex_semantic_bilingual_label_calibration_15x_r3_ready" if ready else "medai_vertex_semantic_bilingual_label_calibration_15x_r3_not_ready")
    print(
        json.dumps(
            {
                "bilingual_label_calibration_passed": summary["bilingual_label_calibration_passed"],
                "cal_cyrillic_natural_label_supported": summary["cal_cyrillic_natural_label_supported"],
                "cal_cyrillic_undeclared_drift_rejected": summary["cal_cyrillic_undeclared_drift_rejected"],
                "evidence_anchor_strictness_preserved": summary["evidence_anchor_strictness_preserved"],
                "cal_uncertainty_protection_preserved": summary["cal_uncertainty_protection_preserved"],
                "fuzzy_or_embedding_or_llm_label_judge_used": summary["fuzzy_or_embedding_or_llm_label_judge_used"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "privacy_result": summary["privacy_result"],
                "retry_recommendation": "READY_TO_RERUN_15X_R4_ONCE" if ready else "DO_NOT_RERUN_15X",
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
