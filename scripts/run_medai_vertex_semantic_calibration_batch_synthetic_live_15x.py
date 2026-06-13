#!/usr/bin/env python3
"""MEDAI-VERTEX-SEMANTIC-CALIBRATION-BATCH-SYNTHETIC-LIVE-15X.

Bounded live Vertex calibration batch over a broader synthetic/redacted fixture
set (one live call per fixture, <=20, no retries, stop on first failure). Posts
only {contents, generationConfig}; JSON-only; temperature=0; maxOutputTokens<=512.
Produces a per-call quality + token/cost ledger. No active MKB writes;
review_required=true; auto_accept=false. Never logs credentials/tokens.

Refuses unless MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED=YES.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.vertex_semantic_calibration_batch import (
    COST_TABLE_NOTE,
    INPUT_USD_PER_1K,
    LIVE_ENV,
    MAX_LIVE_CALLS,
    OUTPUT_USD_PER_1K,
    fixtures_public_dict,
    run_calibration_batch,
)

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_semantic_calibration_batch_synthetic_live_15x"
SUMMARY_JSON = REPORT_DIR / "summary.json"
FIXTURES_JSON = REPORT_DIR / "calibration_fixtures.json"
RESULTS_JSON = REPORT_DIR / "calibration_results.json"
MATRIX_MD = REPORT_DIR / "calibration_matrix.md"
LEDGER_CSV = REPORT_DIR / "token_cost_ledger.csv"
LEDGER_JSON = REPORT_DIR / "token_cost_ledger.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/", "DOB", "MRN", "Accession")
LEDGER_COLUMNS = [
    "fixture_id", "package_family", "category", "provider_route", "model",
    "prompt_token_count", "output_token_count", "total_token_count",
    "estimated_input_cost_usd", "estimated_output_cost_usd", "estimated_total_cost_usd",
    "provider_response_received", "schema_validation_pass", "hallucinated_field_count",
    "privacy_result", "status",
]


def _privacy_scan(payload: Any) -> bool:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return not any(token in blob for token in FORBIDDEN_TOKENS)


def _ledger_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in agg["per_call_results"]:
        rows.append({col: r.get(col, "") for col in LEDGER_COLUMNS})
    return rows


def _ledger_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LEDGER_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def _matrix_markdown(agg: dict[str, Any]) -> str:
    keys = [
        "status", "fixture_count", "live_call_count", "provider_response_received_count",
        "schema_validation_pass_count", "verbatim_evidence_anchor_pass_count",
        "source_visible_body_preserved_count", "evidence_anchor_preserved_count",
        "candidate_facts_separated_count", "unknown_values_explicit_count", "uncertainty_flags_visible_count",
        "posted_body_allowed_top_level_keys_only_count", "review_required_count", "hallucinated_field_count",
        "total_prompt_tokens", "total_output_tokens", "total_token_count_all_calls",
        "estimated_total_cost_usd_all_calls", "estimated_cost_ceiling_usd", "failed_call_count",
        "auto_accept_true_count", "active_written_count", "active_mkb_record_created_count",
        "live_call_made", "external_api_used", "privacy_result", "billing_check_pending",
        "stopped_early", "stop_reason",
    ]
    return "\n".join(
        [
            "# 15X Vertex calibration batch matrix",
            "",
            f"- Cost table: {agg['cost_table_note']}",
            f"- Category breakdown: `{agg['category_breakdown']}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{agg[key]}` |" for key in keys],
            "",
            "Synthetic/redacted fixtures only; bounded <=20 live calls; no active writes; no auto-accept.",
            "",
        ]
    )


def _implementation_markdown(agg: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEDAI-VERTEX-SEMANTIC-CALIBRATION-BATCH-SYNTHETIC-LIVE-15X",
            "",
            f"- Status: `{agg['status']}`",
            f"- Fixtures / live calls / responses: `{agg['fixture_count']}` / `{agg['live_call_count']}` / `{agg['provider_response_received_count']}`",
            f"- Schema pass / hallucinated: `{agg['schema_validation_pass_count']}` / `{agg['hallucinated_field_count']}`",
            f"- Source body / evidence / candidate-sep / unknown / uncertainty preserved: "
            f"`{agg['source_visible_body_preserved_count']}` / `{agg['evidence_anchor_preserved_count']}` / "
            f"`{agg['candidate_facts_separated_count']}` / `{agg['unknown_values_explicit_count']}` / `{agg['uncertainty_flags_visible_count']}`",
            f"- Allowed-keys-only / review-required: `{agg['posted_body_allowed_top_level_keys_only_count']}` / `{agg['review_required_count']}`",
            f"- Tokens (prompt/output/total): `{agg['total_prompt_tokens']}` / `{agg['total_output_tokens']}` / `{agg['total_token_count_all_calls']}`",
            f"- Estimated cost all calls / ceiling (USD): `{agg['estimated_total_cost_usd_all_calls']}` / `{agg['estimated_cost_ceiling_usd']}`",
            f"- Cost constants: input ${INPUT_USD_PER_1K}/1K, output ${OUTPUT_USD_PER_1K}/1K tokens.",
            f"- auto_accept_true_count: `{agg['auto_accept_true_count']}` | active_written_count: `{agg['active_written_count']}` | "
            f"active_mkb_record_created_count: `{agg['active_mkb_record_created_count']}`",
            f"- live_call_made: `{agg['live_call_made']}` | external_api_used: `{agg['external_api_used']}` | "
            f"privacy_result: `{agg['privacy_result']}` | billing_check_pending: `{agg['billing_check_pending']}`",
            f"- stopped_early: `{agg['stopped_early']}` | stop_reason: `{agg['stop_reason']}`",
            "",
            "## Safety",
            "",
            "- Synthetic/redacted fixtures only; <=20 live calls (one per fixture); no retries; stop on first failure.",
            "- Every posted body contains only `contents` + `generationConfig`; no MedAI metadata sent.",
            "- No active MKB writes; output review-bound; no auto-accept; no provider call without the 15X gate.",
            "- Token/cost ledger uses documented conservative local constants; billing_check_pending=true.",
            "",
        ]
    )


def main() -> int:
    agg = run_calibration_batch()
    ledger_rows = _ledger_rows(agg)

    privacy_ok = _privacy_scan(agg) and _privacy_scan(fixtures_public_dict()) and _privacy_scan(ledger_rows)
    agg["privacy_result"] = "passed" if (agg["privacy_result"] == "passed" and privacy_ok) else "failed"

    # R4 provenance markers (this fresh run supersedes prior failed live artifacts).
    agg["run_label"] = "15X-R4"
    agg["prior_15x_failed"] = True
    agg["prior_15x_r2_failed"] = True
    agg["rerun_after_15x_r1_evidence_hardening"] = True
    agg["rerun_after_15x_r3_bilingual_label_calibration"] = True
    agg["live_script_run_exactly_once"] = True

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in agg.items() if k != "per_call_results"}
    summary["per_call_status"] = [{"fixture_id": r["fixture_id"], "status": r["status"]} for r in agg["per_call_results"]]
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    FIXTURES_JSON.write_text(json.dumps({"fixtures": fixtures_public_dict()}, indent=2), encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    MATRIX_MD.write_text(_matrix_markdown(agg), encoding="utf-8")
    LEDGER_CSV.write_text(_ledger_csv(ledger_rows), encoding="utf-8")
    LEDGER_JSON.write_text(json.dumps({"cost_table_note": COST_TABLE_NOTE, "rows": ledger_rows}, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation_markdown(agg), encoding="utf-8")

    print(json.dumps(
        {
            "status": agg["status"],
            "fixture_count": agg["fixture_count"],
            "live_call_count": agg["live_call_count"],
            "schema_validation_pass_count": agg["schema_validation_pass_count"],
            "verbatim_evidence_anchor_pass_count": agg["verbatim_evidence_anchor_pass_count"],
            "hallucinated_field_count": agg["hallucinated_field_count"],
            "total_token_count_all_calls": agg["total_token_count_all_calls"],
            "estimated_total_cost_usd_all_calls": agg["estimated_total_cost_usd_all_calls"],
            "active_written_count": agg["active_written_count"],
            "auto_accept_true_count": agg["auto_accept_true_count"],
            "live_call_made": agg["live_call_made"],
            "external_api_used": agg["external_api_used"],
            "privacy_result": agg["privacy_result"],
            "billing_check_pending": agg["billing_check_pending"],
            "stopped_early": agg["stopped_early"],
            "stop_reason": agg["stop_reason"],
        },
        indent=2,
    ))
    # Gate-absent refusal is exit 0 (not a failure); a real live FAIL is nonzero.
    if agg["status"].startswith("BLOCKED"):
        return 0
    return 0 if agg["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
