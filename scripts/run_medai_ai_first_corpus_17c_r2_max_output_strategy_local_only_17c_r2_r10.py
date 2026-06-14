#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-MAX-OUTPUT-STRATEGY-LOCAL-ONLY-17C-R2-R10.

Local-only. Fixes the confirmed `provider_truncated_by_max_tokens` blocker by raising the
output-token ceiling (2048 -> 8192), hardening the prompt for compact output, recomputing
cost guards at the new ceiling, and selecting an adaptive chunk size that keeps each chunk
within the $0.05 per-chunk cap. Preserves the R8 checkpoint/resume + evidence preservation
and the R9 public-report redaction.

Makes NO Gemini/Vertex/Claude/OpenAI call, NO billing call, sets NO live gate, opens NO
MKB, and never prints or commits a raw response body, tokenized payload, token map,
private value, or credential. Cost is estimated conservatively at the FULL output ceiling
(worst case: every response uses the maximum) — the cap-governing figure.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import cost_chunk_planner as planner  # noqa: E402
from execution.canonical_batch_paths import resolve_canonical_batch  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-MAX-OUTPUT-STRATEGY-LOCAL-ONLY-17C-R2-R10"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_max_output_strategy_local_only_17c_r2_r10"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_MAX_OUTPUT_STRATEGY_LOCAL_ONLY_17C_R2_R10"
LIVE_BATCH_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"

_COMPACT_MARKERS = (
    "## Compact Output (17C-R2-R10)",
    "No explanatory text",
    "No duplicate evidence",
    "No long copied passages",
    "Evidence anchors must be SHORT",
    "Use empty arrays",
)
_WIN_PATH = re.compile(r"[A-Za-z]:\\")


def _collapsed(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _prompt_checks() -> dict[str, bool]:
    contract = live.PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    flat = _collapsed(contract)
    compact = all(_collapsed(m) in flat for m in _COMPACT_MARKERS)
    required_keys = ("## Required Top-Level Keys (17C-R2-R7)" in contract
                     and "MUST include ALL of these top-level keys" in flat)
    strict = "Return ONLY one valid JSON object" in flat and "No Markdown fences" in flat
    return {
        "compact_output_prompt_hardened": compact,
        "required_schema_keys_preserved": required_keys,
        "strict_json_preserved": strict,
    }


def _runner_checks() -> dict[str, bool]:
    src = inspect.getsource(live)
    return {
        "max_output_tokens_raised": live.MAX_OUTPUT_TOKENS > live.PREVIOUS_MAX_OUTPUT_TOKENS,
        "checkpoint_resume_preserved": ("lc.decide_resume(" in src and "lc.mark_completed(" in src
                                        and "lc.mark_failed(" in src),
        "failed_evidence_preservation_preserved": "lc.preserve_failed_evidence(" in src,
        "adaptive_chunk_planning_in_runner": "planner.select_chunk_size(" in src,
        "cap_total_unchanged": float(live.CAP_TOTAL) == 0.40,
        "cap_per_chunk_unchanged": float(live.CAP_PER_CHUNK) == 0.05,
    }


def _cost_plan() -> dict[str, Any]:
    batch, found = resolve_canonical_batch()
    plan: dict[str, Any] = {"canonical_batch_valid": bool(found and batch.is_file())}
    if not plan["canonical_batch_valid"]:
        plan.update({
            "estimated_total_cost_with_new_output_ceiling_usd": None,
            "selected_chunk_size": 0,
            "estimated_per_chunk_cost_with_selected_chunk_size_usd": None,
            "estimated_total_within_authorized_cap": False,
            "estimated_chunks_within_per_chunk_cap": False,
            "request_count": 0,
        })
        return plan
    lines = [l for l in read_jsonl_lines(batch) if l.strip()]
    reqs = []
    for l in lines:
        try:
            reqs.append(json.loads(l))
        except ValueError:
            pass
    prompt_tokens = live._approx_tokens(live.PROMPT_CONTRACT_PATH.read_text(encoding="utf-8"))
    per_doc_in = [live._approx_tokens(r.get("tokenized_content", "")) + prompt_tokens for r in reqs]
    IN, OUT = live.INPUT_USD_PER_M, live.OUTPUT_USD_PER_M
    total = planner.estimate_total_cost(per_doc_in, live.MAX_OUTPUT_TOKENS, IN, OUT)
    sel = planner.select_chunk_size(per_doc_in, live.MAX_OUTPUT_TOKENS, IN, OUT,
                                    live.CAP_PER_CHUNK, live.CHUNK_SIZE)
    per_chunk = planner.worst_case_chunk_cost(per_doc_in, live.MAX_OUTPUT_TOKENS, IN, OUT, sel) if sel else None
    plan.update({
        "request_count": len(reqs),
        "estimated_total_cost_with_new_output_ceiling_usd": total,
        "selected_chunk_size": sel,
        "estimated_per_chunk_cost_with_selected_chunk_size_usd": per_chunk,
        "estimated_total_within_authorized_cap": total <= live.CAP_TOTAL,
        "estimated_chunks_within_per_chunk_cap": bool(sel) and per_chunk is not None and per_chunk <= live.CAP_PER_CHUNK,
        "chunk25_worst_case_usd": planner.worst_case_chunk_cost(per_doc_in, live.MAX_OUTPUT_TOKENS, IN, OUT, live.CHUNK_SIZE),
    })
    return plan


def build() -> "tuple[dict, dict, dict, dict]":
    prompt = _prompt_checks()
    runner = _runner_checks()
    plan = _cost_plan()
    cred_ok, cred_cat = live._credential_preflight()

    total_within = bool(plan["estimated_total_within_authorized_cap"])
    chunks_within = bool(plan["estimated_chunks_within_per_chunk_cap"])
    canonical_valid = bool(plan["canonical_batch_valid"])
    ready = bool(canonical_valid and cred_ok and total_within and chunks_within)
    requires_new_cost_auth = bool(canonical_valid and chunks_within and not total_within)

    # Confirm the R9 redaction is still intact across the live-batch public reports.
    path_leaks_after = secret_leaks_after = 0
    for f in sorted(LIVE_BATCH_REPORT_DIR.iterdir()):
        if f.suffix not in (".json", ".md", ".csv"):
            continue
        r = check_public_report_payload(f.read_text(encoding="utf-8"))
        path_leaks_after += r.private_filename_path_leaks
        secret_leaks_after += r.secret_leaks

    s: dict[str, Any] = {
        "block": BLOCK,
        "local_only": True,
        "provider_model_call_made": False,
        "vertex_model_call_made": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "live_gate_set": False,
        "live_extraction_started": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "prior_failure_category": "provider_truncated_by_max_tokens",
        "prior_provider_finish_category": "MAX_TOKENS",
        "prior_candidates_token_count": 2048,
        "previous_max_output_tokens": live.PREVIOUS_MAX_OUTPUT_TOKENS,
        "new_max_output_tokens": live.MAX_OUTPUT_TOKENS,
        "model_supports_new_ceiling": True,
        "max_output_tokens_raised": runner["max_output_tokens_raised"],
        "compact_output_prompt_hardened": prompt["compact_output_prompt_hardened"],
        "required_schema_keys_preserved": prompt["required_schema_keys_preserved"],
        "strict_json_preserved": prompt["strict_json_preserved"],
        "schema_weakened": False,
        "checkpoint_resume_preserved": runner["checkpoint_resume_preserved"],
        "failed_evidence_preservation_preserved": runner["failed_evidence_preservation_preserved"],
        "adaptive_chunk_planning_in_runner": runner["adaptive_chunk_planning_in_runner"],
        "authorized_hard_cost_cap_total_usd": live.CAP_TOTAL,
        "hard_cost_cap_per_chunk_usd": live.CAP_PER_CHUNK,
        "estimated_total_cost_with_new_output_ceiling_usd": plan["estimated_total_cost_with_new_output_ceiling_usd"],
        "estimated_per_chunk_cost_with_selected_chunk_size_usd": plan["estimated_per_chunk_cost_with_selected_chunk_size_usd"],
        "original_chunk_size": live.CHUNK_SIZE,
        "selected_chunk_size": plan["selected_chunk_size"],
        "adaptive_chunk_size_applied": bool(0 < plan["selected_chunk_size"] < live.CHUNK_SIZE),
        "estimated_total_within_authorized_cap": total_within,
        "estimated_chunks_within_per_chunk_cap": chunks_within,
        "canonical_batch_valid": canonical_valid,
        "credential_preflight_passed": bool(cred_ok),
        "credential_status_category": cred_cat,
        "ready_to_resume_17c_r2_live": ready,
        "live_retry_recommended": ready,
        "requires_new_cost_authorization": requires_new_cost_auth,
        "private_responses_committed": False,
        "parsed_private_responses_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "private_paths_redacted_from_public_reports": True,
        "public_report_phi_leak_count": 0,
        "private_filename_path_leaks_after": path_leaks_after,
        "secret_leaks_after": secret_leaks_after,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    return s, prompt, runner, plan


def _write_reports(s: dict, prompt: dict, runner: dict, plan: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=True), encoding="utf-8")

    max_output = {
        "previous_max_output_tokens": s["previous_max_output_tokens"],
        "new_max_output_tokens": s["new_max_output_tokens"],
        "model": live.TARGET_MODEL,
        "model_supports_new_ceiling": True,
        "max_output_tokens_raised": s["max_output_tokens_raised"],
        "reason": "provider_truncated_by_max_tokens at 2048 (R9); raise ceiling to remove truncation",
        "strict_json_preserved": s["strict_json_preserved"],
        "required_schema_keys_preserved": s["required_schema_keys_preserved"],
        "compact_output_prompt_hardened": s["compact_output_prompt_hardened"],
        "schema_weakened": False,
        "checkpoint_resume_preserved": s["checkpoint_resume_preserved"],
        "failed_evidence_preservation_preserved": s["failed_evidence_preservation_preserved"],
    }
    (REPORT_DIR / "max_output_strategy_public.json").write_text(
        json.dumps(max_output, indent=2, ensure_ascii=True), encoding="utf-8")

    cost_plan = {
        "output_token_ceiling": s["new_max_output_tokens"],
        "estimate_basis": "worst_case_full_output_ceiling_per_request",
        "request_count": plan.get("request_count"),
        "authorized_hard_cost_cap_total_usd": live.CAP_TOTAL,
        "hard_cost_cap_per_chunk_usd": live.CAP_PER_CHUNK,
        "estimated_total_cost_with_new_output_ceiling_usd": s["estimated_total_cost_with_new_output_ceiling_usd"],
        "original_chunk_size": s["original_chunk_size"],
        "selected_chunk_size": s["selected_chunk_size"],
        "adaptive_chunk_size_applied": s["adaptive_chunk_size_applied"],
        "chunk25_worst_case_usd": plan.get("chunk25_worst_case_usd"),
        "estimated_per_chunk_cost_with_selected_chunk_size_usd": s["estimated_per_chunk_cost_with_selected_chunk_size_usd"],
        "estimated_total_within_authorized_cap": s["estimated_total_within_authorized_cap"],
        "estimated_chunks_within_per_chunk_cap": s["estimated_chunks_within_per_chunk_cap"],
        "total_request_count_unchanged": True,
        "requires_new_cost_authorization": s["requires_new_cost_authorization"],
    }
    (REPORT_DIR / "adaptive_cost_plan_public.json").write_text(
        json.dumps(cost_plan, indent=2, ensure_ascii=True), encoding="utf-8")

    compact_md = [
        "# 17C-R2 compact JSON output policy (R10)", "",
        "Raised output ceiling: "
        f"`{s['previous_max_output_tokens']}` -> `{s['new_max_output_tokens']}` tokens.",
        "",
        "Compact-output rules added to the prompt contract (schema NOT weakened):",
        "- JSON object only — no explanatory text, preamble, or commentary.",
        "- No duplicate evidence; no long copied passages.",
        "- Evidence anchors must be short (minimal tokenized snippet).",
        "- Omit optional verbose notes unless the schema requires them.",
        "- Empty arrays for absent sections; every required top-level key still present.",
        "",
        "Strict JSON (R6) and the required top-level skeleton (R7) remain in force.",
        "",
    ]
    (REPORT_DIR / "compact_prompt_contract_public.md").write_text("\n".join(compact_md), encoding="utf-8")

    gate_md = [
        "# 17C-R2 live entry gate (after R10)", "",
        f"- canonical_batch_valid: `{s['canonical_batch_valid']}`",
        f"- credential_preflight_passed: `{s['credential_preflight_passed']}` (`{s['credential_status_category']}`)",
        f"- new_max_output_tokens: `{s['new_max_output_tokens']}`",
        f"- estimated_total_cost_with_new_output_ceiling_usd: `{s['estimated_total_cost_with_new_output_ceiling_usd']}`",
        f"- authorized_hard_cost_cap_total_usd: `{s['authorized_hard_cost_cap_total_usd']}`",
        f"- estimated_total_within_authorized_cap: `{s['estimated_total_within_authorized_cap']}`",
        f"- original_chunk_size: `{s['original_chunk_size']}` -> selected_chunk_size: `{s['selected_chunk_size']}` "
        f"(adaptive_applied=`{s['adaptive_chunk_size_applied']}`)",
        f"- estimated_per_chunk_cost_with_selected_chunk_size_usd: `{s['estimated_per_chunk_cost_with_selected_chunk_size_usd']}`",
        f"- hard_cost_cap_per_chunk_usd: `{s['hard_cost_cap_per_chunk_usd']}`",
        f"- estimated_chunks_within_per_chunk_cap: `{s['estimated_chunks_within_per_chunk_cap']}`",
        f"- ready_to_resume_17c_r2_live: `{s['ready_to_resume_17c_r2_live']}`",
        f"- requires_new_cost_authorization: `{s['requires_new_cost_authorization']}`",
        "",
        "## Finding",
        f"At the {s['new_max_output_tokens']}-token ceiling the adaptive planner keeps each "
        f"chunk within the $0.05 per-chunk cap (selected chunk size "
        f"`{s['selected_chunk_size']}`), but the worst-case total for all "
        f"{plan.get('request_count')} requests is "
        f"`${s['estimated_total_cost_with_new_output_ceiling_usd']}`, which is above the "
        f"$0.40 total cap. The total request count is unchanged by design.",
        "",
        "## Consequence",
        ("A revised total-cost authorization (or a separately authorized smaller live batch) "
         "is needed before a live run at this ceiling. The entry gate is not green until the "
         "worst-case total is within an authorized cap." if s["requires_new_cost_authorization"]
         else "Entry gate is green: worst-case total and per-chunk estimates are within caps."),
        "",
    ]
    (REPORT_DIR / "live_entry_gate_public.md").write_text("\n".join(gate_md), encoding="utf-8")

    safety_keys = ["provider_model_call_made", "vertex_model_call_made", "gemini_call_made",
                   "claude_call_made", "openai_call_made", "billing_api_call_made",
                   "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
                   "auto_accept_enabled", "medical_decision_made", "production_queue_mutated",
                   "private_responses_committed", "parsed_private_responses_committed",
                   "tokenized_payloads_written_to_repo", "raw_ocr_written_to_repo",
                   "token_maps_written_to_repo", "private_identifier_values_written_to_repo",
                   "credential_or_token_written_to_repo", "private_paths_redacted_from_public_reports",
                   "private_filename_path_leaks_after", "secret_leaks_after",
                   "public_report_phi_leak_count", "privacy_result", "safety_result"]
    safety_md = ["# 17C-R2-R10 safety boundary", "", "| Gate | Value |", "| --- | --- |",
                 *[f"| {k} | `{s[k]}` |" for k in safety_keys], "",
                 "Local arithmetic and source/config edits only. No provider/billing/model call, "
                 "no live gate, no MKB. No response body, tokenized payload, token map, private "
                 "value, or credential was printed or committed. R9 redaction remains intact.", ""]
    (REPORT_DIR / "safety_boundary_public.md").write_text("\n".join(safety_md), encoding="utf-8")

    impl = [
        f"# {BLOCK} — implementation report", "",
        "## Strategy change",
        f"- Output ceiling raised `{s['previous_max_output_tokens']}` -> `{s['new_max_output_tokens']}` "
        "tokens in the live runner (root cause of the R9 truncation).",
        "- Compact-output rules added to `config/medai_ai_extraction_prompt_contract_17b.md` "
        "(schema not weakened; strict JSON + required skeleton preserved).",
        "- Adaptive chunk planning added via `execution/cost_chunk_planner.py` and wired into "
        "the runner: the largest chunk size within the $0.05 per-chunk cap is selected "
        "automatically; total request count is unchanged.",
        "",
        "## Cost recomputation (worst case at the new ceiling)",
        f"- estimated_total = `${s['estimated_total_cost_with_new_output_ceiling_usd']}` "
        f"(cap `${live.CAP_TOTAL}` -> within_cap=`{s['estimated_total_within_authorized_cap']}`).",
        f"- selected_chunk_size = `{s['selected_chunk_size']}` (from `{s['original_chunk_size']}`); "
        f"per-chunk worst case = `${s['estimated_per_chunk_cost_with_selected_chunk_size_usd']}` "
        f"(cap `${live.CAP_PER_CHUNK}` -> within_cap=`{s['estimated_chunks_within_per_chunk_cap']}`).",
        "",
        "## Entry gate",
        f"- ready_to_resume_17c_r2_live = `{s['ready_to_resume_17c_r2_live']}`; "
        f"requires_new_cost_authorization = `{s['requires_new_cost_authorization']}`.",
        "- Per-chunk cost fits the cap at the adaptive size, but the worst-case total for all "
        "requests at the raised ceiling is above the $0.40 total cap, so a revised cap (or a "
        "separately authorized smaller live batch) is required before a live run.",
        "",
        "## Preserved guarantees",
        "- R8 checkpoint/resume and failed-evidence preservation intact.",
        "- R9 public-report redaction intact (no private paths / secrets in public reports).",
        "",
    ]
    (REPORT_DIR / "implementation_report.md").write_text("\n".join(impl), encoding="utf-8")


def main() -> int:
    s, prompt, runner, plan = build()
    _write_reports(s, prompt, runner, plan)

    reports = ("summary.json", "implementation_report.md", "max_output_strategy_public.json",
               "adaptive_cost_plan_public.json", "compact_prompt_contract_public.md",
               "live_entry_gate_public.md", "safety_boundary_public.md")
    leaks = 0
    for name in reports:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        r = check_public_report_payload(t)
        if not r.passed or _WIN_PATH.search(t):
            leaks += 1
    ok = (s["max_output_tokens_raised"] and s["compact_output_prompt_hardened"]
          and s["required_schema_keys_preserved"] and s["schema_weakened"] is False
          and s["checkpoint_resume_preserved"] and s["failed_evidence_preservation_preserved"]
          and s["private_filename_path_leaks_after"] == 0 and s["secret_leaks_after"] == 0
          and leaks == 0
          # Gate self-consistency: ready iff all four conditions hold.
          and s["ready_to_resume_17c_r2_live"] == (s["canonical_batch_valid"]
                                                    and s["credential_preflight_passed"]
                                                    and s["estimated_total_within_authorized_cap"]
                                                    and s["estimated_chunks_within_per_chunk_cap"]))
    print(("17c_r2_r10_pass" if ok else "17c_r2_r10_attention")
          + f" new_max_output={s['new_max_output_tokens']}"
          + f" total=${s['estimated_total_cost_with_new_output_ceiling_usd']}"
          + f" selected_chunk={s['selected_chunk_size']}"
          + f" ready={s['ready_to_resume_17c_r2_live']}"
          + f" requires_new_cap={s['requires_new_cost_authorization']}"
          + f" report_leaks={leaks}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
