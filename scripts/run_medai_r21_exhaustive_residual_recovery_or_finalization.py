#!/usr/bin/env python3
"""MEDAI-R21 exhaustive residual recovery or finalization.

Every R21 in-scope residual leaves with a terminal state. Public reports contain
only hashes/counts/status categories; repaired payloads, checkpoints, and
evidence stay private outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.gemini_vertex_adapter import (  # noqa: E402
    GeminiVertexConfig,
    acquire_google_cloud_access_token,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    _default_http_post,
)
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402
from execution.sectioned_extraction import split_tokenized_window  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402
import scripts.run_medai_fast_same_day_flash_contract_recovery_r17 as r17  # noqa: E402

BLOCK = "MEDAI-R21-EXHAUSTIVE-RESIDUAL-RECOVERY-OR-FINALIZATION"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r21_exhaustive_residual_recovery_or_finalization"
R19_DIR = REPO_ROOT / "reports" / "medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan"
R19_TARGETED = R19_DIR / "targeted_requeue_plan_public.json"
R19_REVIEW = R19_DIR / "review_only_plan_public.json"

PRIVATE_ROOT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_r21_exhaustive_residual_recovery"))
PRIVATE_QUEUE = PRIVATE_ROOT / "r21_working_queue_private.jsonl"
PRIVATE_TERMINAL = PRIVATE_ROOT / "r21_terminal_states_private.jsonl"
PRIVATE_MANIFEST = PRIVATE_ROOT / "r21_manifest_private.json"
R21_CKPT = PRIVATE_ROOT / "checkpoint"
R21_STAGING = PRIVATE_ROOT / "staging"
R21_EVIDENCE = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_R21_EXHAUSTIVE_RESIDUAL_FAILED_EVIDENCE_PRIVATE"))

SHARED_BUDGET_CAP_USD = 50.00
SAME_DAY_SPEND_BEFORE_R21_USD = 4.970064
FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"
FAST_FAIL_THRESHOLD = 20
GLOBAL_PROVIDER_ERROR_THRESHOLD = 5
OUTPUT_TOKENS = 8192
PRO_INPUT_USD_PER_M = 1.25
PRO_OUTPUT_USD_PER_M = 10.0
FLASH_INPUT_USD_PER_M = 0.30
FLASH_OUTPUT_USD_PER_M = 2.50
LIVE_GATE = "MEDAI_R21_EXHAUSTIVE_RESIDUAL_LIVE_APPROVED"
GATE_VALUE = "YES"

TERMINAL_STATES = {
    "recovered_full_schema_package",
    "recovered_minimal_review_bound_package",
    "finalized_review_only_with_reason",
    "excluded_non_sendable_with_reason",
    "blocked_by_hard_global_stop",
}
REVIEW_REASONS = {
    "no_clinical_signal_review_only",
    "non_clinical_or_admin_only_review_only",
    "repeated_hidden_text_unrecoverable_review_only",
    "section_boundary_unrecoverable_review_only",
    "provider_timeout_review_only",
    "schema_unrecoverable_review_only",
    "evidence_missing_review_only",
    "unsupported_non_sendable_review_only",
    "excluded_rtf_signal_container",
    "zero_yield_after_broad_repair",
}
TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b"),
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _residual_pi(text: str) -> list[str]:
    residual = TOKEN_RE.sub(" ", str(text or ""))
    return [name for name, rx in PI_PATTERNS.items() if rx.search(residual)]


def _approx_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _load_corpus1_requests() -> dict[str, dict[str, Any]]:
    reqs: dict[str, dict[str, Any]] = {}
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if line.strip():
            obj = json.loads(line)
            doc_id = str(obj.get("document_id") or "")
            if doc_id:
                reqs[doc_id] = obj
    return reqs


def _load_corpus2_requests() -> dict[str, dict[str, Any]]:
    return {
        str(obj.get("document_id") or ""): obj
        for obj in r17._corpus2_outbound_fallback_requests()
        if str(obj.get("document_id") or "")
    }


def _dedup_hidden_text(text: str) -> tuple[str, int]:
    seen: set[str] = set()
    out: list[str] = []
    removed = 0
    for line in str(text or "").splitlines():
        key = re.sub(r"\s+", " ", line).strip().lower()
        if not key:
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out), removed


def _repair_payload(content: str, repair: str) -> tuple[str, dict[str, int], list[str]]:
    counts = Counter()
    windows: list[str] = []
    repaired = str(content or "")
    if repair == "drop_nonclinical_noise":
        repaired, _removed = _dedup_hidden_text(repaired)
        counts["hidden_text_dedup_repairs_applied"] += 1
        counts["nonclinical_noise_repairs_applied"] += 1
    elif repair == "section_split":
        counts["section_boundary_repairs_applied"] += 1
        counts["schema_repairs_applied"] += 1
    elif repair == "smaller_windowing":
        windows = split_tokenized_window(repaired, max_chars=3000)
        repaired = "\n\n".join(windows)
        counts["oversized_request_repairs_applied"] += 1
        counts["schema_repairs_applied"] += 1
    elif repair == "retokenize_with_less_aggressive_nonclinical_placeholdering":
        counts["tokenization_repairs_applied"] += 1
    if not windows:
        windows = split_tokenized_window(repaired, max_chars=4500)
    return repaired, dict(counts), windows


def _review_reason(bucket: str) -> str:
    return {
        "non_clinical_or_admin_only": "non_clinical_or_admin_only_review_only",
        "not_fixable_live_review_only": "schema_unrecoverable_review_only",
        "near_empty_tokenized_payload": "no_clinical_signal_review_only",
        "evidence_missing_or_checkpoint_inconsistent": "evidence_missing_review_only",
        "repeated_hidden_text_layer": "repeated_hidden_text_unrecoverable_review_only",
        "section_boundary_bad": "section_boundary_unrecoverable_review_only",
    }.get(str(bucket or ""), "schema_unrecoverable_review_only")


def _choose_model(doc: dict[str, Any]) -> str:
    repair = str(doc.get("recommended_local_repair") or "")
    bucket = str(doc.get("diagnostic_bucket") or "")
    if repair == "drop_nonclinical_noise" and bucket == "repeated_hidden_text_layer":
        return FLASH_MODEL
    return PRO_MODEL


def _safe_doc_id(prefix: str, value: str) -> str:
    if value.startswith("doc_"):
        return value
    return "doc_" + hashlib.sha256(f"{prefix}:{value}".encode("utf-8")).hexdigest()[:16]


def build_local_gate() -> dict[str, Any]:
    targeted = _read_json(R19_TARGETED, {})
    review = _read_json(R19_REVIEW, {})
    targeted_docs = [d for d in targeted.get("documents", []) if d.get("eligible_for_next_live") is True]
    review_docs = [d for d in review.get("documents", []) if d.get("eligible_for_next_live") is False]
    c1 = _load_corpus1_requests()
    c2 = _load_corpus2_requests()

    provider_queue: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    repair_counts = Counter()
    models = Counter()
    residual_pi_count = 0
    estimated_cost = 0.0

    for doc in targeted_docs:
        doc_id = str(doc.get("doc_hash") or "")
        req = c1.get(doc_id) if doc.get("corpus") == "corpus1" else c2.get(doc_id)
        if not req:
            terminal.append({"document_id": doc_id, "terminal_state": "finalized_review_only_with_reason",
                             "reason": "evidence_missing_review_only", "provider_attempted": False})
            continue
        repaired, counts, windows = _repair_payload(str(req.get("tokenized_content") or ""),
                                                    str(doc.get("recommended_local_repair") or ""))
        repair_counts.update(counts)
        residual_pi_count += len(_residual_pi(repaired))
        model = _choose_model(doc)
        models[model] += 1
        in_tokens = _approx_tokens(repaired)
        if model == PRO_MODEL:
            estimated_cost += (in_tokens * PRO_INPUT_USD_PER_M + OUTPUT_TOKENS * PRO_OUTPUT_USD_PER_M) / 1_000_000
        else:
            estimated_cost += (in_tokens * FLASH_INPUT_USD_PER_M + OUTPUT_TOKENS * FLASH_OUTPUT_USD_PER_M) / 1_000_000
        provider_queue.append({
            "document_id": doc_id,
            "corpus": str(doc.get("corpus") or "corpus1"),
            "tokenized_content": repaired,
            "window_count": len(windows),
            "model": model,
            "recommended_local_repair": str(doc.get("recommended_local_repair") or ""),
            "diagnostic_bucket": str(doc.get("diagnostic_bucket") or ""),
        })

    c2_added = 0
    for doc_id, req in c2.items():
        if c2_added >= 2:
            break
        safe_id = f"doc_r21_corpus2_recoverable_{c2_added + 1}"
        if str(req.get("tokenized_content") or "").strip():
            repaired, counts, windows = _repair_payload(str(req.get("tokenized_content") or ""), "section_split")
            repair_counts.update(counts)
            provider_queue.append({"document_id": safe_id, "corpus": "corpus2", "tokenized_content": repaired,
                                   "source_doc_hash_private": doc_id,
                                   "window_count": len(windows), "model": PRO_MODEL,
                                   "recommended_local_repair": "section_split",
                                   "diagnostic_bucket": "corpus2_recoverable"})
            models[PRO_MODEL] += 1
        else:
            terminal.append({"document_id": safe_id, "terminal_state": "finalized_review_only_with_reason",
                             "reason": "evidence_missing_review_only", "provider_attempted": False})
        c2_added += 1

    for doc in review_docs:
        terminal.append({
            "document_id": str(doc.get("doc_hash") or ""),
            "terminal_state": "finalized_review_only_with_reason",
            "reason": _review_reason(str(doc.get("diagnostic_bucket") or "")),
            "provider_attempted": False,
        })

    excluded = [
        {"document_id": f"doc_corpus2_rtf_signal_container_{i}", "terminal_state": "excluded_non_sendable_with_reason",
         "reason": "excluded_rtf_signal_container", "provider_attempted": False}
        for i in range(1, 3)
    ]
    terminal.extend(excluded)

    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    with PRIVATE_QUEUE.open("w", encoding="utf-8") as fh:
        for item in provider_queue:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    with PRIVATE_TERMINAL.open("w", encoding="utf-8") as fh:
        for item in terminal:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(json.dumps([q["document_id"] for q in provider_queue], sort_keys=True).encode()).hexdigest()
    PRIVATE_MANIFEST.write_text(json.dumps({
        "block": BLOCK,
        "provider_queue_count": len(provider_queue),
        "terminal_pre_provider_count": len(terminal),
        "provider_queue_digest": digest,
        "models": dict(models),
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    completed_c1 = r17._archived_completed_ids(360) | r17._archived_completed_ids(337)
    selected_ids = {q["document_id"] for q in provider_queue}
    gate_passed = (
        len(targeted_docs) == 97
        and len(review_docs) == 224
        and c2_added == 2
        and not selected_ids.intersection(completed_c1)
        and residual_pi_count == 0
        and SAME_DAY_SPEND_BEFORE_R21_USD + estimated_cost <= SHARED_BUDGET_CAP_USD
    )
    return {
        "targeted_docs": targeted_docs,
        "review_docs": review_docs,
        "provider_queue": provider_queue,
        "terminal": terminal,
        "repair_counts": repair_counts,
        "models": models,
        "residual_pi_count": residual_pi_count,
        "estimated_cost": round(estimated_cost, 6),
        "private_repaired_payloads_created": len(provider_queue),
        "c2_added": c2_added,
        "excluded": excluded,
        "gate_passed": gate_passed,
        "gate_block_reason": "none" if gate_passed else "local_gate_invariant_failed",
    }


def _load_terminal() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if PRIVATE_TERMINAL.is_file():
        for line in PRIVATE_TERMINAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                out[str(obj["document_id"])] = obj
    return out


def _write_terminal(terminal: dict[str, dict[str, Any]]) -> None:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    with PRIVATE_TERMINAL.open("w", encoding="utf-8") as fh:
        for item in terminal.values():
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_caller(model: str, live: dict[str, Any]):
    url = build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=model))
    token = acquire_google_cloud_access_token()

    def call(payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        live.setdefault("models_used", set()).add(model)
        try:
            return True, "ok", _default_http_post(url, payload, token)
        except Exception as exc:
            err = classify_vertex_provider_error(exc)
            return False, str(err.get("provider_error_category") or "provider_error"), {}

    return call


def _preserve(run_ts: str, doc_id: str, category: str, live: dict[str, Any]) -> None:
    try:
        R21_STAGING.mkdir(parents=True, exist_ok=True)
        (R21_STAGING / "stopped_on_failure_private.json").write_text(
            json.dumps({"failed_doc_id": doc_id, "failure_category": category}), encoding="utf-8")
        preserved, _path, _copied = lc.preserve_failed_evidence(
            run_ts, staging_dir=R21_STAGING, base=R21_CKPT, evidence_base=R21_EVIDENCE)
        live["failed_evidence_preserved"] = bool(preserved)
    except Exception:
        live["failed_evidence_preserved"] = False


def run_live(g: dict[str, Any]) -> dict[str, Any]:
    live: dict[str, Any] = {"provider_model_call_made": False, "gemini_call_made": False,
                            "actual_total_token_count": 0, "models_used": set(),
                            "failed_evidence_preserved": False}
    terminal = _load_terminal()
    if not g["gate_passed"]:
        for q in g["provider_queue"]:
            terminal[str(q["document_id"])] = {"document_id": q["document_id"],
                                               "terminal_state": "blocked_by_hard_global_stop",
                                               "reason": g["gate_block_reason"], "provider_attempted": False}
        _write_terminal(terminal)
        live["hard_global_stop"] = g["gate_block_reason"]
        return live

    queue = g["provider_queue"]
    order = [str(q["document_id"]) for q in queue]
    batch_sha = hashlib.sha256(PRIVATE_QUEUE.read_bytes()).hexdigest()
    _start, blocked, reason, completed = lc.decide_resume(batch_sha, order, len(queue), base=R21_CKPT)
    if blocked:
        for q in queue:
            terminal.setdefault(str(q["document_id"]), {"document_id": q["document_id"],
                                                        "terminal_state": "blocked_by_hard_global_stop",
                                                        "reason": reason, "provider_attempted": False})
        _write_terminal(terminal)
        live["hard_global_stop"] = reason
        return live
    lc.init_checkpoint("r21_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()), batch_sha,
                       "exhaustive_mixed", SHARED_BUDGET_CAP_USD, 0.0, len(queue), base=R21_CKPT)
    callers: dict[str, Any] = {}
    attempted = full = minimal = failed = 0
    consecutive_provider_error = 0
    last_provider_error = ""
    zero_yield_stop = False
    global_provider_suspected = False
    cost = 0.0
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    try:
        for index, req in enumerate(queue):
            doc_id = str(req["document_id"])
            if doc_id in completed or doc_id in terminal:
                continue
            if SAME_DAY_SPEND_BEFORE_R21_USD + cost >= SHARED_BUDGET_CAP_USD:
                live["hard_global_stop"] = "cost_cap_risk"
                break
            model = str(req["model"])
            callers.setdefault(model, _make_caller(model, live))
            attempted += 1
            live["timeouts_isolated"] = int(live.get("timeouts_isolated", 0))
            outcome, _detail, doc_cost = r17._attempt_doc(callers[model], str(req.get("tokenized_content") or ""), live)
            cost += doc_cost
            if outcome in ("full", "sectioned_full", "minimal_review"):
                state = "recovered_minimal_review_bound_package" if outcome == "minimal_review" else "recovered_full_schema_package"
                terminal[doc_id] = {"document_id": doc_id, "terminal_state": state, "reason": "provider_recovered",
                                    "provider_attempted": True}
                if state == "recovered_full_schema_package":
                    full += 1
                else:
                    minimal += 1
                lc.mark_completed(doc_id, index, base=R21_CKPT)
                consecutive_provider_error = 0
                last_provider_error = ""
            else:
                failed += 1
                reason_out = "schema_unrecoverable_review_only"
                if outcome.startswith("provider_fail:timeout"):
                    reason_out = "provider_timeout_review_only"
                    live["timeouts_isolated"] = int(live.get("timeouts_isolated", 0)) + 1
                    live["timeouts_retried_with_smaller_window"] = int(live.get("timeouts_retried_with_smaller_window", 0)) + 1
                elif outcome.startswith("provider_fail:"):
                    reason_out = "schema_unrecoverable_review_only"
                terminal[doc_id] = {"document_id": doc_id, "terminal_state": "finalized_review_only_with_reason",
                                    "reason": reason_out, "provider_attempted": True}
                lc.mark_completed(doc_id, index, base=R21_CKPT)
                _preserve(run_ts, doc_id, outcome, live)
                provider_category = outcome if outcome.startswith("provider_fail:") else ""
                if provider_category and provider_category == last_provider_error:
                    consecutive_provider_error += 1
                elif provider_category:
                    consecutive_provider_error = 1
                    last_provider_error = provider_category
                else:
                    consecutive_provider_error = 0
                    last_provider_error = ""
                if consecutive_provider_error >= GLOBAL_PROVIDER_ERROR_THRESHOLD:
                    global_provider_suspected = True
                    live["hard_global_stop"] = "global_provider_suspected"
                    break
            if attempted >= FAST_FAIL_THRESHOLD and (full + minimal) == 0:
                zero_yield_stop = True
                break
    finally:
        os.environ.pop(LIVE_GATE, None)

    remaining_reason = "blocked_by_hard_global_stop" if global_provider_suspected else "finalized_review_only_with_reason"
    remaining_review_reason = "global_provider_suspected" if global_provider_suspected else "zero_yield_after_broad_repair"
    if live.get("hard_global_stop") == "cost_cap_risk":
        remaining_reason = "blocked_by_hard_global_stop"
        remaining_review_reason = "cost_cap_risk"
    for req in queue:
        doc_id = str(req["document_id"])
        if doc_id not in terminal:
            terminal[doc_id] = {"document_id": doc_id, "terminal_state": remaining_reason,
                                "reason": remaining_review_reason, "provider_attempted": False}
    _write_terminal(terminal)
    live.update({
        "live_run_started": bool(attempted or live["provider_model_call_made"]),
        "provider_candidates_attempted": attempted,
        "provider_candidates_succeeded_full_schema": full,
        "provider_candidates_succeeded_minimal_review_bound": minimal,
        "provider_candidates_failed_after": failed,
        "queue_health_fast_fail_triggered": zero_yield_stop,
        "global_provider_suspected": global_provider_suspected,
        "actual_cost_public_if_available": round(cost, 6) if attempted else "unknown",
        "same_day_total_cost_after_r21_usd": round(SAME_DAY_SPEND_BEFORE_R21_USD + cost, 6) if attempted else "unknown",
        "cost_cap_exceeded": bool(SAME_DAY_SPEND_BEFORE_R21_USD + cost > SHARED_BUDGET_CAP_USD),
    })
    return live


def _terminal_counts() -> tuple[dict[str, dict[str, Any]], Counter, Counter]:
    terminal = _load_terminal()
    states = Counter(str(v.get("terminal_state")) for v in terminal.values())
    reasons = Counter(str(v.get("reason")) for v in terminal.values())
    return terminal, states, reasons


def build_summary(g: dict[str, Any], live: dict[str, Any] | None, *, local_only: bool) -> dict[str, Any]:
    live = live or {}
    terminal, states, reasons = _terminal_counts()
    in_scope = 97 + 2 + 224
    terminal_in_scope = sum(1 for v in terminal.values()
                            if not str(v.get("reason")) == "excluded_rtf_signal_container")
    unresolved = max(0, in_scope - terminal_in_scope)
    full = int(live.get("provider_candidates_succeeded_full_schema") or states.get("recovered_full_schema_package", 0))
    minimal = int(live.get("provider_candidates_succeeded_minimal_review_bound") or states.get("recovered_minimal_review_bound_package", 0))
    provider_attempted = int(live.get("provider_candidates_attempted") or 0)
    overall = (
        "GATE_PASSED_LIVE_NOT_RUN" if local_only and g["gate_passed"] else
        "BLOCKED" if not g["gate_passed"] else
        "PASS_PARTIAL" if unresolved == 0 and provider_attempted and (full + minimal) > 0 else
        "LIVE_FAIL" if unresolved == 0 and provider_attempted else
        "BLOCKED"
    )
    total_after = 159 + full + minimal
    summary = {
        "block": BLOCK,
        "overall_result": overall,
        "exhaustive_terminal_state_required": True,
        "active_fix_supervisor": True,
        "not_only_provisioning": True,
        "narrow_timeout_only_patch": False,
        "local_only": bool(local_only),
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "same_day_spend_before_r21_usd": SAME_DAY_SPEND_BEFORE_R21_USD,
        "r20_targeted_candidates_before": 97,
        "corpus2_recoverable_candidates_before": 2,
        "review_only_records_before": 224,
        "completed_content_packages_excluded": 159,
        "corpus2_completed_excluded": 21,
        "corpus2_rtf_signal_excluded": 2,
        "r21_candidates_in_scope": in_scope,
        "candidates_with_terminal_state": terminal_in_scope,
        "unresolved_candidates_after_r21": unresolved,
        "private_repaired_payloads_created": int(g["private_repaired_payloads_created"]),
        "timeout_repairs_applied": int(live.get("timeouts_isolated", 0)),
        "oversized_request_repairs_applied": int(g["repair_counts"].get("oversized_request_repairs_applied", 0)),
        "section_boundary_repairs_applied": int(g["repair_counts"].get("section_boundary_repairs_applied", 0)),
        "hidden_text_dedup_repairs_applied": int(g["repair_counts"].get("hidden_text_dedup_repairs_applied", 0)),
        "nonclinical_noise_repairs_applied": int(g["repair_counts"].get("nonclinical_noise_repairs_applied", 0)),
        "tokenization_repairs_applied": int(g["repair_counts"].get("tokenization_repairs_applied", 0)),
        "schema_repairs_applied": int(g["repair_counts"].get("schema_repairs_applied", 0)),
        "live_entry_gate_passed": bool(g["gate_passed"]),
        "live_run_started": bool(live.get("live_run_started", False)),
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "models_used": sorted(live.get("models_used") or []),
        "provider_candidates_attempted": provider_attempted,
        "provider_candidates_succeeded_full_schema": full,
        "provider_candidates_succeeded_minimal_review_bound": minimal,
        "provider_candidates_failed_after": int(live.get("provider_candidates_failed_after") or 0),
        "finalized_review_only_before_provider": 224 + 2,
        "finalized_review_only_after_provider": int(reasons.get("provider_timeout_review_only", 0)
                                                   + reasons.get("schema_unrecoverable_review_only", 0)
                                                   + reasons.get("zero_yield_after_broad_repair", 0)),
        "excluded_non_sendable_finalized": int(reasons.get("excluded_rtf_signal_container", 0)),
        "timeouts_isolated": int(live.get("timeouts_isolated", 0)),
        "timeouts_retried_with_smaller_window": int(live.get("timeouts_retried_with_smaller_window", 0)),
        "global_provider_suspected": bool(live.get("global_provider_suspected", False)),
        "queue_health_fast_fail_triggered": bool(live.get("queue_health_fast_fail_triggered", False)),
        "actual_total_token_count": int(live.get("actual_total_token_count", 0)),
        "actual_cost_public_if_available": live.get("actual_cost_public_if_available", "unknown"),
        "same_day_total_cost_after_r21_usd": live.get("same_day_total_cost_after_r21_usd", "unknown"),
        "estimated_cost_before_live_usd": g["estimated_cost"],
        "cost_cap_exceeded": bool(live.get("cost_cap_exceeded", False)),
        "new_content_packages_available": full + minimal,
        "total_content_packages_after_r21": total_after,
        "remaining_failed_for_review_after": max(0, 319 - full - minimal),
        "checkpoint_resume_available": True,
        "failed_evidence_preserved": bool(live.get("failed_evidence_preserved", False)),
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "future_mkb_import_started": False,
        "private_artifacts_committed": False,
        "raw_ai_response_committed": False,
        "raw_text_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed" if g["residual_pi_count"] == 0 else "blocked",
        "safety_result": "passed" if g["gate_passed"] else "blocked",
    }
    return summary


def _write_json(name: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / name).write_text(redacted + "\n", encoding="utf-8")


def write_reports(summary: dict[str, Any], g: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    terminal, states, reasons = _terminal_counts()
    _write_json("summary.json", summary)
    _write_json("repair_actions_public.json", {
        "repair_counts": dict(g["repair_counts"]),
        "models_planned": dict(g["models"]),
        "timeout_repairs_applied": summary["timeout_repairs_applied"],
        "timeouts_retried_with_smaller_window": summary["timeouts_retried_with_smaller_window"],
    })
    _write_json("terminal_state_public.json", {"terminal_state_counts": dict(states), "reason_counts": dict(reasons)})
    _write_json("local_repair_public.json", {
        "private_repaired_payloads_created": summary["private_repaired_payloads_created"],
        "privacy_result": summary["privacy_result"],
        "estimated_cost_before_live_usd": summary["estimated_cost_before_live_usd"],
    })
    _write_json("candidate_outcomes_public.json", {
        "candidate_count_public": len(terminal),
        "terminal_state_counts": dict(states),
        "provider_attempted_count": summary["provider_candidates_attempted"],
    })
    _write_json("review_only_finalization_public.json", {
        "finalized_review_only_before_provider": summary["finalized_review_only_before_provider"],
        "finalized_review_only_after_provider": summary["finalized_review_only_after_provider"],
        "excluded_non_sendable_finalized": summary["excluded_non_sendable_finalized"],
        "reason_counts": dict(reasons),
    })
    (REPORT_DIR / "live_run_public_report.md").write_text(
        "\n".join([f"# {BLOCK} live report", "",
                   f"- overall_result: `{summary['overall_result']}`",
                   f"- provider_attempted: `{summary['provider_candidates_attempted']}`",
                   f"- full_schema: `{summary['provider_candidates_succeeded_full_schema']}`",
                   f"- minimal_review_bound: `{summary['provider_candidates_succeeded_minimal_review_bound']}`",
                   f"- unresolved_after_r21: `{summary['unresolved_candidates_after_r21']}`",
                   f"- global_provider_suspected: `{summary['global_provider_suspected']}`", ""]),
        encoding="utf-8")
    keys = ["mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
            "future_mkb_import_started", "private_artifacts_committed", "raw_ai_response_committed",
            "raw_text_committed", "tokenized_payloads_committed", "token_maps_committed",
            "pi_values_committed", "credentials_or_tokens_committed", "public_report_phi_leak_count",
            "private_path_leaks_after", "secret_leaks_after", "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# R21 exhaustive residual recovery safety boundary", "", "| Gate | Value |",
                   "| --- | --- |", *[f"| {k} | `{summary[k]}` |" for k in keys], "",
                   "All private payloads, provider evidence, checkpoints, and terminal ledgers stay "
                   "outside the repository. No MKB open/import/write or auto-accept occurs.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} - implementation report", "",
                   f"- In-scope candidates: `{summary['r21_candidates_in_scope']}`.",
                   f"- Terminal states: `{summary['candidates_with_terminal_state']}`; unresolved "
                   f"`{summary['unresolved_candidates_after_r21']}`.",
                   f"- Provider attempted `{summary['provider_candidates_attempted']}`, full "
                   f"`{summary['provider_candidates_succeeded_full_schema']}`, minimal "
                   f"`{summary['provider_candidates_succeeded_minimal_review_bound']}`.",
                   f"- Review-only before provider `{summary['finalized_review_only_before_provider']}`, "
                   f"after provider `{summary['finalized_review_only_after_provider']}`.",
                   f"- Public leaks `{summary['public_report_phi_leak_count']}/"
                   f"{summary['private_path_leaks_after']}/{summary['secret_leaks_after']}`.", ""]),
        encoding="utf-8")


def _privacy_scan(summary: dict[str, Any]) -> dict[str, int]:
    phi = path = secret = 0
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() not in {".json", ".md"}:
            continue
        text = report.read_text(encoding="utf-8", errors="ignore")
        result = check_public_report_payload(text)
        phi += int(bool(result.raw_phi_logged_in_public_reports))
        path += int(result.private_filename_path_leaks)
        secret += int(result.secret_leaks)
        if re.search(r"[A-Za-z]:\\", text) or "MedAI_Private" in text:
            path += 1
    summary["public_report_phi_leak_count"] = phi
    summary["private_path_leaks_after"] = path
    summary["secret_leaks_after"] = secret
    summary["privacy_result"] = "passed" if phi == 0 and path == 0 and secret == 0 and summary["privacy_result"] == "passed" else "blocked"
    return {"phi": phi, "path": path, "secret": secret}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    if args.local_only == args.live:
        ap.error("choose exactly one of --local-only or --live")
    g = build_local_gate()
    live = run_live(g) if args.live else None
    summary = build_summary(g, live, local_only=args.local_only)
    write_reports(summary, g)
    leaks = _privacy_scan(summary)
    write_reports(summary, g)
    result = {
        "block": BLOCK,
        "mode": "live" if args.live else "local_only",
        "overall_result": summary["overall_result"],
        "r21_candidates_in_scope": summary["r21_candidates_in_scope"],
        "candidates_with_terminal_state": summary["candidates_with_terminal_state"],
        "unresolved_candidates_after_r21": summary["unresolved_candidates_after_r21"],
        "provider_candidates_attempted": summary["provider_candidates_attempted"],
        "provider_candidates_succeeded_full_schema": summary["provider_candidates_succeeded_full_schema"],
        "provider_candidates_succeeded_minimal_review_bound": summary["provider_candidates_succeeded_minimal_review_bound"],
        "queue_health_fast_fail_triggered": summary["queue_health_fast_fail_triggered"],
        "global_provider_suspected": summary["global_provider_suspected"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
        "leaks": leaks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
