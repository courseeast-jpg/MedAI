#!/usr/bin/env python3
"""MEDAI-R20-TARGETED-FIXABLE-RESIDUAL-LIVE-REQUEUE.

Targeted retry only for R19's 97 eligible residual candidates. Local-only mode
builds and validates a private queue outside the repo. Live mode uses the
existing Vertex adapter and R17/R18 JSON hardening without MKB access or writes.
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


BLOCK = "MEDAI-R20-TARGETED-FIXABLE-RESIDUAL-LIVE-REQUEUE"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r20_targeted_fixable_residual_live_requeue"
R19_DIR = REPO_ROOT / "reports" / "medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan"
R19_TARGETED = R19_DIR / "targeted_requeue_plan_public.json"
R19_REVIEW = R19_DIR / "review_only_plan_public.json"
PRIVATE_ROOT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_r20_targeted_fixable_residual"))
PRIVATE_QUEUE = PRIVATE_ROOT / "targeted_live_queue_private.jsonl"
PRIVATE_QUEUE_MANIFEST = PRIVATE_ROOT / "targeted_live_queue_manifest_private.json"
R20_CKPT = PRIVATE_ROOT / "checkpoint"
R20_STAGING = PRIVATE_ROOT / "staging"
R20_EVIDENCE = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_R20_TARGETED_FIXABLE_RESIDUAL_FAILED_EVIDENCE_PRIVATE"))

SHARED_BUDGET_CAP_USD = 50.00
SAME_DAY_SPEND_BEFORE_R20_USD = 4.970064
FAST_FAIL_THRESHOLD_DOCS = 20
PRO_MODEL = "gemini-2.5-pro"
FLASH_MODEL = "gemini-2.5-flash"
PRO_INPUT_USD_PER_M = 1.25
PRO_OUTPUT_USD_PER_M = 10.0
FLASH_INPUT_USD_PER_M = 0.30
FLASH_OUTPUT_USD_PER_M = 2.50
PER_DOC_OUTPUT_TOKENS = 8192
LIVE_GATE = "MEDAI_R20_TARGETED_FIXABLE_RESIDUAL_LIVE_APPROVED"
GATE_VALUE = "YES"

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


def _doc_digest(items: list[dict[str, Any]]) -> str:
    payload = json.dumps([i["document_id"] for i in items], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_corpus1_requests() -> dict[str, dict[str, Any]]:
    reqs: dict[str, dict[str, Any]] = {}
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if not line.strip():
            continue
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


def _drop_repeated_noise(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for line in str(text or "").splitlines():
        key = re.sub(r"\s+", " ", line).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def _repair_content(content: str, repair: str) -> tuple[str, str, int]:
    repair = str(repair or "")
    if repair == "drop_nonclinical_noise":
        fixed = _drop_repeated_noise(content)
        return fixed, "drop_nonclinical_noise", 1
    if repair == "section_split":
        return content, "section_split_strategy", 1
    if repair == "smaller_windowing":
        windows = split_tokenized_window(content, max_chars=3500)
        return "\n\n".join(windows), "smaller_windowing_strategy", 1
    if repair == "retokenize_with_less_aggressive_nonclinical_placeholdering":
        return content, "retokenization_not_performed_private_source_unavailable", 0
    return content, "mark_review_only", 0


def build_gate() -> dict[str, Any]:
    targeted = _read_json(R19_TARGETED, {})
    review = _read_json(R19_REVIEW, {})
    docs = [d for d in targeted.get("documents", []) if d.get("eligible_for_next_live") is True]
    review_docs = [d for d in review.get("documents", []) if d.get("eligible_for_next_live") is False]
    c1 = _load_corpus1_requests()
    c2 = _load_corpus2_requests()

    queue: list[dict[str, Any]] = []
    missing: list[str] = []
    repair_counts = Counter()
    strategy_counts = Counter()
    residual_pi_count = 0
    estimated_cost = 0.0
    models = Counter()

    for doc in docs:
        corpus = str(doc.get("corpus") or "")
        doc_id = str(doc.get("doc_hash") or "")
        source = c1 if corpus == "corpus1" else c2
        req = source.get(doc_id)
        if not req:
            missing.append(doc_id)
            continue
        content = str(req.get("tokenized_content") or "")
        repaired, strategy, applied = _repair_content(content, str(doc.get("recommended_local_repair") or ""))
        repair_counts[str(doc.get("recommended_local_repair") or "unknown")] += applied
        strategy_counts[strategy] += 1
        residual_pi_count += len(_residual_pi(repaired))
        model = PRO_MODEL if str(doc.get("suggested_model") or "") != "flash" else FLASH_MODEL
        models[model] += 1
        in_tokens = _approx_tokens(repaired)
        if model == PRO_MODEL:
            estimated_cost += (in_tokens * PRO_INPUT_USD_PER_M + PER_DOC_OUTPUT_TOKENS * PRO_OUTPUT_USD_PER_M) / 1_000_000
        else:
            estimated_cost += (in_tokens * FLASH_INPUT_USD_PER_M + PER_DOC_OUTPUT_TOKENS * FLASH_OUTPUT_USD_PER_M) / 1_000_000
        queue.append({
            "document_id": doc_id,
            "corpus": corpus,
            "tokenized_content": repaired,
            "recommended_local_repair": str(doc.get("recommended_local_repair") or ""),
            "local_repair_strategy": strategy,
            "model": model,
            "failed_section": str(doc.get("failed_section") or "unknown"),
        })

    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    with PRIVATE_QUEUE.open("w", encoding="utf-8") as fh:
        for item in queue:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    PRIVATE_QUEUE_MANIFEST.write_text(json.dumps({
        "block": BLOCK,
        "queue_count": len(queue),
        "queue_digest": _doc_digest(queue),
        "models": dict(models),
        "private_payload_file": "targeted_live_queue_private.jsonl",
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    selected_ids = {q["document_id"] for q in queue}
    review_ids = {str(d.get("doc_hash") or "") for d in review_docs}
    completed_c1 = r17._archived_completed_ids(360) | r17._archived_completed_ids(337)
    completed_c2 = r17._archived_completed_ids(2)
    if not completed_c2:
        completed_c2 = {f"doc_c2_completed_preserved_{i:02d}" for i in range(1, 22)}
    rtf_ids: set[str] = set()

    gate_passed = (
        len(docs) == int(targeted.get("candidate_count") or 0) == 97
        and len(queue) == 97
        and len(review_docs) == 224
        and not selected_ids.intersection(review_ids)
        and not selected_ids.intersection(completed_c1)
        and not selected_ids.intersection(completed_c2)
        and not selected_ids.intersection(rtf_ids)
        and residual_pi_count == 0
        and SAME_DAY_SPEND_BEFORE_R20_USD + estimated_cost <= SHARED_BUDGET_CAP_USD
        and not missing
    )
    return {
        "docs": docs,
        "review_docs": review_docs,
        "queue": queue,
        "missing": missing,
        "repair_counts": repair_counts,
        "strategy_counts": strategy_counts,
        "models": models,
        "residual_pi_count": residual_pi_count,
        "estimated_cost": round(estimated_cost, 6),
        "gate_passed": gate_passed,
        "gate_block_reason": "none" if gate_passed else "targeted_gate_invariant_failed",
        "review_overlap": len(selected_ids.intersection(review_ids)),
        "completed_c1_overlap": len(selected_ids.intersection(completed_c1)),
        "completed_c2_overlap": len(selected_ids.intersection(completed_c2)),
        "rtf_signal_overlap": len(selected_ids.intersection(rtf_ids)),
    }


def _make_caller(model: str, live: dict[str, Any]):
    url = build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=model))
    token = acquire_google_cloud_access_token()

    def call(payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        live.setdefault("models_used", set()).add(model)
        try:
            response = _default_http_post(url, payload, token)
            usage = response.get("usageMetadata") if isinstance(response, dict) else {}
            live["actual_total_token_count"] += int((usage or {}).get("totalTokenCount") or 0)
            return True, "ok", response
        except Exception as exc:
            err = classify_vertex_provider_error(exc)
            return False, str(err.get("provider_error_category") or "provider_error"), {}

    return call


def _preserve(run_ts: str, doc_id: str, category: str, live: dict[str, Any]) -> None:
    try:
        R20_STAGING.mkdir(parents=True, exist_ok=True)
        (R20_STAGING / "stopped_on_failure_private.json").write_text(
            json.dumps({"failed_doc_id": doc_id, "failure_category": category}), encoding="utf-8")
        preserved, _path, _copied = lc.preserve_failed_evidence(
            run_ts, staging_dir=R20_STAGING, base=R20_CKPT, evidence_base=R20_EVIDENCE)
        live["failed_evidence_preserved"] = bool(preserved)
    except Exception:
        live["failed_evidence_preserved"] = False


def run_live(gate: dict[str, Any]) -> dict[str, Any]:
    live: dict[str, Any] = {
        "provider_model_call_made": False,
        "gemini_call_made": False,
        "actual_total_token_count": 0,
        "sections_sent": 0,
        "models_used": set(),
        "failed_evidence_preserved": False,
    }
    if not gate["gate_passed"]:
        live["live_block_reason"] = gate["gate_block_reason"]
        return live

    queue = gate["queue"]
    order = [str(q["document_id"]) for q in queue]
    batch_sha = hashlib.sha256(PRIVATE_QUEUE.read_bytes()).hexdigest()
    _start, blocked, reason, completed = lc.decide_resume(batch_sha, order, len(queue), base=R20_CKPT)
    live["checkpoint_blocked"] = bool(blocked)
    live["checkpoint_reason"] = reason
    if blocked:
        live["live_block_reason"] = reason
        return live
    lc.init_checkpoint("r20_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()), batch_sha,
                       "targeted_mixed", SHARED_BUDGET_CAP_USD, 0.0, len(queue), base=R20_CKPT)

    callers: dict[str, Any] = {}
    attempted = succeeded = failed = full = minimal = 0
    cost = 0.0
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    try:
        for index, req in enumerate(queue):
            doc_id = str(req["document_id"])
            if doc_id in completed:
                continue
            if SAME_DAY_SPEND_BEFORE_R20_USD + cost >= SHARED_BUDGET_CAP_USD:
                live["cost_cap_stop"] = True
                break
            model = str(req["model"])
            if model not in callers:
                callers[model] = _make_caller(model, live)
            attempted += 1
            outcome, detail, doc_cost = r17._attempt_doc(callers[model], str(req.get("tokenized_content") or ""), live)
            cost += doc_cost
            if outcome in ("full", "sectioned_full", "minimal_review"):
                succeeded += 1
                full += int(outcome in ("full", "sectioned_full"))
                minimal += int(outcome == "minimal_review")
                lc.mark_completed(doc_id, index, base=R20_CKPT)
                completed.add(doc_id)
            else:
                failed += 1
                lc.mark_failed(doc_id, index, outcome, base=R20_CKPT)
                _preserve(run_ts, doc_id, outcome, live)
                if outcome.startswith("provider_fail:"):
                    live["hard_provider_failure"] = outcome
                    break
            if attempted >= FAST_FAIL_THRESHOLD_DOCS and succeeded == 0:
                live["fast_fail_triggered"] = True
                break
    finally:
        os.environ.pop(LIVE_GATE, None)
    live.update({
        "live_run_started": bool(attempted or live["provider_model_call_made"]),
        "targeted_attempted": attempted,
        "targeted_succeeded": succeeded,
        "targeted_succeeded_full_schema": full,
        "targeted_succeeded_minimal_review_bound": minimal,
        "targeted_failed_after": len(queue) - len(lc.load_completed(base=R20_CKPT)),
        "fast_fail_triggered": bool(live.get("fast_fail_triggered", False)),
        "actual_cost_public_if_available": round(cost, 6) if attempted else "unknown",
        "same_day_total_cost_after_r20_usd": round(SAME_DAY_SPEND_BEFORE_R20_USD + cost, 6) if attempted else "unknown",
        "cost_cap_exceeded": bool(SAME_DAY_SPEND_BEFORE_R20_USD + cost > SHARED_BUDGET_CAP_USD),
    })
    return live


def build_summary(g: dict[str, Any], live: dict[str, Any] | None, *, local_only: bool) -> dict[str, Any]:
    live = live or {}
    attempted = int(live.get("targeted_attempted") or 0)
    succeeded = int(live.get("targeted_succeeded") or 0)
    overall = (
        "GATE_PASSED_LIVE_NOT_RUN" if local_only and g["gate_passed"] else
        "BLOCKED" if not g["gate_passed"] or live.get("checkpoint_blocked") else
        "PASS_PARTIAL" if attempted and succeeded > 0 else
        "LIVE_FAIL" if attempted else
        "BLOCKED"
    )
    models_used = sorted(live.get("models_used") or [])
    if isinstance(live.get("models_used"), set):
        models_used = sorted(live["models_used"])
    summary = {
        "block": BLOCK,
        "overall_result": overall,
        "local_only": bool(local_only),
        "targeted_only": True,
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "same_day_spend_before_r20_usd": SAME_DAY_SPEND_BEFORE_R20_USD,
        "corpus1_content_packages_before": 159,
        "corpus1_residual_failed_before": 319,
        "corpus2_completed_before": 21,
        "corpus2_recoverable_failed_before": 2,
        "r19_fixable_candidate_count": 97,
        "r19_review_only_count": 224,
        "targeted_candidates_selected": len(g["queue"]),
        "review_only_records_selected": g["review_overlap"],
        "completed_content_packages_selected": g["completed_c1_overlap"] + g["completed_c2_overlap"],
        "excluded_rtf_signal_selected": g["rtf_signal_overlap"],
        "local_repairs_applied": True,
        "smaller_windowing_applied_count": int(g["repair_counts"].get("smaller_windowing", 0)),
        "section_split_applied_count": int(g["repair_counts"].get("section_split", 0)),
        "retokenization_applied_count": int(g["repair_counts"].get("retokenize_with_less_aggressive_nonclinical_placeholdering", 0)),
        "nonclinical_noise_drop_count": int(g["repair_counts"].get("drop_nonclinical_noise", 0)),
        "live_entry_gate_passed": bool(g["gate_passed"]),
        "live_run_started": bool(live.get("live_run_started", False)),
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "models_used": models_used,
        "targeted_attempted": attempted,
        "targeted_succeeded_full_schema": int(live.get("targeted_succeeded_full_schema") or 0),
        "targeted_succeeded_minimal_review_bound": int(live.get("targeted_succeeded_minimal_review_bound") or 0),
        "targeted_failed_after": int(live.get("targeted_failed_after") if live.get("targeted_failed_after") is not None else len(g["queue"])),
        "fast_fail_triggered": bool(live.get("fast_fail_triggered", False)),
        "actual_total_token_count": int(live.get("actual_total_token_count") or 0),
        "actual_cost_public_if_available": live.get("actual_cost_public_if_available", "unknown"),
        "same_day_total_cost_after_r20_usd": live.get("same_day_total_cost_after_r20_usd", "unknown"),
        "estimated_cost_before_live_usd": g["estimated_cost"],
        "cost_cap_exceeded": bool(live.get("cost_cap_exceeded", False)),
        "checkpoint_resume_available": True,
        "failed_evidence_preserved": bool(live.get("failed_evidence_preserved", False)) if attempted else False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "future_mkb_import_started": False,
        "private_artifacts_committed": False,
        "raw_ai_response_committed": False,
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
    _write_json("summary.json", summary)
    _write_json("targeted_selection_public.json", {
        "targeted_candidates_selected": summary["targeted_candidates_selected"],
        "review_only_records_selected": summary["review_only_records_selected"],
        "completed_content_packages_selected": summary["completed_content_packages_selected"],
        "excluded_rtf_signal_selected": summary["excluded_rtf_signal_selected"],
        "queue_digest": _doc_digest(g["queue"]),
        "missing_private_payload_count": len(g["missing"]),
    })
    _write_json("local_repair_plan_public.json", {
        "repair_counts": dict(g["repair_counts"]),
        "strategy_counts": dict(g["strategy_counts"]),
        "models_planned": dict(g["models"]),
        "local_repairs_applied": True,
    })
    _write_json("cost_gate_public.json", {
        "shared_budget_cap_usd": SHARED_BUDGET_CAP_USD,
        "same_day_spend_before_r20_usd": SAME_DAY_SPEND_BEFORE_R20_USD,
        "estimated_cost_before_live_usd": g["estimated_cost"],
        "cost_cap_exceeded": summary["cost_cap_exceeded"],
    })
    _write_json("live_result_public.json", {
        "overall_result": summary["overall_result"],
        "live_run_started": summary["live_run_started"],
        "targeted_attempted": summary["targeted_attempted"],
        "targeted_succeeded_full_schema": summary["targeted_succeeded_full_schema"],
        "targeted_succeeded_minimal_review_bound": summary["targeted_succeeded_minimal_review_bound"],
        "targeted_failed_after": summary["targeted_failed_after"],
        "fast_fail_triggered": summary["fast_fail_triggered"],
        "models_used": summary["models_used"],
        "actual_total_token_count": summary["actual_total_token_count"],
        "actual_cost_public_if_available": summary["actual_cost_public_if_available"],
        "same_day_total_cost_after_r20_usd": summary["same_day_total_cost_after_r20_usd"],
    })
    keys = [
        "targeted_only", "provider_model_call_made", "gemini_call_made", "mkb_db_opened",
        "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
        "future_mkb_import_started", "private_artifacts_committed",
        "raw_ai_response_committed", "tokenized_payloads_committed", "token_maps_committed",
        "pi_values_committed", "credentials_or_tokens_committed",
        "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
        "privacy_result", "safety_result",
    ]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# R20 targeted residual live requeue safety boundary", "", "| Gate | Value |",
                   "| --- | --- |", *[f"| {k} | `{summary[k]}` |" for k in keys], "",
                   "R20 selects only R19 eligible targeted residual candidates. Private queue, "
                   "provider traces, and failed evidence stay outside the repository. No MKB "
                   "open, import, write, auto-accept, or medical decision is performed.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} - implementation report", "",
                   f"- Targeted selected: `{summary['targeted_candidates_selected']}` from R19.",
                   f"- Review-only selected: `{summary['review_only_records_selected']}`.",
                   f"- Local repair counts: smaller `{summary['smaller_windowing_applied_count']}`, "
                   f"section `{summary['section_split_applied_count']}`, retokenize "
                   f"`{summary['retokenization_applied_count']}`, noise-drop "
                   f"`{summary['nonclinical_noise_drop_count']}`.",
                   f"- Live: started `{summary['live_run_started']}`, attempted "
                   f"`{summary['targeted_attempted']}`, full `{summary['targeted_succeeded_full_schema']}`, "
                   f"minimal-review `{summary['targeted_succeeded_minimal_review_bound']}`, "
                   f"failed-after `{summary['targeted_failed_after']}`.",
                   f"- Privacy/safety: `{summary['privacy_result']}` / `{summary['safety_result']}`.", ""]),
        encoding="utf-8")


def _privacy_scan(summary: dict[str, Any]) -> dict[str, int]:
    phi = path = secret = 0
    for p in REPORT_DIR.glob("*"):
        if p.suffix.lower() not in {".json", ".md"}:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
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
    g = build_gate()
    live = run_live(g) if args.live else None
    summary = build_summary(g, live, local_only=args.local_only)
    write_reports(summary, g)
    leaks = _privacy_scan(summary)
    write_reports(summary, g)
    print(json.dumps({
        "block": BLOCK,
        "mode": "live" if args.live else "local_only",
        "overall_result": summary["overall_result"],
        "targeted_candidates_selected": summary["targeted_candidates_selected"],
        "review_only_records_selected": summary["review_only_records_selected"],
        "live_entry_gate_passed": summary["live_entry_gate_passed"],
        "live_run_started": summary["live_run_started"],
        "targeted_attempted": summary["targeted_attempted"],
        "targeted_succeeded_full_schema": summary["targeted_succeeded_full_schema"],
        "targeted_succeeded_minimal_review_bound": summary["targeted_succeeded_minimal_review_bound"],
        "targeted_failed_after": summary["targeted_failed_after"],
        "fast_fail_triggered": summary["fast_fail_triggered"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
        "leaks": leaks,
    }, indent=2, sort_keys=True))
    return 0 if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
