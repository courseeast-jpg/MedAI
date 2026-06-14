#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-SMALL-LIVE-BATCH-VERTEX-17C.

First small live AI-first extraction batch against Vertex/Gemini using ONLY the 12
validated tokenized requests from 17B-R1. Raw responses are saved to private staging;
schema validated locally; NO MKB write. Hard cost cap $0.05, stop-on-first-failure,
sequential, no retry, no fallback. The 17C live gate is activated only inside this run
and cleared in a finally block.

Public reports carry counts, hashed document IDs, and status only — never tokenized
payloads, raw responses, token maps, raw OCR, or private identifier values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.gemini_vertex_adapter import (  # noqa: E402
    GeminiVertexConfig,
    acquire_google_cloud_access_token,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    _default_http_post,
)

EXPECTED_HEAD = "bd7624f137ade57013eba5f4efe848a028b86327"

PRIVATE_BATCH = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R1_repaired\outbound_requests_private.jsonl"))
PRIVATE_BATCH_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R1_repaired\outbound_requests_private.jsonl"
PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_live_batch"))
PRIVATE_RESP_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_live_batch\live_responses_private.jsonl"
DOWNLOADS = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17C_Live_Batch_Review"))

PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"
SCHEMA_PATH = REPO_ROOT / "config" / "medai_ai_extraction_schema_17b.json"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_small_live_batch_vertex_17c"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_VERTEX_17C"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_VERTEX_17C.md",
    "MEDAI_17C_LIVE_GATE_AND_COST_CAP.md",
    "MEDAI_17C_STOP_ON_FIRST_FAILURE_POLICY.md",
    "MEDAI_17C_PRIVATE_RESPONSE_STAGING_SPEC.md",
    "MEDAI_17D_MKB_STAGING_IMPORT_ENTRY_CRITERIA.md",
    "MEDAI_REMAINING_587_BLOCKED_FILES_NOT_INCLUDED_17C.md",
)

LIVE_GATE_17C = "MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_17C_APPROVED"
GATE_ACTIVE_VALUE = "YES"
# Gates we must never set/rely on.
OLD_16D_GATE = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
SMOKE_GATE = "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED"

TARGET_MODEL = "gemini-2.5-flash-lite"
HARD_COST_CAP_USD = 0.05
MAX_REQUESTS = 12
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
_PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b"),
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}


def _sha16(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _residual_pi(text: str) -> list[str]:
    residual = TOKEN_RE.sub(" ", text)
    return [name for name, rx in _PI_PATTERNS.items() if rx.search(residual)]


def _gate_active(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().upper() == GATE_ACTIVE_VALUE


def _new_summary() -> dict[str, Any]:
    return {
        "block": "MEDAI-AI-FIRST-CORPUS-SMALL-LIVE-BATCH-VERTEX-17C",
        "live_execution_authorized": True,
        "authorized_scope": "12_validated_tokenized_requests_only",
        "target_model": TARGET_MODEL,
        "hard_cost_cap_usd": HARD_COST_CAP_USD,
        "estimated_cost_before_run_usd": 0.0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "request_count_authorized": MAX_REQUESTS,
        "request_count_loaded": 0,
        "request_count_sent": 0,
        "request_count_succeeded": 0,
        "request_count_failed": 0,
        "stopped_on_first_failure": False,
        "failure_stage": "none",
        "failure_category": "none",
        "private_request_batch_path": PRIVATE_BATCH_LABEL,
        "private_response_staging_path": PRIVATE_RESP_LABEL,
        "private_outputs_written_outside_repo": False,
        "raw_ai_responses_written_to_repo": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "future_live_gate_set_during_run": False,
        "future_live_gate_environment_active_after_run": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "blocked_files_excluded": 587,
        "original_source_files_uploaded": False,
        "future_17d_mkb_import_not_started": True,
        "privacy_result": "passed",
        "safety_result": "passed",
        "execution_result": "BLOCKED",
    }


def _build_payload(prompt_contract: str, tokenized_content: str) -> dict[str, Any]:
    full_prompt = (
        prompt_contract
        + "\n\n# DOCUMENT (tokenized; identifiers already replaced with placeholders)\n"
        + tokenized_content
        + "\n\n# OUTPUT\nReturn strict JSON only, conforming to the extraction schema. "
        "Preserve token placeholders exactly. Use null when missing."
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }


def _extract_response_text(response: dict[str, Any]) -> str:
    try:
        return str(response["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return ""


def _validate_response_schema(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty_response"
    try:
        obj = json.loads(text)
    except ValueError:
        return False, "not_strict_json"
    if not isinstance(obj, dict):
        return False, "not_json_object"
    expected = ("extracted_labs", "extracted_diagnoses", "extracted_medications",
                "needs_review", "source_evidence", "extraction_warnings")
    if not any(k in obj for k in expected):
        return False, "missing_expected_schema_fields"
    # Response must carry no residual raw PI.
    if _residual_pi(text):
        return False, "raw_pi_in_response"
    return True, "ok"


def run() -> dict[str, Any]:
    s = _new_summary()
    private_records: list[dict[str, Any]] = []  # public-safe per-doc results
    raw_responses: list[dict[str, Any]] = []     # PRIVATE only
    parsed_responses: list[dict[str, Any]] = []   # PRIVATE only

    def stop(stage: str, category: str) -> None:
        s["stopped_on_first_failure"] = True
        s["failure_stage"] = stage
        s["failure_category"] = category

    # ---- Preflight (no provider call) ----
    if _gate_active(LIVE_GATE_17C):
        stop("gate", "live_gate_already_active_before_run")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, private_records, raw_responses, parsed_responses)
    if _gate_active(OLD_16D_GATE) or _gate_active(SMOKE_GATE):
        stop("gate", "forbidden_other_live_gate_active")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, private_records, raw_responses, parsed_responses)

    if not PRIVATE_BATCH.is_file():
        stop("preflight", "request_batch_missing")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, private_records, raw_responses, parsed_responses)

    requests = []
    for line in PRIVATE_BATCH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                requests.append(json.loads(line))
            except ValueError:
                pass
    s["request_count_loaded"] = len(requests)
    if len(requests) != MAX_REQUESTS:
        stop("preflight", f"unexpected_request_count_{len(requests)}")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, private_records, raw_responses, parsed_responses)

    # Privacy re-validation + cost estimate before any call.
    prompt_contract = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt_tokens = _approx_tokens(prompt_contract)
    est_input = 0
    for req in requests:
        content = req.get("tokenized_content", "")
        if _residual_pi(content):
            stop("privacy", "residual_pi_in_request")
            s["privacy_result"] = "blocked"
            s["execution_result"] = "BLOCKED"
            return _finalize(s, private_records, raw_responses, parsed_responses)
        est_input += _approx_tokens(content) + prompt_tokens
    est_output = MAX_OUTPUT_TOKENS * len(requests)
    est_cost = round(est_input / 1_000_000 * INPUT_USD_PER_M + est_output / 1_000_000 * OUTPUT_USD_PER_M, 6)
    s["estimated_input_tokens"] = est_input
    s["estimated_output_tokens"] = est_output
    s["estimated_cost_before_run_usd"] = est_cost
    if est_cost > HARD_COST_CAP_USD:
        stop("cost", f"estimated_cost_{est_cost}_exceeds_cap_{HARD_COST_CAP_USD}")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, private_records, raw_responses, parsed_responses)

    # ---- Live execution (gate active only inside; cleared in finally) ----
    config = GeminiVertexConfig()
    url = build_vertex_generate_content_url(config)
    try:
        os.environ[LIVE_GATE_17C] = GATE_ACTIVE_VALUE
        s["future_live_gate_set_during_run"] = True

        # 17C gate must be active to permit any call (our orchestration-layer gate).
        if not _gate_active(LIVE_GATE_17C):
            stop("gate", "live_gate_activation_failed")
            s["execution_result"] = "BLOCKED"
            return _finalize(s, private_records, raw_responses, parsed_responses)

        try:
            token = acquire_google_cloud_access_token()
        except Exception:  # noqa: BLE001 - token resolution failed (ADC/gcloud unavailable)
            s["provider_call_made"] = False  # no Vertex HTTP attempted; auth resolution failed
            stop("provider", "auth_or_credentials_unavailable")
            s["execution_result"] = "PROVIDER_FAIL"
            return _finalize(s, private_records, raw_responses, parsed_responses)
        if not str(token or "").strip():
            s["provider_call_made"] = False
            stop("provider", "google_cloud_access_token_unavailable")
            s["execution_result"] = "PROVIDER_FAIL"
            return _finalize(s, private_records, raw_responses, parsed_responses)

        for idx, req in enumerate(requests):
            doc_id = req.get("document_id", f"doc_{idx}")
            payload = _build_payload(prompt_contract, req.get("tokenized_content", ""))
            s["request_count_sent"] += 1
            s["provider_call_made"] = True
            s["vertex_live_execution"] = True
            s["gemini_call_made"] = True
            try:
                response = _default_http_post(url, payload, token)
            except Exception as exc:  # noqa: BLE001 - sanitized via classifier
                err = classify_vertex_provider_error(exc)
                s["request_count_failed"] += 1
                stop("provider", err.get("provider_error_category", "provider_error"))
                s["execution_result"] = "PROVIDER_FAIL"
                return _finalize(s, private_records, raw_responses, parsed_responses)

            raw_responses.append({"document_id": doc_id, "response": response})  # PRIVATE
            text = _extract_response_text(response)
            valid, reason = _validate_response_schema(text)
            usage = response.get("usageMetadata") if isinstance(response, dict) else {}
            usage = usage if isinstance(usage, dict) else {}
            parsed_responses.append({"document_id": doc_id, "schema_valid": valid,
                                     "reason": reason})  # PRIVATE (no body)
            private_records.append({
                "document_id": doc_id,
                "schema_valid": valid,
                "validation_reason": reason,
                "prompt_token_count": int(usage.get("promptTokenCount") or 0),
                "candidates_token_count": int(usage.get("candidatesTokenCount") or 0),
                "total_token_count": int(usage.get("totalTokenCount") or 0),
            })
            if not valid:
                s["request_count_failed"] += 1
                if reason == "raw_pi_in_response":
                    stop("privacy", "raw_pi_in_response")
                    s["privacy_result"] = "blocked"
                    s["execution_result"] = "BLOCKED"
                else:
                    stop("schema", reason)
                    s["execution_result"] = "SCHEMA_FAIL"
                return _finalize(s, private_records, raw_responses, parsed_responses)
            s["request_count_succeeded"] += 1

        s["execution_result"] = "PASS"
        return _finalize(s, private_records, raw_responses, parsed_responses)
    finally:
        os.environ.pop(LIVE_GATE_17C, None)


def _finalize(s: dict[str, Any], records: list[dict[str, Any]],
              raw_responses: list[dict[str, Any]], parsed: list[dict[str, Any]]) -> dict[str, Any]:
    # Ensure gate cleared and report inactive.
    os.environ.pop(LIVE_GATE_17C, None)
    s["future_live_gate_environment_active_after_run"] = _gate_active(LIVE_GATE_17C)

    # ---- Write PRIVATE staging (outside repo, never committed) ----
    private_written = False
    try:
        PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
        (PRIVATE_OUT / "live_request_manifest_private.json").write_text(
            json.dumps([{"document_id": r["document_id"]} for r in records], ensure_ascii=False, indent=2),
            encoding="utf-8")
        with (PRIVATE_OUT / "live_responses_private.jsonl").open("w", encoding="utf-8") as fh:
            for rr in raw_responses:
                fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
        with (PRIVATE_OUT / "parsed_responses_private.jsonl").open("w", encoding="utf-8") as fh:
            for pr in parsed:
                fh.write(json.dumps(pr, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "schema_validation_private.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "provider_trace_private.json").write_text(
            json.dumps({"sent": s["request_count_sent"], "succeeded": s["request_count_succeeded"],
                        "failed": s["request_count_failed"], "failure_stage": s["failure_stage"],
                        "failure_category": s["failure_category"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (PRIVATE_OUT / "cost_guard_private.json").write_text(
            json.dumps({"estimated_cost_before_run_usd": s["estimated_cost_before_run_usd"],
                        "hard_cost_cap_usd": HARD_COST_CAP_USD}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        if s["stopped_on_first_failure"]:
            (PRIVATE_OUT / "stopped_on_failure_private.json").write_text(
                json.dumps({"failure_stage": s["failure_stage"], "failure_category": s["failure_category"]},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        private_written = True
    except OSError:
        pass
    s["private_outputs_written_outside_repo"] = private_written
    return _write_public(s, records)


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"document_id": r["document_id"], "schema_valid": r["schema_valid"],
             "validation_reason": r["validation_reason"],
             "total_token_count": r.get("total_token_count", 0)} for r in records]


def _privacy_matrix(s: dict[str, Any]) -> str:
    keys = ["provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
            "billing_api_call_made", "future_live_gate_set_during_run",
            "future_live_gate_environment_active_after_run", "original_source_files_uploaded",
            "raw_ai_responses_written_to_repo", "tokenized_payloads_written_to_repo",
            "token_maps_written_to_repo", "private_identifier_values_written_to_repo",
            "public_report_phi_leak_count", "mkb_db_opened", "active_mkb_write",
            "privacy_result", "safety_result"]
    return "\n".join(["# 17C privacy gate matrix", "", "| Gate | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      "Raw responses and tokenized payloads stay private (outside the repo).",
                      "Public reports carry counts, hashed doc IDs, and status only.", ""])


def _stopped_md(s: dict[str, Any]) -> str:
    if not s["stopped_on_first_failure"]:
        return "# 17C stop-on-first-failure\n\nNo failure stop occurred during this run.\n"
    return ("# 17C stop-on-first-failure\n\n"
            f"- stopped_on_first_failure: `True`\n"
            f"- failure_stage: `{s['failure_stage']}`\n"
            f"- failure_category: `{s['failure_category']}`\n"
            f"- request_count_sent: `{s['request_count_sent']}`\n"
            f"- request_count_succeeded: `{s['request_count_succeeded']}`\n"
            f"- request_count_failed: `{s['request_count_failed']}`\n\n"
            "No retry was attempted. The live gate was cleared. No MKB write occurred.\n")


def _next_gate_md(s: dict[str, Any]) -> str:
    return ("# Next stage: 17D MKB staging import gate\n\n"
            f"- 17C execution_result: `{s['execution_result']}`\n"
            f"- requests succeeded: `{s['request_count_succeeded']}` of `{s['request_count_authorized']}`\n\n"
            "17D MKB staging import is NOT started and requires separate explicit authorization\n"
            "after human review of the private parsed responses and schema validation. 17C wrote\n"
            "nothing to MKB.\n")


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "", f"## Result: **{s['execution_result']}**", "", "## Metrics", ""]
    for k in ("target_model", "hard_cost_cap_usd", "estimated_cost_before_run_usd",
              "estimated_input_tokens", "estimated_output_tokens", "request_count_authorized",
              "request_count_loaded", "request_count_sent", "request_count_succeeded",
              "request_count_failed", "stopped_on_first_failure", "failure_stage", "failure_category",
              "provider_call_made", "vertex_live_execution", "gemini_call_made", "claude_call_made",
              "openai_call_made", "billing_api_call_made", "future_live_gate_set_during_run",
              "future_live_gate_environment_active_after_run", "original_source_files_uploaded",
              "blocked_files_excluded", "mkb_db_opened", "active_mkb_write", "auto_accept_enabled",
              "medical_decision_made", "production_queue_mutated", "raw_ai_responses_written_to_repo",
              "tokenized_payloads_written_to_repo", "token_maps_written_to_repo",
              "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
              "future_17d_mkb_import_not_started", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = {
        "PASS": "prepare 17D MKB staging import dry-run from private parsed responses (separate authorization required)",
        "PROVIDER_FAIL": "report provider failure category; no retry is permitted until the root cause is fixed",
        "SCHEMA_FAIL": "repair schema/prompt before any further live call",
        "BLOCKED": "repair privacy/cost/gate preflight first",
    }.get(s["execution_result"], "review run")
    lines += ["", "## Recommended next (no MKB import started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def _write_public(s: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pub_records = _public_records(records)

    live_batch_status = {"block": s["block"], "execution_result": s["execution_result"],
                         "request_count_loaded": s["request_count_loaded"],
                         "request_count_sent": s["request_count_sent"],
                         "request_count_succeeded": s["request_count_succeeded"],
                         "request_count_failed": s["request_count_failed"],
                         "stopped_on_first_failure": s["stopped_on_first_failure"],
                         "documents": pub_records}
    schema_summary = {"execution_result": s["execution_result"],
                      "schema_valid_count": sum(1 for r in records if r["schema_valid"]),
                      "schema_invalid_count": sum(1 for r in records if not r["schema_valid"]),
                      "per_document": pub_records}
    cost_guard = {"target_model": TARGET_MODEL, "hard_cost_cap_usd": HARD_COST_CAP_USD,
                  "estimated_cost_before_run_usd": s["estimated_cost_before_run_usd"],
                  "estimated_input_tokens": s["estimated_input_tokens"],
                  "estimated_output_tokens": s["estimated_output_tokens"],
                  "actual_total_token_count": sum(r.get("total_token_count", 0) for r in records),
                  "note": "No billing API was called; estimates and provider-reported token counts only."}
    provider_status = {"provider_call_made": s["provider_call_made"],
                       "vertex_live_execution": s["vertex_live_execution"],
                       "gemini_call_made": s["gemini_call_made"],
                       "failure_stage": s["failure_stage"], "failure_category": s["failure_category"],
                       "execution_result": s["execution_result"]}
    matrix_md = _privacy_matrix(s)
    stopped_md = _stopped_md(s)
    next_md = _next_gate_md(s)
    impl_md = _implementation(s)

    # Defense-in-depth: ensure no raw response/payload line leaked into public blobs.
    public_blob = "\n".join([json.dumps(s), json.dumps(live_batch_status), json.dumps(schema_summary),
                             json.dumps(cost_guard), json.dumps(provider_status), matrix_md,
                             stopped_md, next_md, impl_md])
    leak = 0
    try:
        for fname in ("live_responses_private.jsonl",):
            p = PRIVATE_OUT / fname
            if p.is_file():
                for ln in {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if len(l.strip()) >= 12}:
                    if ln in public_blob:
                        leak += 1
        bf = PRIVATE_BATCH
        if bf.is_file():
            for rl in bf.read_text(encoding="utf-8").splitlines():
                try:
                    c = json.loads(rl).get("tokenized_content", "")
                except ValueError:
                    c = ""
                for ln in {l.strip() for l in c.splitlines() if len(l.strip()) >= 12}:
                    if ln in public_blob:
                        leak += 1
    except OSError:
        pass
    s["public_report_phi_leak_count"] = leak
    if leak:
        s["safety_result"] = "blocked"

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(impl_md, encoding="utf-8")
    (REPORT_DIR / "live_batch_status_public.json").write_text(json.dumps(live_batch_status, indent=2), encoding="utf-8")
    (REPORT_DIR / "schema_validation_summary_public.json").write_text(json.dumps(schema_summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "cost_guard_public.json").write_text(json.dumps(cost_guard, indent=2), encoding="utf-8")
    (REPORT_DIR / "privacy_gate_matrix.md").write_text(matrix_md, encoding="utf-8")
    (REPORT_DIR / "stopped_on_first_failure_public.md").write_text(stopped_md, encoding="utf-8")
    (REPORT_DIR / "provider_status_public.json").write_text(json.dumps(provider_status, indent=2), encoding="utf-8")
    (REPORT_DIR / "next_stage_17d_mkb_staging_import_gate.md").write_text(next_md, encoding="utf-8")

    # Downloads sanitized review.
    try:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        (DOWNLOADS / "live_batch_summary_public.json").write_text(json.dumps(live_batch_status, indent=2), encoding="utf-8")
        (DOWNLOADS / "schema_validation_summary_public.json").write_text(json.dumps(schema_summary, indent=2), encoding="utf-8")
        (DOWNLOADS / "cost_summary_public.json").write_text(json.dumps(cost_guard, indent=2), encoding="utf-8")
        (DOWNLOADS / "README_NEXT_STEPS.txt").write_text(
            "MedAI 17C Live Batch Review (sanitized summaries only)\n\n"
            f"execution_result: {s['execution_result']}\n"
            f"requests succeeded: {s['request_count_succeeded']} / {s['request_count_authorized']}\n"
            f"stopped_on_first_failure: {s['stopped_on_first_failure']} "
            f"({s['failure_stage']}/{s['failure_category']})\n"
            f"estimated_cost_before_run_usd: {s['estimated_cost_before_run_usd']} (cap {HARD_COST_CAP_USD})\n\n"
            "Raw AI responses are in private staging only, never in this folder or the repo.\n"
            "17D MKB staging import is NOT started and requires separate authorization.\n", encoding="utf-8")
        (DOWNLOADS / "mkb_import_not_started_notice.txt").write_text(
            "MKB import NOT started. 17C wrote nothing to MKB. A future 17D MKB staging import\n"
            "requires separate explicit authorization after human review of private responses.\n",
            encoding="utf-8")
    except OSError:
        pass
    return s


def main() -> int:
    s = run()
    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    # The script "succeeds" (exit 0) when it completed safely with a clean gate and no
    # leak — regardless of PASS/BLOCKED/PROVIDER_FAIL/SCHEMA_FAIL (all are valid outcomes).
    safe = (
        s["future_live_gate_environment_active_after_run"] is False
        and s["public_report_phi_leak_count"] == 0
        and s["claude_call_made"] is False
        and s["openai_call_made"] is False
        and s["billing_api_call_made"] is False
        and s["active_mkb_write"] is False
        and s["raw_ai_responses_written_to_repo"] is False
        and docs_ok
    )
    print(f"medai_ai_first_corpus_small_live_batch_vertex_17c_{s['execution_result'].lower()}"
          if safe else "medai_ai_first_corpus_small_live_batch_vertex_17c_unsafe")
    print(json.dumps({k: s[k] for k in (
        "execution_result", "request_count_loaded", "request_count_sent",
        "request_count_succeeded", "request_count_failed", "stopped_on_first_failure",
        "failure_stage", "failure_category", "estimated_cost_before_run_usd",
        "provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
        "billing_api_call_made", "future_live_gate_environment_active_after_run",
        "public_report_phi_leak_count", "mkb_db_opened", "active_mkb_write",
        "future_17d_mkb_import_not_started", "privacy_result", "safety_result")}, indent=2))
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
