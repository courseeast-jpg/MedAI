#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-478-LIVE-BATCH-VERTEX-17C-R2.

Corpus-scale live AI-first extraction against Vertex/Gemini for the canonical 478
validated tokenized requests from 17B-R2-R1, in chunks of 25. Strict preflight:
exactly-478 integrity, residual-PI re-validation, local cost caps ($0.05/chunk,
$0.25 total), credential preflight (ADC refresh), and a 17C-R2-specific live gate
activated only inside chunks and cleared in a finally block. Raw/parsed responses to
private staging only; schema validated locally; NO MKB write.

Public reports carry counts, chunk numbers, hashed doc IDs, and status only — never
payload/response bodies, token maps, raw OCR, private identifiers, or credentials.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.gemini_vertex_adapter import (  # noqa: E402
    GeminiVertexConfig,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    _default_http_post,
)

import re  # noqa: E402

from execution.jsonl_framing import read_jsonl_lines  # noqa: E402  physical-newline JSONL framing
from execution.canonical_batch_paths import resolve_canonical_batch  # noqa: E402  robust path resolver
from execution.strict_json import normalize_one_json_object, missing_required_keys  # noqa: E402  strict-JSON helpers
from execution import live_checkpoint as lc  # noqa: E402  durable checkpoint + evidence preservation (17C-R2-R8)
from execution import cost_chunk_planner as planner  # noqa: E402  adaptive cost + chunk planning (17C-R2-R10)
from execution.public_report_redaction import redact_json_value, redact_report_file  # noqa: E402  public report redaction

# Core required top-level schema keys every extraction response must include (17C-R2-R7).
EXPECTED_TOP_LEVEL_FIELDS = ("extracted_labs", "extracted_diagnoses", "extracted_medications",
                             "needs_review", "source_evidence", "extraction_warnings")

EXPECTED_HEAD = "ab4a00c87d0afd49f867e301c7ed9e069be884a1"

CANON_BATCH = resolve_canonical_batch()[0]  # robust resolver (literal -> %LOCALAPPDATA% -> home)
CANON_BATCH_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl"
PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_batch"))
PRIVATE_RESP_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_R2_478_live_batch\live_responses_private.jsonl"
DOWNLOADS = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17C_R2_478_Live_Batch_Review"))

PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_VERTEX_17C_R2"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_VERTEX_17C_R2.md",
    "MEDAI_17C_R2_LIVE_GATE_AND_COST_CAP.md",
    "MEDAI_17C_R2_STOP_ON_FIRST_FAILURE_POLICY.md",
    "MEDAI_17C_R2_PRIVATE_RESPONSE_STAGING_SPEC.md",
    "MEDAI_17D_MKB_STAGING_IMPORT_ENTRY_CRITERIA.md",
    "MEDAI_REMAINING_NON_READY_FILES_NOT_INCLUDED_17C_R2.md",
)

LIVE_GATE = "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_17C_R2_APPROVED"
GATE_VALUE = "YES"
OLD_GATES = ("MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED",
             "MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_17C_APPROVED",
             "MEDAI_VERTEX_LIVE_SMOKE_ALLOWED")

TARGET_MODEL = "gemini-2.5-flash-lite"
CHUNK_SIZE = 25
EXPECTED_REQUESTS = 478
CHUNK_COUNT_PLANNED = math.ceil(EXPECTED_REQUESTS / CHUNK_SIZE)  # 20 (default; adaptive at run time)
CAP_PER_CHUNK = 0.05
# Authorized total cap raised to $10.00 (17C-R2-R11) for the full 478-request corpus
# extraction while expiring Google Cloud credit remains; was $0.40 (R5), $0.25 (pre-R5).
CAP_TOTAL = 10.00
PREVIOUS_CAP_TOTAL = 0.40
# 17C-R2-R10: raised from 2048 -> 8192 to stop MAX_TOKENS mid-JSON truncation (R9 finding).
# gemini-2.5-flash-lite supports >= 8192 output tokens; the adapter posts maxOutputTokens
# verbatim and imposes no lower ceiling.
MAX_OUTPUT_TOKENS = 8192
PREVIOUS_MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
TARGET_PROJECT = "sot-knowledge-ocr"

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
_PI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b"),
    "mrn": re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "insurance_account": re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "accession_specimen": re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}"),
    "local_path": re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)"),
}


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _residual_pi(text: str) -> list[str]:
    residual = TOKEN_RE.sub(" ", text)
    return [name for name, rx in _PI_PATTERNS.items() if rx.search(residual)]


def _gate_active(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().upper() == GATE_VALUE


def _new_summary() -> dict[str, Any]:
    return {
        "block": "MEDAI-AI-FIRST-CORPUS-478-LIVE-BATCH-VERTEX-17C-R2",
        "live_execution_authorized": True,
        "authorized_scope": "canonical_478_validated_tokenized_requests_only",
        "target_model": TARGET_MODEL,
        "chunk_size": CHUNK_SIZE,
        "original_chunk_size": CHUNK_SIZE,
        "selected_chunk_size": CHUNK_SIZE,
        "adaptive_chunk_size_applied": False,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "previous_max_output_tokens": PREVIOUS_MAX_OUTPUT_TOKENS,
        "hard_cost_cap_per_chunk_usd": CAP_PER_CHUNK,
        "hard_cost_cap_total_usd": CAP_TOTAL,
        "estimated_total_cost_before_run_usd": 0.0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "credential_preflight_passed": False,
        "request_count_authorized": EXPECTED_REQUESTS,
        "request_count_loaded": 0,
        "request_count_sent": 0,
        "request_count_succeeded": 0,
        "request_count_failed": 0,
        "chunk_count_planned": CHUNK_COUNT_PLANNED,
        "chunk_count_started": 0,
        "chunk_count_completed": 0,
        "stopped_on_first_failure": False,
        "failure_stage": "none",
        "failure_category": "none",
        "private_request_batch_path": CANON_BATCH_LABEL,
        "private_response_staging_path": PRIVATE_RESP_LABEL,
        "private_outputs_written_outside_repo": False,
        # --- Durable checkpoint + evidence preservation (17C-R2-R8) ---
        "run_id": "",
        "run_timestamp": "",
        "canonical_batch_sha256": "",
        "checkpointing_enabled": True,
        "checkpoint_resume_start_index": 0,
        "checkpoint_resume_reason": "none",
        "checkpoint_completed_on_entry": 0,
        "request_count_skipped_completed": 0,
        "failed_doc_id": None,
        "evidence_preserved": False,
        "evidence_preservation_path": "",
        "evidence_files_copied": 0,
        "raw_ai_responses_written_to_repo": False,
        "parsed_ai_responses_written_to_repo": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "live_gate_set_during_run": False,
        "live_gate_environment_active_after_run": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "old_12_added_separately": False,
        "combined_batch_count": EXPECTED_REQUESTS,
        "blocked_files_excluded": 121,
        "duplicates_excluded": 88,
        "remaining_extraction_unavailable_excluded": 31,
        "unsupported_excluded": 2,
        "original_source_files_uploaded": False,
        "future_17d_mkb_import_not_started": True,
        "privacy_result": "passed",
        "safety_result": "passed",
        "execution_result": "BLOCKED",
    }


def _build_payload(prompt_contract: str, tokenized_content: str) -> dict[str, Any]:
    full = (prompt_contract
            + "\n\n# DOCUMENT (tokenized)\n" + tokenized_content
            + "\n\n# OUTPUT\nReturn strict JSON only per the schema. Preserve token "
              "placeholders exactly. Use null when missing.")
    return {"contents": [{"role": "user", "parts": [{"text": full}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_OUTPUT_TOKENS,
                                 "responseMimeType": "application/json"}}


def _extract_text(response: dict[str, Any]) -> str:
    try:
        return str(response["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return ""


def _validate_schema(text: str) -> tuple[bool, str]:
    # Safe normalization: strip ONE surrounding Markdown fence around a single JSON
    # object; reject prose-wrapped / multiple / truncated responses. Schema is NOT
    # weakened — full field + privacy checks still apply to the normalized object.
    obj, reason = normalize_one_json_object(text)
    if obj is None:
        return False, reason
    # Require ALL core top-level keys (stricter than before; never accepts incomplete
    # output). The schema is not weakened and no field is synthesized/inferred.
    if missing_required_keys(obj, EXPECTED_TOP_LEVEL_FIELDS):
        return False, "missing_expected_schema_fields"
    if _residual_pi(text):
        return False, "raw_pi_in_response"
    return True, "ok"


def _credential_preflight() -> tuple[bool, str]:
    """ADC refresh check for the target project. No model call, no token printed."""
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore
    except Exception:
        return False, "google_auth_unavailable"
    try:
        creds, _proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        return False, ("login_required" if "DefaultCredentials" in type(exc).__name__ else "credentials_unavailable")
    try:
        creds.refresh(google.auth.transport.requests.Request())
    except Exception as exc:
        low = str(exc).lower()
        if any(k in low for k in ("reauth", "invalid_grant", "expired", "login")):
            return False, "login_required"
        return False, "token_refresh_failed"
    if not str(getattr(creds, "token", "") or "").strip():
        return False, "token_empty"
    return True, "pass"


def run() -> dict[str, Any]:
    s = _new_summary()
    public_records: list[dict[str, Any]] = []
    chunk_status: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    parsed_responses: list[dict[str, Any]] = []

    import time  # local import: timestamp/run_id only, no provider use
    s["run_timestamp"] = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    s["run_id"] = "17c_r2_" + s["run_timestamp"]

    def stop(stage: str, cat: str) -> None:
        s["stopped_on_first_failure"] = True
        s["failure_stage"] = stage
        s["failure_category"] = cat

    # Gate inactivity preflight.
    if _gate_active(LIVE_GATE) or any(_gate_active(g) for g in OLD_GATES):
        stop("gate", "live_gate_active_before_run")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    if not CANON_BATCH.is_file():
        stop("preflight", "canonical_batch_missing")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    # Strict integrity load: exactly 478 well-formed, unique requests.
    lines = [l for l in read_jsonl_lines(CANON_BATCH) if l.strip()]  # physical-newline framing
    s["request_count_loaded"] = len(lines)
    requests: list[dict[str, Any]] = []
    malformed = 0
    for l in lines:
        try:
            requests.append(json.loads(l))
        except ValueError:
            malformed += 1
    unique_ids = {r.get("document_id") for r in requests}
    if len(lines) != EXPECTED_REQUESTS or malformed > 0 or len(unique_ids) != EXPECTED_REQUESTS or len(requests) != EXPECTED_REQUESTS:
        stop("preflight",
             f"canonical_batch_integrity_failure_lines_{len(lines)}_parseable_{len(requests)}_unique_{len(unique_ids)}_malformed_{malformed}")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    # ---- Durable checkpoint resume policy (17C-R2-R8) ----
    # Resume from the next unsent request; never re-send a completed document. Block on
    # SHA256 mismatch, an unresolved failed doc, or an inconsistent checkpoint.
    batch_sha = lc.sha256_file(CANON_BATCH)
    s["canonical_batch_sha256"] = batch_sha
    order = [r.get("document_id", "") for r in requests]
    start_index, ck_blocked, ck_reason, completed_set = lc.decide_resume(
        batch_sha, order, EXPECTED_REQUESTS)
    s["checkpoint_resume_start_index"] = start_index
    s["checkpoint_resume_reason"] = ck_reason
    s["checkpoint_completed_on_entry"] = len(completed_set)
    if ck_blocked:
        stop("checkpoint", ck_reason)
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
    lc.init_checkpoint(s["run_id"], batch_sha, TARGET_MODEL, CAP_TOTAL, CAP_PER_CHUNK,
                       EXPECTED_REQUESTS)

    # Privacy re-validation + cost estimate before any provider call.
    prompt_contract = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt_tokens = _approx_tokens(prompt_contract)
    est_input = 0
    per_doc_in: list[int] = []
    for req in requests:
        content = req.get("tokenized_content", "")
        if _residual_pi(content):
            stop("privacy", "residual_pi_in_request")
            s["privacy_result"] = "blocked"
            s["execution_result"] = "BLOCKED"
            return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
        di = _approx_tokens(content) + prompt_tokens
        per_doc_in.append(di)
        est_input += di
    est_output = MAX_OUTPUT_TOKENS * len(requests)
    est_cost = round(est_input / 1_000_000 * INPUT_USD_PER_M + est_output / 1_000_000 * OUTPUT_USD_PER_M, 6)
    s["estimated_input_tokens"] = est_input
    s["estimated_output_tokens"] = est_output
    s["estimated_total_cost_before_run_usd"] = est_cost
    if est_cost > CAP_TOTAL:
        stop("cost", f"estimated_total_{est_cost}_exceeds_cap_{CAP_TOTAL}")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    # ---- Adaptive chunk-size planning (17C-R2-R10) ----
    # Pick the largest chunk size <= CHUNK_SIZE whose worst-case cost stays within the
    # per-chunk cap at the new output ceiling. Total request count never changes.
    selected_chunk_size = planner.select_chunk_size(
        per_doc_in, MAX_OUTPUT_TOKENS, INPUT_USD_PER_M, OUTPUT_USD_PER_M, CAP_PER_CHUNK, CHUNK_SIZE)
    s["selected_chunk_size"] = selected_chunk_size
    s["adaptive_chunk_size_applied"] = 0 < selected_chunk_size < CHUNK_SIZE
    if selected_chunk_size <= 0:
        # Even a single request exceeds the per-chunk cap at this output ceiling.
        stop("cost", f"per_request_cost_exceeds_per_chunk_cap_{CAP_PER_CHUNK}")
        s["execution_result"] = "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
    active_chunk_size = selected_chunk_size
    active_chunk_count = math.ceil(EXPECTED_REQUESTS / active_chunk_size)
    s["chunk_count_planned"] = active_chunk_count
    # Per-chunk estimate at the selected size.
    for ci in range(active_chunk_count):
        seg = per_doc_in[ci * active_chunk_size:(ci + 1) * active_chunk_size]
        c_in = sum(seg)
        c_out = MAX_OUTPUT_TOKENS * len(seg)
        c_cost = round(c_in / 1_000_000 * INPUT_USD_PER_M + c_out / 1_000_000 * OUTPUT_USD_PER_M, 6)
        if c_cost > CAP_PER_CHUNK:
            stop("cost", f"chunk_{ci}_estimate_{c_cost}_exceeds_cap_{CAP_PER_CHUNK}")
            s["execution_result"] = "BLOCKED"
            return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    # Credential preflight (no model call).
    cred_ok, cred_cat = _credential_preflight()
    s["credential_preflight_passed"] = cred_ok
    if not cred_ok:
        stop("credentials", cred_cat)
        s["execution_result"] = "PROVIDER_FAIL" if cred_cat in ("login_required", "credentials_unavailable",
                                                                 "token_refresh_failed", "token_empty",
                                                                 "google_auth_unavailable") else "BLOCKED"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    # ---- Live chunked execution (gate active only inside; cleared in finally) ----
    config = GeminiVertexConfig()
    url = build_vertex_generate_content_url(config)
    from execution.gemini_vertex_adapter import acquire_google_cloud_access_token
    try:
        token = acquire_google_cloud_access_token()
    except Exception:
        token = ""
    if not str(token or "").strip():
        stop("credentials", "token_acquire_failed")
        s["execution_result"] = "PROVIDER_FAIL"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)

    try:
        for ci in range(active_chunk_count):
            chunk = requests[ci * active_chunk_size:(ci + 1) * active_chunk_size]
            if not chunk:
                continue
            os.environ[LIVE_GATE] = GATE_VALUE
            s["live_gate_set_during_run"] = True
            if not _gate_active(LIVE_GATE):
                stop("gate", "live_gate_activation_failed")
                s["execution_result"] = "BLOCKED"
                return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
            s["chunk_count_started"] += 1
            chunk_ok = 0
            try:
                for within, req in enumerate(chunk):
                    gidx = ci * active_chunk_size + within  # global request index across the batch
                    doc_id = req.get("document_id", "")
                    if doc_id in completed_set:
                        # Already completed on a prior run — never re-send / re-charge.
                        s["request_count_skipped_completed"] += 1
                        continue
                    payload = _build_payload(prompt_contract, req.get("tokenized_content", ""))
                    s["request_count_sent"] += 1
                    s["provider_call_made"] = True
                    s["vertex_live_execution"] = True
                    s["gemini_call_made"] = True
                    try:
                        response = _default_http_post(url, payload, token)
                    except Exception as exc:
                        err = classify_vertex_provider_error(exc)
                        cat = err.get("provider_error_category", "provider_error")
                        s["request_count_failed"] += 1
                        s["failed_doc_id"] = doc_id
                        lc.record_provider_trace(cat, gidx)
                        lc.mark_failed(doc_id, gidx, cat)
                        stop("provider", cat)
                        s["execution_result"] = "PROVIDER_FAIL"
                        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
                    raw_responses.append({"document_id": doc_id, "response": response})
                    text = _extract_text(response)
                    valid, reason = _validate_schema(text)
                    parsed_responses.append({"document_id": doc_id, "schema_valid": valid, "reason": reason})
                    usage = response.get("usageMetadata") if isinstance(response, dict) else {}
                    usage = usage if isinstance(usage, dict) else {}
                    public_records.append({"document_id": doc_id, "chunk": ci, "schema_valid": valid,
                                           "validation_reason": reason,
                                           "total_token_count": int(usage.get("totalTokenCount") or 0)})
                    if not valid:
                        s["request_count_failed"] += 1
                        s["failed_doc_id"] = doc_id
                        lc.record_provider_trace(reason, gidx)
                        lc.mark_failed(doc_id, gidx, reason)
                        if reason == "raw_pi_in_response":
                            stop("privacy", "raw_pi_in_response")
                            s["privacy_result"] = "blocked"
                            s["execution_result"] = "BLOCKED"
                        else:
                            stop("schema", reason)
                            s["execution_result"] = "SCHEMA_FAIL"
                        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
                    s["request_count_succeeded"] += 1
                    chunk_ok += 1
                    lc.record_provider_trace("ok", gidx)
                    lc.mark_completed(doc_id, gidx)
            finally:
                os.environ.pop(LIVE_GATE, None)  # clear gate after every chunk
            s["chunk_count_completed"] += 1
            chunk_status.append({"chunk": ci, "sent": len(chunk), "succeeded": chunk_ok})
            lc.record_chunk_status(ci, len(chunk), chunk_ok, "completed")
        s["execution_result"] = "PASS"
        return _finalize(s, public_records, chunk_status, raw_responses, parsed_responses)
    finally:
        os.environ.pop(LIVE_GATE, None)


def _finalize(s, records, chunk_status, raw_responses, parsed):
    os.environ.pop(LIVE_GATE, None)
    s["live_gate_environment_active_after_run"] = _gate_active(LIVE_GATE)
    private_written = False
    try:
        PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
        (PRIVATE_OUT / "live_request_manifest_private.json").write_text(
            json.dumps([{"document_id": r["document_id"]} for r in records], ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "live_chunk_plan_private.json").write_text(
            json.dumps({"chunk_size": CHUNK_SIZE, "chunk_count_planned": CHUNK_COUNT_PLANNED}, indent=2), encoding="utf-8")
        with (PRIVATE_OUT / "live_responses_private.jsonl").open("w", encoding="utf-8") as fh:
            for rr in raw_responses:
                fh.write(json.dumps(rr, ensure_ascii=False) + "\n")
        with (PRIVATE_OUT / "parsed_responses_private.jsonl").open("w", encoding="utf-8") as fh:
            for pr in parsed:
                fh.write(json.dumps(pr, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "schema_validation_private.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        with (PRIVATE_OUT / "chunk_status_private.jsonl").open("w", encoding="utf-8") as fh:
            for cs in chunk_status:
                fh.write(json.dumps(cs, ensure_ascii=False) + "\n")
        (PRIVATE_OUT / "completed_doc_ids_private.json").write_text(
            json.dumps([r["document_id"] for r in records if r["schema_valid"]], ensure_ascii=False, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "provider_trace_private.json").write_text(
            json.dumps({"sent": s["request_count_sent"], "succeeded": s["request_count_succeeded"],
                        "failed": s["request_count_failed"], "failure_stage": s["failure_stage"],
                        "failure_category": s["failure_category"]}, indent=2), encoding="utf-8")
        (PRIVATE_OUT / "cost_guard_private.json").write_text(
            json.dumps({"estimated_total_cost_before_run_usd": s["estimated_total_cost_before_run_usd"],
                        "cap_total": CAP_TOTAL, "cap_per_chunk": CAP_PER_CHUNK}, indent=2), encoding="utf-8")
        if s["stopped_on_first_failure"]:
            (PRIVATE_OUT / "stopped_on_failure_private.json").write_text(
                json.dumps({"failure_stage": s["failure_stage"], "failure_category": s["failure_category"]}, indent=2), encoding="utf-8")
        private_written = True
    except OSError:
        pass
    s["private_outputs_written_outside_repo"] = private_written
    # Immediately preserve failed-response evidence OUT of the volatile staging area
    # (the external writer has cleared MedAI_Private staging before). Private copy only.
    if s.get("failed_doc_id"):
        ts = s.get("run_timestamp") or "unknown"
        try:
            preserved, path, copied = lc.preserve_failed_evidence(ts, staging_dir=PRIVATE_OUT)
        except OSError:
            preserved, path, copied = False, "", 0
        s["evidence_preserved"] = preserved
        s["evidence_preservation_path"] = lc.EVIDENCE_DIR_LABEL + ("\\run_" + ts if preserved else "")
        s["evidence_files_copied"] = copied
    return _write_public(s, records, chunk_status)


def _pub_records(records):
    return [{"document_id": r["document_id"], "chunk": r["chunk"], "schema_valid": r["schema_valid"],
             "validation_reason": r["validation_reason"]} for r in records]


def _privacy_matrix(s):
    keys = ["provider_call_made", "gemini_call_made", "claude_call_made", "openai_call_made",
            "billing_api_call_made", "credential_preflight_passed", "live_gate_set_during_run",
            "live_gate_environment_active_after_run", "original_source_files_uploaded",
            "raw_ai_responses_written_to_repo", "parsed_ai_responses_written_to_repo",
            "tokenized_payloads_written_to_repo", "token_maps_written_to_repo",
            "private_identifier_values_written_to_repo", "credential_or_token_written_to_repo",
            "public_report_phi_leak_count", "mkb_db_opened", "active_mkb_write", "privacy_result", "safety_result"]
    return "\n".join(["# 17C-R2 privacy gate matrix", "", "| Gate | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      "Raw/parsed responses and tokenized payloads stay private (outside the repo).", ""])


def _stopped_md(s):
    if not s["stopped_on_first_failure"]:
        return "# 17C-R2 stop-on-first-failure\n\nNo failure stop occurred during this run.\n"
    return ("# 17C-R2 stop-on-first-failure\n\n"
            f"- stopped_on_first_failure: `True`\n- failure_stage: `{s['failure_stage']}`\n"
            f"- failure_category: `{s['failure_category']}`\n"
            f"- request_count_sent: `{s['request_count_sent']}`\n"
            f"- request_count_succeeded: `{s['request_count_succeeded']}`\n"
            f"- request_count_failed: `{s['request_count_failed']}`\n\n"
            "No retry was attempted. The live gate was cleared. No MKB write occurred.\n")


def _next_md(s):
    return ("# Next stage: 17D MKB staging import gate\n\n"
            f"- 17C-R2 execution_result: `{s['execution_result']}`\n"
            f"- requests succeeded: `{s['request_count_succeeded']}` of `{s['request_count_authorized']}`\n\n"
            "17D MKB staging import is NOT started and requires separate explicit authorization\n"
            "after human review of the private parsed responses and schema validation.\n")


def _implementation(s):
    lines = [f"# {s['block']}", "", f"## Result: **{s['execution_result']}**", "", "## Metrics", ""]
    for k in ("target_model", "chunk_size", "hard_cost_cap_per_chunk_usd", "hard_cost_cap_total_usd",
              "estimated_total_cost_before_run_usd", "estimated_input_tokens", "estimated_output_tokens",
              "credential_preflight_passed", "request_count_authorized", "request_count_loaded",
              "request_count_sent", "request_count_succeeded", "request_count_failed",
              "chunk_count_planned", "chunk_count_started", "chunk_count_completed",
              "stopped_on_first_failure", "failure_stage", "failure_category", "provider_call_made",
              "vertex_live_execution", "gemini_call_made", "claude_call_made", "openai_call_made",
              "billing_api_call_made", "live_gate_set_during_run", "live_gate_environment_active_after_run",
              "original_source_files_uploaded", "old_12_added_separately", "combined_batch_count",
              "blocked_files_excluded", "mkb_db_opened", "active_mkb_write", "auto_accept_enabled",
              "medical_decision_made", "production_queue_mutated", "raw_ai_responses_written_to_repo",
              "parsed_ai_responses_written_to_repo", "tokenized_payloads_written_to_repo",
              "token_maps_written_to_repo", "private_identifier_values_written_to_repo",
              "credential_or_token_written_to_repo", "public_report_phi_leak_count",
              "future_17d_mkb_import_not_started", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = {"PASS": "prepare 17D MKB staging import dry-run from private parsed responses (separate authorization required)",
           "PROVIDER_FAIL": "report provider failure category; no retry until the root cause is fixed",
           "SCHEMA_FAIL": "repair schema/prompt before any further live call",
           "BLOCKED": "repair privacy/cost/gate/batch-integrity preflight first"}.get(s["execution_result"], "review run")
    lines += ["", "## Recommended next (no MKB import started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def _write_public(s, records, chunk_status):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    s_redacted, _np, _ns = redact_json_value(s)
    s.clear()
    s.update(s_redacted)
    pub = _pub_records(records)
    live_status = {"block": s["block"], "execution_result": s["execution_result"],
                   "request_count_loaded": s["request_count_loaded"], "request_count_sent": s["request_count_sent"],
                   "request_count_succeeded": s["request_count_succeeded"], "request_count_failed": s["request_count_failed"],
                   "chunk_count_started": s["chunk_count_started"], "chunk_count_completed": s["chunk_count_completed"],
                   "stopped_on_first_failure": s["stopped_on_first_failure"], "documents": pub}
    schema_summary = {"execution_result": s["execution_result"],
                      "schema_valid_count": sum(1 for r in records if r["schema_valid"]),
                      "schema_invalid_count": sum(1 for r in records if not r["schema_valid"]),
                      "per_document": pub}
    cost_guard = {"target_model": TARGET_MODEL, "hard_cost_cap_per_chunk_usd": CAP_PER_CHUNK,
                  "hard_cost_cap_total_usd": CAP_TOTAL,
                  "estimated_total_cost_before_run_usd": s["estimated_total_cost_before_run_usd"],
                  "estimated_input_tokens": s["estimated_input_tokens"], "estimated_output_tokens": s["estimated_output_tokens"],
                  "actual_total_token_count": sum(r.get("total_token_count", 0) for r in records),
                  "note": "No billing API was called; local estimates and provider-reported tokens only."}
    provider_status = {"provider_call_made": s["provider_call_made"], "vertex_live_execution": s["vertex_live_execution"],
                       "gemini_call_made": s["gemini_call_made"], "credential_preflight_passed": s["credential_preflight_passed"],
                       "failure_stage": s["failure_stage"], "failure_category": s["failure_category"],
                       "execution_result": s["execution_result"]}
    matrix_md, stopped_md, next_md, impl_md = _privacy_matrix(s), _stopped_md(s), _next_md(s), _implementation(s)

    public_blob = "\n".join([json.dumps(s), json.dumps(live_status), json.dumps(schema_summary),
                             json.dumps(cost_guard), json.dumps(provider_status), matrix_md, stopped_md, next_md, impl_md])
    leak = 0
    try:
        p = PRIVATE_OUT / "live_responses_private.jsonl"
        if p.is_file():
            for ln in {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if len(l.strip()) >= 14}:
                if ln in public_blob:
                    leak += 1
                    break
    except OSError:
        pass
    s["public_report_phi_leak_count"] = leak
    if leak:
        s["safety_result"] = "blocked"

    public_outputs = {
        "summary.json": json.dumps(s, indent=2),
        "implementation_report.md": impl_md,
        "live_batch_status_public.json": json.dumps(live_status, indent=2),
        "schema_validation_summary_public.json": json.dumps(schema_summary, indent=2),
        "cost_guard_public.json": json.dumps(cost_guard, indent=2),
        "privacy_gate_matrix.md": matrix_md,
        "stopped_on_first_failure_public.md": stopped_md,
        "provider_status_public.json": json.dumps(provider_status, indent=2),
        "next_stage_17d_mkb_staging_import_gate.md": next_md,
    }
    for name, text in public_outputs.items():
        redacted, _path_count, _secret_count = redact_report_file(text, is_json=name.endswith(".json"))
        (REPORT_DIR / name).write_text(redacted, encoding="utf-8")
    with (REPORT_DIR / "chunk_status_public.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["chunk", "sent", "succeeded"])
        for cs in chunk_status:
            w.writerow([cs["chunk"], cs["sent"], cs["succeeded"]])

    try:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        (DOWNLOADS / "live_batch_summary_public.json").write_text(json.dumps(live_status, indent=2), encoding="utf-8")
        (DOWNLOADS / "schema_validation_summary_public.json").write_text(json.dumps(schema_summary, indent=2), encoding="utf-8")
        (DOWNLOADS / "cost_summary_public.json").write_text(json.dumps(cost_guard, indent=2), encoding="utf-8")
        (DOWNLOADS / "provider_status_public.json").write_text(json.dumps(provider_status, indent=2), encoding="utf-8")
        import shutil
        if (REPORT_DIR / "chunk_status_public.csv").is_file():
            shutil.copyfile(REPORT_DIR / "chunk_status_public.csv", DOWNLOADS / "chunk_status_public.csv")
        (DOWNLOADS / "README_NEXT_STEPS.txt").write_text(
            "MedAI 17C-R2 478 Live Batch Review (sanitized summaries only)\n\n"
            f"execution_result: {s['execution_result']}\n"
            f"requests succeeded: {s['request_count_succeeded']} / {s['request_count_authorized']}\n"
            f"stopped_on_first_failure: {s['stopped_on_first_failure']} ({s['failure_stage']}/{s['failure_category']})\n"
            f"estimated_total_cost_before_run_usd: {s['estimated_total_cost_before_run_usd']} (cap {CAP_TOTAL})\n\n"
            "Raw/parsed AI responses are private staging only. 17D MKB import NOT started.\n", encoding="utf-8")
        (DOWNLOADS / "mkb_import_not_started_notice.txt").write_text(
            "MKB import NOT started. 17C-R2 wrote nothing to MKB. 17D requires separate authorization.\n", encoding="utf-8")
    except OSError:
        pass
    return s


def main() -> int:
    s = run()
    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    safe = (s["live_gate_environment_active_after_run"] is False and s["public_report_phi_leak_count"] == 0
            and s["claude_call_made"] is False and s["openai_call_made"] is False
            and s["billing_api_call_made"] is False and s["active_mkb_write"] is False
            and s["raw_ai_responses_written_to_repo"] is False and docs_ok)
    print(f"medai_ai_first_corpus_478_live_batch_vertex_17c_r2_{s['execution_result'].lower()}"
          if safe else "medai_ai_first_corpus_478_live_batch_vertex_17c_r2_unsafe")
    print(json.dumps({k: s[k] for k in (
        "execution_result", "request_count_loaded", "request_count_sent", "request_count_succeeded",
        "request_count_failed", "chunk_count_completed", "stopped_on_first_failure", "failure_stage",
        "failure_category", "credential_preflight_passed", "estimated_total_cost_before_run_usd",
        "provider_call_made", "gemini_call_made", "live_gate_environment_active_after_run",
        "public_report_phi_leak_count", "mkb_db_opened", "active_mkb_write",
        "future_17d_mkb_import_not_started", "privacy_result", "safety_result")}, indent=2))
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
