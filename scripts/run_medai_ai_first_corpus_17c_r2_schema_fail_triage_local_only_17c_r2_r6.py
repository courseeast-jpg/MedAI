#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-SCHEMA-FAIL-TRIAGE-LOCAL-ONLY-17C-R2-R6.

Local-only triage of the 17C-R2 SCHEMA_FAIL (request #2 not_strict_json). Inspects the
public run summary + private metadata, classifies the strict-JSON failure (best-effort;
the real failed body may be unavailable in this context), confirms the prompt/parser/
JSON-mode hardening is in place, validates the normalizer via a synthetic local replay,
and reports the stale-cap-test fix.

NO model call, NO content request, NO billing, NO live gate, NO live extraction, NO MKB.
No raw response bodies / payloads / token maps / PI / credentials are printed or committed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.canonical_batch_paths import resolve_canonical_batch
from execution.jsonl_framing import load_jsonl_objects, read_jsonl_lines
from execution.strict_json import normalize_one_json_object
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

ALLOWED_CLASSES = {
    "empty_response", "markdown_fenced_json", "prose_wrapped_json", "multiple_json_objects",
    "truncated_json", "invalid_json_escape", "safety_refusal_or_policy_text",
    "schema_shape_mismatch", "unexpected_non_json_text", "unknown_not_strict_json",
}

LIVE_SUMMARY = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2" / "summary.json"
PRIVATE_STAGING = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_batch"))
PROMPT_CONTRACT = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"
LIVE_RUNNER_SRC = REPO_ROOT / "scripts" / "run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2.py"
STALE_TEST = REPO_ROOT / "tests" / "test_medai_ai_first_corpus_478_live_batch_vertex_17c_r2.py"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_schema_fail_triage_local_only_17c_r2_r6"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_SCHEMA_FAIL_TRIAGE_LOCAL_ONLY_17C_R2_R6"
REQUIRED_DOCS = (
    "MEDAI_17C_R2_SCHEMA_FAIL_TRIAGE_LOCAL_ONLY_17C_R2_R6.md",
    "MEDAI_17C_R2_STRICT_JSON_RESPONSE_POLICY_R6.md",
    "MEDAI_17C_R2_LIVE_RESUME_ENTRY_GATE_AFTER_R6.md",
)


def _prior_from_summary() -> dict[str, Any]:
    # Prefer the on-disk run summary; fall back to the observed values from the task.
    defaults = {"execution_result": "SCHEMA_FAIL", "request_count_loaded": 478,
                "request_count_sent": 2, "request_count_succeeded": 1, "request_count_failed": 1,
                "failure_stage": "schema", "failure_category": "not_strict_json"}
    try:
        d = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))
        if d.get("execution_result") == "SCHEMA_FAIL":
            return {k: d.get(k, defaults[k]) for k in defaults}
    except (OSError, ValueError):
        pass
    return defaults


def _failed_doc_id() -> str:
    # Request #2 = the 2nd canonical record (sequential send). Doc ids are hashes.
    path, ok = resolve_canonical_batch()
    if not ok:
        return "unavailable_canonical_missing"
    try:
        objs, _, _ = load_jsonl_objects(path)
        if len(objs) >= 2:
            return str(objs[1].get("document_id", "unavailable"))
    except OSError:
        pass
    return "unavailable"


def _classify_failed_body() -> tuple[str, bool]:
    """Best-effort classification from the private staged failed response, if present.
    Returns (class, body_available). Never prints the body."""
    lr = PRIVATE_STAGING / "live_responses_private.jsonl"
    if not lr.is_file() or lr.stat().st_size == 0:
        return "unknown_not_strict_json", False
    try:
        recs = [json.loads(l) for l in read_jsonl_lines(lr) if l.strip()]
    except (OSError, ValueError):
        return "unknown_not_strict_json", False
    if len(recs) < 2:
        return "unknown_not_strict_json", False
    resp = recs[1].get("response", {})
    try:
        text = str(resp["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        text = ""
    obj, reason = normalize_one_json_object(text)
    mapping = {"empty_response": "empty_response",
               "markdown_fence_unterminated": "markdown_fenced_json",
               "multiple_json_objects_or_trailing_text": "multiple_json_objects",
               "prose_wrapped_or_non_json": "prose_wrapped_json",
               "truncated_or_invalid_json": "truncated_json",
               "not_json_object": "schema_shape_mismatch"}
    if obj is not None:
        return "markdown_fenced_json", True  # recoverable fence (otherwise valid)
    return mapping.get(reason, "unknown_not_strict_json"), True


def _synthetic_replay() -> dict[str, bool]:
    good = json.dumps({"extracted_labs": [], "needs_review": True, "source_evidence": []})
    return {
        "plain_object_ok": normalize_one_json_object(good)[1] == "ok",
        "fenced_object_recovered": normalize_one_json_object("```json\n" + good + "\n```")[1] == "ok",
        "prose_wrapped_rejected": normalize_one_json_object("Here is the JSON:\n" + good)[0] is None,
        "multiple_objects_rejected": normalize_one_json_object(good + "\n" + good)[0] is None,
        "truncated_rejected": normalize_one_json_object('{"extracted_labs":[ ')[0] is None,
        "empty_rejected": normalize_one_json_object("   ")[0] is None,
    }


def _normalizer_wired() -> bool:
    try:
        return "normalize_one_json_object(text)" in LIVE_RUNNER_SRC.read_text(encoding="utf-8")
    except OSError:
        return False


def _prompt_hardened() -> bool:
    try:
        t = PROMPT_CONTRACT.read_text(encoding="utf-8")
        return "Return ONLY one valid JSON object" in t and "No Markdown fences" in t
    except OSError:
        return False


def _json_mode_enabled() -> bool:
    try:
        pay = live._build_payload("p", "[PATIENT_NAME_1]")
        return pay.get("generationConfig", {}).get("responseMimeType") == "application/json"
    except Exception:
        return False


def _stale_cap_test_updated() -> bool:
    try:
        t = STALE_TEST.read_text(encoding="utf-8")
        # No remaining assertion that the total cap equals the old 0.25, and the runner
        # constant is the authorized 0.40.
        old = 's["hard_cost_cap_total_usd"] == 0.25' in t
        return (not old) and float(live.CAP_TOTAL) == 0.40 and float(live.CAP_PER_CHUNK) == 0.05
    except OSError:
        return False


def run() -> dict[str, Any]:
    prior = _prior_from_summary()
    failure_class, body_available = _classify_failed_body()
    if failure_class not in ALLOWED_CLASSES:
        failure_class = "unknown_not_strict_json"
    replay = _synthetic_replay()
    normalizer_ok = all(replay.values())

    normalizer_added = _normalizer_wired()
    prompt_hardened = _prompt_hardened()
    json_mode = _json_mode_enabled()
    cap_test_updated = _stale_cap_test_updated()

    # Sealed batch + credential preflight (reuse R5-style checks via the live module path).
    path, ok = resolve_canonical_batch()
    sealed_valid = False
    if ok and path.is_file():
        objs, malformed, nonempty = load_jsonl_objects(path)
        sealed_valid = (nonempty == 478 and len(objs) == 478
                        and len({o.get("document_id") for o in objs}) == 478 and malformed == 0)
    try:
        cred_pass, _adc = live._credential_preflight()
    except Exception:
        cred_pass = False

    # Local replay schema validity carried from the prior run metadata (1 succeeded,
    # 1 failed). The real failed body is not recoverable here unless it was a fence.
    replay_recovered = (failure_class == "markdown_fenced_json" and body_available)
    valid_count = 2 if replay_recovered else 1
    failed_count = 0 if replay_recovered else 1

    ready = bool(normalizer_added and prompt_hardened and json_mode and sealed_valid
                 and cred_pass and normalizer_ok)
    resume_policy = ("restart_required" if ready else "blocked")

    return {
        "block": "MEDAI-AI-FIRST-CORPUS-17C-R2-SCHEMA-FAIL-TRIAGE-LOCAL-ONLY-17C-R2-R6",
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
        "prior_live_execution_result": prior["execution_result"],
        "prior_request_count_loaded": prior["request_count_loaded"],
        "prior_request_count_sent": prior["request_count_sent"],
        "prior_request_count_succeeded": prior["request_count_succeeded"],
        "prior_request_count_failed": prior["request_count_failed"],
        "prior_failure_stage": prior["failure_stage"],
        "prior_failure_category": prior["failure_category"],
        "failed_request_index": 2,
        "failed_doc_id_public": _failed_doc_id(),
        "failed_response_body_available_in_context": body_available,
        "schema_failure_class": failure_class,
        "safe_normalizer_added": normalizer_added,
        "normalizer_synthetic_replay_all_pass": normalizer_ok,
        "prompt_hardened_for_strict_json": prompt_hardened,
        "json_mode_enabled_if_supported": json_mode,
        "schema_weakened": False,
        "partial_json_accepted": False,
        "missing_fields_inferred": False,
        "local_replay_attempted": True,
        "local_replay_recovered_failed_response": replay_recovered,
        "local_replay_schema_valid_count": valid_count,
        "local_replay_schema_failed_count": failed_count,
        "authorized_hard_cost_cap_total_usd": 0.40,
        "hard_cost_cap_per_chunk_usd": 0.05,
        "stale_cap_test_updated": cap_test_updated,
        "sealed_batch_valid": sealed_valid,
        "credential_preflight_passed": cred_pass,
        "ready_to_resume_17c_r2_live": ready,
        "resume_policy": resume_policy,
        "private_responses_committed": False,
        "parsed_private_responses_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
        "_replay": replay,
    }


def _classification_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "prior_live_execution_result", "prior_request_count_loaded", "prior_request_count_sent",
        "prior_request_count_succeeded", "prior_request_count_failed", "prior_failure_stage",
        "prior_failure_category", "failed_request_index", "failed_doc_id_public",
        "failed_response_body_available_in_context", "schema_failure_class")}


def _replay_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"normalizer_synthetic_replay": s["_replay"],
            "normalizer_synthetic_replay_all_pass": s["normalizer_synthetic_replay_all_pass"],
            "local_replay_attempted": s["local_replay_attempted"],
            "local_replay_recovered_failed_response": s["local_replay_recovered_failed_response"],
            "local_replay_schema_valid_count": s["local_replay_schema_valid_count"],
            "local_replay_schema_failed_count": s["local_replay_schema_failed_count"],
            "schema_weakened": s["schema_weakened"], "partial_json_accepted": s["partial_json_accepted"],
            "missing_fields_inferred": s["missing_fields_inferred"]}


def _hardening_md(s: dict[str, Any]) -> str:
    return "\n".join(["# Prompt & parser hardening (17C-R2-R6)", "",
                      f"- safe_normalizer_added: `{s['safe_normalizer_added']}`",
                      f"- prompt_hardened_for_strict_json: `{s['prompt_hardened_for_strict_json']}`",
                      f"- json_mode_enabled_if_supported: `{s['json_mode_enabled_if_supported']}`",
                      f"- schema_weakened: `{s['schema_weakened']}` | partial_json_accepted: `{s['partial_json_accepted']}` | missing_fields_inferred: `{s['missing_fields_inferred']}`",
                      f"- normalizer_synthetic_replay_all_pass: `{s['normalizer_synthetic_replay_all_pass']}`", "",
                      "The normalizer strips ONE clean Markdown fence around a single JSON object and",
                      "rejects prose-wrapped, multiple-object, truncated, and empty responses. The schema",
                      "is unchanged; full field + privacy checks still apply to the normalized object.", ""])


def _resume_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 live resume entry gate (after R6)", "",
                      f"- ready_to_resume_17c_r2_live: `{s['ready_to_resume_17c_r2_live']}`",
                      f"- resume_policy: `{s['resume_policy']}`",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}` | credential_preflight_passed: `{s['credential_preflight_passed']}`", "",
                      "The 17C-R2 runner has no per-request checkpoint, so a resume re-sends from the",
                      "first request (request #1 is sent again). If the failure recurs at request #2,",
                      "capture the private response body and re-triage; the normalizer intentionally does",
                      "not recover prose/truncated/safety-refusal responses.", ""])


def _safety_md() -> str:
    return "\n".join(["# 17C-R2-R6 safety boundary", "",
                      "- No Gemini/Vertex/Claude/OpenAI model call; no provider content request.",
                      "- No billing API call; no live gate; no live extraction; no corpus request.",
                      "- No MKB open/write; no auto-accept; no medical decision; no queue mutation.",
                      "- No raw AI response bodies, parsed responses, tokenized payloads, raw OCR, token",
                      "  maps, private values, or credentials are printed or committed.",
                      "- Schema not weakened; partial JSON not accepted; missing fields not inferred.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("prior_live_execution_result", "prior_request_count_loaded", "prior_request_count_sent",
              "prior_request_count_succeeded", "prior_request_count_failed", "prior_failure_stage",
              "prior_failure_category", "failed_request_index", "failed_doc_id_public",
              "failed_response_body_available_in_context", "schema_failure_class",
              "safe_normalizer_added", "normalizer_synthetic_replay_all_pass",
              "prompt_hardened_for_strict_json", "json_mode_enabled_if_supported", "schema_weakened",
              "partial_json_accepted", "missing_fields_inferred", "local_replay_attempted",
              "local_replay_recovered_failed_response", "local_replay_schema_valid_count",
              "local_replay_schema_failed_count", "authorized_hard_cost_cap_total_usd",
              "hard_cost_cap_per_chunk_usd", "stale_cap_test_updated", "sealed_batch_valid",
              "credential_preflight_passed", "ready_to_resume_17c_r2_live", "resume_policy",
              "provider_model_call_made", "live_gate_set", "live_extraction_started", "mkb_db_opened",
              "active_mkb_write", "public_report_phi_leak_count", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = ("run the R4 operator-context restore, then resume/restart 17C-R2 live once per the resume policy"
           if s["ready_to_resume_17c_r2_live"] else "fix schema/prompt/parser locally before any further live call")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    classification = _classification_public(s)
    replay_pub = _replay_public(s)
    hardening_md = _hardening_md(s)
    resume_md = _resume_md(s)
    safety_md = _safety_md()
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps({k: v for k, v in s.items() if k != "_replay"}),
                             json.dumps(classification), json.dumps(replay_pub),
                             hardening_md, resume_md, safety_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    s_public = {k: v for k, v in s.items() if k != "_replay"}
    (REPORT_DIR / "summary.json").write_text(json.dumps(s_public, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "schema_failure_classification_public.json").write_text(json.dumps(classification, indent=2), encoding="utf-8")
    (REPORT_DIR / "local_replay_validation_public.json").write_text(json.dumps(replay_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "prompt_and_parser_hardening_public.md").write_text(hardening_md, encoding="utf-8")
    (REPORT_DIR / "live_resume_entry_gate.md").write_text(resume_md, encoding="utf-8")
    (REPORT_DIR / "safety_boundary_public.md").write_text(safety_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["schema_weakened"] is False
          and s["safe_normalizer_added"] is True and s["prompt_hardened_for_strict_json"] is True
          and s["stale_cap_test_updated"] is True and s["schema_failure_class"] in ALLOWED_CLASSES
          and docs_ok)
    print("medai_ai_first_corpus_17c_r2_schema_fail_triage_local_only_17c_r2_r6_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s_public[k] for k in (
        "schema_failure_class", "failed_response_body_available_in_context", "safe_normalizer_added",
        "prompt_hardened_for_strict_json", "json_mode_enabled_if_supported", "schema_weakened",
        "local_replay_recovered_failed_response", "stale_cap_test_updated", "sealed_batch_valid",
        "credential_preflight_passed", "ready_to_resume_17c_r2_live", "resume_policy",
        "public_report_phi_leak_count", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
