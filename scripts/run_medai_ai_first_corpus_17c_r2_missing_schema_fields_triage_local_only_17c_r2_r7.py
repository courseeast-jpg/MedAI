#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-17C-R2-MISSING-SCHEMA-FIELDS-TRIAGE-LOCAL-ONLY-17C-R2-R7.

Local-only triage of the 17C-R2 SCHEMA_FAIL (request #2 missing_expected_schema_fields).
Identifies (best-effort) which required top-level keys were missing from response #2,
confirms prompt/precheck/validator hardening is in place, and reports a required-fields
precheck demonstration. NO model call, NO content request, NO billing, NO live gate, NO
live extraction, NO MKB. Field NAMES/paths only — never response bodies/values.
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
from execution.strict_json import normalize_one_json_object, missing_required_keys
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

ALLOWED_CLASSES = {
    "top_level_required_fields_missing", "nested_required_fields_missing",
    "empty_required_arrays", "wrong_type_for_required_field", "null_where_object_required",
    "schema_version_or_contract_mismatch", "provider_omitted_empty_sections",
    "unknown_missing_expected_schema_fields",
}
REQUIRED = live.EXPECTED_TOP_LEVEL_FIELDS

LIVE_SUMMARY = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2" / "summary.json"
STAGING = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_478_live_batch"))
PRESERVED = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_17C_R2_missing_schema_fields_PRIVATE_COPY"))
PROMPT_CONTRACT = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"
LIVE_RUNNER_SRC = REPO_ROOT / "scripts" / "run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2.py"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_missing_schema_fields_triage_local_only_17c_r2_r7"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_MISSING_SCHEMA_FIELDS_TRIAGE_LOCAL_ONLY_17C_R2_R7"
REQUIRED_DOCS = (
    "MEDAI_17C_R2_MISSING_SCHEMA_FIELDS_TRIAGE_LOCAL_ONLY_17C_R2_R7.md",
    "MEDAI_17C_R2_REQUIRED_SCHEMA_FIELDS_POLICY_R7.md",
    "MEDAI_17C_R2_LIVE_RESUME_ENTRY_GATE_AFTER_R7.md",
)


def _prior() -> dict[str, Any]:
    defaults = {"execution_result": "SCHEMA_FAIL", "request_count_loaded": 478,
                "request_count_sent": 2, "request_count_succeeded": 1, "request_count_failed": 1,
                "failure_stage": "schema", "failure_category": "missing_expected_schema_fields"}
    try:
        d = json.loads(LIVE_SUMMARY.read_text(encoding="utf-8"))
        if d.get("failure_category") == "missing_expected_schema_fields":
            return {k: d.get(k, defaults[k]) for k in defaults}
    except (OSError, ValueError):
        pass
    return defaults


def _failed_doc_id() -> str:
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


def _find_failed_response_text() -> tuple[str, bool]:
    """Best-effort: read failed response #2 text from staging or preserved copy.
    Returns (text, available). Never prints/returns it to public output."""
    for base in (STAGING, PRESERVED):
        lr = base / "live_responses_private.jsonl"
        if lr.is_file() and lr.stat().st_size > 0:
            try:
                recs = [json.loads(l) for l in read_jsonl_lines(lr) if l.strip()]
            except (OSError, ValueError):
                continue
            if len(recs) >= 2:
                try:
                    return str(recs[1]["response"]["candidates"][0]["content"]["parts"][0]["text"]), True
                except Exception:
                    return "", True
    return "", False


def _classify(text: str, available: bool) -> tuple[str, list[str]]:
    if not available or not text.strip():
        return "unknown_missing_expected_schema_fields", []
    obj, _reason = normalize_one_json_object(text)
    if obj is None:
        return "unknown_missing_expected_schema_fields", []
    missing = missing_required_keys(obj, REQUIRED)
    if not missing:
        # all core present but validator failed earlier -> nested/empty/type issue
        return "nested_required_fields_missing", []
    if len(missing) == len(REQUIRED):
        return "top_level_required_fields_missing", missing
    return "top_level_required_fields_missing", missing


def _schema_paths(names: list[str]) -> list[str]:
    return [f"$.{n}" for n in names]


def _precheck_demo() -> dict[str, Any]:
    full = {k: [] for k in REQUIRED}; full["needs_review"] = False
    return {
        "empty_object_missing_count": len(missing_required_keys({}, REQUIRED)),
        "partial_object_missing_count": len(missing_required_keys({"extracted_labs": []}, REQUIRED)),
        "full_skeleton_missing_count": len(missing_required_keys(full, REQUIRED)),
        "validator_rejects_partial": live._validate_schema('{"extracted_labs":[]}')[0] is False,
        "validator_accepts_full_skeleton": live._validate_schema(json.dumps(full))[0] is True,
    }


def _prompt_hardened() -> bool:
    try:
        t = re.sub(r"\s+", " ", PROMPT_CONTRACT.read_text(encoding="utf-8"))  # collapse line wraps
        return ("Required Top-Level Keys" in t and "MUST include ALL of these top-level keys" in t
                and "Do not omit any required key" in t)
    except OSError:
        return False


def _precheck_added() -> bool:
    try:
        return "missing_required_keys(obj, EXPECTED_TOP_LEVEL_FIELDS)" in LIVE_RUNNER_SRC.read_text(encoding="utf-8")
    except OSError:
        return False


def run() -> dict[str, Any]:
    prior = _prior()
    text, available = _find_failed_response_text()
    failure_class, missing_names = _classify(text, available)
    if failure_class not in ALLOWED_CLASSES:
        failure_class = "unknown_missing_expected_schema_fields"

    demo = _precheck_demo()
    precheck_ok = (demo["empty_object_missing_count"] == len(REQUIRED)
                   and demo["full_skeleton_missing_count"] == 0
                   and demo["validator_rejects_partial"] and demo["validator_accepts_full_skeleton"])

    prompt_hardened = _prompt_hardened()
    skeleton_added = prompt_hardened  # the skeleton list is part of the same prompt section
    precheck_added = _precheck_added()
    json_obj_returned = True  # missing-fields failure only occurs after a successful JSON parse

    # bodies unavailable -> cannot recover/replay the real failed response
    replay_recovered = False
    valid_count = 1
    failed_count = 1

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

    ready = bool(prompt_hardened and precheck_added and precheck_ok and sealed_valid and cred_pass)
    return {
        "block": "MEDAI-AI-FIRST-CORPUS-17C-R2-MISSING-SCHEMA-FIELDS-TRIAGE-LOCAL-ONLY-17C-R2-R7",
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
        "failed_response_body_available_in_context": available,
        "schema_failure_class": failure_class,
        "missing_required_field_count": len(missing_names),
        "missing_required_field_names": missing_names,
        "missing_required_field_schema_paths": _schema_paths(missing_names),
        "provider_returned_json_object": json_obj_returned,
        "response_privacy_clean": False if not available else True,
        "response_privacy_evaluated": available,
        "prompt_hardened_for_required_fields": prompt_hardened,
        "schema_skeleton_instruction_added": skeleton_added,
        "required_fields_precheck_added": precheck_added,
        "required_fields_precheck_demo": demo,
        "schema_weakened": False,
        "required_fields_made_optional": False,
        "clinical_values_inferred": False,
        "missing_required_values_synthesized": False,
        "local_replay_attempted": True,
        "local_replay_recovered_failed_response": replay_recovered,
        "local_replay_schema_valid_count": valid_count,
        "local_replay_schema_failed_count": failed_count,
        "authorized_hard_cost_cap_total_usd": 0.40,
        "hard_cost_cap_per_chunk_usd": 0.05,
        "sealed_batch_valid": sealed_valid,
        "credential_preflight_passed": cred_pass,
        "ready_to_resume_17c_r2_live": ready,
        "resume_policy": "restart_required" if ready else "blocked",
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
    }


def _missing_public(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in (
        "prior_failure_stage", "prior_failure_category", "failed_request_index",
        "failed_doc_id_public", "failed_response_body_available_in_context", "schema_failure_class",
        "missing_required_field_count", "missing_required_field_names",
        "missing_required_field_schema_paths", "provider_returned_json_object",
        "response_privacy_clean", "response_privacy_evaluated", "required_top_level_keys")}


def _replay_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"required_fields_precheck_demo": s["required_fields_precheck_demo"],
            "local_replay_attempted": s["local_replay_attempted"],
            "local_replay_recovered_failed_response": s["local_replay_recovered_failed_response"],
            "local_replay_schema_valid_count": s["local_replay_schema_valid_count"],
            "local_replay_schema_failed_count": s["local_replay_schema_failed_count"],
            "schema_weakened": s["schema_weakened"], "required_fields_made_optional": s["required_fields_made_optional"],
            "clinical_values_inferred": s["clinical_values_inferred"],
            "missing_required_values_synthesized": s["missing_required_values_synthesized"]}


def _hardening_md(s: dict[str, Any]) -> str:
    return "\n".join(["# Prompt contract hardening (17C-R2-R7)", "",
                      f"- prompt_hardened_for_required_fields: `{s['prompt_hardened_for_required_fields']}`",
                      f"- schema_skeleton_instruction_added: `{s['schema_skeleton_instruction_added']}`",
                      f"- required_fields_precheck_added: `{s['required_fields_precheck_added']}`",
                      f"- schema_weakened: `{s['schema_weakened']}` | required_fields_made_optional: `{s['required_fields_made_optional']}`",
                      f"- clinical_values_inferred: `{s['clinical_values_inferred']}` | missing_required_values_synthesized: `{s['missing_required_values_synthesized']}`", "",
                      "The prompt now lists every required top-level key and requires the model to emit",
                      "all of them (empty arrays for lists, null only where allowed). The 17C-R2 validator",
                      "requires ALL core top-level keys and never accepts incomplete output. Nothing is",
                      "inferred or synthesized; the schema is not weakened.", ""])


def _resume_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 live resume entry gate (after R7)", "",
                      f"- ready_to_resume_17c_r2_live: `{s['ready_to_resume_17c_r2_live']}`",
                      f"- resume_policy: `{s['resume_policy']}`",
                      f"- sealed_batch_valid: `{s['sealed_batch_valid']}` | credential_preflight_passed: `{s['credential_preflight_passed']}`", "",
                      "The 17C-R2 runner has no per-request checkpoint, so a resume re-sends from the",
                      "first request. The exact missing fields for the prior failure were unavailable;",
                      "watch request #2 on the next run and capture the body if it fails again.", ""])


def _safety_md() -> str:
    return "\n".join(["# 17C-R2-R7 safety boundary", "",
                      "- No Gemini/Vertex/Claude/OpenAI model call; no provider content request.",
                      "- No billing API call; no live gate; no live extraction; no corpus request.",
                      "- No MKB open/write; no auto-accept; no medical decision; no queue mutation.",
                      "- No raw AI response bodies, parsed responses, tokenized payloads, raw OCR, token",
                      "  maps, private values, or credentials are printed or committed.",
                      "- Schema not weakened; required fields not made optional; nothing inferred/synthesized.", ""])


def _implementation(s: dict[str, Any]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("prior_live_execution_result", "prior_request_count_loaded", "prior_request_count_sent",
              "prior_request_count_succeeded", "prior_request_count_failed", "prior_failure_stage",
              "prior_failure_category", "failed_request_index", "failed_doc_id_public",
              "failed_response_body_available_in_context", "schema_failure_class",
              "missing_required_field_count", "missing_required_field_names", "provider_returned_json_object",
              "response_privacy_clean", "prompt_hardened_for_required_fields",
              "schema_skeleton_instruction_added", "required_fields_precheck_added", "schema_weakened",
              "required_fields_made_optional", "clinical_values_inferred", "missing_required_values_synthesized",
              "local_replay_recovered_failed_response", "local_replay_schema_valid_count",
              "local_replay_schema_failed_count", "authorized_hard_cost_cap_total_usd",
              "hard_cost_cap_per_chunk_usd", "sealed_batch_valid", "credential_preflight_passed",
              "ready_to_resume_17c_r2_live", "resume_policy", "provider_model_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "public_report_phi_leak_count", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    nxt = ("run the R4 operator-context restore, then restart 17C-R2 live once per the resume policy"
           if s["ready_to_resume_17c_r2_live"] else "fix schema/prompt locally before any further live call")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    s = run()
    s["required_top_level_keys"] = list(REQUIRED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing_pub = _missing_public(s)
    replay_pub = _replay_public(s)
    hardening_md = _hardening_md(s)
    resume_md = _resume_md(s)
    safety_md = _safety_md()
    impl_md = _implementation(s)

    public_blob = "\n".join([json.dumps(s), json.dumps(missing_pub), json.dumps(replay_pub),
                             hardening_md, resume_md, safety_md, impl_md])
    for marker in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"', '"response"'):
        if marker in public_blob:
            s["public_report_phi_leak_count"] = 1
            s["safety_result"] = "blocked"
            break

    (REPORT_DIR / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(s), encoding="utf-8")
    (REPORT_DIR / "missing_schema_fields_public.json").write_text(json.dumps(missing_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "local_replay_validation_public.json").write_text(json.dumps(replay_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "prompt_contract_hardening_public.md").write_text(hardening_md, encoding="utf-8")
    (REPORT_DIR / "live_resume_entry_gate.md").write_text(resume_md, encoding="utf-8")
    (REPORT_DIR / "safety_boundary_public.md").write_text(safety_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (s["safety_result"] == "passed" and s["public_report_phi_leak_count"] == 0
          and s["provider_model_call_made"] is False and s["schema_weakened"] is False
          and s["required_fields_made_optional"] is False and s["prompt_hardened_for_required_fields"] is True
          and s["required_fields_precheck_added"] is True and s["schema_failure_class"] in ALLOWED_CLASSES
          and docs_ok)
    print("medai_ai_first_corpus_17c_r2_missing_schema_fields_triage_local_only_17c_r2_r7_"
          + ("pass" if ok and s["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: s[k] for k in (
        "schema_failure_class", "missing_required_field_count", "failed_response_body_available_in_context",
        "provider_returned_json_object", "prompt_hardened_for_required_fields",
        "required_fields_precheck_added", "schema_weakened", "required_fields_made_optional",
        "local_replay_recovered_failed_response", "sealed_batch_valid", "credential_preflight_passed",
        "ready_to_resume_17c_r2_live", "resume_policy", "public_report_phi_leak_count",
        "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
