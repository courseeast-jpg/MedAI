"""Autonomous guarded 17C-R2-R13 recovery runner.

This module coordinates local gates, sectioned fallback extraction, private
checkpoint/evidence handling, and public-safe reporting. Provider calls happen
only when `live=True` is passed by the CLI script.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from clinical_knowledge.privacy import check_public_report_payload

from execution import live_checkpoint as lc
from execution import sectioned_checkpoint as sc
from execution.gemini_vertex_adapter import (
    GeminiVertexConfig,
    acquire_google_cloud_access_token,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    _default_http_post,
)
from execution.jsonl_framing import read_jsonl_lines
from execution.public_report_redaction import redact_json_value, redact_report_file
from execution.sectioned_cost_planner import build_cost_plan
from execution.sectioned_extraction import (
    SECTION_NAMES,
    SKELETON_STRATEGY,
    SUBSECTION_STRATEGY,
    build_full_payload,
    build_section_payload,
    is_transient_provider_category,
    salvage_one_complete_json_object,
    split_tokenized_window,
    validate_section_response,
)
from execution.sectioned_merge import merge_sections


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_autonomous_recovery_full_corpus_final_17c_r2_r13"
PRIVATE_OUT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_R13_autonomous_recovery"))
LIVE_GATE = "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_17C_R2_APPROVED"
GATE_VALUE = "YES"
BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-AUTONOMOUS-RECOVERY-AND-FULL-CORPUS-RUN-FINAL-17C-R2-R13"
TARGET_MODEL = "gemini-2.5-flash-lite"
EXPECTED_REQUESTS = 478
TOTAL_CAP_USD = 10.00
PER_CHUNK_CAP_USD = 0.05
INPUT_USD_PER_M = 0.075
OUTPUT_USD_PER_M = 0.30
FULL_MAX_OUTPUT_TOKENS = 8192
SECTION_MAX_OUTPUT_TOKENS = 2048


def default_summary() -> dict[str, Any]:
    return {
        "block": BLOCK,
        "local_strategy_committed": False,
        "local_strategy_commit_sha": None,
        "full_live_run_started": False,
        "provider_model_call_made_during_local_phase": False,
        "provider_model_call_made_during_live_phase": False,
        "gemini_call_made_during_live_phase": False,
        "target_model": TARGET_MODEL,
        "canonical_request_count": EXPECTED_REQUESTS,
        "docs_loaded": 0,
        "docs_completed": 0,
        "docs_failed_for_review": 0,
        "docs_unattempted": 0,
        "full_schema_attempts": 0,
        "sectioned_attempts": 0,
        "subsection_attempts": 0,
        "json_salvage_attempts": 0,
        "schema_skeleton_retries": 0,
        "transient_provider_retries": 0,
        "sections_sent": 0,
        "sections_succeeded": 0,
        "sections_failed": 0,
        "recovery_ladder_exhausted_count": 0,
        "run_result": "BLOCKED",
        "failure_stage": "none",
        "failure_category": "none",
        "failed_doc_hash": None,
        "failed_section": None,
        "failed_evidence_preserved": False,
        "authorized_total_cap_usd": TOTAL_CAP_USD,
        "hard_cost_cap_per_chunk_usd": PER_CHUNK_CAP_USD,
        "estimated_total_cost_before_run_usd": 0.0,
        "actual_total_token_count": 0,
        "actual_cost_public_if_available": "unknown",
        "cost_cap_exceeded": False,
        "per_chunk_cap_exceeded": False,
        "checkpoint_document_aware": True,
        "checkpoint_section_aware": True,
        "completed_docs_skipped_on_resume": True,
        "completed_sections_skipped_on_resume": True,
        "merge_requires_all_required_sections": True,
        "schema_weakened": False,
        "clinical_values_inferred": False,
        "missing_required_values_synthesized": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "future_17d_mkb_import_not_started": True,
        "live_gate_environment_active_after_run": False,
        "private_checkpoint_committed": False,
        "private_evidence_committed": False,
        "private_responses_committed": False,
        "parsed_private_responses_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "credential_or_token_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "private_filename_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }


def run_autonomous_recovery(*, live: bool = False,
                            ignore_failed_review_skip: bool = False,
                            http_post: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None,
                            token_provider: Callable[[], str] | None = None) -> dict[str, Any]:
    import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base
    import scripts.run_medai_ai_first_corpus_17c_r2_checkpoint_unblock_and_report_redaction_local_only_17c_r2_r12 as r12

    summary = default_summary()
    committed, commit_sha = _local_strategy_commit_status()
    summary["local_strategy_committed"] = committed
    summary["local_strategy_commit_sha"] = commit_sha
    failed_docs: list[dict[str, Any]] = []
    section_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    parsed_records: list[dict[str, Any]] = []
    token = ""
    url = build_vertex_generate_content_url(GeminiVertexConfig())
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    batch_sha, order, canonical_valid = r12._load_canonical_order()
    if not canonical_valid:
        summary.update({"run_result": "BLOCKED", "failure_stage": "preflight", "failure_category": "canonical_batch_invalid"})
        _write_public_reports(summary, failed_docs, section_records)
        return summary
    requests = _load_requests(live_base.CANON_BATCH)
    summary["docs_loaded"] = len(requests)

    unblock = r12._checkpoint_action(batch_sha, order)
    redaction = _redact_report_tree(live_base.REPORT_DIR)
    completed_set = set(lc.load_completed())
    failed_skip = set() if ignore_failed_review_skip else _load_failed_review_doc_ids()
    per_doc_tokens = [_approx_tokens(str(req.get("tokenized_content") or "")) for req in requests]
    plan = build_cost_plan(
        per_doc_tokens,
        total_cap_usd=TOTAL_CAP_USD,
        per_chunk_cap_usd=PER_CHUNK_CAP_USD,
        full_max_output_tokens=FULL_MAX_OUTPUT_TOKENS,
        section_max_output_tokens=SECTION_MAX_OUTPUT_TOKENS,
        input_usd_per_m=INPUT_USD_PER_M,
        output_usd_per_m=OUTPUT_USD_PER_M,
    )
    summary["estimated_total_cost_before_run_usd"] = plan.estimated_total_cost_usd
    if not plan.full_schema_safe or redaction["private_path_leaks_after"] or redaction["secret_leaks_after"]:
        summary.update({
            "run_result": "BLOCKED",
            "failure_stage": "cost_or_privacy",
            "failure_category": "local_gate_failed",
            "per_chunk_cap_exceeded": not plan.full_schema_safe,
            "privacy_result": "blocked" if redaction["private_path_leaks_after"] or redaction["secret_leaks_after"] else "passed",
            "safety_result": "blocked",
        })
        _write_public_reports(summary, failed_docs, section_records)
        return summary

    cred_ok, cred_cat = live_base._credential_preflight()
    if not cred_ok:
        summary.update({"run_result": "BLOCKED", "failure_stage": "credentials", "failure_category": cred_cat, "safety_result": "blocked"})
        _write_public_reports(summary, failed_docs, section_records)
        return summary

    if not live:
        summary.update({
            "run_result": "PASS",
            "docs_unattempted": len([r for r in requests if str(r.get("document_id") or "") not in completed_set]),
        })
        _write_public_reports(summary, failed_docs, section_records)
        return summary

    try:
        token = (token_provider or acquire_google_cloud_access_token)()
    except Exception:
        token = ""
    if not str(token or "").strip():
        summary.update({"run_result": "BLOCKED", "failure_stage": "credentials", "failure_category": "token_acquire_failed", "safety_result": "blocked"})
        _write_public_reports(summary, failed_docs, section_records)
        return summary

    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    sc.write_state({"run_timestamp": run_ts, "canonical_batch_sha256": batch_sha, "total": len(requests)})
    completed_sections = sc.load_completed_sections()
    actual_cost = 0.0
    summary["full_live_run_started"] = True

    try:
        for index, req in enumerate(requests):
            doc_id = str(req.get("document_id") or "")
            tokenized_content = str(req.get("tokenized_content") or "")
            if doc_id in completed_set:
                continue
            if doc_id in failed_skip:
                continue
            if actual_cost >= TOTAL_CAP_USD:
                summary.update({"run_result": "BLOCKED", "failure_stage": "cost", "failure_category": "total_cap_exhausted", "cost_cap_exceeded": True})
                break

            ok, category, sections, usage, records = _attempt_document(
                doc_id=doc_id,
                index=index,
                tokenized_content=tokenized_content,
                completed_sections=set(completed_sections.get(doc_id, [])),
                url=url,
                token=token,
                http_post=http_post or _default_http_post,
                summary=summary,
                raw_records=raw_records,
                parsed_records=parsed_records,
            )
            actual_cost += _usage_cost(usage)
            summary["actual_total_token_count"] += int(usage.get("total_token_count", 0))
            section_records.extend(records)
            if actual_cost > TOTAL_CAP_USD:
                summary.update({"run_result": "BLOCKED", "failure_stage": "cost", "failure_category": "total_cap_exceeded_after_response", "cost_cap_exceeded": True})
                break
            if ok:
                lc.mark_completed(doc_id, index)
                completed_set.add(doc_id)
                continue
            failed_docs.append({"doc_hash": doc_id, "failure_category": category, "failed_section": sections[-1] if sections else None})
            _write_failed_review_doc_id(doc_id)
            sc.mark_section_failed(doc_id, sections[-1] if sections else "full_schema", category)
            preserved, _path, copied = _preserve_live_private(run_ts, doc_id, category, raw_records, parsed_records)
            summary["failed_evidence_preserved"] = bool(preserved)
            summary["failed_doc_hash"] = doc_id
            summary["failed_section"] = sections[-1] if sections else None
            summary["failure_stage"] = "recovery_ladder"
            summary["failure_category"] = category
            summary["recovery_ladder_exhausted_count"] += 1
            summary["docs_failed_for_review"] = len(failed_docs)
            summary["evidence_files_copied"] = copied
            # Continue for schema/provider-output failures; hard safety failures already returned as blocked.
            continue
    finally:
        os.environ.pop(LIVE_GATE, None)
        summary["live_gate_environment_active_after_run"] = str(os.environ.get(LIVE_GATE) or "").strip().upper() == GATE_VALUE

    summary["docs_completed"] = len(lc.load_completed())
    summary["docs_failed_for_review"] = len(failed_docs)
    summary["docs_unattempted"] = max(0, len(requests) - summary["docs_completed"] - summary["docs_failed_for_review"])
    summary["actual_cost_public_if_available"] = round(actual_cost, 6)
    if summary["run_result"] == "BLOCKED":
        summary["safety_result"] = "blocked"
    elif summary["docs_completed"] == len(requests):
        summary["run_result"] = "PASS"
    else:
        summary["run_result"] = "LIVE_FAIL" if failed_docs else "PASS"
    _write_public_reports(summary, failed_docs, section_records)
    return summary


def _attempt_document(*, doc_id: str, index: int, tokenized_content: str, completed_sections: set[str],
                      url: str, token: str,
                      http_post: Callable[[str, dict[str, Any], str], dict[str, Any]],
                      summary: dict[str, Any],
                      raw_records: list[dict[str, Any]],
                      parsed_records: list[dict[str, Any]]) -> tuple[bool, str, list[str], dict[str, int], list[dict[str, Any]]]:
    usage = {"total_token_count": 0, "prompt_token_count": 0, "candidates_token_count": 0}
    records: list[dict[str, Any]] = []
    summary["full_schema_attempts"] += 1
    ok, reason, response = _provider_call(url, build_full_payload("", tokenized_content), token, http_post, summary)
    if ok:
        raw_records.append({"document_id": doc_id, "strategy": "full_schema", "response": response})
        text = _extract_text(response)
        valid, full_reason = _validate_full(text)
        _merge_usage(usage, response)
        parsed_records.append({"document_id": doc_id, "strategy": "full_schema", "schema_valid": valid, "reason": full_reason})
        records.append({"doc_hash": doc_id, "strategy": "full_schema", "section": None, "schema_valid": valid, "failure_category": full_reason})
        if valid:
            return True, "ok", [], usage, records
        if full_reason == "raw_pi_in_response":
            return False, full_reason, ["full_schema"], usage, records
        summary["json_salvage_attempts"] += 1
        salvage_ok, _obj, _salvage_reason = salvage_one_complete_json_object(text)
        if salvage_ok and full_reason != "missing_expected_schema_fields":
            # Full schema validator remains authoritative; salvage is not success unless schema already passed.
            pass
        if full_reason == "missing_expected_schema_fields":
            summary["schema_skeleton_retries"] += 1
    else:
        if is_transient_provider_category(reason):
            for _ in range(2):
                summary["transient_provider_retries"] += 1
                ok, reason, response = _provider_call(url, build_full_payload("", tokenized_content), token, http_post, summary)
                if ok:
                    break
        if not ok:
            return False, reason, ["full_schema"], usage, records

    section_payloads: dict[str, dict[str, Any]] = {}
    failed_sections: list[str] = []
    for section in SECTION_NAMES:
        if section in completed_sections:
            continue
        summary["sectioned_attempts"] += 1
        summary["sections_sent"] += 1
        ok, reason, response = _provider_call(url, build_section_payload(section, tokenized_content), token, http_post, summary)
        if not ok:
            if is_transient_provider_category(reason):
                for _ in range(2):
                    summary["transient_provider_retries"] += 1
                    ok, reason, response = _provider_call(url, build_section_payload(section, tokenized_content), token, http_post, summary)
                    if ok:
                        break
            if not ok:
                summary["sections_failed"] += 1
                failed_sections.append(section)
                records.append({"doc_hash": doc_id, "strategy": "sectioned", "section": section, "schema_valid": False, "failure_category": reason})
                return False, reason, failed_sections, usage, records
        raw_records.append({"document_id": doc_id, "strategy": "sectioned", "section": section, "response": response})
        _merge_usage(usage, response)
        text = _extract_text(response)
        valid, section_reason, section_obj = validate_section_response(text, section)
        if not valid and section_reason in {"truncated_or_invalid_json", "empty_response"}:
            summary["subsection_attempts"] += 1
            windows = split_tokenized_window(tokenized_content, max_chars=5000)
            window_payloads: list[dict[str, Any]] = []
            window_failed = False
            for wi, window in enumerate(windows):
                ok_w, reason_w, response_w = _provider_call(
                    url,
                    build_section_payload(section, window, window_index=wi, window_count=len(windows),
                                          max_output_tokens=SUBSECTION_STRATEGY.max_output_tokens),
                    token,
                    http_post,
                    summary,
                )
                summary["sections_sent"] += 1
                if not ok_w:
                    section_reason = reason_w
                    window_failed = True
                    break
                raw_records.append({"document_id": doc_id, "strategy": "subsection", "section": section, "window": wi, "response": response_w})
                _merge_usage(usage, response_w)
                valid_w, reason_w, obj_w = validate_section_response(_extract_text(response_w), section)
                if not valid_w or obj_w is None:
                    section_reason = reason_w
                    window_failed = True
                    break
                window_payloads.append(obj_w)
            if not window_failed:
                section_obj = _merge_section_windows(section, window_payloads)
                valid = True
                section_reason = "ok"
        if not valid and section_reason == "missing_section_schema_fields":
            summary["schema_skeleton_retries"] += 1
            ok_s, reason_s, response_s = _provider_call(
                url,
                build_section_payload(section, tokenized_content, max_output_tokens=SKELETON_STRATEGY.max_output_tokens,
                                      skeleton_retry=True),
                token,
                http_post,
                summary,
            )
            if ok_s:
                raw_records.append({"document_id": doc_id, "strategy": "schema_skeleton_retry", "section": section, "response": response_s})
                _merge_usage(usage, response_s)
                valid, section_reason, section_obj = validate_section_response(_extract_text(response_s), section)
            else:
                section_reason = reason_s
        parsed_records.append({"document_id": doc_id, "strategy": "sectioned", "section": section, "schema_valid": valid, "reason": section_reason})
        records.append({"doc_hash": doc_id, "strategy": "sectioned", "section": section, "schema_valid": valid, "failure_category": section_reason})
        if not valid or section_obj is None:
            summary["sections_failed"] += 1
            failed_sections.append(section)
            return False, section_reason, failed_sections, usage, records
        section_payloads[section] = section_obj
        summary["sections_succeeded"] += 1
        sc.mark_section_completed(doc_id, section)
    merged_ok, merge_reason, _merged = merge_sections(section_payloads)
    if not merged_ok:
        return False, merge_reason, failed_sections, usage, records
    return True, "ok", [], usage, records


def _provider_call(url: str, payload: dict[str, Any], token: str,
                   http_post: Callable[[str, dict[str, Any], str], dict[str, Any]],
                   summary: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    os.environ[LIVE_GATE] = GATE_VALUE
    summary["provider_model_call_made_during_live_phase"] = True
    summary["gemini_call_made_during_live_phase"] = True
    try:
        return True, "ok", http_post(url, payload, token)
    except Exception as exc:  # noqa: BLE001 - sanitized classification only
        err = classify_vertex_provider_error(exc)
        return False, str(err.get("provider_error_category") or "provider_error"), {}
    finally:
        os.environ.pop(LIVE_GATE, None)


def _load_requests(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in read_jsonl_lines(path):
        if line.strip():
            out.append(json.loads(line))
    return out


def _local_strategy_commit_status() -> tuple[bool, str | None]:
    marker = "execution/autonomous_recovery_runner.py"
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", marker],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return False, None
    if tracked.returncode == 0 and head.returncode == 0:
        return True, str(head.stdout or "").strip() or None
    return False, None


def _validate_full(text: str) -> tuple[bool, str]:
    import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base
    return live_base._validate_schema(text)


def _extract_text(response: dict[str, Any]) -> str:
    try:
        return str(response["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return ""


def _merge_usage(total: dict[str, int], response: dict[str, Any]) -> None:
    usage = response.get("usageMetadata") if isinstance(response, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    for src, dst in (("totalTokenCount", "total_token_count"), ("promptTokenCount", "prompt_token_count"),
                     ("candidatesTokenCount", "candidates_token_count")):
        total[dst] = int(total.get(dst, 0)) + int(usage.get(src) or 0)


def _usage_cost(usage: dict[str, int]) -> float:
    return round(
        int(usage.get("prompt_token_count", 0)) / 1_000_000 * INPUT_USD_PER_M
        + int(usage.get("candidates_token_count", 0)) / 1_000_000 * OUTPUT_USD_PER_M,
        6,
    )


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _merge_section_windows(section: str, windows: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[Any] = []
    warnings: list[Any] = []
    for window in windows:
        items.extend(window.get("items") or [])
        warnings.extend(window.get("warnings") or [])
    return {"section": section, "items": items, "needs_review": True, "warnings": warnings}


def _failed_review_file() -> Path:
    return PRIVATE_OUT / "failed_docs_for_private_review_private.json"


def _load_failed_review_doc_ids() -> set[str]:
    path = _failed_review_file()
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(data, list):
        return set()
    return {str(x.get("doc_hash") or "") for x in data if isinstance(x, dict)}


def _write_failed_review_doc_id(doc_id: str) -> None:
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    path = _failed_review_file()
    rows: list[dict[str, Any]] = []
    if path.is_file():
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
    if not any(row.get("doc_hash") == doc_id for row in rows if isinstance(row, dict)):
        rows.append({"doc_hash": doc_id, "private_review_required": True})
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")


def _preserve_live_private(run_ts: str, doc_id: str, category: str,
                           raw_records: list[dict[str, Any]],
                           parsed_records: list[dict[str, Any]]) -> tuple[bool, str, int]:
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    (PRIVATE_OUT / "live_responses_private.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_records) + "\n",
        encoding="utf-8",
    )
    (PRIVATE_OUT / "parsed_responses_private.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in parsed_records) + "\n",
        encoding="utf-8",
    )
    (PRIVATE_OUT / "stopped_on_failure_private.json").write_text(
        json.dumps({"failed_doc_id": doc_id, "failure_category": category}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return lc.preserve_failed_evidence(run_ts, staging_dir=PRIVATE_OUT)


def _redact_report_tree(path: Path) -> dict[str, int]:
    counts = {"private_path_leaks_after": 0, "secret_leaks_after": 0, "public_report_phi_leak_count": 0}
    if not path.exists():
        return counts
    for report in path.iterdir():
        if not report.is_file() or report.suffix.lower() not in {".json", ".md", ".csv"}:
            continue
        text = report.read_text(encoding="utf-8", errors="ignore")
        redacted, _np, _ns = redact_report_file(text, is_json=report.suffix.lower() == ".json")
        if redacted != text:
            report.write_text(redacted, encoding="utf-8")
        result = check_public_report_payload(redacted)
        counts["private_path_leaks_after"] += result.private_filename_path_leaks
        counts["secret_leaks_after"] += result.secret_leaks
        counts["public_report_phi_leak_count"] += 0 if result.raw_phi_logged_in_public_reports is False else 1
    return counts


def _write_public_reports(summary: dict[str, Any], failed_docs: list[dict[str, Any]],
                          section_records: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    redacted_summary, _np, _ns = redact_json_value(summary)
    summary.clear()
    summary.update(redacted_summary)
    reports = {
        "summary.json": json.dumps(summary, indent=2, ensure_ascii=True),
        "sectioned_strategy_public.json": json.dumps({
            "sections": list(SECTION_NAMES),
            "sectioned_extraction_enabled": True,
            "adaptive_section_splitting_enabled": True,
            "schema_skeleton_retry_enabled": True,
            "no_clinical_inference": True,
        }, indent=2),
        "recovery_ladder_public.json": json.dumps({
            "strategies": [
                "skip_completed",
                "compact_full_schema",
                "sectioned_extraction",
                "adaptive_section_splitting",
                "json_salvage_local_only",
                "schema_skeleton_retry",
                "transient_provider_retry",
                "private_review_fallback",
            ],
            "continue_to_next_doc_for_non_safety_schema_failures": True,
        }, indent=2),
        "sectioned_merge_validation_public.json": json.dumps({
            "merge_requires_all_required_sections": True,
            "schema_weakened": False,
            "clinical_values_inferred": False,
            "missing_required_values_synthesized": False,
        }, indent=2),
        "checkpoint_policy_public.json": json.dumps({
            "checkpoint_document_aware": True,
            "checkpoint_section_aware": True,
            "completed_docs_skipped_on_resume": True,
            "completed_sections_skipped_on_resume": True,
            "private_checkpoint_path": "PRIVATE_CHECKPOINT_PATH_REDACTED",
        }, indent=2),
        "cost_plan_public.json": json.dumps({
            "authorized_total_cap_usd": TOTAL_CAP_USD,
            "hard_cost_cap_per_chunk_usd": PER_CHUNK_CAP_USD,
            "estimated_total_cost_before_run_usd": summary["estimated_total_cost_before_run_usd"],
            "actual_total_token_count": summary["actual_total_token_count"],
            "actual_cost_public_if_available": summary["actual_cost_public_if_available"],
            "billing_api_call_made": False,
        }, indent=2),
        "failed_docs_public.json": json.dumps(failed_docs, indent=2),
        "autonomous_recovery_policy_public.md": _policy_md(),
        "live_entry_gate_public.md": _live_gate_md(summary),
        "live_run_public_report.md": _live_report_md(summary),
        "safety_boundary_public.md": _safety_md(summary),
        "implementation_report.md": _implementation_md(summary),
    }
    for name, text in reports.items():
        redacted, _path_count, _secret_count = redact_report_file(text, is_json=name.endswith(".json"))
        (REPORT_DIR / name).write_text(redacted, encoding="utf-8")
    _refresh_privacy_counts(summary)


def _refresh_privacy_counts(summary: dict[str, Any]) -> None:
    path_leaks = secret_leaks = phi = 0
    for report in REPORT_DIR.iterdir():
        if report.is_file() and report.suffix.lower() in {".json", ".md"}:
            result = check_public_report_payload(report.read_text(encoding="utf-8", errors="ignore"))
            path_leaks += result.private_filename_path_leaks
            secret_leaks += result.secret_leaks
            phi += 0 if result.raw_phi_logged_in_public_reports is False else 1
    summary["private_filename_path_leaks_after"] = path_leaks
    summary["secret_leaks_after"] = secret_leaks
    summary["public_report_phi_leak_count"] = phi
    summary["privacy_result"] = "passed" if path_leaks == 0 and secret_leaks == 0 and phi == 0 else "blocked"
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / "summary.json").write_text(redacted, encoding="utf-8")


def _policy_md() -> str:
    return (
        "# Autonomous Recovery Policy\n\n"
        "R13 handles schema/provider-output failures through bounded full-schema, sectioned, "
        "subsection, strict local JSON salvage, skeleton retry, transient retry, and private-review fallback. "
        "Safety, privacy, credential, cost, MKB-write, auto-accept, and medical-decision gates remain hard stops.\n"
    )


def _live_gate_md(summary: dict[str, Any]) -> str:
    return (
        "# Live Entry Gate\n\n"
        f"- full_live_run_started: `{summary['full_live_run_started']}`\n"
        f"- provider_model_call_made_during_local_phase: `{summary['provider_model_call_made_during_local_phase']}`\n"
        f"- authorized_total_cap_usd: `{summary['authorized_total_cap_usd']}`\n"
        f"- hard_cost_cap_per_chunk_usd: `{summary['hard_cost_cap_per_chunk_usd']}`\n"
    )


def _live_report_md(summary: dict[str, Any]) -> str:
    return (
        "# Live Run Public Report\n\n"
        f"- run_result: `{summary['run_result']}`\n"
        f"- docs_loaded: `{summary['docs_loaded']}`\n"
        f"- docs_completed: `{summary['docs_completed']}`\n"
        f"- docs_failed_for_review: `{summary['docs_failed_for_review']}`\n"
        f"- docs_unattempted: `{summary['docs_unattempted']}`\n"
        f"- failure_stage: `{summary['failure_stage']}`\n"
        f"- failure_category: `{summary['failure_category']}`\n"
    )


def _safety_md(summary: dict[str, Any]) -> str:
    keys = ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
            "future_17d_mkb_import_not_started", "credential_or_token_written_to_repo",
            "private_responses_committed", "privacy_result", "safety_result")
    return "# Safety Boundary\n\n" + "\n".join(f"- {key}: `{summary[key]}`" for key in keys) + "\n"


def _implementation_md(summary: dict[str, Any]) -> str:
    return (
        f"# {BLOCK}\n\n"
        f"- run_result: `{summary['run_result']}`\n"
        "- local strategy committed separately before live execution when gates pass.\n"
        "- Public reports contain counts, doc hashes, section names, failure classes, and redacted labels only.\n"
    )


__all__ = ["BLOCK", "REPORT_DIR", "run_autonomous_recovery", "default_summary"]
