#!/usr/bin/env python3
"""R15 resume supervisor and final live continuation wrapper.

Local-only mode classifies the current partial checkpoint and writes public-safe
preflight/simulation reports. Live mode first repeats those gates, then runs the
R13 autonomous recovery runner with stale failed-review skip metadata ignored so
completed docs are preserved and remaining docs can be attempted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.gemini_vertex_adapter import GeminiVertexConfig  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.public_report_redaction import redact_json_value, redact_report_file  # noqa: E402
from execution.resume_supervisor import (  # noqa: E402
    EXPECTED_COMPLETED_FOR_R15,
    EXPECTED_DOCS,
    PER_CHUNK_CAP_USD,
    TARGET_MODEL,
    TOTAL_CAP_USD,
    archive_failed_review_skip_file,
    build_preflight_matrix,
    classify_checkpoint_state,
    section_completed_doc_count,
    simulate_resume,
)
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402
import scripts.run_medai_ai_first_corpus_17c_r2_checkpoint_unblock_and_report_redaction_local_only_17c_r2_r12 as r12  # noqa: E402
from execution.autonomous_recovery_runner import run_autonomous_recovery  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-RESUME-SUPERVISOR-AND-FINAL-LIVE-CONTINUATION-R15"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_resume_supervisor_and_final_live_continuation_r15"


def _summary_base() -> dict[str, Any]:
    return {
        "block": BLOCK,
        "local_strategy_committed": False,
        "local_strategy_commit_sha": None,
        "live_continuation_started": False,
        "provider_model_call_made_during_live_phase": False,
        "gemini_call_made_during_live_phase": False,
        "checkpoint_state_before": "unknown_checkpoint_state",
        "docs_loaded": EXPECTED_DOCS,
        "docs_completed_before": 0,
        "docs_remaining_selected_before": 0,
        "completed_docs_preserved": False,
        "completed_docs_skipped_on_resume": True,
        "completed_sections_skipped_on_resume": True,
        "unresolved_failed_checkpoint_before": False,
        "checkpoint_corruption_detected": False,
        "canonical_sha256_match": False,
        "preflight_matrix_passed": False,
        "dry_run_resume_simulation_passed": False,
        "authorized_total_cap_usd": TOTAL_CAP_USD,
        "hard_cost_cap_per_chunk_usd": PER_CHUNK_CAP_USD,
        "estimated_remaining_cost_usd": 0.0,
        "cost_caps_passed": False,
        "target_model": TARGET_MODEL,
        "docs_completed_after": 0,
        "docs_failed_for_review_after": 0,
        "docs_unattempted_after": 0,
        "sections_sent_live": 0,
        "sections_succeeded_live": 0,
        "sections_failed_live": 0,
        "run_result": "BLOCKED",
        "failure_stage": "none",
        "failure_category": "none",
        "failed_doc_hash": None,
        "failed_section": None,
        "failed_evidence_preserved": False,
        "actual_total_token_count": 0,
        "actual_cost_public_if_available": "unknown",
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


def _load_order() -> tuple[str, list[str], bool]:
    return r12._load_canonical_order()


def _load_requests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _git_status() -> tuple[bool, str | None]:
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch",
             "scripts/run_medai_ai_first_corpus_17c_r2_resume_supervisor_and_final_live_continuation_r15.py"],
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


def build_local_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    summary = _summary_base()
    committed, sha = _git_status()
    summary["local_strategy_committed"] = committed
    summary["local_strategy_commit_sha"] = sha
    batch_sha, order, canonical_valid = _load_order()
    requests = _load_requests() if canonical_valid else []
    classification = classify_checkpoint_state(batch_sha=batch_sha, order=order, total=EXPECTED_DOCS)
    cred_ok, _cred_reason = live_base._credential_preflight()
    per_doc_tokens = [_approx_tokens(str(req.get("tokenized_content") or "")) for req in requests]
    config = GeminiVertexConfig()
    preflight = build_preflight_matrix(
        classification=classification,
        canonical_batch_valid=canonical_valid and len(requests) == EXPECTED_DOCS and len(set(order)) == EXPECTED_DOCS,
        credential_preflight_passed=cred_ok,
        project_id=config.vertex_project_id,
        vertex_route_configured=config.provider_route == "vertex" and config.vertex_model == TARGET_MODEL,
        per_doc_input_tokens=per_doc_tokens,
    )
    simulation = simulate_resume(order=order, classification=classification)
    summary.update({
        "checkpoint_state_before": classification.state,
        "docs_loaded": len(requests),
        "docs_completed_before": classification.completed_count,
        "docs_remaining_selected_before": classification.remaining_count,
        "completed_docs_preserved": classification.completed_docs_preserved and classification.completed_count == EXPECTED_COMPLETED_FOR_R15,
        "completed_docs_skipped_on_resume": classification.completed_docs_skipped_on_resume,
        "completed_sections_skipped_on_resume": section_completed_doc_count() > 0,
        "unresolved_failed_checkpoint_before": classification.unresolved_failed_checkpoint,
        "checkpoint_corruption_detected": classification.checkpoint_corruption_detected,
        "canonical_sha256_match": classification.canonical_sha256_match,
        "preflight_matrix_passed": preflight["preflight_matrix_passed"],
        "dry_run_resume_simulation_passed": simulation["simulation_passed"],
        "estimated_remaining_cost_usd": preflight["estimated_remaining_cost_usd"],
        "cost_caps_passed": preflight["cost_caps_passed"],
    })
    if summary["preflight_matrix_passed"] and summary["dry_run_resume_simulation_passed"]:
        summary["run_result"] = "PASS"
        summary["safety_result"] = "passed"
    else:
        summary["run_result"] = "BLOCKED"
        summary["failure_stage"] = "local_preflight_block"
        summary["failure_category"] = "resume_supervisor_preflight_failed"
        summary["safety_result"] = "blocked"
    return summary, preflight, simulation, _checkpoint_public(classification), order


def run(live: bool = False) -> dict[str, Any]:
    summary, preflight, simulation, checkpoint_public, _order = build_local_state()
    failed_docs: list[dict[str, Any]] = []
    if live:
        if not (summary["preflight_matrix_passed"] and summary["dry_run_resume_simulation_passed"]):
            _write_reports(summary, checkpoint_public, preflight, simulation, failed_docs)
            return summary
        archive_failed_review_skip_file()
        r13 = run_autonomous_recovery(live=True, ignore_failed_review_skip=True)
        failed_docs = _load_r13_failed_docs_public()
        summary.update({
            "live_continuation_started": bool(r13.get("full_live_run_started")),
            "provider_model_call_made_during_live_phase": bool(r13.get("provider_model_call_made_during_live_phase")),
            "gemini_call_made_during_live_phase": bool(r13.get("gemini_call_made_during_live_phase")),
            "docs_completed_after": int(r13.get("docs_completed") or len(lc.load_completed())),
            "docs_failed_for_review_after": int(r13.get("docs_failed_for_review") or 0),
            "docs_unattempted_after": int(r13.get("docs_unattempted") or 0),
            "sections_sent_live": int(r13.get("sections_sent") or 0),
            "sections_succeeded_live": int(r13.get("sections_succeeded") or 0),
            "sections_failed_live": int(r13.get("sections_failed") or 0),
            "failure_stage": str(r13.get("failure_stage") or "none"),
            "failure_category": str(r13.get("failure_category") or "none"),
            "failed_doc_hash": r13.get("failed_doc_hash"),
            "failed_section": r13.get("failed_section"),
            "failed_evidence_preserved": bool(r13.get("failed_evidence_preserved")),
            "actual_total_token_count": int(r13.get("actual_total_token_count") or 0),
            "actual_cost_public_if_available": r13.get("actual_cost_public_if_available", "unknown"),
            "live_gate_environment_active_after_run": bool(r13.get("live_gate_environment_active_after_run")),
            "active_mkb_write": bool(r13.get("active_mkb_write")),
            "mkb_db_opened": bool(r13.get("mkb_db_opened")),
            "auto_accept_enabled": bool(r13.get("auto_accept_enabled")),
            "medical_decision_made": bool(r13.get("medical_decision_made")),
        })
        if r13.get("run_result") == "PASS":
            summary["run_result"] = "PASS"
            summary["failure_stage"] = "none"
            summary["failure_category"] = "none"
            summary["safety_result"] = "passed"
        elif summary["live_continuation_started"]:
            summary["run_result"] = "LIVE_FAIL"
            summary["failure_stage"] = "provider_live_fail" if summary["failure_category"] in {
                "api_disabled_or_permission", "quota_or_billing", "model_or_endpoint_not_found",
                "timeout", "unknown_provider_error"
            } else summary["failure_stage"]
            summary["safety_result"] = "passed" if not summary["active_mkb_write"] else "blocked"
        else:
            summary["run_result"] = "BLOCKED"
            summary["failure_stage"] = "local_preflight_block"
            summary["safety_result"] = "blocked"
    _write_reports(summary, checkpoint_public, preflight, simulation, failed_docs)
    return summary


def _checkpoint_public(classification) -> dict[str, Any]:
    return {
        "checkpoint_state": classification.state,
        "completed_doc_count": classification.completed_count,
        "remaining_doc_count": classification.remaining_count,
        "canonical_sha256_match": classification.canonical_sha256_match,
        "unresolved_failed_checkpoint": classification.unresolved_failed_checkpoint,
        "checkpoint_corruption_detected": classification.checkpoint_corruption_detected,
        "first_continuation_doc_id": classification.first_continuation_doc_id,
        "completed_docs_skipped_on_resume": classification.completed_docs_skipped_on_resume,
        "failed_review_count_private_metadata": classification.failed_review_count_private,
    }


def _load_r13_failed_docs_public() -> list[dict[str, Any]]:
    path = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_autonomous_recovery_full_corpus_final_17c_r2_r13" / "failed_docs_public.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    path.write_text(redacted + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    redacted, _np, _ns = redact_report_file(text, is_json=False)
    path.write_text(redacted, encoding="utf-8")


def _write_reports(summary: dict[str, Any], checkpoint: dict[str, Any],
                   preflight: dict[str, Any], simulation: dict[str, Any],
                   failed_docs: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    redacted_summary, _np, _ns = redact_json_value(summary)
    summary.clear()
    summary.update(redacted_summary)
    _write_json(REPORT_DIR / "summary.json", summary)
    _write_json(REPORT_DIR / "checkpoint_state_public.json", checkpoint)
    _write_json(REPORT_DIR / "preflight_matrix_public.json", preflight)
    _write_json(REPORT_DIR / "dry_run_resume_simulation_public.json", simulation)
    _write_json(REPORT_DIR / "failed_docs_public.json", failed_docs)
    _write_md(REPORT_DIR / "live_continuation_public_report.md", _live_md(summary))
    _write_md(REPORT_DIR / "safety_boundary_public.md", _safety_md(summary))
    _write_md(REPORT_DIR / "implementation_report.md", _implementation_md(summary))
    _refresh_privacy(summary)


def _refresh_privacy(summary: dict[str, Any]) -> None:
    path_leaks = secret_leaks = phi = 0
    for path in REPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            result = check_public_report_payload(path.read_text(encoding="utf-8", errors="ignore"))
            path_leaks += result.private_filename_path_leaks
            secret_leaks += result.secret_leaks
            phi += 0 if result.raw_phi_logged_in_public_reports is False else 1
    summary["private_filename_path_leaks_after"] = path_leaks
    summary["secret_leaks_after"] = secret_leaks
    summary["public_report_phi_leak_count"] = phi
    summary["privacy_result"] = "passed" if path_leaks == 0 and secret_leaks == 0 and phi == 0 else "blocked"
    text = json.dumps(summary, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / "summary.json").write_text(redacted + "\n", encoding="utf-8")


def _live_md(summary: dict[str, Any]) -> str:
    return (
        "# R15 live continuation public report\n\n"
        f"- run_result: `{summary['run_result']}`\n"
        f"- live_continuation_started: `{summary['live_continuation_started']}`\n"
        f"- docs_completed_after: `{summary['docs_completed_after']}`\n"
        f"- docs_failed_for_review_after: `{summary['docs_failed_for_review_after']}`\n"
        f"- docs_unattempted_after: `{summary['docs_unattempted_after']}`\n"
        f"- failure_stage: `{summary['failure_stage']}`\n"
        f"- failure_category: `{summary['failure_category']}`\n"
    )


def _safety_md(summary: dict[str, Any]) -> str:
    keys = ("mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
            "future_17d_mkb_import_not_started", "live_gate_environment_active_after_run",
            "private_checkpoint_committed", "private_evidence_committed", "private_responses_committed",
            "credential_or_token_written_to_repo", "privacy_result", "safety_result")
    return "# R15 safety boundary\n\n" + "\n".join(f"- {key}: `{summary[key]}`" for key in keys) + "\n"


def _implementation_md(summary: dict[str, Any]) -> str:
    return (
        f"# {BLOCK}\n\n"
        "R15 classifies clean partial checkpoints as resumable, preserves completed doc IDs, "
        "skips completed docs, and isolates stale failed-review skip metadata before supervised live continuation.\n\n"
        f"- checkpoint_state_before: `{summary['checkpoint_state_before']}`\n"
        f"- preflight_matrix_passed: `{summary['preflight_matrix_passed']}`\n"
        f"- dry_run_resume_simulation_passed: `{summary['dry_run_resume_simulation_passed']}`\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    summary = run(live=bool(args.live))
    print(f"{BLOCK}_{summary['run_result']}")
    print(json.dumps({
        "run_result": summary["run_result"],
        "checkpoint_state_before": summary["checkpoint_state_before"],
        "docs_completed_before": summary["docs_completed_before"],
        "docs_remaining_selected_before": summary["docs_remaining_selected_before"],
        "preflight_matrix_passed": summary["preflight_matrix_passed"],
        "dry_run_resume_simulation_passed": summary["dry_run_resume_simulation_passed"],
        "live_continuation_started": summary["live_continuation_started"],
        "provider_model_call_made_during_live_phase": summary["provider_model_call_made_during_live_phase"],
        "docs_completed_after": summary["docs_completed_after"],
        "docs_failed_for_review_after": summary["docs_failed_for_review_after"],
        "docs_unattempted_after": summary["docs_unattempted_after"],
        "failure_stage": summary["failure_stage"],
        "failure_category": summary["failure_category"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
    }, indent=2, sort_keys=True))
    return 0 if summary["run_result"] in {"PASS", "LIVE_FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
