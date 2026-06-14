#!/usr/bin/env python3
"""Local-only R14 provider hard-stop diagnosis for 17C-R2.

Reads public R13 reports and private metadata/checkpoint files only. It does not
call Vertex/Gemini, does not refresh credentials, and does not inspect raw
provider response bodies.
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

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402
from execution.sectioned_checkpoint import load_completed_sections  # noqa: E402

BLOCK = "MEDAI-AI-FIRST-CORPUS-17C-R2-PROVIDER-HARD-STOP-DIAGNOSIS-LOCAL-ONLY-R14"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_provider_hard_stop_diagnosis_local_only_r14"
R13_REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_autonomous_recovery_full_corpus_final_17c_r2_r13"
R13_SUMMARY_PATH = R13_REPORT_DIR / "summary.json"
R13_PRIVATE_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17C_R2_R13_autonomous_recovery"))
FAILED_REVIEW_FILE = R13_PRIVATE_DIR / "failed_docs_for_private_review_private.json"
STOPPED_FILE = R13_PRIVATE_DIR / "stopped_on_failure_private.json"
EXPECTED_TOTAL = 478


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _safe_message(message: str) -> str:
    text = str(message or "")
    text = re.sub(r"ya29\.[0-9A-Za-z_\-.]+", "<redacted-credential>", text)
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}", "<redacted-credential>", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "<redacted-credential>", text)
    text = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<redacted-path>", text)
    return text[:300] if len(text) > 300 else text


def _classify_error(prior_category: str, stopped: dict[str, Any] | None) -> dict[str, Any]:
    message = ""
    status = "unknown"
    category = str(prior_category or "unknown_provider_hard_stop")
    if isinstance(stopped, dict):
        message = str(stopped.get("provider_error_message_sanitized") or stopped.get("error_message") or "")
        status = str(stopped.get("provider_error_status_code") or stopped.get("status_code") or "unknown")
        category = str(stopped.get("failure_category") or category)
    diagnostic_low = " ".join([message, status]).lower()
    category_low = category.lower()
    low = " ".join([category, message, status]).lower()
    exact = "unknown_provider_hard_stop"
    if "quota" in diagnostic_low or "resourceexhausted" in diagnostic_low or "429" in diagnostic_low:
        exact = "quota_exhausted"
    elif "billing" in diagnostic_low or "credit" in diagnostic_low:
        exact = "billing_or_credit_block"
    elif "permission" in diagnostic_low or "forbidden" in diagnostic_low or "unauth" in diagnostic_low or "403" in diagnostic_low:
        exact = "permission_denied"
    elif "disabled" in diagnostic_low or "api has not been used" in diagnostic_low or "serviceusage" in diagnostic_low:
        exact = "api_disabled"
    elif "not found" in diagnostic_low or "404" in diagnostic_low or "model" in diagnostic_low:
        exact = "model_unavailable"
    elif "region" in diagnostic_low or "location" in diagnostic_low:
        exact = "region_unavailable"
    elif "rate" in diagnostic_low or "timeout" in diagnostic_low or "unavailable" in diagnostic_low:
        exact = "rate_limit_or_transient"
    elif "invalid" in diagnostic_low or "400" in diagnostic_low or "malformed" in diagnostic_low:
        exact = "malformed_request"
    return {
        "exact_provider_error_class": exact,
        "exact_provider_status_code_public": status or "unknown",
        "exact_provider_error_message_sanitized": _safe_message(message) if message else "unavailable_from_private_metadata",
        "quota_or_billing_suspected": exact in {"quota_exhausted", "billing_or_credit_block"},
        "api_or_permission_suspected": exact in {"api_disabled", "permission_denied"} or category_low == "api_disabled_or_permission",
        "malformed_request_suspected": exact == "malformed_request",
    }


def _load_failed_review_count() -> int:
    rows = _read_json(FAILED_REVIEW_FILE)
    if not isinstance(rows, list):
        return 0
    return sum(1 for row in rows if isinstance(row, dict) and row.get("doc_hash"))


def build_summary() -> dict[str, Any]:
    r13 = _read_json(R13_SUMMARY_PATH) or {}
    stopped = _read_json(STOPPED_FILE)
    completed_docs = lc.load_completed()
    completed_count = len(set(completed_docs))
    failed_review_count = _load_failed_review_count()
    completed_sections = load_completed_sections()
    trace_available = STOPPED_FILE.is_file() or FAILED_REVIEW_FILE.is_file() or bool(completed_sections)
    error = _classify_error(str(r13.get("failure_category") or ""), stopped if isinstance(stopped, dict) else None)
    remaining = int(r13.get("docs_failed_for_review") or failed_review_count or max(0, EXPECTED_TOTAL - completed_count))
    resume_supported = (
        completed_count == int(r13.get("docs_completed") or completed_count)
        and bool(r13.get("completed_docs_skipped_on_resume", True))
        and completed_count > 0
    )
    requires_code_change = bool(error["malformed_request_suspected"])
    requires_provider_action = not requires_code_change and (
        error["api_or_permission_suspected"] or error["quota_or_billing_suspected"]
        or error["exact_provider_error_class"] == "unknown_provider_hard_stop"
    )
    ready_to_resume = resume_supported and requires_provider_action and not requires_code_change
    return {
        "block": BLOCK,
        "local_only": True,
        "provider_generation_call_made": False,
        "billing_api_call_made": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "prior_r13_result": str(r13.get("run_result") or "unknown"),
        "prior_failure_stage": str(r13.get("failure_stage") or "unknown"),
        "prior_failure_category": str(r13.get("failure_category") or "unknown"),
        "prior_docs_completed": int(r13.get("docs_completed") or 0),
        "prior_docs_failed_for_review": int(r13.get("docs_failed_for_review") or 0),
        "prior_actual_token_count": int(r13.get("actual_total_token_count") or 0),
        "prior_actual_cost_public_if_available": f"${float(r13.get('actual_cost_public_if_available') or 0):.6f}",
        **error,
        "provider_trace_available_private": bool(trace_available),
        "credential_preflight_passed": bool(r13.get("provider_model_call_made_during_live_phase") is True),
        "completed_docs_checkpointed": completed_count == int(r13.get("docs_completed") or 0),
        "completed_doc_count_checkpointed": completed_count,
        "completed_section_doc_count_checkpointed": len(completed_sections),
        "resume_without_completed_doc_resend_supported": resume_supported,
        "remaining_docs_estimated": remaining,
        "ready_to_resume_after_provider_fix": ready_to_resume,
        "requires_code_change_before_resume": requires_code_change,
        "requires_provider_account_action": requires_provider_action,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed" if trace_available and completed_count == int(r13.get("docs_completed") or 0) else "blocked",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    path.write_text(redacted + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    redacted, _np, _ns = redact_report_file(text, is_json=False)
    path.write_text(redacted, encoding="utf-8")


def write_reports(summary: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "prior_failure_category": summary["prior_failure_category"],
        "exact_provider_error_class": summary["exact_provider_error_class"],
        "exact_provider_status_code_public": summary["exact_provider_status_code_public"],
        "exact_provider_error_message_sanitized": summary["exact_provider_error_message_sanitized"],
        "quota_or_billing_suspected": summary["quota_or_billing_suspected"],
        "api_or_permission_suspected": summary["api_or_permission_suspected"],
        "malformed_request_suspected": summary["malformed_request_suspected"],
        "provider_trace_available_private": summary["provider_trace_available_private"],
    }
    checkpoint = {
        "completed_docs_checkpointed": summary["completed_docs_checkpointed"],
        "completed_doc_count_checkpointed": summary["completed_doc_count_checkpointed"],
        "remaining_docs_estimated": summary["remaining_docs_estimated"],
        "resume_without_completed_doc_resend_supported": summary["resume_without_completed_doc_resend_supported"],
        "ready_to_resume_after_provider_fix": summary["ready_to_resume_after_provider_fix"],
    }
    _write_json(REPORT_DIR / "summary.json", summary)
    _write_json(REPORT_DIR / "provider_hard_stop_diagnosis_public.json", diagnosis)
    _write_json(REPORT_DIR / "checkpoint_resume_status_public.json", checkpoint)
    _write_md(
        REPORT_DIR / "recommended_next_action_public.md",
        "# Recommended Next Action\n\n"
        f"- exact_provider_error_class: `{summary['exact_provider_error_class']}`\n"
        f"- requires_code_change_before_resume: `{summary['requires_code_change_before_resume']}`\n"
        f"- requires_provider_account_action: `{summary['requires_provider_account_action']}`\n\n"
        "If provider/account access is repaired, resume with the R13 autonomous runner. "
        "The completed-doc checkpoint supports skipping already completed documents. "
        "No code repair is indicated unless new provider metadata proves a malformed request.\n",
    )
    _write_md(
        REPORT_DIR / "safety_boundary_public.md",
        "# Safety Boundary\n\n"
        "- local_only: `true`\n"
        "- provider_generation_call_made: `false`\n"
        "- billing_api_call_made: `false`\n"
        "- mkb_db_opened: `false`\n"
        "- active_mkb_write: `false`\n"
        "- raw provider bodies inspected: `false`\n"
        "- private artifacts committed: `false`\n",
    )
    _write_md(
        REPORT_DIR / "implementation_report.md",
        f"# {BLOCK}\n\n"
        f"- prior R13 result: `{summary['prior_r13_result']}`\n"
        f"- exact provider error class: `{summary['exact_provider_error_class']}`\n"
        f"- completed docs checkpointed: `{summary['completed_doc_count_checkpointed']}`\n"
        f"- remaining docs estimated: `{summary['remaining_docs_estimated']}`\n"
        f"- resume without completed-doc resend supported: `{summary['resume_without_completed_doc_resend_supported']}`\n"
        f"- provider/account action required: `{summary['requires_provider_account_action']}`\n",
    )
    return _refresh_privacy(summary)


def _refresh_privacy(summary: dict[str, Any]) -> dict[str, Any]:
    path_leaks = secret_leaks = phi = 0
    for path in REPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            result = check_public_report_payload(path.read_text(encoding="utf-8", errors="ignore"))
            path_leaks += result.private_filename_path_leaks
            secret_leaks += result.secret_leaks
            phi += 0 if result.raw_phi_logged_in_public_reports is False else 1
    summary["private_path_leaks_after"] = path_leaks
    summary["secret_leaks_after"] = secret_leaks
    summary["public_report_phi_leak_count"] = phi
    summary["privacy_result"] = "passed" if path_leaks == 0 and secret_leaks == 0 and phi == 0 else "blocked"
    _write_json(REPORT_DIR / "summary.json", summary)
    return summary


def main() -> int:
    summary = build_summary()
    summary = write_reports(summary)
    result = "PASS" if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else "BLOCKED"
    print(f"{BLOCK}_{result}")
    print(json.dumps({
        "exact_provider_error_class": summary["exact_provider_error_class"],
        "exact_provider_status_code_public": summary["exact_provider_status_code_public"],
        "provider_trace_available_private": summary["provider_trace_available_private"],
        "completed_doc_count_checkpointed": summary["completed_doc_count_checkpointed"],
        "resume_without_completed_doc_resend_supported": summary["resume_without_completed_doc_resend_supported"],
        "requires_code_change_before_resume": summary["requires_code_change_before_resume"],
        "requires_provider_account_action": summary["requires_provider_account_action"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
    }, indent=2, sort_keys=True))
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
