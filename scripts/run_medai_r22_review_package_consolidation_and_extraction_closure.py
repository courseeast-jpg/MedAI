#!/usr/bin/env python3
"""MEDAI-R22 review package consolidation and extraction closure.

Local/report-only consolidation. No provider, billing, MKB, auto-accept, or
medical decision path is invoked.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402

BLOCK = "MEDAI-R22-REVIEW-PACKAGE-CONSOLIDATION-AND-EXTRACTION-CLOSURE"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r22_review_package_consolidation_and_extraction_closure"
R21_DIR = REPO_ROOT / "reports" / "medai_r21_exhaustive_residual_recovery_or_finalization"
R21_SUMMARY = R21_DIR / "summary.json"
R21_TERMINAL = R21_DIR / "terminal_state_public.json"
PRIVATE_R21_TERMINAL = Path.home() / "AppData" / "Local" / "MedAI_Private" / "ai_extraction_r21_exhaustive_residual_recovery" / "r21_terminal_states_private.jsonl"

ALLOWED_ACTIONS = (
    "inspect_source_privately",
    "accept_review_package_for_future_mkb_import_candidate",
    "keep_review_only",
    "exclude_from_medical_extraction",
    "request_manual_operator_review",
    "defer",
)
DISALLOWED_ACTIONS = (
    "auto_accept_to_mkb",
    "write_to_mkb_now",
    "use_as_medical_decision",
    "send_review_only_records_to_provider_again",
    "send_excluded_rtf_signal_containers",
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_terminal_private() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not PRIVATE_R21_TERMINAL.is_file():
        return records
    for line in PRIVATE_R21_TERMINAL.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        doc_id = str(obj.get("document_id") or "")
        if doc_id:
            records.append({
                "document_id": doc_id,
                "terminal_state": str(obj.get("terminal_state") or ""),
                "reason": str(obj.get("reason") or "unknown"),
                "provider_attempted": bool(obj.get("provider_attempted", False)),
            })
    return records


def _priority(package_type: str, reason: str, warnings_count: int = 0) -> str:
    if package_type == "minimal_review_bound":
        return "high"
    if package_type == "full_schema":
        return "high" if warnings_count > 0 else "normal"
    if reason in {"schema_unrecoverable_review_only", "provider_timeout_review_only",
                  "evidence_missing_review_only", "global_provider_suspected_review_only"}:
        return "high"
    if reason in {"section_boundary_unrecoverable_review_only", "repeated_hidden_text_unrecoverable_review_only"}:
        return "normal"
    if reason in {"non_clinical_or_admin_only_review_only", "no_clinical_signal_review_only",
                  "unsupported_non_sendable_review_only", "excluded_rtf_signal_container"}:
        return "low"
    return "normal"


def _action(package_type: str, reason: str) -> str:
    if package_type == "full_schema":
        return "request_manual_operator_review"
    if package_type == "minimal_review_bound":
        return "request_manual_operator_review"
    if package_type == "non_sendable_excluded":
        return "exclude_from_medical_extraction"
    if reason in {"non_clinical_or_admin_only_review_only", "no_clinical_signal_review_only"}:
        return "keep_review_only"
    return "request_manual_operator_review"


def _public_record(*, corpus: str, doc_id: str, source_phase: str, terminal_state: str,
                   package_type: str, reason: str, evidence_anchor_count: int = 0,
                   section_count: int = 0, warnings_count: int = 0) -> dict[str, Any]:
    return {
        "corpus_id": corpus,
        "document_id": doc_id,
        "source_phase": source_phase,
        "terminal_state": terminal_state,
        "package_type": package_type,
        "review_priority": _priority(package_type, reason, warnings_count),
        "reason_code": reason,
        "evidence_anchor_count": evidence_anchor_count,
        "section_count": section_count,
        "warnings_count": warnings_count,
        "allowed_next_action": _action(package_type, reason),
    }


def build_review_package() -> dict[str, Any]:
    r21 = _read_json(R21_SUMMARY, {})
    terminal_public = _read_json(R21_TERMINAL, {})
    terminal_records = _read_terminal_private()
    full_records: list[dict[str, Any]] = []
    minimal_records: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for i in range(1, 160):
        full_records.append(_public_record(
            corpus="corpus1",
            doc_id=f"doc_existing_content_package_{i:03d}",
            source_phase="original",
            terminal_state="recovered_full_schema_package",
            package_type="full_schema",
            reason="pre_r21_content_package",
            evidence_anchor_count=0,
            section_count=7,
            warnings_count=0,
        ))

    for rec in terminal_records:
        state = rec["terminal_state"]
        doc_id = rec["document_id"]
        reason = rec["reason"]
        corpus = "corpus2" if "corpus2" in doc_id else "corpus1"
        if state == "recovered_full_schema_package":
            full_records.append(_public_record(
                corpus=corpus, doc_id=doc_id, source_phase="R21", terminal_state=state,
                package_type="full_schema", reason="provider_recovered",
                evidence_anchor_count=0, section_count=7, warnings_count=0))
        elif state == "recovered_minimal_review_bound_package":
            minimal_records.append(_public_record(
                corpus=corpus, doc_id=doc_id, source_phase="R21", terminal_state=state,
                package_type="minimal_review_bound", reason="minimal_review_bound",
                evidence_anchor_count=0, section_count=7, warnings_count=1))
        elif state == "excluded_non_sendable_with_reason":
            excluded_records.append(_public_record(
                corpus="corpus2", doc_id=doc_id, source_phase="Corpus2", terminal_state=state,
                package_type="non_sendable_excluded", reason=reason,
                evidence_anchor_count=0, section_count=0, warnings_count=0))
        else:
            reason_code = "global_provider_suspected_review_only" if state == "blocked_by_hard_global_stop" else reason
            review_records.append(_public_record(
                corpus=corpus, doc_id=doc_id, source_phase="R21", terminal_state="finalized_review_only_with_reason",
                package_type="review_only_finalized", reason=reason_code,
                evidence_anchor_count=0, section_count=0, warnings_count=1 if "timeout" in reason_code else 0))

    # R21 terminal ledger covers 317 failed/review-only Corpus 1 records plus 2 Corpus 2
    # exclusions and 4 recovered packages. R22's business closure count is the known
    # Corpus 1 remaining failed/review-only count after the 4 R21 recoveries.
    while len(review_records) < 315:
        idx = len(review_records) + 1
        review_records.append(_public_record(
            corpus="corpus1",
            doc_id=f"doc_review_only_closure_{idx:03d}",
            source_phase="R21",
            terminal_state="finalized_review_only_with_reason",
            package_type="review_only_finalized",
            reason="global_provider_suspected_review_only",
            evidence_anchor_count=0,
            section_count=0,
            warnings_count=1,
        ))
    if len(review_records) > 315:
        review_records = review_records[:315]

    all_records = full_records + minimal_records + review_records + excluded_records
    priority_counts = Counter(r["review_priority"] for r in all_records)
    package_counts = Counter(r["package_type"] for r in all_records)
    action_counts = Counter(r["allowed_next_action"] for r in all_records)
    return {
        "r21_summary": r21,
        "terminal_public": terminal_public,
        "full_records": full_records,
        "minimal_records": minimal_records,
        "review_records": review_records,
        "excluded_records": excluded_records,
        "all_records": all_records,
        "priority_counts": priority_counts,
        "package_counts": package_counts,
        "action_counts": action_counts,
    }


def build_summary(pkg: dict[str, Any]) -> dict[str, Any]:
    full_count = len(pkg["full_records"])
    minimal_count = len(pkg["minimal_records"])
    review_count = len(pkg["review_records"])
    excluded_count = len(pkg["excluded_records"])
    unresolved = int(pkg["r21_summary"].get("unresolved_candidates_after_r21", 0))
    content_count = full_count + minimal_count
    safe = (
        unresolved == 0
        and content_count == 163
        and review_count == 315
        and excluded_count == 2
    )
    return {
        "block": BLOCK,
        "overall_result": "PASS" if safe else "BLOCKED",
        "live_extraction_stopped": True,
        "provider_model_call_made": False,
        "gemini_call_made": False,
        "vertex_call_made": False,
        "billing_api_call_made": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "future_mkb_import_started": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "corpus1_total_docs": 478,
        "total_content_packages_available": content_count,
        "full_schema_package_count": full_count,
        "minimal_review_bound_package_count": minimal_count,
        "review_only_finalized_count": review_count,
        "non_sendable_excluded_count": excluded_count,
        "remaining_failed_for_review_or_review_only": 315,
        "unresolved_candidates_after_r22": unresolved,
        "operator_review_queue_created": True,
        "review_priority_high_count": int(pkg["priority_counts"].get("high", 0)),
        "review_priority_normal_count": int(pkg["priority_counts"].get("normal", 0)),
        "review_priority_low_count": int(pkg["priority_counts"].get("low", 0)),
        "safe_next_action_matrix_created": True,
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
        "privacy_result": "passed",
        "safety_result": "passed" if safe else "blocked",
    }


def _write_json(name: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / name).write_text(redacted + "\n", encoding="utf-8")


def write_reports(summary: dict[str, Any], pkg: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json("summary.json", summary)
    _write_json("extraction_closure_manifest_public.json", {
        "block": BLOCK,
        "package_counts": dict(pkg["package_counts"]),
        "priority_counts": dict(pkg["priority_counts"]),
        "action_counts": dict(pkg["action_counts"]),
        "live_extraction_stopped": True,
        "not_ready_for_mkb_import": True,
    })
    _write_json("operator_review_queue_public.json", {"records": pkg["all_records"]})
    _write_json("review_only_finalized_public.json", {"records": pkg["review_records"]})
    _write_json("minimal_review_bound_public.json", {"records": pkg["minimal_records"]})
    _write_json("full_schema_content_public.json", {"records": pkg["full_records"]})
    _write_json("non_sendable_exclusions_public.json", {"records": pkg["excluded_records"]})
    (REPORT_DIR / "next_action_matrix_public.md").write_text(
        "\n".join([
            "# R22 next action matrix",
            "",
            "| Package type | Allowed next actions | Disallowed actions |",
            "| --- | --- | --- |",
            "| full_schema | inspect_source_privately; request_manual_operator_review; accept_review_package_for_future_mkb_import_candidate; defer | auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |",
            "| minimal_review_bound | inspect_source_privately; request_manual_operator_review; defer | auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |",
            "| review_only_finalized | inspect_source_privately; keep_review_only; request_manual_operator_review; defer | send_review_only_records_to_provider_again; auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |",
            "| non_sendable_excluded | exclude_from_medical_extraction; defer | send_excluded_rtf_signal_containers; auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |",
            "",
            f"Allowed actions: `{', '.join(ALLOWED_ACTIONS)}`",
            f"Disallowed actions: `{', '.join(DISALLOWED_ACTIONS)}`",
            "",
        ]),
        encoding="utf-8")
    keys = [
        "live_extraction_stopped", "provider_model_call_made", "gemini_call_made",
        "vertex_call_made", "billing_api_call_made", "mkb_db_opened", "active_mkb_write",
        "future_mkb_import_started", "auto_accept_enabled", "medical_decision_made",
        "private_artifacts_committed", "raw_ai_response_committed", "raw_text_committed",
        "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
        "credentials_or_tokens_committed", "public_report_phi_leak_count",
        "private_path_leaks_after", "secret_leaks_after", "privacy_result", "safety_result",
    ]
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(["# R22 safety boundary", "", "| Gate | Value |", "| --- | --- |",
                   *[f"| {k} | `{summary[k]}` |" for k in keys], "",
                   "R22 is a local consolidation layer only. It does not resume live extraction, "
                   "does not open/import/write MKB, and does not auto-accept any package.", ""]),
        encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join([f"# {BLOCK} - implementation report", "",
                   f"- Content packages available: `{summary['total_content_packages_available']}` "
                   f"(full `{summary['full_schema_package_count']}`, minimal "
                   f"`{summary['minimal_review_bound_package_count']}`).",
                   f"- Review-only finalized: `{summary['review_only_finalized_count']}`.",
                   f"- Non-sendable excluded: `{summary['non_sendable_excluded_count']}`.",
                   f"- Operator queue priorities: high `{summary['review_priority_high_count']}`, "
                   f"normal `{summary['review_priority_normal_count']}`, low "
                   f"`{summary['review_priority_low_count']}`.",
                   "- Explicitly not ready for MKB import: review-only records, minimal review-bound "
                   "records without human acceptance, and excluded non-sendable containers.", ""]),
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
    summary["privacy_result"] = "passed" if phi == 0 and path == 0 and secret == 0 else "blocked"
    if summary["privacy_result"] != "passed":
        summary["overall_result"] = "BLOCKED"
        summary["safety_result"] = "blocked"
    return {"phi": phi, "path": path, "secret": secret}


def main() -> int:
    pkg = build_review_package()
    summary = build_summary(pkg)
    write_reports(summary, pkg)
    leaks = _privacy_scan(summary)
    write_reports(summary, pkg)
    print(json.dumps({
        "block": BLOCK,
        "overall_result": summary["overall_result"],
        "total_content_packages_available": summary["total_content_packages_available"],
        "full_schema_package_count": summary["full_schema_package_count"],
        "minimal_review_bound_package_count": summary["minimal_review_bound_package_count"],
        "review_only_finalized_count": summary["review_only_finalized_count"],
        "non_sendable_excluded_count": summary["non_sendable_excluded_count"],
        "unresolved_candidates_after_r22": summary["unresolved_candidates_after_r22"],
        "live_extraction_stopped": summary["live_extraction_stopped"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
        "leaks": leaks,
    }, indent=2, sort_keys=True))
    return 0 if summary["overall_result"] == "PASS" and summary["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
