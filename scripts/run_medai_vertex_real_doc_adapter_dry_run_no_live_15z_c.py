#!/usr/bin/env python3
"""MEDAI-VERTEX-REAL-DOC-ADAPTER-DRY-RUN-NO-LIVE-15Z-C."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload
from execution.vertex_real_doc_adapter_dry_run import (  # noqa: E402
    AdapterDryRunFixture,
    build_adapter_dry_run_fixtures,
    evaluate_adapter_readiness_with_gates,
    evaluate_all_adapter_dry_run_cases,
)
from execution.vertex_real_doc_pii_stripping_proof import (  # noqa: E402
    build_isolated_pii_vault_record,
    redact_pii_like_values,
)
from execution.vertex_real_doc_readiness_gates import sanitize_readiness_report_payload  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_adapter_dry_run_no_live_15z_c"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CASES_JSON = REPORT_DIR / "adapter_dry_run_cases.json"
MATRIX_MD = REPORT_DIR / "request_shape_matrix.md"
HANDOFF_JSON = REPORT_DIR / "review_queue_handoff_records.json"
REQUEST_PREVIEW_MD = REPORT_DIR / "would_be_request_preview.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

FORBIDDEN_PUBLISHED_TOKENS = ("ya29.", "AIza", "Bearer ", "Authorization", "C:\\", "/home/")


def build_reports() -> dict[str, Any]:
    report = evaluate_all_adapter_dry_run_cases()
    summary = {
        "block": "MEDAI-VERTEX-REAL-DOC-SYNTHETIC-TO-REAL-ADAPTER-DRY-RUN-NO-LIVE-15Z-C",
        **report["summary"],
    }
    cases = report["cases"]
    handoffs = report["handoff_records"]
    matrix = _matrix_markdown(cases, summary)
    preview = _request_preview(cases)
    implementation = _implementation_markdown(summary)
    privacy = _privacy_passes(summary, cases, handoffs, matrix, preview, implementation)
    summary["privacy_result"] = "passed" if privacy else "failed"
    implementation = _implementation_markdown(summary)
    return {
        "summary": summary,
        "cases": {"cases": cases},
        "handoffs": {"review_queue_handoff_records": handoffs},
        "matrix": matrix,
        "preview": preview,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CASES_JSON.write_text(
        json.dumps(sanitize_readiness_report_payload(reports["cases"]), indent=2),
        encoding="utf-8",
    )
    HANDOFF_JSON.write_text(
        json.dumps(sanitize_readiness_report_payload(reports["handoffs"]), indent=2),
        encoding="utf-8",
    )
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    REQUEST_PREVIEW_MD.write_text(reports["preview"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def build_case_results() -> list[Any]:
    return [evaluate_adapter_readiness_with_gates(f) for f in build_adapter_dry_run_fixtures()]


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str) for p in payloads)
    if any(token in published for token in FORBIDDEN_PUBLISHED_TOKENS):
        return False
    if _token_map_signature_present(published):
        return False
    for fixture in build_adapter_dry_run_fixtures():
        red = redact_pii_like_values(fixture.raw_text)
        raw_values = list(red._detected_values)
        vault = build_isolated_pii_vault_record(red)
        if vault._serialized_map and vault._serialized_map in published:
            return False
        if any(raw and raw in published for raw in raw_values):
            return False
    return bool(check_public_report_payload(payloads).passed)


def _token_map_signature_present(text: str) -> bool:
    import re

    return bool(re.search(r'"\[[A-Z_]+_\d+\]"\s*:\s*"[^"]+"', text))


def _matrix_markdown(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = [
        "| {case_id} | {status} | {shape} | {keys} | {handoff} | {ready} | {blocked} |".format(
            case_id=c["case_id"],
            status=c["adapter_status"],
            shape=c["request_shape_valid"],
            keys=", ".join(c["request_top_level_keys"]),
            handoff=c["review_queue_handoff_record_created"],
            ready=c["readiness_status"],
            blocked=c["blocked"],
        )
        for c in cases
    ]
    return "\n".join(
        [
            "# 15Z-C request shape matrix",
            "",
            "| Case | Adapter status | Shape valid | Request keys | Handoff | Readiness | Blocked |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| adapter_cases_total | `{summary['adapter_cases_total']}` |",
            f"| request_shape_valid_count | `{summary['request_shape_valid_count']}` |",
            f"| request_top_level_keys_exact_count | `{summary['request_top_level_keys_exact_count']}` |",
            f"| forbidden_metadata_rejected_count | `{summary['forbidden_metadata_rejected_count']}` |",
            f"| generation_config_valid_count | `{summary['generation_config_valid_count']}` |",
            f"| review_queue_handoff_records_created_count | `{summary['review_queue_handoff_records_created_count']}` |",
            f"| blocked_case_count | `{summary['blocked_case_count']}` |",
            f"| future_authorization_only_count | `{summary['future_authorization_only_count']}` |",
            f"| real_doc_live_allowed_count | `{summary['real_doc_live_allowed_count']}` |",
            "",
        ]
    )


def _request_preview(cases: list[dict[str, Any]]) -> str:
    accepted = [
        c for c in cases
        if c["adapter_status"] in {
            "DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED",
            "READY_FOR_FUTURE_AUTHORIZATION_ONLY",
        }
    ]
    lines = [
        "# 15Z-C would-be Vertex request preview",
        "",
        "Preview is fingerprint-only. The would-be request body is validated offline",
        "with exactly `contents` and `generationConfig` top-level keys. No source",
        "text, token map, vault contents, credentials, PDF, image, OCR payload, or",
        "provider call is included in this report.",
        "",
        "| Case | Request keys | Request fingerprint | Payload fingerprint | Vault fingerprint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in accepted:
        lines.append(
            f"| {c['case_id']} | {', '.join(c['request_top_level_keys'])} | "
            f"`{c['would_be_request_fingerprint']}` | `{c['outbound_payload_fingerprint']}` | "
            f"`{c['vault_record_fingerprint']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _implementation_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "adapter_dry_run_created",
        "adapter_cases_total",
        "adapter_cases_passed",
        "request_shape_valid_count",
        "request_top_level_keys_exact_count",
        "forbidden_metadata_rejected_count",
        "generation_config_valid_count",
        "would_be_request_fingerprints_created_count",
        "outbound_payload_fingerprints_created_count",
        "vault_fingerprints_referenced_count",
        "review_queue_handoff_records_created_count",
        "review_queue_handoff_report_only_count",
        "readiness_cases_fed_count",
        "no_live_replay_allowed_count",
        "blocked_case_count",
        "future_authorization_only_count",
        "real_doc_live_allowed_count",
        "raw_pii_in_request_count",
        "raw_pii_in_report_count",
        "token_map_in_request_count",
        "token_map_in_report_count",
        "live_call_made",
        "external_api_used",
        "active_written_count",
        "active_mkb_record_created_count",
        "auto_accept_true_count",
        "privacy_result",
        "billing_check_pending",
    ]
    lines = ["# MEDAI-VERTEX-REAL-DOC-ADAPTER-DRY-RUN-NO-LIVE-15Z-C", ""]
    lines.extend(f"- {k}: `{summary[k]}`" for k in keys)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Builds the would-be Vertex request shape offline from 15Z-B tokenized payloads.",
            "- Keeps provider metadata local to dry-run proof and review handoff records.",
            "- Writes handoff records only to this report folder; no active MKB or production queue write occurs.",
            "- Keeps `live_call_allowed=false`, `active_write_allowed=false`, `auto_accept_allowed=false`, and `review_required=true`.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["adapter_dry_run_created"] is True,
            summary["adapter_cases_total"] == 14,
            summary["adapter_cases_passed"] == 14,
            summary["request_shape_valid_count"] == 5,
            summary["request_top_level_keys_exact_count"] == 5,
            summary["forbidden_metadata_rejected_count"] == 1,
            summary["generation_config_valid_count"] == 5,
            summary["would_be_request_fingerprints_created_count"] == 14,
            summary["outbound_payload_fingerprints_created_count"] == 14,
            summary["vault_fingerprints_referenced_count"] == 14,
            summary["review_queue_handoff_records_created_count"] == 5,
            summary["review_queue_handoff_report_only_count"] == 5,
            summary["readiness_cases_fed_count"] == 14,
            summary["no_live_replay_allowed_count"] == 4,
            summary["blocked_case_count"] == 9,
            summary["future_authorization_only_count"] == 1,
            summary["real_doc_live_allowed_count"] == 0,
            summary["raw_pii_in_request_count"] == 0,
            summary["raw_pii_in_report_count"] == 0,
            summary["token_map_in_request_count"] == 0,
            summary["token_map_in_report_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["privacy_result"] == "passed",
        ]
    )


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    ready = _ready(summary)
    print(
        "medai_vertex_real_doc_adapter_dry_run_no_live_15z_c_ready"
        if ready
        else "medai_vertex_real_doc_adapter_dry_run_no_live_15z_c_not_ready"
    )
    print(
        json.dumps(
            {
                "adapter_cases_total": summary["adapter_cases_total"],
                "adapter_cases_passed": summary["adapter_cases_passed"],
                "request_shape_valid_count": summary["request_shape_valid_count"],
                "request_top_level_keys_exact_count": summary["request_top_level_keys_exact_count"],
                "review_queue_handoff_records_created_count": summary["review_queue_handoff_records_created_count"],
                "blocked_case_count": summary["blocked_case_count"],
                "future_authorization_only_count": summary["future_authorization_only_count"],
                "real_doc_live_allowed_count": summary["real_doc_live_allowed_count"],
                "raw_pii_in_request_count": summary["raw_pii_in_request_count"],
                "raw_pii_in_report_count": summary["raw_pii_in_report_count"],
                "token_map_in_request_count": summary["token_map_in_request_count"],
                "token_map_in_report_count": summary["token_map_in_report_count"],
                "live_call_made": summary["live_call_made"],
                "external_api_used": summary["external_api_used"],
                "privacy_result": summary["privacy_result"],
                "billing_check_pending": summary["billing_check_pending"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
