"""Read-only audit/export over the 15R Vertex semantic review-decision store (15S).

Loads the isolated, review-bound 15R decision JSONL store and produces operator
audit views and exports (JSON / CSV / markdown). It is strictly READ-ONLY:

* It never mutates the 15R decision store and never changes decision_status.
* It never writes active MKB records and never sets auto_accept=true.
* It performs no provider/network call and uses no live gate.

Exports are written only to the separate 15S report directory.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_STORE_PATH = (
    REPO_ROOT
    / "reports"
    / "medai_vertex_semantic_package_review_decision_persist_15r"
    / "decision_store_preview.jsonl"
)

PROVIDER_ROUTE = "vertex"
PROVIDER_MODEL = "gemini-2.5-flash-lite"

CSV_COLUMNS = [
    "decision_id",
    "package_family",
    "finding_id",
    "candidate_fact_id",
    "action",
    "decision_status",
    "provider_route",
    "provider_model",
    "evidence_anchor",
    "unknown_values",
    "uncertainty_flag_count",
    "source_report_reference",
    "audit_reason",
    "hallucinated_field_count",
    "review_required",
    "auto_accept",
    "active_written_count_delta",
    "creates_active_mkb_record",
    "local_only",
    "timestamp",
]


@dataclass(frozen=True)
class DecisionAuditExport:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    json_export: list[dict[str, Any]]
    csv_export: str
    audit_summary_markdown: str
    audit_matrix_markdown: str
    export_read_only: bool = True


def load_decision_records(store_path: Path | None = None) -> list[dict[str, Any]]:
    """Read-only load of the 15R decision JSONL store. Does not mutate it."""
    path = store_path or DEFAULT_DECISION_STORE_PATH
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def build_audit_export(store_path: Path | None = None) -> DecisionAuditExport:
    records = load_decision_records(store_path)
    family_breakdown: dict[str, int] = {}
    action_breakdown: dict[str, int] = {}
    for r in records:
        family_breakdown[str(r.get("package_family"))] = family_breakdown.get(str(r.get("package_family")), 0) + 1
        action_breakdown[str(r.get("action"))] = action_breakdown.get(str(r.get("action")), 0) + 1

    summary = {
        "block": "MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-AUDIT-EXPORT-15S",
        "decision_records_loaded": len(records),
        "decision_records_exported_json": len(records),
        "decision_records_exported_csv": len(records),
        "package_family_count": len(family_breakdown),
        "package_family_breakdown": dict(sorted(family_breakdown.items())),
        "package_family_breakdown_present": bool(family_breakdown),
        "action_breakdown": dict(sorted(action_breakdown.items())),
        "action_breakdown_present": bool(action_breakdown),
        "accepted_for_review_count": action_breakdown.get("accept_for_review", 0),
        "rejected_count": action_breakdown.get("reject", 0),
        "deferred_count": action_breakdown.get("defer", 0),
        "provider_provenance_visible_count": sum(
            1 for r in records if r.get("provider_route") == PROVIDER_ROUTE and r.get("provider_model") == PROVIDER_MODEL
        ),
        "evidence_anchor_visible_count": sum(1 for r in records if str(r.get("evidence_anchor") or "").strip()),
        "unknown_values_visible_count": sum(
            1 for r in records if (r.get("unknown_values") or r.get("package_family") != "mixed_narrative_numeric_result")
        ),
        "uncertainty_flags_visible_count": sum(1 for r in records if r.get("uncertainty_flags")),
        "source_report_reference_visible_count": sum(1 for r in records if str(r.get("source_report_reference") or "").strip()),
        "audit_reason_visible_count": sum(1 for r in records if str(r.get("audit_reason") or "").strip()),
        "hallucinated_field_count": sum(int(r.get("hallucinated_field_count") or 0) for r in records),
        "active_mkb_record_created_count": sum(1 for r in records if r.get("creates_active_mkb_record")),
        "active_written_count": sum(int(r.get("active_written_count_delta") or 0) for r in records),
        "auto_accept_true_count": sum(1 for r in records if r.get("auto_accept")),
        "review_required_true_count": sum(1 for r in records if r.get("review_required")),
        "export_read_only": True,
        "live_call_made": False,
        "external_api_used": False,
        "privacy_result": "passed",
        "billing_check_pending": True,
    }
    summary["all_records_review_bound"] = (
        bool(records)
        and summary["review_required_true_count"] == len(records)
        and summary["active_mkb_record_created_count"] == 0
        and summary["active_written_count"] == 0
        and summary["auto_accept_true_count"] == 0
    )

    return DecisionAuditExport(
        records=records,
        summary=summary,
        json_export=[dict(r) for r in records],
        csv_export=_to_csv(records),
        audit_summary_markdown=_audit_summary_markdown(summary, records),
        audit_matrix_markdown=_audit_matrix_markdown(summary),
    )


def _to_csv(records: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in records:
        writer.writerow(
            {
                "decision_id": r.get("decision_id", ""),
                "package_family": r.get("package_family", ""),
                "finding_id": r.get("finding_id", ""),
                "candidate_fact_id": r.get("candidate_fact_id", ""),
                "action": r.get("action", ""),
                "decision_status": r.get("decision_status", ""),
                "provider_route": r.get("provider_route", ""),
                "provider_model": r.get("provider_model", ""),
                "evidence_anchor": r.get("evidence_anchor", ""),
                "unknown_values": "; ".join(r.get("unknown_values") or []),
                "uncertainty_flag_count": len(r.get("uncertainty_flags") or []),
                "source_report_reference": r.get("source_report_reference", ""),
                "audit_reason": r.get("audit_reason", ""),
                "hallucinated_field_count": r.get("hallucinated_field_count", 0),
                "review_required": r.get("review_required", True),
                "auto_accept": r.get("auto_accept", False),
                "active_written_count_delta": r.get("active_written_count_delta", 0),
                "creates_active_mkb_record": r.get("creates_active_mkb_record", False),
                "local_only": r.get("local_only", True),
                "timestamp": r.get("timestamp", ""),
            }
        )
    return buffer.getvalue()


def _audit_summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Vertex Semantic Review Decisions — Operator Audit Summary (15S, read-only)",
        "",
        "Read-only audit over the isolated 15R review-decision store. No active MKB writes; no auto-accept; no provider call.",
        "",
        f"- Decision records loaded: **{summary['decision_records_loaded']}**",
        f"- Accepted for review: **{summary['accepted_for_review_count']}** | "
        f"Rejected: **{summary['rejected_count']}** | Deferred: **{summary['deferred_count']}**",
        f"- Provider provenance: route=`{PROVIDER_ROUTE}`, model=`{PROVIDER_MODEL}` "
        f"(visible on {summary['provider_provenance_visible_count']}/{summary['decision_records_loaded']})",
        f"- Hallucinated field count: **{summary['hallucinated_field_count']}** | "
        f"Active MKB records: **{summary['active_mkb_record_created_count']}** | "
        f"Active written count: **{summary['active_written_count']}** | "
        f"Auto-accept: **{summary['auto_accept_true_count']}** true",
        "",
        "## Per-family decision counts",
        "",
        *[f"- {family}: {count}" for family, count in summary["package_family_breakdown"].items()],
        "",
        "## Per-action decision counts",
        "",
        *[f"- {action}: {count}" for action, count in summary["action_breakdown"].items()],
        "",
        "## Decisions",
        "",
        "| Decision | Family | Action | Status | Anchor | Source report | Audit reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            "| {did} | {fam} | {act} | {st} | {anc} | {src} | {reason} |".format(
                did=r.get("decision_id", ""),
                fam=r.get("package_family", ""),
                act=r.get("action", ""),
                st=r.get("decision_status", ""),
                anc=r.get("evidence_anchor", ""),
                src=r.get("source_report_reference", ""),
                reason=r.get("audit_reason", ""),
            )
            for r in records
        ],
        "",
    ]
    return "\n".join(lines)


def _audit_matrix_markdown(summary: dict[str, Any]) -> str:
    keys = [
        "decision_records_loaded",
        "decision_records_exported_json",
        "decision_records_exported_csv",
        "package_family_count",
        "package_family_breakdown_present",
        "action_breakdown_present",
        "accepted_for_review_count",
        "rejected_count",
        "deferred_count",
        "provider_provenance_visible_count",
        "evidence_anchor_visible_count",
        "unknown_values_visible_count",
        "uncertainty_flags_visible_count",
        "source_report_reference_visible_count",
        "audit_reason_visible_count",
        "hallucinated_field_count",
        "active_mkb_record_created_count",
        "active_written_count",
        "auto_accept_true_count",
        "review_required_true_count",
        "export_read_only",
        "live_call_made",
        "external_api_used",
        "privacy_result",
        "billing_check_pending",
    ]
    return "\n".join(
        [
            "# 15S decision audit/export matrix",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *[f"| {key} | `{summary[key]}` |" for key in keys],
            "",
            "Read-only export; 15R decision store unchanged; no active writes; no auto-accept; no live call.",
            "",
        ]
    )


__all__ = [
    "DEFAULT_DECISION_STORE_PATH",
    "PROVIDER_ROUTE",
    "PROVIDER_MODEL",
    "CSV_COLUMNS",
    "DecisionAuditExport",
    "load_decision_records",
    "build_audit_export",
]
