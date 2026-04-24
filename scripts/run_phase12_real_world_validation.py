from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "test_data" / "final_batch_50"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "phase12_real_world_validation"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.audit import StageAuditLogger
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from mkb.sqlite_store import SQLiteStore


def is_external_quota_error(error: Exception | str) -> bool:
    message = str(error).lower()
    quota_markers = (
        "quota exceeded",
        "current quota",
        "rate limit",
        "rate-limit",
        "retry in",
        "generativelanguage.googleapis.com/generate_content",
        "429",
    )
    return any(marker in message for marker in quota_markers)


def list_pdf_documents(dataset_dir: Path, limit: int) -> list[Path]:
    pdfs = sorted(path for path in dataset_dir.glob("*.pdf") if path.is_file())
    if limit > 0:
        return pdfs[:limit]
    return pdfs


def build_validation_pipeline(runtime_dir: Path) -> tuple[ExecutionPipeline, SQLiteStore, dict[str, Any]]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    db_path = runtime_dir / "mkb.db"
    audit_path = runtime_dir / "execution_audit.jsonl"
    stage_audit_path = runtime_dir / "pipeline_stages.jsonl"
    review_queue_path = runtime_dir / "review_queue.jsonl"

    sql_store = SQLiteStore(db_path)
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        audit_logger=AuditLogger(path=audit_path),
        stage_audit_logger=StageAuditLogger(path=stage_audit_path),
        review_queue_path=review_queue_path,
    )
    component_state = {
        "sql_store": True,
        "vector_store": False,
        "quality_gate": False,
        "medication_gate": False,
        "governance_active": True,
        "review_queue_path": str(review_queue_path),
        "audit_log_path": str(audit_path),
        "stage_audit_log_path": str(stage_audit_path),
        "runtime_db_path": str(db_path),
    }
    return pipeline, sql_store, component_state


def summarize_document(
    pdf_path: Path,
    result,
    error: Exception | None = None,
    *,
    quota_safe: bool = False,
) -> dict[str, Any]:
    if error is not None:
        if quota_safe and is_external_quota_error(error):
            return {
                "document": pdf_path.name,
                "status": "external_quota_blocked",
                "error": str(error),
                "outcome": "external_quota_blocked",
                "validation_status": "skipped_external_quota",
                "extractor": None,
                "extractor_actual": None,
                "confidence": 0.0,
                "entity_count": 0,
                "written_count": 0,
                "queued_count": 0,
                "blocked_count": 0,
                "review_reasons": [],
                "notes": [],
            }
        return {
            "document": pdf_path.name,
            "status": "error",
            "error": str(error),
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
        }

    review_reasons: list[str] = []
    for item in result.validation_errors:
        code = item.get("code")
        if code:
            review_reasons.append(str(code))
    for record in result.queued_records + result.blocked_records:
        if record.status:
            review_reasons.append(str(record.status))
        if record.ddi_status:
            review_reasons.append(str(record.ddi_status))

    return {
        "document": pdf_path.name,
        "status": "processed",
        "error": None,
        "outcome": result.outcome,
        "validation_status": result.validation_status,
        "extractor": result.audit.get("extractor", result.extractor_result.get("extractor")),
        "extractor_actual": result.audit.get("extractor_actual", result.extractor_result.get("actual_extractor")),
        "confidence": float(result.audit.get("confidence", result.extractor_result.get("confidence", 0.0))),
        "entity_count": int(result.audit.get("entity_count", len(result.extractor_result.get("entities", [])))),
        "written_count": result.written_count,
        "queued_count": result.queued_count,
        "blocked_count": len(result.blocked_records),
        "review_reasons": sorted(set(review_reasons)),
        "notes": list(result.notes),
    }


def build_phase12_summary(
    *,
    dataset_dir: Path,
    requested_limit: int,
    documents: list[dict[str, Any]],
    runtime_counts: dict[str, int],
    component_state: dict[str, Any],
) -> dict[str, Any]:
    processed = [item for item in documents if item["status"] == "processed"]
    external_quota_blocked = [item for item in documents if item["status"] == "external_quota_blocked"]
    errors = [item for item in documents if item["status"] == "error"]
    coverage_count = len(processed) + len(external_quota_blocked)

    outcome_counts = Counter(item["outcome"] for item in processed)
    validation_counts = Counter(item["validation_status"] for item in processed)
    extractor_counts = Counter(
        item["extractor_actual"] or item["extractor"] or "unknown"
        for item in processed
    )
    avg_confidence = round(
        sum(float(item["confidence"]) for item in processed) / len(processed),
        3,
    ) if processed else 0.0

    review_reasons = Counter()
    for item in processed:
        for reason in item["review_reasons"]:
            review_reasons[reason] += 1

    recommendations: list[str] = []
    if coverage_count < 10:
        recommendations.append("Expand the Phase 12 run to at least 10 documents to match the baseline target window.")
    if errors:
        recommendations.append("Investigate per-document processing errors before treating this run as representative.")
    if external_quota_blocked:
        recommendations.append("Some documents were skipped due to external API quota exhaustion; rerun later to complete the sample without changing pipeline behavior.")
    if outcome_counts.get("queued_for_review", 0) > 0:
        recommendations.append("Inspect review-queued documents and classify whether queues are expected governance behavior or extraction misses.")
    if outcome_counts.get("blocked_ddi", 0) > 0:
        recommendations.append("Review medication safety blocks manually and confirm whether DDI blocks are clinically appropriate.")
    if avg_confidence < 0.5:
        recommendations.append("Average extraction confidence is low; review OCR quality and routed extractor behavior on the sampled PDFs.")
    if not recommendations:
        recommendations.append("Run a second Phase 12 batch from a different document mix to confirm stability beyond this initial sample.")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 12 Real-World Validation",
        "dataset_dir": str(dataset_dir),
        "requested_limit": requested_limit,
        "documents_selected": len(documents),
        "documents_processed": len(processed),
        "written": outcome_counts.get("written", 0),
        "queued_for_review": outcome_counts.get("queued_for_review", 0),
        "external_quota_blocked": len(external_quota_blocked),
        "hard_failures": len(errors),
        "documents_failed": len(errors),
        "target_window_met": 10 <= coverage_count <= 20,
        "phase12_started": coverage_count > 0,
        "run_passed": coverage_count > 0 and len(errors) == 0,
        "component_state": component_state,
        "aggregate": {
            "outcomes": dict(sorted(outcome_counts.items())),
            "validation_statuses": dict(sorted(validation_counts.items())),
            "extractors": dict(sorted(extractor_counts.items())),
            "avg_confidence": avg_confidence,
            "total_entities": sum(int(item["entity_count"]) for item in processed),
            "total_written": sum(int(item["written_count"]) for item in processed),
            "total_queued": sum(int(item["queued_count"]) for item in processed),
            "total_blocked": sum(int(item["blocked_count"]) for item in processed),
            "review_reasons": dict(sorted(review_reasons.items())),
        },
        "mkb_counts": runtime_counts,
        "documents": documents,
        "recommendations": recommendations,
    }


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase12_real_world_validation_summary.json"
    md_path = output_dir / "phase12_real_world_validation_summary.md"
    jsonl_path = output_dir / "phase12_real_world_validation_documents.jsonl"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in summary["documents"]:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    md_lines = [
        "# Phase 12 Real-World Validation Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Documents processed: {summary['documents_processed']}/{summary['documents_selected']}",
        f"- Written: {summary['written']}",
        f"- Queued for review: {summary['queued_for_review']}",
        f"- External quota blocked: {summary['external_quota_blocked']}",
        f"- Hard failures: {summary['hard_failures']}",
        f"- Target window met (10-20 docs): {summary['target_window_met']}",
        f"- Run passed: {summary['run_passed']}",
        "",
        "## Aggregate",
        "",
        f"- Outcomes: {summary['aggregate']['outcomes']}",
        f"- Validation statuses: {summary['aggregate']['validation_statuses']}",
        f"- Extractors: {summary['aggregate']['extractors']}",
        f"- Average confidence: {summary['aggregate']['avg_confidence']}",
        f"- Total entities: {summary['aggregate']['total_entities']}",
        f"- Total written: {summary['aggregate']['total_written']}",
        f"- Total queued: {summary['aggregate']['total_queued']}",
        f"- Total blocked: {summary['aggregate']['total_blocked']}",
        "",
        "## Runtime MKB",
        "",
        f"- Counts: {summary['mkb_counts']}",
        "",
        "## Component State",
        "",
    ]
    for key, value in summary["component_state"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    md_lines.extend(f"- {item}" for item in summary["recommendations"])
    md_lines.extend([
        "",
        "## Documents",
        "",
    ])
    for item in summary["documents"]:
        md_lines.append(
            f"- `{item['document']}` -> status={item['status']} outcome={item['outcome']} "
            f"validation={item['validation_status']} confidence={item['confidence']}"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--specialty", default="general")
    parser.add_argument("--quota-safe", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "runtime"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    pdfs = list_pdf_documents(dataset_dir, args.limit)
    pipeline, sql_store, component_state = build_validation_pipeline(runtime_dir)

    documents: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        try:
            result = pipeline.process_pdf(
                pdf_path,
                specialty=args.specialty,
                session_id=f"phase12-{pdf_path.stem}",
            )
            documents.append(summarize_document(pdf_path, result))
        except Exception as exc:  # pragma: no cover - defensive path for live validation
            documents.append(summarize_document(pdf_path, None, error=exc, quota_safe=args.quota_safe))

    summary = build_phase12_summary(
        dataset_dir=dataset_dir,
        requested_limit=args.limit,
        documents=documents,
        runtime_counts=sql_store.count_records(),
        component_state=component_state,
    )
    write_outputs(output_dir, summary)

    print(json.dumps(summary, indent=2))
    return 0 if summary["run_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
