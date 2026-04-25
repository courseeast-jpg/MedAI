from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewQueueWriter:
    """Deterministic JSONL writer for operator-review records."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "timestamp": record.get("timestamp") or _utc_now(),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def append_validation_review(
        self,
        *,
        run_id: str,
        document_id: str,
        source_filename: str,
        reason: str,
        reasons: list[str],
        confidence: float,
        extractor_route: str | None,
        extractor_actual: str | None,
        recommended_action: str,
        raw_evidence_path: str | None,
        requested_extractor_route: str | None,
        validation_status: str,
        validation_errors: list[dict[str, Any]],
        entity_count: int,
        notes: list[str],
        raw_confidence: float | None = None,
        calibrated_confidence: float | None = None,
        confidence_band: str | None = None,
        calibration_reason: str | None = None,
        route_mismatch_flag: bool | None = None,
        review_recommendation: str | None = None,
    ) -> dict[str, Any]:
        return self.append({
            "queue_category": "validation_review",
            "run_id": run_id,
            "document_id": document_id,
            "source_filename": source_filename,
            "reason": reason,
            "reasons": reasons,
            "confidence": float(confidence),
            "extractor_route": extractor_route,
            "extractor_actual": extractor_actual,
            "recommended_action": recommended_action,
            "raw_evidence_path": raw_evidence_path,
            "requested_extractor_route": requested_extractor_route,
            "validation_status": validation_status,
            "validation_errors": validation_errors,
            "entity_count": int(entity_count),
            "notes": notes,
            "raw_confidence": float(confidence if raw_confidence is None else raw_confidence),
            "calibrated_confidence": (
                float(confidence if calibrated_confidence is None else calibrated_confidence)
            ),
            "confidence_band": confidence_band,
            "calibration_reason": calibration_reason,
            "route_mismatch_flag": bool(route_mismatch_flag) if route_mismatch_flag is not None else False,
            "review_recommendation": review_recommendation,
        })

    def append_external_quota_block(
        self,
        *,
        run_id: str,
        document_id: str,
        source_filename: str,
        reason: str,
        recommended_action: str,
        raw_evidence_path: str | None,
        error: str,
        retry_visibility: dict[str, Any],
        raw_confidence: float | None = None,
        calibrated_confidence: float | None = None,
        confidence_band: str | None = None,
        calibration_reason: str | None = None,
        route_mismatch_flag: bool | None = None,
        review_recommendation: str | None = None,
    ) -> dict[str, Any]:
        return self.append({
            "queue_category": "external_quota_block",
            "run_id": run_id,
            "document_id": document_id,
            "source_filename": source_filename,
            "reason": reason,
            "confidence": 0.0,
            "extractor_route": None,
            "extractor_actual": None,
            "recommended_action": recommended_action,
            "raw_evidence_path": raw_evidence_path,
            "validation_status": "skipped_external_quota",
            "error": error,
            "retry_visibility": retry_visibility,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "confidence_band": confidence_band,
            "calibration_reason": calibration_reason,
            "route_mismatch_flag": bool(route_mismatch_flag) if route_mismatch_flag is not None else False,
            "review_recommendation": review_recommendation,
        })

    def append_resolution_review(
        self,
        *,
        run_id: str,
        document_id: str,
        source_filename: str,
        reason: str,
        confidence: float,
        extractor_route: str | None,
        extractor_actual: str | None,
        recommended_action: str,
        raw_evidence_path: str | None,
        requested_extractor_route: str | None,
        validation_status: str,
        resolution_action: str,
        resolution_confidence: float,
        record_id: str,
        fact_type: str,
        content: str,
        raw_confidence: float | None = None,
        calibrated_confidence: float | None = None,
        confidence_band: str | None = None,
        calibration_reason: str | None = None,
        route_mismatch_flag: bool | None = None,
        review_recommendation: str | None = None,
    ) -> dict[str, Any]:
        return self.append({
            "queue_category": "truth_resolution_review",
            "run_id": run_id,
            "document_id": document_id,
            "source_filename": source_filename,
            "reason": reason,
            "confidence": float(confidence),
            "extractor_route": extractor_route,
            "extractor_actual": extractor_actual,
            "recommended_action": recommended_action,
            "raw_evidence_path": raw_evidence_path,
            "requested_extractor_route": requested_extractor_route,
            "validation_status": validation_status,
            "resolution_action": resolution_action,
            "resolution_confidence": float(resolution_confidence),
            "record_id": record_id,
            "fact_type": fact_type,
            "content": content,
            "raw_confidence": float(confidence if raw_confidence is None else raw_confidence),
            "calibrated_confidence": (
                float(confidence if calibrated_confidence is None else calibrated_confidence)
            ),
            "confidence_band": confidence_band,
            "calibration_reason": calibration_reason,
            "route_mismatch_flag": bool(route_mismatch_flag) if route_mismatch_flag is not None else False,
            "review_recommendation": review_recommendation,
        })

    def append_medication_review(
        self,
        *,
        run_id: str,
        document_id: str,
        source_filename: str,
        reason: str,
        confidence: float,
        extractor_route: str | None,
        extractor_actual: str | None,
        recommended_action: str,
        raw_evidence_path: str | None,
        requested_extractor_route: str | None,
        validation_status: str,
        record_id: str,
        fact_type: str,
        content: str,
        ddi_status: str | None,
        ddi_findings: list[dict[str, Any]],
        safety_action: str | None,
        requires_review: bool,
        raw_confidence: float | None = None,
        calibrated_confidence: float | None = None,
        confidence_band: str | None = None,
        calibration_reason: str | None = None,
        route_mismatch_flag: bool | None = None,
        review_recommendation: str | None = None,
    ) -> dict[str, Any]:
        return self.append({
            "queue_category": "medication_review",
            "run_id": run_id,
            "document_id": document_id,
            "source_filename": source_filename,
            "reason": reason,
            "confidence": float(confidence),
            "extractor_route": extractor_route,
            "extractor_actual": extractor_actual,
            "recommended_action": recommended_action,
            "raw_evidence_path": raw_evidence_path,
            "requested_extractor_route": requested_extractor_route,
            "validation_status": validation_status,
            "record_id": record_id,
            "fact_type": fact_type,
            "content": content,
            "ddi_status": ddi_status,
            "ddi_findings": ddi_findings,
            "safety_action": safety_action,
            "requires_review": requires_review,
            "raw_confidence": float(confidence if raw_confidence is None else raw_confidence),
            "calibrated_confidence": (
                float(confidence if calibrated_confidence is None else calibrated_confidence)
            ),
            "confidence_band": confidence_band,
            "calibration_reason": calibration_reason,
            "route_mismatch_flag": bool(route_mismatch_flag) if route_mismatch_flag is not None else False,
            "review_recommendation": review_recommendation,
        })


def read_review_queue(path: Path | str) -> list[dict[str, Any]]:
    queue_path = Path(path)
    if not queue_path.exists():
        return []
    return [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
