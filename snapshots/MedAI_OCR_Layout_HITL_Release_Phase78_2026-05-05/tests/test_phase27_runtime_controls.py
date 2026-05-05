from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from execution.runtime_controls import (
    FAILURE_EXTERNAL_QUOTA,
    FAILURE_EXTERNAL_TIMEOUT,
    FAILURE_OPERATOR_REVIEW,
    RuntimeLockError,
    RuntimeRunGuard,
    classify_document_failures,
    classify_failure,
    deterministic_run_id,
    is_retry_eligible,
)
from monitoring.observability import build_phase27_metrics, write_phase27_outputs


def make_summary(tmp_path: Path) -> dict:
    return {
        "generated_at": "2026-04-25T16:00:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 3,
        "documents_processed": 2,
        "written": 1,
        "queued_for_review": 1,
        "external_quota_blocked": 1,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "runtime_controls": {
            "run_id": "phase12_validation-abc123",
            "script_name": "run_phase12_real_world_validation.py",
            "lock_path": str(tmp_path / "validation.lock"),
            "run_lock_acquired": True,
            "run_lock_released": True,
            "stale_lock_recovered": False,
            "retry_eligible_count": 1,
            "non_retryable_failure_count": 1,
            "timeout_count": 0,
            "cleanup_completed": False,
            "failure_category_counts": {
                "external_quota_block": 1,
                "none": 1,
                "operator_review_required": 1,
            },
        },
        "documents": [
            {
                "document": "written.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "error": None,
            },
            {
                "document": "review.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "error": None,
            },
            {
                "document": "quota.pdf",
                "status": "external_quota_blocked",
                "outcome": "external_quota_blocked",
                "validation_status": "skipped_external_quota",
                "error": "429 quota exceeded; retry in 12 seconds",
            },
        ],
    }


def test_second_concurrent_run_is_blocked(tmp_path: Path):
    lock_path = tmp_path / "runtime.lock"
    first = RuntimeRunGuard(
        script_name="run_phase18_full_cycle.py",
        run_id=deterministic_run_id(scope="phase18", values={"a": 1}),
        lock_path=lock_path,
    )
    second = RuntimeRunGuard(
        script_name="run_phase18_full_cycle.py",
        run_id=deterministic_run_id(scope="phase18", values={"a": 1}),
        lock_path=lock_path,
    )

    first.acquire()
    try:
        try:
            second.acquire()
        except RuntimeLockError as exc:
            assert "active runtime lock present" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected concurrent lock acquisition to fail")
    finally:
        first.release()


def test_stale_lock_can_be_recovered_deterministically(tmp_path: Path):
    lock_path = tmp_path / "runtime.lock"
    stale_dir = tmp_path / "stale_runtime"
    stale_dir.mkdir()
    (stale_dir / "temp.txt").write_text("stale", encoding="utf-8")
    payload = {
        "run_id": "old-run",
        "script_name": "run_phase18_full_cycle.py",
        "pid": 1234,
        "created_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    guard = RuntimeRunGuard(
        script_name="run_phase18_full_cycle.py",
        run_id=deterministic_run_id(scope="phase18", values={"a": 2}),
        lock_path=lock_path,
        cleanup_paths=[stale_dir],
        stale_after_seconds=60,
    )

    state = guard.acquire()
    try:
        assert state.stale_lock_recovered is True
        assert state.cleanup_completed is True
        assert not stale_dir.exists()
    finally:
        guard.release()


def test_external_quota_block_is_not_hard_failure():
    category = classify_failure(status="external_quota_blocked", validation_status="skipped_external_quota")

    assert category == FAILURE_EXTERNAL_QUOTA
    assert is_retry_eligible(failure_category=category) is True


def test_operator_review_required_is_not_hard_failure():
    category = classify_failure(status="queued_for_review", validation_status="needs_review")

    assert category == FAILURE_OPERATOR_REVIEW
    assert is_retry_eligible(failure_category=category) is False


def test_timeout_is_classified_separately():
    category = classify_failure(error=TimeoutError("connector timeout"))

    assert category == FAILURE_EXTERNAL_TIMEOUT
    assert is_retry_eligible(failure_category=category) is True


def test_cleanup_occurs_after_simulated_interrupted_run(tmp_path: Path):
    lock_path = tmp_path / "runtime.lock"
    stale_file = tmp_path / "stale.json"
    stale_file.write_text("{}", encoding="utf-8")
    lock_path.write_text(
        json.dumps({
            "run_id": "old-run",
            "script_name": "run_phase12_real_world_validation.py",
            "pid": 1234,
            "created_at": (datetime.now(UTC) - timedelta(hours=4)).isoformat(),
        }),
        encoding="utf-8",
    )
    guard = RuntimeRunGuard(
        script_name="run_phase12_real_world_validation.py",
        run_id=deterministic_run_id(scope="phase12", values={"dataset": "x"}),
        lock_path=lock_path,
        cleanup_paths=[stale_file],
        stale_after_seconds=60,
    )

    state = guard.acquire()
    try:
        assert state.cleanup_completed is True
        assert not stale_file.exists()
    finally:
        guard.release()


def test_document_failure_classification_counts_are_deterministic():
    documents = [
        {"document": "written.pdf", "status": "processed", "outcome": "written", "validation_status": "accepted", "error": None},
        {"document": "review.pdf", "status": "processed", "outcome": "queued_for_review", "validation_status": "needs_review", "error": None},
        {"document": "quota.pdf", "status": "external_quota_blocked", "outcome": "external_quota_blocked", "validation_status": "skipped_external_quota", "error": "429 quota exceeded"},
        {"document": "timeout.pdf", "status": "error", "outcome": "error", "validation_status": "error", "error": "connector timeout"},
    ]

    first = classify_document_failures(documents)
    second = classify_document_failures(documents)

    assert first == second
    assert first["failure_category_counts"] == {
        "external_quota_block": 1,
        "external_timeout": 1,
        "none": 1,
        "operator_review_required": 1,
    }
    assert first["retry_eligible_count"] == 2
    assert first["non_retryable_failure_count"] == 1
    assert first["timeout_count"] == 1


def test_phase27_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase27" / "runtime_controls.json"
    report_path = tmp_path / "reports" / "phase27" / "production_hardening_report.md"

    metrics = write_phase27_outputs(make_summary(tmp_path), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 27 Production Hardening Report" in report_path.read_text(encoding="utf-8")


def test_phase27_metrics_preserve_phase26_aggregate_behavior(tmp_path: Path):
    metrics = build_phase27_metrics(make_summary(tmp_path))

    assert metrics["written_documents"] == 1
    assert metrics["queued_for_review_documents"] == 1
    assert metrics["external_quota_blocked"] == 1
    assert metrics["hard_failures"] == 0
    assert metrics["run_lock_acquired"] is True
    assert metrics["run_lock_released"] is True
