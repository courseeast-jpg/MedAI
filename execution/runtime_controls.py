from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


FAILURE_NONE = "none"
FAILURE_ENVIRONMENT = "environment_failure"
FAILURE_INPUT = "input_failure"
FAILURE_LOGIC = "logic_failure"
FAILURE_EXTERNAL_QUOTA = "external_quota_block"
FAILURE_EXTERNAL_TIMEOUT = "external_timeout"
FAILURE_OPERATOR_REVIEW = "operator_review_required"

RETRY_ELIGIBLE_FAILURES = {FAILURE_EXTERNAL_QUOTA, FAILURE_EXTERNAL_TIMEOUT}


class RuntimeLockError(RuntimeError):
    """Raised when a runtime lock cannot be acquired safely."""


@dataclass
class RuntimeControlState:
    run_id: str
    script_name: str
    lock_path: str
    run_lock_acquired: bool = False
    run_lock_released: bool = False
    stale_lock_recovered: bool = False
    retry_eligible_count: int = 0
    non_retryable_failure_count: int = 0
    timeout_count: int = 0
    cleanup_completed: bool = False
    failure_category_counts: dict[str, int] = field(default_factory=dict)
    lock_created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_run_id(*, scope: str, values: dict[str, Any]) -> str:
    normalized_values = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(f"{scope}|{normalized_values}".encode("utf-8")).hexdigest()
    return f"{scope}-{digest[:16]}"


def classify_failure(
    *,
    error: Exception | str | None = None,
    status: str | None = None,
    validation_status: str | None = None,
) -> str:
    if status in {"queued_for_review", "written_with_review"} or validation_status in {"needs_review", "rejected"}:
        return FAILURE_OPERATOR_REVIEW
    if status == "external_quota_blocked":
        return FAILURE_EXTERNAL_QUOTA
    if error is None:
        return FAILURE_NONE

    message = str(error).lower()
    if _is_timeout_text(message) or isinstance(error, TimeoutError):
        return FAILURE_EXTERNAL_TIMEOUT
    if _is_external_quota_text(message):
        return FAILURE_EXTERNAL_QUOTA
    if isinstance(error, (FileNotFoundError, ValueError)):
        return FAILURE_INPUT
    if isinstance(error, (ImportError, ModuleNotFoundError, OSError)):
        return FAILURE_ENVIRONMENT
    return FAILURE_LOGIC


def is_retry_eligible(*, failure_category: str) -> bool:
    return failure_category in RETRY_ELIGIBLE_FAILURES


def classify_document_failures(documents: list[dict[str, Any]]) -> dict[str, Any]:
    failure_category_counts: dict[str, int] = {}
    retry_eligible_count = 0
    non_retryable_failure_count = 0
    timeout_count = 0

    for item in documents:
        category = classify_failure(
            error=item.get("error"),
            status=str(item.get("status") or item.get("outcome") or ""),
            validation_status=str(item.get("validation_status") or ""),
        )
        failure_category_counts[category] = failure_category_counts.get(category, 0) + 1
        if category == FAILURE_EXTERNAL_TIMEOUT:
            timeout_count += 1
        if is_retry_eligible(failure_category=category):
            retry_eligible_count += 1
        elif category != FAILURE_NONE:
            non_retryable_failure_count += 1

    return {
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "retry_eligible_count": retry_eligible_count,
        "non_retryable_failure_count": non_retryable_failure_count,
        "timeout_count": timeout_count,
    }


class RuntimeRunGuard:
    def __init__(
        self,
        *,
        script_name: str,
        run_id: str,
        lock_path: Path | str,
        cleanup_paths: list[Path | str] | None = None,
        stale_after_seconds: int = 3600,
    ) -> None:
        self.script_name = script_name
        self.run_id = run_id
        self.lock_path = Path(lock_path)
        self.cleanup_paths = [Path(path) for path in (cleanup_paths or [])]
        self.stale_after_seconds = stale_after_seconds
        self.state = RuntimeControlState(
            run_id=run_id,
            script_name=script_name,
            lock_path=str(self.lock_path),
        )

    def acquire(self) -> RuntimeControlState:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            if self._is_stale_lock():
                self.state.stale_lock_recovered = True
                self._safe_cleanup()
                self.lock_path.unlink(missing_ok=True)
            else:
                raise RuntimeLockError(f"active runtime lock present: {self.lock_path}")

        payload = {
            "run_id": self.run_id,
            "script_name": self.script_name,
            "pid": os.getpid(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.state.run_lock_acquired = True
        self.state.lock_created_at = str(payload["created_at"])
        return self.state

    def release(self) -> RuntimeControlState:
        self.lock_path.unlink(missing_ok=True)
        self.state.run_lock_released = True
        return self.state

    def finalize(self, *, documents: list[dict[str, Any]] | None = None) -> RuntimeControlState:
        metrics = classify_document_failures(documents or [])
        self.state.retry_eligible_count = int(metrics["retry_eligible_count"])
        self.state.non_retryable_failure_count = int(metrics["non_retryable_failure_count"])
        self.state.timeout_count = int(metrics["timeout_count"])
        self.state.failure_category_counts = dict(metrics["failure_category_counts"])
        return self.state

    def _is_stale_lock(self) -> bool:
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except Exception:
            return True

        created_at = payload.get("created_at")
        if not created_at:
            return True
        try:
            created_time = datetime.fromisoformat(str(created_at))
        except ValueError:
            return True
        return created_time <= datetime.now(UTC) - timedelta(seconds=self.stale_after_seconds)

    def _safe_cleanup(self) -> None:
        for path in self.cleanup_paths:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        self.state.cleanup_completed = True


def _is_external_quota_text(message: str) -> bool:
    quota_markers = (
        "quota exceeded",
        "current quota",
        "rate limit",
        "rate-limit",
        "generativelanguage.googleapis.com/generate_content",
        "429",
    )
    return any(marker in message for marker in quota_markers)


def _is_timeout_text(message: str) -> bool:
    timeout_markers = ("timeout", "timed out", "deadline exceeded")
    return any(marker in message for marker in timeout_markers)
