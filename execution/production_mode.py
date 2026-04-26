from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODE_OFF = "OFF"
MODE_DRY_RUN = "DRY_RUN"
MODE_CONTROLLED = "CONTROLLED"
MODE_LIVE = "LIVE"
VALID_MODES = {MODE_OFF, MODE_DRY_RUN, MODE_CONTROLLED, MODE_LIVE}


@dataclass(frozen=True)
class ProductionModeConfig:
    mode: str = MODE_OFF
    max_documents_per_run: int = 0
    max_concurrent_runs: int = 1
    audit_required: bool = False
    require_snapshot_before_run: bool = False
    run_approval: bool = False
    review_queue_acknowledged: bool = False
    required_snapshot_dir: str | None = None
    required_snapshot_zip: str | None = None

    def normalized_mode(self) -> str:
        value = str(self.mode or MODE_OFF).upper()
        if value not in VALID_MODES:
            raise ValueError(f"unsupported production mode: {self.mode}")
        return value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.normalized_mode()
        return payload


@dataclass
class ProductionModeState:
    production_mode: str
    max_documents_per_run: int
    max_concurrent_runs: int
    audit_required: bool
    require_snapshot_before_run: bool
    run_approval: bool
    review_queue_acknowledged: bool
    required_snapshot_dir: str | None
    required_snapshot_zip: str | None
    production_gate_passed: bool
    production_gate_failed_reason: str | None
    dry_run_executed: bool
    controlled_run_limit_applied: bool
    run_blocked_by_gate: bool
    previous_run_completed_cleanly: bool
    deterministic_outputs_verified: bool
    unresolved_runtime_lock: bool
    snapshot_verified: bool
    audit_report_available: bool
    review_queue_items: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: Path | str | None) -> dict[str, Any]:
    if path in {None, ""}:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def previous_run_completed_cleanly(summary_path: Path | str | None) -> bool:
    summary = read_json(summary_path)
    if not summary:
        return False
    return bool(summary.get("success")) and not summary.get("failed_step")


def deterministic_outputs_verified(report_path: Path | str | None) -> bool:
    if report_path in {None, ""}:
        return False
    path = Path(report_path)
    if not path.exists():
        return False
    return "- Overall status: `STABLE`" in path.read_text(encoding="utf-8")


def snapshot_verified(*, snapshot_dir: Path | str | None, snapshot_zip: Path | str | None) -> bool:
    if snapshot_dir in {None, ""} or snapshot_zip in {None, ""}:
        return False
    return Path(snapshot_dir).exists() and Path(snapshot_zip).exists()


def current_review_queue_items(phase12_summary_path: Path | str | None) -> int:
    summary = read_json(phase12_summary_path)
    review_queue = summary.get("review_queue", {})
    return int(review_queue.get("items", 0))


def unresolved_lock_exists(lock_path: Path | str | None) -> bool:
    if lock_path in {None, ""}:
        return False
    return Path(lock_path).exists()


def evaluate_production_mode(
    config: ProductionModeConfig,
    *,
    previous_summary_path: Path | str | None,
    stability_report_path: Path | str | None,
    lock_path: Path | str | None,
    phase12_summary_path: Path | str | None,
) -> ProductionModeState:
    mode = config.normalized_mode()
    review_queue_items = current_review_queue_items(phase12_summary_path)
    snapshot_ok = snapshot_verified(
        snapshot_dir=config.required_snapshot_dir,
        snapshot_zip=config.required_snapshot_zip,
    )
    previous_clean = previous_run_completed_cleanly(previous_summary_path)
    deterministic_ok = deterministic_outputs_verified(stability_report_path)
    unresolved_lock = unresolved_lock_exists(lock_path)
    audit_report_available = previous_clean

    failed_reason: str | None = None
    if mode != MODE_OFF:
        if config.require_snapshot_before_run and not snapshot_ok:
            failed_reason = "snapshot_missing"
        elif unresolved_lock:
            failed_reason = "runtime_lock_present"
        elif not previous_clean:
            failed_reason = "previous_run_not_clean"
        elif not deterministic_ok:
            failed_reason = "determinism_not_verified"
        elif mode in {MODE_CONTROLLED, MODE_LIVE} and review_queue_items > 0 and not config.review_queue_acknowledged:
            failed_reason = "review_queue_unacknowledged"
        elif mode in {MODE_CONTROLLED, MODE_LIVE} and config.audit_required and not audit_report_available:
            failed_reason = "audit_report_missing"
        elif mode == MODE_LIVE and not config.run_approval:
            failed_reason = "run_approval_missing"

    production_gate_passed = failed_reason is None
    return ProductionModeState(
        production_mode=mode,
        max_documents_per_run=int(config.max_documents_per_run),
        max_concurrent_runs=int(config.max_concurrent_runs),
        audit_required=bool(config.audit_required),
        require_snapshot_before_run=bool(config.require_snapshot_before_run),
        run_approval=bool(config.run_approval),
        review_queue_acknowledged=bool(config.review_queue_acknowledged),
        required_snapshot_dir=config.required_snapshot_dir,
        required_snapshot_zip=config.required_snapshot_zip,
        production_gate_passed=production_gate_passed,
        production_gate_failed_reason=failed_reason,
        dry_run_executed=mode == MODE_DRY_RUN,
        controlled_run_limit_applied=mode == MODE_CONTROLLED and int(config.max_documents_per_run) > 0,
        run_blocked_by_gate=not production_gate_passed,
        previous_run_completed_cleanly=previous_clean,
        deterministic_outputs_verified=deterministic_ok,
        unresolved_runtime_lock=unresolved_lock,
        snapshot_verified=snapshot_ok,
        audit_report_available=audit_report_available,
        review_queue_items=review_queue_items,
    )
