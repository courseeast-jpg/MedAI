"""Safe privacy audit helpers for reports and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validation_baselines.compare_holdout_baseline import (
    tracked_report_archive_or_review_files,
    tracked_report_phi_files,
)


def assert_no_raw_samples(report: dict[str, Any], forbidden_samples: list[str]) -> bool:
    serialized = json.dumps(report, ensure_ascii=False, default=str)
    return not any(sample and sample in serialized for sample in forbidden_samples)


def phi_artifact_tracking_status() -> dict[str, Any]:
    tracked_phi = tracked_report_phi_files()
    tracked_archive_review = tracked_report_archive_or_review_files()
    return {
        "tracked_report_phi_files": tracked_phi,
        "tracked_report_archive_or_review_files": tracked_archive_review,
        "passed": not tracked_phi and not tracked_archive_review,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
