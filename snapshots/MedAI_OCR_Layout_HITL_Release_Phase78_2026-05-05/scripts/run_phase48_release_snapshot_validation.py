"""Validate the Phase48 operator release package and snapshot."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_baselines.compare_holdout_baseline import (
    tracked_report_archive_or_review_files,
    tracked_report_phi_files,
)


PACKAGE_DIR = ROOT / "reports" / "phase48_operator_release_package"
REQUIRED_FILES = {
    "release_summary": PACKAGE_DIR / "medai_v2_ocr_layout_release_summary.md",
    "operator_review_guide": PACKAGE_DIR / "operator_review_guide.md",
    "validation_runbook": PACKAGE_DIR / "validation_runbook.md",
    "release_snapshot_json": PACKAGE_DIR / "release_snapshot.json",
    "release_snapshot_md": PACKAGE_DIR / "release_snapshot.md",
}
PHASE47_REPORT = ROOT / "reports" / "phase47_final_regression_hardening" / "phase47_final_regression_hardening_report.json"
JSON_REPORT = PACKAGE_DIR / "phase48_release_validation_report.json"
MD_REPORT = PACKAGE_DIR / "phase48_release_validation_report.md"


def run_validation() -> dict[str, Any]:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    current_head = _git_head()
    missing_docs = [name for name, path in REQUIRED_FILES.items() if not path.exists()]
    snapshot = _load_json(REQUIRED_FILES["release_snapshot_json"])
    phase47 = _load_json(PHASE47_REPORT)
    tracked_phi = tracked_report_phi_files()
    tracked_archive_review = tracked_report_archive_or_review_files()
    doc_checks = _document_checks(snapshot=snapshot, current_head=current_head)
    safety_report_missing = not PHASE47_REPORT.exists() or not bool(phase47.get("release_candidate_ready"))
    blocked_by_phi = bool(tracked_phi or tracked_archive_review)
    conclusion = _conclusion(
        missing_docs=missing_docs,
        blocked_by_phi=blocked_by_phi,
        safety_report_missing=safety_report_missing,
        doc_checks=doc_checks,
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 48 Operator Review Package + Release Snapshot",
        "snapshot_id": snapshot.get("snapshot_id"),
        "current_head": current_head,
        "validated_source_commit": snapshot.get("validated_source_commit"),
        "phase47_report": str(PHASE47_REPORT),
        "phase47_release_candidate_ready": bool(phase47.get("release_candidate_ready")),
        "required_files": {name: str(path) for name, path in REQUIRED_FILES.items()},
        "missing_docs": missing_docs,
        "doc_checks": doc_checks,
        "phi_artifact_check": {
            "tracked_report_phi_files": tracked_phi,
            "tracked_report_archive_or_review_files": tracked_archive_review,
            "passed": not blocked_by_phi,
        },
        "release_snapshot_schema_valid": _snapshot_schema_valid(snapshot),
        "conclusion": conclusion,
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def _document_checks(*, snapshot: dict[str, Any], current_head: str) -> dict[str, bool]:
    texts = {name: path.read_text(encoding="utf-8") for name, path in REQUIRED_FILES.items() if path.exists() and path.suffix == ".md"}
    combined = "\n".join(texts.values())
    phase48_package_commit = snapshot.get("phase48_package_commit")
    return {
        "tests_baseline_referenced": "411 passed, 5 warnings" in combined or snapshot.get("test_result", {}).get("result") == "411 passed, 5 warnings",
        "phase47_commit_referenced": snapshot.get("phase47_commit") == "31024c7f18b65144addf0141876b040fbf92eaaf",
        "phase48_package_commit_referenced": phase48_package_commit == "020375413ff7c455c86adfdf362880bd8c4ad9c2" and phase48_package_commit in combined,
        "count_convention_documented": "total == accepted + review" in combined,
        "safety_guarantees_documented": all(
            phrase in combined
            for phrase in [
                "Poor OCR cannot become accepted",
                "Empty extraction cannot become accepted",
                "Lab normalizer cannot produce accepted",
                "Cyrillic non-lab reconciliation cannot produce accepted",
            ]
        ),
        "not_medical_device_documented": "not a medical device" in combined,
    }


def _snapshot_schema_valid(snapshot: dict[str, Any]) -> bool:
    required = {
        "snapshot_id",
        "release_name",
        "validated_source_commit",
        "completed_phases",
        "final_metrics",
        "safety_invariants",
        "count_convention",
        "known_limitations",
        "resume_commands",
        "validation_commands",
        "next_valid_future_work",
    }
    return required.issubset(snapshot) and snapshot.get("snapshot_id") == "MedAI_Snapshot_Phase48_2026-05-01"


def _conclusion(
    *,
    missing_docs: list[str],
    blocked_by_phi: bool,
    safety_report_missing: bool,
    doc_checks: dict[str, bool],
) -> str:
    if blocked_by_phi:
        return "blocked_by_phi_artifacts"
    if safety_report_missing:
        return "blocked_by_safety_report_missing"
    if missing_docs or not all(doc_checks.values()):
        return "blocked_by_missing_docs"
    return "release_snapshot_ready"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return (result.stdout or "").strip()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 48 Release Snapshot Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Snapshot ID: `{report.get('snapshot_id')}`",
        f"- Current HEAD: `{report['current_head']}`",
        f"- Validated source commit: `{report.get('validated_source_commit')}`",
        f"- Phase47 release_candidate_ready: `{report['phase47_release_candidate_ready']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Required Files",
        "",
    ]
    for name, path in report["required_files"].items():
        missing = name in report["missing_docs"]
        lines.append(f"- {name}: `{path}` present `{not missing}`")
    lines += [
        "",
        "## Document Checks",
        "",
    ]
    for name, passed in report["doc_checks"].items():
        lines.append(f"- {name}: `{passed}`")
    phi = report["phi_artifact_check"]
    lines += [
        "",
        "## PHI Artifact Check",
        "",
        f"- passed: `{phi['passed']}`",
        f"- tracked_report_phi_files: `{phi['tracked_report_phi_files']}`",
        f"- tracked_report_archive_or_review_files: `{phi['tracked_report_archive_or_review_files']}`",
        "",
        "## Snapshot Schema",
        "",
        f"- release_snapshot_schema_valid: `{report['release_snapshot_schema_valid']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_validation()
    print("MedAI Phase 48 release snapshot validation complete.")
    print(f"snapshot_id: {report.get('snapshot_id')}")
    print(f"phase47_release_candidate_ready: {report['phase47_release_candidate_ready']}")
    print(f"release_snapshot_schema_valid: {report['release_snapshot_schema_valid']}")
    print(f"phi_artifact_check_passed: {report['phi_artifact_check']['passed']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0 if report["conclusion"] in {"release_snapshot_ready", "ready_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
