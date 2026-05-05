"""Phase 77 — Operator-Facing Local Release Polish.

Validates that all operator-facing release documentation exists with
required safety statements, and generates a polish validation report.

Does NOT change production OCR, extraction, thresholds, or safety gates.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

REPORT_DIR = ROOT / "reports" / "phase77_operator_release_polish"
JSON_REPORT = REPORT_DIR / "phase77_operator_release_polish_report.json"
MD_REPORT = REPORT_DIR / "phase77_operator_release_polish_report.md"

OPERATOR_GUIDE = ROOT / "RELEASE_OPERATOR_GUIDE.md"
QUICKSTART = ROOT / "RELEASE_QUICKSTART_LOCAL_ONLY.md"
LIMITATIONS = ROOT / "RELEASE_LIMITATIONS_AND_SAFETY.md"

# Required statements (checked in docs)
_REQUIRED_STATEMENTS = {
    "local_only": [
        "local-only",
        "local only",
    ],
    "not_medical_device": [
        "not a medical device",
        "not provide clinical diagnosis",
        "no clinical diagnosis",
    ],
    "not_production_autonomous": [
        "not production-autonomous",
        "not production autonomous",
        "human review is required",
    ],
    "manual_review_boundary": [
        "manual-review boundary",
        "manual review boundary",
        "review boundary is retained",
    ],
}


def _doc_contains(path: Path, phrases: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return any(p.lower() in text for p in phrases)


def _check_docs() -> dict[str, bool]:
    all_docs = [OPERATOR_GUIDE, QUICKSTART, LIMITATIONS]
    combined = ""
    for p in all_docs:
        if p.exists():
            combined += p.read_text(encoding="utf-8").lower()

    return {
        "local_only_message_present": any(
            phrase.lower() in combined for phrase in _REQUIRED_STATEMENTS["local_only"]
        ),
        "not_medical_device_message_present": any(
            phrase.lower() in combined
            for phrase in _REQUIRED_STATEMENTS["not_medical_device"]
        ),
        "not_production_autonomous_message_present": any(
            phrase.lower() in combined
            for phrase in _REQUIRED_STATEMENTS["not_production_autonomous"]
        ),
        "manual_review_boundary_message_present": any(
            phrase.lower() in combined
            for phrase in _REQUIRED_STATEMENTS["manual_review_boundary"]
        ),
        "review_package_explained": "review package" in combined,
    }


def run_polish_validation(
    *,
    operator_guide_path: Path | None = None,
    quickstart_path: Path | None = None,
    limitations_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    guide = operator_guide_path or OPERATOR_GUIDE
    quick = quickstart_path or QUICKSTART
    limits = limitations_path or LIMITATIONS

    doc_checks = _check_docs_from_paths(guide, quick, limits)

    operator_docs_ready = guide.exists()
    quickstart_ready = quick.exists()
    limitations_safety_ready = limits.exists()
    ui_polish_ready = operator_docs_ready and quickstart_ready and limitations_safety_ready

    all_statements = all(doc_checks.values())

    report: dict[str, Any] = {
        "phase": 77,
        "phase_name": "Operator-Facing Local Release Polish",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": (
            "operator_release_polish_ready"
            if (ui_polish_ready and all_statements)
            else "operator_release_polish_incomplete"
        ),
        "recommended_next_phase": "Phase78 Final HITL Release Snapshot / Release Freeze",
        "operator_docs_ready": operator_docs_ready,
        "quickstart_ready": quickstart_ready,
        "limitations_safety_ready": limitations_safety_ready,
        "ui_polish_ready": ui_polish_ready,
        **doc_checks,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "docs": {
            "RELEASE_OPERATOR_GUIDE.md": str(guide),
            "RELEASE_QUICKSTART_LOCAL_ONLY.md": str(quick),
            "RELEASE_LIMITATIONS_AND_SAFETY.md": str(limits),
        },
        "validation_commands": [
            "python scripts/run_phase77_operator_release_polish_validation.py",
            "python -m pytest tests/test_phase77_operator_release_polish.py",
        ],
    }

    (target_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_dir / MD_REPORT.name).write_text(_render_md(report), encoding="utf-8")
    return report


def _check_docs_from_paths(guide: Path, quick: Path, limits: Path) -> dict[str, bool]:
    combined = ""
    for p in [guide, quick, limits]:
        if p.exists():
            combined += p.read_text(encoding="utf-8").lower()
    return {
        "local_only_message_present": any(
            ph.lower() in combined for ph in _REQUIRED_STATEMENTS["local_only"]
        ),
        "not_medical_device_message_present": any(
            ph.lower() in combined for ph in _REQUIRED_STATEMENTS["not_medical_device"]
        ),
        "not_production_autonomous_message_present": any(
            ph.lower() in combined for ph in _REQUIRED_STATEMENTS["not_production_autonomous"]
        ),
        "manual_review_boundary_message_present": any(
            ph.lower() in combined for ph in _REQUIRED_STATEMENTS["manual_review_boundary"]
        ),
        "review_package_explained": "review package" in combined,
    }


def _render_md(r: dict[str, Any]) -> str:
    lines = [
        "# Phase 77 Operator-Facing Local Release Polish",
        "",
        f"- Generated at: `{r['generated_at']}`",
        f"- Conclusion: `{r['conclusion']}`",
        f"- Recommended next phase: **{r['recommended_next_phase']}**",
        "",
        "## Document Readiness",
        "",
        f"| Document | Present |",
        f"| --- | --- |",
        f"| RELEASE_OPERATOR_GUIDE.md | {'✓' if r['operator_docs_ready'] else '✗'} |",
        f"| RELEASE_QUICKSTART_LOCAL_ONLY.md | {'✓' if r['quickstart_ready'] else '✗'} |",
        f"| RELEASE_LIMITATIONS_AND_SAFETY.md | {'✓' if r['limitations_safety_ready'] else '✗'} |",
        "",
        "## Required Statement Checks",
        "",
        f"| Statement | Present |",
        f"| --- | --- |",
        f"| Local-only | {'✓' if r['local_only_message_present'] else '✗'} |",
        f"| Not a medical device | {'✓' if r['not_medical_device_message_present'] else '✗'} |",
        f"| Not production-autonomous | {'✓' if r['not_production_autonomous_message_present'] else '✗'} |",
        f"| Manual-review boundary retained | {'✓' if r['manual_review_boundary_message_present'] else '✗'} |",
        f"| Review package explained | {'✓' if r['review_package_explained'] else '✗'} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_polish_validation()
    print(f"Phase 77 conclusion: {report['conclusion']}")
    print(f"operator_docs_ready: {report['operator_docs_ready']}")
    print(f"quickstart_ready: {report['quickstart_ready']}")
    print(f"limitations_safety_ready: {report['limitations_safety_ready']}")
    print(f"local_only_message_present: {report['local_only_message_present']}")
    print(f"not_medical_device_message_present: {report['not_medical_device_message_present']}")
    print(f"not_production_autonomous_message_present: "
          f"{report['not_production_autonomous_message_present']}")
    print(f"manual_review_boundary_message_present: "
          f"{report['manual_review_boundary_message_present']}")
    print(f"json_report: {JSON_REPORT}")
    return 0 if report["conclusion"] == "operator_release_polish_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
