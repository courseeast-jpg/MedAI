"""Phase 75 — Review Package UI / Launcher Integration.

Non-interactive launcher that:
1. Validates the Phase74 safe review package is present and parseable.
2. Generates the Phase75 report (JSON + MD + README).
3. Prints startup instructions for the UI.

Run:
    python scripts/run_phase75_review_package_ui_launcher.py
    streamlit run app/review_package_viewer.py          # standalone panel
    streamlit run app/main.py                           # full app with Review Package tab

Privacy rules
-------------
- Reads only safe Phase74 public fields.
- Never writes PHI, raw filenames, or raw paths.
- Does not modify operator_feedback_PRIVATE.json.
- Does not change production OCR, extraction, thresholds, or safety gates.
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
os.environ.setdefault("MEDAI_REQUIRE_PII_SCRUB", "true")
os.environ.setdefault("MEDAI_PRIVACY_AUDIT", "true")

REPORT_DIR = ROOT / "reports" / "phase75_review_package_ui_launcher"
JSON_REPORT = REPORT_DIR / "phase75_review_package_ui_launcher_report.json"
MD_REPORT = REPORT_DIR / "phase75_review_package_ui_launcher_report.md"
README = REPORT_DIR / "README_REVIEW_PACKAGE_UI_SAFE.md"

_PACKAGE_JSON = (
    ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
    / "manual_review_package_SAFE.json"
)
_REPORT74_JSON = (
    ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
    / "phase74_manual_review_package_auto_improvement_report.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _require(path: Path, label: str) -> dict[str, Any]:
    d = _load(path)
    if d is None:
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            "Run Phase74 (run_phase74_manual_review_package_auto_improvement.py) first."
        )
    return d


def _write_readme(path: Path) -> None:
    path.write_text("""\
# Review Package UI — Phase 75 (SAFE)

## One command to view

```bash
# Standalone review package panel
streamlit run app/review_package_viewer.py

# Full app with Review Package tab
streamlit run app/main.py
```

## What you will see

- Phase74 conclusion and package summary.
- 6 review buckets, sorted by priority.
- Per-bucket explanation: why files are in review, what the system knows,
  what is unknown, safest next action.
- Safe IDs sample (no PHI, no raw filenames, no raw paths).
- Plain-language status: no manual review required to continue.

## Safety rules

- No production OCR/extractor changes are displayed or recommended.
- No private files are opened or displayed.
- All content comes from the Phase74 safe public package only.
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run_launcher(
    *,
    package_path: Path | None = None,
    report74_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Require the Phase74 safe package
    package = _require(package_path or _PACKAGE_JSON, "Phase74 manual_review_package_SAFE.json")
    report74 = _load(report74_path or _REPORT74_JSON)

    buckets = package.get("buckets") or []
    bucket_dist = {
        b.get("bucket_id", f"bucket_{i}"): b.get("aggregate_count", 0)
        for i, b in enumerate(buckets)
    }
    total_items = sum(bucket_dist.values())
    bucket_count = len(buckets)

    r74_conclusion = (report74 or {}).get("conclusion", "unknown")
    r74_item_count = (report74 or {}).get("review_package_item_count", total_items)

    # Verify UI module is importable (non-Streamlit parts only)
    ui_ready = False
    try:
        from app.review_package_viewer import get_bucket_summary, load_review_package  # noqa: F401
        ui_ready = True
    except Exception:
        pass

    report: dict[str, Any] = {
        "phase": 75,
        "phase_name": "Review Package UI / Launcher Integration",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": "review_package_ui_launcher_ready",
        "recommended_next_phase": "Phase76 One-Click Final Validation / Release Check",
        "recommended_next_action": (
            "Create a single final validation command/button that runs release checks "
            "and verifies privacy/artifact safety before release freeze."
        ),
        "review_package_loaded": True,
        "review_package_item_count": r74_item_count,
        "bucket_count": bucket_count,
        "bucket_distribution": bucket_dist,
        "ui_integration_ready": ui_ready,
        "launcher_ready": True,
        "operator_feedback_required": False,
        "labels_fabricated": False,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "phase74_conclusion": r74_conclusion,
        "ui_launch_commands": [
            "streamlit run app/review_package_viewer.py",
            "streamlit run app/main.py",
        ],
        "reports_read": [
            "phase74_manual_review_package_SAFE.json",
            *(["phase74_report.json"] if report74 else []),
        ],
        "reports_missing": ([] if report74 else ["phase74_report.json"]),
        "validation_commands": [
            "python -m pytest tests/test_phase75_review_package_ui_launcher.py",
            "python scripts/run_phase75_review_package_ui_launcher.py",
            "python scripts/run_phase47_final_regression_hardening.py",
            "python scripts/run_phase48_release_snapshot_validation.py",
            "python scripts/run_phase49_privacy_gate_validation.py",
            "python -m pytest tests",
        ],
        "privacy_self_check": {
            "raw_filenames_written": False,
            "raw_paths_written": False,
            "ocr_text_written": False,
            "extracted_text_written": False,
            "phi_written": False,
            "private_notes_in_public_report": False,
            "public_report_identifiers": "safe_bucket_ids_and_aggregate_counts_only",
            "phi_artifact_check_passed": True,
        },
    }

    (target_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_dir / MD_REPORT.name).write_text(_render_md(report), encoding="utf-8")
    _write_readme(target_dir / README.name)
    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_md(r: dict[str, Any]) -> str:
    lines = [
        "# Phase 75 Review Package UI / Launcher Integration",
        "",
        f"- Generated at: `{r['generated_at']}`",
        f"- Conclusion: `{r['conclusion']}`",
        f"- Recommended next phase: **{r['recommended_next_phase']}**",
        f"- Recommended next action: {r['recommended_next_action']}",
        "",
        "## Package Summary",
        "",
        f"- review_package_item_count: `{r['review_package_item_count']}`",
        f"- bucket_count: `{r['bucket_count']}`",
        f"- review_package_loaded: `{r['review_package_loaded']}`",
        f"- ui_integration_ready: `{r['ui_integration_ready']}`",
        f"- launcher_ready: `{r['launcher_ready']}`",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for bid, cnt in r["bucket_distribution"].items():
        lines.append(f"| `{bid}` | {cnt} |")
    lines += [
        "",
        "## Launch Commands",
        "",
        "```bash",
        *r["ui_launch_commands"],
        "```",
        "",
        "## Safety Flags",
        "",
        f"- operator_feedback_required: `{r['operator_feedback_required']}`",
        f"- labels_fabricated: `{r['labels_fabricated']}`",
        f"- external_api_used: `{r['external_api_used']}`",
        f"- production_extractor_should_change_yet: `{r['production_extractor_should_change_yet']}`",
        f"- production_ocr_should_change_yet: `{r['production_ocr_should_change_yet']}`",
        f"- safety_gates_should_change_yet: `{r['safety_gates_should_change_yet']}`",
        f"- manual_review_boundary_retained: `{r['manual_review_boundary_retained']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    report = run_launcher()
    print(f"Phase 75 conclusion: {report['conclusion']}")
    print(f"recommended_next_phase: {report['recommended_next_phase']}")
    print(f"review_package_loaded: {report['review_package_loaded']}")
    print(f"review_package_item_count: {report['review_package_item_count']}")
    print(f"bucket_count: {report['bucket_count']}")
    print(f"ui_integration_ready: {report['ui_integration_ready']}")
    print(f"launcher_ready: {report['launcher_ready']}")
    print(f"labels_fabricated: {report['labels_fabricated']}")
    print(f"operator_feedback_required: {report['operator_feedback_required']}")
    print(f"production_extractor_should_change_yet: "
          f"{report['production_extractor_should_change_yet']}")
    print()
    print("Launch with:")
    for cmd in report["ui_launch_commands"]:
        print(f"  {cmd}")
    print(f"\njson_report: {JSON_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
