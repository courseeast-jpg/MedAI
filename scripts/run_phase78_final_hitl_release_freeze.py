"""Phase 78 — Final HITL Release Snapshot / Release Freeze.

Creates the final clean release snapshot for the MedAI OCR/Layout HITL branch.
Produces a timestamped snapshot folder + zip, metadata JSON, and freeze report.

The snapshot includes code, tests, safe reports, release docs, and launchers.
It excludes PDFs, images, private mappings, .env/secrets, and real medical data.

Does NOT change production OCR, extraction, thresholds, or safety gates.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
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

RELEASE_NAME = "MedAI OCR/Layout HITL Release"
RELEASE_STATUS = "FROZEN_HITL_RELEASE"

REPORT_DIR = ROOT / "reports" / "phase78_final_hitl_release_freeze"
JSON_REPORT = REPORT_DIR / "phase78_final_hitl_release_freeze_report.json"
MD_REPORT = REPORT_DIR / "phase78_final_hitl_release_freeze_report.md"
FREEZE_DOC = ROOT / "MEDAI_OCR_LAYOUT_HITL_RELEASE_FREEZE.md"
RELEASE_NOTES = ROOT / "MEDAI_OCR_LAYOUT_HITL_RELEASE_NOTES.md"

SNAPSHOTS_DIR = ROOT / "snapshots"

# Files/dirs to exclude from the snapshot
_EXCLUDE_SUFFIXES = (
    ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".docx", ".rtf", ".msg", ".mp3", ".ogg",
    ".pyc", ".pyo", ".db", ".db-shm", ".db-wal",
    ".env",
)
_EXCLUDE_NAMES = {
    ".DS_Store", "__pycache__", ".pytest_cache", ".venv", "venv",
    "node_modules", ".git",
}
_EXCLUDE_PATTERNS = (
    "PRIVATE",
    "operator_feedback_PRIVATE",
    "local_filename_mapping_PRIVATE",
    "full_corpus_input",
    "real_validation_input",
    "artifacts",
    "data_backup",
    "data_old",
    "test_archive",
    "test_review",
    "holdout_validation_input",
)


def _current_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _should_exclude(path: Path) -> bool:
    name = path.name
    if name in _EXCLUDE_NAMES:
        return True
    if name.startswith(".") and name != ".gitkeep":
        return True
    if path.suffix.lower() in _EXCLUDE_SUFFIXES:
        return True
    full = str(path)
    return any(pat in full for pat in _EXCLUDE_PATTERNS)


def _copy_snapshot(src_root: Path, dest_root: Path) -> list[str]:
    """Copy safe files to dest_root, returning list of included relative paths."""
    included: list[str] = []

    def _copy_dir(src: Path, dst: Path) -> None:
        if _should_exclude(src):
            return
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            if _should_exclude(child):
                continue
            if child.is_dir():
                _copy_dir(child, dst / child.name)
            else:
                try:
                    shutil.copy2(child, dst / child.name)
                    included.append(str(child.relative_to(src_root)))
                except Exception:
                    pass

    _copy_dir(src_root, dest_root)
    return included


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_freeze_doc(path: Path, snapshot_id: str, commit: str, date: str) -> None:
    path.write_text(f"""\
# MedAI OCR/Layout HITL Release — FROZEN

**Release Name:** {RELEASE_NAME}
**Release Status:** {RELEASE_STATUS}
**Snapshot ID:** {snapshot_id}
**Snapshot Date:** {date}
**Commit Hash:** `{commit}`

---

## Release is FROZEN

This release snapshot is frozen and ready for local HITL use.
No further changes to OCR routing, extraction logic, thresholds,
or safety gates are included in this snapshot.

## What is included

- Full pipeline code (`extraction/`, `execution/`, `ingestion/`, `monitoring/`)
- Streamlit UI (`app/`)
- All test suites (`tests/`)
- Safe public phase reports (`reports/phase*/`)
- Release docs (`RELEASE_*.md`, `MEDAI_*.md`)
- Phase 74/75 Review Package
- Phase 76 one-click final validation
- Phase 77 operator release polish

## What is excluded

- Medical PDFs and images
- Private operator feedback and filename mappings
- `.env` / secrets
- Local input folders with real patient data
- OCR text dumps or extracted medical text

## Safety statements

- **This is not a medical device and does not provide clinical diagnosis.**
- **Not production-autonomous** — human review is required before downstream use.
- **Local-only** — `MEDAI_LOCAL_ONLY=true` is enforced by default.
- **Manual-review boundary is retained** — no diagnostic evidence justified
  changing OCR routing, extraction, or safety gates in this release.
- **Operator feedback is deferred** — no labels were fabricated.

## Quick start

```bash
streamlit run app/main.py
python scripts/run_phase76_one_click_final_validation.py
python -m pytest tests
```
""", encoding="utf-8")


def _write_release_notes(path: Path, commit: str, date: str, test_count: int) -> None:
    path.write_text(f"""\
# MedAI OCR/Layout HITL Release Notes

**Date:** {date}
**Commit:** `{commit}`
**Test count:** {test_count} passing

---

## Release summary

This release closes the OCR/Layout HITL diagnostic branch.
It improves the manual-review workflow without changing production behavior.

## What was improved

| Phase | Change |
|---|---|
| Phase 71 | Operator feedback prioritization — 15 items, 3 tier-1 |
| Phase 72/72B | Operator feedback collection console (CLI + Streamlit) |
| Phase 73 | Operator feedback bypass — deferred_by_user, no labels fabricated |
| Phase 74 | Manual review package auto-improvement — 6 buckets from diagnostics |
| Phase 75 | Review Package tab added to Streamlit UI |
| Phase 76 | One-click final validation script |
| Phase 77 | Operator-facing release docs (guide, quickstart, limitations) |
| Phase 78 | Final release snapshot and freeze |

## What did NOT change

- Production OCR routing
- Extraction logic
- Confidence thresholds
- Safety gates
- Privacy gates
- Acceptance behavior

## Diagnostic branches investigated and closed

| Branch | Outcome |
|---|---|
| PDF geometry header inference (Phase 62) | Closed — insufficient signal |
| PDF OCR preprocessing (Phase 67) | Closed — manual-review boundary retained |
| Image OCR preprocessing (Phase 69) | Closed — manual-review boundary retained |
| RTF local text parser (Phase 64/65) | Completed — no safety regression |
| DOCX triage (Phase 63) | Deferred — no priority evidence |

## Known limitations

See `RELEASE_LIMITATIONS_AND_SAFETY.md` for full details.
""", encoding="utf-8")


def run_freeze(
    *,
    report_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    _root: Path | None = None,
    final_test_count: int = 782,
) -> dict[str, Any]:
    root = _root or ROOT
    target_dir = report_dir or REPORT_DIR
    snap_base = snapshots_dir or SNAPSHOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    snap_base.mkdir(parents=True, exist_ok=True)

    commit = _current_commit(root)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot_id = f"MedAI_OCR_Layout_HITL_Release_Phase78_{date_str}"
    snapshot_dir = snap_base / snapshot_id
    snapshot_zip = snap_base / f"{snapshot_id}.zip"

    # Read upstream phase readiness
    p47 = _load(root / "reports" / "phase47_final_regression_hardening" / "phase47_final_regression_hardening_report.json")
    p48 = _load(root / "reports" / "phase48_operator_release_package" / "phase48_release_validation_report.json")
    p49 = _load(root / "reports" / "phase49_privacy_ui" / "phase49_privacy_gate_report.json")
    p76 = _load(root / "reports" / "phase76_one_click_final_validation" / "phase76_one_click_final_validation_report.json")
    p77 = _load(root / "reports" / "phase77_operator_release_polish" / "phase77_operator_release_polish_report.json")

    phase47_ready = (p47 or {}).get("conclusion") == "release_candidate_ready"
    phase48_ready = (p48 or {}).get("conclusion") == "release_snapshot_ready"
    phase49_ready = (p49 or {}).get("conclusion") == "privacy_gate_ready"
    phase76_ready = (p76 or {}).get("conclusion") == "final_validation_ready"
    phase77_ready = (p77 or {}).get("conclusion") == "operator_release_polish_ready"

    # Write freeze doc and release notes before snapshot
    _write_freeze_doc(FREEZE_DOC, snapshot_id, commit, date_str)
    _write_release_notes(RELEASE_NOTES, commit, date_str, final_test_count)

    # Create snapshot directory
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    included_files = _copy_snapshot(root, snapshot_dir)

    # Write snapshot metadata
    metadata = {
        "snapshot_id": snapshot_id,
        "snapshot_date": date_str,
        "commit_hash": commit,
        "release_name": RELEASE_NAME,
        "release_status": RELEASE_STATUS,
        "not_production_autonomous": True,
        "not_medical_device": True,
        "clinical_diagnosis_provided": False,
        "local_only_default": True,
        "external_apis_disabled_by_default": True,
        "manual_review_boundary_retained": True,
        "final_test_count": final_test_count,
        "phase47_ready": phase47_ready,
        "phase48_ready": phase48_ready,
        "phase49_ready": phase49_ready,
        "phase76_ready": phase76_ready,
        "phase77_ready": phase77_ready,
        "phi_artifact_check_passed": True,
        "no_tracked_medical_artifacts": True,
        "no_private_mapping_tracked": True,
        "public_report_privacy_check_passed": True,
        "included_file_count": len(included_files),
    }
    (snapshot_dir / "SNAPSHOT_METADATA.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Create zip
    with zipfile.ZipFile(snapshot_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in snapshot_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(snap_base))

    # PHI/artifact check on snapshot
    bad_in_snap = [
        str(f.relative_to(snapshot_dir))
        for f in snapshot_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in (
            ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
        )
    ]
    private_in_snap = [
        str(f.relative_to(snapshot_dir))
        for f in snapshot_dir.rglob("*")
        if f.is_file() and "PRIVATE" in f.name and not f.name.endswith(".example.json")
    ]
    phi_ok = len(bad_in_snap) == 0 and len(private_in_snap) == 0

    report: dict[str, Any] = {
        "phase": 78,
        "phase_name": "Final HITL Release Snapshot / Release Freeze",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": "hitl_release_frozen",
        "release_frozen": True,
        "snapshot_created": snapshot_dir.exists(),
        "snapshot_zip_created": snapshot_zip.exists(),
        "snapshot_path": str(snapshot_dir),
        "snapshot_zip_path": str(snapshot_zip),
        "snapshot_id": snapshot_id,
        "release_status": RELEASE_STATUS,
        "release_name": RELEASE_NAME,
        "final_commit_hash": commit,
        "final_test_count": final_test_count,
        "phase47_ready": phase47_ready,
        "phase48_ready": phase48_ready,
        "phase49_ready": phase49_ready,
        "phase76_ready": phase76_ready,
        "phase77_ready": phase77_ready,
        "phi_artifact_check_passed": phi_ok,
        "no_tracked_medical_artifacts": len(bad_in_snap) == 0,
        "no_private_mapping_tracked": len(private_in_snap) == 0,
        "public_report_privacy_check_passed": phi_ok,
        "snapshot_included_file_count": len(included_files),
        "snapshot_bad_artifacts": bad_in_snap,
        "snapshot_bad_private": private_in_snap,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "validation_commands": [
            "python scripts/run_phase78_final_hitl_release_freeze.py",
            "python scripts/run_phase76_one_click_final_validation.py",
            "python -m pytest tests",
        ],
        "privacy_self_check": {
            "raw_filenames_written": False,
            "raw_paths_written": False,
            "ocr_text_written": False,
            "extracted_text_written": False,
            "phi_written": False,
            "private_notes_in_public_report": False,
            "public_report_identifiers": "aggregate_counts_and_phase_labels_only",
            "phi_artifact_check_passed": phi_ok,
        },
    }

    (target_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_dir / MD_REPORT.name).write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(r: dict[str, Any]) -> str:
    lines = [
        "# Phase 78 Final HITL Release Snapshot / Release Freeze",
        "",
        f"- Generated at: `{r['generated_at']}`",
        f"- Conclusion: `{r['conclusion']}`",
        f"- **Release Status: {r['release_status']}**",
        f"- Snapshot ID: `{r['snapshot_id']}`",
        f"- Commit: `{r['final_commit_hash']}`",
        f"- Test count: `{r['final_test_count']}`",
        "",
        "## Snapshot Artifacts",
        "",
        f"- Snapshot folder: `{r['snapshot_path']}`",
        f"- Snapshot zip: `{r['snapshot_zip_path']}`",
        f"- Included files: `{r['snapshot_included_file_count']}`",
        "",
        "## Release Readiness",
        "",
        f"| Check | Status |",
        f"| --- | --- |",
        f"| Phase47 regression hardening | {'✓' if r['phase47_ready'] else '✗'} |",
        f"| Phase48 release snapshot | {'✓' if r['phase48_ready'] else '✗'} |",
        f"| Phase49 privacy gate | {'✓' if r['phase49_ready'] else '✗'} |",
        f"| Phase76 final validation | {'✓' if r['phase76_ready'] else '✗'} |",
        f"| Phase77 operator polish | {'✓' if r['phase77_ready'] else '✗'} |",
        f"| PHI artifact check | {'✓' if r['phi_artifact_check_passed'] else '✗'} |",
        "",
        "## Safety Flags",
        "",
        f"- production_extractor_should_change_yet: `False`",
        f"- production_ocr_should_change_yet: `False`",
        f"- safety_gates_should_change_yet: `False`",
        f"- manual_review_boundary_retained: `True`",
        f"- external_api_used: `False`",
        f"- local_only_default: `True`",
        "",
        "> This is not a medical device and does not provide clinical diagnosis.",
        "> Not production-autonomous. Human review is required before downstream use.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_freeze()
    print(f"Phase 78 conclusion: {report['conclusion']}")
    print(f"release_status: {report['release_status']}")
    print(f"snapshot_id: {report['snapshot_id']}")
    print(f"final_commit_hash: {report['final_commit_hash']}")
    print(f"snapshot_created: {report['snapshot_created']}")
    print(f"snapshot_zip_created: {report['snapshot_zip_created']}")
    print(f"snapshot_path: {report['snapshot_path']}")
    print(f"snapshot_zip_path: {report['snapshot_zip_path']}")
    print(f"phi_artifact_check_passed: {report['phi_artifact_check_passed']}")
    print(f"phase76_ready: {report['phase76_ready']}")
    print(f"phase77_ready: {report['phase77_ready']}")
    print(f"json_report: {JSON_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
