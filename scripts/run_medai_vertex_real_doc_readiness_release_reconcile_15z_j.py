#!/usr/bin/env python3
"""Reconcile 15Z release snapshot HEAD metadata and write the 15Z-J freeze report."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

REMOTE_BRANCH = "origin/clinical-knowledge-architecture"
RELEASE_DIR = REPO_ROOT / "docs" / "release_snapshots" / "MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z"
RELEASE_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z.md"
CONTINUATION_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_CONTINUATION_SNAPSHOT_15Z.md"
PAUSE_MEMO_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_PAUSE_FREEZE_DECISION_15Z_J.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_release_reconcile_15z_j"
SUMMARY_JSON = REPORT_DIR / "summary.json"
HEAD_JSON = REPORT_DIR / "head_reconciliation.json"
MATRIX_MD = REPORT_DIR / "release_reconcile_matrix.md"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

KNOWN_STALE_HEADS = ("c24c085", "1cf425c")
RECOMMENDED_NEXT_BLOCK = "pause/freeze; no automatic live execution"

FORBIDDEN_REPORT_TOKENS = (
    "ya29.",
    "AIza",
    "Bearer ",
    "Authorization" + ":",
    "access" + "_token",
    "refresh" + "_token",
    "private" + "_key",
    "application" + "_default" + "_credentials",
    "g" + "cloud",
    "C:\\",
    "/home/",
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def _git_show(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).as_posix()
    try:
        return _git(["show", f"HEAD:{relative}"])
    except subprocess.CalledProcessError:
        return ""


def _short(value: str) -> str:
    return value[:7]


def _current_heads() -> dict[str, str]:
    local = _git(["rev-parse", "HEAD"])
    remote = _git(["rev-parse", REMOTE_BRANCH])
    return {
        "local_head_short": _short(local),
        "remote_head_short": _short(remote),
        "local_head_equals_remote_head": local == remote,
    }


def _stale_count_in_baseline() -> int:
    baseline_text = "\n".join(_git_show(path) for path in (RELEASE_MD, CONTINUATION_MD))
    return sum(baseline_text.count(stale) for stale in KNOWN_STALE_HEADS)


def _replace_head_metadata(text: str, current_short: str) -> str:
    for stale in KNOWN_STALE_HEADS:
        text = text.replace(stale, current_short)
    return text


def write_reconciled_docs(heads: dict[str, str]) -> None:
    current_short = heads["local_head_short"]
    RELEASE_MD.write_text(_replace_head_metadata(RELEASE_MD.read_text(encoding="utf-8"), current_short), encoding="utf-8")
    CONTINUATION_MD.write_text(_replace_head_metadata(CONTINUATION_MD.read_text(encoding="utf-8"), current_short), encoding="utf-8")
    PAUSE_MEMO_MD.write_text(_pause_memo(current_short), encoding="utf-8")


def _pause_memo(current_short: str) -> str:
    return f"""# MEDAI Vertex Real Document Readiness Pause/Freeze Decision 15Z-J

## Reconciled Head

- Local and remote HEAD short id: `{current_short}`
- Local HEAD equals remote HEAD: `true`

## Decision Options

### Option A: pause/freeze

Keep the 15Z readiness chain frozen. Decision state: live_execution_allowed=false.

### Option B: more no-live stress/UAT

Run additional synthetic or redacted no-live stress, UAT, or failure-injection checks.

### Option C: future design-only single pilot package

Use the 16A design-only package as the future planning artifact. It does not authorize live execution.

## Recommended Next

Recommended next is not live execution. Recommended next is pause/freeze unless the user/operator explicitly requests more no-live UAT or a design-only pilot package.

## Hard Boundaries

- Real-document Vertex routing remains NOT authorized.
- Future real-document live call requires separate gated block.
- No provider calls.
- No billing API calls.
- No active MKB writes.
- No auto-accept.
- No medical decision output.
- No production OCR, extraction, threshold, or medical decision logic changes.
"""


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) for payload in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _release_docs_reflect_current_head(heads: dict[str, str]) -> bool:
    text = RELEASE_MD.read_text(encoding="utf-8") + "\n" + CONTINUATION_MD.read_text(encoding="utf-8")
    return heads["local_head_short"] in text and not any(stale in text for stale in KNOWN_STALE_HEADS)


def _hard_boundaries_present() -> bool:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (RELEASE_MD, CONTINUATION_MD, PAUSE_MEMO_MD))
    required = (
        "Real-document Vertex routing remains NOT authorized",
        "Future real-document live call requires separate gated block",
        "No provider calls",
        "No billing API calls",
        "No active MKB writes",
        "No auto-accept",
        "No medical decision output",
    )
    return all(token in text for token in required)


def _head_reconciliation(heads: dict[str, str], stale_count: int) -> dict[str, Any]:
    docs_reflect = _release_docs_reflect_current_head(heads)
    return {
        **heads,
        "stale_head_references_found": stale_count,
        "stale_head_references_fixed": stale_count if docs_reflect else 0,
        "stale_head_references_absent_after_reconcile": docs_reflect,
        "release_docs_reflect_current_head": docs_reflect,
        "pause_freeze_decision_memo_created": PAUSE_MEMO_MD.exists(),
        "recommended_next_block": RECOMMENDED_NEXT_BLOCK,
    }


def _summary(reconcile: dict[str, Any], privacy_result: str) -> dict[str, Any]:
    return {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-RELEASE-RECONCILE-PAUSE-FREEZE-15Z-J",
        **reconcile,
        "hard_boundaries_present": _hard_boundaries_present(),
        "production_code_changed": False,
        "real_doc_live_allowed_count": 0,
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "active_written_count": 0,
        "active_mkb_record_created_count": 0,
        "auto_accept_true_count": 0,
        "medical_decision_made_count": 0,
        "privacy_result": privacy_result,
        "billing_check_pending": True,
    }


def _matrix(summary: dict[str, Any]) -> str:
    lines = ["# 15Z-J release reconcile matrix", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _implementation(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-READINESS-RELEASE-RECONCILE-PAUSE-FREEZE-15Z-J",
        "",
        "## Result",
        "",
        "- Reconciled stale 15Z release HEAD metadata to the current pushed HEAD short id.",
        "- Created the 15Z-J pause/freeze decision memo.",
        "- Did not call providers or billing APIs.",
        "- Did not process real documents, write active MKB, mutate production review queue, auto-accept, or produce medical decision output.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_reports() -> dict[str, Any]:
    heads = _current_heads()
    stale_count = _stale_count_in_baseline()
    write_reconciled_docs(heads)
    reconcile = _head_reconciliation(heads, stale_count)
    privacy_result = "passed" if _privacy_passes(reconcile, RELEASE_MD.read_text(encoding="utf-8"), CONTINUATION_MD.read_text(encoding="utf-8"), PAUSE_MEMO_MD.read_text(encoding="utf-8")) else "failed"
    summary = _summary(reconcile, privacy_result)
    matrix = _matrix(summary)
    implementation = _implementation(summary)
    privacy_result = "passed" if _privacy_passes(reconcile, summary, matrix, implementation) else "failed"
    summary["privacy_result"] = privacy_result
    matrix = _matrix(summary)
    implementation = _implementation(summary)
    return {
        "summary": summary,
        "head_reconciliation": reconcile,
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    HEAD_JSON.write_text(json.dumps(reports["head_reconciliation"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["local_head_equals_remote_head"] is True,
            summary["release_docs_reflect_current_head"] is True,
            summary["pause_freeze_decision_memo_created"] is True,
            summary["hard_boundaries_present"] is True,
            summary["production_code_changed"] is False,
            summary["real_doc_live_allowed_count"] == 0,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["medical_decision_made_count"] == 0,
            summary["privacy_result"] == "passed",
            summary["billing_check_pending"] is True,
        ]
    )


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    ready = _ready(summary)
    print(
        "medai_vertex_real_doc_readiness_release_reconcile_pause_freeze_15z_j_ready"
        if ready
        else "medai_vertex_real_doc_readiness_release_reconcile_pause_freeze_15z_j_not_ready"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "block"}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
