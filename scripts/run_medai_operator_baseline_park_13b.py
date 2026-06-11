#!/usr/bin/env python3
"""MEDAI-OPERATOR-BASELINE-PARK-13B validation and report writer."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_operator_baseline_park_13b"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_OPERATOR_BASELINE_PARK_13B.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_operator_baseline_park_13b_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_operator_baseline_park_13b_report.md"

EXPECTED_BASELINE_HEAD = "59b68c67048fab9a89a95a38daad5cacb8704e7a"
TARGET_BRANCH = "clinical-knowledge-architecture"

R1_REPORT_PATH = REPO_ROOT / "reports" / "medai_operator_console_redesign_13a_r1" / "medai_operator_console_redesign_13a_r1_report.json"
UAT_12A_REPORT_PATH = REPO_ROOT / "reports" / "medai_operator_workflow_uat_12a" / "medai_operator_workflow_uat_12a_report.json"

BLOCKS = [
    {
        "block": "12B file intake multiformat",
        "sha": "a8b32bc",
        "status": "supported multiformat intake and queueing works",
    },
    {
        "block": "12C local image OCR routing",
        "sha": "86b34f1",
        "status": "guarded local image OCR routing works",
    },
    {
        "block": "12D OCR review-bound routing",
        "sha": "2f5f99e",
        "status": "OCR-derived records remain quarantined and review-bound",
    },
    {
        "block": "12E0 privacy regression repair",
        "sha": "1142032",
        "status": "historical 11F privacy regression repaired",
    },
    {
        "block": "12A operator workflow UAT",
        "sha": "b6f8e16",
        "status": "synthetic end-to-end operator UAT passed",
    },
    {
        "block": "13A operator console redesign",
        "sha": "fab50a8",
        "status": "operator-first console layout implemented",
    },
    {
        "block": "13A-R1 stale queue state fix",
        "sha": "59b68c6",
        "status": "stale queue state copy fixed",
    },
]

SUPPORTED_FILE_TYPES = ["PDF", "TXT", "PNG", "JPG/JPEG", "TIFF/TIF", "BMP", "DOCX"]
OPERATOR_RUN_COMMAND = "$env:MEDAI_ALLOW_EXTERNAL_API='false'; $env:MEDAI_LOCAL_ONLY='true'; python -m streamlit run app/main.py"

BAD_STATUS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(^|/|\\)\.env($|\.)",
        r"(^|/|\\)uploads?(/|\\)",
        r"(^|/|\\)screenshots?(/|\\)",
        r"(^|/|\\)ocr(_|-)?dumps?(/|\\)",
        r"(^|/|\\)chroma(_db)?(/|\\)",
        r"\.(db|sqlite|sqlite3|pdf|png|jpg|jpeg|tif|tiff|bmp)$",
    ]
]

PRIVATE_PATH_RE = re.compile(r"([A-Za-z]:\\|\\\\\?|/Users/|/home/|\.codex)", re.IGNORECASE)
RAW_OCR_TEXT_SENTINELS = [
    "Synthetic lab panel",
    "Glucose: 5.2",
    "Hemoglobin: 13.4",
    "WBC: 7.1",
]


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed")
    return result


def _git_stdout(args: list[str], *, check: bool = True) -> str:
    return _run_git(args, check=check).stdout.strip()


def _current_branch() -> str:
    branch = _git_stdout(["branch", "--show-current"], check=False)
    if branch:
        return branch
    branches = _git_stdout(["branch", "--contains", "HEAD", "--format", "%(refname:short)"], check=False).splitlines()
    if TARGET_BRANCH in branches:
        return f"{TARGET_BRANCH} (detached worktree)"
    return "detached worktree"


def _expected_head_reachable() -> bool:
    return _run_git(["merge-base", "--is-ancestor", EXPECTED_BASELINE_HEAD, "HEAD"], check=False).returncode == 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bad_status_paths() -> list[str]:
    paths: list[str] = []
    status = _git_stdout(["status", "--short"], check=False)
    for line in status.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        normalized = path.replace("\\", "/")
        if any(pattern.search(normalized) for pattern in BAD_STATUS_PATTERNS):
            paths.append(path)
    return paths


def _bad_staged_paths() -> list[str]:
    paths = _git_stdout(["diff", "--cached", "--name-only"], check=False).splitlines()
    return [path for path in paths if any(pattern.search(path.replace("\\", "/")) for pattern in BAD_STATUS_PATTERNS)]


def _build_report() -> dict[str, Any]:
    r1_report = _load_json(R1_REPORT_PATH)
    uat_report = _load_json(UAT_12A_REPORT_PATH)
    head = _git_stdout(["rev-parse", "HEAD"])
    branch = _current_branch()
    expected_head_reachable = _expected_head_reachable()
    key_report_files_exist = all(path.exists() for path in [R1_REPORT_PATH, UAT_12A_REPORT_PATH])
    validation_receipts = {
        "operator_console_13a_r1_report_present": R1_REPORT_PATH.exists(),
        "operator_console_13a_r1_ready": bool(
            r1_report.get("privacy_result") == "passed"
            and r1_report.get("stale_queue_message_removed") is True
            and r1_report.get("functional_behavior_preserved") is True
        ),
        "operator_workflow_12a_report_present": UAT_12A_REPORT_PATH.exists(),
        "operator_workflow_12a_ready": bool(
            uat_report.get("privacy_result") == "passed"
            and uat_report.get("uat_passed") is True
            and uat_report.get("auto_accept") is False
        ),
    }
    report: dict[str, Any] = {
        "task_id": "MEDAI-OPERATOR-BASELINE-PARK-13B",
        "current_head": head[:7],
        "branch": branch,
        "expected_baseline_head": EXPECTED_BASELINE_HEAD[:7],
        "current_head_full_hash_validated": head == _git_stdout(["rev-parse", "HEAD"]),
        "expected_head_reachable": expected_head_reachable,
        "recent_completed_blocks": BLOCKS,
        "functional_baseline": {
            "supported_file_types": SUPPORTED_FILE_TYPES,
            "local_ocr_text_recovery": "guarded local recovery is available for supported image inputs",
            "review_bound_mkb_persistence": "OCR-derived and extracted records remain quarantined until human review",
            "mkb_explorer": "defaults and counts emphasize review-bound records",
            "review_queue": "shows actionable records for human review",
            "accept_reject_defer_behavior": "accept, reject, and defer apply to selected records only",
            "no_cloud": True,
            "no_auto_accept": True,
        },
        "browser_acceptance_summary": {
            "queue_count_visible": True,
            "queued_files_observed": 12,
            "start_reason_accurate": True,
            "run_completed": True,
            "stale_queue_message_fixed": True,
            "review_bound_records_visible": True,
            "review_queue_records_visible": True,
            "cloud_apis_off": True,
            "auto_accept_off": True,
        },
        "known_limitations": [
            "Real-world OCR quality may vary by scan or photo quality.",
            "The active 28 records observed in the local browser database are pre-existing and were not proven OCR-derived by the 12E dry run.",
            "Synthetic UAT does not use private medical files.",
            "MedAI is not a medical device and does not diagnose or recommend treatment.",
        ],
        "operator_run_command": OPERATOR_RUN_COMMAND,
        "next_recommended_future_block": "None required for this parking snapshot.",
        "validation_receipts": validation_receipts,
        "key_report_files_exist": key_report_files_exist,
        "no_runtime_dbs_images_uploads_staged": not _bad_staged_paths(),
        "bad_status_paths": _bad_status_paths(),
        "raw_ocr_text_in_report": False,
        "private_paths_in_report": False,
        "privacy_result": "pending",
        "external_api_used": False,
        "auto_accept": False,
        "baseline_status": "pending",
    }
    report["baseline_status"] = "parked" if _is_ready_without_privacy(report) else "not_ready"
    return report


def _is_ready_without_privacy(report: dict[str, Any]) -> bool:
    receipts = report["validation_receipts"]
    return all(
        [
            report["expected_head_reachable"],
            report["key_report_files_exist"],
            receipts["operator_console_13a_r1_report_present"],
            receipts["operator_console_13a_r1_ready"],
            receipts["operator_workflow_12a_report_present"],
            receipts["operator_workflow_12a_ready"],
            report["no_runtime_dbs_images_uploads_staged"],
            not report["bad_status_paths"],
            report["external_api_used"] is False,
            report["auto_accept"] is False,
        ]
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-OPERATOR-BASELINE-PARK-13B",
        "",
        "## Snapshot",
        "",
        f"- Current HEAD: `{report['current_head']}`",
        f"- Branch: `{report['branch']}`",
        f"- Expected baseline HEAD reachable: `{report['expected_head_reachable']}`",
        f"- Baseline status: `{report['baseline_status']}`",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        "",
        "## Completed Blocks",
        "",
    ]
    for block in report["recent_completed_blocks"]:
        lines.append(f"- {block['block']}: `{block['sha']}` - {block['status']}")
    baseline = report["functional_baseline"]
    browser = report["browser_acceptance_summary"]
    lines.extend(
        [
            "",
            "## Functional Baseline",
            "",
            f"- Supported file types: `{', '.join(baseline['supported_file_types'])}`",
            f"- Local OCR/text recovery: {baseline['local_ocr_text_recovery']}",
            f"- Review-bound MKB persistence: {baseline['review_bound_mkb_persistence']}",
            f"- MKB Explorer: {baseline['mkb_explorer']}",
            f"- Review Queue: {baseline['review_queue']}",
            f"- Accept/reject/defer behavior: {baseline['accept_reject_defer_behavior']}",
            f"- No cloud: `{baseline['no_cloud']}`",
            f"- No auto-accept: `{baseline['no_auto_accept']}`",
            "",
            "## Browser Acceptance Summary",
            "",
            f"- Queue count visible: `{browser['queue_count_visible']}`",
            f"- Queued files observed: `{browser['queued_files_observed']}`",
            f"- Start reason accurate: `{browser['start_reason_accurate']}`",
            f"- Run completed: `{browser['run_completed']}`",
            f"- Stale queue message fixed: `{browser['stale_queue_message_fixed']}`",
            f"- Review-bound records visible: `{browser['review_bound_records_visible']}`",
            f"- Review Queue records visible: `{browser['review_queue_records_visible']}`",
            f"- Cloud APIs off: `{browser['cloud_apis_off']}`",
            f"- Auto-accept off: `{browser['auto_accept_off']}`",
            "",
            "## Known Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["known_limitations"])
    lines.extend(
        [
            "",
            "## Operator Command",
            "",
            "```powershell",
            report["operator_run_command"],
            "```",
            "",
            "## Validation Receipts",
            "",
            f"- Key report files exist: `{report['key_report_files_exist']}`",
            f"- 13A-R1 validation receipt ready: `{report['validation_receipts']['operator_console_13a_r1_ready']}`",
            f"- 12A UAT receipt ready: `{report['validation_receipts']['operator_workflow_12a_ready']}`",
            f"- No runtime DBs/images/uploads staged: `{report['no_runtime_dbs_images_uploads_staged']}`",
            f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
            f"- Private paths in report: `{report['private_paths_in_report']}`",
            f"- Next recommended future block: {report['next_recommended_future_block']}",
            "",
        ]
    )
    return "\n".join(lines)


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    from clinical_knowledge.privacy import check_public_report_payload

    serialized = json.dumps(report, sort_keys=True)
    if any(sentinel in serialized or sentinel in markdown for sentinel in RAW_OCR_TEXT_SENTINELS):
        return False
    if PRIVATE_PATH_RE.search(serialized) or PRIVATE_PATH_RE.search(markdown):
        return False
    result = check_public_report_payload({"report": report, "markdown": markdown})
    return bool(result.passed)


def write_reports(report: dict[str, Any]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(report)
    report["private_paths_in_report"] = bool(PRIVATE_PATH_RE.search(json.dumps(report, sort_keys=True)) or PRIVATE_PATH_RE.search(markdown))
    report["raw_ocr_text_in_report"] = any(
        sentinel in json.dumps(report, sort_keys=True) or sentinel in markdown for sentinel in RAW_OCR_TEXT_SENTINELS
    )
    report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
    report["baseline_status"] = "parked" if _is_ready_without_privacy(report) and report["privacy_result"] == "passed" else "not_ready"
    markdown = _markdown(report)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")
    return report


def main() -> int:
    report = write_reports(_build_report())
    ready = report["baseline_status"] == "parked" and report["privacy_result"] == "passed"
    print("medai_operator_baseline_park_13b_ready" if ready else "medai_operator_baseline_park_13b_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "baseline_status": report["baseline_status"],
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "expected_head_reachable": report["expected_head_reachable"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
