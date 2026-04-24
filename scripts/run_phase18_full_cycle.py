from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.run_comparator import write_stability_report

PHASE18_REPORT_DIR = ROOT / "reports" / "phase18"
PHASE11_AUDIT_PATH = ROOT / "artifacts" / "phase11_integration" / "phase11_integration_audit.json"
PHASE12_SUMMARY_PATH = ROOT / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
PHASE17_DASHBOARD_PATH = ROOT / "reports" / "phase17" / "dashboard_latest.md"

PHASE18_STEPS: list[tuple[str, list[str]]] = [
    ("tests", [sys.executable, "-m", "pytest", "tests"]),
    ("phase11_audit", [sys.executable, "scripts\\run_phase11_integration_audit.py"]),
    (
        "validation",
        [
            sys.executable,
            "scripts\\run_phase12_real_world_validation.py",
            "--dataset-dir",
            "test_data\\final_batch_50",
            "--quota-safe",
        ],
    ),
    ("dashboard_latest", [sys.executable, "scripts\\run_phase17_dashboard.py", "--latest"]),
    ("dashboard_export", [sys.executable, "scripts\\run_phase17_dashboard.py", "--export"]),
]

PYTEST_SUMMARY_RE = re.compile(r"=+\s+(\d+)\s+passed(?:,.*)?\s+=+")


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def summarize_pytest(stdout: str) -> str:
    match = PYTEST_SUMMARY_RE.search(stdout)
    if match:
        return f"{match.group(1)} passed"
    return "unknown"


def git_commit_hash() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def git_status_state() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return "clean" if not completed.stdout.strip() else "dirty"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(*, commands: list[dict], started_at: datetime, ended_at: datetime) -> dict:
    phase11 = load_json(PHASE11_AUDIT_PATH) if PHASE11_AUDIT_PATH.exists() else {}
    phase12 = load_json(PHASE12_SUMMARY_PATH) if PHASE12_SUMMARY_PATH.exists() else {}
    pytest_step = next((item for item in commands if item["name"] == "tests"), None)
    failed_step = next((item["name"] for item in commands if item["returncode"] != 0), None)

    return {
        "generated_at": ended_at.isoformat(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "commit_hash": git_commit_hash(),
        "git_status": git_status_state(),
        "steps": [
            {
                "name": item["name"],
                "command": item["command"],
                "returncode": item["returncode"],
            }
            for item in commands
        ],
        "success": failed_step is None,
        "failed_step": failed_step,
        "test_result": summarize_pytest(pytest_step["stdout"]) if pytest_step else "unknown",
        "phase11_audit_result": "passed" if phase11.get("merge_recommended") else "failed",
        "validation_result": {
            "attempted": int(phase12.get("documents_selected", 0)),
            "processed": int(phase12.get("documents_processed", 0)),
            "written": int(phase12.get("written", 0)),
            "queued_for_review": int(phase12.get("queued_for_review", 0)),
            "external_quota_blocked": int(phase12.get("external_quota_blocked", 0)),
            "hard_failures": int(phase12.get("hard_failures", 0)),
            "avg_confidence": float(phase12.get("aggregate", {}).get("avg_confidence", 0.0)),
        },
        "dashboard_export_path": str(PHASE17_DASHBOARD_PATH),
        "stability_report_path": str(ROOT / "reports" / "phase19" / "stability_report.md"),
    }


def write_summary_reports(summary: dict, report_dir: Path = PHASE18_REPORT_DIR) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "full_cycle_summary.json"
    md_path = report_dir / "full_cycle_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validation = summary["validation_result"]
    lines = [
        "# Phase 18 Full Cycle Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Commit hash: `{summary['commit_hash']}`",
        f"- Git status: `{summary['git_status']}`",
        f"- Success: `{summary['success']}`",
        f"- Failed step: `{summary['failed_step']}`",
        f"- Test result: `{summary['test_result']}`",
        f"- Phase 11 audit result: `{summary['phase11_audit_result']}`",
        f"- Validation attempted: `{validation['attempted']}`",
        f"- Validation processed: `{validation['processed']}`",
        f"- Validation written: `{validation['written']}`",
        f"- Validation queued_for_review: `{validation['queued_for_review']}`",
        f"- Validation external_quota_blocked: `{validation['external_quota_blocked']}`",
        f"- Validation hard_failures: `{validation['hard_failures']}`",
        f"- Validation avg_confidence: `{validation['avg_confidence']}`",
        f"- Dashboard export path: `{summary['dashboard_export_path']}`",
        f"- Stability report path: `{summary['stability_report_path']}`",
        f"- Duration seconds: `{summary['duration_seconds']}`",
        "",
        "## Steps",
        "",
    ]
    lines.extend(
        f"- `{item['name']}` -> returncode={item['returncode']} command={' '.join(item['command'])}"
        for item in summary["steps"]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def execute_steps(
    steps: list[tuple[str, list[str]]] = PHASE18_STEPS,
    *,
    runner=run_command,
) -> list[dict]:
    results: list[dict] = []
    for name, command in steps:
        result = runner(command)
        results.append({
            "name": name,
            **result,
        })
        if result["returncode"] != 0:
            break
    return results


def main() -> int:
    started_at = datetime.now(UTC)
    results = execute_steps()
    ended_at = datetime.now(UTC)
    summary = build_summary(commands=results, started_at=started_at, ended_at=ended_at)
    write_summary_reports(summary)
    write_stability_report()
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
