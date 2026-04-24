from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, UTC
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "artifacts" / "phase11_integration"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def direct_checks() -> dict:
    from app.config import (
        ENABLE_DECISION_SCORING,
        ENABLE_HYPOTHESIS_TIER,
        ENABLE_TRUTH_RESOLUTION,
    )
    from execution.pipeline import ExecutionPipeline  # noqa: F401
    from governance.decision_scoring import GovernanceDecisionScoring
    from governance.governance_ledger import GovernanceLedger
    from governance.hypothesis_tier import GovernanceHypothesisTier
    from governance.truth_resolution import GovernanceTruthResolutionAdapter
    from tests.test_phase11_governance import make_record

    flags_enabled = {
        "ENABLE_HYPOTHESIS_TIER": ENABLE_HYPOTHESIS_TIER,
        "ENABLE_TRUTH_RESOLUTION": ENABLE_TRUTH_RESOLUTION,
        "ENABLE_DECISION_SCORING": ENABLE_DECISION_SCORING,
    }

    tiering = GovernanceHypothesisTier(enabled=True)
    active = make_record(name="Epilepsy", tier="active")
    hypothesis = tiering.classify_record(make_record(
        name="AI differential",
        trust_level=3,
        source_type="ai_response",
        extraction_method="gemini",
    ))
    active_context = tiering.active_context([active, hypothesis])

    existing = make_record(name="Lamotrigine", fact_type="medication", dose="100mg")
    candidate = make_record(name="Lamotrigine", fact_type="medication", dose="200mg")
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = GovernanceLedger(Path(tmpdir) / "governance_ledger.jsonl")
        adapter = GovernanceTruthResolutionAdapter(
            enabled=True,
            existing_records_provider=lambda record: [existing],
            ledger=ledger,
        )
        resolution = adapter.resolve_batch([candidate])
        ledger_lines = ledger.path.read_text(encoding="utf-8").splitlines()

    scoring = GovernanceDecisionScoring(enabled=True).score(
        content="Epilepsy management is appropriate because Lamotrigine is already listed.",
        mkb_context=[active],
        citations=["guideline-1"],
        ddi_safety_score=0.9,
    )
    rollback_scoring = GovernanceDecisionScoring(enabled=False).score(content="short note")

    return {
        "flags_enabled": flags_enabled,
        "hypothesis_excluded_from_active_context": active_context == [active],
        "truth_resolution_non_destructive_overwrite": resolution.records_to_write == [] and len(resolution.quarantined_records) == 2,
        "governance_ledger_records_produced": len(ledger_lines) > 0,
        "decision_scoring_deterministic": scoring.enabled and scoring.final_score == 0.93,
        "rollback_path_exists": rollback_scoring.enabled is False and rollback_scoring.score_breakdown == {},
    }


def summarize_failures(pytest_stdout: str) -> list[str]:
    failures: list[str] = []
    for line in pytest_stdout.splitlines():
        if line.startswith("FAILED "):
            failures.append(line.replace("FAILED ", "", 1))
    return failures


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_suite = run_command([sys.executable, "-m", "pytest", "tests"])
    phase10 = run_command([sys.executable, "scripts\\run_phase10_hardening.py", "--output-dir", "artifacts\\phase10"])
    phase11 = run_command([sys.executable, "scripts\\run_phase11_activation.py"])
    checks = direct_checks()

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": "phase11-governance",
        "commands": [
            {"command": full_suite["command"], "returncode": full_suite["returncode"]},
            {"command": phase10["command"], "returncode": phase10["returncode"]},
            {"command": phase11["command"], "returncode": phase11["returncode"]},
        ],
        "full_suite_passed": full_suite["returncode"] == 0,
        "phase10_runner_passed": phase10["returncode"] == 0,
        "phase11_activation_passed": phase11["returncode"] == 0,
        "direct_checks": checks,
        "full_suite_failures": summarize_failures(full_suite["stdout"]),
        "merge_recommended": (
            full_suite["returncode"] == 0
            and phase10["returncode"] == 0
            and phase11["returncode"] == 0
            and all(checks.values())
        ),
    }

    (OUTPUT_DIR / "phase11_integration_audit_full_suite_stdout.txt").write_text(full_suite["stdout"], encoding="utf-8")
    (OUTPUT_DIR / "phase11_integration_audit_full_suite_stderr.txt").write_text(full_suite["stderr"], encoding="utf-8")
    (OUTPUT_DIR / "phase11_integration_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Phase 11 Integration Audit",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Full suite passed: {summary['full_suite_passed']}",
        f"- Phase 10 runner passed: {summary['phase10_runner_passed']}",
        f"- Phase 11 activation runner passed: {summary['phase11_activation_passed']}",
        f"- Governance flags enabled: {checks['flags_enabled']}",
        f"- Hypothesis excluded from active context: {checks['hypothesis_excluded_from_active_context']}",
        f"- Truth resolution non-destructive: {checks['truth_resolution_non_destructive_overwrite']}",
        f"- Governance ledger produced records: {checks['governance_ledger_records_produced']}",
        f"- Decision scoring deterministic: {checks['decision_scoring_deterministic']}",
        f"- Rollback path exists: {checks['rollback_path_exists']}",
        f"- Merge recommended now: {summary['merge_recommended']}",
        "",
        "## Full Suite Failures",
        "",
    ]
    if summary["full_suite_failures"]:
        md_lines.extend(f"- {item}" for item in summary["full_suite_failures"])
    else:
        md_lines.append("- None")
    (OUTPUT_DIR / "phase11_integration_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if summary["merge_recommended"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
