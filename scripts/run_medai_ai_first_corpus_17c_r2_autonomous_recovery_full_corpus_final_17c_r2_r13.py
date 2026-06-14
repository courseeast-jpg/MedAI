#!/usr/bin/env python3
"""CLI for 17C-R2-R13 autonomous guarded recovery and full corpus run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.autonomous_recovery_runner import BLOCK, run_autonomous_recovery  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local-only", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    summary = run_autonomous_recovery(live=bool(args.live))
    result = str(summary.get("run_result") or "BLOCKED")
    print(f"{BLOCK}_{result}")
    print(json.dumps({
        "run_result": summary.get("run_result"),
        "full_live_run_started": summary.get("full_live_run_started"),
        "provider_model_call_made_during_local_phase": summary.get("provider_model_call_made_during_local_phase"),
        "provider_model_call_made_during_live_phase": summary.get("provider_model_call_made_during_live_phase"),
        "docs_loaded": summary.get("docs_loaded"),
        "docs_completed": summary.get("docs_completed"),
        "docs_failed_for_review": summary.get("docs_failed_for_review"),
        "docs_unattempted": summary.get("docs_unattempted"),
        "failure_stage": summary.get("failure_stage"),
        "failure_category": summary.get("failure_category"),
        "actual_total_token_count": summary.get("actual_total_token_count"),
        "privacy_result": summary.get("privacy_result"),
        "safety_result": summary.get("safety_result"),
        "live_gate_environment_active_after_run": summary.get("live_gate_environment_active_after_run"),
        "active_mkb_write": summary.get("active_mkb_write"),
    }, indent=2, sort_keys=True))
    return 0 if result in {"PASS", "LIVE_FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
