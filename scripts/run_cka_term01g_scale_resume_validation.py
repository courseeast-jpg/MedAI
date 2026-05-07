"""Validate CKA-TERM-01G synthetic scale/resume harness."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term01g_scale_resume"
REPORT_JSON = REPORT_DIR / "cka_term01g_scale_resume_report.json"
REPORT_MD = REPORT_DIR / "cka_term01g_scale_resume_report.md"
GUIDE_MD = REPORT_DIR / "CKA_TERM01G_SCALE_RESUME_GUIDE.md"


def _run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "passed": proc.returncode == 0, "stdout_tail": proc.stdout[-800:], "stderr_tail": proc.stderr[-800:]}


def _git_staged_has(prefix: str) -> bool:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return False
    return any(line.strip().replace("\\", "/").startswith(prefix) for line in proc.stdout.splitlines())


def _privacy_clean(payload: dict[str, Any]) -> dict[str, Any]:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload

    check = check_public_report_payload(payload)
    return {
        "passed": check.passed,
        "raw_phi_logged_in_public_reports": check.raw_phi_logged_in_public_reports,
        "private_filename_path_leaks": check.private_filename_path_leaks,
        "secret_leaks": check.secret_leaks,
    }


def main() -> int:
    from clinical_knowledge.terminology.import_limits import build_import_limits
    from clinical_knowledge.terminology.import_resume import simulate_chunked_import_with_resume
    from clinical_knowledge.terminology.import_scale import build_scale_fixtures, parse_scale_fixture

    limits = build_import_limits(max_rows_per_file=50, chunk_size=10)
    with tempfile.TemporaryDirectory(prefix="medai_term01g_validation_") as tmp:
        fixtures = build_scale_fixtures(Path(tmp), rows_per_system=120)
        fixture_summaries = [fixture.safe_public_summary() for fixture in fixtures]
        parse_results = [parse_scale_fixture(fixture, max_rows=limits.max_rows_per_file_default) for fixture in fixtures]
        metrics = [
            simulate_chunked_import_with_resume(parse_result, limits=limits, interrupt_after_chunks=2)
            for parse_result in parse_results
        ]

    baseline_validations = {
        "TERM-01B": _run([sys.executable, "scripts/run_cka_term01b_import_planner_validation.py"]),
        "TERM-01C": _run([sys.executable, "scripts/run_cka_term01c_import_executor_validation.py"]),
        "TERM-01D": _run([sys.executable, "scripts/run_cka_term01d_qa_validation.py"]),
        "TERM-01E": _run([sys.executable, "scripts/run_cka_term01e_operator_readiness_ui_validation.py"]),
    }
    payload: dict[str, Any] = {
        "block_id": "CKA-TERM-01G",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term01g_scale_resume_ready",
        "synthetic_scale_fixtures_ready": len(fixture_summaries) == 4,
        "streaming_parser_guard_ready": all(m.streaming_parser_guard_passed for m in metrics),
        "row_cap_enforced": all(m.row_cap_enforced for m in metrics),
        "chunking_verified": all(m.chunking_verified for m in metrics),
        "checkpoint_resume_verified": all(m.resume_performed and m.checkpoints_written > 0 for m in metrics),
        "duplicate_prevention_verified": all(m.duplicate_prevention_passed for m in metrics),
        "fixture_summaries": fixture_summaries,
        "performance_metrics": [m.safe_public_summary() for m in metrics],
        "baseline_validations": {name: {"passed": result["passed"], "returncode": result["returncode"]} for name, result in baseline_validations.items()},
        "no_real_terminology_import_performed": True,
        "no_real_terminology_files_used": True,
        "terminology_data_staged": _git_staged_has("terminology_data/"),
        "data_terminology_staged": _git_staged_has("data/terminology/"),
        "external_api_used": False,
        "external_terminology_api_used": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "next_manual_action": "operator downloads licensed terminology files and creates private LICENSE_ACK_PRIVATE.json",
        "next_code_action_after_manual_files": "CKA-TERM-02 controlled local terminology import",
    }
    privacy = _privacy_clean(payload)
    payload.update(privacy)
    if (
        not payload["synthetic_scale_fixtures_ready"]
        or not payload["streaming_parser_guard_ready"]
        or not payload["row_cap_enforced"]
        or not payload["chunking_verified"]
        or not payload["checkpoint_resume_verified"]
        or not payload["duplicate_prevention_verified"]
        or payload["terminology_data_staged"]
        or payload["data_terminology_staged"]
        or not all(result["passed"] for result in baseline_validations.values())
        or not payload["passed"]
    ):
        payload["conclusion"] = "cka_term01g_scale_resume_blocked"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    GUIDE_MD.write_text(_guide(), encoding="utf-8")
    print(json.dumps({"conclusion": payload["conclusion"], "systems_tested": [m["system"] for m in payload["performance_metrics"]]}, indent=2))
    return 0 if payload["conclusion"] == "cka_term01g_scale_resume_ready" else 1


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CKA-TERM-01G Scale Resume Report",
        "",
        f"Conclusion: `{payload['conclusion']}`",
        "",
        "## Summary",
        f"- Synthetic scale fixtures ready: `{payload['synthetic_scale_fixtures_ready']}`",
        f"- Streaming parser guard ready: `{payload['streaming_parser_guard_ready']}`",
        f"- Row cap enforced: `{payload['row_cap_enforced']}`",
        f"- Chunking verified: `{payload['chunking_verified']}`",
        f"- Checkpoint resume verified: `{payload['checkpoint_resume_verified']}`",
        f"- Duplicate prevention verified: `{payload['duplicate_prevention_verified']}`",
        "",
        "## Metrics",
    ]
    for metric in payload["performance_metrics"]:
        lines.append(
            f"- {metric['system']}: rows_seen `{metric['rows_seen']}`, rows_imported `{metric['rows_imported']}`, chunks `{metric['chunks_processed']}`, elapsed bucket `{metric['elapsed_seconds_safe_bucket']}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            f"- No real terminology import performed: `{payload['no_real_terminology_import_performed']}`",
            f"- No real terminology files used: `{payload['no_real_terminology_files_used']}`",
            f"- Terminology data staged: `{payload['terminology_data_staged']}`",
            f"- Data terminology staged: `{payload['data_terminology_staged']}`",
            f"- External API used: `{payload['external_api_used']}`",
            "",
            "## Next Action",
            payload["next_manual_action"],
        ]
    )
    return "\n".join(lines) + "\n"


def _guide() -> str:
    return (
        "# CKA-TERM-01G Scale Resume Guide\n\n"
        "This block uses synthetic temporary terminology fixtures only. It checks row caps, chunking, checkpoint resume, duplicate prevention, and parser streaming behavior before any future TERM-02 real import.\n\n"
        "Run `python scripts/cka_terminology_synthetic_scale_test.py` for a local synthetic-only scale check.\n\n"
        "No real terminology data is required or imported.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
