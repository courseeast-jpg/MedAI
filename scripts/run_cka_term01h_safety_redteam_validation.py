"""Validate CKA-TERM-01H terminology safety red-team pack."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term01h_safety_redteam"
REPORT_JSON = REPORT_DIR / "cka_term01h_safety_redteam_report.json"
REPORT_MD = REPORT_DIR / "cka_term01h_safety_redteam_report.md"
GUIDE_MD = REPORT_DIR / "CKA_TERM01H_TERMINOLOGY_SAFETY_REDTEAM_GUIDE.md"


def _run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-800:],
        "stderr_tail": proc.stderr[-800:],
    }


def _privacy_clean(payload: dict[str, Any]) -> dict[str, Any]:
    from clinical_knowledge.terminology.privacy_regression import assert_public_report_safe

    return assert_public_report_safe(payload)


def main() -> int:
    from clinical_knowledge.terminology.safety_redteam import run_terminology_safety_redteam

    redteam = run_terminology_safety_redteam().safe_public_summary()
    baseline_validations = {
        "TERM-01G": _run([sys.executable, "scripts/run_cka_term01g_scale_resume_validation.py"]),
        "TERM-01F": _run([sys.executable, "scripts/run_cka_term01f_manual_return_pack_validation.py"]),
        "TERM-01": _run([sys.executable, "scripts/run_cka_term01_real_terminology_readiness_validation.py"]),
        "FINAL_CKA": _run([sys.executable, "scripts/run_cka_final_mvp_release_validation.py"]),
    }
    payload: dict[str, Any] = {
        **redteam,
        "timestamp": datetime.now(UTC).isoformat(),
        "baseline_validations": {
            name: {"passed": result["passed"], "returncode": result["returncode"]}
            for name, result in baseline_validations.items()
        },
        "validation_commands": [
            "python -m pytest tests/test_cka_term01h_safety_redteam.py",
            "python scripts/run_cka_term01h_safety_redteam_validation.py",
            "python scripts/run_cka_term01g_scale_resume_validation.py",
            "python scripts/run_cka_term01f_manual_return_pack_validation.py",
            "python scripts/run_cka_term01_real_terminology_readiness_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
    }
    privacy = _privacy_clean(payload)
    payload.update(privacy)
    if payload["conclusion"] != "cka_term01h_safety_redteam_ready" or not all(
        result["passed"] for result in baseline_validations.values()
    ):
        payload["conclusion"] = "cka_term01h_safety_redteam_blocked"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    GUIDE_MD.write_text(_guide(), encoding="utf-8")
    print(json.dumps({"conclusion": payload["conclusion"], "scenarios": len(payload["scenario_results"])}, indent=2))
    return 0 if payload["conclusion"] == "cka_term01h_safety_redteam_ready" else 1


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CKA-TERM-01H Safety Red-Team Report",
        "",
        f"Conclusion: `{payload['conclusion']}`",
        "",
        "## Blocked Or Flagged Scenarios",
    ]
    for key in (
        "raw_path_leak_blocked",
        "license_text_leak_blocked",
        "fake_ack_blocked",
        "ack_mismatch_blocked",
        "terminology_data_staging_detected",
        "data_terminology_staging_detected",
        "zip_slip_blocked",
        "malformed_rows_skipped",
        "csv_formula_injection_neutralized",
        "ambiguity_not_silently_resolved",
        "unknown_code_not_hallucinated",
        "b07_hypothesis_promotion_blocked",
        "b07_ddi_clear_blocked",
        "external_api_blocked",
        "clinical_advice_absent",
    ):
        lines.append(f"- {key}: `{payload[key]}`")
    lines.extend(
        [
            "",
            "## Safety",
            f"- No real import performed: `{payload['no_real_import_performed']}`",
            f"- Real terminology files committed: `{payload['real_terminology_files_committed']}`",
            f"- External API used: `{payload['external_api_used']}`",
            f"- Raw PHI logged in public reports: `{payload['raw_phi_logged_in_public_reports']}`",
            f"- Private filename/path leaks: `{payload['private_filename_path_leaks']}`",
            f"- Secret leaks: `{payload['secret_leaks']}`",
            f"- License text written to public reports: `{payload['license_text_written_to_public_reports']}`",
            f"- Clinical advice flag generated: `{payload['clinical_recommendations_generated']}`",
            f"- Dosing-advice flag generated: `{payload['prescription_dosing_advice_generated']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _guide() -> str:
    return (
        "# CKA-TERM-01H Terminology Safety Red-Team Guide\n\n"
        "This pack is synthetic-only. It validates terminology privacy, license, staging, parser, lookup, and B07 boundary protections before any real licensed terminology files are provided.\n\n"
        "Run `python scripts/run_cka_term01h_safety_redteam_validation.py` to regenerate the public safety report.\n\n"
        "Keep real terminology files, private acknowledgments, database files, and medical artifacts outside this public report folder.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
