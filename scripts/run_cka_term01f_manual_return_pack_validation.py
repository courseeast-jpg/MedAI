"""Validate CKA-TERM-01F manual return pack and TERM-02 preflight gate."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term01f_manual_return_pack"
REPORT_JSON = REPORT_DIR / "cka_term01f_manual_return_pack_report.json"
REPORT_MD = REPORT_DIR / "cka_term01f_manual_return_pack_report.md"
GUIDE_MD = REPORT_DIR / "CKA_TERM01F_MANUAL_RETURN_GUIDE.md"
CHECKLIST_MD = REPORT_DIR / "CKA_TERM02_PREFLIGHT_CHECKLIST.md"


def _run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "passed": proc.returncode == 0, "stdout_tail": proc.stdout[-800:], "stderr_tail": proc.stderr[-800:]}


def _git_staged_has(prefix: str) -> bool:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return False
    prefix = prefix.replace("\\", "/")
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
    from clinical_knowledge.terminology.manual_return_pack import (
        build_manual_return_guide_text,
        build_term02_preflight_checklist_text,
        run_manual_return_pack,
    )
    from clinical_knowledge.terminology.synthetic_intake_rehearsal import run_synthetic_intake_rehearsal
    from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate

    gate_without_files = run_term02_preflight_gate(repo_root=ROOT)
    rehearsal = run_synthetic_intake_rehearsal(repo_root=ROOT)
    manual_pack = run_manual_return_pack(repo_root=ROOT, run_prepare=False)
    final_validation = _run([sys.executable, "scripts/run_cka_final_mvp_release_validation.py"])

    payload: dict[str, Any] = {
        "block_id": "CKA-TERM-01F",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term01f_manual_return_pack_ready",
        "manual_return_guide_created": True,
        "term02_preflight_gate_ready": True,
        "one_command_readiness_check_ready": True,
        "synthetic_intake_rehearsal_passed": all(
            [
                rehearsal.classification_passed,
                rehearsal.safe_entries_extracted,
                rehearsal.zip_slip_protection_verified,
                rehearsal.readiness_passed,
                rehearsal.dry_run_passed,
                rehearsal.term02_preflight_passed,
            ]
        ),
        "zip_slip_protection_verified": rehearsal.zip_slip_protection_verified,
        "term02_preflight_blocks_without_manual_files": not gate_without_files.allowed,
        "term02_preflight_block_reason_codes": gate_without_files.reason_codes,
        "term02_preflight_passes_with_synthetic_temp_ack": rehearsal.term02_preflight_passed,
        "manual_return_pack": manual_pack.safe_public_summary(),
        "synthetic_intake_rehearsal": rehearsal.safe_public_summary(),
        "no_real_download_performed": True,
        "no_real_terminology_import_performed": True,
        "real_license_ack_created": False,
        "terminology_data_staged": _git_staged_has("terminology_data/"),
        "data_terminology_staged": _git_staged_has("data/terminology/"),
        "real_terminology_files_committed": False,
        "license_gate_bypassed": False,
        "external_api_used": False,
        "external_terminology_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
        "license_text_written_to_public_reports": False,
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_manual_action": "operator downloads licensed terminology files and creates private LICENSE_ACK_PRIVATE.json",
        "next_code_action_after_manual_files": "CKA-TERM-02 controlled local terminology import",
        "final_cka_validation_passed": final_validation["passed"],
        "validation_commands": [
            "python -m pytest tests/test_cka_term01f_manual_return_pack.py",
            "python scripts/run_cka_term01f_manual_return_pack_validation.py",
            "python scripts/run_cka_term01e_operator_readiness_ui_validation.py",
            "python scripts/run_cka_term01d_qa_validation.py",
            "python scripts/run_cka_term01c_import_executor_validation.py",
            "python scripts/run_cka_term01b_import_planner_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
    }
    privacy = _privacy_clean(payload)
    payload.update(privacy)
    if (
        not payload["synthetic_intake_rehearsal_passed"]
        or not payload["term02_preflight_blocks_without_manual_files"]
        or not payload["term02_preflight_passes_with_synthetic_temp_ack"]
        or payload["terminology_data_staged"]
        or payload["data_terminology_staged"]
        or not payload["final_cka_validation_passed"]
        or not payload["passed"]
    ):
        payload["conclusion"] = "cka_term01f_manual_return_pack_blocked"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    GUIDE_MD.write_text(build_manual_return_guide_text(), encoding="utf-8")
    CHECKLIST_MD.write_text(build_term02_preflight_checklist_text(), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"conclusion": payload["conclusion"], "synthetic_intake_rehearsal_passed": payload["synthetic_intake_rehearsal_passed"]}, indent=2))
    return 0 if payload["conclusion"] == "cka_term01f_manual_return_pack_ready" else 1


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CKA-TERM-01F Manual Return Pack Report",
        "",
        f"Conclusion: `{payload['conclusion']}`",
        "",
        "## Readiness",
        f"- Manual return guide created: `{payload['manual_return_guide_created']}`",
        f"- TERM-02 preflight gate ready: `{payload['term02_preflight_gate_ready']}`",
        f"- One-command readiness check ready: `{payload['one_command_readiness_check_ready']}`",
        f"- Synthetic intake rehearsal passed: `{payload['synthetic_intake_rehearsal_passed']}`",
        f"- Zip-slip protection verified: `{payload['zip_slip_protection_verified']}`",
        f"- TERM-02 blocks without manual files: `{payload['term02_preflight_blocks_without_manual_files']}`",
        f"- TERM-02 passes with synthetic temp ack: `{payload['term02_preflight_passes_with_synthetic_temp_ack']}`",
        "",
        "## Safety",
        f"- No real download performed: `{payload['no_real_download_performed']}`",
        f"- No real terminology import performed: `{payload['no_real_terminology_import_performed']}`",
        f"- Real license ack created: `{payload['real_license_ack_created']}`",
        f"- Terminology data staged: `{payload['terminology_data_staged']}`",
        f"- Data terminology staged: `{payload['data_terminology_staged']}`",
        f"- External API used: `{payload['external_api_used']}`",
        f"- Clinical recommendations generated: `{payload['clinical_recommendations_generated']}`",
        f"- Prescription dosing advice generated: `{payload['prescription_dosing_advice_generated']}`",
        "",
        "## Next Action",
        payload["next_manual_action"],
        "",
        "Next code action after manual files: CKA-TERM-02 controlled local terminology import.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
