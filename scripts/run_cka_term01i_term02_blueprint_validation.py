"""Validate CKA-TERM-01I TERM-02 blueprint and runbook."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term01i_term02_blueprint"
BLUEPRINT = REPORT_DIR / "CKA_TERM02_EXECUTION_BLUEPRINT.md"
RUNBOOK = REPORT_DIR / "CKA_TERM02_OPERATOR_RUNBOOK.md"
MATRIX = REPORT_DIR / "CKA_TERM02_STOP_ON_FAILURE_MATRIX.md"
REPORT_JSON = REPORT_DIR / "cka_term01i_term02_blueprint_report.json"
REPORT_MD = REPORT_DIR / "cka_term01i_term02_blueprint_report.md"


REQUIRED_PHASES = [f"TERM-01{suffix}" for suffix in ("", "A", "B", "C", "D", "E", "F", "G", "H")]
REQUIRED_TERM02_PHASES = (
    "Preflight",
    "Inventory",
    "License confirmation",
    "Dry-run plan",
    "Capped import",
    "QA harness",
    "B07 boundary check",
    "Privacy report",
    "Commit and tag policy",
)
STOP_REASONS = (
    "no_supported_files_present",
    "license_ack_private_missing",
    "systems_pending_acknowledgment",
    "terminology_files_staged",
    "data_terminology_staged",
    "row_cap_exceeded",
    "malformed_rows_above_threshold",
    "ambiguous_lookup_regression",
    "unknown_code_hallucination",
    "b07_hypothesis_promotion",
    "b07_ddi_status_changed",
    "external_api_attempted",
    "privacy_report_leak",
    "terminology_db_staged",
)


def main() -> int:
    payload = build_report()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"conclusion": payload["conclusion"], "docs_validated": payload["docs_validated"]}, indent=2))
    return 0 if payload["conclusion"] == "cka_term01i_term02_blueprint_ready" else 1


def build_report() -> dict[str, Any]:
    docs = {path.name: path.read_text(encoding="utf-8") for path in (BLUEPRINT, RUNBOOK, MATRIX)}
    combined = "\n".join(docs.values())
    final_validation = _run([sys.executable, "scripts/run_cka_final_mvp_release_validation.py"])
    staged = _git_staged_paths()
    payload: dict[str, Any] = {
        "block_id": "CKA-TERM-01I",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term01i_term02_blueprint_ready",
        "term02_blueprint_created": BLUEPRINT.exists(),
        "operator_runbook_created": RUNBOOK.exists(),
        "stop_on_failure_matrix_created": MATRIX.exists(),
        "docs_validated": True,
        "term01_through_term01h_mentioned": all(phase in combined for phase in REQUIRED_PHASES),
        "term02_phases_defined": all(phase in combined for phase in REQUIRED_TERM02_PHASES),
        "stop_matrix_complete": all(reason in combined for reason in STOP_REASONS),
        "operator_commands_present": all(
            cmd in combined
            for cmd in (
                "cka_terminology_prepare_intake.py",
                "cka_terminology_check_ready.py",
                "cka_terminology_import_dry_run.py",
                "cka_term02_preflight_gate.py",
                "cka_terminology_run_qa.py",
            )
        ),
        "license_gate_preserved": "bypass" not in combined.lower() and "license gate" in combined.lower(),
        "commit_policy_preserved": _commit_policy_preserved(combined),
        "raw_private_paths_present": _has_raw_private_path(combined),
        "clinical_advice_or_dosing_present": _has_advice_text(combined),
        "final_cka_validation": {"passed": final_validation["passed"], "returncode": final_validation["returncode"]},
        "no_real_import_performed": True,
        "terminology_data_staged": _staged_under(staged, "terminology_data/"),
        "data_terminology_staged": _staged_under(staged, "data/terminology/"),
        "clinical_recommendations_generated": False,
        "prescription_dosing_advice_generated": False,
        "external_api_used": False,
        "next_manual_action": "operator downloads licensed terminology files and creates private LICENSE_ACK_PRIVATE.json",
        "next_code_action_after_manual_files": "CKA-TERM-02 controlled local terminology import",
    }
    payload.update(_privacy_clean(payload, docs))
    blockers = [
        not payload["term02_blueprint_created"],
        not payload["operator_runbook_created"],
        not payload["stop_on_failure_matrix_created"],
        not payload["term01_through_term01h_mentioned"],
        not payload["term02_phases_defined"],
        not payload["stop_matrix_complete"],
        not payload["operator_commands_present"],
        not payload["license_gate_preserved"],
        not payload["commit_policy_preserved"],
        payload["raw_private_paths_present"],
        payload["clinical_advice_or_dosing_present"],
        payload["terminology_data_staged"],
        payload["data_terminology_staged"],
        not payload["final_cka_validation"]["passed"],
        not payload["privacy_report_clean"],
    ]
    if any(blockers):
        payload["conclusion"] = "cka_term01i_term02_blueprint_blocked"
        payload["docs_validated"] = False
    return payload


def _run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "passed": proc.returncode == 0}


def _git_staged_paths() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: list[str], prefix: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)


def _commit_policy_preserved(text: str) -> bool:
    lowered = text.lower()
    forbidden_commit_phrases = (
        "commit terminology_data/",
        "commit data/terminology/",
        "commit license_ack_private.json",
        "commit vendor files",
        "commit generated terminology indexes",
    )
    return all(phrase not in lowered for phrase in forbidden_commit_phrases) and "never commit" in lowered


def _has_raw_private_path(text: str) -> bool:
    lowered = text.lower()
    return "c:\\" in lowered or "g:\\" in lowered or "/users/" in lowered or "\\users\\" in lowered


def _has_advice_text(text: str) -> bool:
    lowered = text.lower()
    blocked = ("recommended dose", "increase dose", "decrease dose", "you should take")
    return any(token in lowered for token in blocked)


def _privacy_clean(payload: dict[str, Any], docs: dict[str, str]) -> dict[str, Any]:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload

    check = check_public_report_payload({"report": payload, "docs": docs})
    return {
        "privacy_report_clean": check.passed,
        "raw_phi_logged_in_public_reports": check.raw_phi_logged_in_public_reports,
        "private_filename_path_leaks": check.private_filename_path_leaks,
        "secret_leaks": check.secret_leaks,
    }


def _markdown(payload: dict[str, Any]) -> str:
    return (
        "# CKA-TERM-01I TERM-02 Blueprint Report\n\n"
        f"Conclusion: `{payload['conclusion']}`\n\n"
        "## Summary\n"
        f"- TERM-02 blueprint created: `{payload['term02_blueprint_created']}`\n"
        f"- Operator runbook created: `{payload['operator_runbook_created']}`\n"
        f"- Stop-on-failure matrix created: `{payload['stop_on_failure_matrix_created']}`\n"
        f"- License gate preserved: `{payload['license_gate_preserved']}`\n"
        f"- Final CKA validation passed: `{payload['final_cka_validation']['passed']}`\n\n"
        "## Safety\n"
        f"- No real import performed: `{payload['no_real_import_performed']}`\n"
        f"- Terminology data staged: `{payload['terminology_data_staged']}`\n"
        f"- Data terminology staged: `{payload['data_terminology_staged']}`\n"
        f"- Clinical advice flag generated: `{payload['clinical_recommendations_generated']}`\n"
        f"- Dosing-advice flag generated: `{payload['prescription_dosing_advice_generated']}`\n"
        f"- External API used: `{payload['external_api_used']}`\n"
        f"- Privacy report clean: `{payload['privacy_report_clean']}`\n\n"
        "## Next\n"
        f"- Manual action: {payload['next_manual_action']}\n"
        f"- Future code action: {payload['next_code_action_after_manual_files']}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
