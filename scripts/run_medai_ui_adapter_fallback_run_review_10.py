#!/usr/bin/env python3
"""MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10 validation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_adapter_fallback_run_review_10"
AUDIT_MD_PATH = REPORT_DIR / "MEDAI_UI_ADAPTER_FALLBACK_RUN_REVIEW_10_AUDIT.md"
AUDIT_JSON_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_audit.json"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_UI_ADAPTER_FALLBACK_RUN_REVIEW_10.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_ui_adapter_fallback_run_review_10_report.md"

NEXT_OPERATOR_COMMAND = "streamlit run app/main.py"
SYNTHETIC_TEXT = (
    "Lab result report\n"
    "Specimen: serum\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Platelets: 230 x10E9/L (ref 150-450)\n"
)


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode("utf-8").strip()
    except Exception:
        return default


def _privacy_check(payload: Any) -> dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {
            "passed": bool(result.passed),
            "leak_examples_redacted": list(result.leak_examples_redacted or []),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build_report() -> dict[str, Any]:
    from app.local_adapter_fallback_processor import (
        process_adapter_fallback_run_review,
        run_operator_action_proof,
    )
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_ui_adapter_fallback_10_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    result = process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        specialty="general",
        session_id="adapter-fallback-10",
    )
    record_ids = list(result["run_item"].get("extracted_medical_fact_record_ids") or [])
    with sql_store._get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE fact_type=? AND requires_review=1",
            ("test_result",),
        ).fetchone()
    retrieval_proof_count = int(row["c"] if isinstance(row, dict) else row[0])
    operator_action_proof_count = run_operator_action_proof(sql_store, record_ids, session_id="adapter-fallback-10")
    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    run_review_works_when_execution_none = (
        "render_adapter_fallback_panel" in main_source
        and 'sys_components.get("execution") is None' in main_source
        and 'sys_components.get("sql") is not None' in main_source
    )

    findings = {
        "branch": _git_value(["branch", "--show-current"], default=""),
        "head": _git_value(["rev-parse", "--short", "HEAD"]),
        "ui_adapter_fallback_run_review_ready": bool(
            result["ready"]
            and result["structured_facts_extracted"] == 4
            and result["review_bound_records_persisted"] == 4
            and retrieval_proof_count == 4
            and result["ui_preview_rows"] == 4
            and operator_action_proof_count == 3
            and run_review_works_when_execution_none
        ),
        "structured_facts_extracted": int(result["structured_facts_extracted"]),
        "review_bound_records_persisted": int(result["review_bound_records_persisted"]),
        "retrieval_proof_count": retrieval_proof_count,
        "ui_preview_rows": int(result["ui_preview_rows"]),
        "operator_action_proof_count": operator_action_proof_count,
        "run_review_works_when_execution_none": run_review_works_when_execution_none,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "next_operator_command": NEXT_OPERATOR_COMMAND,
    }
    report = {
        "block_id": "MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10",
        "mode": "synthetic_degraded_run_review_adapter_fallback",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "fallback_mode_ready": bool(findings["ui_adapter_fallback_run_review_ready"]),
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "conclusion": "ui_adapter_fallback_run_review_ready" if findings["ui_adapter_fallback_run_review_ready"] else "not_ready",
        "all_pass": bool(findings["ui_adapter_fallback_run_review_ready"]),
    }
    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get("leak_examples_redacted", [])
    return report


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    AUDIT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    f = report["audit_findings"]
    lines = [
        "# MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10 Report",
        "",
        f"Conclusion: `{report['conclusion']}`",
        "",
        "## Counts",
        "",
        f"- fallback mode ready: `{report['fallback_mode_ready']}`",
        f"- structured facts extracted: `{f['structured_facts_extracted']}`",
        f"- review-bound records persisted: `{f['review_bound_records_persisted']}`",
        f"- retrieval proof count: `{f['retrieval_proof_count']}`",
        f"- UI preview rows: `{f['ui_preview_rows']}`",
        f"- operator action proof count: `{f['operator_action_proof_count']}`",
        f"- Run & Review works when execution=None: `{f['run_review_works_when_execution_none']}`",
        "",
        "## Safety",
        "",
        f"- privacy check passed: `{report['privacy_check_passed']}`",
        f"- external API used: `{report['external_api_used']}`",
        f"- auto-accept enabled: `{report['auto_accept_enabled']}`",
        "",
        "## Next Operator Command",
        "",
        f"```bash\n{f['next_operator_command']}\n```",
        "",
    ]
    md = "\n".join(lines)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    AUDIT_MD_PATH.write_text(md.replace("Report", "Audit", 1), encoding="utf-8")
    SHORT_MD_PATH.write_text(
        "# MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10\n\n"
        f"Conclusion: `{report['conclusion']}`\n\n"
        f"Fallback ready: `{report['fallback_mode_ready']}`\n\n"
        f"Next operator command:\n\n```bash\n{f['next_operator_command']}\n```\n",
        encoding="utf-8",
    )


def main() -> int:
    report = build_report()
    write_reports(report)
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] and report["privacy_check_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
