"""Run CKA-TERM-02 controlled local terminology import validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "cka_term02_controlled_local_import"
REPORT_JSON = REPORT_DIR / "cka_term02_controlled_local_import_report.json"
REPORT_MD = REPORT_DIR / "cka_term02_controlled_local_import_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term02_controlled_import import run_controlled_local_import

    result = run_controlled_local_import(repo_root=ROOT)
    payload: dict[str, Any] = {
        "block_id": "CKA-TERM-02",
        "phase_name": "Controlled Local Terminology Import",
        "timestamp": datetime.now(UTC).isoformat(),
        **result.safe_public_summary(),
        "validation_commands": [
            "python -m pytest tests/test_cka_term02_controlled_local_import.py",
            "python scripts/run_cka_term02_controlled_local_import_validation.py",
            "python scripts/cka_term02_preflight_gate.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
        "next_recommended_action": "Run terminology QA and keep imported DB/index private and gitignored.",
    }
    check = check_public_report_payload(payload)
    payload.update(
        {
            "privacy_report_clean": check.passed,
            "raw_phi_logged_in_public_reports": check.raw_phi_logged_in_public_reports,
            "private_filename_path_leaks": check.private_filename_path_leaks,
            "secret_leaks": check.secret_leaks,
        }
    )
    if not check.passed:
        payload["conclusion"] = "cka_term02_controlled_local_import_blocked"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"conclusion": payload["conclusion"], "imported_systems": payload["imported_systems"]}, indent=2))
    return 0 if payload["conclusion"] == "cka_term02_controlled_local_import_ready" else 1


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CKA-TERM-02 Controlled Local Import Report",
        "",
        f"Conclusion: `{payload['conclusion']}`",
        "",
        "## Imported Systems",
    ]
    for summary in payload["file_summaries"]:
        lines.append(
            f"- {summary['system']}: rows seen `{summary['rows_seen']}`, imported `{summary['records_imported']}`, skipped `{summary['records_skipped']}`, chunks `{summary['chunks_processed']}`"
        )
    lines.extend(
        [
            "",
            "## Lookup Validation",
        ]
    )
    for key, value in payload["lookup_validation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Safety",
            f"- Preflight allowed: `{payload['preflight_allowed']}`",
            f"- TERM-02 completed: `{payload['term02_completed']}`",
            f"- Real import performed: `{payload['real_import_performed']}`",
            f"- External API used: `{payload['external_api_used']}`",
            f"- Terminology data staged: `{payload['terminology_data_staged']}`",
            f"- Data terminology staged: `{payload['data_terminology_staged']}`",
            f"- Private ack staged: `{payload['license_ack_private_staged']}`",
            f"- Privacy report clean: `{payload['privacy_report_clean']}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
