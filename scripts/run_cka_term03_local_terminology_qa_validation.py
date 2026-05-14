"""Run CKA-TERM-03 local terminology QA validation."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term03_local_terminology_qa"
REPORT_JSON = REPORT_DIR / "cka_term03_local_terminology_qa_report.json"
REPORT_MD = REPORT_DIR / "cka_term03_local_terminology_qa_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term03_local_qa import Term03QABlocked, run_local_terminology_qa

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = run_local_terminology_qa(repo_root=ROOT)
        payload = {
            "block_id": "CKA-TERM-03",
            "phase_name": "Local Terminology QA / Lookup Regression Expansion",
            "timestamp": datetime.now(UTC).isoformat(),
            **result.safe_public_summary(),
            "recommended_next_action": "Use TERM-03 QA as a regression gate before any terminology-backed clinical integration.",
            "validation_commands": [
                "python -m pytest tests/test_cka_term03_local_terminology_qa.py",
                "python scripts/run_cka_term03_local_terminology_qa_validation.py",
                "python scripts/run_cka_term02_controlled_local_import_validation.py",
                "python scripts/cka_term02_preflight_gate.py",
                "python scripts/run_cka_final_mvp_release_validation.py",
            ],
        }
    except Term03QABlocked as exc:
        payload = {
            "block_id": "CKA-TERM-03",
            "phase_name": "Local Terminology QA / Lookup Regression Expansion",
            "timestamp": datetime.now(UTC).isoformat(),
            "conclusion": "cka_term03_local_terminology_qa_blocked",
            "blocked_reason": str(exc),
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": False,
            "private_filename_path_leaks": 0,
            "secret_leaks": 0,
        }

    privacy = check_public_report_payload(payload)
    payload["privacy_report_clean"] = privacy.passed
    payload["raw_phi_logged_in_public_reports"] = privacy.raw_phi_logged_in_public_reports
    payload["private_filename_path_leaks"] = privacy.private_filename_path_leaks
    payload["secret_leaks"] = privacy.secret_leaks
    if not privacy.passed:
        payload["conclusion"] = "cka_term03_local_terminology_qa_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "qa_case_counts": payload.get("qa_case_counts", {}),
        "source_systems_detected": payload.get("source_systems_detected", []),
    }, indent=2))
    return 0 if payload["conclusion"] == "cka_term03_local_terminology_qa_ready" else 1


def _render_markdown(payload: dict) -> str:
    counts = payload.get("qa_case_counts", {})
    lines = [
        "# CKA-TERM-03 Local Terminology QA Report",
        "",
        f"Conclusion: `{payload.get('conclusion')}`",
        "",
        "## Store",
        f"- Store available: `{payload.get('store_available')}`",
        f"- Read-only mode: `{payload.get('read_only_mode')}`",
        f"- Source systems detected: `{', '.join(payload.get('source_systems_detected', []))}`",
        f"- Aggregate concept count: `{payload.get('store_summary', {}).get('concepts_count')}`",
        "",
        "## QA Counts",
        f"- Total cases: `{counts.get('total', 0)}`",
        f"- Passed: `{counts.get('passed', 0)}`",
        f"- Failed: `{counts.get('failed', 0)}`",
        f"- Skipped: `{counts.get('skipped', 0)}`",
        "",
        "## Behavior Checks",
        f"- Unknown remains unmapped: `{payload.get('unknown_unmapped_passed')}`",
        f"- Ambiguous remains manual-review: `{payload.get('ambiguous_manual_review_passed')}`",
        f"- Determinism passed: `{payload.get('determinism_passed')}`",
        f"- Source filter isolation passed: `{payload.get('source_filter_isolation_passed')}`",
        f"- Code lookup passed: `{payload.get('code_lookup_passed')}`",
        f"- Synonym/alias supported by imported fields: `{payload.get('synonym_alias_supported')}`",
        "",
        "## Safety",
        f"- External API used: `{payload.get('external_api_used')}`",
        f"- Clinical recommendations generated: `{payload.get('clinical_recommendations_generated')}`",
        f"- Prescription dosing advice generated: `{payload.get('prescription_dosing_advice_generated')}`",
        f"- Coding promotes hypothesis: `{payload.get('coding_promotes_hypothesis')}`",
        f"- Coding clears DDI status: `{payload.get('coding_clears_ddi_status')}`",
        f"- No code hallucinated: `{payload.get('no_code_hallucinated')}`",
        f"- Terminology data staged: `{payload.get('terminology_data_staged')}`",
        f"- Data terminology staged: `{payload.get('data_terminology_staged')}`",
        f"- Private ack staged: `{payload.get('license_ack_private_staged')}`",
        f"- Privacy report clean: `{payload.get('privacy_report_clean')}`",
        "",
        "## Next Action",
        str(payload.get("recommended_next_action", "Keep TERM-03 QA as a read-only regression gate.")),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
