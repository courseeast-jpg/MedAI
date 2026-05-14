"""Run CKA-TERM-05 synthetic read-only terminology adapter validation."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term05_synthetic_adapter"
REPORT_JSON = REPORT_DIR / "cka_term05_synthetic_adapter_report.json"
REPORT_MD = REPORT_DIR / "cka_term05_synthetic_adapter_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term05_read_only_adapter import run_term05_synthetic_adapter_validation

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_term05_synthetic_adapter_validation()
    payload = {
        "block_id": "CKA-TERM-05",
        "phase_name": "Synthetic Read-Only Terminology Adapter",
        "timestamp": datetime.now(UTC).isoformat(),
        **result.safe_public_summary(),
        "terminology_data_staged": False,
        "data_terminology_staged": False,
        "license_ack_private_staged": False,
        "source_terminology_files_staged": False,
        "clinical_logic_changed": False,
        "ocr_extractor_safety_gates_changed": False,
        "recommended_next_action": "Proceed to TERM-06 private-store read-only adapter validation only after explicit approval.",
        "validation_commands": [
            "python -m pytest tests/test_cka_term05_synthetic_adapter.py -vv",
            "python scripts/run_cka_term05_synthetic_adapter_validation.py",
            "python scripts/run_cka_term04_integration_readiness_validation.py",
            "python scripts/run_cka_term03_local_terminology_qa_validation.py",
            "python scripts/run_cka_term01h_safety_redteam_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
    }
    privacy = check_public_report_payload(payload)
    payload["privacy_report_clean"] = privacy.passed
    payload["raw_phi_logged_in_public_reports"] = privacy.raw_phi_logged_in_public_reports
    payload["private_filename_path_leaks"] = privacy.private_filename_path_leaks
    payload["secret_leaks"] = privacy.secret_leaks
    if not privacy.passed:
        payload["conclusion"] = "cka_term05_synthetic_adapter_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "cases_total": payload["cases_total"],
        "cases_passed": payload["cases_passed"],
        "cases_failed": payload["cases_failed"],
    }, indent=2))
    return 0 if payload["conclusion"] == "cka_term05_synthetic_adapter_ready" else 1


def _render_markdown(payload: dict) -> str:
    lines = [
        "# CKA-TERM-05 Synthetic Read-Only Terminology Adapter Report",
        "",
        f"Conclusion: `{payload.get('conclusion')}`",
        "",
        "## Adapter Behavior",
        f"- Synthetic only: `{payload.get('synthetic_only')}`",
        f"- Read-only: `{payload.get('read_only')}`",
        f"- Exact rxnorm lookup passed: `{payload.get('exact_rxnorm_passed')}`",
        f"- Exact loinc lookup passed: `{payload.get('exact_loinc_passed')}`",
        f"- Code lookup passed: `{payload.get('code_lookup_passed')}`",
        f"- Source filter isolation passed: `{payload.get('source_filter_isolation_passed')}`",
        f"- Unknown unmapped passed: `{payload.get('unknown_unmapped_passed')}`",
        f"- Ambiguous manual-review passed: `{payload.get('ambiguous_manual_review_passed')}`",
        f"- Determinism passed: `{payload.get('determinism_passed')}`",
        f"- Normalization passed: `{payload.get('normalization_passed')}`",
        "",
        "## Case Counts",
        f"- Total: `{payload.get('cases_total')}`",
        f"- Passed: `{payload.get('cases_passed')}`",
        f"- Failed: `{payload.get('cases_failed')}`",
        "",
        "## Safety",
        f"- External API used: `{payload.get('external_api_used')}`",
        f"- Private store accessed: `{payload.get('private_store_accessed')}`",
        f"- Terminology data accessed: `{payload.get('terminology_data_accessed')}`",
        f"- Data terminology accessed: `{payload.get('data_terminology_accessed')}`",
        f"- MKB write performed: `{payload.get('mkb_write_performed')}`",
        f"- B07 integrated: `{payload.get('b07_integrated')}`",
        f"- DDI status cleared: `{payload.get('ddi_status_cleared')}`",
        f"- Hypothesis promoted: `{payload.get('hypothesis_promoted')}`",
        f"- No code hallucinated: `{payload.get('no_code_hallucinated')}`",
        f"- Privacy report clean: `{payload.get('privacy_report_clean')}`",
        "",
        "## Next Action",
        str(payload.get("recommended_next_action")),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
