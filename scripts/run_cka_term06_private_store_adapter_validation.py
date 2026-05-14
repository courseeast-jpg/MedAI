"""Run CKA-TERM-06 private-store read-only adapter validation."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term06_private_store_adapter_validation"
REPORT_JSON = REPORT_DIR / "cka_term06_private_store_adapter_validation_report.json"
REPORT_MD = REPORT_DIR / "cka_term06_private_store_adapter_validation_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term06_private_store_adapter_validation import (
        Term06ValidationBlocked,
        run_private_store_adapter_validation,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = run_private_store_adapter_validation(repo_root=ROOT)
        payload = {
            "block_id": "CKA-TERM-06",
            "phase_name": "Private-Store Read-Only Adapter Validation",
            "timestamp": datetime.now(UTC).isoformat(),
            **result.safe_public_summary(),
            "clinical_logic_changed": False,
            "ocr_extractor_safety_gates_changed": False,
            "source_terminology_files_staged": False,
            "recommended_next_action": "Proceed to TERM-07 UI-only terminology lookup panel only after explicit approval.",
            "validation_commands": [
                "python -m pytest tests/test_cka_term06_private_store_adapter_validation.py -vv",
                "python scripts/run_cka_term06_private_store_adapter_validation.py",
                "python scripts/run_cka_term05_synthetic_adapter_validation.py",
                "python scripts/run_cka_term04_integration_readiness_validation.py",
                "python scripts/run_cka_term03_local_terminology_qa_validation.py",
                "python scripts/run_cka_term02_controlled_local_import_validation.py",
                "python scripts/run_cka_term01h_safety_redteam_validation.py",
                "python scripts/run_cka_final_mvp_release_validation.py",
            ],
        }
    except Term06ValidationBlocked as exc:
        payload = {
            "block_id": "CKA-TERM-06",
            "phase_name": "Private-Store Read-Only Adapter Validation",
            "timestamp": datetime.now(UTC).isoformat(),
            "conclusion": "cka_term06_private_store_adapter_validation_blocked",
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
        payload["conclusion"] = "cka_term06_private_store_adapter_validation_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "qa_case_counts": payload.get("qa_case_counts", {}),
        "source_systems_detected": payload.get("source_systems_detected", []),
    }, indent=2))
    return 0 if payload["conclusion"] == "cka_term06_private_store_adapter_validation_ready" else 1


def _render_markdown(payload: dict) -> str:
    counts = payload.get("qa_case_counts", {})
    lines = [
        "# CKA-TERM-06 Private-Store Read-Only Adapter Validation Report",
        "",
        f"Conclusion: `{payload.get('conclusion')}`",
        "",
        "## Store",
        f"- Store available: `{payload.get('store_available')}`",
        f"- Store opened read-only: `{payload.get('store_opened_read_only')}`",
        f"- Write attempt blocked: `{payload.get('write_attempt_blocked')}`",
        f"- Source systems detected: `{', '.join(payload.get('source_systems_detected', []))}`",
        f"- Aggregate concept count: `{payload.get('store_summary', {}).get('concepts_count')}`",
        "",
        "## QA Counts",
        f"- Total cases: `{counts.get('total', 0)}`",
        f"- Passed: `{counts.get('passed', 0)}`",
        f"- Failed: `{counts.get('failed', 0)}`",
        f"- Skipped: `{counts.get('skipped', 0)}`",
        "",
        "## Adapter Checks",
        f"- Exact rxnorm lookup passed: `{payload.get('exact_rxnorm_passed')}`",
        f"- Exact loinc lookup passed: `{payload.get('exact_loinc_passed')}`",
        f"- Code lookup passed: `{payload.get('code_lookup_passed')}`",
        f"- Source filter isolation passed: `{payload.get('source_filter_isolation_passed')}`",
        f"- Unknown unmapped passed: `{payload.get('unknown_unmapped_passed')}`",
        f"- Ambiguous manual-review passed: `{payload.get('ambiguous_manual_review_passed')}`",
        f"- Determinism passed: `{payload.get('determinism_passed')}`",
        f"- Normalization passed: `{payload.get('normalization_passed')}`",
        "",
        "## Safety",
        f"- Real import performed: `{payload.get('real_import_performed')}`",
        f"- Store recreated: `{payload.get('store_recreated')}`",
        f"- External API used: `{payload.get('external_api_used')}`",
        f"- Clinical advice generated: `{payload.get('clinical_advice_generated')}`",
        f"- Dosing advice generated: `{payload.get('dosing_advice_generated')}`",
        f"- MKB write performed: `{payload.get('mkb_write_performed')}`",
        f"- Automatic annotation created: `{payload.get('automatic_annotation_created')}`",
        f"- B07 integrated: `{payload.get('b07_integrated')}`",
        f"- DDI status cleared: `{payload.get('ddi_status_cleared')}`",
        f"- Hypothesis promoted: `{payload.get('hypothesis_promoted')}`",
        f"- No code hallucinated: `{payload.get('no_code_hallucinated')}`",
        f"- Privacy report clean: `{payload.get('privacy_report_clean')}`",
        "",
        "## Next Action",
        str(payload.get("recommended_next_action", "Keep TERM-06 as a read-only validation gate.")),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
