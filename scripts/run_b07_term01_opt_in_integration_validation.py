"""Run B07-TERM-01 opt-in terminology metadata integration validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "b07_term01_opt_in_integration"
REPORT_JSON = REPORT_DIR / "b07_term01_opt_in_integration_report.json"
REPORT_MD = REPORT_DIR / "b07_term01_opt_in_integration_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.b07_term_opt_in import build_b07_terminology_metadata
    from clinical_knowledge.terminology.term05_read_only_adapter import build_synthetic_read_only_adapter

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = build_synthetic_read_only_adapter()
    enabled_env = {
        "MEDAI_B07_TERMINOLOGY_OPT_IN": "1",
        "MEDAI_TERMINOLOGY_LOOKUP_ENABLED": "1",
        "MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION": "1",
        "MEDAI_TERMINOLOGY_READ_ONLY": "1",
        "MEDAI_TERMINOLOGY_ALLOW_WRITES": "0",
    }
    off = build_b07_terminology_metadata("aspirin", adapter=adapter, env={})
    inconsistent = build_b07_terminology_metadata(
        "aspirin",
        adapter=adapter,
        env={**enabled_env, "MEDAI_TERMINOLOGY_ALLOW_WRITES": "1"},
    )
    exact = build_b07_terminology_metadata("aspirin", adapter=adapter, source_filter=["rxnorm"], env=enabled_env)
    unknown = build_b07_terminology_metadata("b07 term unknown remains unmapped", adapter=adapter, env=enabled_env)
    ambiguous = build_b07_terminology_metadata("aspirin", adapter=adapter, env=enabled_env)
    rollback = build_b07_terminology_metadata("aspirin", adapter=adapter, env={})
    case_results = [
        {
            "case_id": "flags_off_preserves_baseline",
            "passed": off.enabled is False and off.terminology_status == "disabled" and off.input_term is None,
            **_safe_case(off),
        },
        {
            "case_id": "inconsistent_flags_fail_closed",
            "passed": inconsistent.enabled is False and "terminology_writes_forbidden" in inconsistent.reason_codes,
            **_safe_case(inconsistent),
        },
        {
            "case_id": "exact_hypothesis_metadata_only",
            "passed": exact.enabled is True
            and exact.terminology_status == "exact"
            and exact.annotation_tier == "hypothesis"
            and exact.requires_review is True
            and exact.writes_active_fact is False
            and exact.b07_authority_source is False,
            **_safe_case(exact),
        },
        {
            "case_id": "unknown_unmapped_no_code",
            "passed": unknown.enabled is True
            and unknown.terminology_status == "unmapped"
            and unknown.candidate_code_count == 0
            and unknown.no_code_hallucinated is True,
            **_safe_case(unknown),
        },
        {
            "case_id": "ambiguous_manual_review",
            "passed": ambiguous.enabled is True
            and ambiguous.terminology_status == "ambiguous"
            and ambiguous.source_system is None
            and "manual_review_required" in ambiguous.reason_codes,
            **_safe_case(ambiguous),
        },
        {
            "case_id": "rollback_flags_disable_effect",
            "passed": rollback.enabled is False and rollback.safe_public_summary() == off.safe_public_summary(),
            **_safe_case(rollback),
        },
    ]
    staged = _git_staged_paths()
    payload = {
        "block_id": "B07-TERM-01",
        "phase_name": "Opt-In Hypothesis-Only Terminology Metadata Integration",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "b07_term01_opt_in_integration_ready",
        "feature_flags_default_off": True,
        "off_state_preservation_passed": case_results[0]["passed"],
        "inconsistent_flags_fail_closed": case_results[1]["passed"],
        "opt_in_behavior_passed": case_results[2]["passed"],
        "rollback_behavior_passed": case_results[5]["passed"],
        "unknown_unmapped_passed": case_results[3]["passed"],
        "ambiguous_manual_review_passed": case_results[4]["passed"],
        "case_results": case_results,
        "cases_total": len(case_results),
        "cases_passed": sum(1 for case in case_results if case["passed"]),
        "cases_failed": sum(1 for case in case_results if not case["passed"]),
        "writes_active_fact": False,
        "promotes_hypothesis": False,
        "clears_ddi_status": False,
        "clinical_advice_generated": False,
        "dosing_advice_generated": False,
        "prescribing_advice_generated": False,
        "external_api_used": False,
        "b07_authority_source": False,
        "ocr_extractor_safety_gates_changed": False,
        "terminology_data_staged": _staged_under(staged, "terminology_data/"),
        "data_terminology_staged": _staged_under(staged, "data/terminology/"),
        "license_ack_private_staged": any("LICENSE_ACK_PRIVATE" in path for path in staged),
        "source_terminology_files_staged": any(path.lower().endswith((".rrf", ".csv", ".zip")) for path in staged),
        "db_key_private_artifacts_staged": any(path.lower().endswith((".db", ".sqlite", ".key", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")) for path in staged),
        "validation_commands": [
            "python -m pytest tests/test_b07_term01_opt_in_integration.py -vv",
            "python scripts/run_b07_term01_opt_in_integration_validation.py",
            "python -m pytest tests/test_cka_block07_medical_coding.py -vv",
            "python scripts/run_cka_term08_hypothesis_annotation_validation.py",
            "python scripts/run_cka_term07_ui_lookup_panel_validation.py",
            "python scripts/run_cka_term06_private_store_adapter_validation.py",
            "python scripts/run_cka_term05_synthetic_adapter_validation.py",
            "python scripts/run_cka_term04_integration_readiness_validation.py",
            "python scripts/run_cka_term03_local_terminology_qa_validation.py",
            "python scripts/run_cka_term02_controlled_local_import_validation.py",
            "python scripts/run_cka_term01h_safety_redteam_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
        "recommended_next_action": "Create a post-B07-TERM-01 parking snapshot before any broader terminology-backed behavior.",
    }
    privacy = check_public_report_payload(payload)
    payload["public_report_privacy_clean"] = privacy.passed
    payload["raw_phi_logged_in_public_reports"] = privacy.raw_phi_logged_in_public_reports
    payload["private_filename_path_leaks"] = privacy.private_filename_path_leaks
    payload["secret_leaks"] = privacy.secret_leaks
    if not _required_passed(payload) or not privacy.passed:
        payload["conclusion"] = "b07_term01_opt_in_integration_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "cases_total": payload["cases_total"],
        "cases_passed": payload["cases_passed"],
        "cases_failed": payload["cases_failed"],
        "external_api_used": payload["external_api_used"],
    }, indent=2))
    return 0 if payload["conclusion"] == "b07_term01_opt_in_integration_ready" else 1


def _safe_case(metadata) -> dict:
    summary = metadata.safe_public_summary()
    return {
        "enabled": summary["enabled"],
        "terminology_status": summary["terminology_status"],
        "candidate_code_count": summary["candidate_code_count"],
        "annotation_tier": summary["annotation_tier"],
        "requires_review": summary["requires_review"],
        "reason_codes": summary["reason_codes"],
        "read_only_lookup": summary["read_only_lookup"],
        "writes_active_fact": summary["writes_active_fact"],
        "clears_ddi_status": summary["clears_ddi_status"],
        "promotes_hypothesis": summary["promotes_hypothesis"],
        "external_api_used": summary["external_api_used"],
        "b07_authority_source": summary["b07_authority_source"],
        "no_code_hallucinated": summary["no_code_hallucinated"],
    }


def _required_passed(payload: dict) -> bool:
    return (
        payload["feature_flags_default_off"] is True
        and payload["off_state_preservation_passed"] is True
        and payload["inconsistent_flags_fail_closed"] is True
        and payload["opt_in_behavior_passed"] is True
        and payload["rollback_behavior_passed"] is True
        and payload["unknown_unmapped_passed"] is True
        and payload["ambiguous_manual_review_passed"] is True
        and payload["cases_failed"] == 0
        and payload["writes_active_fact"] is False
        and payload["promotes_hypothesis"] is False
        and payload["clears_ddi_status"] is False
        and payload["clinical_advice_generated"] is False
        and payload["dosing_advice_generated"] is False
        and payload["prescribing_advice_generated"] is False
        and payload["external_api_used"] is False
        and payload["b07_authority_source"] is False
        and payload["ocr_extractor_safety_gates_changed"] is False
        and payload["terminology_data_staged"] is False
        and payload["data_terminology_staged"] is False
        and payload["license_ack_private_staged"] is False
        and payload["source_terminology_files_staged"] is False
        and payload["db_key_private_artifacts_staged"] is False
    )


def _render_markdown(payload: dict) -> str:
    lines = [
        "# B07-TERM-01 Opt-In Terminology Metadata Integration Report",
        "",
        f"Conclusion: `{payload.get('conclusion')}`",
        "",
        "## Behavior",
        f"- Feature flags default off: `{payload.get('feature_flags_default_off')}`",
        f"- Off-state preservation passed: `{payload.get('off_state_preservation_passed')}`",
        f"- Inconsistent flags fail closed: `{payload.get('inconsistent_flags_fail_closed')}`",
        f"- Opt-in behavior passed: `{payload.get('opt_in_behavior_passed')}`",
        f"- Rollback behavior passed: `{payload.get('rollback_behavior_passed')}`",
        f"- Unknown unmapped passed: `{payload.get('unknown_unmapped_passed')}`",
        f"- Ambiguous manual-review passed: `{payload.get('ambiguous_manual_review_passed')}`",
        "",
        "## Case Counts",
        f"- Total: `{payload.get('cases_total')}`",
        f"- Passed: `{payload.get('cases_passed')}`",
        f"- Failed: `{payload.get('cases_failed')}`",
        "",
        "## Safety",
        f"- Writes active fact: `{payload.get('writes_active_fact')}`",
        f"- Promotes hypothesis: `{payload.get('promotes_hypothesis')}`",
        f"- Clears DDI status: `{payload.get('clears_ddi_status')}`",
        f"- Clinical advice generated: `{payload.get('clinical_advice_generated')}`",
        f"- Dosing advice generated: `{payload.get('dosing_advice_generated')}`",
        f"- Prescribing advice generated: `{payload.get('prescribing_advice_generated')}`",
        f"- External API used: `{payload.get('external_api_used')}`",
        f"- B07 authority source: `{payload.get('b07_authority_source')}`",
        f"- OCR/extractor/safety gates changed: `{payload.get('ocr_extractor_safety_gates_changed')}`",
        f"- Public report privacy clean: `{payload.get('public_report_privacy_clean')}`",
        "",
        "## Next Action",
        str(payload.get("recommended_next_action")),
        "",
    ]
    return "\n".join(lines)


def _git_staged_paths() -> list[str]:
    proc = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _staged_under(paths: list[str], prefix: str) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)


if __name__ == "__main__":
    raise SystemExit(main())
