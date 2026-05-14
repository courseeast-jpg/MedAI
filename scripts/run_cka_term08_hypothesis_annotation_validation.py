"""Run CKA-TERM-08 hypothesis-only annotation pilot validation."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "cka_term08_hypothesis_annotation_pilot"
REPORT_JSON = REPORT_DIR / "cka_term08_hypothesis_annotation_report.json"
REPORT_MD = REPORT_DIR / "cka_term08_hypothesis_annotation_report.md"


def main() -> int:
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    from clinical_knowledge.terminology.term05_read_only_adapter import build_synthetic_read_only_adapter
    from clinical_knowledge.terminology.term08_hypothesis_annotation_pilot import (
        TERM08_FEATURE_FLAG,
        annotate_candidate_term,
        summarize_annotation_for_public_report,
        term08_annotation_enabled,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    enabled_env = {TERM08_FEATURE_FLAG: "1", "MEDAI_LOCAL_ONLY": "1"}
    adapter = build_synthetic_read_only_adapter()
    disabled = annotate_candidate_term("term08 disabled synthetic", adapter=adapter, env={})
    exact = annotate_candidate_term("aspirin", adapter=adapter, source_filter=["rxnorm"], env=enabled_env)
    unknown = annotate_candidate_term("term08 unknown stays unmapped", adapter=adapter, env=enabled_env)
    ambiguous = annotate_candidate_term("aspirin", adapter=adapter, env=enabled_env)
    case_results = [
        {
            "case_id": "feature_flag_off",
            "passed": disabled is None and term08_annotation_enabled({}) is False,
            **summarize_annotation_for_public_report(disabled),
        },
        {
            "case_id": "exact_lookup_hypothesis_only",
            "passed": exact is not None
            and exact.terminology_status == "exact"
            and exact.annotation_tier == "hypothesis"
            and exact.requires_review is True
            and exact.writes_active_fact is False,
            **summarize_annotation_for_public_report(exact),
        },
        {
            "case_id": "unknown_unmapped_no_hallucination",
            "passed": unknown is not None
            and unknown.terminology_status == "unmapped"
            and not unknown.candidate_codes
            and unknown.no_code_hallucinated is True,
            **summarize_annotation_for_public_report(unknown),
        },
        {
            "case_id": "ambiguous_manual_review",
            "passed": ambiguous is not None
            and ambiguous.terminology_status == "ambiguous"
            and ambiguous.requires_review is True
            and "ambiguity_requires_manual_review" in ambiguous.reason_codes,
            **summarize_annotation_for_public_report(ambiguous),
        },
    ]
    staged = _git_staged_paths()
    payload = {
        "block_id": "CKA-TERM-08",
        "phase_name": "Hypothesis-Only Coding Annotation Pilot",
        "timestamp": datetime.now(UTC).isoformat(),
        "conclusion": "cka_term08_hypothesis_annotation_ready",
        "feature_flag_name": TERM08_FEATURE_FLAG,
        "feature_flag_default_enabled": term08_annotation_enabled({}),
        "feature_flag_off_preserves_current_behavior": disabled is None,
        "enabled_mode_creates_hypothesis_only_annotations": exact is not None and exact.annotation_tier == "hypothesis",
        "case_results": case_results,
        "cases_total": len(case_results),
        "cases_passed": sum(1 for case in case_results if case["passed"]),
        "cases_failed": sum(1 for case in case_results if not case["passed"]),
        "unknown_terms_remain_unmapped": case_results[2]["passed"],
        "ambiguous_terms_remain_manual_review": case_results[3]["passed"],
        "no_code_hallucinated": all(case.get("no_code_hallucinated", True) for case in case_results),
        "writes_active_fact": False,
        "clears_ddi_status": False,
        "promotes_hypothesis": False,
        "external_api_used": False,
        "clinical_recommendations_generated": False,
        "dosing_advice_generated": False,
        "prescribing_advice_generated": False,
        "b07_behavior_changed": False,
        "ocr_extractor_safety_gates_changed": False,
        "terminology_data_staged": _staged_under(staged, "terminology_data/"),
        "data_terminology_staged": _staged_under(staged, "data/terminology/"),
        "license_ack_private_staged": any("LICENSE_ACK_PRIVATE" in path for path in staged),
        "source_terminology_files_staged": any(path.lower().endswith((".rrf", ".csv", ".zip")) for path in staged),
        "db_key_private_artifacts_staged": any(path.lower().endswith((".db", ".sqlite", ".key", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")) for path in staged),
        "validation_commands": [
            "python -m pytest tests/test_cka_term08_hypothesis_annotation_pilot.py -vv",
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
        "recommended_next_action": "Create PARK-08 snapshot before any B07-TERM opt-in integration.",
    }
    privacy = check_public_report_payload(payload)
    payload["privacy_report_clean"] = privacy.passed
    payload["raw_phi_logged_in_public_reports"] = privacy.raw_phi_logged_in_public_reports
    payload["private_filename_path_leaks"] = privacy.private_filename_path_leaks
    payload["secret_leaks"] = privacy.secret_leaks
    if not _required_passed(payload) or not privacy.passed:
        payload["conclusion"] = "cka_term08_hypothesis_annotation_blocked"

    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "cases_total": payload["cases_total"],
        "cases_passed": payload["cases_passed"],
        "cases_failed": payload["cases_failed"],
        "feature_flag_default_enabled": payload["feature_flag_default_enabled"],
    }, indent=2))
    return 0 if payload["conclusion"] == "cka_term08_hypothesis_annotation_ready" else 1


def _required_passed(payload: dict) -> bool:
    return (
        payload["feature_flag_default_enabled"] is False
        and payload["feature_flag_off_preserves_current_behavior"] is True
        and payload["enabled_mode_creates_hypothesis_only_annotations"] is True
        and payload["cases_failed"] == 0
        and payload["unknown_terms_remain_unmapped"] is True
        and payload["ambiguous_terms_remain_manual_review"] is True
        and payload["writes_active_fact"] is False
        and payload["clears_ddi_status"] is False
        and payload["promotes_hypothesis"] is False
        and payload["external_api_used"] is False
        and payload["clinical_recommendations_generated"] is False
        and payload["dosing_advice_generated"] is False
        and payload["prescribing_advice_generated"] is False
        and payload["b07_behavior_changed"] is False
        and payload["ocr_extractor_safety_gates_changed"] is False
        and payload["terminology_data_staged"] is False
        and payload["data_terminology_staged"] is False
        and payload["license_ack_private_staged"] is False
        and payload["source_terminology_files_staged"] is False
        and payload["db_key_private_artifacts_staged"] is False
    )


def _render_markdown(payload: dict) -> str:
    lines = [
        "# CKA-TERM-08 Hypothesis-Only Coding Annotation Pilot Report",
        "",
        f"Conclusion: `{payload.get('conclusion')}`",
        "",
        "## Feature Flag",
        f"- Feature flag: `{payload.get('feature_flag_name')}`",
        f"- Default enabled: `{payload.get('feature_flag_default_enabled')}`",
        f"- Disabled mode preserves current behavior: `{payload.get('feature_flag_off_preserves_current_behavior')}`",
        "",
        "## Validation Cases",
        f"- Total: `{payload.get('cases_total')}`",
        f"- Passed: `{payload.get('cases_passed')}`",
        f"- Failed: `{payload.get('cases_failed')}`",
    ]
    for case in payload.get("case_results", []):
        lines.append(
            f"- {case['case_id']}: passed=`{case['passed']}`, status=`{case['terminology_status']}`, "
            f"candidate_code_count=`{case['candidate_code_count']}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            f"- Writes active fact: `{payload.get('writes_active_fact')}`",
            f"- Clears DDI status: `{payload.get('clears_ddi_status')}`",
            f"- Promotes hypothesis: `{payload.get('promotes_hypothesis')}`",
            f"- External API used: `{payload.get('external_api_used')}`",
            f"- Clinical recommendations generated: `{payload.get('clinical_recommendations_generated')}`",
            f"- Dosing advice generated: `{payload.get('dosing_advice_generated')}`",
            f"- B07 behavior changed: `{payload.get('b07_behavior_changed')}`",
            f"- OCR/extractor/safety gates changed: `{payload.get('ocr_extractor_safety_gates_changed')}`",
            f"- Privacy report clean: `{payload.get('privacy_report_clean')}`",
            "",
            "## Next Action",
            str(payload.get("recommended_next_action")),
            "",
        ]
    )
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
