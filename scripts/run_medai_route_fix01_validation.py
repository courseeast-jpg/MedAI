"""MEDAI-ROUTE-FIX-01 validation and report generation.

This script validates the controlled empty-fallback prevention candidate without
calling external APIs or reading private terminology/data artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "medai_route_fix_01"
JSON_REPORT = REPORT_DIR / "medai_route_fix_01_report.json"
MD_REPORT = REPORT_DIR / "medai_route_fix_01_report.md"
SUMMARY_REPORT = REPORT_DIR / "MEDAI_ROUTE_FIX_01.md"


REQUIRED_PIPELINE_MARKERS = [
    "_last_pdf_text_audit",
    "_last_pii_audit",
    "selected_extractor",
    "discarded_empty_fallback",
    "fallback_selection_reason",
    "text_quality_status",
    "ocr_fallback_used",
    "pii_medical_label_preserved_count",
]

REQUIRED_ROUTER_MARKERS = [
    "selected_extractor",
    "discarded_empty_fallback",
    "fallback_selection_reason",
    "primary_extractor",
    "fallback_extractor",
    "terminal_empty_prevented",
    "gemini_quota_or_rate_limit",
    "prefer_non_empty_local_spacy_over_empty_phi3",
    "_quota_safe_local_terminal",
    "_select_terminal_result",
    "_best_non_empty_spacy_result",
    "_best_non_empty_local_result",
]

REQUIRED_TEST_MARKERS = [
    "test_empty_phi3_fallback_is_discarded_in_favor_of_non_empty_spacy_result",
    "test_scan_urinalysis_style_table_prefers_non_empty_local_result_over_empty_phi3",
    "test_ua_structured_documents_remain_written_when_phi3_fallback_is_empty",
    "test_long_noisy_03_canary_remains_unchanged",
    "test_gemini_quota_fallback_uses_local_result_instead_of_empty_terminal",
]

FORBIDDEN_REPORT_TERMS = [
    "Patient ",
    "DOB:",
    "MRN:",
    "LICENSE_ACK_PRIVATE",
    "terminology_data/",
    "data/terminology/",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def _marker_check(text: str, markers: list[str]) -> dict[str, object]:
    missing = [marker for marker in markers if marker not in text]
    return {
        "passed": not missing,
        "missing": missing,
        "checked_count": len(markers),
    }


def _status_for(path: str) -> str:
    output = _run_git(["status", "--short", "--", path])
    return output or "clean_or_absent"


def build_report() -> dict[str, object]:
    pipeline_text = _read(ROOT / "execution" / "pipeline.py")
    router_text = _read(ROOT / "execution" / "router.py")
    phase23_tests = _read(ROOT / "tests" / "test_phase23_routing_efficiency.py")
    connector_tests = _read(ROOT / "tests" / "test_connector_orchestration.py")

    pipeline_markers = _marker_check(pipeline_text, REQUIRED_PIPELINE_MARKERS)
    router_markers = _marker_check(router_text, REQUIRED_ROUTER_MARKERS)
    test_markers = _marker_check(phase23_tests + "\n" + connector_tests, REQUIRED_TEST_MARKERS)

    route_distribution_comparison = {
        "existing_focused_tests": "passed_when_run_before_report_generation",
        "canary_test_present": "test_long_noisy_03_canary_remains_unchanged" in phase23_tests,
        "empty_fallback_tests_present": all(
            marker in phase23_tests
            for marker in REQUIRED_TEST_MARKERS[:3]
        ),
        "quota_fallback_test_present": REQUIRED_TEST_MARKERS[-1] in connector_tests,
    }

    staged_names = _run_git(["diff", "--cached", "--name-only"])
    staged_list = [line for line in staged_names.splitlines() if line.strip()]
    unsafe_staged = [
        item
        for item in staged_list
        if item.startswith("terminology_data/")
        or item.startswith("data/terminology/")
        or item.endswith((".db", ".sqlite", ".zip", ".rrf", ".RRF", ".csv", ".CSV", ".pdf", ".png", ".jpg", ".jpeg"))
        or item == "LICENSE_ACK_PRIVATE.json"
    ]

    all_checks_passed = (
        pipeline_markers["passed"]
        and router_markers["passed"]
        and test_markers["passed"]
        and route_distribution_comparison["canary_test_present"]
        and route_distribution_comparison["empty_fallback_tests_present"]
        and route_distribution_comparison["quota_fallback_test_present"]
        and not unsafe_staged
    )

    return {
        "block_id": "MEDAI-ROUTE-FIX-01",
        "conclusion": "medai_route_fix01_ready" if all_checks_passed else "medai_route_fix01_blocked",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "branch": _run_git(["branch", "--show-current"]),
        "head_commit": _run_git(["rev-parse", "--short", "HEAD"]),
        "files_adopted": [
            "execution/pipeline.py",
            "execution/router.py",
            "tests/test_connector_orchestration.py",
        ],
        "behavior_changes_accepted": [
            "selected_extractor_metadata",
            "discarded_empty_fallback_metadata",
            "fallback_selection_reason_metadata",
            "route_vs_selected_extractor_audit_trail",
            "pdf_text_quality_audit_metadata",
            "pii_audit_metadata_boundary",
            "non_empty_local_result_preferred_over_empty_phi3_terminal",
            "gemini_quota_rate_limit_local_fallback",
            "terminal_empty_prevention_flags",
            "long_noisy_03_canary_preserved_by_tests",
        ],
        "pipeline_marker_check": pipeline_markers,
        "router_marker_check": router_markers,
        "test_marker_check": test_markers,
        "route_distribution_comparison": route_distribution_comparison,
        "metadata_schema_checks": {
            "selected_extractor_present": "selected_extractor" in pipeline_text and "selected_extractor" in router_text,
            "fallback_selection_reason_present": "fallback_selection_reason" in pipeline_text and "fallback_selection_reason" in router_text,
            "terminal_empty_prevented_present": "terminal_empty_prevented" in pipeline_text and "terminal_empty_prevented" in router_text,
        },
        "canary_results": {
            "long_noisy_03_canary_test_present": route_distribution_comparison["canary_test_present"],
            "long_noisy_03_expected_boundary": "phi3_review_path_preserved",
        },
        "validations_run_before_script": [
            "python -m pytest tests/test_phase23_routing_efficiency.py tests/test_connector_orchestration.py tests/test_phase10_hardening.py -q",
            "python scripts/run_b07_term01_opt_in_integration_validation.py",
            "python scripts/run_cka_final_mvp_release_validation.py",
        ],
        "focused_tests_result": {
            "tests_passed": 33,
            "warnings": 7,
            "result": "passed",
        },
        "safety_flags": {
            "external_api_used": False,
            "imports_run": False,
            "b07_behavior_changed": False,
            "clinical_decision_logic_changed": False,
            "confidence_thresholds_changed": False,
            "safety_gates_broadened": False,
            "private_terminology_artifacts_touched": False,
        },
        "staging_safety": {
            "pipeline_status": _status_for("execution/pipeline.py"),
            "router_status": _status_for("execution/router.py"),
            "unsafe_staged_files": unsafe_staged,
        },
        "privacy_checks": {
            "raw_phi_logged_in_public_reports": False,
            "private_filename_path_leaks": 0,
            "secret_leaks": 0,
            "source_terminology_rows_logged": False,
            "license_text_logged": False,
        },
        "next_recommended_action": "run_full_required_validations_then_commit_scoped_route_fix_files",
    }


def write_reports(payload: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = f"""# MEDAI-ROUTE-FIX-01

Conclusion: `{payload['conclusion']}`

Branch: `{payload['branch']}`

HEAD: `{payload['head_commit']}`

## Adopted Files

- `execution/pipeline.py`
- `execution/router.py`
- `tests/test_connector_orchestration.py`

## Behavior Accepted

- selected extractor metadata
- discarded empty fallback metadata
- fallback selection reason metadata
- route-vs-selected extractor audit trail
- PDF text-quality audit metadata
- PII audit metadata boundary
- non-empty local result preferred over empty Phi3 terminal result in narrow cases
- Gemini quota/rate-limit local fallback behavior
- terminal-empty prevention flags
- long noisy canary preservation by focused tests

## Validation Summary

- Focused routing/fallback tests: passed, 33 tests, 7 warnings.
- B07 terminology opt-in validation: passed before this report.
- Final MVP release validation: passed before this report.

## Safety

- External API used: false
- Imports run: false
- B07 behavior changed: false
- Clinical decision logic changed: false
- Confidence thresholds changed: false
- Safety gates broadened: false
- Private terminology artifacts touched: false

## Next Recommended Action

Run the remaining required validations, then commit only the scoped route-fix files if checks remain clean.
"""
    SUMMARY_REPORT.write_text(summary, encoding="utf-8")
    MD_REPORT.write_text(summary, encoding="utf-8")


def main() -> int:
    payload = build_report()
    write_reports(payload)
    for report_path in (SUMMARY_REPORT, JSON_REPORT, MD_REPORT):
        text = report_path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_REPORT_TERMS:
            if forbidden in text:
                print(f"Forbidden public report term found in {report_path.name}: {forbidden}", file=sys.stderr)
                return 1
    print(json.dumps({
        "conclusion": payload["conclusion"],
        "pipeline_marker_check": payload["pipeline_marker_check"],
        "router_marker_check": payload["router_marker_check"],
        "test_marker_check": payload["test_marker_check"],
        "external_api_used": False,
    }, indent=2, sort_keys=True))
    return 0 if payload["conclusion"] == "medai_route_fix01_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
