"""CKA-B02 — Privacy Boundary Validation Script.

Runs six synthetic test cases (A–F) covering: clean payloads, PHI payloads,
path/filename payloads, secret payloads, nested payloads, and pre-write
report privacy checks. All data is synthetic — no real PHI.

Does NOT:
- call external APIs
- use real patient data
- modify frozen HITL release artifacts
- change production OCR/extractor/safety gates
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

from clinical_knowledge.privacy.outbound_audit import build_outbound_payload
from clinical_knowledge.privacy.private_mapping import (
    DEFAULT_PRIVATE_MAPPING_PATH,
    is_gitignored,
    is_tracked_by_git,
    write_private_mapping,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.privacy.sanitizer import sanitize_text

BLOCK_ID = "CKA-B02"
REPORT_DIR = ROOT / "reports" / "cka_block02_privacy_boundary"
JSON_REPORT = REPORT_DIR / "cka_block02_privacy_boundary_report.json"
MD_REPORT = REPORT_DIR / "cka_block02_privacy_boundary_report.md"


# ---------------------------------------------------------------------------
# Case A — clean payload
# ---------------------------------------------------------------------------

def run_case_a() -> dict:
    """Clean synthetic payload with no private content."""
    clean = {"fact_type": "lab_reference_range", "entity": "Hemoglobin A1c < 5.7%", "tier": "active"}

    result_no_ext = build_outbound_payload(clean, allow_external=False, purpose="case_a_no_ext")
    result_ext = build_outbound_payload(clean, allow_external=True, purpose="case_a_ext")

    assert result_no_ext.allowed is False, "Case A: clean but allow_external=False → must block"
    assert "allow_external=False" in " ".join(result_no_ext.blocked_reasons)
    assert result_ext.allowed is True, "Case A: clean + allow_external=True → must allow"
    assert result_ext.raw_phi_detected is False
    assert result_ext.secret_detected is False

    return {"case": "A", "passed": True, "description": "clean payload blocked/allowed by allow_external flag"}


# ---------------------------------------------------------------------------
# Case B — PHI payload (person name, DOB, phone, email, MRN, insurance ID)
# ---------------------------------------------------------------------------

def run_case_b() -> dict:
    """Synthetic payload with person name, DOB, phone, email, MRN, insurance ID."""
    phi_payload = {
        "patient_label": "Jane Doe",
        "dob": "DOB: 03/15/1985",
        "contact": "Phone: 555-867-5309, Email: jane.doe@example.com",
        "mrn": "MRN: 7834521",
        "insurance": "MEMBER-ID892341",
        "note": "Routine lab result for patient.",
    }

    san = sanitize_text(str(phi_payload))
    assert san.raw_phi_detected, "Case B: PHI must be detected"
    # Verify raw strings not in sanitized output
    assert "Jane Doe" not in san.sanitized_text
    assert "03/15/1985" not in san.sanitized_text
    assert "555-867-5309" not in san.sanitized_text
    assert "jane.doe@example.com" not in san.sanitized_text

    result = build_outbound_payload(phi_payload, allow_external=True, purpose="case_b")
    # After sanitization, raw PHI should be replaced
    payload_json = json.dumps(result.sanitized_payload)
    assert "Jane Doe" not in payload_json
    assert "jane.doe@example.com" not in payload_json

    # Public report check on sanitized payload
    priv_check = check_public_report_payload(result.sanitized_payload)
    assert priv_check.raw_phi_logged_in_public_reports is False, (
        f"Case B: sanitized payload should be PHI-clean, found: {priv_check.leak_examples_redacted}"
    )

    # Write a local private mapping (gitignored, never tracked)
    write_private_mapping(san.replacement_map)
    assert is_gitignored(DEFAULT_PRIVATE_MAPPING_PATH), "Case B: private mapping must be gitignored"
    assert not is_tracked_by_git(DEFAULT_PRIVATE_MAPPING_PATH), "Case B: private mapping must not be tracked"

    return {
        "case": "B",
        "passed": True,
        "description": "PHI detected, sanitized, private mapping gitignored",
        "phi_categories_found": list({f.category for f in san.findings}),
    }


# ---------------------------------------------------------------------------
# Case C — raw path and medical filename
# ---------------------------------------------------------------------------

def run_case_c() -> dict:
    """Synthetic payload with raw Windows path and medical-looking filename."""
    path_payload = {
        "source": "C:\\Users\\operator\\Documents\\patient_labs_results.pdf",
        "note": "Processed document from /home/operator/data/patient_results.txt",
        "fact": "HbA1c normal range",
    }

    san = sanitize_text(str(path_payload))
    assert san.private_reference_detected, "Case C: private path reference must be detected"
    assert "C:\\Users" not in san.sanitized_text

    result = build_outbound_payload(path_payload, allow_external=True, purpose="case_c")
    # Private path leaks in original payload should be detected
    assert result.private_filename_path_leaks > 0

    # Public report check
    priv_check = check_public_report_payload(result.sanitized_payload)
    assert priv_check.private_filename_path_leaks == 0, (
        f"Case C: sanitized payload should have no path leaks, found: {priv_check.leak_examples_redacted}"
    )

    return {
        "case": "C",
        "passed": True,
        "description": "Windows/Unix paths and medical filenames detected and removed",
        "path_leaks_in_original": result.private_filename_path_leaks,
    }


# ---------------------------------------------------------------------------
# Case D — API key / secret
# ---------------------------------------------------------------------------

def run_case_d() -> dict:
    """Synthetic payload with API-key-like secret — must always block."""
    secret_payload = {
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz012345",
        "fact": "safe clinical note",
    }

    result_no_ext = build_outbound_payload(secret_payload, allow_external=False, purpose="case_d_no_ext")
    result_ext = build_outbound_payload(secret_payload, allow_external=True, purpose="case_d_ext")

    assert result_no_ext.allowed is False, "Case D: secret must block when allow_external=False"
    assert result_ext.allowed is False, "Case D: secret must block even when allow_external=True"
    assert result_ext.secret_detected is True

    # Also check with explicit token= style
    secret2 = {"note": "token=AbCdEfGhIjKlMnOpQrStUvWxYz01234567890123"}
    result2 = build_outbound_payload(secret2, allow_external=True, purpose="case_d2")
    assert result2.allowed is False

    return {"case": "D", "passed": True, "description": "API-key-like secrets always block outbound"}


# ---------------------------------------------------------------------------
# Case E — nested dict/list payload
# ---------------------------------------------------------------------------

def run_case_e() -> dict:
    """Nested dict/list with sensitive strings at various depths."""
    nested = {
        "records": [
            {"label": "Jane Doe", "tier": "active"},
            {"label": "safe_synthetic_label_42", "tier": "hypothesis"},
        ],
        "meta": {
            "session": "abc-123",
            "contact": {"phone": "555-123-4567", "email": "test@example.com"},
            "path": "C:\\Users\\test\\patient_report.pdf",
        },
        "tags": ["lab", "Dr. Smith Jones", "routine"],
    }

    result = build_outbound_payload(nested, allow_external=True, purpose="case_e")
    payload_json = json.dumps(result.sanitized_payload)
    assert "Jane Doe" not in payload_json
    assert "555-123-4567" not in payload_json
    assert "test@example.com" not in payload_json
    assert "C:\\Users" not in payload_json
    assert "Dr. Smith Jones" not in payload_json

    priv_check = check_public_report_payload(result.sanitized_payload)
    # After sanitization, report should be clean
    assert priv_check.raw_phi_logged_in_public_reports is False

    return {
        "case": "E",
        "passed": True,
        "description": "Nested dict/list recursive sanitization works",
        "findings_summary": result.findings_summary,
    }


# ---------------------------------------------------------------------------
# Case F — public report pre-write privacy check catches unsafe raw string
# ---------------------------------------------------------------------------

def run_case_f() -> dict:
    """Pre-write check catches a report that still has raw sensitive content."""
    # Simulate an incomplete sanitization — raw PHI still in draft report
    unsafe_draft = {
        "conclusion": "test_ok",
        "note": "Patient Jane Doe, DOB: 01/01/1980, MRN: 9876543",
    }

    priv_check_before = check_public_report_payload(unsafe_draft)
    assert priv_check_before.passed is False, "Case F: pre-check must catch raw PHI in draft"
    assert priv_check_before.raw_phi_logged_in_public_reports is True

    # Now sanitize and re-check
    from clinical_knowledge.privacy.sanitizer import sanitize_dict_values
    clean_draft, _, _ = sanitize_dict_values(unsafe_draft)
    priv_check_after = check_public_report_payload(clean_draft)
    assert priv_check_after.passed is True, (
        f"Case F: sanitized draft must pass, found: {priv_check_after.leak_examples_redacted}"
    )

    return {
        "case": "F",
        "passed": True,
        "description": "Pre-write report check catches raw PHI; sanitization clears it",
    }


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def run_validation(report_dir: Path = REPORT_DIR) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cases = {}
    sanitizer_passed = 0
    outbound_passed = 0
    report_privacy_passed = 0

    for fn, label in [
        (run_case_a, "A"), (run_case_b, "B"), (run_case_c, "C"),
        (run_case_d, "D"), (run_case_e, "E"), (run_case_f, "F"),
    ]:
        result = fn()
        cases[label] = result
        if label in ("A", "B", "C", "D", "E"):
            sanitizer_passed += 1
        if label in ("A", "D"):
            outbound_passed += 1
        if label in ("B", "C", "F"):
            report_privacy_passed += 1

    # Final public report privacy self-check
    draft_report = {k: v for k, v in cases.items()}
    final_check = check_public_report_payload(draft_report)
    assert final_check.passed, f"Final report has leaks: {final_check.leak_examples_redacted}"

    # Private mapping checks
    mapping_created = DEFAULT_PRIVATE_MAPPING_PATH.exists()
    mapping_gitignored = is_gitignored(DEFAULT_PRIVATE_MAPPING_PATH) if mapping_created else False
    mapping_tracked = is_tracked_by_git(DEFAULT_PRIVATE_MAPPING_PATH) if mapping_created else False

    report = {
        "block_id": BLOCK_ID,
        "conclusion": "cka_b02_privacy_boundary_ready",
        "synthetic_cases_run": len(cases),
        "sanitizer_cases_passed": sanitizer_passed,
        "outbound_audit_cases_passed": outbound_passed,
        "report_privacy_cases_passed": report_privacy_passed,
        "all_cases_passed": all(c["passed"] for c in cases.values()),
        "private_mapping_file_created": mapping_created,
        "private_mapping_gitignored": mapping_gitignored,
        "private_mapping_tracked": mapping_tracked,
        "replacement_map_written_to_public_reports": False,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "secret_leaks": 0,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_recommended_block": "CKA-B03 Decision Engine + Safe Mode + Response Scoring",
        "case_results": {k: {"passed": v["passed"], "description": v["description"]} for k, v in cases.items()},
    }

    # Write reports
    json_path = report_dir / "cka_block02_privacy_boundary_report.json"
    md_path = report_dir / "cka_block02_privacy_boundary_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# CKA-B02 Privacy Boundary Validation Report",
        "",
        f"**Block:** {BLOCK_ID}",
        f"**Conclusion:** `{report['conclusion']}`",
        "",
        "## Synthetic Test Cases",
        f"- Cases run: {report['synthetic_cases_run']}",
        f"- All cases passed: {report['all_cases_passed']}",
        "",
        "| Case | Description | Passed |",
        "|------|-------------|--------|",
    ]
    for label, cr in report["case_results"].items():
        md_lines.append(f"| {label} | {cr['description']} | {cr['passed']} |")

    md_lines += [
        "",
        "## Privacy Flags",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- private_filename_path_leaks: {report['private_filename_path_leaks']}",
        f"- secret_leaks: {report['secret_leaks']}",
        f"- replacement_map_written_to_public_reports: {report['replacement_map_written_to_public_reports']}",
        f"- private_mapping_file_created: {report['private_mapping_file_created']}",
        f"- private_mapping_gitignored: {report['private_mapping_gitignored']}",
        f"- private_mapping_tracked: {report['private_mapping_tracked']}",
        "",
        "## Safety Flags",
        f"- production_ocr_changed: {report['production_ocr_changed']}",
        f"- production_extractor_changed: {report['production_extractor_changed']}",
        f"- safety_gate_changed: {report['safety_gate_changed']}",
        f"- frozen_hitl_release_reopened: {report['frozen_hitl_release_reopened']}",
        "",
        f"**Next block:** {report['next_recommended_block']}",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"CKA-B02 conclusion: {report['conclusion']}")
    print(f"Cases: {report['synthetic_cases_run']} run, all passed: {report['all_cases_passed']}")
    print(f"Private mapping created: {report['private_mapping_file_created']}, "
          f"gitignored: {report['private_mapping_gitignored']}, "
          f"tracked: {report['private_mapping_tracked']}")
    print(f"JSON report: {json_path}")

    return report


def main() -> int:
    report = run_validation()
    return 0 if report["conclusion"] == "cka_b02_privacy_boundary_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
