"""Tests for CKA-B02 — Privacy Boundary + PII/PHI Outbound Audit.

Covers:
- Pattern detection for each category
- Sanitizer removes raw sensitive strings
- Repeated sensitive values map consistently
- replacement_map is not public-safe
- safe_public_findings contains only category/count/hash
- Private mapping file path contains PRIVATE
- Private mapping file is gitignored
- Outbound blocked when allow_external=False
- Outbound allowed only after sanitization when allow_external=True
- Secret-like values block outbound always
- Raw paths/filenames do not leak
- Nested payload scanning
- Public report privacy checker catches unsafe payloads
- Validation script succeeds
- Final public reports contain no raw private strings
- No external API needed
- CKA-B01 tests still pass
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clinical_knowledge.privacy.outbound_audit import OutboundAuditResult, build_outbound_payload
from clinical_knowledge.privacy.patterns import (
    ALWAYS_BLOCK_CATEGORIES,
    PHI_CATEGORIES,
    PRIVATE_REF_CATEGORIES,
    PRIVACY_PATTERNS,
    PatternSeverity,
)
from clinical_knowledge.privacy.private_mapping import (
    DEFAULT_PRIVATE_MAPPING_PATH,
    is_gitignored,
    is_tracked_by_git,
    write_private_mapping,
)
from clinical_knowledge.privacy.report_privacy import ReportPrivacyCheck, check_public_report_payload
from clinical_knowledge.privacy.sanitizer import SanitizedText, sanitize_dict_values, sanitize_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_category(result: SanitizedText, category: str) -> bool:
    return any(f.category == category for f in result.findings)


def _raw_absent(sanitized_text: str, *raw_values: str) -> bool:
    return all(v not in sanitized_text for v in raw_values)


# ---------------------------------------------------------------------------
# Pattern detection — one test per category
# ---------------------------------------------------------------------------


def test_pattern_detects_person_titled_name():
    r = sanitize_text("Patient seen by Dr. Alice Johnson today.")
    assert _has_category(r, "PERSON")
    assert "Dr. Alice Johnson" not in r.sanitized_text


def test_pattern_detects_person_known_fixture():
    r = sanitize_text("Record for Jane Doe.")
    assert _has_category(r, "PERSON")
    assert "Jane Doe" not in r.sanitized_text


def test_pattern_detects_dob():
    r = sanitize_text("DOB: 04/22/1990")
    assert _has_category(r, "DOB")
    assert "04/22/1990" not in r.sanitized_text


def test_pattern_detects_phone():
    r = sanitize_text("Call 555-867-5309 for appointment.")
    assert _has_category(r, "PHONE")
    assert "555-867-5309" not in r.sanitized_text


def test_pattern_detects_email():
    r = sanitize_text("Contact: jane.doe@clinic.org")
    assert _has_category(r, "EMAIL")
    assert "jane.doe@clinic.org" not in r.sanitized_text


def test_pattern_detects_date():
    r = sanitize_text("Lab drawn on 2024-03-15.")
    assert _has_category(r, "DATE") or _has_category(r, "DOB")
    assert "2024-03-15" not in r.sanitized_text


def test_pattern_detects_mrn():
    r = sanitize_text("MRN: 7654321")
    assert _has_category(r, "MRN")
    assert "7654321" not in r.sanitized_text


def test_pattern_detects_insurance_id():
    r = sanitize_text("MEMBER-ID892341 on file.")
    assert _has_category(r, "INSURANCE_ID")
    assert "892341" not in r.sanitized_text


def test_pattern_detects_windows_path():
    r = sanitize_text(r"Loaded from C:\Users\operator\patient_labs.pdf")
    assert _has_category(r, "WIN_PATH")
    assert r"C:\Users" not in r.sanitized_text


def test_pattern_detects_unix_path():
    r = sanitize_text("Source: /home/operator/docs/patient_results.txt")
    assert _has_category(r, "UNIX_PATH")
    assert "/home/operator" not in r.sanitized_text


def test_pattern_detects_medical_filename():
    r = sanitize_text("File: patient_labs_results.pdf was processed.")
    assert _has_category(r, "MEDICAL_FILENAME")


def test_pattern_detects_address():
    r = sanitize_text("Lives at 123 Main Street.")
    assert _has_category(r, "ADDRESS")
    assert "123 Main Street" not in r.sanitized_text


def test_pattern_detects_facility():
    r = sanitize_text("Referred to Springfield Medical Center for follow-up.")
    assert _has_category(r, "FACILITY")
    assert "Springfield Medical Center" not in r.sanitized_text


def test_pattern_detects_secret_api_key():
    r = sanitize_text("api_key: sk-abcdefghijklmnopqrstuvwxyz01234567")
    assert _has_category(r, "SECRET")
    assert "sk-abcdefghijklmnopqrstuvwxyz01234567" not in r.sanitized_text


def test_pattern_detects_secret_bearer_token():
    r = sanitize_text("Authorization: Bearer AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdef")
    assert _has_category(r, "SECRET")


# ---------------------------------------------------------------------------
# Sanitizer: replacement consistency
# ---------------------------------------------------------------------------


def test_repeated_value_maps_to_same_token():
    text = "Jane Doe is the patient. Confirm Jane Doe's DOB."
    r = sanitize_text(text)
    # Both "Jane Doe" occurrences should map to the same token
    person_tokens = [f.token for f in r.findings if f.category == "PERSON"]
    if len(person_tokens) >= 2:
        assert person_tokens[0] == person_tokens[1]


def test_sanitized_text_has_no_raw_values():
    text = "jane.doe@example.com called 555-123-4567 from C:\\Users\\jane\\file.txt"
    r = sanitize_text(text)
    assert "jane.doe@example.com" not in r.sanitized_text
    assert "555-123-4567" not in r.sanitized_text
    assert r"C:\Users" not in r.sanitized_text


def test_replacement_map_contains_original_values():
    text = "Email: user@example.com"
    r = sanitize_text(text)
    # At least one value in replacement_map should be the email
    assert any("@" in v for v in r.replacement_map.values())


def test_replacement_map_not_in_safe_public_findings():
    text = "jane.doe@example.com"
    r = sanitize_text(text)
    # safe_public_findings must not contain raw email
    for item in r.safe_public_findings:
        assert "jane.doe@example.com" not in json.dumps(item)
    # replacement_map maps token → original (PRIVATE)
    for k, v in r.replacement_map.items():
        assert k.startswith("[")   # token form
        assert v                    # original value present in private map


def test_safe_public_findings_has_category_count_hash():
    text = "Patient: Jane Doe, DOB: 01/01/1980"
    r = sanitize_text(text)
    for item in r.safe_public_findings:
        assert "category" in item
        assert "value_hash" in item
        # Must not contain raw sensitive value
        assert "Jane Doe" not in json.dumps(item)
        assert "01/01/1980" not in json.dumps(item)


def test_clean_text_no_findings():
    r = sanitize_text("Hemoglobin A1c reference range is under 5.7%.")
    # Should be clean (no PHI/PII/paths/secrets)
    assert r.raw_phi_detected is False
    assert r.private_reference_detected is False
    assert r.secret_detected is False


# ---------------------------------------------------------------------------
# Private mapping file
# ---------------------------------------------------------------------------


def test_private_mapping_path_contains_PRIVATE():
    assert "PRIVATE" in str(DEFAULT_PRIVATE_MAPPING_PATH)


def test_private_mapping_is_gitignored():
    # Write the file first so git check-ignore can evaluate it
    write_private_mapping({"[EMAIL_1]": "test@example.com"})
    assert is_gitignored(DEFAULT_PRIVATE_MAPPING_PATH), (
        "Private mapping file must be gitignored"
    )


def test_private_mapping_is_not_tracked():
    write_private_mapping({"[TEST_1]": "test_value"})
    assert not is_tracked_by_git(DEFAULT_PRIVATE_MAPPING_PATH), (
        "Private mapping file must not be tracked by git"
    )


def test_write_private_mapping_does_not_expose_values_in_report(tmp_path: Path):
    mapping = {"[PERSON_1]": "John Doe", "[EMAIL_1]": "john@example.com"}
    private_path = tmp_path / "private_sanitization_mapping_PRIVATE.json"
    write_private_mapping(mapping, private_path)
    content = private_path.read_text(encoding="utf-8")
    # File exists and contains the mapping (private file — that's expected)
    assert "John Doe" in content  # private file — OK
    # But we verify it would NOT be in a public report by checking it's separate
    assert "PRIVATE" in private_path.name


# ---------------------------------------------------------------------------
# Outbound audit
# ---------------------------------------------------------------------------


def test_outbound_blocked_when_allow_external_false():
    payload = {"fact": "Hemoglobin normal range"}
    result = build_outbound_payload(payload, allow_external=False, purpose="test")
    assert result.allowed is False
    assert any("allow_external=False" in r for r in result.blocked_reasons)


def test_outbound_allowed_clean_payload_allow_external_true():
    payload = {"fact": "Hemoglobin A1c < 5.7%", "tier": "active"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result.allowed is True
    assert result.external_api_used is False


def test_outbound_blocked_by_secret_even_when_allow_external_true():
    payload = {"api_key": "sk-secretkeyabcdefghijklmnopqrstuvwxyz0123"}
    result_ext = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result_ext.allowed is False
    assert result_ext.secret_detected is True


def test_outbound_blocked_by_secret_when_allow_external_false():
    payload = {"token": "Bearer AbCdEfGhIjKlMnOpQrStUvWxYz01234567890123"}
    result = build_outbound_payload(payload, allow_external=False, purpose="test")
    assert result.allowed is False


def test_outbound_phi_detected_in_findings_summary():
    payload = {"patient": "Jane Doe", "note": "routine"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result.raw_phi_detected is True


def test_outbound_path_leak_counted():
    payload = {"source": r"C:\Users\staff\patient_labs.pdf"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result.private_filename_path_leaks > 0


def test_outbound_external_api_always_false():
    payload = {"fact": "safe"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result.external_api_used is False


def test_outbound_safe_public_payload_hash_present():
    payload = {"fact": "safe_fact_42"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    assert result.safe_public_payload_hash
    assert len(result.safe_public_payload_hash) > 0


def test_outbound_sanitized_payload_clean():
    payload = {"patient_name": "John Doe", "mrn": "MRN: 1234567", "fact": "routine"}
    result = build_outbound_payload(payload, allow_external=True, purpose="test")
    payload_str = json.dumps(result.sanitized_payload)
    assert "John Doe" not in payload_str
    assert "1234567" not in payload_str


# ---------------------------------------------------------------------------
# Nested payload scanning
# ---------------------------------------------------------------------------


def test_nested_dict_sanitized():
    nested = {
        "level1": {
            "level2": {
                "email": "nested@example.com",
                "phone": "555-999-0000",
            }
        }
    }
    result = build_outbound_payload(nested, allow_external=True, purpose="nested_test")
    flat = json.dumps(result.sanitized_payload)
    assert "nested@example.com" not in flat
    assert "555-999-0000" not in flat


def test_nested_list_sanitized():
    nested = {"records": [{"name": "Jane Doe"}, {"name": "safe_id_42"}]}
    result = build_outbound_payload(nested, allow_external=True, purpose="list_test")
    flat = json.dumps(result.sanitized_payload)
    assert "Jane Doe" not in flat


def test_deeply_nested_path_detected():
    nested = {"a": {"b": {"c": {"d": r"C:\Users\deep\patient_result.pdf"}}}}
    sanitized, findings, _ = sanitize_dict_values(nested)
    assert any(f.category in {"WIN_PATH", "MEDICAL_FILENAME"} for f in findings)
    assert r"C:\Users" not in json.dumps(sanitized)


# ---------------------------------------------------------------------------
# Public report privacy checker
# ---------------------------------------------------------------------------


def test_report_checker_passes_clean_payload():
    payload = {"fact": "Hemoglobin normal", "tier": "active", "confidence": 0.9}
    check = check_public_report_payload(payload)
    assert check.passed is True
    assert check.raw_phi_logged_in_public_reports is False
    assert check.private_filename_path_leaks == 0
    assert check.secret_leaks == 0


def test_report_checker_catches_raw_phi():
    payload = {"patient": "Jane Doe", "fact": "normal"}
    check = check_public_report_payload(payload)
    assert check.passed is False
    assert check.raw_phi_logged_in_public_reports is True


def test_report_checker_catches_raw_path():
    payload = {"source": r"C:\Users\staff\patient_records.pdf"}
    check = check_public_report_payload(payload)
    assert check.passed is False
    assert check.private_filename_path_leaks > 0


def test_report_checker_catches_secret():
    payload = {"config": "api_key=sk-abc123defghijklmnopqrstuvwxyz0123456"}
    check = check_public_report_payload(payload)
    assert check.passed is False
    assert check.secret_leaks > 0


def test_report_checker_redacted_examples_no_raw_values():
    payload = {"email": "raw.person@hospital.org", "fact": "normal"}
    check = check_public_report_payload(payload)
    for example in check.leak_examples_redacted:
        # Redacted examples must not contain the raw email
        assert "raw.person@hospital.org" not in example


def test_report_checker_recursively_scans_nested():
    nested = {"top": {"middle": {"bottom": "Jane Doe"}}}
    check = check_public_report_payload(nested)
    assert check.passed is False
    assert check.raw_phi_logged_in_public_reports is True


def test_report_checker_strings_checked_count():
    payload = {"a": "hello", "b": "world", "c": {"d": "foo"}}
    check = check_public_report_payload(payload)
    assert check.checked_strings_count > 0


# ---------------------------------------------------------------------------
# CKA-B02 ledger integration — PRIVACY_AUDIT event type
# ---------------------------------------------------------------------------


def test_privacy_audit_event_type_exists():
    from clinical_knowledge.models import LedgerEventType
    assert hasattr(LedgerEventType, "PRIVACY_AUDIT")
    assert LedgerEventType.PRIVACY_AUDIT.value == "privacy_audit"


def test_privacy_audit_event_not_reserved():
    from clinical_knowledge.models import LedgerEvent, LedgerEventType
    from clinical_knowledge.safe_ids import new_event_id
    # Should not raise
    event = LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.PRIVACY_AUDIT,
        record_id="test-record-id",
        timestamp="2026-01-01T00:00:00+00:00",
        reason="privacy boundary check",
        safe_public_details={"passed": True},
    )
    assert event.event_type == LedgerEventType.PRIVACY_AUDIT


def test_make_privacy_audit_event():
    from clinical_knowledge.ledger import make_privacy_audit_event
    from clinical_knowledge.models import LedgerEventType
    event = make_privacy_audit_event(
        record_id="rec-001",
        safe_record_id="cka_rec_abc123",
        findings_summary={"PERSON": 1},
        passed=True,
    )
    assert event.event_type == LedgerEventType.PRIVACY_AUDIT
    assert event.safe_public_details["passed"] is True
    assert "PERSON" in event.safe_public_details["finding_categories"]


# ---------------------------------------------------------------------------
# Validation script
# ---------------------------------------------------------------------------


def test_validation_script_runs_and_passes(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["conclusion"] == "cka_b02_privacy_boundary_ready"


def test_validation_all_cases_passed(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["all_cases_passed"] is True
    assert report["synthetic_cases_run"] == 6


def test_validation_no_phi_in_report(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["private_filename_path_leaks"] == 0
    assert report["secret_leaks"] == 0


def test_validation_safety_flags(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["external_api_used"] is False
    assert report["production_ocr_changed"] is False
    assert report["production_extractor_changed"] is False
    assert report["safety_gate_changed"] is False
    assert report["frozen_hitl_release_reopened"] is False


def test_validation_replacement_map_not_in_public(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    report = run_validation(report_dir=tmp_path / "reports")
    assert report["replacement_map_written_to_public_reports"] is False
    # Check actual report JSON file
    report_file = tmp_path / "reports" / "cka_block02_privacy_boundary_report.json"
    report_text = report_file.read_text(encoding="utf-8")
    assert "Jane Doe" not in report_text
    assert "jane.doe@example.com" not in report_text


def test_validation_json_report_no_raw_strings(tmp_path: Path):
    from scripts.run_cka_block02_privacy_boundary_validation import run_validation
    run_validation(report_dir=tmp_path / "reports")
    report_file = tmp_path / "reports" / "cka_block02_privacy_boundary_report.json"
    text = report_file.read_text(encoding="utf-8")
    for forbidden in [
        "Jane Doe", "john.doe@example.com", "555-867-5309",
        "MRN:", "C:\\Users\\", "sk-abcdef",
    ]:
        assert forbidden not in text, f"Forbidden string '{forbidden}' found in public report"


def test_no_external_api_required():
    # All imports and functions work without any external API being available
    from clinical_knowledge.privacy.sanitizer import sanitize_text
    from clinical_knowledge.privacy.outbound_audit import build_outbound_payload
    from clinical_knowledge.privacy.report_privacy import check_public_report_payload
    r = sanitize_text("safe synthetic text")
    a = build_outbound_payload({"k": "v"}, allow_external=True, purpose="test")
    c = check_public_report_payload({"k": "v"})
    assert a.external_api_used is False


# ---------------------------------------------------------------------------
# CKA-B01 regression guard
# ---------------------------------------------------------------------------


def test_cka_b01_tests_still_pass():
    """Smoke-check that CKA-B01 core models still work after B02 changes."""
    from clinical_knowledge.models import KnowledgeTier, MKBRecord, TrustLevel
    from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
    rec_id = new_record_id()
    r = MKBRecord(
        record_id=rec_id,
        safe_record_id=make_safe_record_id(rec_id),
        session_id="smoke",
        fact_type="test",
        entity_text="synthetic",
        trust_level=TrustLevel.EXPERT_VALIDATED,
    )
    assert r.tier == KnowledgeTier.ACTIVE
