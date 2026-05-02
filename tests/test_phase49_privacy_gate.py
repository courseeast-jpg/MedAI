from __future__ import annotations

import json

from privacy.outbound_gate import guard_external_payload
from privacy.pii_detector import detect_pii
from privacy.pii_redactor import redact_pii
from scripts.run_phase49_privacy_gate_validation import RAW_SAMPLES, run_validation
from validation_baselines.compare_holdout_baseline import tracked_report_phi_files


ENGLISH_PII = "Patient John Smith DOB 01/02/1980 MRN AB123456 phone 212-555-1212 email john@example.com."
RUSSIAN_PII = "Пациент Иван Петров дата рождения 01.02.1980 полис АБ123456 телефон +7 495 123-45-67."
ID_PII = "Insurance Policy ID ZX-778899 and SSN 123-45-6789."


def test_phase49_english_synthetic_pii_is_detected():
    report = detect_pii(ENGLISH_PII)

    assert report.pii_found is True
    assert {"PERSON", "DOB", "MRN", "PHONE", "EMAIL"}.issubset(report.counts_by_type)
    assert report.risk_level == "high"


def test_phase49_russian_cyrillic_synthetic_pii_is_detected():
    report = detect_pii(RUSSIAN_PII)

    assert report.pii_found is True
    assert {"RU_PERSON", "RU_DOB", "RU_INSURANCE_ID", "RU_PHONE"}.issubset(report.counts_by_type)
    assert report.risk_level == "high"


def test_phase49_mrn_and_insurance_like_ids_are_detected():
    report = detect_pii(ID_PII)

    assert report.pii_found is True
    assert report.counts_by_type["INSURANCE_ID"] == 1
    assert report.counts_by_type["SSN"] == 1


def test_phase49_redactor_replaces_detected_pii_with_tokens():
    result = redact_pii(ENGLISH_PII + "\n" + RUSSIAN_PII)

    assert result.redaction_passed is True
    assert "[PERSON_1]" in result.redacted_text
    assert "[RU_PERSON_1]" in result.redacted_text
    assert "John Smith" not in result.redacted_text
    assert "Иван Петров" not in result.redacted_text


def test_phase49_outbound_gate_blocks_raw_pii():
    decision = guard_external_payload(
        provider="gemini",
        text=ENGLISH_PII,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=False,
    )

    assert decision.allowed is False
    assert decision.mode == "blocked_pii_detected"


def test_phase49_outbound_gate_allows_only_redacted_payload_when_external_enabled():
    decision = guard_external_payload(
        provider="gemini",
        text=ENGLISH_PII,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=True,
    )

    assert decision.allowed is True
    assert decision.mode == "external_allowed_redacted"
    assert "John Smith" not in decision.payload_text
    assert decision.redaction_counts


def test_phase49_local_only_mode_blocks_cloud_calls():
    for provider in ("gemini", "claude", "openai"):
        decision = guard_external_payload(
            provider=provider,
            text=ENGLISH_PII,
            local_only=True,
            allow_external_api=True,
            require_pii_scrub=True,
        )
        assert decision.allowed is False
        assert decision.mode == "local_only"


def test_phase49_redaction_failure_blocks_external_calls():
    decision = guard_external_payload(
        provider="openai",
        text=ENGLISH_PII,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=True,
        redaction_failed=True,
    )

    assert decision.allowed is False
    assert decision.mode == "blocked_redaction_failed"


def test_phase49_audit_report_does_not_contain_raw_pii_samples():
    report = run_validation()
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["conclusion"] == "privacy_gate_ready"
    for sample in RAW_SAMPLES:
        assert sample not in serialized


def test_phase49_no_pdfs_tracked_under_reports():
    assert tracked_report_phi_files() == []
