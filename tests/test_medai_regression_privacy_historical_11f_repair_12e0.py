"""Focused tests for MEDAI-REGRESSION-PRIVACY-HISTORICAL-11F-REPAIR-12E0."""
from __future__ import annotations

from clinical_knowledge.privacy import check_public_report_payload


def test_public_report_checker_allows_uuid_record_id_mrn_like_fragment() -> None:
    payload = {
        "record_ids": [
            "651e318c-baad-4430-82bc-915638ed328a",
            "136e7ad7-3bcc-4bc1-94ed-49460d37eefb",
        ]
    }

    result = check_public_report_payload(payload)

    assert result.passed is True
    assert result.raw_phi_logged_in_public_reports is False
    assert result.leak_examples_redacted == []


def test_public_report_checker_still_blocks_real_mrn_values() -> None:
    result = check_public_report_payload({"raw_identifier": "MRN: 1234567"})

    assert result.passed is False
    assert result.raw_phi_logged_in_public_reports is True
    assert result.leak_examples_redacted
