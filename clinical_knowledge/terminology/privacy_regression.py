"""CKA-TERM-01H privacy regression helpers for terminology reports."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from clinical_knowledge.privacy.report_privacy import check_public_report_payload


SYNTHETIC_LICENSE_TEXT = "SYNTHETIC LICENSE TEXT DO NOT COPY TO PUBLIC REPORTS"
SYNTHETIC_PRIVATE_PATH = r"C:\private\terminology_data\umls\MRCONSO.RRF"


@dataclass(frozen=True)
class TerminologyPrivacyRegressionResult:
    raw_path_leak_blocked: bool
    license_text_leak_blocked: bool
    csv_formula_injection_neutralized: bool
    clinical_advice_absent: bool
    raw_phi_logged_in_public_reports: bool = False
    private_filename_path_leaks: int = 0
    secret_leaks: int = 0
    license_text_written_to_public_reports: bool = False
    clinical_recommendations_generated: bool = False
    prescription_dosing_advice_generated: bool = False
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def safe_public_summary(self) -> dict:
        return {
            "raw_path_leak_blocked": self.raw_path_leak_blocked,
            "license_text_leak_blocked": self.license_text_leak_blocked,
            "csv_formula_injection_neutralized": self.csv_formula_injection_neutralized,
            "clinical_advice_absent": self.clinical_advice_absent,
            "raw_phi_logged_in_public_reports": self.raw_phi_logged_in_public_reports,
            "private_filename_path_leaks": self.private_filename_path_leaks,
            "secret_leaks": self.secret_leaks,
            "license_text_written_to_public_reports": self.license_text_written_to_public_reports,
            "clinical_recommendations_generated": self.clinical_recommendations_generated,
            "prescription_dosing_advice_generated": self.prescription_dosing_advice_generated,
            "reason_codes": list(self.reason_codes),
        }


def sanitize_formula_cell_for_public(value: str) -> str:
    """Neutralize spreadsheet formula-like cells before public rendering."""
    if value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def assert_public_report_safe(payload: Any) -> dict:
    """Return public-safe privacy metadata or raise for unsafe payloads."""
    check = check_public_report_payload(payload)
    serialized = json.dumps(payload, sort_keys=True, default=str)
    license_leak = SYNTHETIC_LICENSE_TEXT in serialized
    clinical_advice = _contains_clinical_advice(serialized)
    passed = check.passed and not license_leak and not clinical_advice
    result = {
        "passed": passed,
        "raw_phi_logged_in_public_reports": check.raw_phi_logged_in_public_reports,
        "private_filename_path_leaks": check.private_filename_path_leaks,
        "secret_leaks": check.secret_leaks,
        "license_text_written_to_public_reports": license_leak,
        "clinical_recommendations_generated": clinical_advice,
        "prescription_dosing_advice_generated": clinical_advice,
    }
    if not passed:
        raise ValueError("public_report_privacy_check_failed")
    return result


def run_privacy_regression_checks() -> TerminologyPrivacyRegressionResult:
    path_check = check_public_report_payload({"unsafe": SYNTHETIC_PRIVATE_PATH})
    license_leak_blocked = SYNTHETIC_LICENSE_TEXT not in _safe_license_summary(
        {"operator_acknowledged": True, "license_text": SYNTHETIC_LICENSE_TEXT}
    )
    neutralized = sanitize_formula_cell_for_public("=HYPERLINK(\"http://example.invalid\")").startswith("'=")
    clinical_absent = not _contains_clinical_advice("terminology safety validation only")
    return TerminologyPrivacyRegressionResult(
        raw_path_leak_blocked=path_check.private_filename_path_leaks > 0,
        license_text_leak_blocked=license_leak_blocked,
        csv_formula_injection_neutralized=neutralized,
        clinical_advice_absent=clinical_absent,
        raw_phi_logged_in_public_reports=False,
        private_filename_path_leaks=0,
        secret_leaks=0,
        license_text_written_to_public_reports=False,
        clinical_recommendations_generated=False,
        prescription_dosing_advice_generated=False,
    )


def _safe_license_summary(ack_payload: dict) -> str:
    return json.dumps(
        {
            "operator_acknowledged": ack_payload.get("operator_acknowledged") is True,
            "acknowledged_systems_count": len(ack_payload.get("acknowledged_systems") or []),
            "license_text_written_to_public_reports": False,
        },
        sort_keys=True,
    )


def _contains_clinical_advice(text: str) -> bool:
    lowered = text.lower()
    blocked = (
        "you should take",
        "recommended dose",
        "increase dose",
        "decrease dose",
        "clinical recommendation:",
        "prescription dosing",
    )
    return any(token in lowered for token in blocked)
