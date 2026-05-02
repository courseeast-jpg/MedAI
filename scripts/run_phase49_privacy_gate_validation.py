"""Validate Phase49 privacy gate and operator UI safety metadata."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import MEDAI_ALLOW_EXTERNAL_API, MEDAI_LOCAL_ONLY, MEDAI_PRIVACY_AUDIT, MEDAI_REQUIRE_PII_SCRUB
from app.operator_safety import operator_guidance, privacy_mode_labels
from privacy.outbound_gate import guard_external_payload
from privacy.pii_detector import detect_pii
from privacy.pii_redactor import redact_pii
from privacy.privacy_audit import assert_no_raw_samples, phi_artifact_tracking_status, write_json


REPORT_DIR = ROOT / "reports" / "phase49_privacy_ui"
JSON_REPORT = REPORT_DIR / "phase49_privacy_gate_report.json"
MD_REPORT = REPORT_DIR / "phase49_privacy_gate_report.md"

SYNTHETIC_ENGLISH = "Patient John Smith DOB 01/02/1980 MRN AB123456 phone 212-555-1212 email john@example.com."
SYNTHETIC_RUSSIAN = "Пациент Иван Петров дата рождения 01.02.1980 полис АБ123456 телефон +7 495 123-45-67 адрес Москва."
SYNTHETIC_ID = "Insurance Policy ID ZX-778899 and SSN 123-45-6789."
RAW_SAMPLES = ["John Smith", "01/02/1980", "212-555-1212", "john@example.com", "Иван Петров", "АБ123456"]


def run_validation() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    english_report = detect_pii(SYNTHETIC_ENGLISH)
    russian_report = detect_pii(SYNTHETIC_RUSSIAN)
    id_report = detect_pii(SYNTHETIC_ID)
    redacted = redact_pii(SYNTHETIC_ENGLISH + "\n" + SYNTHETIC_RUSSIAN + "\n" + SYNTHETIC_ID)

    blocked_raw = guard_external_payload(
        provider="gemini",
        text=SYNTHETIC_ENGLISH,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=False,
    )
    allowed_redacted = guard_external_payload(
        provider="gemini",
        text=SYNTHETIC_ENGLISH,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=True,
    )
    local_only_block = guard_external_payload(
        provider="claude",
        text=SYNTHETIC_ENGLISH,
        local_only=True,
        allow_external_api=True,
        require_pii_scrub=True,
    )
    redaction_failure = guard_external_payload(
        provider="openai",
        text=SYNTHETIC_ENGLISH,
        local_only=False,
        allow_external_api=True,
        require_pii_scrub=True,
        redaction_failed=True,
    )

    phi_artifacts = phi_artifact_tracking_status()
    ui_labels = privacy_mode_labels()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 49 Privacy Gate + Operator UI Safety Panel",
        "local_only_default": MEDAI_LOCAL_ONLY,
        "external_api_default_allowed": MEDAI_ALLOW_EXTERNAL_API,
        "pii_scrub_required": MEDAI_REQUIRE_PII_SCRUB,
        "privacy_audit_enabled": MEDAI_PRIVACY_AUDIT,
        "pii_detection_tests": {
            "english_synthetic_pii_detected": english_report.pii_found,
            "russian_cyrillic_synthetic_pii_detected": russian_report.pii_found,
            "mrn_insurance_like_ids_detected": id_report.pii_found,
            "english_counts_by_type": english_report.counts_by_type,
            "russian_counts_by_type": russian_report.counts_by_type,
            "id_counts_by_type": id_report.counts_by_type,
        },
        "pii_redaction_tests": {
            "redaction_passed": redacted.redaction_passed,
            "redaction_counts": redacted.redaction_counts,
            "redacted_payload_has_raw_samples": not assert_no_raw_samples({"payload": redacted.redacted_text}, RAW_SAMPLES),
        },
        "outbound_gate_tests": {
            "raw_pii_blocked": not blocked_raw.allowed and blocked_raw.mode == "blocked_pii_detected",
            "redacted_payload_allowed_when_external_enabled": allowed_redacted.allowed and allowed_redacted.mode == "external_allowed_redacted",
            "local_only_blocks_external": not local_only_block.allowed and local_only_block.mode == "local_only",
            "redaction_failure_blocks_external": not redaction_failure.allowed and redaction_failure.mode == "blocked_redaction_failed",
            "allowed_payload_hash": allowed_redacted.payload_hash,
        },
        "ui_safety_panel": {
            "accepted_guidance": operator_guidance("accepted"),
            "review_guidance": operator_guidance("review"),
            "review_ocr_quality_guidance": operator_guidance("review_ocr_quality"),
            "empty_guidance": operator_guidance("empty"),
            "local_only_label": ui_labels.local_only,
            "external_apis_label": ui_labels.external_apis,
            "pii_scrub_required_label": ui_labels.pii_scrub_required,
            "privacy_warning": ui_labels.warning,
        },
        "cloud_payload_contains_raw_pii": not assert_no_raw_samples({"payload": allowed_redacted.payload_text}, RAW_SAMPLES),
        "raw_phi_logged_in_reports": False,
        "phi_report_artifacts_tracked": not phi_artifacts["passed"],
        "phi_artifact_check": phi_artifacts,
    }
    report["raw_phi_logged_in_reports"] = not assert_no_raw_samples(report, RAW_SAMPLES)
    report["conclusion"] = _conclusion(report)

    write_json(JSON_REPORT, report)
    MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def _conclusion(report: dict[str, Any]) -> str:
    if report["phi_report_artifacts_tracked"]:
        return "blocked_by_pii_leak_risk"
    if report["cloud_payload_contains_raw_pii"] or report["raw_phi_logged_in_reports"]:
        return "blocked_by_pii_leak_risk"
    if not all(report["pii_detection_tests"][key] for key in ("english_synthetic_pii_detected", "russian_cyrillic_synthetic_pii_detected", "mrn_insurance_like_ids_detected")):
        return "blocked_by_missing_gate"
    if not report["pii_redaction_tests"]["redaction_passed"]:
        return "blocked_by_pii_leak_risk"
    if not all(report["outbound_gate_tests"][key] for key in ("raw_pii_blocked", "redacted_payload_allowed_when_external_enabled", "local_only_blocks_external", "redaction_failure_blocks_external")):
        return "blocked_by_missing_gate"
    if not report["local_only_default"] or report["external_api_default_allowed"]:
        return "ready_with_warnings"
    return "privacy_gate_ready"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 49 Privacy Gate Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Local-only default: `{report['local_only_default']}`",
        f"- External API default allowed: `{report['external_api_default_allowed']}`",
        f"- PII scrub required: `{report['pii_scrub_required']}`",
        f"- Cloud payload contains raw PII: `{report['cloud_payload_contains_raw_pii']}`",
        f"- Raw PHI logged in reports: `{report['raw_phi_logged_in_reports']}`",
        f"- PHI/report artifacts tracked: `{report['phi_report_artifacts_tracked']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Detection",
        "",
    ]
    for key, value in report["pii_detection_tests"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Redaction", ""]
    for key, value in report["pii_redaction_tests"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Outbound Gate", ""]
    for key, value in report["outbound_gate_tests"].items():
        if key != "allowed_payload_hash":
            lines.append(f"- {key}: `{value}`")
    lines += ["", "## UI Safety Panel", ""]
    lines.append(f"- local_only_label: `{report['ui_safety_panel']['local_only_label']}`")
    lines.append(f"- external_apis_label: `{report['ui_safety_panel']['external_apis_label']}`")
    lines.append(f"- pii_scrub_required_label: `{report['ui_safety_panel']['pii_scrub_required_label']}`")
    lines.append(f"- privacy_warning: `{report['ui_safety_panel']['privacy_warning']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_validation()
    print("MedAI Phase 49 privacy gate validation complete.")
    print(f"local_only_default: {report['local_only_default']}")
    print(f"external_api_default_allowed: {report['external_api_default_allowed']}")
    print(f"cloud_payload_contains_raw_pii: {report['cloud_payload_contains_raw_pii']}")
    print(f"raw_phi_logged_in_reports: {report['raw_phi_logged_in_reports']}")
    print(f"phi_report_artifacts_tracked: {report['phi_report_artifacts_tracked']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0 if report["conclusion"] in {"privacy_gate_ready", "ready_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
