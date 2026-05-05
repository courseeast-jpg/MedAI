"""CKA privacy boundary package (CKA-B02).

Provides deterministic local PII/PHI detection, sanitization,
outbound payload auditing, and public report privacy checking.
No external APIs. No Presidio dependency.
"""
from clinical_knowledge.privacy.sanitizer import SanitizedText, sanitize_text
from clinical_knowledge.privacy.outbound_audit import OutboundAuditResult, build_outbound_payload
from clinical_knowledge.privacy.report_privacy import ReportPrivacyCheck, check_public_report_payload

__all__ = [
    "SanitizedText",
    "sanitize_text",
    "OutboundAuditResult",
    "build_outbound_payload",
    "ReportPrivacyCheck",
    "check_public_report_payload",
]
