"""Operator-facing release, status, and privacy labels for the Streamlit UI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import MEDAI_ALLOW_EXTERNAL_API, MEDAI_LOCAL_ONLY, MEDAI_REQUIRE_PII_SCRUB


SNAPSHOT_ID = "MedAI_Snapshot_Phase49_2026-05-01"
RELEASE_NAME = "MedAI v2 OCR/Layout HITL Release"
PHASE52_SAFETY_WARNING = (
    "Not production-autonomous. Human review is required before any extracted fact is used downstream. "
    "This is not a medical device and does not provide clinical diagnosis."
)
PRIVACY_INVARIANT_GUIDANCE = (
    "Local-only mode, empty extraction, and poor OCR cannot become accepted. "
    "Cyrillic non-lab routing cannot bypass review."
)


STATUS_GUIDANCE = {
    "accepted": "Spot-check against source before use.",
    "review": "Manual review required before relying on output.",
    "review_ocr_quality": "Do not trust extraction. OCR/input quality is insufficient.",
    "empty": "No usable extraction. Check document quality or format.",
    "error": "Processing failed.",
}

DETAILED_STATUS_GUIDANCE = {
    "accepted": (
        "Confidence and safety gates passed. Spot-check the labeled values against the source PDF before downstream use. "
        "The system never auto-promotes lab normalizations."
    ),
    "review": (
        "Manual review required. The extractor produced output below the acceptance gate or matched a non-acceptable "
        "routing rule. Open side-by-side and reconcile."
    ),
    "review_ocr_quality": (
        "Do not trust the extraction. The OCR/input quality was too low to be reliable. Re-scan, rekey, or request "
        "a typed copy from the source system."
    ),
    "empty": (
        "No usable extraction. The page contained too little recognizable text to process. Verify the file is a real "
        "medical document before re-uploading."
    ),
    "error": (
        "Processing failed before extraction could complete. Review the reason codes, resolve the underlying file issue, "
        "and re-run."
    ),
}

STATUS_LABELS = {
    "accepted": "Accepted",
    "review": "Review",
    "review_ocr_quality": "OCR Review",
    "ocr_review": "OCR Review",
    "empty": "Empty",
    "error": "Error",
}

STATUS_BADGE_STYLES = {
    "accepted": {"label": "Accepted", "class": "badge-accepted", "color": "green"},
    "review": {"label": "Review", "class": "badge-review", "color": "amber"},
    "review_ocr_quality": {"label": "OCR Review", "class": "badge-ocr", "color": "orange"},
    "ocr_review": {"label": "OCR Review", "class": "badge-ocr", "color": "orange"},
    "empty": {"label": "Empty", "class": "badge-empty", "color": "gray"},
    "error": {"label": "Error", "class": "badge-error", "color": "red"},
    "privacy_local_only": {"label": "SAFE LOCAL MODE", "class": "badge-privacy", "color": "blue"},
}

STATUS_SUBLABELS = {
    "accepted": "passed all gates · spot-check",
    "review": "manual reconciliation",
    "review_ocr_quality": "do not trust extraction",
    "empty": "no usable extraction",
    "error": "processing failed",
}


@dataclass(frozen=True)
class PrivacyModeLabels:
    local_only: str
    external_apis: str
    pii_scrub_required: str
    warning: str


def operator_guidance(status: str) -> str:
    return STATUS_GUIDANCE.get(normalize_status(status), STATUS_GUIDANCE["review"])


def detailed_operator_guidance(status: str) -> str:
    return DETAILED_STATUS_GUIDANCE.get(normalize_status(status), DETAILED_STATUS_GUIDANCE["review"])


def normalize_status(status: str | None) -> str:
    value = str(status or "review").strip().lower()
    if value in {"ocr_review", "review_ocr", "ocr quality", "ocr review"}:
        return "review_ocr_quality"
    return value if value in STATUS_LABELS else "review"


def status_label(status: str | None) -> str:
    return STATUS_LABELS.get(normalize_status(status), "Review")


def status_badge(status: str | None) -> dict[str, str]:
    normalized = normalize_status(status)
    return dict(STATUS_BADGE_STYLES.get(normalized, STATUS_BADGE_STYLES["review"]))


def operator_guidance_catalog() -> dict[str, str]:
    return {
        "Accepted": DETAILED_STATUS_GUIDANCE["accepted"],
        "Review": DETAILED_STATUS_GUIDANCE["review"],
        "OCR Review": DETAILED_STATUS_GUIDANCE["review_ocr_quality"],
        "Empty": DETAILED_STATUS_GUIDANCE["empty"],
        "Error": DETAILED_STATUS_GUIDANCE["error"],
        "Privacy invariant": PRIVACY_INVARIANT_GUIDANCE,
    }


def privacy_mode_labels(
    *,
    local_only: bool = MEDAI_LOCAL_ONLY,
    allow_external_api: bool = MEDAI_ALLOW_EXTERNAL_API,
    require_pii_scrub: bool = MEDAI_REQUIRE_PII_SCRUB,
) -> PrivacyModeLabels:
    if local_only:
        warning = "Local-only mode active. External APIs blocked."
    elif allow_external_api:
        warning = "External APIs are enabled. Only redacted payloads may be sent. Do not process real PHI unless privacy gate passes."
    else:
        warning = "External APIs disabled by default. Privacy status must be explicit before cloud processing."
    return PrivacyModeLabels(
        local_only="ON" if local_only else "OFF",
        external_apis="ENABLED" if allow_external_api and not local_only else "DISABLED",
        pii_scrub_required="YES" if require_pii_scrub else "NO",
        warning=warning,
    )


def status_from_execution_result(result: Any) -> str:
    errors = getattr(result, "validation_errors", None) or []
    codes = {str(error.get("code", "")) for error in errors if isinstance(error, dict)}
    audit = getattr(result, "audit", {}) or {}
    if "empty_extraction" in codes or bool(audit.get("empty_extraction_flag")):
        return "empty"
    reason_codes = set(_as_list(audit.get("reason_codes")))
    if "poor_input_ocr" in reason_codes or "review_ocr_quality" in reason_codes:
        return "review_ocr_quality"
    outcome = getattr(result, "outcome", "")
    validation_status = getattr(result, "validation_status", "")
    if outcome == "written" and validation_status == "accepted":
        return "accepted"
    return "review"


def build_result_summary(result: Any) -> dict[str, Any]:
    audit = getattr(result, "audit", {}) or {}
    extracted = getattr(result, "extractor_result", {}) or {}
    privacy_gate = extracted.get("privacy_gate") or audit.get("privacy_gate") or {}
    status = status_from_execution_result(result)
    reason_codes = _as_list(audit.get("reason_codes")) or [
        str(error.get("code", "")) for error in (getattr(result, "validation_errors", None) or []) if isinstance(error, dict)
    ]
    return {
        "final_status": status,
        "operator_next_action": operator_guidance(status),
        "reason_codes": [code for code in reason_codes if code],
        "ocr_layout_quality_band": audit.get("input_quality_band") or extracted.get("input_quality_band") or "unknown",
        "selected_ocr_engine": audit.get("selected_engine") or extracted.get("selected_engine") or audit.get("extractor") or "unknown",
        "cyrillic_ratio": audit.get("cyrillic_ratio") or extracted.get("cyrillic_ratio"),
        "lab_table_detected": bool(audit.get("lab_table_detected") or extracted.get("lab_table_detected")),
        "parsed_lab_row_count": int(audit.get("parsed_lab_row_count") or extracted.get("parsed_lab_row_count") or 0),
        "lab_coverage_band": audit.get("lab_coverage_band") or extracted.get("lab_coverage_band") or "unknown",
        "document_type": audit.get("document_type") or extracted.get("document_type") or "unknown",
        "privacy_gate_status": privacy_gate.get("mode", "local_only" if MEDAI_LOCAL_ONLY else "unknown"),
        "external_api_used": bool(privacy_gate.get("allowed") and privacy_gate.get("provider") not in {"local", "spacy", "phi3"}),
        "payload_redacted": bool(privacy_gate.get("payload_redacted") or privacy_gate.get("redaction_counts")),
    }


def current_commit(root: Path | None = None) -> str:
    repo_root = root or Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return (result.stdout or "unknown").strip() or "unknown"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple | set):
        return [str(item) for item in value]
    return [str(value)]
