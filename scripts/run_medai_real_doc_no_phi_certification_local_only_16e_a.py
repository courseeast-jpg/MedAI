#!/usr/bin/env python3
"""MEDAI-REAL-DOC-NO-PHI-CERTIFICATION-LOCAL-ONLY-16E-A.

Local-only no-PHI certification assessment for the single approved image that
previously blocked 16D. NO provider call, NO network, NO billing, NO live gate.

Raw artifacts (raw OCR text, token map, tokenized payload, operator-review material)
are written ONLY to a private location outside the repo. Public repo reports carry
counts, hashes, and pass/fail flags only — never raw OCR text, token maps, or PHI.

This block never auto-certifies a real document as safe to send. For unstructured
real OCR text the honest outcome is NEEDS_HUMAN_REVIEW.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly one approved input. No CLI argument, no folder, no substitution.
APPROVED_INPUT_PATH = r"G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG"

# Private artifacts live OUTSIDE the repo (never staged/committed/pushed).
PRIVATE_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\16E_A"))
# Public label for the private location — env-var form, never the resolved home path.
PRIVATE_LOCATION_LABEL = r"%LOCALAPPDATA%\MedAI_Private\16E_A"

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_no_phi_certification_local_only_16e_a"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
DETECTION_MATRIX_MD = REPORT_DIR / "no_phi_detection_matrix.md"
PUBLIC_ASSESSMENT_JSON = REPORT_DIR / "public_sanitized_payload_assessment.json"

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_REAL_DOC_NO_PHI_CERTIFICATION_LOCAL_ONLY_16E_A"
REQUIRED_DOCS = (
    "MEDAI_REAL_DOC_NO_PHI_CERTIFICATION_LOCAL_ONLY_16E_A.md",
    "MEDAI_LOCAL_OCR_DEID_REVIEW_RULES_16E_A.md",
    "MEDAI_OPERATOR_NO_PHI_ATTESTATION_TEMPLATE_16E_A.md",
    "MEDAI_16D_RETRY_ENTRY_CRITERIA_AFTER_16E_A.md",
    "MEDAI_NEXT_DECISION_16E_A.md",
)

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

DETECTION_CLASSES = (
    "patient_name", "dob", "dates", "address", "phone", "email", "mrn",
    "insurance_id", "account_id", "provider_name", "facility_name",
    "accession_specimen_id", "filename_path", "embedded_metadata",
    "ocr_artifacts", "free_text_identifiers", "rare_reidentification_combinations",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def _gate_active() -> bool:
    v = os.environ.get(DEDICATED_GATE_NAME)
    return v is not None and v.strip() in ACTIVE_VALUES


# ---------------------------------------------------------------------------
# Local detection. Returns COUNTS only (never raw values) for public reporting,
# plus a private token map / tokenized text for the private operator-review file.
# ---------------------------------------------------------------------------
_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b")),
    ("dates", re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")),
    ("filename_path", re.compile(r"(?:[A-Za-z]:\\[^\s]+|\b[\w\-]+\.(?:png|jpg|jpeg|pdf|tif|tiff)\b)", re.IGNORECASE)),
    ("mrn", re.compile(r"\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}", re.IGNORECASE)),
    ("accession_specimen_id", re.compile(r"\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}", re.IGNORECASE)),
    ("insurance_id", re.compile(r"\b(?:Insurance|Policy|Member)\b[:#]?\s*[A-Z0-9\-]{3,}", re.IGNORECASE)),
    ("account_id", re.compile(r"\b(?:Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}", re.IGNORECASE)),
    ("address", re.compile(r"\b\d{1,6}\s+[A-Za-z0-9.\s]{2,40}\b(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive|Ct|Court|Way)\b", re.IGNORECASE)),
    ("free_text_identifiers", re.compile(r"\b\d{4,}\b")),
    ("provider_name", re.compile(r"\b(?:Dr\.?|MD|DO|NP|PA|Physician|Provider)\b[ :]+[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+)*")),
    ("patient_name", re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")),
)
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_DOB_CONTEXT = re.compile(r"\b(?:DOB|D\.O\.B|Date of Birth|Birth ?date)\b", re.IGNORECASE)
_RESIDUAL = (
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    re.compile(r"\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}"),
    re.compile(r"\b\d{4,}\b"),
    re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
)


def detect_and_tokenize(text: str) -> dict[str, Any]:
    counts: dict[str, int] = {c: 0 for c in DETECTION_CLASSES}
    token_map: dict[str, str] = {}
    counters: dict[str, int] = {}
    tokenized = text

    def _tok(cls: str, value: str) -> str:
        for tok, val in token_map.items():
            if val == value and tok.startswith(f"[{cls.upper()}_"):
                return tok
        counters[cls] = counters.get(cls, 0) + 1
        tok = f"[{cls.upper()}_{counters[cls]}]"
        token_map[tok] = value
        return tok

    for cls, rx in _PATTERNS:
        spans = list(rx.finditer(tokenized))
        # Replace right-to-left to keep offsets valid.
        for m in reversed(spans):
            value = m.group(0)
            counts[cls] += 1
            tok = _tok(cls, value)
            tokenized = tokenized[: m.start()] + tok + tokenized[m.end():]

    # DOB = dates appearing near a DOB context word (counted, not separately tokenized).
    counts["dob"] = len(_DOB_CONTEXT.findall(text))
    # Address ZIP contributes to address signal.
    counts["address"] += len(_ZIP.findall(text))
    # Embedded metadata + OCR artifacts are filled by the caller (image/text level).

    # Residual scan AFTER tokenization, excluding token placeholders.
    no_tokens = re.sub(r"\[[A-Z_]+_\d+\]", " ", tokenized)
    residual = 0
    for rx in _RESIDUAL:
        residual += len(rx.findall(no_tokens))

    # Rare re-identification combination: DOB context present AND a ZIP present.
    counts["rare_reidentification_combinations"] = 1 if (counts["dob"] > 0 and len(_ZIP.findall(text)) > 0) else 0

    return {
        "counts": counts,
        "token_map": token_map,          # PRIVATE
        "tokenized_text": tokenized,     # PRIVATE
        "residual_candidate_count": residual,
        "total_detected": sum(counts.values()),
    }


def _try_ocr(path: Path) -> tuple[bool, str, int]:
    """Attempt local OCR. Returns (available, text, embedded_metadata_count)."""
    try:
        import pytesseract  # local OCR wrapper
        from PIL import Image
    except Exception:
        return False, "", 0
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False, "", 0
    try:
        with Image.open(path) as img:
            meta_count = len(getattr(img, "text", {}) or {}) + len(
                {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}
            )
            text = pytesseract.image_to_string(img)
        return True, text or "", meta_count
    except Exception:
        return False, "", 0


def run_assessment() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []
    approved = Path(APPROVED_INPUT_PATH)

    # Hard guards.
    if _gate_active():
        raise SystemExit("ABORT: live gate is active; refusing to run local assessment.")
    if not approved.is_file():
        notes.append("approved_file_missing_or_not_a_file")

    file_exists = approved.is_file()
    file_size = approved.stat().st_size if file_exists else 0
    file_hash = _sha256_file(approved) if file_exists else ""

    ocr_available, raw_text, meta_count = (False, "", 0)
    if file_exists:
        ocr_available, raw_text, meta_count = _try_ocr(approved)
        if not ocr_available:
            notes.append("ocr_tool_unavailable")

    detection = {"counts": {c: 0 for c in DETECTION_CLASSES}, "token_map": {},
                 "tokenized_text": "", "residual_candidate_count": 0, "total_detected": 0}
    if ocr_available and raw_text.strip():
        detection = detect_and_tokenize(raw_text)
        detection["counts"]["embedded_metadata"] = meta_count
        # crude OCR-artifact heuristic: count of non-printable / replacement chars
        detection["counts"]["ocr_artifacts"] = sum(1 for ch in raw_text if ord(ch) > 0x2000)

    # Write PRIVATE artifacts outside the repo (never committed).
    private_written = False
    try:
        PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
        (PRIVATE_DIR / "raw_ocr.txt").write_text(raw_text, encoding="utf-8")
        (PRIVATE_DIR / "token_map.json").write_text(
            json.dumps(detection["token_map"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (PRIVATE_DIR / "tokenized_payload.txt").write_text(detection["tokenized_text"], encoding="utf-8")
        (PRIVATE_DIR / "operator_review.json").write_text(
            json.dumps(
                {
                    "approved_basename": approved.name,
                    "instructions": "Human review required. Confirm no raw identifiers remain "
                                    "in tokenized_payload.txt before any later live send.",
                    "detection_counts": detection["counts"],
                    "residual_candidate_count": detection["residual_candidate_count"],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        private_written = True
    except OSError:
        notes.append("private_artifact_write_failed")

    # Recommendation: a real unstructured document is never auto-PASSed here.
    if not file_exists:
        recommendation = "FAIL"
    elif not ocr_available:
        recommendation = "NEEDS_HUMAN_REVIEW"
    else:
        # Even with zero residual, automated detection cannot prove absence -> human review.
        recommendation = "NEEDS_HUMAN_REVIEW"

    privacy_result = "passed" if recommendation == "PASS" else "needs_human_review"
    if not file_exists:
        privacy_result = "needs_human_review"

    summary = {
        "block": "MEDAI-REAL-DOC-NO-PHI-CERTIFICATION-LOCAL-ONLY-16E-A",
        "local_only": True,
        "approved_input_path": APPROVED_INPUT_PATH,
        "approved_basename": approved.name,
        "one_document_only": True,
        "folder_processed": False,
        "additional_file_processed": False,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "billing_api_call_made": False,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": _gate_active(),
        "ocr_or_local_extraction_attempted": True,
        "ocr_available": ocr_available,
        "approved_file_exists": file_exists,
        "approved_file_size_bytes": file_size,
        "approved_file_sha256": file_hash,
        "private_artifacts_written_outside_repo": private_written,
        "private_artifact_location": PRIVATE_LOCATION_LABEL,
        "raw_ocr_written_to_repo": False,
        "token_map_written_to_repo": False,
        "detected_total": detection["total_detected"],
        "residual_candidate_count": detection["residual_candidate_count"],
        "operator_review_artifact_generated": private_written,
        "public_report_phi_leak_count": 0,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "operator_review_required_before_live_retry": True,
        "future_16d_retry_not_started": True,
        "recommendation": recommendation,
        "privacy_result": privacy_result,
        "safety_result": "passed",
    }

    public_assessment = {
        "block": summary["block"],
        "approved_basename": approved.name,
        "approved_file_sha256": file_hash,
        "approved_file_size_bytes": file_size,
        "ocr_available": ocr_available,
        "detection_class_counts": detection["counts"],
        "detected_total": detection["total_detected"],
        "residual_candidate_count": detection["residual_candidate_count"],
        "tokenized_preview_safe": False,  # never asserted safe for a real doc here
        "raw_identifier_leak_count": 0,    # nothing raw is published
        "token_map_in_public_report": False,
        "recommendation": recommendation,
        "note": "Counts only. No raw OCR text, no token map, no identifiers are published. "
                "Human review of the private tokenized payload is required before any live send.",
    }
    return summary, public_assessment, notes


def _detection_matrix(public: dict[str, Any]) -> str:
    rows = [f"| {cls} | {public['detection_class_counts'].get(cls, 0)} |" for cls in DETECTION_CLASSES]
    return "\n".join(
        [
            "# 16E-A no-PHI detection matrix (counts only)",
            "",
            f"Approved file (basename only): `{public['approved_basename']}`",
            f"File hash: `{public['approved_file_sha256']}`  |  Size: `{public['approved_file_size_bytes']}` bytes",
            f"OCR available: `{public['ocr_available']}`",
            "",
            "| Detection class | Count |",
            "| --- | --- |",
            *rows,
            "",
            f"Detected total: `{public['detected_total']}`  |  Residual candidate count after tokenization: "
            f"`{public['residual_candidate_count']}`",
            f"Tokenized-preview asserted safe: `{public['tokenized_preview_safe']}`",
            f"Raw identifier leak count (public): `{public['raw_identifier_leak_count']}`",
            f"Recommendation: **{public['recommendation']}**",
            "",
            "Counts only — no raw OCR text, no token maps, no identifiers are published. The raw",
            "OCR text, token map, and tokenized payload remain private, outside git.",
            "",
        ]
    )


def _implementation(summary: dict[str, Any], notes: list[str]) -> str:
    lines = [
        "# MEDAI-REAL-DOC-NO-PHI-CERTIFICATION-LOCAL-ONLY-16E-A",
        "",
        f"## Recommendation: **{summary['recommendation']}**",
        "",
        "## Result",
        "",
        "- Local-only assessment of exactly one approved image. No provider call, no network,",
        "  no billing, no live gate activation.",
        "- Raw OCR text, token map, and tokenized payload written ONLY to a private location",
        "  outside the repo; public reports carry counts, hashes, and flags only.",
        "- A real unstructured document is never auto-certified here; human/operator review is",
        "  required before any later live send.",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "local_only", "approved_basename", "approved_file_exists", "approved_file_size_bytes",
        "ocr_or_local_extraction_attempted", "ocr_available", "detected_total",
        "residual_candidate_count", "private_artifacts_written_outside_repo",
        "raw_ocr_written_to_repo", "token_map_written_to_repo", "public_report_phi_leak_count",
        "provider_call_made", "vertex_live_execution", "billing_api_call_made",
        "future_live_gate_set", "future_live_gate_environment_active", "mkb_db_opened",
        "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
        "production_queue_mutated", "operator_review_required_before_live_retry",
        "future_16d_retry_not_started", "privacy_result", "safety_result",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    lines += [
        "",
        "## Recommended next (NO live retry started automatically)",
        "",
        "- Human/operator completes the No-PHI attestation after reviewing the private",
        "  tokenized payload. A future 16D retry requires explicit new authorization and is",
        "  not started by this block.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary, public_assessment, notes = run_assessment()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_md = _detection_matrix(public_assessment)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: make sure no raw private content leaked into public blobs.
    private_blobs = []
    try:
        for name in ("raw_ocr.txt", "token_map.json", "tokenized_payload.txt"):
            p = PRIVATE_DIR / name
            if p.exists():
                private_blobs.append(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    public_blob = "\n".join([json.dumps(summary), json.dumps(public_assessment), matrix_md, impl_md])
    leak = 0
    for raw in private_blobs:
        for line in raw.splitlines():
            s = line.strip()
            if len(s) >= 6 and s in public_blob:
                leak += 1
    summary["public_report_phi_leak_count"] = leak
    public_assessment["raw_identifier_leak_count"] = 0 if leak == 0 else leak
    if leak:
        summary["safety_result"] = "failed"
        summary["privacy_result"] = "failed"

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PUBLIC_ASSESSMENT_JSON.write_text(json.dumps(public_assessment, indent=2), encoding="utf-8")
    DETECTION_MATRIX_MD.write_text(_detection_matrix(public_assessment), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(summary, notes), encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (
        summary["provider_call_made"] is False
        and summary["vertex_live_execution"] is False
        and summary["future_live_gate_environment_active"] is False
        and summary["public_report_phi_leak_count"] == 0
        and summary["safety_result"] == "passed"
        and docs_ok
    )
    print(
        "medai_real_doc_no_phi_certification_local_only_16e_a_ok"
        if ok
        else "medai_real_doc_no_phi_certification_local_only_16e_a_attention"
    )
    print(json.dumps(
        {k: summary[k] for k in (
            "recommendation", "privacy_result", "safety_result", "ocr_available",
            "detected_total", "residual_candidate_count", "public_report_phi_leak_count",
            "provider_call_made", "vertex_live_execution", "future_live_gate_environment_active",
            "private_artifacts_written_outside_repo", "future_16d_retry_not_started",
        )}, indent=2,
    ))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
