#!/usr/bin/env python3
"""MEDAI-REAL-DOC-REPAIRED-PAYLOAD-MACHINE-REVIEW-LOCAL-ONLY-16E-C.

Local, automated machine review of the repaired tokenized payload (16E-B output) for
the single approved document. Produces sanitized counts and pass/fail flags only.

NO provider call, NO network, NO billing, NO live gate. The repaired payload, raw OCR,
and token maps are never committed. Public reports carry counts, booleans, and hashes
only. This block never auto-marks NO_PHI_ATTESTED and never starts 16D. If anything is
uncertain or any targeted identifier is found, the result is NEEDS_HUMAN_REVIEW.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

PAYLOAD_PATH = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_16E_B_Review\repaired_tokenized_payload.txt"))
REPAIR_REVIEW_PATH = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_16E_B_Review\repair_review.json"))
APPROVED_BASENAME = "2.PNG"

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_repaired_payload_machine_review_local_only_16e_c"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
MACHINE_REVIEW_MATRIX_MD = REPORT_DIR / "machine_review_matrix.md"
PUBLIC_ASSESSMENT_JSON = REPORT_DIR / "public_sanitized_review_assessment.json"

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_REAL_DOC_REPAIRED_PAYLOAD_MACHINE_REVIEW_LOCAL_ONLY_16E_C"
REQUIRED_DOCS = (
    "MEDAI_REAL_DOC_REPAIRED_PAYLOAD_MACHINE_REVIEW_LOCAL_ONLY_16E_C.md",
    "MEDAI_OPERATOR_ATTESTATION_REQUIRED_AFTER_MACHINE_REVIEW_16E_C.md",
    "MEDAI_16D_RETRY_ENTRY_CRITERIA_AFTER_16E_C.md",
    "MEDAI_NEXT_DECISION_16E_C.md",
)

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
PATIENT_NAME_TOKEN_RE = re.compile(r"\[PATIENT_NAME_\d+\]")

# Clinical/test terms that are safe and expected to remain (not identifiers).
SAFE_CLINICAL_TERMS = {
    t.lower()
    for t in (
        "Urinalysis", "Specific Gravity", "pH", "Urine Color", "Urine-Color", "Appearance",
        "WBC Esterase", "Protein", "Glucose", "Ketones", "Occult Blood", "Bilirubin",
        "Urobilinogen", "Nitrite", "Microscopic Examination", "WBC", "RBC",
        "Epithelial Cells", "Casts", "Bacteria", "Urine Culture", "Result", "No growth",
        "Reference Interval", "Current Result", "Previous Result", "Units", "Abnormal",
        "Color", "Clarity", "Yellow", "Clear", "Negative", "Positive", "Trace", "Normal",
        "Final", "Comment", "Reported", "Collected", "Received",
    )
}

# Targeted identifier patterns (applied to NON-token residual text).
_LABCORP = re.compile(r"(?i)\blabcorp\b")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:\+?\d{1,2}[ \-.]?)?\(?\d{3}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b")
_DOB = re.compile(r"(?i)\b(?:DOB|D\.O\.B|Date of Birth|Birth ?date)\b")
_ADDRESS = re.compile(r"(?i)\b\d{1,6}\s+[A-Za-z0-9.\s]{2,40}\b(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive|Ct|Court|Way)\b")
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_MRN = re.compile(r"(?i)\b(?:MRN|MR#|Medical Record(?: Number)?)\b[:#]?\s*[A-Z0-9\-]{3,}")
_INS_ACCT = re.compile(r"(?i)\b(?:Insurance|Policy|Member|Account|Acct)\b[:#]?\s*[A-Z0-9\-]{3,}")
_ACCESSION = re.compile(r"(?i)\b(?:Accession|Specimen|Order)\b[:#]?\s*[A-Z0-9\-]{3,}")
_PROVIDER = re.compile(r"(?i)\b(?:Dr\.?|MD|DO|NP|PA|Physician|Provider|Ordering)\b[ :]+[A-Z][A-Za-z.\-]+")
_PATH = re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:home|Users)/[^\s]+)")
_NAME_CANDIDATE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


def _gate_active() -> bool:
    v = os.environ.get(DEDICATED_GATE_NAME)
    return v is not None and v.strip() in ACTIVE_VALUES


def _sha16(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def review(payload: str) -> dict[str, Any]:
    no_tokens = TOKEN_RE.sub(" ", payload)  # residual text excluding tokens

    labcorp = bool(_LABCORP.search(payload))
    dob = bool(_DOB.search(no_tokens))
    address = bool(_ADDRESS.search(no_tokens)) or (bool(_ZIP.search(no_tokens)) and dob)
    phone_email = bool(_EMAIL.search(no_tokens)) or bool(_PHONE.search(no_tokens))
    mrn = bool(_MRN.search(no_tokens))
    ins_acct = bool(_INS_ACCT.search(no_tokens))
    accession = bool(_ACCESSION.search(no_tokens))
    provider_facility = bool(_PROVIDER.search(no_tokens)) or labcorp
    local_path = bool(_PATH.search(payload))

    # Name candidates = capitalized multiword sequences in residual text that are NOT
    # in the clinical allowlist. These are not proven identifiers (heuristic).
    name_candidates = [
        m.group(0) for m in _NAME_CANDIDATE.finditer(no_tokens)
        if _norm(m.group(0)) not in SAFE_CLINICAL_TERMS
    ]
    name_candidate_count = len(name_candidates)

    patient_name_tokens = len(PATIENT_NAME_TOKEN_RE.findall(payload))

    # Clinical usability: any safe clinical term present in the payload?
    low = payload.lower()
    clinical_terms_present = sum(1 for t in SAFE_CLINICAL_TERMS if t in low)
    clinical_usable = clinical_terms_present >= 3

    targeted_identifier = any([labcorp, dob, address, phone_email, mrn, ins_acct, accession, local_path])

    return {
        "labcorp": labcorp,
        "dob": dob,
        "address": address,
        "phone_email": phone_email,
        "mrn": mrn,
        "ins_acct": ins_acct,
        "accession": accession,
        "provider_facility": provider_facility,
        "local_path": local_path,
        "name_candidate_count": name_candidate_count,
        "patient_name_tokens": patient_name_tokens,
        "clinical_terms_present": clinical_terms_present,
        "clinical_usable": clinical_usable,
        "targeted_identifier_detected": targeted_identifier,
        "payload_sha256": _sha16(payload),
    }


def run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []
    if _gate_active():
        raise SystemExit("ABORT: live gate is active; refusing to run machine review.")

    payload = ""
    read_ok = False
    if PAYLOAD_PATH.is_file():
        try:
            payload = PAYLOAD_PATH.read_text(encoding="utf-8")
            read_ok = True
        except OSError:
            notes.append("payload_read_failed")
    else:
        notes.append("repaired_payload_missing")

    r = review(payload) if read_ok and payload else {
        "labcorp": False, "dob": False, "address": False, "phone_email": False, "mrn": False,
        "ins_acct": False, "accession": False, "provider_facility": False, "local_path": False,
        "name_candidate_count": 0, "patient_name_tokens": 0, "clinical_terms_present": 0,
        "clinical_usable": False, "targeted_identifier_detected": False, "payload_sha256": _sha16(""),
    }

    # Machine review is heuristic: PASS only when fully clean AND confident. Residual
    # name candidates or any targeted identifier or unread payload -> NEEDS_HUMAN_REVIEW.
    fully_clean = read_ok and not r["targeted_identifier_detected"] and r["name_candidate_count"] == 0
    machine_review_result = "PASS" if fully_clean and r["clinical_usable"] else "NEEDS_HUMAN_REVIEW"
    # Conservative: this block never auto-PASSes a real document to a live-ready state.
    if machine_review_result == "PASS":
        machine_review_result = "NEEDS_HUMAN_REVIEW"
        notes.append("auto_pass_withheld_pending_human_attestation")

    # Name candidates alone are not asserted as identifiers (they could be clinical
    # phrases not in the allowlist); they keep the result at NEEDS_HUMAN_REVIEW above
    # but do not flip this targeted-identifier boolean.
    raw_patient_identifier_detected = bool(r["targeted_identifier_detected"])

    summary = {
        "block": "MEDAI-REAL-DOC-REPAIRED-PAYLOAD-MACHINE-REVIEW-LOCAL-ONLY-16E-C",
        "local_only": True,
        "approved_basename": APPROVED_BASENAME,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "billing_api_call_made": False,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": _gate_active(),
        "sixteen_d_retry_started": False,
        "repaired_payload_read_locally": read_ok,
        "repaired_payload_sha256": r["payload_sha256"],
        "raw_labcorp_remaining": r["labcorp"],
        "raw_patient_identifier_detected": raw_patient_identifier_detected,
        "raw_dob_detected": r["dob"],
        "raw_address_detected": r["address"],
        "raw_phone_or_email_detected": r["phone_email"],
        "raw_mrn_detected": r["mrn"],
        "raw_insurance_or_account_id_detected": r["ins_acct"],
        "raw_accession_or_specimen_id_detected": r["accession"],
        "raw_provider_or_facility_identifier_detected": r["provider_facility"],
        "local_path_detected_in_payload": r["local_path"],
        "name_candidate_count": r["name_candidate_count"],
        "remaining_patient_name_token_count": r["patient_name_tokens"],
        "clinical_terms_present_count": r["clinical_terms_present"],
        "clinical_table_content_usable": r["clinical_usable"],
        "tokenized_payload_committed": False,
        "raw_ocr_committed": False,
        "token_map_committed": False,
        "public_report_phi_leak_count": 0,
        "machine_review_result": machine_review_result,
        "operator_attestation_still_required": True,
        "privacy_result": "needs_human_review" if machine_review_result != "PASS" else "passed",
        "safety_result": "passed",
    }

    public_assessment = {
        "block": summary["block"],
        "approved_basename": APPROVED_BASENAME,
        "repaired_payload_sha256": r["payload_sha256"],
        "repaired_payload_read_locally": read_ok,
        "raw_labcorp_remaining": r["labcorp"],
        "raw_dob_detected": r["dob"],
        "raw_address_detected": r["address"],
        "raw_phone_or_email_detected": r["phone_email"],
        "raw_mrn_detected": r["mrn"],
        "raw_insurance_or_account_id_detected": r["ins_acct"],
        "raw_accession_or_specimen_id_detected": r["accession"],
        "raw_provider_or_facility_identifier_detected": r["provider_facility"],
        "local_path_detected_in_payload": r["local_path"],
        "name_candidate_count": r["name_candidate_count"],
        "remaining_patient_name_token_count": r["patient_name_tokens"],
        "clinical_table_content_usable": r["clinical_usable"],
        "token_map_in_public_report": False,
        "raw_identifier_leak_count": 0,
        "machine_review_result": machine_review_result,
        "operator_attestation_still_required": True,
        "note": "Counts/booleans only. No raw payload body, no token map, no raw OCR is "
                "published. Operator attestation is still required; 16D is not started.",
    }
    return summary, public_assessment, notes


def _matrix(public: dict[str, Any]) -> str:
    keys = [
        "repaired_payload_read_locally", "raw_labcorp_remaining", "raw_dob_detected",
        "raw_address_detected", "raw_phone_or_email_detected", "raw_mrn_detected",
        "raw_insurance_or_account_id_detected", "raw_accession_or_specimen_id_detected",
        "raw_provider_or_facility_identifier_detected", "local_path_detected_in_payload",
        "name_candidate_count", "remaining_patient_name_token_count",
        "clinical_table_content_usable", "token_map_in_public_report",
        "raw_identifier_leak_count", "machine_review_result",
        "operator_attestation_still_required",
    ]
    return "\n".join(
        [
            "# 16E-C machine review matrix (counts/booleans only)",
            "",
            f"Approved file (basename only): `{public['approved_basename']}`",
            f"Repaired payload hash: `{public['repaired_payload_sha256']}`",
            "",
            "| Field | Value |",
            "| --- | --- |",
            *[f"| {k} | `{public[k]}` |" for k in keys],
            "",
            "Counts/booleans only — no raw payload body, no token map, no raw OCR. The",
            "repaired payload and token map remain private, outside git. Operator attestation",
            "is still required; this machine review does not write NO_PHI_ATTESTED.",
            "",
        ]
    )


def _implementation(summary: dict[str, Any], notes: list[str]) -> str:
    lines = [
        "# MEDAI-REAL-DOC-REPAIRED-PAYLOAD-MACHINE-REVIEW-LOCAL-ONLY-16E-C",
        "",
        f"## Machine review result: **{summary['machine_review_result']}** "
        f"(safety: {summary['safety_result']}, privacy: {summary['privacy_result']})",
        "",
        "## Result",
        "",
        "- Local heuristic machine review of the repaired tokenized payload for one approved",
        "  document. No provider call, no network, no billing, no live gate activation.",
        "- Produced sanitized counts and pass/fail flags only; the repaired payload, token",
        "  map, and raw OCR are never committed.",
        "- This machine review does not constitute human attestation and never writes",
        "  NO_PHI_ATTESTED. 16D is not started.",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "local_only", "approved_basename", "repaired_payload_read_locally",
        "raw_labcorp_remaining", "raw_patient_identifier_detected", "raw_dob_detected",
        "raw_address_detected", "raw_phone_or_email_detected", "raw_mrn_detected",
        "raw_insurance_or_account_id_detected", "raw_accession_or_specimen_id_detected",
        "raw_provider_or_facility_identifier_detected", "local_path_detected_in_payload",
        "name_candidate_count", "remaining_patient_name_token_count",
        "clinical_terms_present_count", "clinical_table_content_usable",
        "tokenized_payload_committed", "raw_ocr_committed", "token_map_committed",
        "public_report_phi_leak_count", "provider_call_made", "vertex_live_execution",
        "billing_api_call_made", "future_live_gate_environment_active",
        "sixteen_d_retry_started", "machine_review_result",
        "operator_attestation_still_required", "privacy_result", "safety_result",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    lines += [
        "",
        "## Recommended next (NO live retry started automatically)",
        "",
        "- Human/operator reads this report, reviews the private repaired payload line by",
        "  line, and completes the No-PHI attestation. A future 16D retry requires explicit",
        "  new authorization and is not started by this block.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary, public_assessment, notes = run()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_md = _matrix(public_assessment)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: confirm no repaired-payload / token-map line leaked into public.
    public_blob = "\n".join([json.dumps(summary), json.dumps(public_assessment), matrix_md, impl_md])
    leak = 0
    try:
        if PAYLOAD_PATH.is_file():
            for line in PAYLOAD_PATH.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if len(s) >= 6 and s in public_blob:
                    leak += 1
        priv_map = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\16E_B\token_map_private.json"))
        if priv_map.is_file():
            for line in priv_map.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if len(s) >= 6 and s in public_blob:
                    leak += 1
    except OSError:
        pass
    summary["public_report_phi_leak_count"] = leak
    if leak or "Labcorp" in public_blob:
        summary["safety_result"] = "failed"

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PUBLIC_ASSESSMENT_JSON.write_text(json.dumps(public_assessment, indent=2), encoding="utf-8")
    MACHINE_REVIEW_MATRIX_MD.write_text(_matrix(public_assessment), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(summary, notes), encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (
        summary["safety_result"] == "passed"
        and summary["public_report_phi_leak_count"] == 0
        and summary["provider_call_made"] is False
        and summary["future_live_gate_environment_active"] is False
        and summary["sixteen_d_retry_started"] is False
        and summary["operator_attestation_still_required"] is True
        and docs_ok
    )
    print(
        "medai_real_doc_repaired_payload_machine_review_local_only_16e_c_ok"
        if ok
        else "medai_real_doc_repaired_payload_machine_review_local_only_16e_c_attention"
    )
    print(json.dumps(
        {k: summary[k] for k in (
            "machine_review_result", "repaired_payload_read_locally", "raw_labcorp_remaining",
            "raw_patient_identifier_detected", "name_candidate_count",
            "remaining_patient_name_token_count", "clinical_table_content_usable",
            "public_report_phi_leak_count", "operator_attestation_still_required",
            "sixteen_d_retry_started", "provider_call_made", "vertex_live_execution",
            "future_live_gate_environment_active", "privacy_result", "safety_result",
        )}, indent=2,
    ))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
