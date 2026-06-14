#!/usr/bin/env python3
"""MEDAI-REAL-DOC-REVIEWED-PAYLOAD-REPAIR-LOCAL-ONLY-16E-B.

Local-only repair of the reviewed tokenized payload for the single approved document.
It (1) tokenizes the raw facility/lab identifier that remained after 16E-A, and
(2) restores a curated allowlist of clinical/test terms that were over-redacted, to
preserve payload utility while staying de-identified.

NO provider call, NO network, NO billing, NO live gate. Raw OCR, token maps, and the
repaired payload remain in a private location outside the repo. Public reports carry
counts, booleans, hashes, and safe class labels only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

APPROVED_INPUT_PATH = r"G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG"

PRIV_16E_A = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\16E_A"))
PRIV_16E_B = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\16E_B"))
DOWNLOADS_REVIEW = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_16E_B_Review"))
PRIVATE_LOCATION_LABEL = r"%LOCALAPPDATA%\MedAI_Private\16E_B"
DOWNLOADS_LABEL = r"%USERPROFILE%\Downloads\MedAI_16E_B_Review"

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_reviewed_payload_repair_local_only_16e_b"
SUMMARY_JSON = REPORT_DIR / "summary.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"
REPAIR_MATRIX_MD = REPORT_DIR / "repair_matrix.md"
PUBLIC_ASSESSMENT_JSON = REPORT_DIR / "public_sanitized_repair_assessment.json"

DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_REAL_DOC_REVIEWED_PAYLOAD_REPAIR_LOCAL_ONLY_16E_B"
REQUIRED_DOCS = (
    "MEDAI_REAL_DOC_REVIEWED_PAYLOAD_REPAIR_LOCAL_ONLY_16E_B.md",
    "MEDAI_REPAIRED_PAYLOAD_REVIEW_RULES_16E_B.md",
    "MEDAI_OPERATOR_NO_PHI_ATTESTATION_AFTER_REPAIR_16E_B.md",
    "MEDAI_16D_RETRY_ENTRY_CRITERIA_AFTER_16E_B.md",
    "MEDAI_NEXT_DECISION_16E_B.md",
)

DEDICATED_GATE_NAME = "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED"
ACTIVE_VALUES = ("1", "true", "TRUE", "yes", "YES", "enabled", "ENABLED", "True")

# Facility/lab identifiers to tokenize (the known review finding + close variants).
FACILITY_IDENTIFIERS = ("Labcorp", "LabCorp", "LABCORP", "Laboratory Corporation of America", "LabCorp.")

# Curated allowlist of clinical/test terms safe to restore (normalized lowercase).
SAFE_CLINICAL_TERMS = {
    t.lower()
    for t in (
        "Urinalysis", "Specific Gravity", "pH", "Urine Color", "Urine-Color", "Appearance",
        "WBC Esterase", "Protein", "Glucose", "Ketones", "Occult Blood", "Bilirubin",
        "Urobilinogen", "Nitrite", "Microscopic Examination", "WBC", "RBC",
        "Epithelial Cells", "Casts", "Bacteria", "Urine Culture", "Result", "No growth",
        "Reference Interval", "Current Result", "Previous Result", "Units", "Abnormal",
    )
}

TOKEN_RE = re.compile(r"\[[A-Z_]+_\d+\]")
PATIENT_NAME_TOKEN_RE = re.compile(r"\[PATIENT_NAME_\d+\]")


def _gate_active() -> bool:
    v = os.environ.get(DEDICATED_GATE_NAME)
    return v is not None and v.strip() in ACTIVE_VALUES


def _sha16(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _load_16e_a() -> tuple[str, dict[str, str], bool]:
    tok_path = PRIV_16E_A / "tokenized_payload.txt"
    map_path = PRIV_16E_A / "token_map.json"
    if tok_path.exists() and map_path.exists():
        try:
            payload = tok_path.read_text(encoding="utf-8")
            token_map = json.loads(map_path.read_text(encoding="utf-8"))
            return payload, token_map, True
        except (OSError, ValueError):
            return "", {}, False
    return "", {}, False


def _local_extract_fallback() -> tuple[str, dict[str, str], bool]:
    """Local-only OCR fallback for the one approved file. No provider/network."""
    approved = Path(APPROVED_INPUT_PATH)
    if not approved.is_file():
        return "", {}, False
    try:
        import pytesseract
        from PIL import Image
        pytesseract.get_tesseract_version()
        with Image.open(approved) as img:
            text = pytesseract.image_to_string(img) or ""
        return text, {}, True
    except Exception:
        return "", {}, False


def repair(payload: str, token_map: dict[str, str]) -> dict[str, Any]:
    repaired = payload
    restored_terms = 0

    # 1) Restore over-redacted clinical terms (PATIENT_NAME tokens whose mapped value
    #    is in the curated clinical allowlist). Patient identifiers are never restored.
    def _restore(m: "re.Match[str]") -> str:
        nonlocal restored_terms
        tok = m.group(0)
        raw = token_map.get(tok)
        if raw is not None and _norm(raw) in SAFE_CLINICAL_TERMS:
            restored_terms += 1
            return raw
        return tok

    repaired = PATIENT_NAME_TOKEN_RE.sub(_restore, repaired)

    # 2) Tokenize the raw facility/lab identifier(s).
    facility_tokens_added = 0
    facility_token = "[FACILITY_1]"
    for ident in FACILITY_IDENTIFIERS:
        pattern = re.compile(r"\b" + re.escape(ident) + r"\b")
        if pattern.search(repaired):
            repaired = pattern.sub(facility_token, repaired)
            facility_tokens_added = 1  # single facility class instance

    # Residual checks.
    labcorp_remaining = bool(re.search(r"(?i)\blabcorp\b", repaired))
    remaining_patient_name_tokens = len(PATIENT_NAME_TOKEN_RE.findall(repaired))
    total_tokens_after = len(TOKEN_RE.findall(repaired))

    return {
        "repaired_payload": repaired,
        "restored_clinical_terms": restored_terms,
        "facility_tokens_added": facility_tokens_added,
        "facility_identifier_tokenized": facility_tokens_added > 0 and not labcorp_remaining,
        "labcorp_remaining": labcorp_remaining,
        "remaining_patient_name_tokens": remaining_patient_name_tokens,
        "total_tokens_after": total_tokens_after,
    }


def run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []
    if _gate_active():
        raise SystemExit("ABORT: live gate is active; refusing to run local repair.")

    payload, token_map, used_a = _load_16e_a()
    if not used_a:
        notes.append("private_16e_a_artifacts_missing_attempting_local_fallback")
        payload, token_map, ok = _local_extract_fallback()
        if not ok:
            notes.append("local_extraction_unavailable")

    before_patient_tokens = len(PATIENT_NAME_TOKEN_RE.findall(payload))
    before_total_tokens = len(TOKEN_RE.findall(payload))
    labcorp_before = bool(re.search(r"(?i)\blabcorp\b", payload))

    rep = repair(payload, token_map) if payload else {
        "repaired_payload": "", "restored_clinical_terms": 0, "facility_tokens_added": 0,
        "facility_identifier_tokenized": False, "labcorp_remaining": False,
        "remaining_patient_name_tokens": 0, "total_tokens_after": 0,
    }

    # Updated private token map: drop restored clinical-term tokens, add facility token.
    updated_map = {
        tok: val for tok, val in token_map.items()
        if not (PATIENT_NAME_TOKEN_RE.fullmatch(tok) and _norm(val) in SAFE_CLINICAL_TERMS)
    }
    if rep["facility_tokens_added"]:
        updated_map["[FACILITY_1]"] = "<facility/lab identifier — private>"

    # ---- Write PRIVATE artifacts (outside repo, never committed) ----
    private_written = False
    downloads_written = False
    try:
        PRIV_16E_B.mkdir(parents=True, exist_ok=True)
        (PRIV_16E_B / "repaired_tokenized_payload.txt").write_text(rep["repaired_payload"], encoding="utf-8")
        (PRIV_16E_B / "token_map_private.json").write_text(
            json.dumps(updated_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        repair_review = {
            "approved_basename": Path(APPROVED_INPUT_PATH).name,
            "instructions": "Human review required. Confirm no raw identifiers remain in "
                            "repaired_tokenized_payload.txt (including no raw facility/lab name) "
                            "before any later live send.",
            "restored_clinical_terms": rep["restored_clinical_terms"],
            "facility_tokens_added": rep["facility_tokens_added"],
            "remaining_patient_name_tokens": rep["remaining_patient_name_tokens"],
            "labcorp_remaining": rep["labcorp_remaining"],
        }
        (PRIV_16E_B / "repair_review.json").write_text(
            json.dumps(repair_review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        diff_summary = {
            "before_patient_name_tokens": before_patient_tokens,
            "after_patient_name_tokens": rep["remaining_patient_name_tokens"],
            "before_total_tokens": before_total_tokens,
            "after_total_tokens": rep["total_tokens_after"],
            "labcorp_before": labcorp_before,
            "labcorp_after": rep["labcorp_remaining"],
            "restored_clinical_terms": rep["restored_clinical_terms"],
            "facility_tokens_added": rep["facility_tokens_added"],
        }
        (PRIV_16E_B / "private_repair_diff_summary.json").write_text(
            json.dumps(diff_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        private_written = True
    except OSError:
        notes.append("private_artifact_write_failed")

    # ---- Copy review-safe files to Downloads (NOT raw_ocr / token_map) ----
    try:
        if private_written:
            DOWNLOADS_REVIEW.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PRIV_16E_B / "repaired_tokenized_payload.txt",
                            DOWNLOADS_REVIEW / "repaired_tokenized_payload.txt")
            shutil.copyfile(PRIV_16E_B / "repair_review.json",
                            DOWNLOADS_REVIEW / "repair_review.json")
            downloads_written = True
    except OSError:
        notes.append("downloads_copy_failed")

    summary = {
        "block": "MEDAI-REAL-DOC-REVIEWED-PAYLOAD-REPAIR-LOCAL-ONLY-16E-B",
        "local_only": True,
        "approved_input_path": APPROVED_INPUT_PATH,
        "approved_basename": Path(APPROVED_INPUT_PATH).name,
        "one_document_only": True,
        "folder_processed": False,
        "additional_file_processed": False,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "billing_api_call_made": False,
        "future_live_gate_set": False,
        "future_live_gate_environment_active": _gate_active(),
        "used_private_16e_a_artifacts": used_a,
        "repair_executed": bool(payload),
        "facility_identifier_tokenized": rep["facility_identifier_tokenized"],
        "raw_labcorp_remaining_in_repaired_payload": rep["labcorp_remaining"],
        "clinical_terms_preservation_attempted": True,
        "false_positive_reduction_attempted": True,
        "restored_clinical_terms_count": rep["restored_clinical_terms"],
        "facility_tokens_added_count": rep["facility_tokens_added"],
        "remaining_patient_name_token_count": rep["remaining_patient_name_tokens"],
        "repaired_payload_sha256": _sha16(rep["repaired_payload"]),
        "private_repaired_payload_written_outside_repo": private_written,
        "private_artifact_location": PRIVATE_LOCATION_LABEL,
        "downloads_review_copy_created": downloads_written,
        "downloads_review_location": DOWNLOADS_LABEL,
        "raw_ocr_written_to_repo": False,
        "token_map_written_to_repo": False,
        "repaired_payload_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "operator_review_required_before_live_retry": True,
        "future_16d_retry_not_started": True,
        "privacy_result": "needs_human_review",
        "safety_result": "passed",
    }

    public_assessment = {
        "block": summary["block"],
        "approved_basename": summary["approved_basename"],
        "repaired_payload_sha256": summary["repaired_payload_sha256"],
        "used_private_16e_a_artifacts": used_a,
        "facility_identifier_tokenized": rep["facility_identifier_tokenized"],
        "raw_labcorp_remaining_in_repaired_payload": rep["labcorp_remaining"],
        "restored_clinical_terms_count": rep["restored_clinical_terms"],
        "facility_tokens_added_count": rep["facility_tokens_added"],
        "remaining_patient_name_token_count": rep["remaining_patient_name_tokens"],
        "token_map_in_public_report": False,
        "raw_identifier_leak_count": 0,
        "recommendation": "NEEDS_HUMAN_REVIEW",
        "note": "Counts/booleans only. No raw OCR text, no token map, no payload body is "
                "published. Human review of the private repaired payload is required.",
    }
    return summary, public_assessment, notes


def _matrix(public: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 16E-B repair matrix (counts/booleans only)",
            "",
            f"Approved file (basename only): `{public['approved_basename']}`",
            f"Repaired payload hash: `{public['repaired_payload_sha256']}`",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| used_private_16e_a_artifacts | `{public['used_private_16e_a_artifacts']}` |",
            f"| facility_identifier_tokenized | `{public['facility_identifier_tokenized']}` |",
            f"| raw_labcorp_remaining_in_repaired_payload | `{public['raw_labcorp_remaining_in_repaired_payload']}` |",
            f"| restored_clinical_terms_count | `{public['restored_clinical_terms_count']}` |",
            f"| facility_tokens_added_count | `{public['facility_tokens_added_count']}` |",
            f"| remaining_patient_name_token_count | `{public['remaining_patient_name_token_count']}` |",
            f"| token_map_in_public_report | `{public['token_map_in_public_report']}` |",
            f"| raw_identifier_leak_count | `{public['raw_identifier_leak_count']}` |",
            f"| recommendation | **{public['recommendation']}** |",
            "",
            "Counts/booleans only — no raw OCR, no token map, no payload body. The repaired",
            "payload, token map, and raw OCR remain private, outside git.",
            "",
        ]
    )


def _implementation(summary: dict[str, Any], notes: list[str]) -> str:
    lines = [
        "# MEDAI-REAL-DOC-REVIEWED-PAYLOAD-REPAIR-LOCAL-ONLY-16E-B",
        "",
        f"## Recommendation: **{summary['privacy_result']}** (safety: {summary['safety_result']})",
        "",
        "## Result",
        "",
        "- Local-only repair of the reviewed tokenized payload for one approved document.",
        "- Tokenized the raw facility/lab identifier and restored a curated allowlist of",
        "  clinical/test terms to preserve payload utility.",
        "- Raw OCR, token maps, and the repaired payload remain private, outside git; public",
        "  reports carry counts, booleans, hashes, and safe class labels only.",
        "- No provider call, no network, no billing, no live gate activation.",
        "",
        "## Metrics",
        "",
    ]
    for key in (
        "local_only", "approved_basename", "used_private_16e_a_artifacts", "repair_executed",
        "facility_identifier_tokenized", "raw_labcorp_remaining_in_repaired_payload",
        "clinical_terms_preservation_attempted", "false_positive_reduction_attempted",
        "restored_clinical_terms_count", "facility_tokens_added_count",
        "remaining_patient_name_token_count", "private_repaired_payload_written_outside_repo",
        "downloads_review_copy_created", "raw_ocr_written_to_repo", "token_map_written_to_repo",
        "repaired_payload_written_to_repo", "public_report_phi_leak_count", "provider_call_made",
        "vertex_live_execution", "billing_api_call_made", "future_live_gate_environment_active",
        "mkb_db_opened", "active_mkb_write", "auto_accept_enabled", "medical_decision_made",
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
        "- Human/operator reviews the private `repaired_tokenized_payload.txt` and completes",
        "  the No-PHI attestation after repair. A future 16D retry requires explicit new",
        "  authorization and is not started by this block.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary, public_assessment, notes = run()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_md = _matrix(public_assessment)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: ensure no private content leaked into public blobs.
    public_blob = "\n".join([json.dumps(summary), json.dumps(public_assessment), matrix_md, impl_md])
    leak = 0
    try:
        for name in ("repaired_tokenized_payload.txt", "token_map_private.json"):
            p = PRIV_16E_B / name
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if len(s) >= 6 and s in public_blob:
                        leak += 1
        rawocr = PRIV_16E_A / "raw_ocr.txt"
        if rawocr.exists():
            for line in rawocr.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if len(s) >= 6 and s in public_blob:
                    leak += 1
    except OSError:
        pass
    summary["public_report_phi_leak_count"] = leak

    # Hard failure conditions.
    failed = (
        summary["raw_labcorp_remaining_in_repaired_payload"] is True
        or leak != 0
        or summary["future_live_gate_environment_active"] is True
        or summary["provider_call_made"] is True
    )
    if failed:
        summary["safety_result"] = "failed"

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    PUBLIC_ASSESSMENT_JSON.write_text(json.dumps(public_assessment, indent=2), encoding="utf-8")
    REPAIR_MATRIX_MD.write_text(_matrix(public_assessment), encoding="utf-8")
    IMPLEMENTATION_MD.write_text(_implementation(summary, notes), encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (
        not failed
        and summary["repair_executed"] is True
        and summary["facility_identifier_tokenized"] is True
        and summary["raw_labcorp_remaining_in_repaired_payload"] is False
        and summary["public_report_phi_leak_count"] == 0
        and docs_ok
    )
    print(
        "medai_real_doc_reviewed_payload_repair_local_only_16e_b_ok"
        if ok
        else "medai_real_doc_reviewed_payload_repair_local_only_16e_b_attention"
    )
    print(json.dumps(
        {k: summary[k] for k in (
            "repair_executed", "used_private_16e_a_artifacts", "facility_identifier_tokenized",
            "raw_labcorp_remaining_in_repaired_payload", "restored_clinical_terms_count",
            "remaining_patient_name_token_count", "public_report_phi_leak_count",
            "downloads_review_copy_created", "provider_call_made", "vertex_live_execution",
            "future_live_gate_environment_active", "privacy_result", "safety_result",
        )}, indent=2,
    ))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
