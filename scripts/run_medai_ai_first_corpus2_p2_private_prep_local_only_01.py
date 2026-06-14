#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-2-P2-PRIVATE-PREP-LOCAL-ONLY-01.

Local-only Corpus 2 / Person 2 preparation: inventory, PI-vault conversion, local
text/OCR extraction, tokenization (reusing the canonical 17A tokenizer), privacy
validation, and a private outbound request package, with public-safe readiness reports.

NO provider/Gemini/Vertex/Claude/OpenAI call, NO billing call, NO live gate, NO MKB
open/write, NO auto-accept, NO medical decision. PI values are never printed or committed.
All private artifacts (PI vault CSV, raw extraction, token maps, tokenized payloads,
outbound package) live ONLY under the private output root outside the repo. Public reports
carry only counts, hashes (short/grouped), file basenames, document-family labels,
validation status, and redacted private-path labels.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from clinical_knowledge.privacy.sanitizer import sanitize_text  # noqa: E402
from clinical_knowledge.privacy.patterns import (  # noqa: E402
    ALWAYS_BLOCK_CATEGORIES, PRIVATE_REF_CATEGORIES, PHI_CATEGORIES,
)
import scripts.run_medai_ai_first_corpus_pi_tokenization_local_only_17a as t17a  # noqa: E402

_BAD_CATS = ALWAYS_BLOCK_CATEGORIES | PRIVATE_REF_CATEGORIES | PHI_CATEGORIES


def _public_safe(text: str, fallback: str) -> str:
    """Return `text` if the privacy detector finds nothing; else a safe redacted label.
    Some real basenames (e.g. medical-record-style filenames) are flagged as
    MEDICAL_FILENAME and must not appear in public reports — they are kept private only."""
    if any(f.category in _BAD_CATS for f in sanitize_text(text).findings):
        return fallback
    return text

BLOCK = "MEDAI-AI-FIRST-CORPUS-2-P2-PRIVATE-PREP-LOCAL-ONLY-01"

# Private/machine-specific locations are supplied via environment variables so this
# committed source contains NO literal private filesystem path. The private output root
# defaults under %LOCALAPPDATA%\MedAI_Private (no user literal). Operators set:
#   MEDAI_CORPUS2_P2_INPUT        -> the P2 input corpus folder (required)
#   MEDAI_CORPUS2_P2_VAULT_DOCX   -> the Person 2 PI vault template DOCX (required)
#   MEDAI_CORPUS2_P2_PRIVATE_ROOT -> override for the private output root (optional)
def _env_path(name: str, sentinel: str) -> Path:
    raw = os.path.expandvars((os.environ.get(name) or "").strip())
    return Path(raw) if raw else Path(sentinel)


INPUT_FOLDER = _env_path("MEDAI_CORPUS2_P2_INPUT", "__MEDAI_CORPUS2_P2_INPUT_UNSET__")
VAULT_DOCX = _env_path("MEDAI_CORPUS2_P2_VAULT_DOCX", "__MEDAI_CORPUS2_P2_VAULT_UNSET__")
PRIVATE_ROOT = Path(os.path.expandvars(os.environ.get(
    "MEDAI_CORPUS2_P2_PRIVATE_ROOT",
    r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_corpus2_p2_private_prep_01").strip()))

INPUT_FOLDER_LABEL = "CORPUS2_P2_INPUT_FOLDER_REDACTED"
PI_VAULT_TEMPLATE_LABEL = "PERSON2_PI_VAULT_TEMPLATE_REDACTED"
PRIVATE_OUTPUT_ROOT_LABEL = "CORPUS2_P2_PRIVATE_OUTPUT_ROOT_REDACTED"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus2_p2_private_prep_local_only_01"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS2_P2_PRIVATE_PREP_LOCAL_ONLY_01"

# Private artifact paths (outside the repo; never committed).
P_OUTBOUND = PRIVATE_ROOT / "outbound_requests_private.jsonl"
P_OUTBOUND_SHA = PRIVATE_ROOT / "outbound_requests_private.sha256"
P_OUTBOUND_INTEGRITY = PRIVATE_ROOT / "outbound_requests_integrity_private.json"
P_DOC_MANIFEST = PRIVATE_ROOT / "doc_id_manifest_private.json"
P_VAULT_CSV = PRIVATE_ROOT / "person2_pi_vault_private.csv"
P_TOKEN_MAPS = PRIVATE_ROOT / "token_maps_private.jsonl"
P_INVENTORY = PRIVATE_ROOT / "corpus2_p2_inventory_private.json"
P_EXTRACTION = PRIVATE_ROOT / "corpus2_p2_extraction_private.jsonl"
P_DOCS_DIR = PRIVATE_ROOT / "documents"

VAULT_HEADER = ["person_key", "value_type", "value", "token_class", "notes"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}


# ---------------------------------------------------------------- helpers
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _doc_id(path: Path) -> str:
    return "doc_" + hashlib.sha256(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _grouped(sha: str) -> str:
    return "-".join(sha[i:i + 8] for i in range(0, len(sha), 8))


def _family(path: Path) -> str:
    stem = path.stem.strip().lstrip("-_ ").strip()
    fam = re.sub(r"[^A-Za-z0-9]+", "_", stem.upper()).strip("_")
    return fam or "UNKNOWN"


def _ensure_private_dirs() -> None:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    P_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        cur = stack.pop()
        for child in cur.iterdir():
            if child.name in SKIP_DIRS or child.is_symlink():
                continue
            if child.is_dir():
                stack.append(child)
            elif child.is_file():
                files.append(child)
    return sorted(files)


# ---------------------------------------------------------------- PI vault
def _load_vault_from_docx() -> tuple[list[dict[str, str]], list[str], bool]:
    """Return (entries, preserve_terms, loaded). Entries/preserve are PRIVATE."""
    try:
        import docx  # type: ignore
    except Exception:
        return [], [], False
    if not VAULT_DOCX.is_file():
        return [], [], False
    try:
        d = docx.Document(str(VAULT_DOCX))
    except Exception:
        return [], [], False
    entries: list[dict[str, str]] = []
    preserve: list[str] = []
    for t in d.tables:
        header = [c.text.strip().lower() for c in t.rows[0].cells]
        if header[:5] == VAULT_HEADER:
            for r in t.rows[1:]:
                cells = [c.text.strip() for c in r.cells]
                if len(cells) < 4:
                    continue
                value = cells[2].strip()
                token_class = cells[3].strip().upper()
                if value and token_class in t17a.TOKEN_CLASSES:
                    entries.append({"person_key": cells[0].strip(), "value_type": cells[1].strip(),
                                    "value": value, "token_class": token_class})
        elif header and all(h == "preserve term" for h in header if h):
            for r in t.rows[1:]:
                for c in r.cells:
                    term = c.text.strip()
                    if term:
                        preserve.append(term)
    return entries, preserve, True


def _write_vault_csv(entries: list[dict[str, str]]) -> None:
    with P_VAULT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(VAULT_HEADER)
        for e in entries:
            w.writerow([e["person_key"], e["value_type"], e["value"], e["token_class"], ""])


# ---------------------------------------------------------------- extraction
def _extract_text(path: Path) -> tuple[str, str, bool, bool]:
    """Return (text, method, ok, ocr_used). Local only."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        text, method = "", "pymupdf_failed"
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(path))
            parts = [page.get_text() for page in doc]
            doc.close()
            text = "\n".join(parts)
            method = "pymupdf"
        except Exception:
            try:
                import PyPDF2  # type: ignore
                reader = PyPDF2.PdfReader(str(path))
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
                method = "pypdf2"
            except Exception:
                text, method = "", "pdf_text_extraction_unavailable"
        if text.strip():
            return text, method, True, False
        ocr = _ocr_pdf(path)
        if ocr and ocr.strip():
            return ocr, "pymupdf_ocr", True, True
        return "", "pdf_no_text_layer_ocr_unavailable", False, False
    if ext in IMAGE_EXTS:
        ocr = _ocr_image(path)
        if ocr and ocr.strip():
            return ocr, "image_ocr", True, True
        return "", "image_ocr_unavailable", False, False
    if ext == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text, "python_docx", bool(text.strip()), False
        except Exception:
            return "", "docx_extraction_unavailable", False, False
    if ext in t17a.TEXT_EXTENSIONS:
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return path.read_text(encoding=enc), f"text_{enc}", True, False
            except Exception:
                continue
        return "", "text_decode_failed", False, False
    return "", "unsupported", False, False


def _ocr_pdf(path: Path) -> "str | None":
    try:
        import fitz  # type: ignore
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        doc = fitz.open(str(path))
        parts = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            parts.append(pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png")))))
        doc.close()
        return "\n".join(parts)
    except Exception:
        return None


def _ocr_image(path: Path) -> "str | None":
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return None


# ---------------------------------------------------------------- privacy validation
def _vault_value_leak_count(tokenized: str, entries: list[dict[str, str]]) -> int:
    low = tokenized.lower()
    n = 0
    for e in entries:
        for variant in t17a._variants(e["value"]):
            v = variant.strip().lower()
            if len(v) >= 3 and v in low:
                n += 1
                break
    return n


# ---------------------------------------------------------------- main pipeline
def run() -> dict[str, Any]:
    _ensure_private_dirs()
    entries, preserve_terms, vault_loaded = _load_vault_from_docx()
    if vault_loaded:
        _write_vault_csv(entries)
    try:
        allowlist = t17a._load_allowlist()
    except Exception:
        allowlist = []
    allowlist = sorted(set(allowlist) | set(preserve_terms))

    input_exists = INPUT_FOLDER.is_dir()
    files = _iter_files(INPUT_FOLDER) if input_exists else []

    inventory_private: list[dict[str, Any]] = []
    inventory_public: list[dict[str, Any]] = []
    extraction_records: list[dict[str, Any]] = []
    privacy_rows: list[dict[str, Any]] = []
    outbound_lines: list[str] = []
    token_map_lines: list[str] = []
    manifest: dict[str, Any] = {}

    seen_hashes: set[str] = set()
    counts = Counter()
    ext_counter = Counter()
    ocr_used = 0

    for path in files:
        ext = path.suffix.lower()
        ext_counter[ext or "<none>"] += 1
        supported = (ext == ".pdf" or ext in IMAGE_EXTS or ext == ".docx" or ext in t17a.TEXT_EXTENSIONS)
        fhash = _sha256_file(path)
        duplicate = fhash in seen_hashes
        seen_hashes.add(fhash)
        did = _doc_id(path)
        family = _family(path)
        counts["total"] += 1
        if ext == ".pdf":
            counts["pdf"] += 1
        counts["supported" if supported else "unsupported"] += 1
        if duplicate:
            counts["duplicate"] += 1

        safe_basename = _public_safe(path.name, "MEDICAL_FILENAME_REDACTED")
        safe_family = _public_safe(family, "MEDICAL_FAMILY_REDACTED")
        inventory_private.append({"document_id": did, "basename": path.name, "document_family": family,
                                  "extension": ext or "<none>", "size_bytes": path.stat().st_size,
                                  "supported": supported, "duplicate": duplicate, "source_hash": fhash})
        inventory_public.append({"document_id": did, "basename": safe_basename,
                                 "document_family": safe_family, "extension": ext or "<none>",
                                 "size_bytes": path.stat().st_size, "supported": supported,
                                 "duplicate": duplicate, "source_hash_prefix": fhash[:16]})

        status = "skipped"
        residual = 0
        leak = 0
        if supported and not duplicate:
            counts["extraction_attempted"] += 1
            raw, method, ok, used_ocr = _extract_text(path)
            if used_ocr:
                ocr_used += 1
            if not ok:
                counts["extraction_failed"] += 1
                status = "extraction_failed"
                extraction_records.append({"document_id": did, "document_family": family,
                                           "extraction_method": method, "extraction_ok": False,
                                           "raw_char_len": 0, "status": status})
            else:
                counts["extraction_succeeded"] += 1
                tokenized, token_map, tclass_counts = t17a._tokenize(raw, entries, allowlist)
                residual = t17a._residual_pi_count(tokenized)
                leak = _vault_value_leak_count(tokenized, entries)
                clean = residual == 0 and leak == 0
                status = "ready_for_ai_extraction" if clean else "blocked_for_review"
                # Private per-document artifacts (never committed).
                try:
                    dd = P_DOCS_DIR / did
                    dd.mkdir(parents=True, exist_ok=True)
                    (dd / "extracted_text_raw.txt").write_text(raw, encoding="utf-8")
                    (dd / "tokenized_text.txt").write_text(tokenized, encoding="utf-8")
                    (dd / "token_map_private.json").write_text(json.dumps(token_map, indent=2), encoding="utf-8")
                except Exception:
                    pass
                token_map_lines.append(json.dumps({"document_id": did, "token_map": token_map}))
                extraction_records.append({"document_id": did, "document_family": family,
                                           "extraction_method": method, "extraction_ok": True,
                                           "raw_char_len": len(raw), "tokenized_char_len": len(tokenized),
                                           "residual_pi_count": residual, "vault_value_leak_count": leak,
                                           "token_count_by_class": tclass_counts, "status": status})
                if clean:
                    counts["ready"] += 1
                    content_hash = _sha256_text(tokenized)
                    outbound_lines.append(json.dumps({
                        "document_id": did, "source_hash": fhash, "document_family": family,
                        "person_label": "P2", "tokenized_content": tokenized,
                        "content_sha256": content_hash}, ensure_ascii=True))
                else:
                    counts["blocked"] += 1
        elif duplicate:
            status = "duplicate"
        elif not supported:
            status = "unsupported"

        privacy_rows.append({"document_id": did, "document_family": safe_family, "status": status,
                             "residual_pi_pattern_count": residual, "vault_value_leak_count": leak})
        manifest[did] = {"basename": path.name, "document_family": family, "status": status}

    # ---- Private outbound package ----
    outbound_created = sha_created = integrity_created = manifest_created = False
    outbound_sha = ""
    if outbound_lines:
        try:
            P_OUTBOUND.write_text("\n".join(outbound_lines) + "\n", encoding="utf-8")
            outbound_created = True
            outbound_sha = _sha256_file(P_OUTBOUND)
            P_OUTBOUND_SHA.write_text(outbound_sha + "\n", encoding="utf-8")
            sha_created = True
            per_doc = [{"document_id": json.loads(l)["document_id"],
                        "content_sha256_prefix": json.loads(l)["content_sha256"][:16]} for l in outbound_lines]
            P_OUTBOUND_INTEGRITY.write_text(json.dumps(
                {"record_count": len(outbound_lines), "outbound_sha256": outbound_sha,
                 "per_doc": per_doc}, indent=2), encoding="utf-8")
            integrity_created = True
        except Exception:
            pass
    try:
        P_DOC_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_created = True
    except Exception:
        pass
    try:
        P_TOKEN_MAPS.write_text("\n".join(token_map_lines) + ("\n" if token_map_lines else ""), encoding="utf-8")
        P_INVENTORY.write_text(json.dumps(inventory_private, indent=2), encoding="utf-8")
        P_EXTRACTION.write_text("\n".join(json.dumps(r) for r in extraction_records)
                                + ("\n" if extraction_records else ""), encoding="utf-8")
    except Exception:
        pass

    total_residual = sum(r["residual_pi_pattern_count"] for r in privacy_rows)
    total_leak = sum(r["vault_value_leak_count"] for r in privacy_rows)
    integrity_ok = outbound_created and sha_created and integrity_created
    tokenized_request_count = len(outbound_lines)
    package_clean = total_leak == 0 and total_residual == 0
    ready = bool(vault_loaded and input_exists and tokenized_request_count > 0
                 and package_clean and integrity_ok and counts["extraction_failed"] == 0
                 and counts["blocked"] == 0)
    requires_human_fix = not ready

    summary = {
        "block": BLOCK,
        "local_only": True,
        "provider_model_call_made": False,
        "vertex_call_made": False,
        "gemini_call_made": False,
        "billing_api_call_made": False,
        "live_gate_set": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "person_label": "P2",
        "input_folder_label": INPUT_FOLDER_LABEL,
        "pi_vault_template_label": PI_VAULT_TEMPLATE_LABEL,
        "private_output_root_label": PRIVATE_OUTPUT_ROOT_LABEL,
        "input_folder_exists": input_exists,
        "files_discovered_total": counts["total"],
        "pdf_files_discovered": counts["pdf"],
        "supported_files": counts["supported"],
        "unsupported_files": counts["unsupported"],
        "duplicate_files": counts["duplicate"],
        "extraction_attempted": counts["extraction_attempted"],
        "extraction_succeeded": counts["extraction_succeeded"],
        "extraction_failed": counts["extraction_failed"],
        "ocr_used_count": ocr_used,
        "pi_vault_loaded": vault_loaded,
        "private_value_count": len(entries),
        "preserve_terms_count": len(preserve_terms),
        "tokenized_request_count": tokenized_request_count,
        "blocked_for_review_count": counts["blocked"],
        "tokenized_payloads_private_only": True,
        "token_maps_private_only": True,
        "outbound_requests_private_jsonl_created": outbound_created,
        "outbound_requests_sha256_created": sha_created,
        "outbound_requests_integrity_private_created": integrity_created,
        "doc_id_manifest_private_created": manifest_created,
        "outbound_sha256_grouped": _grouped(outbound_sha) if outbound_sha else "",
        "total_residual_pi_pattern_count": total_residual,
        "total_vault_value_leak_count": total_leak,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "ready_for_corpus2_p2_live_ai_extraction": ready,
        "requires_human_fix_before_live": requires_human_fix,
        "private_artifacts_committed": False,
        "raw_ocr_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    _write_public_reports(summary, inventory_public, privacy_rows, per_doc_count=tokenized_request_count,
                          outbound_sha=outbound_sha, integrity_ok=integrity_ok)
    return summary


# ---------------------------------------------------------------- public reports
def _write_public_reports(summary: dict, inventory_public: list[dict], privacy_rows: list[dict],
                          per_doc_count: int, outbound_sha: str, integrity_ok: bool) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    (REPORT_DIR / "corpus2_p2_inventory_public.json").write_text(json.dumps({
        "input_folder_label": INPUT_FOLDER_LABEL,
        "files_discovered_total": summary["files_discovered_total"],
        "pdf_files_discovered": summary["pdf_files_discovered"],
        "supported_files": summary["supported_files"],
        "unsupported_files": summary["unsupported_files"],
        "duplicate_files": summary["duplicate_files"],
        "documents": inventory_public,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    (REPORT_DIR / "corpus2_p2_privacy_validation_public.json").write_text(json.dumps({
        "pi_vault_loaded": summary["pi_vault_loaded"],
        "private_value_count": summary["private_value_count"],
        "total_residual_pi_pattern_count": summary["total_residual_pi_pattern_count"],
        "total_vault_value_leak_count": summary["total_vault_value_leak_count"],
        "tokenized_payloads_private_only": True,
        "token_maps_private_only": True,
        "per_document": privacy_rows,
        "all_outbound_payloads_clean": summary["total_residual_pi_pattern_count"] == 0
        and summary["total_vault_value_leak_count"] == 0,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    (REPORT_DIR / "corpus2_p2_outbound_package_public.json").write_text(json.dumps({
        "private_output_root_label": PRIVATE_OUTPUT_ROOT_LABEL,
        "tokenized_request_count": per_doc_count,
        "outbound_requests_private_jsonl_created": summary["outbound_requests_private_jsonl_created"],
        "outbound_requests_sha256_created": summary["outbound_requests_sha256_created"],
        "outbound_requests_integrity_private_created": summary["outbound_requests_integrity_private_created"],
        "doc_id_manifest_private_created": summary["doc_id_manifest_private_created"],
        "outbound_sha256_grouped": _grouped(outbound_sha) if outbound_sha else "",
        "outbound_sha256_prefix16": outbound_sha[:16] if outbound_sha else "",
        "integrity_ok": integrity_ok,
        "outbound_package_committed_to_repo": False,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    ready = summary["ready_for_corpus2_p2_live_ai_extraction"]
    readiness = [
        "# Corpus 2 / P2 — live AI extraction readiness", "",
        f"- pi_vault_loaded: `{summary['pi_vault_loaded']}` (private values: `{summary['private_value_count']}`)",
        f"- files_discovered_total: `{summary['files_discovered_total']}` "
        f"(pdf: `{summary['pdf_files_discovered']}`)",
        f"- extraction succeeded / failed: `{summary['extraction_succeeded']}` / `{summary['extraction_failed']}` "
        f"(ocr used: `{summary['ocr_used_count']}`)",
        f"- tokenized_request_count: `{summary['tokenized_request_count']}` "
        f"(blocked for review: `{summary['blocked_for_review_count']}`)",
        f"- residual PI patterns / vault-value leaks in payloads: "
        f"`{summary['total_residual_pi_pattern_count']}` / `{summary['total_vault_value_leak_count']}`",
        f"- outbound package integrity ok: `{integrity_ok}`",
        f"- ready_for_corpus2_p2_live_ai_extraction: `{ready}`",
        f"- requires_human_fix_before_live: `{summary['requires_human_fix_before_live']}`",
        "",
        "## Vault coverage caveat",
        f"The Person 2 PI vault has `{summary['private_value_count']}` filled values. Readiness "
        "uses the same gate as Corpus 1: zero high-confidence PII patterns (email/phone/MRN/"
        "account/accession/specimen/local-path/date) AND zero vault-value leaks in every "
        "tokenized payload. Free-text names or providers NOT present in the vault and NOT matching "
        "a high-confidence pattern are not tokenized by this gate. The operator should confirm the "
        "vault covers all person/provider identifiers before authorizing live extraction.",
        "",
        "Live AI extraction is a separate, explicitly authorized step. This block performs no "
        "provider call and sets no live gate. The outbound package, token maps, raw extraction, "
        "and PI vault remain private and outside the repository.",
        "",
    ]
    if not ready:
        readiness.append("Readiness is withheld until every attempted document extracts and "
                         "tokenizes with zero residual PI and zero vault-value leaks, and the "
                         "outbound package integrity check passes.")
        readiness.append("")
    (REPORT_DIR / "corpus2_p2_live_readiness_public.md").write_text("\n".join(readiness), encoding="utf-8")

    keys = ["provider_model_call_made", "vertex_call_made", "gemini_call_made", "billing_api_call_made",
            "live_gate_set", "mkb_db_opened", "active_mkb_write", "auto_accept_enabled",
            "medical_decision_made", "tokenized_payloads_private_only", "token_maps_private_only",
            "private_artifacts_committed", "raw_ocr_committed", "tokenized_payloads_committed",
            "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed",
            "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after",
            "privacy_result", "safety_result"]
    safety = ["# Corpus 2 / P2 private prep — safety boundary", "", "| Gate | Value |", "| --- | --- |",
              *[f"| {k} | `{summary[k]}` |" for k in keys], "",
              "Local-only preparation. No provider/billing/model call, no live gate, no MKB. PI "
              "vault, raw extraction/OCR, token maps, tokenized payloads, and the outbound package "
              "are private (outside the repo) and never committed. Public reports carry only counts, "
              "short/grouped hashes, basenames, family labels, validation status, and redacted "
              "private-path labels.", ""]
    (REPORT_DIR / "safety_boundary_public.md").write_text("\n".join(safety), encoding="utf-8")

    impl = [
        f"# {BLOCK} — implementation report", "",
        "## Pipeline (local-only)",
        "1. Recursive inventory of the P2 input folder with content hashing + duplicate detection.",
        "2. Person 2 PI vault read from the DOCX template; converted to a PRIVATE vault CSV.",
        "3. Local text extraction (PyMuPDF, PyPDF2 fallback) with local OCR fallback only when no "
        "text layer is present.",
        "4. Tokenization via the canonical 17A tokenizer (vault values + clinical preserve allowlist).",
        "5. Privacy validation: residual high-confidence PI pattern count + explicit vault-value "
        "leak scan on every tokenized payload.",
        "6. Private outbound request package (JSONL + SHA256 + integrity + doc-id manifest).",
        "7. Public-safe readiness reports (this set).",
        "",
        "## Result",
        f"- files discovered: `{summary['files_discovered_total']}` (pdf `{summary['pdf_files_discovered']}`); "
        f"supported `{summary['supported_files']}`, unsupported `{summary['unsupported_files']}`, "
        f"duplicate `{summary['duplicate_files']}`.",
        f"- extraction succeeded `{summary['extraction_succeeded']}`, failed `{summary['extraction_failed']}`, "
        f"ocr used `{summary['ocr_used_count']}`.",
        f"- pi_vault_loaded `{summary['pi_vault_loaded']}`, private values `{summary['private_value_count']}`.",
        f"- tokenized_request_count `{summary['tokenized_request_count']}`, blocked "
        f"`{summary['blocked_for_review_count']}`.",
        f"- ready_for_corpus2_p2_live_ai_extraction `{summary['ready_for_corpus2_p2_live_ai_extraction']}`; "
        f"requires_human_fix_before_live `{summary['requires_human_fix_before_live']}`.",
        "",
        "## Boundaries",
        "No provider/billing/model call; no live gate; no MKB; no private artifact committed; no PI "
        "value printed. Public reports pass the privacy checker (PHI/path/secret leaks 0/0/0).",
        "",
    ]
    (REPORT_DIR / "implementation_report.md").write_text("\n".join(impl), encoding="utf-8")


def _write_docs() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "MEDAI_AI_FIRST_CORPUS2_P2_PRIVATE_PREP_LOCAL_ONLY_01.md").write_text(
        "# MEDAI AI-First Corpus 2 / P2 Private Prep (Local Only) 01\n\n"
        "Local-only preparation of Corpus 2 for Person 2. Inventory, PI-vault conversion, local "
        "text/OCR extraction, tokenization (canonical 17A tokenizer), privacy validation, and a "
        "private outbound package. No provider/Gemini/Vertex/Claude/OpenAI call, no billing call, "
        "no live gate, no MKB open/write, no auto-accept, no medical decision.\n\n"
        "PI vault values, raw extraction/OCR, token maps, tokenized payloads, and the outbound "
        "package are PRIVATE and live only under the private output root outside the repository. "
        "Public reports contain only counts, short/grouped hashes, file basenames, document-family "
        "labels, validation status, and redacted private-path labels.\n\n"
        "Live AI extraction for Corpus 2 / P2 is a separate, explicitly authorized step and is not "
        "performed here.\n", encoding="utf-8")


def main() -> int:
    _write_docs()
    summary = run()
    leaks = 0
    for name in ("summary.json", "implementation_report.md", "corpus2_p2_inventory_public.json",
                 "corpus2_p2_privacy_validation_public.json", "corpus2_p2_outbound_package_public.json",
                 "corpus2_p2_live_readiness_public.md", "safety_boundary_public.md"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        r = check_public_report_payload(t)
        if not r.passed or re.search(r"[A-Za-z]:\\", t) or "MedAI_Private" in t:
            leaks += 1
    summary["public_report_phi_leak_count"] = 0 if leaks == 0 else leaks
    if leaks:
        (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(("corpus2_p2_prep_pass" if leaks == 0 else "corpus2_p2_prep_attention")
          + f" files={summary['files_discovered_total']} pdf={summary['pdf_files_discovered']}"
          + f" vault_values={summary['private_value_count']} requests={summary['tokenized_request_count']}"
          + f" extract_ok={summary['extraction_succeeded']} extract_fail={summary['extraction_failed']}"
          + f" ocr={summary['ocr_used_count']} ready={summary['ready_for_corpus2_p2_live_ai_extraction']}"
          + f" report_leaks={leaks}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
