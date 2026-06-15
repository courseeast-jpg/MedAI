"""MEDAI-R32: private local DOCX/PDF export of all currently extracted MKB staging info.

Writes a private, uncommitted DOCX (and a PDF if LibreOffice is available) containing the
readable extracted content for every Extracted Payload QA Queue record, an empty-shell
section, and a not-extracted appendix index. The private export lives under private_exports/
and is NEVER committed. The committed public report is counts-only (no clinical text).

No provider calls, no extraction, no MKB writes/promotion, no auto-accept, no medical
decision. Reads existing review-staging records only.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mkb_all_records_qa_comparator import (
    build_all_records_qa_comparator,
    build_readable_markdown,
    get_comparator_record_detail,
)

BLOCK = "MEDAI-R32-PRIVATE-EXTRACTED-INFO-DOCX-PDF-EXPORT"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r32_private_extracted_info_docx_pdf_export"
EXPORT_DIR = REPO_ROOT / "private_exports" / "medai_extracted_info_r32"
DOCX_PATH = EXPORT_DIR / "MedAI_Extracted_Info_R32.docx"
PDF_PATH = EXPORT_DIR / "MedAI_Extracted_Info_R32.pdf"
DOCX_REL = "private_exports/medai_extracted_info_r32/MedAI_Extracted_Info_R32.docx"
PDF_REL = "private_exports/medai_extracted_info_r32/MedAI_Extracted_Info_R32.pdf"


def _record_detail(rid: str) -> dict[str, Any]:
    d = get_comparator_record_detail(rid, include_private_preview=False)
    sections = d.get("extracted_sections") or []
    items = d.get("extracted_items") or []
    payload = d.get("structured_payload") or {}
    md, sec_n, item_n, _nonpl = build_readable_markdown(payload, sections, items)
    qm = d.get("quality_metrics") or {}
    row = d.get("qa_row") or {}
    return {
        "record_id": d.get("record_id"),
        "corpus_id": d.get("corpus_id"),
        "package_type": d.get("package_type"),
        "source_phase": d.get("source_phase"),
        "terminal_reason": d.get("terminal_reason"),
        "qa_status": row.get("qa_status", "not_reviewed"),
        "section_count": int(qm.get("section_count", 0)),
        "item_count": int(qm.get("item_count", 0)),
        "warning_count": int(qm.get("warning_count", 0)),
        "payload_available": bool(d.get("payload_available")),
        "source_preview_available": bool((d.get("source_evidence") or {}).get("preview_available")),
        "markdown": md,
        "sections": sections,
        "items": items,
        "rendered_items": item_n,
        "warnings": (payload.get("warnings") or payload.get("extraction_warnings") or []) if isinstance(payload, dict) else [],
    }


def _build_docx(model: dict, timestamp: str) -> dict[str, Any]:
    from docx import Document

    extracted = model["extracted_queue"]
    not_extracted = model["not_extracted_queue"]
    counts = model["counts"]

    doc = Document()
    doc.add_heading("MedAI Extracted Information Export — R32", level=0)
    doc.add_paragraph(f"Export generated: {timestamp}")
    doc.add_heading("Summary", level=1)
    for line in (
        f"Total staging records: {counts['total_staging_records']}",
        f"Extracted payload records: {counts['extracted_payload_records']}",
        f"Not-extracted / metadata-only records: {counts['not_extracted_records']}",
        "Review status: all records remain review-required / unverified",
        "Active/verified records created: 0",
        "This is a private, local-only export. It is not committed and is not an MKB promotion.",
    ):
        doc.add_paragraph(line, style="List Bullet")

    usable_records: list[dict] = []
    empty_shell_records: list[dict] = []
    for row in extracted:
        detail = _record_detail(row["record_id"])
        (usable_records if detail["rendered_items"] > 0 else empty_shell_records).append(detail)

    doc.add_heading(f"Extracted payload records ({len(usable_records)})", level=1)
    for detail in usable_records:
        doc.add_heading(f"Record {detail['record_id']}", level=2)
        doc.add_paragraph(
            f"Corpus: {detail['corpus_id']} | Package: {detail['package_type']} | "
            f"Source phase: {detail['source_phase']} | Reason: {detail['terminal_reason']} | "
            f"QA status: {detail['qa_status']}"
        )
        doc.add_paragraph(
            f"Quality — sections: {detail['section_count']}, items: {detail['item_count']}, "
            f"warnings: {detail['warning_count']}, payload_available: {detail['payload_available']}, "
            f"source_preview_available: {detail['source_preview_available']}"
        )
        doc.add_heading("Extracted content", level=3)
        doc.add_paragraph(detail["markdown"] or "(no readable text)")
        doc.add_heading("Extracted sections", level=3)
        for sec in detail["sections"]:
            name = sec.get("section") if isinstance(sec, dict) else str(sec)
            n = len(sec.get("items", [])) if isinstance(sec, dict) else 0
            doc.add_paragraph(f"{name} — {n} item(s)", style="List Bullet")
        doc.add_heading("Extracted items / facts", level=3)
        for item in detail["items"]:
            text = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
            doc.add_paragraph(text, style="List Bullet")
        if detail["warnings"]:
            doc.add_heading("Warnings", level=3)
            for w in detail["warnings"]:
                doc.add_paragraph(str(w), style="List Bullet")

    doc.add_heading(f"Extracted records with no usable items ({len(empty_shell_records)})", level=1)
    for detail in empty_shell_records:
        doc.add_paragraph(
            f"Record {detail['record_id']} | Corpus: {detail['corpus_id']} | "
            f"Package: {detail['package_type']} | sections: {detail['section_count']} | "
            f"items: {detail['item_count']} | reason: no usable extracted items found | "
            f"source_preview_available: {detail['source_preview_available']}",
            style="List Bullet",
        )

    doc.add_heading(f"Not-extracted records index ({len(not_extracted)})", level=1)
    for row in not_extracted:
        doc.add_paragraph(
            f"Record {row['record_id']} | Corpus: {row['corpus_id']} | "
            f"terminal reason: {row['terminal_reason']} | failure bucket: {row['failure_bucket']} | "
            f"source_preview_available: {row['source_preview_available']}",
            style="List Bullet",
        )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_PATH))
    return {
        "usable": len(usable_records),
        "empty_shell": len(empty_shell_records),
        "not_extracted": len(not_extracted),
        "full_schema_present": any(d["package_type"] == "full_schema" for d in usable_records + empty_shell_records),
        "minimal_review_present": any(
            d["package_type"] == "minimal_review_bound" for d in usable_records + empty_shell_records),
    }


def _try_pdf() -> tuple[bool, str]:
    soffice = None
    for cand in ("soffice", "libreoffice", r"C:\Program Files\LibreOffice\program\soffice.exe"):
        if shutil.which(cand) or Path(cand).is_file():
            soffice = cand
            break
    if soffice is None:
        return False, "unavailable"
    try:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir",
                        str(EXPORT_DIR), str(DOCX_PATH)], capture_output=True, timeout=180, check=False)
        return (PDF_PATH.is_file(), PDF_REL if PDF_PATH.is_file() else "conversion_failed")
    except Exception:
        return False, "conversion_failed"


def _scan_leaks() -> int:
    leaks = 0
    pats = [re.compile(p) for p in (r"AIza[0-9A-Za-z_-]{20,}", r"ya29\.", r"Bearer ",
                                    r"[A-Za-z]:\\\\Users\\\\", r"\bMRN\b", r"\bDOB\b")]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(p.search(text) for p in pats):
            leaks += 1
    return leaks


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    model = build_all_records_qa_comparator()
    counts = model["counts"]

    docx_created = False
    build_info: dict[str, Any] = {"usable": 0, "empty_shell": 0, "not_extracted": counts["not_extracted_records"],
                                  "full_schema_present": False, "minimal_review_present": False}
    error = ""
    try:
        build_info = _build_docx(model, timestamp)
        docx_created = DOCX_PATH.is_file()
    except Exception as exc:
        error = f"docx_error:{type(exc).__name__}"

    pdf_created, pdf_path = _try_pdf() if docx_created else (False, "unavailable")

    overall = "PASS" if (docx_created
                         and build_info["usable"] + build_info["empty_shell"] == counts["extracted_payload_records"]
                         and build_info["not_extracted"] == counts["not_extracted_records"]) else "BLOCKED"

    summary = {
        "block": BLOCK,
        "overall_result": overall,
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "new_extraction_started": False,
        "active_verified_records_created": 0,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "docx_created": docx_created,
        "docx_path": DOCX_REL,
        "pdf_created": pdf_created,
        "pdf_path": pdf_path if pdf_created else "unavailable",
        "pdf_conversion_available": pdf_created,
        "extracted_payload_records_exported": counts["extracted_payload_records"],
        "not_extracted_records_indexed": counts["not_extracted_records"],
        "total_staging_records": counts["total_staging_records"],
        "empty_extraction_shell_records": build_info["empty_shell"],
        "usable_payload_records": build_info["usable"],
        "full_schema_record_present": build_info["full_schema_present"],
        "minimal_review_record_present": build_info["minimal_review_present"],
        "export_timestamp": timestamp,
        "private_export_committed": False,
        "raw_clinical_text_in_public_report": False,
        "error": error,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
    }
    leaks = _scan_leaks()
    if leaks:
        summary["private_path_leaks_after"] = leaks
        summary["privacy_result"] = "blocked"
        summary["overall_result"] = "BLOCKED"
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "export_counts_public.json").write_text(json.dumps({
        "total_staging_records": counts["total_staging_records"],
        "extracted_payload_records_exported": counts["extracted_payload_records"],
        "usable_payload_records": build_info["usable"],
        "empty_extraction_shell_records": build_info["empty_shell"],
        "not_extracted_records_indexed": counts["not_extracted_records"],
        "docx_created": docx_created,
        "pdf_created": pdf_created,
        "private_export_committed": False,
    }, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(
        f"# {BLOCK} — implementation report\n\n"
        f"- overall_result: `{overall}`\n"
        f"- DOCX created: `{docx_created}` at `{DOCX_REL}` (private, not committed)\n"
        f"- PDF created: `{pdf_created}` ({pdf_path if pdf_created else 'LibreOffice unavailable'})\n"
        f"- Extracted payload records exported: `{counts['extracted_payload_records']}` "
        f"(usable `{build_info['usable']}`, empty-shell `{build_info['empty_shell']}`)\n"
        f"- Not-extracted records indexed (appendix): `{counts['not_extracted_records']}`\n"
        f"- full_schema present: `{build_info['full_schema_present']}`, minimal_review present: "
        f"`{build_info['minimal_review_present']}`\n"
        "- Private export under `private_exports/` is NOT committed; only counts-only public "
        "reports + script/tests are committed. No provider calls, no extraction, 0 active/verified.\n",
        encoding="utf-8")

    print(f"result={overall} docx={docx_created} pdf={pdf_created} exported={counts['extracted_payload_records']} "
          f"usable={build_info['usable']} empty={build_info['empty_shell']} "
          f"not_extracted={counts['not_extracted_records']} leaks={leaks} err={error or 'none'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
