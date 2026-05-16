from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
LATEST_RUN_REPORT = ROOT / "reports" / "test_runs" / "latest_test_run.json"
REPORT_DIR = ROOT / "reports" / "medai_ru_lab_text_vis_01"
REPORT_JSON = REPORT_DIR / "medai_ru_lab_text_vis_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ru_lab_text_vis_01_report.md"
REPORT_MAIN = REPORT_DIR / "MEDAI_RU_LAB_TEXT_VIS_01.md"

_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_TABLE_SIGNAL_RE = re.compile(r"\s{2,}|[|;:\t]")


def normalize_visibility_text(text: str | None) -> str:
    """Normalize text for visibility metrics while preserving Cyrillic letters."""
    normalized = str(text or "").replace("\u0451", "\u0435").replace("\u0401", "\u0415")
    return re.sub(r"\s+", " ", normalized).strip()


def bucket_text_length(length: int) -> str:
    if length <= 0:
        return "none"
    if length < 80:
        return "tiny"
    if length < 500:
        return "short"
    if length < 2000:
        return "medium"
    return "long"


def bucket_density(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.05:
        return "low"
    if ratio < 0.25:
        return "medium"
    return "high"


def bucket_cyrillic_density(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.2:
        return "low"
    if ratio < 0.6:
        return "medium"
    return "high"


def has_table_like_numeric_text(text: str | None) -> bool:
    numeric_table_lines = 0
    for line in str(text or "").splitlines():
        if _DIGIT_RE.search(line) and _TABLE_SIGNAL_RE.search(line):
            numeric_table_lines += 1
    return numeric_table_lines >= 2


def bucket_ocr_language_config() -> str:
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:
        return "not_checked"
    if completed.returncode != 0:
        return "unknown"
    languages = {
        line.strip().lower()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip() and not line.lower().startswith("list of available")
    }
    has_russian = any(lang in {"rus", "rus_old"} or lang.startswith("rus") for lang in languages)
    has_english = "eng" in languages
    if has_russian and len(languages) > 1:
        return "multilingual"
    if has_russian:
        return "russian_available"
    if has_english:
        return "english_only"
    return "unknown"


def safe_text_visibility_summary(
    *,
    safe_id: str,
    text: str | None,
    extractor_path_used: str,
    pdf_text_extraction_attempted: bool,
    ocr_attempted: bool,
    ocr_engine: str | None,
    ocr_language_config_bucket: str,
) -> dict[str, Any]:
    raw = str(text or "")
    normalized = normalize_visibility_text(raw)
    compact = re.sub(r"\s+", "", normalized)
    denominator = max(1, len(compact))
    cyrillic_count = len(_CYRILLIC_RE.findall(normalized))
    digit_count = len(_DIGIT_RE.findall(normalized))
    ascii_alpha_count = len(_ASCII_ALPHA_RE.findall(normalized))
    table_like = has_table_like_numeric_text(raw)
    cyrillic_missing_despite_table = bool(table_like and digit_count > 0 and cyrillic_count == 0)
    non_ascii_count = sum(1 for char in normalized if ord(char) > 127)
    likely_reason = likely_failure_reason(
        text_available=bool(normalized),
        cyrillic_count=cyrillic_count,
        digit_count=digit_count,
        table_like_pattern_detected=table_like,
        ocr_attempted=ocr_attempted,
    )
    return {
        "safe_id": safe_id,
        "extractor_path_used": extractor_path_used,
        "pdf_text_extraction_attempted": bool(pdf_text_extraction_attempted),
        "ocr_attempted": bool(ocr_attempted),
        "ocr_engine": safe_ocr_engine(ocr_engine, ocr_attempted=ocr_attempted),
        "ocr_language_config_bucket": ocr_language_config_bucket,
        "text_length_bucket": bucket_text_length(len(normalized)),
        "cyrillic_detected": cyrillic_count > 0,
        "cyrillic_density_bucket": bucket_cyrillic_density(cyrillic_count / denominator),
        "digit_density_bucket": bucket_density(digit_count / denominator),
        "ascii_letter_density_bucket": bucket_density(ascii_alpha_count / denominator),
        "table_like_pattern_detected": table_like,
        "non_ascii_preserved": bool(non_ascii_count > 0) if normalized else "unknown",
        "cyrillic_missing_despite_table_text": cyrillic_missing_despite_table,
        "likely_failure_reason": likely_reason,
    }


def likely_failure_reason(
    *,
    text_available: bool,
    cyrillic_count: int,
    digit_count: int,
    table_like_pattern_detected: bool,
    ocr_attempted: bool,
) -> str:
    if not text_available:
        return "unknown"
    if cyrillic_count == 0 and table_like_pattern_detected and digit_count > 0 and not ocr_attempted:
        return "ocr_skipped_due_to_numeric_readable_text"
    if cyrillic_count == 0 and digit_count > 0:
        return "classifier_receives_numeric_only_text"
    if cyrillic_count == 0:
        return "embedded_text_has_no_cyrillic"
    return "unknown"


def safe_ocr_engine(engine: str | None, *, ocr_attempted: bool) -> str:
    if not ocr_attempted:
        return "none"
    if not engine:
        return "unavailable"
    if "tesseract" in str(engine).lower():
        return "tesseract"
    return "local"


def extract_pymupdf_text(path: Path, *, page_limit: int = 8) -> str:
    try:
        import fitz

        doc = fitz.open(str(path))
        pages: list[str] = []
        for index, page in enumerate(doc, start=1):
            if index > page_limit:
                break
            pages.append(page.get_text() or "")
        doc.close()
        return "\n".join(pages)
    except Exception:
        return ""


def load_latest_unknown_candidates(report_path: Path = LATEST_RUN_REPORT) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = list(data.get("results") or [])
    return [
        item
        for item in results
        if str(item.get("document_type") or "").strip().lower() == "unknown"
        and str(item.get("status") or "").strip().lower() == "review"
    ][:2]


def build_report(report_path: Path = LATEST_RUN_REPORT) -> dict[str, Any]:
    candidates = load_latest_unknown_candidates(report_path)
    language_bucket = bucket_ocr_language_config()
    summaries: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        processed_path = Path(str(item.get("processed_path") or ""))
        text = ""
        pdf_attempted = False
        if processed_path.exists() and processed_path.suffix.lower() == ".pdf":
            pdf_attempted = True
            text = extract_pymupdf_text(processed_path)
        summaries.append(
            safe_text_visibility_summary(
                safe_id=f"file_{index:03d}",
                text=text,
                extractor_path_used="pymupdf_native_text" if pdf_attempted else "unknown",
                pdf_text_extraction_attempted=pdf_attempted,
                ocr_attempted=False,
                ocr_engine=None,
                ocr_language_config_bucket=language_bucket,
            )
        )

    reason_counts = Counter(item["likely_failure_reason"] for item in summaries)
    likely_primary = reason_counts.most_common(1)[0][0] if reason_counts else "unknown"
    cyrillic_issue = any(not bool(item["cyrillic_detected"]) for item in summaries)
    numeric_table_visible = any(bool(item["table_like_pattern_detected"]) for item in summaries)
    root_candidates = root_cause_candidates(likely_primary, cyrillic_issue, numeric_table_visible)
    return {
        "conclusion": "medai_ru_lab_text_vis_01_completed",
        "diagnostic_type": "russian_pdf_cyrillic_text_visibility",
        "baseline_ru_lab_extract_diag_commit_short": "56bfc279",
        "files_analyzed_count": len(summaries),
        "root_cause_candidates_ranked": root_candidates,
        "likely_primary_cause": likely_primary,
        "safe_per_file_diagnostic_summary": summaries,
        "cyrillic_text_visibility_issue_confirmed": cyrillic_issue,
        "numeric_table_text_visible": numeric_table_visible,
        "ocr_routing_change_recommended": bool(cyrillic_issue and numeric_table_visible),
        "text_normalization_fix_applied": False,
        "metadata_routing_fix_applied": False,
        "recommended_next_block": "MEDAI-RU-LAB-OCR-GATE-01 - Cyrillic visibility OCR gate diagnostic",
        "recommended_not_to_do_yet": [
            "no external OCR",
            "no auto-accept expansion",
            "no confidence threshold changes",
            "no parser rewrite until Cyrillic text visibility is understood",
        ],
        "clinical_logic_changed": False,
        "clinical_interpretation_added": False,
        "medication_advice_added": False,
        "ddi_logic_changed": False,
        "confidence_thresholds_changed": False,
        "auto_acceptance_changed": False,
        "ocr_engine_changed": False,
        "external_api_enabled": False,
        "cloud_api_used": False,
        "extraction_parser_changed": False,
        "lab_value_parser_added": False,
        "treatment_parser_added": False,
        "safety_gate_changed": False,
        "b07_terminology_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_behavior_changed": False,
        "allowlist_changed": False,
        "private_files_staged": False,
        "source_documents_staged": False,
        "test_input_files_staged": False,
        "real_validation_input_files_staged": False,
        "no_raw_phi_in_report": True,
        "no_raw_filenames_in_report": True,
        "no_raw_document_text_in_report": True,
        "no_private_paths_in_report": True,
        "no_secrets_in_report": True,
    }


def root_cause_candidates(likely_primary: str, cyrillic_issue: bool, numeric_table_visible: bool) -> list[str]:
    if likely_primary == "ocr_skipped_due_to_numeric_readable_text":
        return [
            "ocr_skipped_due_to_numeric_readable_text",
            "embedded_text_has_no_cyrillic",
            "classifier_receives_numeric_only_text",
            "tesseract_russian_language_missing",
        ]
    if cyrillic_issue and numeric_table_visible:
        return [
            "classifier_receives_numeric_only_text",
            "embedded_text_has_no_cyrillic",
            "ocr_skipped_due_to_numeric_readable_text",
        ]
    if cyrillic_issue:
        return [
            "embedded_text_has_no_cyrillic",
            "cyrillic_stripped_by_cleaning",
            "classifier_receives_numeric_only_text",
        ]
    return ["unknown"]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-RU-LAB-TEXT-VIS-01",
        "",
        f"Conclusion: {report['conclusion']}",
        f"Diagnostic type: {report['diagnostic_type']}",
        f"Files analyzed: {report['files_analyzed_count']}",
        f"Cyrillic text visibility issue confirmed: {str(report['cyrillic_text_visibility_issue_confirmed']).lower()}",
        f"Numeric table text visible: {str(report['numeric_table_text_visible']).lower()}",
        f"OCR routing change recommended: {str(report['ocr_routing_change_recommended']).lower()}",
        "",
        "## Root Cause Candidates",
        "",
    ]
    for index, candidate in enumerate(report["root_cause_candidates_ranked"], start=1):
        lines.append(f"{index}. {candidate}")
    lines.extend(
        [
            "",
            f"Likely primary cause: {report['likely_primary_cause']}",
            "",
            "## Safe Per-File Diagnostic Summary",
            "",
            "| Safe ID | Extractor path | PDF text | OCR attempted | OCR engine | OCR language | Text | Cyrillic | Digits | ASCII letters | Table-like | Non-ASCII | Likely reason |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in report["safe_per_file_diagnostic_summary"]:
        lines.append(
            "| {safe_id} | {extractor_path_used} | {pdf_text_extraction_attempted} | {ocr_attempted} | "
            "{ocr_engine} | {ocr_language_config_bucket} | {text_length_bucket} | {cyrillic_density_bucket} | "
            "{digit_density_bucket} | {ascii_letter_density_bucket} | {table_like_pattern_detected} | "
            "{non_ascii_preserved} | {likely_failure_reason} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Recommended next block: {report['recommended_next_block']}",
            "",
            "Deferred actions:",
        ]
    )
    deferred = {
        "no external OCR": "External OCR: not recommended",
        "no auto-accept expansion": "Auto-accept expansion: not recommended",
        "no confidence threshold changes": "Confidence threshold changes: not recommended",
        "no parser rewrite until Cyrillic text visibility is understood": "Parser rewrite: wait until Cyrillic text visibility is understood",
    }
    for item in report["recommended_not_to_do_yet"]:
        lines.append(f"- {deferred.get(item, item)}")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Clinical logic changed: false",
            "- Clinical interpretation added: false",
            "- Medication advice added: false",
            "- DDI logic changed: false",
            "- Confidence thresholds changed: false",
            "- Auto-acceptance changed: false",
            "- OCR engine changed: false",
            "- External API enabled: false",
            "- Cloud API used: false",
            "- Extraction parser changed: false",
            "- Lab value parser added: false",
            "- Treatment parser added: false",
            "- Safety gate changed: false",
            "- B07 terminology changed: false",
            "- ROUTE-FIX changed: false",
            "- DB schema changed: false",
            "- Command behavior changed: false",
            "- Allowlist changed: false",
            "",
            "## Privacy",
            "",
            "- No raw PHI in report: true",
            "- No raw filenames in report: true",
            "- No raw document text in report: true",
            "- No private paths in report: true",
            "- No secrets in report: true",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    REPORT_MAIN.write_text(markdown, encoding="utf-8")


def main() -> None:
    report = build_report()
    write_reports(report)
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "files_analyzed_count": report["files_analyzed_count"],
                "likely_primary_cause": report["likely_primary_cause"],
                "cyrillic_text_visibility_issue_confirmed": report["cyrillic_text_visibility_issue_confirmed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
