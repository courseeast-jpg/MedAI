from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
LATEST_RUN_REPORT = ROOT / "reports" / "test_runs" / "latest_test_run.json"
REPORT_DIR = ROOT / "reports" / "medai_ru_lab_extract_diag_01"
REPORT_JSON = REPORT_DIR / "medai_ru_lab_extract_diag_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ru_lab_extract_diag_01_report.md"
REPORT_MAIN = REPORT_DIR / "MEDAI_RU_LAB_EXTRACT_DIAG_01.md"

LAB_CUE_TERMS = {
    "lab_header": ("лабораторное исследование", "анализ", "общий анализ крови", "биохимия"),
    "result_terms": ("результат", "результаты", "показатель", "значение"),
    "reference_terms": ("референсные значения", "референс", "норма", "нормы"),
    "analyte_terms": (
        "гемоглобин",
        "лейкоциты",
        "эритроциты",
        "тромбоциты",
        "глюкоза",
        "креатинин",
        "мочевина",
        "билирубин",
        "холестерин",
        "алт",
        "аст",
    ),
    "units_terms": ("единицы", "ед", "мг", "мл", "ммоль", "г/л", "10^", "%"),
    "blood_panel_terms": ("оак", "общий анализ крови", "кровь", "биохимия"),
    "admin_collection_terms": ("дата", "взят", "получен", "забор", "исследование", "материал"),
}


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е")).strip()


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


def bucket_line_count(count: int) -> str:
    if count <= 0:
        return "none"
    if count < 5:
        return "few"
    if count < 25:
        return "medium"
    return "many"


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


def bucket_confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if confidence < 0.5:
        return "low_under_0_50"
    if confidence < 0.65:
        return "moderate_0_50_to_0_64"
    if confidence < 0.85:
        return "review_band_0_65_to_0_84"
    return "high_0_85_or_above"


def cue_categories(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    detected: list[str] = []
    for category, terms in LAB_CUE_TERMS.items():
        if any(term in normalized for term in terms):
            detected.append(category)
    return detected


def table_like_pattern_detected(text: str | None) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    numeric_lines = 0
    for line in lines:
        has_digit = bool(re.search(r"\d", line))
        has_table_signal = bool(re.search(r"\s{2,}|[|;:\t]", line))
        if has_digit and has_table_signal:
            numeric_lines += 1
    return numeric_lines >= 2


def likely_failure_reason(*, text: str | None, categories: list[str], current_document_type: str | None) -> str:
    normalized = normalize_text(text)
    cyrillic_count = len(re.findall(r"[а-яе]", normalized, flags=re.I))
    if not normalized:
        return "no_text_available"
    if cyrillic_count == 0:
        return "no_cyrillic_text_detected"
    if len(normalized) < 80:
        return "text_too_sparse"
    if not categories:
        return "lab_cues_absent"
    if table_like_pattern_detected(text) and len(categories) < 3:
        return "table_structure_lost"
    if str(current_document_type or "").strip().lower() == "unknown" and len(categories) >= 3:
        return "classifier_input_missing"
    return "unknown"


def safe_text_visibility_diagnostic(
    *,
    safe_id: str,
    document_type_current: str | None,
    extractor: str | None,
    confidence: Any,
    ocr_quality: str | None,
    text: str | None,
) -> dict[str, Any]:
    raw = str(text or "")
    normalized = normalize_text(raw)
    compact = normalized.replace(" ", "")
    cyrillic_count = len(re.findall(r"[а-яе]", normalized, flags=re.I))
    digit_count = len(re.findall(r"\d", normalized))
    categories = cue_categories(raw)
    line_count = len([line for line in raw.splitlines() if line.strip()])
    return {
        "safe_id": safe_id,
        "document_type_current": document_type_current or "Unknown",
        "extractor": extractor,
        "confidence_bucket": bucket_confidence(confidence),
        "ocr_quality": ocr_quality,
        "text_available": bool(normalized),
        "text_length_bucket": bucket_text_length(len(normalized)),
        "cyrillic_detected": cyrillic_count > 0,
        "cyrillic_density_bucket": bucket_cyrillic_density(cyrillic_count / len(compact) if compact else 0.0),
        "digit_density_bucket": bucket_density(digit_count / len(compact) if compact else 0.0),
        "table_like_pattern_detected": table_like_pattern_detected(raw),
        "line_count_bucket": bucket_line_count(line_count),
        "russian_lab_cue_category_count": len(categories),
        "russian_lab_cue_categories_detected": categories,
        "likely_failure_reason": likely_failure_reason(
            text=raw,
            categories=categories,
            current_document_type=document_type_current,
        ),
    }


def extract_pdf_text_local(path: Path, *, page_limit: int = 8) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        texts: list[str] = []
        for page in reader.pages[:page_limit]:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""


def load_latest_unknown_lab_candidates(report_path: Path = LATEST_RUN_REPORT) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = list(data.get("results") or [])
    unknowns = [
        item
        for item in results
        if str(item.get("document_type") or "").strip().lower() == "unknown"
        and str(item.get("status") or "").strip().lower() == "review"
    ]
    return unknowns[:2]


def build_report() -> dict[str, Any]:
    candidates = load_latest_unknown_lab_candidates()
    summaries: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        processed_path = Path(str(item.get("processed_path") or ""))
        text = extract_pdf_text_local(processed_path) if processed_path.exists() and processed_path.suffix.lower() == ".pdf" else ""
        summaries.append(
            safe_text_visibility_diagnostic(
                safe_id=f"file_{index:03d}",
                document_type_current=str(item.get("document_type") or "Unknown"),
                extractor=item.get("selected_extractor"),
                confidence=item.get("confidence"),
                ocr_quality=item.get("ocr_quality_band"),
                text=text,
            )
        )
    reason_counts = Counter(item["likely_failure_reason"] for item in summaries)
    likely_primary = reason_counts.most_common(1)[0][0] if reason_counts else "unknown"
    root_candidates = [
        "text_extraction_visible_but_lab_cues_absent",
        "text_too_sparse_or_table_structure_lost",
        "classifier_input_missing",
        "cue_normalization_gap",
        "no_text_available",
    ]
    if likely_primary == "no_text_available":
        root_candidates = ["no_text_available", "text_too_sparse_or_table_structure_lost", "classifier_input_missing"]
    elif likely_primary == "no_cyrillic_text_detected":
        root_candidates = ["no_cyrillic_text_detected", "text_extraction_visible_but_lab_cues_absent", "text_too_sparse_or_table_structure_lost"]
    elif likely_primary == "classifier_input_missing":
        root_candidates = ["classifier_input_missing", "cue_normalization_gap", "text_too_sparse_or_table_structure_lost"]
    elif likely_primary in {"lab_cues_absent", "table_structure_lost", "text_too_sparse"}:
        root_candidates = ["text_too_sparse_or_table_structure_lost", "text_extraction_visible_but_lab_cues_absent", "cue_normalization_gap"]

    return {
        "conclusion": "medai_ru_lab_extract_diag_01_completed",
        "diagnostic_type": "russian_lab_pdf_text_visibility",
        "files_analyzed_count": len(summaries),
        "root_cause_candidates_ranked": root_candidates,
        "likely_primary_cause": likely_primary,
        "safe_per_file_diagnostic_summary": summaries,
        "recommended_next_block": "MEDAI-RU-LAB-TEXT-VIS-01 - Russian lab extraction text visibility repair",
        "recommended_not_to_do_yet": [
            "no external OCR yet",
            "no threshold changes",
            "no auto-accept expansion",
            "no parser rewrite until text visibility is understood",
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


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    REPORT_MAIN.write_text(markdown, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-RU-LAB-EXTRACT-DIAG-01",
        "",
        f"Conclusion: {report['conclusion']}",
        f"Diagnostic type: {report['diagnostic_type']}",
        f"Files analyzed: {report['files_analyzed_count']}",
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
            "| Safe ID | Current type | Extractor | Confidence | OCR quality | Text | Cyrillic | Digits | Table-like | Lines | Cue count | Cue categories | Likely reason |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in report["safe_per_file_diagnostic_summary"]:
        categories = ", ".join(item["russian_lab_cue_categories_detected"]) or "none"
        lines.append(
            "| {safe_id} | {document_type_current} | {extractor} | {confidence_bucket} | {ocr_quality} | {text_length_bucket} | "
            "{cyrillic_density_bucket} | {digit_density_bucket} | {table_like_pattern_detected} | {line_count_bucket} | "
            "{russian_lab_cue_category_count} | {categories} | {likely_failure_reason} |".format(
                categories=categories,
                **item,
            )
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
    markdown_recommendations = {
        "no external OCR yet": "External OCR: not recommended yet",
        "no threshold changes": "Threshold changes: not recommended",
        "no auto-accept expansion": "Auto-accept expansion: not recommended",
        "no parser rewrite until text visibility is understood": "Parser rewrite: wait until text visibility is understood",
    }
    for item in report["recommended_not_to_do_yet"]:
        lines.append(f"- {markdown_recommendations.get(item, item)}")
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


def main() -> None:
    report = build_report()
    write_reports(report)
    print(json.dumps({"conclusion": report["conclusion"], "files_analyzed_count": report["files_analyzed_count"]}, indent=2))


if __name__ == "__main__":
    main()
