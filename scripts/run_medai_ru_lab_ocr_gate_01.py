from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TEXT_VIS_REPORT = ROOT / "reports" / "medai_ru_lab_text_vis_01" / "medai_ru_lab_text_vis_01_report.json"
REPORT_DIR = ROOT / "reports" / "medai_ru_lab_ocr_gate_01"
REPORT_JSON = REPORT_DIR / "medai_ru_lab_ocr_gate_01_report.json"
REPORT_MD = REPORT_DIR / "medai_ru_lab_ocr_gate_01_report.md"
REPORT_MAIN = REPORT_DIR / "MEDAI_RU_LAB_OCR_GATE_01.md"

_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")


def bucket_density(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio < 0.05:
        return "low"
    if ratio < 0.25:
        return "medium"
    return "high"


def bucket_cyrillic_density(text: str | None) -> str:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return "none"
    return bucket_density(len(_CYRILLIC_RE.findall(compact)) / len(compact))


def list_tesseract_languages() -> tuple[bool, set[str]]:
    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:
        return False, set()
    if completed.returncode != 0:
        return False, set()
    languages = {
        line.strip().lower()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip() and not line.lower().startswith("list of available")
    }
    return True, languages


def installed_language_buckets(languages: set[str]) -> list[str]:
    buckets: list[str] = []
    has_english = "eng" in languages
    has_russian = any(lang in {"rus", "rus_old"} or lang.startswith("rus") for lang in languages)
    if has_english:
        buckets.append("english_available")
    if has_russian:
        buckets.append("russian_available")
    if has_english and has_russian:
        buckets.append("multilingual_available")
    if not buckets:
        buckets.append("unknown")
    return buckets


def safe_ocr_gate_decision(
    *,
    native_text_length_bucket: str,
    digit_density_bucket: str,
    cyrillic_density_bucket: str,
    table_like_pattern_detected: bool,
    current_ocr_skipped: bool,
) -> dict[str, Any]:
    has_substantial_text = native_text_length_bucket in {"medium", "long"}
    has_table_digits = digit_density_bucket in {"medium", "high"} and bool(table_like_pattern_detected)
    missing_cyrillic = cyrillic_density_bucket == "none"
    gate_needed = bool(has_substantial_text and has_table_digits and missing_cyrillic and current_ocr_skipped)
    reason = "not_recommended"
    if gate_needed:
        reason = "native_numeric_table_text_without_cyrillic"
    elif not has_substantial_text:
        reason = "native_text_not_substantial"
    elif not has_table_digits:
        reason = "numeric_table_signal_not_strong"
    elif not missing_cyrillic:
        reason = "language_text_visible"
    elif not current_ocr_skipped:
        reason = "ocr_already_attempted"
    return {
        "native_text_length_bucket": native_text_length_bucket,
        "digit_density_bucket": digit_density_bucket,
        "cyrillic_density_bucket": cyrillic_density_bucket,
        "table_like_pattern_detected": bool(table_like_pattern_detected),
        "current_ocr_skipped": bool(current_ocr_skipped),
        "proposed_gate_reason": reason,
        "cyrillic_visibility_ocr_gate_needed": gate_needed,
        "safe_mode": "review_only",
        "auto_acceptance_allowed": False,
    }


def synthetic_cyrillic_ocr_probe(*, tesseract_available: bool, languages: set[str]) -> dict[str, Any]:
    has_russian = any(lang in {"rus", "rus_old"} or lang.startswith("rus") for lang in languages)
    if not tesseract_available:
        return {
            "attempted": False,
            "cyrillic_detected_bucket": "unavailable",
            "raw_text_recorded": False,
        }
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return {
            "attempted": False,
            "cyrillic_detected_bucket": "unavailable",
            "raw_text_recorded": False,
        }

    language_args = ["-l", "rus"] if has_russian else ["-l", "eng"]
    synthetic_text = "\u0410\u043d\u0430\u043b\u0438\u0437 \u043a\u0440\u043e\u0432\u0438 5.1 \u043d\u043e\u0440\u043c\u0430"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "synthetic.png"
            output_base = temp_path / "ocr_output"
            image = Image.new("RGB", (900, 140), color="white")
            draw = ImageDraw.Draw(image)
            font = load_cyrillic_font(size=36)
            draw.text((24, 42), synthetic_text, fill="black", font=font)
            image.save(image_path)
            completed = subprocess.run(
                ["tesseract", str(image_path), str(output_base), *language_args, "--psm", "6"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if completed.returncode != 0:
                return {
                    "attempted": True,
                    "cyrillic_detected_bucket": "unavailable",
                    "raw_text_recorded": False,
                }
            text_path = output_base.with_suffix(".txt")
            ocr_text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path.exists() else ""
            return {
                "attempted": True,
                "cyrillic_detected_bucket": bucket_cyrillic_density(ocr_text),
                "raw_text_recorded": False,
            }
    except Exception:
        return {
            "attempted": False,
            "cyrillic_detected_bucket": "unavailable",
            "raw_text_recorded": False,
        }


def load_cyrillic_font(size: int):
    from PIL import ImageFont

    for candidate in ("arial.ttf", "segoeui.ttf", "times.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_current_gate_analysis(report_path: Path = TEXT_VIS_REPORT) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = list(data.get("safe_per_file_diagnostic_summary") or [])
    analyses: list[dict[str, Any]] = []
    for item in summaries:
        analyses.append(
            safe_ocr_gate_decision(
                native_text_length_bucket=str(item.get("text_length_bucket") or "none"),
                digit_density_bucket=str(item.get("digit_density_bucket") or "none"),
                cyrillic_density_bucket=str(item.get("cyrillic_density_bucket") or "none"),
                table_like_pattern_detected=bool(item.get("table_like_pattern_detected", False)),
                current_ocr_skipped=not bool(item.get("ocr_attempted", False)),
            )
        )
    return analyses


def build_report() -> dict[str, Any]:
    tesseract_available, languages = list_tesseract_languages()
    language_buckets = installed_language_buckets(languages)
    synthetic_probe = synthetic_cyrillic_ocr_probe(tesseract_available=tesseract_available, languages=languages)
    gate_analyses = load_current_gate_analysis()
    gate_needed = any(item["cyrillic_visibility_ocr_gate_needed"] for item in gate_analyses)
    reason_counts = Counter(item["proposed_gate_reason"] for item in gate_analyses)
    likely_primary = reason_counts.most_common(1)[0][0] if reason_counts else "unknown"
    root_candidates = root_cause_candidates(gate_needed=gate_needed, russian_available="russian_available" in language_buckets or "multilingual_available" in language_buckets)
    return {
        "conclusion": "medai_ru_lab_ocr_gate_01_completed",
        "diagnostic_type": "cyrillic_visibility_ocr_gate",
        "baseline_ru_lab_text_vis_commit_short": "bb0de4b",
        "tesseract_available": bool(tesseract_available),
        "installed_language_buckets": language_buckets,
        "russian_ocr_language_available": bool("russian_available" in language_buckets or "multilingual_available" in language_buckets),
        "synthetic_cyrillic_probe_attempted": bool(synthetic_probe["attempted"]),
        "synthetic_cyrillic_probe_result_bucket": synthetic_probe["cyrillic_detected_bucket"],
        "synthetic_cyrillic_ocr_probe": synthetic_probe,
        "current_gate_analysis": gate_analyses,
        "proposed_future_gate": {
            "cyrillic_visibility_ocr_gate_needed": gate_needed,
            "trigger_condition_summary": "medium_or_long_numeric_table_text_with_zero_cyrillic_and_ocr_skipped",
            "safe_mode": "review_only",
            "auto_acceptance_allowed": False,
        },
        "cyrillic_visibility_ocr_gate_needed": gate_needed,
        "root_cause_candidates_ranked": root_candidates,
        "likely_primary_cause": likely_primary,
        "proposed_future_gate_summary": (
            "Evaluate a future review-only local OCR gate when native PDF text is medium/long, table-like, numeric, "
            "and has zero Cyrillic visibility."
        ),
        "auto_acceptance_changed": False,
        "confidence_thresholds_changed": False,
        "confidence_scoring_changed": False,
        "production_ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "external_api_enabled": False,
        "cloud_api_used": False,
        "extraction_parser_changed": False,
        "lab_value_parser_added": False,
        "clinical_logic_changed": False,
        "clinical_interpretation_added": False,
        "medication_advice_added": False,
        "ddi_logic_changed": False,
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
        "recommended_next_block": "MEDAI-RU-LAB-OCR-GATE-02 - Local Cyrillic OCR Gate Implementation, only if diagnostic supports it",
    }


def root_cause_candidates(*, gate_needed: bool, russian_available: bool) -> list[str]:
    if gate_needed and russian_available:
        return [
            "numeric_table_readability_masks_missing_cyrillic_text",
            "current_ocr_gate_lacks_language_visibility_check",
            "native_pdf_text_layer_missing_cyrillic",
            "classifier_receives_numeric_only_text",
        ]
    if gate_needed:
        return [
            "numeric_table_readability_masks_missing_cyrillic_text",
            "tesseract_russian_language_missing",
            "current_ocr_gate_lacks_language_visibility_check",
            "native_pdf_text_layer_missing_cyrillic",
        ]
    return ["unknown"]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-RU-LAB-OCR-GATE-01",
        "",
        f"Conclusion: {report['conclusion']}",
        f"Diagnostic type: {report['diagnostic_type']}",
        f"Baseline text visibility commit: {report['baseline_ru_lab_text_vis_commit_short']}",
        f"Tesseract available: {str(report['tesseract_available']).lower()}",
        f"Russian OCR language available: {str(report['russian_ocr_language_available']).lower()}",
        f"Synthetic Cyrillic probe attempted: {str(report['synthetic_cyrillic_probe_attempted']).lower()}",
        f"Synthetic Cyrillic probe result bucket: {report['synthetic_cyrillic_probe_result_bucket']}",
        f"Cyrillic visibility OCR gate needed: {str(report['cyrillic_visibility_ocr_gate_needed']).lower()}",
        "",
        "## Installed Language Buckets",
        "",
    ]
    for bucket in report["installed_language_buckets"]:
        lines.append(f"- {bucket}")
    lines.extend(["", "## Current Gate Analysis", ""])
    lines.extend(
        [
            "| Native text | Digits | Cyrillic | Table-like | OCR skipped | Proposed reason | Gate needed |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in report["current_gate_analysis"]:
        lines.append(
            "| {native_text_length_bucket} | {digit_density_bucket} | {cyrillic_density_bucket} | "
            "{table_like_pattern_detected} | {current_ocr_skipped} | {proposed_gate_reason} | "
            "{cyrillic_visibility_ocr_gate_needed} |".format(**item)
        )
    lines.extend(["", "## Root Cause Candidates", ""])
    for index, candidate in enumerate(report["root_cause_candidates_ranked"], start=1):
        lines.append(f"{index}. {candidate}")
    lines.extend(
        [
            "",
            f"Likely primary cause: {report['likely_primary_cause']}",
            "",
            "## Proposed Future Gate",
            "",
            f"Summary: {report['proposed_future_gate_summary']}",
            f"Trigger: {report['proposed_future_gate']['trigger_condition_summary']}",
            f"Safe mode: {report['proposed_future_gate']['safe_mode']}",
            f"Auto-acceptance allowed: {str(report['proposed_future_gate']['auto_acceptance_allowed']).lower()}",
            "",
            "## Recommendation",
            "",
            f"Recommended next block: {report['recommended_next_block']}",
            "",
            "## Safety",
            "",
            "- Auto-acceptance changed: false",
            "- Confidence thresholds changed: false",
            "- Confidence scoring changed: false",
            "- Production OCR routing changed: false",
            "- OCR engine changed: false",
            "- External API enabled: false",
            "- Cloud API used: false",
            "- Extraction parser changed: false",
            "- Lab value parser added: false",
            "- Clinical logic changed: false",
            "- Clinical interpretation added: false",
            "- Medication advice added: false",
            "- DDI logic changed: false",
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
                "tesseract_available": report["tesseract_available"],
                "russian_ocr_language_available": report["russian_ocr_language_available"],
                "cyrillic_visibility_ocr_gate_needed": report["cyrillic_visibility_ocr_gate_needed"],
                "likely_primary_cause": report["likely_primary_cause"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
