"""Phase67 local OCR preprocessing comparison prototype.

Diagnostic only. It renders candidate PDFs to temporary images, compares OCR
preprocessing variants locally, and writes only bucketed metrics by safe ID.
No OCR text, rendered images, filenames, paths, or PHI are written to public
reports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MEDAI_LOCAL_ONLY"] = "true"
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
os.environ["MEDAI_PRIVACY_AUDIT"] = "true"

import app.config as app_config
import privacy.outbound_gate as outbound_gate
from ocr_layout.ocr_capabilities import OcrCapabilityReport, get_ocr_capability_report
from ocr_layout.text_quality import assess_text_quality
from privacy.privacy_audit import phi_artifact_tracking_status, write_json


PHASE66_REPORT = ROOT / "reports" / "phase66_pdf_ocr_low_quality_diagnostic" / "phase66_pdf_ocr_low_quality_diagnostic_report.json"
PHASE57_PRIVATE_MAPPING = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "local_filename_mapping_PRIVATE.json"
FULL_CORPUS_INPUT = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase67_ocr_preprocessing_comparison"
JSON_REPORT = REPORT_DIR / "phase67_ocr_preprocessing_comparison_report.json"
MD_REPORT = REPORT_DIR / "phase67_ocr_preprocessing_comparison_report.md"

VARIANTS = ("baseline_render_150dpi", "higher_dpi_250dpi", "grayscale", "contrast_sharpen", "threshold_binarization")
_TESSERACT_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class VariantMetric:
    safe_file_id: str
    variant_name: str
    text_length_bucket: str
    ocr_quality_bucket: str
    improvement_bucket: str
    recommended_next_action: str
    warnings: list[str]
    error_category: str | None = None


def force_local_only_runtime() -> None:
    os.environ["MEDAI_LOCAL_ONLY"] = "true"
    os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
    os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
    os.environ["MEDAI_PRIVACY_AUDIT"] = "true"
    app_config.MEDAI_LOCAL_ONLY = True
    app_config.MEDAI_ALLOW_EXTERNAL_API = False
    app_config.MEDAI_REQUIRE_PII_SCRUB = True
    app_config.MEDAI_PRIVACY_AUDIT = True
    outbound_gate.MEDAI_LOCAL_ONLY = True
    outbound_gate.MEDAI_ALLOW_EXTERNAL_API = False
    outbound_gate.MEDAI_REQUIRE_PII_SCRUB = True


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_comparison(
    *,
    phase66_report_path: Path = PHASE66_REPORT,
    private_mapping_path: Path = PHASE57_PRIVATE_MAPPING,
    input_dir: Path = FULL_CORPUS_INPUT,
    report_dir: Path = REPORT_DIR,
    capability: OcrCapabilityReport | None = None,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    phase66 = load_json(phase66_report_path)
    private_mapping = load_json(private_mapping_path)
    candidate_ids = candidate_safe_ids(phase66)
    cap = capability if capability is not None else get_ocr_capability_report()
    variant_metrics: list[dict[str, Any]] = []
    per_file_summary: list[dict[str, Any]] = []
    for safe_id in candidate_ids:
        source_path = resolve_private_source_path(safe_id, private_mapping, input_dir)
        if source_path is None:
            metrics = missing_source_metrics(safe_id)
        else:
            metrics = compare_pdf_variants(safe_id, source_path, cap)
        metric_dicts = [metric.__dict__ for metric in metrics]
        variant_metrics.extend(metric_dicts)
        per_file_summary.append(summarize_file_metrics(safe_id, metric_dicts))

    meaningful_files = [
        item["safe_file_id"]
        for item in per_file_summary
        if item["best_improvement_bucket"] in {"meaningful", "strong"}
    ]
    all_files_meaningful = bool(candidate_ids) and len(meaningful_files) == len(candidate_ids)
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase67 OCR Preprocessing Comparison Prototype",
        "source_phase66_report": safe_repo_path(phase66_report_path),
        "candidate_source": "Phase66 prototype_recommendation.candidate_safe_file_ids",
        "candidate_count": len(candidate_ids),
        "candidate_safe_file_ids": candidate_ids,
        "variant_names": list(VARIANTS),
        "variant_metrics": variant_metrics,
        "per_file_summary": per_file_summary,
        "meaningful_improvement_file_count": len(meaningful_files),
        "meaningful_improvement_safe_file_ids": meaningful_files,
        "phase68_controlled_ocr_fallback_sandbox_recommended": all_files_meaningful,
        "recommended_next_action": (
            "recommend_phase68_controlled_ocr_fallback_sandbox"
            if all_files_meaningful
            else "keep_manual_review_boundary"
        ),
        "production_ocr_routing_changed": False,
        "production_extraction_logic_changed": False,
        "thresholds_changed": False,
        "safety_gates_changed": False,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "local_only_forced": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "rendered_images_written_to_reports": False,
        "ocr_text_written_to_public_reports": False,
        "private_mapping_used_locally": bool(private_mapping),
        "private_mapping_path_public": "[PRIVATE_MAPPING_REDACTED]",
        "tesseract_available": bool(cap.tesseract_available),
        "phi_artifact_check": phi_artifacts,
    }
    report["conclusion"] = conclusion_for(report)
    write_json(report_dir / JSON_REPORT.name, report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    if public_reports_contain_private_values(report_dir, private_mapping):
        report["raw_phi_logged_in_public_reports"] = True
        report["conclusion"] = "BLOCKED_PRIVACY_RISK"
        write_json(report_dir / JSON_REPORT.name, report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def candidate_safe_ids(phase66: dict[str, Any]) -> list[str]:
    recommendation = phase66.get("prototype_recommendation") or {}
    ids = recommendation.get("candidate_safe_file_ids") or []
    return [str(value) for value in ids]


def resolve_private_source_path(safe_id: str, private_mapping: dict[str, Any], input_dir: Path) -> Path | None:
    entry = (private_mapping.get("files") or {}).get(safe_id)
    if not isinstance(entry, dict):
        return None
    relative = str(entry.get("original_relative_path") or "")
    if not relative:
        return None
    candidate = (input_dir / relative).resolve()
    try:
        candidate.relative_to(input_dir.resolve())
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def missing_source_metrics(safe_id: str) -> list[VariantMetric]:
    return [
        VariantMetric(
            safe_file_id=safe_id,
            variant_name=variant,
            text_length_bucket="unavailable",
            ocr_quality_bucket="unavailable",
            improvement_bucket="unavailable",
            recommended_next_action="keep_manual_review_boundary",
            warnings=["private_source_unavailable"],
            error_category="private_source_unavailable",
        )
        for variant in VARIANTS
    ]


def compare_pdf_variants(safe_id: str, source_path: Path, capability: OcrCapabilityReport) -> list[VariantMetric]:
    if not capability.tesseract_available:
        return [
            VariantMetric(
                safe_file_id=safe_id,
                variant_name=variant,
                text_length_bucket="unavailable",
                ocr_quality_bucket="unavailable",
                improvement_bucket="unavailable",
                recommended_next_action="keep_manual_review_boundary",
                warnings=["tesseract_unavailable"],
                error_category="tesseract_unavailable",
            )
            for variant in VARIANTS
        ]
    try:
        scores = run_variant_scores(source_path, capability)
    except Exception as exc:  # noqa: BLE001
        return [
            VariantMetric(
                safe_file_id=safe_id,
                variant_name=variant,
                text_length_bucket="unavailable",
                ocr_quality_bucket="unavailable",
                improvement_bucket="unavailable",
                recommended_next_action="keep_manual_review_boundary",
                warnings=["variant_comparison_failed"],
                error_category=f"variant_comparison_failed:{exc.__class__.__name__}",
            )
            for variant in VARIANTS
        ]
    baseline = scores.get("baseline_render_150dpi", {"text_length": 0, "quality_score": 0.0})
    metrics: list[VariantMetric] = []
    for variant in VARIANTS:
        score = scores.get(variant, {"text_length": 0, "quality_score": 0.0, "warnings": ["variant_missing"]})
        improvement = improvement_bucket(score, baseline)
        metrics.append(
            VariantMetric(
                safe_file_id=safe_id,
                variant_name=variant,
                text_length_bucket=text_length_bucket(int(score.get("text_length", 0))),
                ocr_quality_bucket=ocr_quality_bucket(float(score.get("quality_score", 0.0))),
                improvement_bucket=improvement,
                recommended_next_action=next_action_for_improvement(improvement),
                warnings=list(score.get("warnings") or []),
                error_category=score.get("error_category"),
            )
        )
    return metrics


def run_variant_scores(source_path: Path, capability: OcrCapabilityReport) -> dict[str, dict[str, Any]]:
    try:
        import fitz  # type: ignore
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception as exc:  # noqa: BLE001
        return {variant: failed_score(f"dependency_unavailable:{exc.__class__.__name__}") for variant in VARIANTS}

    scores: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="phase67_ocr_compare_") as tmp:
        tmp_dir = Path(tmp)
        doc = fitz.open(str(source_path))
        try:
            page = doc.load_page(0)
            base_150 = render_page(page, dpi=150, output_path=tmp_dir / "baseline_150.png")
            base_250 = render_page(page, dpi=250, output_path=tmp_dir / "higher_250.png")
        finally:
            doc.close()
        scores["baseline_render_150dpi"] = score_image(base_150, capability)
        scores["higher_dpi_250dpi"] = score_image(base_250, capability)

        image = Image.open(base_250)
        try:
            grayscale_path = tmp_dir / "grayscale.png"
            grayscale = ImageOps.grayscale(image)
            grayscale.save(grayscale_path)
            scores["grayscale"] = score_image(grayscale_path, capability)

            contrast_path = tmp_dir / "contrast_sharpen.png"
            contrast = ImageEnhance.Contrast(grayscale).enhance(1.8).filter(ImageFilter.SHARPEN)
            contrast.save(contrast_path)
            scores["contrast_sharpen"] = score_image(contrast_path, capability)

            threshold_path = tmp_dir / "threshold.png"
            threshold = grayscale.point(lambda pixel: 255 if pixel > 180 else 0)
            threshold.save(threshold_path)
            scores["threshold_binarization"] = score_image(threshold_path, capability)
        finally:
            image.close()
    return scores


def render_page(page: Any, *, dpi: int, output_path: Path) -> Path:
    zoom = dpi / 72.0
    matrix = page.parent.Matrix(zoom, zoom) if hasattr(page.parent, "Matrix") else None
    if matrix is None:
        import fitz  # type: ignore

        matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(str(output_path))
    return output_path


def score_image(image_path: Path, capability: OcrCapabilityReport) -> dict[str, Any]:
    text, warnings, error_category = run_tesseract(image_path, capability)
    quality = assess_text_quality(text)
    return {
        "text_length": len(text.strip()),
        "quality_score": quality.score,
        "quality_band": quality.band,
        "warnings": list(warnings) + list(quality.warnings),
        "error_category": error_category,
    }


def run_tesseract(image_path: Path, capability: OcrCapabilityReport) -> tuple[str, list[str], str | None]:
    binary = capability.tesseract_path or "tesseract"
    language = "rus+eng" if capability.russian_available and capability.english_available else "eng"
    try:
        result = subprocess.run(
            [binary, str(image_path), "stdout", "-l", language],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TESSERACT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", ["ocr_timeout"], "ocr_timeout"
    except (FileNotFoundError, OSError) as exc:
        return "", ["ocr_invocation_failed"], f"ocr_invocation_failed:{exc.__class__.__name__}"
    warnings: list[str] = []
    error_category = None
    if result.returncode != 0:
        warnings.append("tesseract_nonzero_returncode")
        error_category = f"tesseract_returncode_{result.returncode}"
    return (result.stdout or "").strip(), warnings, error_category


def failed_score(error_category: str) -> dict[str, Any]:
    return {
        "text_length": 0,
        "quality_score": 0.0,
        "quality_band": "empty",
        "warnings": [error_category],
        "error_category": error_category,
    }


def text_length_bucket(length: int) -> str:
    if length <= 0:
        return "empty"
    if length < 80:
        return "very_short"
    if length < 250:
        return "short"
    if length < 800:
        return "moderate"
    return "long"


def ocr_quality_bucket(score: float) -> str:
    if score <= 0.0:
        return "empty"
    if score < 0.35:
        return "poor"
    if score < 0.55:
        return "weak"
    if score < 0.72:
        return "usable_with_review"
    return "good"


def improvement_bucket(score: dict[str, Any], baseline: dict[str, Any]) -> str:
    length = int(score.get("text_length", 0))
    base_length = int(baseline.get("text_length", 0))
    quality = float(score.get("quality_score", 0.0))
    base_quality = float(baseline.get("quality_score", 0.0))
    if length <= 0 and quality <= 0:
        return "none"
    length_gain = length - base_length
    quality_gain = quality - base_quality
    if length_gain >= 500 or quality_gain >= 0.30:
        return "strong"
    if length_gain >= 150 or quality_gain >= 0.15:
        return "meaningful"
    if length_gain >= 40 or quality_gain >= 0.05:
        return "minor"
    return "none"


def next_action_for_improvement(bucket: str) -> str:
    if bucket in {"meaningful", "strong"}:
        return "candidate_for_phase68_sandbox"
    return "keep_manual_review_boundary"


def summarize_file_metrics(safe_id: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    order = {"unavailable": -1, "none": 0, "minor": 1, "meaningful": 2, "strong": 3}
    best = max(metrics, key=lambda item: order.get(str(item.get("improvement_bucket")), -1), default={})
    return {
        "safe_file_id": safe_id,
        "best_variant": best.get("variant_name"),
        "best_text_length_bucket": best.get("text_length_bucket"),
        "best_ocr_quality_bucket": best.get("ocr_quality_bucket"),
        "best_improvement_bucket": best.get("improvement_bucket"),
        "recommended_next_action": best.get("recommended_next_action", "keep_manual_review_boundary"),
    }


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged_in_public_reports"] or not report["phi_artifact_check"].get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if report["candidate_count"] == 0:
        return "no_phase66_ocr_preprocessing_candidates"
    if report["phase68_controlled_ocr_fallback_sandbox_recommended"]:
        return "phase68_sandbox_recommended"
    return "manual_review_boundary_retained"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase67 OCR Preprocessing Comparison Prototype",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Meaningful improvement files: `{report['meaningful_improvement_file_count']}`",
        f"- Phase68 sandbox recommended: `{report['phase68_controlled_ocr_fallback_sandbox_recommended']}`",
        f"- Recommended next action: `{report['recommended_next_action']}`",
        f"- production_ocr_routing_changed: `{report['production_ocr_routing_changed']}`",
        f"- production_extraction_logic_changed: `{report['production_extraction_logic_changed']}`",
        f"- thresholds_changed: `{report['thresholds_changed']}`",
        f"- safety_gates_changed: `{report['safety_gates_changed']}`",
        f"- external_api_used: `{report['external_api_used']}`",
        f"- raw_phi_logged_in_public_reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- conclusion: `{report['conclusion']}`",
        "",
        "## Per-File Summary",
        "",
        "| Safe File ID | Best Variant | Text Length Bucket | OCR Quality Bucket | Improvement | Next Action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["per_file_summary"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id')}`",
                    f"`{item.get('best_variant')}`",
                    f"`{item.get('best_text_length_bucket')}`",
                    f"`{item.get('best_ocr_quality_bucket')}`",
                    f"`{item.get('best_improvement_bucket')}`",
                    f"`{item.get('recommended_next_action')}` |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Variant Metrics",
            "",
            "| Safe File ID | Variant | Text Length Bucket | OCR Quality Bucket | Improvement | Next Action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["variant_metrics"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id')}`",
                    f"`{item.get('variant_name')}`",
                    f"`{item.get('text_length_bucket')}`",
                    f"`{item.get('ocr_quality_bucket')}`",
                    f"`{item.get('improvement_bucket')}`",
                    f"`{item.get('recommended_next_action')}` |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Diagnostic only; production OCR routing is unchanged.",
            "- Rendered page images are temporary and are not written to reports.",
            "- OCR text is not written to public reports.",
            "- Public reports use safe file IDs only.",
        ]
    )
    return "\n".join(lines) + "\n"


def public_reports_contain_private_values(report_dir: Path, private_mapping: dict[str, Any]) -> bool:
    files = private_mapping.get("files", {}) if isinstance(private_mapping, dict) else {}
    values: list[str] = []
    for entry in files.values():
        for key in ("original_filename", "original_relative_path"):
            value = str(entry.get(key) or "")
            if value:
                values.append(value)
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in [report_dir / JSON_REPORT.name, report_dir / MD_REPORT.name]
        if path.exists()
    )
    return any(value and value in public_text for value in values)


def safe_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    report = run_comparison()
    print("MedAI Phase67 OCR preprocessing comparison complete.")
    print(f"candidate_count: {report['candidate_count']}")
    print(f"meaningful_improvement_file_count: {report['meaningful_improvement_file_count']}")
    print(f"phase68_controlled_ocr_fallback_sandbox_recommended: {report['phase68_controlled_ocr_fallback_sandbox_recommended']}")
    print(f"recommended_next_action: {report['recommended_next_action']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
