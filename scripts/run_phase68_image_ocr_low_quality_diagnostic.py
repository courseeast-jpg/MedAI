"""Phase68 image OCR low-quality diagnostic.

Diagnostic only. This script inspects Phase57 safe metadata for the
image_ocr_low_quality cluster and, when private local files are available,
uses local image metadata only to bucket likely causes. It never writes raw
filenames, paths, images, OCR text, extracted text, or PHI to public reports.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
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
from privacy.privacy_audit import phi_artifact_tracking_status, write_json


PHASE57_REPORT = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_inventory_audit_report.json"
PHASE57_CLUSTERS = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "phase57_full_corpus_problem_clusters.json"
PHASE57_PRIVATE_MAPPING = ROOT / "reports" / "phase57_full_corpus_inventory_audit" / "local_filename_mapping_PRIVATE.json"
FULL_CORPUS_INPUT = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase68_image_ocr_low_quality_diagnostic"
JSON_REPORT = REPORT_DIR / "phase68_image_ocr_low_quality_diagnostic_report.json"
MD_REPORT = REPORT_DIR / "phase68_image_ocr_low_quality_diagnostic_report.md"

TARGET_CLUSTER = "image_ocr_low_quality"


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


def run_diagnostic(
    *,
    phase57_report_path: Path = PHASE57_REPORT,
    phase57_clusters_path: Path = PHASE57_CLUSTERS,
    private_mapping_path: Path = PHASE57_PRIVATE_MAPPING,
    input_dir: Path = FULL_CORPUS_INPUT,
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    phase57 = load_json(phase57_report_path)
    clusters = load_json(phase57_clusters_path)
    private_mapping = load_json(private_mapping_path)
    target_ids = [str(value) for value in clusters.get(TARGET_CLUSTER) or []]
    records = select_target_records(phase57, target_ids)
    diagnostics = [
        diagnose_record(
            item,
            source_path=resolve_private_source_path(str(item.get("safe_file_id") or item.get("file_id")), private_mapping, input_dir),
        )
        for item in records
    ]
    root_cause_buckets = bucket_counts(diagnostics)
    preprocessing = preprocessing_recommendation(diagnostics)
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase68 Image OCR Low-Quality Diagnostic",
        "source_phase57_report": safe_repo_path(phase57_report_path),
        "source_phase57_clusters": safe_repo_path(phase57_clusters_path),
        "target_cluster": TARGET_CLUSTER,
        "target_count": len(target_ids),
        "diagnosed_count": len(diagnostics),
        "missing_target_ids": [safe_id for safe_id in target_ids if safe_id not in {d["safe_file_id"] for d in diagnostics}],
        "root_cause_buckets": root_cause_buckets,
        "per_file_diagnostics": diagnostics,
        "ocr_quality_band_distribution": dict(Counter(str(item.get("ocr_quality_band") or "unknown") for item in diagnostics)),
        "narrow_image_preprocessing_prototype_justified": preprocessing["justified"],
        "prototype_recommendation": preprocessing,
        "manual_review_boundary_preserved": True,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "production_ocr_routing_changed": False,
        "production_extraction_logic_changed": False,
        "thresholds_changed": False,
        "safety_gates_changed": False,
        "local_only_forced": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "images_written_to_public_reports": False,
        "ocr_text_written_to_public_reports": False,
        "private_mapping_used_locally": bool(private_mapping),
        "private_mapping_path_public": "[PRIVATE_MAPPING_REDACTED]",
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


def select_target_records(phase57: dict[str, Any], target_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(target_ids)
    records = []
    for item in phase57.get("results", []):
        safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
        if safe_id in wanted:
            records.append(item)
    return sorted(records, key=lambda item: str(item.get("safe_file_id") or item.get("file_id") or ""))


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


def diagnose_record(item: dict[str, Any], *, source_path: Path | None) -> dict[str, Any]:
    reason_codes = set(str(code) for code in item.get("classification_reason_codes") or item.get("reason_codes") or [])
    safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
    ocr_band = str(item.get("ocr_quality_band") or item.get("ocr_status") or "unknown")
    empty = bool(item.get("empty_extraction_flag")) or "empty_or_near_empty_text" in reason_codes
    metadata = inspect_image_metadata(source_path)
    root_cause = classify_root_cause(
        ocr_band=ocr_band,
        empty=empty,
        reason_codes=reason_codes,
        metadata=metadata,
        file_size_bytes=int(item.get("file_size_bytes") or 0),
    )
    return {
        "safe_file_id": safe_id,
        "file_type": "image",
        "status": item.get("status"),
        "ocr_quality_band": ocr_band,
        "ocr_quality_score_bucket": score_bucket(item.get("ocr_quality_score")),
        "selected_ocr_engine": item.get("selected_ocr_engine") or item.get("ocr_engine") or "unknown",
        "confidence_bucket": score_bucket(item.get("confidence")),
        "empty_extraction_flag": empty,
        "local_image_metadata_available": metadata["available"],
        "dimension_bucket": metadata["dimension_bucket"],
        "contrast_bucket": metadata["contrast_bucket"],
        "brightness_bucket": metadata["brightness_bucket"],
        "frame_count_bucket": metadata["frame_count_bucket"],
        "root_cause_bucket": root_cause["bucket"],
        "root_cause_reason_code": root_cause["reason_code"],
        "recommended_operator_action": root_cause["operator_action"],
        "prototype_signal": root_cause["prototype_signal"],
        "classification_reason_codes": sorted(reason_codes),
        "review_reason_codes": list(item.get("review_reason_codes") or []),
        "external_api_used": bool(item.get("external_api_used")),
    }


def inspect_image_metadata(source_path: Path | None) -> dict[str, Any]:
    base = {
        "available": False,
        "dimension_bucket": "unavailable",
        "contrast_bucket": "unavailable",
        "brightness_bucket": "unavailable",
        "frame_count_bucket": "unavailable",
    }
    if source_path is None:
        return base
    try:
        from PIL import Image, ImageOps, ImageSequence, ImageStat
    except Exception:  # noqa: BLE001
        return {**base, "available": False}
    try:
        with Image.open(source_path) as image:
            frame_count = sum(1 for _ in ImageSequence.Iterator(image))
            width, height = image.size
            sample = ImageOps.grayscale(ImageOps.exif_transpose(image.copy()))
            sample.thumbnail((512, 512))
            stat = ImageStat.Stat(sample)
            brightness = float(stat.mean[0]) if stat.mean else 0.0
            contrast = float(stat.stddev[0]) if stat.stddev else 0.0
    except Exception:  # noqa: BLE001
        return {**base, "available": False, "dimension_bucket": "unreadable"}
    megapixels = (width * height) / 1_000_000.0
    return {
        "available": True,
        "dimension_bucket": dimension_bucket(megapixels),
        "contrast_bucket": contrast_bucket(contrast),
        "brightness_bucket": brightness_bucket(brightness),
        "frame_count_bucket": frame_count_bucket(frame_count),
    }


def classify_root_cause(
    *,
    ocr_band: str,
    empty: bool,
    reason_codes: set[str],
    metadata: dict[str, Any],
    file_size_bytes: int,
) -> dict[str, str | bool]:
    if not metadata["available"]:
        return {
            "bucket": "image_metadata_unavailable",
            "reason_code": "image_metadata_unavailable",
            "operator_action": "Keep manual review boundary; source image could not be inspected locally without exposing private data.",
            "prototype_signal": False,
        }
    if metadata["dimension_bucket"] in {"tiny", "low_resolution"}:
        return {
            "bucket": "source_resolution_likely_too_low",
            "reason_code": "low_resolution_image_source",
            "operator_action": "Manual review boundary; request higher-resolution source image.",
            "prototype_signal": False,
        }
    if metadata["contrast_bucket"] == "low_contrast" or metadata["brightness_bucket"] in {"very_dark", "very_bright"}:
        return {
            "bucket": "local_preprocessing_candidate",
            "reason_code": "contrast_or_brightness_preprocessing_candidate",
            "operator_action": "Manual review boundary; a future local-only preprocessing comparison may be justified.",
            "prototype_signal": True,
        }
    if empty or ocr_band == "empty" or "empty_or_near_empty_text" in reason_codes:
        return {
            "bucket": "image_ocr_empty_output",
            "reason_code": "image_ocr_empty_output_without_obvious_metadata_fix",
            "operator_action": "Manual review boundary; re-scan/re-capture before relying on extraction.",
            "prototype_signal": file_size_bytes >= 150_000,
        }
    if ocr_band == "poor_ocr" or "poor_input_ocr" in reason_codes:
        return {
            "bucket": "image_ocr_noise_or_capture_quality",
            "reason_code": "poor_image_ocr_quality",
            "operator_action": "Manual review boundary; source capture quality likely limits automation.",
            "prototype_signal": file_size_bytes >= 250_000,
        }
    return {
        "bucket": "manual_review_boundary",
        "reason_code": "insufficient_safe_signal_for_ocr_change",
        "operator_action": "Keep review_ocr_quality and inspect manually.",
        "prototype_signal": False,
    }


def preprocessing_recommendation(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = [item["safe_file_id"] for item in diagnostics if item.get("prototype_signal")]
    strong_path = len(candidate_ids) >= max(1, math.ceil(len(diagnostics) * 0.6)) if diagnostics else False
    return {
        "justified": bool(candidate_ids),
        "strong_improvement_path_exists": strong_path,
        "candidate_count": len(candidate_ids),
        "candidate_safe_file_ids": candidate_ids,
        "scope": "diagnostic_only_local_image_preprocessing_comparison",
        "rationale": (
            "Some image OCR low-quality files have safe metadata consistent with local contrast/brightness or capture-quality preprocessing candidates. "
            "A future prototype may compare local image preprocessing variants, but production OCR remains unchanged."
            if candidate_ids
            else "The image OCR low-quality files do not show a strong safe metadata signal for preprocessing; keep the manual-review boundary."
        ),
    }


def bucket_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in diagnostics:
        counts[str(item.get("root_cause_bucket") or "unknown")] += 1
    return dict(sorted(counts.items()))


def dimension_bucket(megapixels: float) -> str:
    if megapixels < 0.25:
        return "tiny"
    if megapixels < 1.0:
        return "low_resolution"
    if megapixels < 8.0:
        return "standard"
    return "large"


def contrast_bucket(stddev: float) -> str:
    if stddev < 18.0:
        return "low_contrast"
    if stddev < 45.0:
        return "moderate"
    return "high_contrast"


def brightness_bucket(mean: float) -> str:
    if mean < 45.0:
        return "very_dark"
    if mean > 225.0:
        return "very_bright"
    return "normal"


def frame_count_bucket(frame_count: int) -> str:
    if frame_count <= 0:
        return "unknown"
    if frame_count == 1:
        return "single_frame"
    return "multi_frame"


def score_bucket(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if score <= 0.0:
        return "empty_or_zero"
    if score < 0.35:
        return "poor"
    if score < 0.55:
        return "weak"
    if score < 0.72:
        return "usable_with_review"
    return "good"


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged_in_public_reports"] or not report["phi_artifact_check"].get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if report["target_count"] == 0:
        return "no_image_ocr_low_quality_targets"
    if report["missing_target_ids"]:
        return "ready_with_missing_targets"
    if report["prototype_recommendation"]["strong_improvement_path_exists"]:
        return "image_preprocessing_prototype_justified"
    return "manual_review_boundary_retained"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase68 Image OCR Low-Quality Diagnostic",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Target cluster: `{report['target_cluster']}`",
        f"- Target count: `{report['target_count']}`",
        f"- Diagnosed count: `{report['diagnosed_count']}`",
        f"- Root-cause buckets: `{json.dumps(report['root_cause_buckets'], sort_keys=True)}`",
        f"- Narrow image preprocessing prototype justified: `{report['narrow_image_preprocessing_prototype_justified']}`",
        f"- Production OCR should change yet: `{report['production_ocr_should_change_yet']}`",
        f"- Manual-review boundary preserved: `{report['manual_review_boundary_preserved']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Per-File Diagnostics",
        "",
        "| Safe File ID | Status | OCR Band | Dimension | Contrast | Brightness | Root Cause | Prototype Signal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["per_file_diagnostics"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id')}`",
                    f"`{item.get('status')}`",
                    f"`{item.get('ocr_quality_band')}`",
                    f"`{item.get('dimension_bucket')}`",
                    f"`{item.get('contrast_bucket')}`",
                    f"`{item.get('brightness_bucket')}`",
                    f"`{item.get('root_cause_bucket')}`",
                    f"`{item.get('prototype_signal')}` |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Prototype Recommendation",
            "",
            f"- Scope: `{report['prototype_recommendation']['scope']}`",
            f"- Candidate count: `{report['prototype_recommendation']['candidate_count']}`",
            f"- Candidate safe IDs: `{', '.join(report['prototype_recommendation']['candidate_safe_file_ids'])}`",
            f"- Strong improvement path exists: `{report['prototype_recommendation']['strong_improvement_path_exists']}`",
            f"- Rationale: {report['prototype_recommendation']['rationale']}",
            "",
            "## Safety",
            "",
            "- This phase is diagnostic only.",
            "- No production OCR routing, extraction logic, thresholds, or safety gates changed.",
            "- No image files, OCR text, extracted text, filenames, or paths are written to public reports.",
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
    report = run_diagnostic()
    print("MedAI Phase68 image OCR low-quality diagnostic complete.")
    print(f"target_count: {report['target_count']}")
    print(f"diagnosed_count: {report['diagnosed_count']}")
    print(f"root_cause_buckets: {report['root_cause_buckets']}")
    print(f"narrow_image_preprocessing_prototype_justified: {report['narrow_image_preprocessing_prototype_justified']}")
    print(f"production_ocr_should_change_yet: {report['production_ocr_should_change_yet']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
