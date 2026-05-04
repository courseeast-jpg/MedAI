"""Phase66 PDF OCR low-quality follow-up diagnostic.

This is report-only analysis over Phase57 safe metadata. It does not read raw
PDF text, does not render pages, and does not alter OCR/extraction behavior.
"""

from __future__ import annotations

import json
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
REPORT_DIR = ROOT / "reports" / "phase66_pdf_ocr_low_quality_diagnostic"
JSON_REPORT = REPORT_DIR / "phase66_pdf_ocr_low_quality_diagnostic_report.json"
MD_REPORT = REPORT_DIR / "phase66_pdf_ocr_low_quality_diagnostic_report.md"

TARGET_CLUSTER = "pdf_ocr_low_quality"


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
    report_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    force_local_only_runtime()
    report_dir.mkdir(parents=True, exist_ok=True)
    phase57 = load_json(phase57_report_path)
    clusters = load_json(phase57_clusters_path)
    private_mapping = load_json(private_mapping_path)
    target_ids = list(clusters.get(TARGET_CLUSTER) or [])
    records = select_target_records(phase57, target_ids)
    diagnostics = [diagnose_record(item) for item in records]
    root_cause_buckets = bucket_counts(diagnostics)
    preprocessing = preprocessing_recommendation(diagnostics)
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase66 PDF OCR Low-Quality Follow-up Diagnostic",
        "source_phase57_report": safe_repo_path(phase57_report_path),
        "source_phase57_clusters": safe_repo_path(phase57_clusters_path),
        "target_cluster": TARGET_CLUSTER,
        "target_count": len(target_ids),
        "diagnosed_count": len(diagnostics),
        "missing_target_ids": [safe_id for safe_id in target_ids if safe_id not in {d["safe_file_id"] for d in diagnostics}],
        "root_cause_buckets": root_cause_buckets,
        "per_file_diagnostics": diagnostics,
        "ocr_engine_distribution": dict(Counter(str(item.get("selected_ocr_engine") or "unknown") for item in diagnostics)),
        "ocr_quality_band_distribution": dict(Counter(str(item.get("ocr_quality_band") or "unknown") for item in diagnostics)),
        "document_type_distribution": dict(Counter(str(item.get("document_type") or "unknown") for item in diagnostics)),
        "page_count_distribution": dict(Counter(str(item.get("page_count") or "unknown") for item in diagnostics)),
        "narrow_ocr_preprocessing_prototype_justified": preprocessing["justified"],
        "prototype_recommendation": preprocessing,
        "manual_review_boundary_preserved": True,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "thresholds_changed": False,
        "safety_gates_changed": False,
        "local_only_forced": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_mapping_used_for_leak_check_only": bool(private_mapping),
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


def diagnose_record(item: dict[str, Any]) -> dict[str, Any]:
    reason_codes = set(str(code) for code in item.get("classification_reason_codes") or item.get("reason_codes") or [])
    ocr_band = str(item.get("ocr_quality_band") or item.get("ocr_status") or "unknown")
    selected_engine = str(item.get("selected_ocr_engine") or "unknown")
    entity_count = int(item.get("entity_count") or 0)
    empty = bool(item.get("empty_extraction_flag")) or "empty_or_near_empty_text" in reason_codes
    low_density = "low_text_density" in reason_codes
    page_count = item.get("page_count")
    root_cause = classify_root_cause(
        ocr_band=ocr_band,
        selected_engine=selected_engine,
        empty=empty,
        low_density=low_density,
        reason_codes=reason_codes,
    )
    return {
        "safe_file_id": item.get("safe_file_id") or item.get("file_id"),
        "filename_hash": item.get("filename_hash"),
        "content_hash": item.get("content_hash"),
        "file_type": item.get("file_type"),
        "status": item.get("status"),
        "document_type": item.get("document_type"),
        "page_count": page_count,
        "ocr_quality_band": ocr_band,
        "ocr_quality_score": item.get("ocr_quality_score"),
        "ocr_layout_route": item.get("ocr_layout_route"),
        "selected_ocr_engine": selected_engine,
        "confidence": item.get("confidence"),
        "entity_count": entity_count,
        "empty_extraction_flag": empty,
        "low_text_density": low_density,
        "possible_multi_document_pdf": bool(item.get("possible_multi_document_pdf")),
        "pdf_embedded_files_detected": bool(item.get("pdf_embedded_files_detected")),
        "root_cause_bucket": root_cause["bucket"],
        "root_cause_reason": root_cause["reason"],
        "recommended_operator_action": root_cause["operator_action"],
        "prototype_signal": root_cause["prototype_signal"],
        "classification_reason_codes": sorted(reason_codes),
        "review_reason_codes": list(item.get("review_reason_codes") or []),
        "external_api_used": bool(item.get("external_api_used")),
    }


def classify_root_cause(
    *,
    ocr_band: str,
    selected_engine: str,
    empty: bool,
    low_density: bool,
    reason_codes: set[str],
) -> dict[str, str | bool]:
    if ocr_band == "empty" or "empty_or_near_empty_text" in reason_codes:
        if selected_engine == "pymupdf_native_text":
            return {
                "bucket": "page_rendering_or_ocr_fallback_gap",
                "reason": "Native PDF text produced empty/near-empty extraction on scanned_pdf metadata; a render-to-image OCR diagnostic may be warranted.",
                "operator_action": "Manual review boundary; verify source scan and consider OCR preprocessing only in a future diagnostic prototype.",
                "prototype_signal": True,
            }
        return {
            "bucket": "scan_quality_or_blank_page_likely",
            "reason": "OCR/input quality is empty or near-empty with no extracted entities.",
            "operator_action": "Manual review boundary; request a clearer scan or typed copy if the source is clinically important.",
            "prototype_signal": False,
        }
    if ocr_band == "poor_ocr" and low_density:
        if selected_engine == "existing_pdf_pipeline":
            return {
                "bucket": "scan_quality_low_text_density",
                "reason": "Existing PDF pipeline produced poor OCR quality with low text density and sparse entities.",
                "operator_action": "Manual review boundary; source quality likely limits reliable automation.",
                "prototype_signal": False,
            }
        return {
            "bucket": "ocr_configuration_or_page_rendering_candidate",
            "reason": "Poor OCR quality from native text path suggests a local render/OCR preprocessing comparison may be useful.",
            "operator_action": "Manual review boundary; do not accept without source verification.",
            "prototype_signal": True,
        }
    return {
        "bucket": "manual_review_boundary",
        "reason": "Metadata is insufficient to safely distinguish OCR configuration from source quality.",
        "operator_action": "Keep in review_ocr_quality and inspect manually.",
        "prototype_signal": False,
    }


def bucket_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in diagnostics:
        counts[str(item.get("root_cause_bucket") or "unknown")] += 1
    return dict(sorted(counts.items()))


def preprocessing_recommendation(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = [item["safe_file_id"] for item in diagnostics if item.get("prototype_signal")]
    return {
        "justified": bool(candidate_ids),
        "candidate_count": len(candidate_ids),
        "candidate_safe_file_ids": candidate_ids,
        "scope": "diagnostic_only_local_render_to_image_ocr_comparison",
        "rationale": (
            "A narrow preprocessing prototype is justified only for files whose metadata suggests native text/page rendering may be the limiting factor. "
            "It must remain local-only, compare outputs diagnostically, and must not change production OCR or acceptance behavior."
            if candidate_ids
            else "The target files primarily indicate scan quality / low text density, so production OCR should not change yet."
        ),
    }


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged_in_public_reports"] or not report["phi_artifact_check"].get("passed", False):
        return "BLOCKED_PRIVACY_RISK"
    if report["target_count"] == 0:
        return "no_pdf_ocr_low_quality_targets"
    if report["missing_target_ids"]:
        return "ready_with_missing_targets"
    return "pdf_ocr_low_quality_diagnostic_complete"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase66 PDF OCR Low-Quality Follow-up Diagnostic",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Target cluster: `{report['target_cluster']}`",
        f"- Target count: `{report['target_count']}`",
        f"- Diagnosed count: `{report['diagnosed_count']}`",
        f"- Root-cause buckets: `{json.dumps(report['root_cause_buckets'], sort_keys=True)}`",
        f"- Narrow OCR preprocessing prototype justified: `{report['narrow_ocr_preprocessing_prototype_justified']}`",
        f"- Production extractor should change yet: `{report['production_extractor_should_change_yet']}`",
        f"- Production OCR should change yet: `{report['production_ocr_should_change_yet']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Per-File Diagnostics",
        "",
        "| Safe File ID | Filename Hash | Status | OCR Band | Engine | Page Count | Root Cause | Prototype Signal |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in report["per_file_diagnostics"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id')}`",
                    f"`{item.get('filename_hash')}`",
                    f"`{item.get('status')}`",
                    f"`{item.get('ocr_quality_band')}`",
                    f"`{item.get('selected_ocr_engine')}`",
                    f"`{item.get('page_count')}`",
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
            f"- Rationale: {report['prototype_recommendation']['rationale']}",
            "",
            "## Safety",
            "",
            "- This phase is diagnostic only.",
            "- No production OCR, extraction, routing, thresholds, or safety gates changed.",
            "- Public reports use safe IDs and hashes only.",
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
    print("MedAI Phase66 PDF OCR low-quality diagnostic complete.")
    print(f"target_count: {report['target_count']}")
    print(f"diagnosed_count: {report['diagnosed_count']}")
    print(f"root_cause_buckets: {report['root_cause_buckets']}")
    print(f"narrow_ocr_preprocessing_prototype_justified: {report['narrow_ocr_preprocessing_prototype_justified']}")
    print(f"production_extractor_should_change_yet: {report['production_extractor_should_change_yet']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    return 1 if report["conclusion"] == "BLOCKED_PRIVACY_RISK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
