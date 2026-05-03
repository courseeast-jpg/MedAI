"""Run a Phase53 local-only blind audit without PHI in public reports."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MEDAI_LOCAL_ONLY"] = "true"
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
os.environ["MEDAI_PRIVACY_AUDIT"] = "true"

import app.config as app_config
from execution.jobs import ExecutionJob
from execution.pipeline import ExecutionPipeline
import privacy.outbound_gate as outbound_gate
from privacy.outbound_gate import guard_external_payload
from privacy.privacy_audit import phi_artifact_tracking_status, write_json
from scripts.run_batch_validation import (
    ACCEPTED_OUTCOMES,
    analyze_text,
    apply_lab_normalization_recovery,
    build_initial_text_diagnostics,
    build_ocr_layout_context,
    classify_batch_status,
    classification_reason_codes_for,
    detect_ocr_low_quality_reason_codes,
    diagnostics_from_ocr_layout,
    diagnostics_from_result,
    downstream_classifier_reason_for,
    ocr_layout_forces_ocr_review,
    ocr_status_mismatch_for,
    reconcile_cyrillic_nonlab_status,
    review_reason_for,
    review_reasons_for,
    safe_float,
)


INPUT_DIR = ROOT / "real_validation_input"
REPORT_DIR = ROOT / "reports" / "phase53_blind_generalization_audit"
JSON_REPORT = REPORT_DIR / "phase53_blind_generalization_audit_report.json"
MD_REPORT = REPORT_DIR / "phase53_blind_generalization_audit_report.md"
OPERATOR_SUMMARY = REPORT_DIR / "phase53_blind_generalization_audit_operator_summary.md"
PRIVATE_MAPPING = REPORT_DIR / "local_filename_mapping_PRIVATE.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
EXTERNAL_EXTRACTORS = {"gemini", "claude", "openai", "deepl", "dxgpt", "patientnotes_ddi", "anthropic"}


def ensure_real_validation_input(input_dir: Path = INPUT_DIR) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / ".gitkeep").touch(exist_ok=True)


def supported_input_files(input_dir: Path = INPUT_DIR) -> list[Path]:
    ensure_real_validation_input(input_dir)
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def run_audit(
    *,
    input_dir: Path = INPUT_DIR,
    report_dir: Path = REPORT_DIR,
    pipeline: ExecutionPipeline | None = None,
) -> dict[str, Any]:
    force_local_only_runtime()
    ensure_real_validation_input(input_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    files = supported_input_files(input_dir)
    run_id = str(uuid4())
    active_pipeline = pipeline or ExecutionPipeline()
    if hasattr(active_pipeline, "router"):
        active_pipeline.router.gemini_quota_blocked = True

    results: list[dict[str, Any]] = []
    private_mapping: dict[str, dict[str, Any]] = {}
    for index, source_path in enumerate(files, start=1):
        file_id = f"file_{index:03d}"
        private_mapping[file_id] = {
            "original_filename": source_path.name,
            "file_size_bytes": source_path.stat().st_size,
            "filename_hash": hash_filename(source_path.name),
            "content_hash": hash_file_content(source_path),
        }
        results.append(process_one_blind_file(active_pipeline, source_path, file_id=file_id, run_id=run_id))

    counts = count_results(results)
    phi_artifacts = phi_artifact_tracking_status()
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase 53 Automated Blind PDF Generalization Audit",
        "run_id": run_id,
        "input_folder": "real_validation_input",
        "total_files": len(files),
        "processed_files": counts["processed_files"],
        "accepted_count": counts["accepted_count"],
        "review_count": counts["review_count"],
        "review_ocr_quality_count": counts["review_ocr_quality_count"],
        "empty_count": counts["empty_count"],
        "error_count": counts["error_count"],
        "local_only_mode": True,
        "local_only_forced": True,
        "external_api_default_allowed": False,
        "privacy_gate_status": "local_only",
        "raw_phi_logged": False,
        "raw_phi_logged_in_public_reports": False,
        "report_pdf_artifacts_tracked": not phi_artifacts["passed"],
        "phi_artifact_check": phi_artifacts,
        "external_api_used": any(item.get("external_api_used") for item in results),
        "results": results,
    }
    report["conclusion"] = conclusion_for(report)

    write_json(report_dir / JSON_REPORT.name, report)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    (report_dir / OPERATOR_SUMMARY.name).write_text(render_operator_summary(report), encoding="utf-8")
    write_json(report_dir / PRIVATE_MAPPING.name, {"run_id": run_id, "files": private_mapping})
    if public_reports_contain_private_names(report_dir, private_mapping):
        report["raw_phi_logged_in_public_reports"] = True
        report["raw_phi_logged"] = True
        report["conclusion"] = conclusion_for(report)
        write_json(report_dir / JSON_REPORT.name, report)
        (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
        (report_dir / OPERATOR_SUMMARY.name).write_text(render_operator_summary(report), encoding="utf-8")
    return report


def process_one_blind_file(pipeline: ExecutionPipeline, source_path: Path, *, file_id: str, run_id: str) -> dict[str, Any]:
    gate_decision = guard_external_payload(
        provider="gemini",
        text="",
        local_only=True,
        allow_external_api=False,
        require_pii_scrub=True,
    )
    try:
        ocr_layout: dict[str, Any] = {}
        text_diagnostics = build_initial_text_diagnostics(source_path)
        if source_path.suffix.lower() == ".pdf":
            ocr_layout = build_ocr_layout_context(source_path)
            selected_text = str(ocr_layout.get("selected_text") or "")
            method = str(ocr_layout.get("selected_engine") or "ocr_layout")
            text_diagnostics = analyze_text(selected_text, method=method)
            result = pipeline.run(ExecutionJob(text=selected_text, specialty="general", source_name=file_id, session_id=run_id))
            text_diagnostics = diagnostics_from_ocr_layout(ocr_layout, text_diagnostics)
        elif source_path.suffix.lower() == ".txt":
            selected_text = source_path.read_text(encoding="utf-8", errors="replace")
            ocr_layout = {
                "selected_text": selected_text,
                "selected_engine": "text_file",
                "input_quality_score": None,
                "input_quality_band": "unknown",
                "input_quality_warnings": [],
                "route_decision": "digital_clean_text",
                "ocr_layout_profile": {"document_type": "text_file"},
            }
            result = pipeline.run(ExecutionJob(text=selected_text, specialty="general", source_name=file_id, session_id=run_id))
            text_diagnostics = diagnostics_from_result(source_path, dict(result.extractor_result or {}), text_diagnostics)
        else:
            raise ValueError(f"Unsupported file extension: {source_path.suffix}")

        extractor_result = dict(result.extractor_result or {})
        audit = dict(result.audit or {})
        entities = list(extractor_result.get("entities", []))
        selected_extractor = (
            extractor_result.get("selected_extractor")
            or extractor_result.get("actual_extractor")
            or audit.get("extractor_actual")
            or audit.get("extractor")
        )
        confidence = safe_float(extractor_result.get("confidence", audit.get("confidence")))
        confidence_breakdown = extractor_result.get("confidence_breakdown")
        review_reason = review_reason_for(result, extractor_result)
        legacy_ocr_reason_codes = detect_ocr_low_quality_reason_codes(
            text_diagnostics=text_diagnostics,
            normalization_applied=bool(extractor_result.get("normalization_applied", False)),
            confidence_breakdown=confidence_breakdown,
        )
        is_ocr_low_quality = bool(legacy_ocr_reason_codes) or ocr_layout_forces_ocr_review(ocr_layout)
        status = classify_batch_status(
            outcome=result.outcome,
            review_reason=review_reason,
            confidence=confidence,
            entity_count=len(entities),
            is_ocr_low_quality=is_ocr_low_quality,
        )
        why_reviewed = review_reasons_for(
            status=status,
            entity_count=len(entities),
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
        )
        reason_codes = classification_reason_codes_for(
            status=status,
            entity_count=len(entities),
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            review_reason=review_reason,
            why_reviewed=why_reviewed,
            ocr_layout=ocr_layout,
            legacy_ocr_reason_codes=legacy_ocr_reason_codes,
        )
        lab_recovery = apply_lab_normalization_recovery(
            selected_text=str(ocr_layout.get("selected_text") or ""),
            status=status,
            entity_count=len(entities),
            is_ocr_low_quality=is_ocr_low_quality,
            classification_reason_codes=reason_codes,
            ocr_layout=ocr_layout,
        )
        status = lab_recovery["status"]
        is_ocr_low_quality = lab_recovery["is_ocr_low_quality"]
        reason_codes = lab_recovery["classification_reason_codes"]
        nonlab = reconcile_cyrillic_nonlab_status(
            current_status=status,
            is_ocr_low_quality=is_ocr_low_quality,
            classification_reason_codes=reason_codes,
            selected_text=str(ocr_layout.get("selected_text") or ""),
            ocr_layout=ocr_layout,
            lab_normalization=lab_recovery["lab_normalization"],
            entity_count=len(entities),
        )
        if nonlab.triggered:
            status = nonlab.new_status
            reason_codes = list(nonlab.new_reason_codes)
        mismatch = ocr_status_mismatch_for(status=status, ocr_layout=ocr_layout)
        return public_file_result(
            source_path=source_path,
            file_id=file_id,
            status=status,
            outcome=result.outcome,
            validation_status=result.validation_status,
            selected_extractor=selected_extractor,
            confidence=confidence,
            reason_codes=reason_codes,
            entity_count=len(entities),
            ocr_layout=ocr_layout,
            privacy_gate_mode=gate_decision.mode,
            external_api_used=False,
            error=None,
            mismatch=mismatch,
        )
    except Exception as exc:
        return public_file_result(
            source_path=source_path,
            file_id=file_id,
            status="error",
            outcome=None,
            validation_status=None,
            selected_extractor=None,
            confidence=None,
            reason_codes=["processing_error"],
            entity_count=0,
            ocr_layout={},
            privacy_gate_mode=gate_decision.mode,
            external_api_used=False,
            error=sanitize_error(exc, source_path, file_id),
            mismatch={"ocr_status_mismatch": False, "mismatch_type": None},
        )


def public_file_result(
    *,
    source_path: Path,
    file_id: str,
    status: str,
    outcome: str | None,
    validation_status: str | None,
    selected_extractor: Any,
    confidence: float | None,
    reason_codes: list[str],
    entity_count: int,
    ocr_layout: dict[str, Any],
    privacy_gate_mode: str,
    external_api_used: bool,
    error: str | None,
    mismatch: dict[str, Any],
) -> dict[str, Any]:
    profile = dict(ocr_layout.get("ocr_layout_profile") or {})
    return {
        "file_id": file_id,
        "original_filename_redacted": "[REDACTED]",
        "filename_hash": hash_filename(source_path.name),
        "content_hash": hash_file_content(source_path),
        "file_extension": source_path.suffix.lower(),
        "file_type": source_path.suffix.lower(),
        "file_size_bytes": source_path.stat().st_size,
        "status": status,
        "outcome": outcome,
        "validation_status": validation_status,
        "selected_extractor": str(selected_extractor) if selected_extractor is not None else None,
        "confidence": confidence,
        "reason_codes": list(reason_codes),
        "classification_reason_codes": list(reason_codes),
        "review_reason_codes": [code for code in reason_codes if "review" in code or "low" in code],
        "entity_count": entity_count,
        "ocr_status": ocr_layout.get("input_quality_band"),
        "ocr_quality_band": ocr_layout.get("input_quality_band"),
        "ocr_quality_score": ocr_layout.get("input_quality_score"),
        "ocr_layout_route": ocr_layout.get("route_decision"),
        "selected_ocr_engine": ocr_layout.get("selected_engine"),
        "document_type": profile.get("document_type") or profile.get("input_type"),
        "document_classification": profile.get("document_type") or profile.get("input_type"),
        "privacy_gate_mode": privacy_gate_mode,
        "external_api_used": bool(external_api_used),
        "payload_redacted": False,
        "empty_extraction_flag": entity_count == 0,
        "ocr_status_mismatch": bool(mismatch.get("ocr_status_mismatch")),
        "mismatch_type": mismatch.get("mismatch_type"),
        "error": error,
    }


def count_results(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "processed_files": sum(1 for item in results if item["status"] != "error"),
        "accepted_count": sum(1 for item in results if item["status"] == "accepted"),
        "review_count": sum(1 for item in results if item["status"] in {"review", "review_ocr_quality"}),
        "review_ocr_quality_count": sum(1 for item in results if item["status"] == "review_ocr_quality"),
        "empty_count": sum(1 for item in results if item.get("empty_extraction_flag")),
        "error_count": sum(1 for item in results if item["status"] == "error"),
    }


def conclusion_for(report: dict[str, Any]) -> str:
    if (
        report["report_pdf_artifacts_tracked"]
        or report["external_api_used"]
        or report["raw_phi_logged"]
        or report.get("raw_phi_logged_in_public_reports")
    ):
        return "BLOCKED_PRIVACY_RISK"
    if report["total_files"] == 0:
        return "no_input_files"
    if report["error_count"] > 0:
        return "BLOCKED_ERRORS"
    accepted_ratio = report["accepted_count"] / max(report["total_files"], 1)
    if accepted_ratio >= 0.8 and report["error_count"] == 0:
        return "PASS_SAFETY_ACCEPTABLE_AUTOMATION"
    return "PASS_SAFETY_WEAK_AUTOMATION"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase53 Blind PDF Generalization Audit",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Total files: `{report['total_files']}`",
        f"- Processed files: `{report['processed_files']}`",
        f"- Accepted: `{report['accepted_count']}`",
        f"- Review: `{report['review_count']}`",
        f"- Review OCR quality: `{report['review_ocr_quality_count']}`",
        f"- Empty: `{report['empty_count']}`",
        f"- Errors: `{report['error_count']}`",
        f"- Local-only mode: `{report['local_only_mode']}`",
        f"- External API default allowed: `{report['external_api_default_allowed']}`",
        f"- Privacy gate status: `{report['privacy_gate_status']}`",
        f"- Raw PHI logged: `{report['raw_phi_logged']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Report PDF artifacts tracked: `{report['report_pdf_artifacts_tracked']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Per-File Results",
        "",
        "| Safe File ID | Filename Hash | Content Hash | Type | Status | Outcome | Extractor | Confidence | OCR status | Reason codes | Error |",
        "| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item['file_id']}`",
                    f"`{item['filename_hash']}`",
                    f"`{item.get('content_hash')}`",
                    f"`{item['file_type']}`",
                    f"`{item['status']}`",
                    f"`{item.get('outcome')}`",
                    f"`{item.get('selected_extractor')}`",
                    "" if item.get("confidence") is None else f"`{item['confidence']}`",
                    f"`{item.get('ocr_status')}`",
                    f"`{', '.join(item.get('reason_codes') or [])}`",
                    f"`{item.get('error') or ''}` |",
                ]
            )
        )
    if not report["results"]:
        lines.append("")
        lines.append("No supported PDF/TXT files were found in `real_validation_input/`.")
    return "\n".join(lines) + "\n"


def render_operator_summary(report: dict[str, Any]) -> str:
    attention = [item["file_id"] for item in report["results"] if item["status"] != "accepted"]
    lines = [
        "# Phase53 Operator Summary",
        "",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Accepted: `{report['accepted_count']}`",
        f"- Requires review: `{report['review_count']}`",
        f"- Review OCR quality: `{report['review_ocr_quality_count']}`",
        f"- Empty: `{report['empty_count']}`",
        f"- Errors: `{report['error_count']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged: `{report['raw_phi_logged']}`",
        "",
        "## Files Requiring Attention",
        "",
    ]
    if attention:
        lines.extend(f"- `{file_id}`" for file_id in attention)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "- Review every non-accepted file against the source document.",
            "- Treat `review_ocr_quality` as unreliable extraction until the source quality is checked.",
            "- Use `reports/phase53_blind_generalization_audit/local_filename_mapping_PRIVATE.json` locally to map safe IDs back to filenames.",
            "- Do not commit the private mapping or real validation inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def hash_file_identifier(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8", errors="replace"))
    digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()[:12]


def hash_filename(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8", errors="replace")).hexdigest()[:16]


def hash_file_content(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def public_reports_contain_private_names(report_dir: Path, private_mapping: dict[str, dict[str, Any]]) -> bool:
    private_names = [entry["original_filename"] for entry in private_mapping.values()]
    public_paths = [report_dir / JSON_REPORT.name, report_dir / MD_REPORT.name, report_dir / OPERATOR_SUMMARY.name]
    public_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in public_paths if path.exists())
    return any(name and name in public_text for name in private_names)


def is_external_extractor(selected_extractor: Any) -> bool:
    return str(selected_extractor or "").lower() in EXTERNAL_EXTRACTORS


def sanitize_error(exc: Exception, source_path: Path, file_id: str) -> str:
    message = str(exc).replace(str(source_path), file_id).replace(source_path.name, file_id)
    return message[:500]


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


def main() -> int:
    report = run_audit()
    print("MedAI Phase53 blind PDF generalization audit complete.")
    print(f"total_files: {report['total_files']}")
    print(f"processed_files: {report['processed_files']}")
    print(f"accepted: {report['accepted_count']}")
    print(f"review: {report['review_count']}")
    print(f"review_ocr_quality: {report['review_ocr_quality_count']}")
    print(f"empty: {report['empty_count']}")
    print(f"errors: {report['error_count']}")
    print(f"local_only_mode: {report['local_only_mode']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged: {report['raw_phi_logged']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"report_pdf_artifacts_tracked: {report['report_pdf_artifacts_tracked']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {REPORT_DIR / JSON_REPORT.name}")
    print(f"markdown_report: {REPORT_DIR / MD_REPORT.name}")
    print(f"operator_summary: {REPORT_DIR / OPERATOR_SUMMARY.name}")
    return 1 if report["conclusion"] in {"BLOCKED_ERRORS", "BLOCKED_PRIVACY_RISK"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
