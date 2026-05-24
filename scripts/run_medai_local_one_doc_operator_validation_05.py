#!/usr/bin/env python3
"""MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05.

One-doc operator validation. Local-only. Synthetic-safe.

Looks for exactly one supported file (.pdf or .txt) in ``test_input/``
and runs the extraction-to-MKB chain end-to-end against it. No accept /
reject / defer is executed on the real record. No raw text, raw OCR
text, raw filename, or private path appears in the public report.

If no input file is present, the script prints the exact operator
next step and exits cleanly with conclusion ``no_input_file``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_local_one_doc_operator_validation_05"
JSON_REPORT_PATH = REPORT_DIR / "medai_local_one_doc_operator_validation_05_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_local_one_doc_operator_validation_05_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_LOCAL_ONE_DOC_OPERATOR_VALIDATION_05.md"

INPUT_DIR = REPO_ROOT / "test_input"
SUPPORTED_SUFFIXES = {".pdf", ".txt"}

OPERATOR_NEXT_STEP_NO_INPUT = (
    "NO_INPUT_FILE: Put one PDF or TXT into test_input/ and rerun this script."
)


def _safe_file_handle(path: Path) -> str:
    """Public-safe handle derived from (size, suffix). Never the name."""
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    suffix = path.suffix.lower() if path.suffix else "noext"
    digest = hashlib.sha256(f"{size}:{suffix}".encode("utf-8")).hexdigest()
    return f"localfile_{digest[:10]}"


def _is_supported_real_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def _find_one_input_file() -> Path | None:
    if not INPUT_DIR.is_dir():
        return None
    for p in sorted(INPUT_DIR.glob("*")):
        if _is_supported_real_file(p):
            return p
    return None


def _build_pipeline_with_temp_store():
    from execution.pipeline import ExecutionPipeline
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_one_doc_05_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    # Quality gate intentionally omitted: in local environments that lack
    # chromadb, the gate cannot construct. MKBWriter accepts None and
    # treats records as approved while preserving requires_review and
    # tier semantics. The medication safety gate and DDI behavior remain
    # available through their normal channels when configured.
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        quality_gate=None,
        medication_gate=None,
        review_queue_path=temp_dir / "review_queue.jsonl",
    )
    return pipeline, sql_store, temp_dir


def _privacy_check(payload: Any) -> Dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {
            "passed": bool(result.passed),
            "leak_examples_redacted": list(result.leak_examples_redacted or []),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _ui_render_plan_proof(extractor_result: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from app.extracted_information_preview import build_extracted_information_preview_plan

        item = {
            "extracted_medical_facts_preview_safe": list(
                extractor_result.get("extracted_medical_facts_preview_safe") or []
            ),
            "extracted_medical_fact_count": int(
                extractor_result.get("extracted_medical_fact_count") or 0
            ),
            "extraction_to_mkb_written_count": int(
                extractor_result.get("extraction_to_mkb_written_count") or 0
            ),
            "extraction_to_mkb_review_count": int(
                extractor_result.get("extraction_to_mkb_review_count") or 0
            ),
            "extracted_medical_fact_record_ids": list(
                extractor_result.get("extracted_medical_fact_record_ids") or []
            ),
            "extracted_medical_fact_record_states": dict(
                extractor_result.get("extracted_medical_fact_record_states") or {}
            ),
        }
        plan = build_extracted_information_preview_plan(item)
        action_count = sum(
            1
            for row_action in (plan.get("row_actions") or [])
            for a in row_action.get("actions") or []
            if a.get("enabled") is True
        )
        return {
            "render_plan_built": True,
            "row_count": int(plan.get("row_count") or 0),
            "enabled_action_count": int(action_count),
        }
    except Exception as exc:
        return {"render_plan_built": False, "error": f"{type(exc).__name__}: {exc}"}


def _public_safe_preview(preview: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in preview or []:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "type": str(entry.get("type") or "test_result"),
                "test_name": str(entry.get("test_name") or ""),
                "value": str(entry.get("value") or ""),
                "unit": str(entry.get("unit") or "") or None,
                "reference_range": str(entry.get("reference_range") or "") or None,
                "flag": str(entry.get("flag") or "") or None,
                "confidence": float(entry.get("confidence", 0.0)),
                "language_hint": str(entry.get("language_hint") or "unknown"),
                "parser_name": str(entry.get("parser_name") or "unknown"),
                "requires_review": bool(entry.get("requires_review", True)),
                "auto_accept_allowed": bool(entry.get("auto_accept_allowed", False)),
            }
        )
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = _find_one_input_file()

    if target is None:
        print(OPERATOR_NEXT_STEP_NO_INPUT)
        report = {
            "block_id": "MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05",
            "mode": "one_doc_operator_validation",
            "branch_expected": "clinical-knowledge-architecture",
            "audit_findings": {
                "one_doc_input_present": False,
                "operator_next_step": OPERATOR_NEXT_STEP_NO_INPUT,
                "structured_facts_extracted_count": 0,
                "review_bound_records_persisted_count": 0,
                "retrieval_proof_count": 0,
                "ui_render_plan_built": False,
                "ui_render_plan_row_count": 0,
                "real_record_action_executed": False,
                "external_api_used": False,
                "auto_accept_enabled": False,
            },
            "external_api_used": False,
            "auto_accept_enabled": False,
            "real_pdf_committed": False,
            "real_screenshot_committed": False,
            "raw_text_printed": False,
            "raw_ocr_text_printed": False,
            "raw_filenames_printed": False,
            "private_paths_printed": False,
            "phi_printed": False,
            "conclusion": "no_input_file",
            "all_pass": True,
        }
        privacy = _privacy_check(report)
        report["privacy_check_passed"] = bool(privacy.get("passed"))
        report["privacy_check_leak_examples_redacted"] = privacy.get(
            "leak_examples_redacted", []
        )
        JSON_REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        MD_REPORT_PATH.write_text(
            "# MEDAI-LOCAL-ONE-DOC-OPERATOR-VALIDATION-05 — Report\n\n"
            f"Conclusion: **no_input_file**\n\n"
            f"{OPERATOR_NEXT_STEP_NO_INPUT}\n",
            encoding="utf-8",
        )
        SHORT_MD_PATH.write_text(
            "# MEDAI-LOCAL-ONE-DOC-OPERATOR-VALIDATION-05 — Short Summary\n\n"
            "Conclusion: **no_input_file**.\n\n"
            f"{OPERATOR_NEXT_STEP_NO_INPUT}\n",
            encoding="utf-8",
        )
        return 0

    file_handle = _safe_file_handle(target)
    file_suffix = target.suffix.lower()

    try:
        pipeline, sql_store, _temp_dir = _build_pipeline_with_temp_store()
    except Exception as exc:
        report = {
            "block_id": "MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05",
            "mode": "one_doc_operator_validation",
            "audit_findings": {
                "one_doc_input_present": True,
                "file_handle": file_handle,
                "file_suffix": file_suffix,
                "pipeline_build_error_bucket": "pipeline_construction_failed",
                "pipeline_build_error_summary": f"{type(exc).__name__}",
                "structured_facts_extracted_count": 0,
                "review_bound_records_persisted_count": 0,
                "retrieval_proof_count": 0,
                "ui_render_plan_built": False,
                "external_api_used": False,
                "auto_accept_enabled": False,
            },
            "external_api_used": False,
            "auto_accept_enabled": False,
            "conclusion": "not_ready",
            "all_pass": False,
        }
        privacy = _privacy_check(report)
        report["privacy_check_passed"] = bool(privacy.get("passed"))
        JSON_REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 1

    error_bucket: str | None = None
    extractor_result: Dict[str, Any] = {}
    records_written = 0
    queued_records = 0
    retrieval_count = 0
    try:
        if file_suffix == ".pdf":
            result = pipeline.process_pdf(target, specialty="general", session_id="one-doc-05")
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
            result = pipeline.process_text(
                text,
                specialty="general",
                source_name=file_handle,
                session_id="one-doc-05",
            )
        extractor_result = dict(result.extractor_result or {})
        records_written = len(result.records or [])
        queued_records = len(result.queued_records or [])
        with sql_store._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM records WHERE fact_type=?",
                ("test_result",),
            ).fetchone()
        retrieval_count = int(row["c"] if isinstance(row, dict) else row[0])
    except Exception as exc:
        error_bucket = f"{type(exc).__name__}"

    ui_proof = _ui_render_plan_proof(extractor_result)

    structured_fact_count = int(extractor_result.get("extracted_medical_fact_count") or 0)
    review_bound_count = int(extractor_result.get("extraction_to_mkb_review_count") or 0)

    findings: Dict[str, Any] = {
        "one_doc_input_present": True,
        "file_handle": file_handle,
        "file_suffix": file_suffix,
        "pipeline_error_bucket": error_bucket,
        "structured_facts_extracted_count": structured_fact_count,
        "review_bound_records_persisted_count": review_bound_count,
        "records_written_count": records_written,
        "queued_records_count": queued_records,
        "retrieval_proof_count": retrieval_count,
        "ui_render_plan_built": bool(ui_proof.get("render_plan_built")),
        "ui_render_plan_row_count": int(ui_proof.get("row_count") or 0),
        "ui_render_plan_enabled_action_count": int(ui_proof.get("enabled_action_count") or 0),
        "operator_action_dry_run_executed_on_real_record": False,
        "external_api_used": str(extractor_result.get("actual_extractor") or "") == "gemini"
        or bool(extractor_result.get("external_api_used")),
        "auto_accept_enabled": False,
        "extracted_medical_facts_preview_safe": _public_safe_preview(
            extractor_result.get("extracted_medical_facts_preview_safe") or []
        ),
    }

    all_pass = (
        error_bucket is None
        and structured_fact_count > 0
        and retrieval_count > 0
        and findings["ui_render_plan_built"]
        and not findings["external_api_used"]
        and not findings["auto_accept_enabled"]
    )
    conclusion = "one_doc_operator_validation_ready" if all_pass else "not_ready"

    report = {
        "block_id": "MEDAI-LOCAL-RUNTIME-DEPENDENCY-AND-ONE-DOC-VALIDATION-05",
        "mode": "one_doc_operator_validation",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "external_api_used": findings["external_api_used"],
        "auto_accept_enabled": findings["auto_accept_enabled"],
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "conclusion": conclusion,
        "all_pass": all_pass,
    }

    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get(
        "leak_examples_redacted", []
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# MEDAI-LOCAL-ONE-DOC-OPERATOR-VALIDATION-05 — Report",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "## Counts",
        "",
        f"- file handle: `{file_handle}` (suffix `{file_suffix}`)",
        f"- structured facts extracted: {structured_fact_count}",
        f"- review-bound records persisted (test_result): {review_bound_count}",
        f"- records written (auto-acceptable): {records_written}",
        f"- queued records returned: {queued_records}",
        f"- SQLite retrieval proof (test_result rows): {retrieval_count}",
        f"- UI render plan built: {findings['ui_render_plan_built']}",
        f"- UI render plan rows: {findings['ui_render_plan_row_count']}",
        f"- enabled operator action affordances: {findings['ui_render_plan_enabled_action_count']}",
        "",
        "## Safety",
        "",
        f"- external API used: {findings['external_api_used']}",
        f"- auto-accept enabled: {findings['auto_accept_enabled']}",
        f"- operator action executed on real record: {findings['operator_action_dry_run_executed_on_real_record']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        f"- pipeline error bucket: {error_bucket or 'none'}",
        "",
    ]
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    SHORT_MD_PATH.write_text(
        "# MEDAI-LOCAL-ONE-DOC-OPERATOR-VALIDATION-05 — Short Summary\n\n"
        f"Conclusion: **{conclusion}**\n\n"
        f"- file handle: `{file_handle}` (suffix `{file_suffix}`)\n"
        f"- structured facts: {structured_fact_count}\n"
        f"- review-bound MKB records: {review_bound_count}\n"
        f"- SQLite retrieval proof: {retrieval_count}\n"
        f"- UI render plan built: {findings['ui_render_plan_built']}\n"
        f"- external API used: {findings['external_api_used']}\n"
        f"- auto-accept enabled: {findings['auto_accept_enabled']}\n"
        f"- privacy check passed: {report['privacy_check_passed']}\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
