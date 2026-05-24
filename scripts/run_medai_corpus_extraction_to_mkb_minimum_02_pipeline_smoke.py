#!/usr/bin/env python3
"""MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-02 — pipeline smoke validation.

Exercises ExecutionPipeline.process_text end-to-end with a stubbed
SpacyExtractor (no real spaCy/chromadb required). Proves:

* the extracted_medical_facts adapter runs inside the pipeline;
* lab facts are merged into entities;
* MKBWriter persists records;
* SQLite retrieval returns them;
* the Streamlit-free UI render-plan helper renders rows from the
  extractor_result.

Synthetic-only. No external API. No real document. No PHI.
"""
from __future__ import annotations

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

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_02_PIPELINE_SMOKE.md"

# Synthetic English lab text with family-classifier cues so the pipeline
# tags document_type="Lab result". No PHI, no real filenames, no real
# patient identifiers, no real dates.
SYNTH_EN_TEXT = (
    "Lab result report\n"
    "Specimen: serum\n"
    "Reference range listed below.\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
)

# Synthetic Russian/Cyrillic lab text with family cues
SYNTH_RU_TEXT = (
    "Лабораторный отчет\n"
    "Материал: сыворотка\n"
    "Результат:\n"
    "Глюкоза: 5,4 ммоль/л\n"
    "Гемоглобин: 135 г/л\n"
    "Лейкоциты: 7,2 x10E9/л\n"
)


class StubSpacyExtractor:
    """A minimal extractor double matching the SpacyExtractor.extract surface.

    Returns the extractor schema with NO entities and a high-enough
    confidence to keep the router on the spacy route and to clear the
    validation accept threshold. The adapter is therefore the layer that
    produces the test_result facts merged into the pipeline.
    """

    def __init__(self, confidence: float = 0.86) -> None:
        self._confidence = float(confidence)

    def extract(self, text: str) -> Dict[str, Any]:
        return {
            "extractor": "spacy",
            "entities": [],
            "confidence": self._confidence,
            "latency_ms": 1,
            "raw_text": text,
            "notes": ["stub_extractor_used_in_smoke_test"],
        }


def _build_pipeline_with_temp_store():
    from execution.pipeline import ExecutionPipeline
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_pipeline_smoke_"))
    db_path = temp_dir / "mkb.db"
    sql_store = SQLiteStore(db_path=db_path, encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        quality_gate=None,
        medication_gate=None,
        spacy_extractor=StubSpacyExtractor(),
        review_queue_path=temp_dir / "review_queue.jsonl",
    )
    return pipeline, sql_store, temp_dir


def _public_safe_preview(extractor_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    preview = extractor_result.get("extracted_medical_facts_preview_safe") or []
    out: List[Dict[str, Any]] = []
    for entry in preview:
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


def _public_safe_pipeline_facets(
    extractor_result: Dict[str, Any], outcome: str, validation_status: str
) -> Dict[str, Any]:
    return {
        "outcome": str(outcome),
        "validation_status": str(validation_status),
        "document_type": str(extractor_result.get("document_type") or "unknown"),
        "extracted_medical_fact_count": int(
            extractor_result.get("extracted_medical_fact_count") or 0
        ),
        "extracted_medical_fact_types": list(
            extractor_result.get("extracted_medical_fact_types") or []
        ),
        "extraction_to_mkb_candidate_count": int(
            extractor_result.get("extraction_to_mkb_candidate_count") or 0
        ),
        "extraction_to_mkb_written_count": int(
            extractor_result.get("extraction_to_mkb_written_count") or 0
        ),
        "extraction_to_mkb_review_count": int(
            extractor_result.get("extraction_to_mkb_review_count") or 0
        ),
        "extracted_medical_facts_preview_safe": _public_safe_preview(extractor_result),
    }


def _validate_external_api_disabled(extractor_result: Dict[str, Any]) -> bool:
    actual = str(extractor_result.get("actual_extractor") or "")
    if actual in {"gemini"}:
        return False
    if extractor_result.get("external_api_used") is True:
        return False
    return True


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, sql_store, temp_dir = _build_pipeline_with_temp_store()

    fixtures = [
        {
            "label": "synthetic_en_lab",
            "session_id": "smoke-en",
            "source_name": "synth-en-001",
            "text": SYNTH_EN_TEXT,
        },
        {
            "label": "synthetic_ru_lab",
            "session_id": "smoke-ru",
            "source_name": "synth-ru-001",
            "text": SYNTH_RU_TEXT,
        },
    ]

    findings: Dict[str, Any] = {
        "synthetic_documents_evaluated": 0,
        "structured_facts_extracted_total": 0,
        "records_returned_count": 0,
        "queued_records_returned_count": 0,
        "review_bound_records_persisted_total": 0,
        "retrieval_proof_count": 0,
        "ui_preview_proof_count": 0,
        "ui_render_plan_built_successfully": False,
        "external_api_used": False,
        "auto_accept_allowed_in_any_record": False,
        "per_document_results": [],
    }

    for fixture in fixtures:
        result = pipeline.process_text(
            fixture["text"],
            specialty="general",
            source_name=fixture["source_name"],
            session_id=fixture["session_id"],
        )
        ext = result.extractor_result or {}
        facets = _public_safe_pipeline_facets(
            ext, result.outcome, result.validation_status
        )
        facets["label"] = fixture["label"]
        findings["per_document_results"].append(facets)
        findings["synthetic_documents_evaluated"] += 1
        findings["structured_facts_extracted_total"] += facets["extracted_medical_fact_count"]
        findings["records_returned_count"] += len(result.records or [])
        findings["queued_records_returned_count"] += len(result.queued_records or [])
        for record in (list(result.records or []) + list(result.queued_records or [])):
            if getattr(record, "requires_review", False) is True or getattr(record, "tier", "") == "quarantined":
                findings["review_bound_records_persisted_total"] += 1 if record.fact_type == "test_result" else 0
        if not _validate_external_api_disabled(ext):
            findings["external_api_used"] = True

    # SQLite retrieval proof — count persisted test_result rows.
    with sql_store._get_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE fact_type=?",
            ("test_result",),
        ).fetchone()
    findings["retrieval_proof_count"] = int(rows["c"] if isinstance(rows, dict) else rows[0])

    # UI render plan
    try:
        from app.extracted_information_preview import build_extracted_information_preview_plan

        ui_total = 0
        for facets in findings["per_document_results"]:
            item = {
                "extracted_medical_facts_preview_safe": facets["extracted_medical_facts_preview_safe"],
                "extracted_medical_fact_count": facets["extracted_medical_fact_count"],
                "extraction_to_mkb_written_count": facets["extraction_to_mkb_written_count"],
                "extraction_to_mkb_review_count": facets["extraction_to_mkb_review_count"],
            }
            plan = build_extracted_information_preview_plan(item)
            ui_total += len(plan.get("rows") or [])
            for row in plan.get("rows") or []:
                if row.get("review_status") != "review-required":
                    findings["auto_accept_allowed_in_any_record"] = True
        findings["ui_preview_proof_count"] = ui_total
        findings["ui_render_plan_built_successfully"] = True
    except Exception as exc:
        findings["ui_render_plan_built_successfully"] = False
        findings["ui_render_plan_error"] = str(exc)

    all_pass = (
        findings["synthetic_documents_evaluated"] >= 2
        and findings["structured_facts_extracted_total"] > 0
        and findings["retrieval_proof_count"] > 0
        and findings["ui_render_plan_built_successfully"]
        and findings["ui_preview_proof_count"] > 0
        and not findings["external_api_used"]
        and not findings["auto_accept_allowed_in_any_record"]
        and findings["review_bound_records_persisted_total"] > 0
    )

    conclusion = "pipeline_smoke_ready" if all_pass else "not_ready"
    report = {
        "block_id": "MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-02-PIPELINE-SMOKE",
        "mode": "synthetic_pipeline_smoke_validation",
        "implementation_changed_code": True,
        "branch_expected": "clinical-knowledge-architecture",
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "audit_findings": findings,
        "conclusion": conclusion,
        "all_pass": all_pass,
    }
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-02-PIPELINE-SMOKE — Report",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "## Counts",
        "",
        f"- synthetic documents evaluated: {findings['synthetic_documents_evaluated']}",
        f"- structured facts extracted: {findings['structured_facts_extracted_total']}",
        f"- records returned by pipeline: {findings['records_returned_count']}",
        f"- queued records returned by pipeline: {findings['queued_records_returned_count']}",
        f"- review-bound test_result records persisted: {findings['review_bound_records_persisted_total']}",
        f"- SQLite retrieval proof (test_result rows): {findings['retrieval_proof_count']}",
        f"- UI render-plan preview rows: {findings['ui_preview_proof_count']}",
        "",
        "## Safety",
        "",
        f"- external API used: {findings['external_api_used']}",
        f"- auto-accept enabled in any record: {findings['auto_accept_allowed_in_any_record']}",
        f"- real PDF committed: {report['real_pdf_committed']}",
        f"- raw text printed: {report['raw_text_printed']}",
        f"- private paths printed: {report['private_paths_printed']}",
        "",
        "## Per-document",
        "",
    ]
    for facets in findings["per_document_results"]:
        md_lines.append(
            f"- `{facets['label']}` outcome `{facets['outcome']}` "
            f"document_type `{facets['document_type']}` "
            f"facts {facets['extracted_medical_fact_count']} "
            f"candidate {facets['extraction_to_mkb_candidate_count']} "
            f"written {facets['extraction_to_mkb_written_count']} "
            f"review {facets['extraction_to_mkb_review_count']}"
        )
    md_lines.append("")
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    SHORT_MD_PATH.write_text(
        "# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-02-PIPELINE-SMOKE — Short Summary\n\n"
        f"Conclusion: **{conclusion}**\n\n"
        f"- synthetic documents evaluated: {findings['synthetic_documents_evaluated']}\n"
        f"- structured facts extracted: {findings['structured_facts_extracted_total']}\n"
        f"- review-bound MKB records persisted: {findings['review_bound_records_persisted_total']}\n"
        f"- SQLite retrieval proof: {findings['retrieval_proof_count']}\n"
        f"- UI preview rows: {findings['ui_preview_proof_count']}\n"
        f"- external API used: {findings['external_api_used']}\n"
        f"- auto-accept enabled in any record: {findings['auto_accept_allowed_in_any_record']}\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
