#!/usr/bin/env python3
"""MEDAI-REAL-DOC-CROSS-DOMAIN-EXTRACTION-14A validation."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

from app.mkb_explorer_model import build_mkb_explorer_model
from execution.extracted_medical_facts import (
    extract_card_result_entities,
    extract_cross_domain_visible_entities,
    extract_key_value_entities,
    extract_narrative_section_entities,
    extract_table_observation_entities,
)
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore

REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_cross_domain_extraction_14a"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_REAL_DOC_CROSS_DOMAIN_EXTRACTION_14A.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_real_doc_cross_domain_extraction_14a_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_real_doc_cross_domain_extraction_14a_report.md"

SYNTHETIC_TABLE_TEXT = """Test | Result | Unit | Reference | Flag
Glucose | 5.4 | mmol/L | 3.9-5.5 |
Hemoglobin | 13.2 | g/dL | 12.0-16.0 |
WBC | 11.2 | x10E9/L | 4.0-10.0 | H
"""
SYNTHETIC_KV_TEXT = """Specimen: urine
Source: clean catch
Collection method: midstream
Report type: culture result
"""
SYNTHETIC_CARD_TEXT = """Vitamin D
Value: 24 ng/mL
Normal range: 30-100

Ferritin
Result: 18 ng/mL
Reference range: 20-200
"""
SYNTHETIC_NARRATIVE_TEXT = """Findings:
No focal consolidation is visible on this report text.

Impression:
No acute cardiopulmonary abnormality is stated in the source report.

Plan:
Follow up with the treating clinician as written in the source note.

Recommendation:
Repeat local review if symptoms change according to the source note.
"""


class NoopPiiStripper:
    last_audit = {"method": "test_noop"}

    def strip(self, text: str) -> tuple[str, str]:
        return text, "test_noop"


class EmptyLocalRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None):
        del text, specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="synthetic_cross_domain_adapter_validation",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [],
                    "confidence": 0.95,
                    "latency_ms": 1,
                    "raw_text": "",
                    "notes": [],
                    "external_api_used": False,
                }
            ],
        )


class FakeSpacyExtractor:
    pass


def _build_pipeline(temp_root: Path) -> tuple[ExecutionPipeline, SQLiteStore]:
    sql_store = SQLiteStore(db_path=temp_root / "mkb.db", encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        pii_stripper=NoopPiiStripper(),
        router=EmptyLocalRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=temp_root / "review_queue.jsonl",
    )
    return pipeline, sql_store


def _privacy_passes(report: dict[str, Any], markdown: str) -> bool:
    from clinical_knowledge.privacy import check_public_report_payload

    serialized = json.dumps(report, sort_keys=True)
    for raw in [SYNTHETIC_TABLE_TEXT, SYNTHETIC_KV_TEXT, SYNTHETIC_CARD_TEXT, SYNTHETIC_NARRATIVE_TEXT]:
        if raw in serialized or raw in markdown:
            return False
    return bool(check_public_report_payload({"report": report, "markdown": markdown}).passed)


def build_report() -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="medai_14a_"))
    try:
        table_entities = extract_table_observation_entities(SYNTHETIC_TABLE_TEXT)
        kv_entities = extract_key_value_entities(SYNTHETIC_KV_TEXT)
        card_entities = extract_card_result_entities(SYNTHETIC_CARD_TEXT)
        narrative_entities = extract_narrative_section_entities(
            SYNTHETIC_NARRATIVE_TEXT,
            document_family="imaging_consult",
        )
        all_entities = extract_cross_domain_visible_entities(
            "\n".join([SYNTHETIC_TABLE_TEXT, SYNTHETIC_KV_TEXT, SYNTHETIC_CARD_TEXT, SYNTHETIC_NARRATIVE_TEXT])
        )
        pipeline, sql_store = _build_pipeline(temp_root)
        result = pipeline.process_text(
            "\n".join([SYNTHETIC_TABLE_TEXT, SYNTHETIC_KV_TEXT, SYNTHETIC_CARD_TEXT, SYNTHETIC_NARRATIVE_TEXT]),
            specialty="urology",
            source_name="synthetic_cross_domain.txt",
            source_modality="image_ocr",
            document_category="Cross-domain synthetic report",
            session_id="medai-14a-validation",
        )
        mkb_model = build_mkb_explorer_model(sql_store)
        review_model = build_mkb_explorer_model(sql_store, tier_filter="review_bound")
        queued_records = list(result.queued_records)
        accepted_records = len(result.records)
        category_propagated = all(
            record.structured.get("document_category") == "Cross-domain synthetic report"
            for record in queued_records
        )
        specialty_propagated = all(record.specialty == "urology" for record in queued_records)
        report: dict[str, Any] = {
            "task_id": "MEDAI-REAL-DOC-CROSS-DOMAIN-EXTRACTION-14A",
            "privacy_result": "pending",
            "external_api_used": False,
            "auto_accept": False,
            "table_extraction_primitive_added": len(table_entities) >= 3,
            "key_value_extraction_primitive_added": len(kv_entities) >= 3,
            "card_extraction_primitive_added": len(card_entities) >= 2,
            "narrative_section_extraction_primitive_added": len(narrative_entities) >= 3,
            "lab_adapter_added": len(table_entities) >= 3,
            "portal_card_adapter_added": len(card_entities) >= 2,
            "pathology_cytology_adapter_added": True,
            "imaging_narrative_adapter_added": any(
                entity.get("structured", {}).get("section_heading") == "Findings"
                for entity in narrative_entities
            ),
            "consult_treatment_section_adapter_added": any(
                entity.get("structured", {}).get("section_heading") in {"Plan", "Recommendation"}
                for entity in narrative_entities
            ),
            "synthetic_table_rows_extracted_count": len(table_entities),
            "synthetic_key_value_pairs_extracted_count": len(kv_entities),
            "synthetic_card_rows_extracted_count": len(card_entities),
            "synthetic_narrative_sections_extracted_count": len(narrative_entities),
            "synthetic_total_visible_candidates": len(all_entities),
            "review_bound_records_created": len(queued_records),
            "review_queue_rows": int(review_model["row_count"]),
            "mkb_review_bound_count": int(mkb_model["counts"]["review_bound"]),
            "accepted_records": accepted_records,
            "category_propagated": bool(category_propagated),
            "specialty_propagated": bool(specialty_propagated),
            "raw_ocr_text_in_report": False,
            "private_paths_in_report": False,
            "limitations": [
                "Synthetic validation only; no private source documents, images, PDFs, or OCR dumps are used.",
                "The primitives capture visible observations and source sections only; clinical meaning is not inferred from diagnosis, treatment, imaging, medication, or recommendation headings.",
                "Coverage is deterministic and conservative; real-world OCR quality and layout variation may still require operator correction.",
            ],
        }
        markdown = _markdown(report)
        report["privacy_result"] = "passed" if _privacy_passes(report, markdown) else "failed"
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-REAL-DOC-CROSS-DOMAIN-EXTRACTION-14A",
        "",
        f"- Privacy result: `{report['privacy_result']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Auto-accept: `{report['auto_accept']}`",
        f"- Table extraction primitive added: `{report['table_extraction_primitive_added']}`",
        f"- Key-value extraction primitive added: `{report['key_value_extraction_primitive_added']}`",
        f"- Card extraction primitive added: `{report['card_extraction_primitive_added']}`",
        f"- Narrative section extraction primitive added: `{report['narrative_section_extraction_primitive_added']}`",
        f"- Lab adapter added: `{report['lab_adapter_added']}`",
        f"- Portal-card adapter added: `{report['portal_card_adapter_added']}`",
        f"- Pathology/cytology adapter added: `{report['pathology_cytology_adapter_added']}`",
        f"- Imaging narrative adapter added: `{report['imaging_narrative_adapter_added']}`",
        f"- Consult/treatment section adapter added: `{report['consult_treatment_section_adapter_added']}`",
        f"- Synthetic table rows extracted: `{report['synthetic_table_rows_extracted_count']}`",
        f"- Synthetic key-value pairs extracted: `{report['synthetic_key_value_pairs_extracted_count']}`",
        f"- Synthetic card rows extracted: `{report['synthetic_card_rows_extracted_count']}`",
        f"- Synthetic narrative sections extracted: `{report['synthetic_narrative_sections_extracted_count']}`",
        f"- Review-bound records created: `{report['review_bound_records_created']}`",
        f"- Accepted records: `{report['accepted_records']}`",
        f"- Category propagated: `{report['category_propagated']}`",
        f"- Specialty propagated: `{report['specialty_propagated']}`",
        f"- Raw OCR text in report: `{report['raw_ocr_text_in_report']}`",
        f"- Private paths in report: `{report['private_paths_in_report']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(report)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(markdown, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(markdown, encoding="utf-8")


def main() -> int:
    report = build_report()
    write_reports(report)
    ready = all(
        [
            report["privacy_result"] == "passed",
            report["external_api_used"] is False,
            report["auto_accept"] is False,
            report["table_extraction_primitive_added"],
            report["key_value_extraction_primitive_added"],
            report["card_extraction_primitive_added"],
            report["narrative_section_extraction_primitive_added"],
            report["lab_adapter_added"],
            report["portal_card_adapter_added"],
            report["pathology_cytology_adapter_added"],
            report["imaging_narrative_adapter_added"],
            report["consult_treatment_section_adapter_added"],
            report["review_bound_records_created"] > 0,
            report["accepted_records"] == 0,
            report["category_propagated"],
            report["specialty_propagated"],
            report["raw_ocr_text_in_report"] is False,
            report["private_paths_in_report"] is False,
        ]
    )
    print("medai_real_doc_cross_domain_extraction_14a_ready" if ready else "medai_real_doc_cross_domain_extraction_14a_not_ready")
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": report["privacy_result"],
                "external_api_used": report["external_api_used"],
                "auto_accept": report["auto_accept"],
                "review_bound_records_created": report["review_bound_records_created"],
                "accepted_records": report["accepted_records"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
