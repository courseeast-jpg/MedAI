from __future__ import annotations

from pathlib import Path

import pytest

import app.test_launcher as launcher
from app.mkb_explorer_model import build_mkb_explorer_model
from execution.pipeline import ExecutionPipeline
from execution.router import RoutedExtraction
from mkb.sqlite_store import SQLiteStore


RAW_OCR_TEXT = "SECRET OCR TEXT SHOULD NOT APPEAR"


class FakePiiStripper:
    last_audit = {"method": "test_noop", "placeholder_count": 0}

    def strip(self, text: str) -> tuple[str, str]:
        return text, "test_noop"


class FakeRouter:
    def execute(self, text: str, *, specialty: str = "general", source_name: str | None = None) -> RoutedExtraction:
        del specialty, source_name
        return RoutedExtraction(
            extractor_route="spacy",
            extractor_actual="spacy",
            requested_route="spacy",
            intended_route="spacy",
            selected_extractor="spacy",
            decision_reason="test_local_route",
            route_score=0.99,
            results=[
                {
                    "extractor": "spacy",
                    "actual_extractor": "spacy",
                    "entities": [
                        {
                            "type": "test_result",
                            "text": "safe public test result",
                            "confidence": 0.95,
                            "structured": {
                                "test_name": "safe synthetic analyte",
                                "value": "present",
                                "unit": "",
                            },
                        }
                    ],
                    "confidence": 0.95,
                    "latency_ms": 1,
                    "raw_text": text,
                    "notes": [],
                    "external_api_used": False,
                }
            ],
        )


class FakeSpacyExtractor:
    pass


def configure_launcher_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "TEST_INPUT_DIR", tmp_path / "test_input")
    monkeypatch.setattr(launcher, "TEST_REVIEW_DIR", tmp_path / "test_review")
    monkeypatch.setattr(launcher, "TEST_ARCHIVE_DIR", tmp_path / "test_archive")
    monkeypatch.setattr(launcher, "TEST_OUTPUT_DIR", tmp_path / "test_output")
    monkeypatch.setattr(launcher, "TEST_RUN_REPORT_DIR", tmp_path / "reports" / "test_runs")
    monkeypatch.setattr(launcher, "LATEST_JSON_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.json")
    monkeypatch.setattr(launcher, "LATEST_MD_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.md")


def build_pipeline(sql_store: SQLiteStore) -> ExecutionPipeline:
    return ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        pii_stripper=FakePiiStripper(),
        router=FakeRouter(),
        spacy_extractor=FakeSpacyExtractor(),
        review_queue_path=sql_store.db_path.parent / "review_queue.jsonl",
    )


def run_image_ocr_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    specialty: str = "urology",
    document_category: str = "Urinalysis",
):
    configure_launcher_dirs(monkeypatch, tmp_path)
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()
    (input_dir / "source.png").write_bytes(b"synthetic image")
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text=RAW_OCR_TEXT,
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        ),
    )
    sql_store = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    summary = launcher.run_medai_test_batch(
        build_pipeline(sql_store),
        specialty=specialty,
        document_category=document_category,
    )
    model = build_mkb_explorer_model(sql_store)
    review_model = build_mkb_explorer_model(sql_store, tier_filter="review_bound")
    rows = review_model["rows"]
    assert rows
    record = sql_store.get_record(rows[0]["record_id_full"])
    return summary, sql_store, model, review_model, record


def test_image_ocr_recovered_text_creates_review_bound_records_not_active(monkeypatch, tmp_path):
    summary, _store, model, review_model, record = run_image_ocr_batch(monkeypatch, tmp_path)

    assert summary.accepted_count == 0
    assert summary.review_count == 1
    assert model["counts"]["active"] == 0
    assert model["counts"]["review_bound"] == 1
    assert review_model["row_count"] == 1
    assert record.tier == "quarantined"
    assert record.status == "pending_validation_review"
    assert record.requires_review is True
    assert record.source_type == "image_ocr"
    assert record.extraction_method.startswith("local_image_ocr/")


def test_run_review_needs_review_count_matches_review_queue_count(monkeypatch, tmp_path):
    summary, _store, _model, review_model, _record = run_image_ocr_batch(monkeypatch, tmp_path)

    assert summary.review_count == review_model["counts"]["review_bound"] == review_model["row_count"]


def test_selected_specialty_and_document_category_propagate(monkeypatch, tmp_path):
    _summary, _store, _model, review_model, record = run_image_ocr_batch(
        monkeypatch,
        tmp_path,
        specialty="urology",
        document_category="Urinalysis",
    )

    assert record.specialty == "urology"
    assert record.structured["document_category"] == "Urinalysis"
    assert record.structured["source_modality"] == "image_ocr"
    assert record.structured["operator_review_status"] == "pending"
    assert record.structured["auto_accept_allowed"] is False
    assert review_model["rows"][0]["specialty"] == "urology"


@pytest.mark.parametrize(
    ("specialty", "document_category"),
    [
        ("general", "General"),
        ("hematology", "Lab result"),
        ("urology", "Urinalysis"),
    ],
)
def test_image_ocr_review_bound_is_cross_domain(monkeypatch, tmp_path, specialty, document_category):
    _summary, _store, model, _review_model, record = run_image_ocr_batch(
        monkeypatch,
        tmp_path,
        specialty=specialty,
        document_category=document_category,
    )

    assert model["counts"]["active"] == 0
    assert model["counts"]["review_bound"] == 1
    assert record.specialty == specialty
    assert record.structured["document_category"] == document_category


def test_no_raw_ocr_text_appears_in_public_reports(monkeypatch, tmp_path):
    run_image_ocr_batch(monkeypatch, tmp_path)

    assert RAW_OCR_TEXT not in launcher.LATEST_JSON_REPORT.read_text(encoding="utf-8")
    assert RAW_OCR_TEXT not in launcher.LATEST_MD_REPORT.read_text(encoding="utf-8")


def test_external_api_false_and_auto_accept_false(monkeypatch, tmp_path):
    summary, _store, _model, _review_model, record = run_image_ocr_batch(monkeypatch, tmp_path)
    result = summary.results[0]

    assert result["external_api_used"] is False
    assert result["image_ocr_auto_accept_allowed"] is False
    assert record.structured["auto_accept_allowed"] is False
