from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import app.test_launcher as launcher


NUMERIC_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
        "101 102 103 104 105 106 107 108",
    ]
    * 35
)


CYRILLIC_TEXT = (
    "\u041b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u044b\u0439 "
    "\u0430\u043d\u0430\u043b\u0438\u0437 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 "
    "\u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c "
    "\u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 "
    "\u0435\u0434\u0438\u043d\u0438\u0446\u044b\n"
) * 60


class FakePipeline:
    def __init__(self, extractor_result: dict):
        self.extractor_result = extractor_result

    def process_pdf(self, source_path: Path, *, specialty: str, session_id: str):
        return SimpleNamespace(
            outcome="queued_for_review",
            validation_status="rejected",
            validation_errors=[{"code": "confidence_below_reject_threshold"}],
            extractor_result=dict(self.extractor_result),
            audit={},
        )


def _process_with_result(tmp_path, monkeypatch, extractor_result: dict):
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(launcher, "TEST_REVIEW_DIR", tmp_path / "review")
    monkeypatch.setattr(launcher, "TEST_ARCHIVE_DIR", tmp_path / "archive")
    return launcher._process_one_file(FakePipeline(extractor_result), source, specialty="general", run_id="run_1")


def test_rules_based_numeric_text_without_cyrillic_gets_runtime_marker(tmp_path, monkeypatch) -> None:
    result = _process_with_result(
        tmp_path,
        monkeypatch,
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        },
    )

    assert result.language_text_visibility == "incomplete"
    assert result.cyrillic_ocr_recommended is True
    assert result.ocr_gate_reason == "numeric_table_text_without_cyrillic"


def test_marker_metadata_is_copied_to_run_review_raw_record(tmp_path, monkeypatch) -> None:
    result = _process_with_result(
        tmp_path,
        monkeypatch,
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        },
    )
    raw_record = result.__dict__

    assert raw_record["language_text_visibility"] == "incomplete"
    assert raw_record["cyrillic_ocr_recommended"] is True
    assert raw_record["ocr_gate_reason"] == "numeric_table_text_without_cyrillic"


def test_missing_cyrillic_density_does_not_prevent_runtime_marker() -> None:
    marker = launcher.runtime_cyrillic_ocr_marker_for_result(
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        }
    )

    assert marker["language_text_visibility"] == "incomplete"
    assert marker["cyrillic_ocr_recommended"] is True


def test_cyrillic_visible_text_does_not_trigger_runtime_marker() -> None:
    marker = launcher.runtime_cyrillic_ocr_marker_for_result(
        {
            "raw_text": CYRILLIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        }
    )

    assert marker["language_text_visibility"] == "visible"
    assert marker["cyrillic_ocr_recommended"] is False
    assert marker["ocr_gate_reason"] == "cyrillic_visible"


def test_no_text_result_does_not_trigger_runtime_marker() -> None:
    marker = launcher.runtime_cyrillic_ocr_marker_for_result(
        {
            "raw_text": "",
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "empty",
        }
    )

    assert marker["cyrillic_ocr_recommended"] is False
    assert marker["ocr_gate_reason"] == "insufficient_native_text_for_shadow_gate"


def test_runtime_marker_does_not_change_confidence_or_acceptance(tmp_path, monkeypatch) -> None:
    result = _process_with_result(
        tmp_path,
        monkeypatch,
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        },
    )

    assert result.confidence == 0.45
    assert result.status == "review"
    assert result.validation_status == "rejected"


def test_runtime_marker_does_not_execute_ocr_fallback() -> None:
    marker = launcher.runtime_cyrillic_ocr_marker_for_result(
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        }
    )

    assert marker["ocr_gate_review_only"] is True
    assert marker["ocr_gate_auto_accept_allowed"] is False
    assert marker["ocr_gate_fallback_executed"] is False


def test_runtime_marker_metadata_contains_no_raw_text() -> None:
    marker = launcher.runtime_cyrillic_ocr_marker_for_result(
        {
            "raw_text": NUMERIC_TEXT,
            "selected_extractor": "rules_based",
            "confidence": 0.45,
            "text_quality_status": "readable_native",
        }
    )

    assert "raw_text" not in marker
    assert "text" not in marker
    assert NUMERIC_TEXT not in str(marker)
