from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts import run_phase69_image_ocr_preprocessing_comparison as phase69


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fake_capability() -> SimpleNamespace:
    return SimpleNamespace(
        tesseract_available=True,
        tesseract_path="tesseract",
        russian_available=False,
        english_available=True,
    )


def test_phase69_reads_phase68_image_empty_output_candidates(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase69, "compare_image_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["candidate_safe_file_ids"] == [
        "corpus_file_000145",
        "corpus_file_000204",
        "corpus_file_000205",
        "corpus_file_000206",
        "corpus_file_000573",
    ]
    assert report["candidate_count"] == 5
    assert len(report["variant_metrics"]) == len(phase69.VARIANTS) * 5


def test_phase69_recommends_phase70_when_most_files_improve(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)

    def compare(safe_id, source_path, capability):
        return fake_metrics(safe_id, "meaningful" if safe_id in {"corpus_file_000145", "corpus_file_000204", "corpus_file_000205"} else "minor")

    monkeypatch.setattr(phase69, "compare_image_variants", compare)

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["meaningful_improvement_file_count"] == 3
    assert report["phase70_controlled_image_ocr_fallback_sandbox_recommended"] is True
    assert report["recommended_next_action"] == "recommend_phase70_controlled_image_ocr_fallback_sandbox"
    assert report["conclusion"] == "phase70_sandbox_recommended"


def test_phase69_keeps_manual_review_boundary_when_not_most_files_improve(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)

    def compare(safe_id, source_path, capability):
        return fake_metrics(safe_id, "meaningful" if safe_id == "corpus_file_000145" else "minor")

    monkeypatch.setattr(phase69, "compare_image_variants", compare)

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["meaningful_improvement_file_count"] == 1
    assert report["phase70_controlled_image_ocr_fallback_sandbox_recommended"] is False
    assert report["recommended_next_action"] == "keep_manual_review_boundary"
    assert report["manual_review_boundary_preserved"] is True
    assert report["conclusion"] == "manual_review_boundary_retained"


def test_phase69_public_reports_use_safe_ids_only_and_no_private_values(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase69, "compare_image_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )
    public_json = (report_dir / phase69.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase69.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000145" in public_json
    assert "Patient Jane Doe" not in public_json
    assert "Secret Folder" not in public_md
    assert "fixture.jpg" not in public_json
    assert ".jpg" not in public_json
    assert "SYNTHETIC OCR TEXT" not in public_json
    assert report["raw_phi_logged_in_public_reports"] is False


def test_phase69_blocks_if_private_value_leaks(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase69, "compare_image_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))
    monkeypatch.setattr(phase69, "render_markdown", lambda report: "Patient Jane Doe 000145 fixture.jpg")

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["raw_phi_logged_in_public_reports"] is True
    assert report["conclusion"] == "BLOCKED_PRIVACY_RISK"


def test_phase69_missing_private_source_routes_to_manual_boundary(tmp_path: Path):
    phase68 = tmp_path / "phase68.json"
    private = tmp_path / "private.json"
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports"
    input_dir.mkdir()
    write_json(phase68, {"per_file_diagnostics": [{"safe_file_id": "corpus_file_missing", "root_cause_bucket": "image_ocr_empty_output"}]})
    write_json(private, {"files": {}})

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["candidate_count"] == 1
    assert report["per_file_summary"][0]["recommended_next_action"] == "keep_manual_review_boundary"
    assert report["variant_metrics"][0]["error_category"] == "private_source_unavailable"


def test_phase69_forces_local_only_and_does_not_change_production_flags(tmp_path: Path, monkeypatch):
    phase68, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase69, "compare_image_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase69.run_comparison(
        phase68_report_path=phase68,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["local_only_forced"] is True
    assert report["external_api_used"] is False
    assert report["production_ocr_routing_changed"] is False
    assert report["production_extraction_logic_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["safety_gates_changed"] is False
    assert phase69.app_config.MEDAI_LOCAL_ONLY is True
    assert phase69.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase69_bucket_helpers_are_stable():
    assert phase69.text_length_bucket(0) == "empty"
    assert phase69.text_length_bucket(100) == "short"
    assert phase69.ocr_quality_bucket(0.1) == "poor"
    assert phase69.ocr_quality_bucket(0.8) == "good"
    assert phase69.improvement_bucket({"text_length": 220, "quality_score": 0.5}, {"text_length": 20, "quality_score": 0.3}) == "meaningful"


def test_phase69_real_variant_scoring_handles_synthetic_image_without_cloud(tmp_path: Path):
    image_path = tmp_path / "synthetic.png"
    Image.new("RGB", (120, 60), color=(255, 255, 255)).save(image_path)

    report = phase69.compare_image_variants("corpus_file_synthetic", image_path, SimpleNamespace(tesseract_available=False))

    assert len(report) == len(phase69.VARIANTS)
    assert all(item.safe_file_id == "corpus_file_synthetic" for item in report)
    assert all(item.recommended_next_action == "keep_manual_review_boundary" for item in report)


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    phase68 = tmp_path / "phase68.json"
    private = tmp_path / "private.json"
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports"
    folder = input_dir / "Secret Folder"
    folder.mkdir(parents=True)
    for safe_id in ["000145", "000204", "000205", "000206", "000573"]:
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(folder / f"Patient Jane Doe {safe_id} fixture.jpg")
    ids = [
        "corpus_file_000145",
        "corpus_file_000204",
        "corpus_file_000205",
        "corpus_file_000206",
        "corpus_file_000573",
    ]
    write_json(
        phase68,
        {
            "per_file_diagnostics": [
                {"safe_file_id": safe_id, "root_cause_bucket": "image_ocr_empty_output"}
                for safe_id in ids
            ]
        },
    )
    write_json(
        private,
        {
            "files": {
                safe_id: {
                    "original_filename": f"Patient Jane Doe {safe_id[-6:]} fixture.jpg",
                    "original_relative_path": f"Secret Folder/Patient Jane Doe {safe_id[-6:]} fixture.jpg",
                }
                for safe_id in ids
            }
        },
    )
    return phase68, private, input_dir, report_dir


def fake_metrics(safe_id: str, improvement: str) -> list[phase69.VariantMetric]:
    metrics = []
    for variant in phase69.VARIANTS:
        bucket = improvement if variant == "contrast_enhancement" else "none"
        metrics.append(
            phase69.VariantMetric(
                safe_file_id=safe_id,
                variant_name=variant,
                text_length_bucket="moderate" if bucket in {"meaningful", "strong"} else "very_short",
                ocr_quality_bucket="usable_with_review" if bucket in {"meaningful", "strong"} else "poor",
                improvement_bucket=bucket,
                recommended_next_action=phase69.next_action_for_improvement(bucket),
                warnings=[],
                error_category=None,
            )
        )
    return metrics
