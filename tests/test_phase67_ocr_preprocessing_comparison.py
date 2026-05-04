from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_phase67_ocr_preprocessing_comparison as phase67


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def fake_capability() -> SimpleNamespace:
    return SimpleNamespace(
        tesseract_available=True,
        tesseract_path="tesseract",
        russian_available=False,
        english_available=True,
    )


def test_phase67_reads_phase66_candidate_safe_ids(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase67, "compare_pdf_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["candidate_safe_file_ids"] == ["corpus_file_000537", "corpus_file_000601"]
    assert report["candidate_count"] == 2
    assert len(report["variant_metrics"]) == len(phase67.VARIANTS) * 2


def test_phase67_recommends_phase68_only_when_both_files_improve(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)

    def compare(safe_id, source_path, capability):
        return fake_metrics(safe_id, "meaningful")

    monkeypatch.setattr(phase67, "compare_pdf_variants", compare)

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["meaningful_improvement_file_count"] == 2
    assert report["phase68_controlled_ocr_fallback_sandbox_recommended"] is True
    assert report["recommended_next_action"] == "recommend_phase68_controlled_ocr_fallback_sandbox"
    assert report["conclusion"] == "phase68_sandbox_recommended"


def test_phase67_keeps_manual_review_boundary_when_not_all_improve(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)

    def compare(safe_id, source_path, capability):
        return fake_metrics(safe_id, "meaningful" if safe_id.endswith("537") else "minor")

    monkeypatch.setattr(phase67, "compare_pdf_variants", compare)

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["meaningful_improvement_file_count"] == 1
    assert report["phase68_controlled_ocr_fallback_sandbox_recommended"] is False
    assert report["recommended_next_action"] == "keep_manual_review_boundary"
    assert report["conclusion"] == "manual_review_boundary_retained"


def test_phase67_public_reports_use_safe_ids_only_and_no_private_values(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase67, "compare_pdf_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )
    public_json = (report_dir / phase67.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase67.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000537" in public_json
    assert "Patient Jane Doe" not in public_json
    assert "Secret Folder" not in public_md
    assert "fixture.pdf" not in public_json
    assert report["raw_phi_logged_in_public_reports"] is False


def test_phase67_blocks_if_private_value_leaks(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase67, "compare_pdf_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))
    monkeypatch.setattr(phase67, "render_markdown", lambda report: "Patient Jane Doe fixture.pdf")

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["raw_phi_logged_in_public_reports"] is True
    assert report["conclusion"] == "BLOCKED_PRIVACY_RISK"


def test_phase67_missing_private_source_routes_to_manual_boundary(tmp_path: Path):
    phase66 = tmp_path / "phase66.json"
    private = tmp_path / "private.json"
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports"
    input_dir.mkdir()
    write_json(phase66, {"prototype_recommendation": {"candidate_safe_file_ids": ["corpus_file_missing"]}})
    write_json(private, {"files": {}})

    report = phase67.run_comparison(
        phase66_report_path=phase66,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
        capability=fake_capability(),
    )

    assert report["candidate_count"] == 1
    assert report["per_file_summary"][0]["recommended_next_action"] == "keep_manual_review_boundary"
    assert report["variant_metrics"][0]["error_category"] == "private_source_unavailable"


def test_phase67_forces_local_only_and_does_not_change_production_flags(tmp_path: Path, monkeypatch):
    phase66, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase67, "compare_pdf_variants", lambda safe_id, source_path, capability: fake_metrics(safe_id, "minor"))

    report = phase67.run_comparison(
        phase66_report_path=phase66,
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
    assert phase67.app_config.MEDAI_LOCAL_ONLY is True
    assert phase67.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase67_bucket_helpers_are_stable():
    assert phase67.text_length_bucket(0) == "empty"
    assert phase67.text_length_bucket(100) == "short"
    assert phase67.ocr_quality_bucket(0.1) == "poor"
    assert phase67.ocr_quality_bucket(0.8) == "good"
    assert phase67.improvement_bucket({"text_length": 200, "quality_score": 0.5}, {"text_length": 20, "quality_score": 0.3}) == "meaningful"


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    phase66 = tmp_path / "phase66.json"
    private = tmp_path / "private.json"
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports"
    folder = input_dir / "Secret Folder"
    folder.mkdir(parents=True)
    (folder / "Patient Jane Doe fixture.pdf").write_bytes(b"%PDF-1.4 synthetic")
    write_json(
        phase66,
        {
            "prototype_recommendation": {
                "candidate_safe_file_ids": ["corpus_file_000537", "corpus_file_000601"]
            }
        },
    )
    write_json(
        private,
        {
            "files": {
                "corpus_file_000537": {
                    "original_filename": "Patient Jane Doe fixture.pdf",
                    "original_relative_path": "Secret Folder/Patient Jane Doe fixture.pdf",
                },
                "corpus_file_000601": {
                    "original_filename": "Patient Jane Doe fixture.pdf",
                    "original_relative_path": "Secret Folder/Patient Jane Doe fixture.pdf",
                },
            }
        },
    )
    return phase66, private, input_dir, report_dir


def fake_metrics(safe_id: str, improvement: str) -> list[phase67.VariantMetric]:
    metrics = []
    for variant in phase67.VARIANTS:
        bucket = improvement if variant == "contrast_sharpen" else "none"
        metrics.append(
            phase67.VariantMetric(
                safe_file_id=safe_id,
                variant_name=variant,
                text_length_bucket="moderate" if bucket in {"meaningful", "strong"} else "very_short",
                ocr_quality_bucket="usable_with_review" if bucket in {"meaningful", "strong"} else "poor",
                improvement_bucket=bucket,
                recommended_next_action=phase67.next_action_for_improvement(bucket),
                warnings=[],
                error_category=None,
            )
        )
    return metrics
