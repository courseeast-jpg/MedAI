from __future__ import annotations

import json
from pathlib import Path

from scripts import run_phase66_pdf_ocr_low_quality_diagnostic as phase66


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase66_selects_pdf_ocr_low_quality_targets(tmp_path: Path):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["target_count"] == 3
    assert report["diagnosed_count"] == 3
    assert [item["safe_file_id"] for item in report["per_file_diagnostics"]] == [
        "corpus_file_000001",
        "corpus_file_000002",
        "corpus_file_000003",
    ]


def test_phase66_root_cause_buckets_are_computed(tmp_path: Path):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["root_cause_buckets"] == {
        "ocr_configuration_or_page_rendering_candidate": 1,
        "page_rendering_or_ocr_fallback_gap": 1,
        "scan_quality_low_text_density": 1,
    }


def test_phase66_preprocessing_prototype_recommendation_is_diagnostic_only(tmp_path: Path):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["narrow_ocr_preprocessing_prototype_justified"] is True
    assert report["prototype_recommendation"]["scope"] == "diagnostic_only_local_render_to_image_ocr_comparison"
    assert report["production_extractor_should_change_yet"] is False
    assert report["production_ocr_should_change_yet"] is False


def test_phase66_public_reports_use_safe_ids_only(tmp_path: Path):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )
    public_json = (report_dir / phase66.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase66.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000001" in public_json
    assert "hash_1" in public_md
    assert "Patient Jane Doe" not in public_json
    assert "Private Folder" not in public_md
    assert report["raw_phi_logged_in_public_reports"] is False


def test_phase66_blocks_if_private_values_leak(tmp_path: Path, monkeypatch):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase66, "render_markdown", lambda report: "Patient Jane Doe leaked")

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["raw_phi_logged_in_public_reports"] is True
    assert report["conclusion"] == "BLOCKED_PRIVACY_RISK"


def test_phase66_forces_local_only_and_external_disabled(tmp_path: Path):
    phase57, clusters, private, report_dir = write_inputs(tmp_path)

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["local_only_forced"] is True
    assert report["external_api_used"] is False
    assert phase66.app_config.MEDAI_LOCAL_ONLY is True
    assert phase66.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase66_handles_missing_targets_without_crashing(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(phase57, {"results": []})
    write_json(clusters, {"pdf_ocr_low_quality": ["corpus_file_missing"]})
    write_json(private, {"files": {}})

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["target_count"] == 1
    assert report["diagnosed_count"] == 0
    assert report["missing_target_ids"] == ["corpus_file_missing"]
    assert report["conclusion"] == "ready_with_missing_targets"


def test_phase66_no_targets_is_safe(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(phase57, {"results": []})
    write_json(clusters, {"pdf_ocr_low_quality": []})
    write_json(private, {"files": {}})

    report = phase66.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["target_count"] == 0
    assert report["conclusion"] == "no_pdf_ocr_low_quality_targets"


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(
        clusters,
        {"pdf_ocr_low_quality": ["corpus_file_000001", "corpus_file_000002", "corpus_file_000003"]},
    )
    write_json(
        phase57,
        {
            "results": [
                make_record("corpus_file_000001", "poor_ocr", "existing_pdf_pipeline"),
                make_record("corpus_file_000002", "poor_ocr", "pymupdf_native_text"),
                make_record("corpus_file_000003", "empty", "pymupdf_native_text", empty_code=True),
            ]
        },
    )
    write_json(
        private,
        {
            "files": {
                "corpus_file_000001": {
                    "original_filename": "Patient Jane Doe leaked",
                    "original_relative_path": "Private Folder/Patient Jane Doe leaked",
                }
            }
        },
    )
    return phase57, clusters, private, report_dir


def make_record(safe_id: str, ocr_band: str, engine: str, *, empty_code: bool = False) -> dict:
    codes = [
        "extraction_low_confidence",
        "extraction_low_coverage",
        "extraction_sparse_entities",
        "low_text_density",
        "safety_gate_low_confidence",
        "table_structure_loss",
    ]
    if ocr_band == "poor_ocr":
        codes.append("poor_input_ocr")
    if empty_code:
        codes.append("empty_or_near_empty_text")
    return {
        "safe_file_id": safe_id,
        "filename_hash": f"hash_{safe_id[-1]}",
        "content_hash": f"content_{safe_id[-1]}",
        "file_type": "pdf",
        "status": "review_ocr_quality",
        "document_type": "scanned_pdf",
        "page_count": 1,
        "ocr_quality_band": ocr_band,
        "ocr_quality_score": 0.4,
        "ocr_layout_route": ocr_band,
        "selected_ocr_engine": engine,
        "confidence": 0.45,
        "entity_count": 0,
        "empty_extraction_flag": True,
        "classification_reason_codes": codes,
        "review_reason_codes": ["extraction_low_confidence"],
        "external_api_used": False,
    }
