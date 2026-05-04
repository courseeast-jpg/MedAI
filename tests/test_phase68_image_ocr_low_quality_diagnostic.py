from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts import run_phase68_image_ocr_low_quality_diagnostic as phase68


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase68_selects_image_ocr_low_quality_targets(tmp_path: Path):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    assert report["target_count"] == 3
    assert report["diagnosed_count"] == 3
    assert [item["safe_file_id"] for item in report["per_file_diagnostics"]] == [
        "corpus_file_000001",
        "corpus_file_000002",
        "corpus_file_000003",
    ]


def test_phase68_root_cause_buckets_are_computed(tmp_path: Path):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    assert report["root_cause_buckets"] == {
        "image_ocr_empty_output": 1,
        "local_preprocessing_candidate": 1,
        "source_resolution_likely_too_low": 1,
    }


def test_phase68_preprocessing_recommendation_is_diagnostic_only(tmp_path: Path):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    assert report["narrow_image_preprocessing_prototype_justified"] is True
    assert report["prototype_recommendation"]["scope"] == "diagnostic_only_local_image_preprocessing_comparison"
    assert report["production_ocr_should_change_yet"] is False
    assert report["production_ocr_routing_changed"] is False
    assert report["production_extraction_logic_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["safety_gates_changed"] is False


def test_phase68_public_reports_use_safe_ids_only(tmp_path: Path):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )
    public_json = (report_dir / phase68.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase68.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000001" in public_json
    assert "hash_1" not in public_json
    assert "hash_1" not in public_md
    assert "Patient Jane Doe image.jpg" not in public_json
    assert "Private Folder" not in public_md
    assert "SYNTHETIC OCR TEXT" not in public_json
    assert report["raw_phi_logged_in_public_reports"] is False


def test_phase68_blocks_if_private_values_leak(tmp_path: Path, monkeypatch):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)
    monkeypatch.setattr(phase68, "render_markdown", lambda report: "Patient Jane Doe image.jpg leaked")

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    assert report["raw_phi_logged_in_public_reports"] is True
    assert report["conclusion"] == "BLOCKED_PRIVACY_RISK"


def test_phase68_forces_local_only_and_external_disabled(tmp_path: Path):
    phase57, clusters, private, input_dir, report_dir = write_inputs(tmp_path)

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=input_dir,
        report_dir=report_dir,
    )

    assert report["local_only_forced"] is True
    assert report["external_api_used"] is False
    assert phase68.app_config.MEDAI_LOCAL_ONLY is True
    assert phase68.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase68_missing_private_source_keeps_manual_review_boundary(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(phase57, {"results": [make_record("corpus_file_000001", "poor_ocr", ".jpg")]})
    write_json(clusters, {"image_ocr_low_quality": ["corpus_file_000001"]})
    write_json(private, {"files": {}})

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=tmp_path / "input",
        report_dir=report_dir,
    )

    assert report["per_file_diagnostics"][0]["root_cause_bucket"] == "image_metadata_unavailable"
    assert report["per_file_diagnostics"][0]["prototype_signal"] is False
    assert report["manual_review_boundary_preserved"] is True


def test_phase68_no_targets_is_safe(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(phase57, {"results": []})
    write_json(clusters, {"image_ocr_low_quality": []})
    write_json(private, {"files": {}})

    report = phase68.run_diagnostic(
        phase57_report_path=phase57,
        phase57_clusters_path=clusters,
        private_mapping_path=private,
        input_dir=tmp_path / "input",
        report_dir=report_dir,
    )

    assert report["target_count"] == 0
    assert report["conclusion"] == "no_image_ocr_low_quality_targets"


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    input_dir = tmp_path / "full_corpus_input"
    private_folder = input_dir / "Private Folder"
    private_folder.mkdir(parents=True)
    create_image(private_folder / "Patient Jane Doe image.jpg", size=(100, 100), color=(245, 245, 245))
    create_image(private_folder / "Patient Jane Doe low contrast.jpg", size=(1400, 1400), color=(240, 240, 240))
    create_checkerboard(private_folder / "Patient Jane Doe empty.jpg", size=(1400, 1400))

    phase57 = tmp_path / "phase57.json"
    clusters = tmp_path / "clusters.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(
        clusters,
        {"image_ocr_low_quality": ["corpus_file_000001", "corpus_file_000002", "corpus_file_000003"]},
    )
    write_json(
        phase57,
        {
            "results": [
                make_record("corpus_file_000001", "poor_ocr", ".jpg", score=0.42, size=10_000),
                make_record("corpus_file_000002", "poor_ocr", ".jpg", score=0.41, size=200_000),
                make_record("corpus_file_000003", "empty", ".jpg", score=0.0, size=200_000, empty_code=True),
            ]
        },
    )
    write_json(
        private,
        {
            "files": {
                "corpus_file_000001": {
                    "original_filename": "Patient Jane Doe image.jpg",
                    "original_relative_path": "Private Folder/Patient Jane Doe image.jpg",
                },
                "corpus_file_000002": {
                    "original_filename": "Patient Jane Doe low contrast.jpg",
                    "original_relative_path": "Private Folder/Patient Jane Doe low contrast.jpg",
                },
                "corpus_file_000003": {
                    "original_filename": "Patient Jane Doe empty.jpg",
                    "original_relative_path": "Private Folder/Patient Jane Doe empty.jpg",
                },
            }
        },
    )
    return phase57, clusters, private, input_dir, report_dir


def create_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color=color).save(path)


def create_checkerboard(path: Path, *, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, color=(255, 255, 255))
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            if ((x // 40) + (y // 40)) % 2:
                pixels[x, y] = (0, 0, 0)
    image.save(path)


def make_record(safe_id: str, band: str, extension: str, *, score: float = 0.4, size: int = 200_000, empty_code: bool = False) -> dict:
    codes = [
        "table_structure_loss",
        "extraction_low_coverage",
        "extraction_low_confidence",
        "safety_gate_low_confidence",
        "extraction_sparse_entities",
        "classifier_legacy_ocr_flag",
    ]
    if band == "poor_ocr":
        codes.append("poor_input_ocr")
    if empty_code:
        codes.append("empty_or_near_empty_text")
    return {
        "safe_file_id": safe_id,
        "filename_hash": f"hash_{safe_id[-1]}",
        "content_hash": f"content_{safe_id[-1]}",
        "file_type": "image",
        "extension": extension,
        "image_extension": extension,
        "status": "review_ocr_quality",
        "document_type": "image_ocr",
        "ocr_quality_band": band,
        "ocr_quality_score": score,
        "selected_ocr_engine": "image_ocr_tesseract",
        "confidence": 0.45,
        "entity_count": 0,
        "empty_extraction_flag": True,
        "file_size_bytes": size,
        "classification_reason_codes": codes,
        "review_reason_codes": ["extraction_low_confidence"],
        "external_api_used": False,
    }
