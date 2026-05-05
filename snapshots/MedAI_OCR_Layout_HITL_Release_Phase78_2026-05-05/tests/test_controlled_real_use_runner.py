from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import run_controlled_real_use as runner


def _write_pdf(path: Path, body: bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n") -> None:
    path.write_bytes(body)


def test_controlled_real_use_runner_stages_valid_files_and_limits_processing(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "data" / "pdfs"
    input_dir.mkdir(parents=True)
    artifacts_root = tmp_path / "artifacts" / "controlled_real_use"
    reports_root = tmp_path / "reports" / "controlled_real_use"

    valid_names = [
        "a_valid.pdf",
        "b_valid.pdf",
        "c_valid.pdf",
        "d_valid.txt",
        "e_valid.txt",
        "f_valid.txt",
    ]
    for name in valid_names[:3]:
        _write_pdf(input_dir / name)
    for name in valid_names[3:]:
        (input_dir / name).write_text(f"Clinical note for {name}", encoding="utf-8")

    (input_dir / "ignore.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (input_dir / "too_small.pdf").write_bytes(b"%PDF-")
    (input_dir / "corrupt.pdf").write_bytes(b"not-a-real-pdf-but-large-enough-to-fail-header-check")

    captured: dict[str, object] = {}

    def fake_execute_controlled_validation(
        staged_files,
        *,
        dataset_dir,
        runtime_dir,
        requested_limit,
        specialty,
        quota_safe,
    ):
        captured["dataset_dir"] = dataset_dir
        captured["runtime_dir"] = runtime_dir
        captured["requested_limit"] = requested_limit
        captured["staged_names"] = [item.staged_path.name for item in staged_files]
        documents = [
            {
                "document": item.staged_path.name,
                "status": "processed",
                "error": None,
                "outcome": "written",
                "validation_status": "accepted",
                "confidence": 0.7,
            }
            for item in staged_files[:requested_limit]
        ]
        return {
            "generated_at": "2026-04-26T16:00:00+00:00",
            "dataset_dir": str(dataset_dir),
            "documents_selected": min(requested_limit, len(staged_files)),
            "documents_processed": min(requested_limit, len(staged_files)),
            "written": min(3, min(requested_limit, len(staged_files))),
            "queued_for_review": 0,
            "external_quota_blocked": 0,
            "hard_failures": 0,
            "review_queue": {
                "path": str(runtime_dir / "review_queue.jsonl"),
                "items": 0,
            },
            "documents": documents,
        }

    monkeypatch.setattr(runner, "ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr(runner, "REPORTS_ROOT", reports_root)
    monkeypatch.setattr(runner, "execute_controlled_validation", fake_execute_controlled_validation)

    summary, exit_code = runner.run_controlled_real_use(
        input_dir=input_dir,
        now=datetime(2026, 4, 26, 16, 0, 0, tzinfo=UTC),
    )

    assert exit_code == 0
    assert summary["valid_files_found"] == 6
    assert captured["requested_limit"] == 5
    assert "final_batch_50" not in str(captured["dataset_dir"])
    assert summary["dataset_dir"] == str(captured["dataset_dir"])

    staged_input_dir = artifacts_root / "20260426T160000Z" / "inputs"
    assert staged_input_dir.exists()
    assert sorted(path.name for path in staged_input_dir.iterdir()) == sorted(valid_names)
    assert sorted(captured["staged_names"]) == sorted(valid_names)

    skipped_reasons = {item["name"]: item["reason"] for item in summary["skipped_files"]}
    assert skipped_reasons["too_small.pdf"].startswith("pdf_too_small")
    assert skipped_reasons["corrupt.pdf"] == "pdf_header_missing_or_corrupt"
    assert summary["ignored_files"][0]["name"] == "ignore.png"
    assert summary["ignored_files"][0]["reason"] == "ignored_unsupported_extension"
    assert len(summary["deferred_files"]) == 1
    assert summary["deferred_files"][0]["reason"] == "controlled_mode_max_documents_limit_5"

    summary_path = artifacts_root / "20260426T160000Z" / "controlled_run_summary.json"
    report_path = reports_root / "20260426T160000Z" / "controlled_run_report.md"
    assert summary_path.exists()
    assert report_path.exists()
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["attempted"] == 5
    assert written_summary["synthetic_dataset_used"] is False
    report_text = report_path.read_text(encoding="utf-8")
    assert "Controlled Real-Use Run Report" in report_text
    assert "Synthetic dataset used: `False`" in report_text

    for name in valid_names + ["ignore.png", "too_small.pdf", "corrupt.pdf"]:
        assert (input_dir / name).exists()


def test_controlled_real_use_runner_fails_if_input_folder_missing(tmp_path: Path):
    missing_dir = tmp_path / "data" / "pdfs"

    with pytest.raises(SystemExit, match="Controlled real-use input folder not found"):
        runner.run_controlled_real_use(input_dir=missing_dir)


def test_controlled_real_use_runner_fails_if_no_supported_files(tmp_path: Path):
    input_dir = tmp_path / "data" / "pdfs"
    input_dir.mkdir(parents=True)
    (input_dir / "photo.jpg").write_bytes(b"jpg")

    with pytest.raises(SystemExit, match=r"No supported input files found .* Supported extensions: \.pdf, \.txt"):
        runner.run_controlled_real_use(input_dir=input_dir)
