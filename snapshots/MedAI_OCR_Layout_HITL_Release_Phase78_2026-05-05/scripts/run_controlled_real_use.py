from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.production_mode import MODE_CONTROLLED
from execution.review_queue import ReviewQueueWriter, read_review_queue
from execution.runtime_controls import RuntimeRunGuard, deterministic_run_id
from scripts.run_phase12_real_world_validation import (
    build_phase12_summary,
    build_validation_pipeline,
    summarize_document,
)


INPUT_DIR = ROOT / "data" / "pdfs"
ARTIFACTS_ROOT = ROOT / "artifacts" / "controlled_real_use"
REPORTS_ROOT = ROOT / "reports" / "controlled_real_use"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
IGNORED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_DOCUMENTS_PER_RUN = 5
MIN_PDF_BYTES = 32
LOCK_PATH = ROOT / "artifacts" / "controlled_real_use" / "controlled_real_use.lock"


@dataclass(frozen=True)
class FileDecision:
    name: str
    path: str
    category: str
    reason: str


@dataclass(frozen=True)
class StagedFile:
    original_path: Path
    staged_path: Path
    file_type: str


def timestamp_token(now: datetime | None = None) -> str:
    value = now or datetime.now(UTC)
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_read_bytes(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def classify_input_files(input_dir: Path) -> tuple[list[Path], list[FileDecision], list[FileDecision]]:
    valid_files: list[Path] = []
    skipped_files: list[FileDecision] = []
    ignored_files: list[FileDecision] = []

    for path in sorted(item for item in input_dir.iterdir() if item.is_file()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            reason = "ignored_unsupported_extension"
            category = "ignored"
            if suffix not in IGNORED_EXTENSIONS:
                reason = "ignored_non_supported_extension"
            ignored_files.append(FileDecision(name=path.name, path=str(path), category=category, reason=reason))
            continue

        if suffix == ".pdf":
            size = path.stat().st_size
            if size < MIN_PDF_BYTES:
                skipped_files.append(
                    FileDecision(
                        name=path.name,
                        path=str(path),
                        category="skipped",
                        reason=f"pdf_too_small_lt_{MIN_PDF_BYTES}_bytes",
                    )
                )
                continue
            header = _safe_read_bytes(path, 5)
            if header != b"%PDF-":
                skipped_files.append(
                    FileDecision(
                        name=path.name,
                        path=str(path),
                        category="skipped",
                        reason="pdf_header_missing_or_corrupt",
                    )
                )
                continue

        valid_files.append(path)

    return valid_files, skipped_files, ignored_files


def copy_valid_files(valid_files: list[Path], staged_input_dir: Path) -> list[StagedFile]:
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    staged_files: list[StagedFile] = []
    for original_path in valid_files:
        staged_path = staged_input_dir / original_path.name
        shutil.copy2(original_path, staged_path)
        staged_files.append(
            StagedFile(
                original_path=original_path,
                staged_path=staged_path,
                file_type=original_path.suffix.lower().lstrip("."),
            )
        )
    return staged_files


def execute_controlled_validation(
    staged_files: list[StagedFile],
    *,
    dataset_dir: Path,
    runtime_dir: Path,
    requested_limit: int = MAX_DOCUMENTS_PER_RUN,
    specialty: str = "general",
    quota_safe: bool = True,
) -> dict[str, Any]:
    run_id = deterministic_run_id(
        scope="controlled_real_use",
        values={
            "dataset_dir": str(dataset_dir.resolve()),
            "runtime_dir": str(runtime_dir.resolve()),
            "requested_limit": requested_limit,
            "specialty": specialty,
            "quota_safe": bool(quota_safe),
            "files": [item.staged_path.name for item in staged_files],
        },
    )
    guard = RuntimeRunGuard(
        script_name="run_controlled_real_use.py",
        run_id=run_id,
        lock_path=LOCK_PATH,
        cleanup_paths=[runtime_dir],
    )
    summary: dict[str, Any] | None = None
    guard.acquire()
    try:
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        pipeline, sql_store, component_state = build_validation_pipeline(runtime_dir)
        review_queue = ReviewQueueWriter(Path(component_state["review_queue_path"]))
        documents: list[dict[str, Any]] = []
        for staged in staged_files[:requested_limit]:
            started = time.perf_counter()
            try:
                if staged.file_type == "txt":
                    document_text = staged.staged_path.read_text(encoding="utf-8")
                    result = pipeline.process_text(
                        document_text,
                        specialty=specialty,
                        source_name=staged.staged_path.name,
                        session_id=f"controlled-real-use-{staged.staged_path.stem}",
                    )
                else:
                    result = pipeline.process_pdf(
                        staged.staged_path,
                        specialty=specialty,
                        session_id=f"controlled-real-use-{staged.staged_path.stem}",
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                documents.append(summarize_document(staged.staged_path, result, processing_time_ms=elapsed_ms))
            except Exception as exc:  # pragma: no cover - defensive live path
                elapsed_ms = (time.perf_counter() - started) * 1000
                document = summarize_document(
                    staged.staged_path,
                    None,
                    error=exc,
                    quota_safe=quota_safe,
                    processing_time_ms=elapsed_ms,
                )
                if document["status"] == "external_quota_blocked":
                    review_queue.append_external_quota_block(
                        run_id=f"controlled-real-use-{staged.staged_path.stem}",
                        document_id=staged.staged_path.name,
                        source_filename=staged.staged_path.name,
                        reason="external_quota_blocked",
                        recommended_action="operator_retry_after_quota_reset",
                        raw_evidence_path=str(staged.staged_path),
                        error=str(exc),
                        retry_visibility=document["retry_visibility"],
                    )
                documents.append(document)

        summary = build_phase12_summary(
            dataset_dir=dataset_dir,
            requested_limit=requested_limit,
            documents=documents,
            runtime_counts=sql_store.count_records(),
            component_state=component_state,
        )
        guard.finalize(documents=documents)
    finally:
        guard.release()

    summary["runtime_controls"] = guard.state.to_dict()
    return summary


def determine_next_human_action(*, summary: dict[str, Any], review_queue_path: str | None) -> str:
    if summary["hard_failures"] > 0:
        return "Inspect the controlled run report and per-document errors before retrying."
    if summary["external_quota_blocked"] > 0:
        return "Retry the controlled run after external quota resets."
    if summary["queued_for_review"] > 0 or summary["review_queue"]["items"] > 0:
        queue_target = review_queue_path or summary["review_queue"]["path"]
        return f"Open the review queue at {queue_target} and resolve review items before broader use."
    if summary["documents_processed"] == 0:
        return "Add valid .pdf or .txt files to data/pdfs and rerun."
    return "Review the markdown report, verify clinical correctness, and proceed with the next controlled batch if appropriate."


def build_controlled_run_summary(
    *,
    run_token: str,
    input_dir: Path,
    archive_dir: Path,
    staged_input_dir: Path,
    report_path: Path,
    valid_files: list[Path],
    staged_files: list[StagedFile],
    skipped_files: list[FileDecision],
    ignored_files: list[FileDecision],
    deferred_files: list[FileDecision],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    review_queue_path = validation_summary.get("review_queue", {}).get("path")
    next_action = determine_next_human_action(summary=validation_summary, review_queue_path=review_queue_path)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_token": run_token,
        "production_mode": MODE_CONTROLLED,
        "max_documents_per_run": MAX_DOCUMENTS_PER_RUN,
        "review_queue_acknowledged": False,
        "live_mode_allowed": False,
        "input_dir": str(input_dir),
        "dataset_dir": str(staged_input_dir),
        "archive_dir": str(archive_dir),
        "staged_input_dir": str(staged_input_dir),
        "report_path": str(report_path),
        "valid_files_found": len(valid_files),
        "valid_files": [
            {
                "name": path.name,
                "original_path": str(path),
                "staged_path": str(next(item.staged_path for item in staged_files if item.original_path == path)),
            }
            for path in valid_files
        ],
        "skipped_files": [asdict(item) for item in skipped_files],
        "ignored_files": [asdict(item) for item in ignored_files],
        "deferred_files": [asdict(item) for item in deferred_files],
        "attempted": int(validation_summary["documents_selected"]),
        "processed": int(validation_summary["documents_processed"]),
        "written": int(validation_summary["written"]),
        "queued_for_review": int(validation_summary["queued_for_review"]),
        "external_quota_blocked": int(validation_summary["external_quota_blocked"]),
        "hard_failures": int(validation_summary["hard_failures"]),
        "review_queue": {
            "path": review_queue_path,
            "items": int(validation_summary.get("review_queue", {}).get("items", 0)),
        },
        "validation_summary": validation_summary,
        "next_human_action": next_action,
        "synthetic_dataset_used": False,
    }


def build_controlled_run_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Controlled Real-Use Run Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Production mode: `{summary['production_mode']}`",
        f"- Max documents per run: `{summary['max_documents_per_run']}`",
        f"- Review queue acknowledged: `{summary['review_queue_acknowledged']}`",
        f"- Input dir: `{summary['input_dir']}`",
        f"- Staged input dir: `{summary['staged_input_dir']}`",
        f"- Synthetic dataset used: `{summary['synthetic_dataset_used']}`",
        f"- Valid files found: `{summary['valid_files_found']}`",
        f"- Attempted: `{summary['attempted']}`",
        f"- Processed: `{summary['processed']}`",
        f"- Written: `{summary['written']}`",
        f"- Queued for review: `{summary['queued_for_review']}`",
        f"- External quota blocked: `{summary['external_quota_blocked']}`",
        f"- Hard failures: `{summary['hard_failures']}`",
        f"- Review queue path: `{summary['review_queue']['path']}`",
        f"- Review queue items: `{summary['review_queue']['items']}`",
        f"- Next human action: `{summary['next_human_action']}`",
        "",
        "## Valid Files",
        "",
    ]
    if summary["valid_files"]:
        for item in summary["valid_files"]:
            lines.append(
                f"- `{item['name']}` -> original=`{item['original_path']}` staged=`{item['staged_path']}`"
            )
    else:
        lines.append("- No valid files were staged.")

    lines.extend([
        "",
        "## Skipped Files",
        "",
    ])
    all_skipped = summary["skipped_files"] + summary["deferred_files"]
    if all_skipped:
        for item in all_skipped:
            lines.append(f"- `{item['name']}` -> reason=`{item['reason']}` path=`{item['path']}`")
    else:
        lines.append("- No files were skipped.")

    lines.extend([
        "",
        "## Ignored Files",
        "",
    ])
    if summary["ignored_files"]:
        for item in summary["ignored_files"]:
            lines.append(f"- `{item['name']}` -> reason=`{item['reason']}` path=`{item['path']}`")
    else:
        lines.append("- No unsupported files were ignored.")

    lines.extend([
        "",
        "## Validation Documents",
        "",
    ])
    for item in summary["validation_summary"]["documents"]:
        lines.append(
            f"- `{item['document']}` -> status={item['status']} outcome={item['outcome']} validation={item['validation_status']} confidence={item['confidence']}"
        )
    return "\n".join(lines) + "\n"


def write_controlled_run_outputs(summary: dict[str, Any], *, archive_dir: Path, report_path: Path) -> tuple[Path, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = archive_dir / "controlled_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_controlled_run_report(summary), encoding="utf-8")
    return summary_path, report_path


def render_console_summary(summary: dict[str, Any]) -> str:
    skipped = summary["skipped_files"] + summary["deferred_files"]
    skipped_lines = ["none"]
    if skipped:
        skipped_lines = [f"{item['name']}: {item['reason']}" for item in skipped]
    report_path = summary["report_path"]
    review_queue_path = summary["review_queue"]["path"] or "none"
    lines = [
        "Controlled MedAI real-use run complete.",
        f"Valid files found: {summary['valid_files_found']}",
        "Skipped files with reasons:",
    ]
    lines.extend(f"- {line}" for line in skipped_lines)
    lines.extend(
        [
            f"Attempted: {summary['attempted']}",
            f"Processed: {summary['processed']}",
            f"Written: {summary['written']}",
            f"Queued for review: {summary['queued_for_review']}",
            f"External quota blocked: {summary['external_quota_blocked']}",
            f"Hard failures: {summary['hard_failures']}",
            f"Report path: {report_path}",
            f"Review queue path: {review_queue_path}",
            f"Next human action: {summary['next_human_action']}",
        ]
    )
    return "\n".join(lines)


def run_controlled_real_use(*, input_dir: Path = INPUT_DIR, now: datetime | None = None) -> tuple[dict[str, Any], int]:
    if not input_dir.exists():
        raise SystemExit(f"Controlled real-use input folder not found: {input_dir}")

    valid_files, skipped_files, ignored_files = classify_input_files(input_dir)
    supported_count = len(valid_files) + len(skipped_files)
    if supported_count == 0:
        raise SystemExit(
            f"No supported input files found in {input_dir}. Supported extensions: .pdf, .txt"
        )
    if not valid_files:
        skipped_text = ", ".join(f"{item.name}={item.reason}" for item in skipped_files) or "none"
        raise SystemExit(f"No valid supported files available after validation. Skipped: {skipped_text}")

    run_token = timestamp_token(now)
    archive_dir = ARTIFACTS_ROOT / run_token
    staged_input_dir = archive_dir / "inputs"
    report_dir = REPORTS_ROOT / run_token
    report_path = report_dir / "controlled_run_report.md"

    staged_files = copy_valid_files(valid_files, staged_input_dir)
    deferred_files = [
        FileDecision(
            name=item.original_path.name,
            path=str(item.original_path),
            category="deferred",
            reason=f"controlled_mode_max_documents_limit_{MAX_DOCUMENTS_PER_RUN}",
        )
        for item in staged_files[MAX_DOCUMENTS_PER_RUN:]
    ]
    validation_summary = execute_controlled_validation(
        staged_files,
        dataset_dir=staged_input_dir,
        runtime_dir=archive_dir / "runtime",
        requested_limit=MAX_DOCUMENTS_PER_RUN,
        specialty="general",
        quota_safe=True,
    )
    summary = build_controlled_run_summary(
        run_token=run_token,
        input_dir=input_dir,
        archive_dir=archive_dir,
        staged_input_dir=staged_input_dir,
        report_path=report_path,
        valid_files=valid_files,
        staged_files=staged_files,
        skipped_files=skipped_files,
        ignored_files=ignored_files,
        deferred_files=deferred_files,
        validation_summary=validation_summary,
    )
    write_controlled_run_outputs(summary, archive_dir=archive_dir, report_path=report_path)
    exit_code = 0 if summary["hard_failures"] == 0 else 1
    return summary, exit_code


def main() -> int:
    summary, exit_code = run_controlled_real_use()
    print(render_console_summary(summary))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
