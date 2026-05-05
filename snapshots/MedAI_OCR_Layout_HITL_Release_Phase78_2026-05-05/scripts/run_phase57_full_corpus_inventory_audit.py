"""Run a local-only full-corpus inventory audit with PHI-safe public reports."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


# ---------------------------------------------------------------------------
# Phase 57A — Filesystem reconciliation helpers
#
# Phase 57's earlier discovery silently dropped files starting with "." and
# files named ".gitkeep", which left a gap between the public report's
# `total_discovered` and the host filesystem's true count. Phase 57A adds a
# parallel enumerator that walks every regular file and every folder without
# any filtering, then classifies each file into one of seven accounting
# categories so the reconciliation block can prove that
# `total_filesystem_files == accounted_total`.
# ---------------------------------------------------------------------------

# Names that are unambiguously OS / VCS metadata, not corpus content.
_IGNORED_SYSTEM_FILE_NAMES = frozenset({
    ".gitkeep",
    ".gitignore",
    ".gitattributes",
    ".DS_Store",
    "._.DS_Store",
    "desktop.ini",
    "Desktop.ini",
    "DESKTOP.INI",
    "Thumbs.db",
    "thumbs.db",
    "ehthumbs.db",
    "ehthumbs_vista.db",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    ".AppleDouble",
    ".LSOverride",
    ".VolumeIcon.icns",
})

# Patterns for OS / app temporary or lock files (not corpus content).
_IGNORED_SYSTEM_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^~\$"),         # Office temporary lock files
    re.compile(r"^\.~lock\."),   # LibreOffice lock files
    re.compile(r"^\._"),         # macOS resource-fork metadata
    re.compile(r"\.tmp$", re.I),
    re.compile(r"\.bak$", re.I),
    re.compile(r"\.swp$", re.I),
    re.compile(r"~$"),           # Editor backup files (must follow .swp)
)

ACCOUNTING_CATEGORIES: tuple[str, ...] = (
    "supported_processed",
    "unsupported_extension",
    "ignored_system_file",
    "skipped_policy",
    "processing_error",
    "inaccessible_file",
    "unknown_unclassified",
)


def is_ignored_system_file(path: Path) -> bool:
    name = path.name
    if name in _IGNORED_SYSTEM_FILE_NAMES:
        return True
    for pattern in _IGNORED_SYSTEM_FILE_PATTERNS:
        if pattern.search(name):
            return True
    return False


def enumerate_filesystem_inventory(input_dir: Path) -> dict[str, Any]:
    """Walk ``input_dir`` recursively and account for every regular file and
    folder. No filtering. Folders are counted as the number of subdirectories
    inside ``input_dir`` (not including the root itself), matching the
    Windows folder-properties convention.
    """
    total_files = 0
    total_folders = 0
    total_bytes = 0
    files: list[Path] = []
    inaccessible_paths: list[dict[str, str]] = []
    if not input_dir.exists():
        return {
            "total_filesystem_files": 0,
            "total_filesystem_folders": 0,
            "total_filesystem_bytes": 0,
            "files": [],
            "inaccessible_paths": [],
        }
    for root, dirs, names in os.walk(str(input_dir)):
        root_path = Path(root)
        for _ in dirs:
            total_folders += 1
        for name in names:
            path = root_path / name
            try:
                if path.is_file():
                    total_files += 1
                    files.append(path)
                    try:
                        total_bytes += path.stat().st_size
                    except OSError as exc:
                        inaccessible_paths.append({
                            "filename_hash": _safe_filename_hash(name),
                            "reason": f"stat_failed:{exc.__class__.__name__}",
                        })
            except OSError as exc:
                inaccessible_paths.append({
                    "filename_hash": _safe_filename_hash(name),
                    "reason": f"is_file_failed:{exc.__class__.__name__}",
                })
    files.sort(key=lambda item: str(item).lower())
    return {
        "total_filesystem_files": total_files,
        "total_filesystem_folders": total_folders,
        "total_filesystem_bytes": total_bytes,
        "files": files,
        "inaccessible_paths": inaccessible_paths,
    }


def _safe_filename_hash(name: str) -> str:
    # Lazy import to avoid binding at module load if hashing helper changes.
    return hash_filename(name)


def _safe_size_or_none(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def ignored_system_file_result(
    source_path: Path,
    safe_id: str,
    filename_hash: str,
    content_hash: str | None,
) -> dict[str, Any]:
    try:
        size = source_path.stat().st_size
    except OSError:
        size = None
    return {
        "safe_file_id": safe_id,
        "file_id": safe_id,
        "original_filename_redacted": "[REDACTED]",
        "filename_hash": filename_hash,
        "content_hash": content_hash,
        "extension": source_path.suffix.lower(),
        "file_extension": source_path.suffix.lower(),
        "file_type": "ignored_system_file",
        "file_size_bytes": size,
        "processed": False,
        "status": "ignored",
        "outcome": None,
        "selected_extractor": None,
        "actual_extractor": None,
        "confidence": None,
        "validation_status": None,
        "document_classification": None,
        "document_type": None,
        "language_hint": None,
        "ocr_status": None,
        "image_extension": None,
        "ocr_engine": None,
        "classification_reason_codes": ["ignored_system_file"],
        "review_reason_codes": [],
        "reason_codes": ["ignored_system_file"],
        "external_api_used": False,
        "error_category": None,
        "error": None,
        "skip_reason": "ignored_system_file",
        "accounting_category": "ignored_system_file",
    }


def inaccessible_file_result(
    source_path: Path,
    safe_id: str,
    filename_hash: str,
    content_hash: str | None,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "safe_file_id": safe_id,
        "file_id": safe_id,
        "original_filename_redacted": "[REDACTED]",
        "filename_hash": filename_hash,
        "content_hash": content_hash,
        "extension": source_path.suffix.lower(),
        "file_extension": source_path.suffix.lower(),
        "file_type": "inaccessible",
        "file_size_bytes": None,
        "processed": False,
        "status": "error",
        "outcome": None,
        "selected_extractor": None,
        "actual_extractor": None,
        "confidence": None,
        "validation_status": None,
        "document_classification": None,
        "document_type": None,
        "language_hint": None,
        "ocr_status": None,
        "image_extension": None,
        "ocr_engine": None,
        "classification_reason_codes": ["inaccessible_file"],
        "review_reason_codes": ["inaccessible_file"],
        "reason_codes": ["inaccessible_file"],
        "external_api_used": False,
        "error_category": "inaccessible_file",
        "error": f"inaccessible_file:{exc.__class__.__name__}",
        "skip_reason": "inaccessible_file",
        "accounting_category": "inaccessible_file",
    }


def assign_accounting_category(item: dict[str, Any]) -> str:
    """Decide the accounting category for a processed item.

    ``item`` may already have ``accounting_category`` set by the helpers above
    (ignored / inaccessible). Otherwise classify based on file_type and status.
    """
    existing = item.get("accounting_category")
    if existing in ACCOUNTING_CATEGORIES:
        return existing
    if item.get("file_type") == "unsupported":
        return "unsupported_extension"
    if item.get("file_type") == "ignored_system_file":
        return "ignored_system_file"
    if item.get("file_type") == "inaccessible":
        return "inaccessible_file"
    if item.get("status") == "error":
        return "processing_error"
    if item.get("skip_reason") == "skipped_policy":
        return "skipped_policy"
    if item.get("processed"):
        return "supported_processed"
    return "unknown_unclassified"


def compute_filesystem_reconciliation(
    fs_inventory: dict[str, Any],
    inventory_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(item.get("accounting_category") or "unknown_unclassified") for item in inventory_entries)
    supported_processed = counts.get("supported_processed", 0)
    unsupported_extension = counts.get("unsupported_extension", 0)
    ignored_system_files = counts.get("ignored_system_file", 0)
    skipped_policy = counts.get("skipped_policy", 0)
    processing_errors = counts.get("processing_error", 0)
    inaccessible_files = counts.get("inaccessible_file", 0)
    unknown_unclassified = counts.get("unknown_unclassified", 0)
    accounted_total = (
        supported_processed
        + unsupported_extension
        + ignored_system_files
        + skipped_policy
        + processing_errors
        + inaccessible_files
        + unknown_unclassified
    )
    total_filesystem_files = int(fs_inventory.get("total_filesystem_files") or 0)
    unexplained_count = total_filesystem_files - accounted_total
    reconciliation_passed = unexplained_count == 0 and unknown_unclassified == 0
    return {
        "total_filesystem_files": total_filesystem_files,
        "total_filesystem_folders": int(fs_inventory.get("total_filesystem_folders") or 0),
        "total_filesystem_bytes": int(fs_inventory.get("total_filesystem_bytes") or 0),
        "total_supported_files": supported_processed + processing_errors,
        "total_supported_processed": supported_processed,
        "total_unsupported_extension": unsupported_extension,
        "total_ignored_system_files": ignored_system_files,
        "total_skipped_policy": skipped_policy,
        "total_processing_errors": processing_errors,
        "total_inaccessible_files": inaccessible_files,
        "total_unknown_unclassified": unknown_unclassified,
        "accounted_total": accounted_total,
        "unexplained_count": unexplained_count,
        "reconciliation_passed": reconciliation_passed,
        "inaccessible_paths_during_walk": list(fs_inventory.get("inaccessible_paths") or []),
    }


def build_extension_distribution(inventory_entries: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in inventory_entries:
        ext = str(item.get("extension") or "").strip().lower()
        if not ext:
            counter["no_extension"] += 1
        else:
            counter[ext] += 1
    return dict(sorted(counter.items()))


def build_unsupported_extension_distribution(inventory_entries: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in inventory_entries:
        if item.get("accounting_category") != "unsupported_extension":
            continue
        ext = str(item.get("extension") or "").strip().lower()
        if not ext:
            counter["no_extension"] += 1
        else:
            counter[ext] += 1
    return dict(sorted(counter.items()))


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["MEDAI_LOCAL_ONLY"] = "true"
os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ["MEDAI_REQUIRE_PII_SCRUB"] = "true"
os.environ["MEDAI_PRIVACY_AUDIT"] = "true"

from execution.pipeline import ExecutionPipeline
from privacy.privacy_audit import phi_artifact_tracking_status, write_json
from scripts.run_phase53_blind_pdf_generalization_audit import (
    SUPPORTED_EXTENSIONS,
    force_local_only_runtime,
    hash_file_content,
    hash_filename,
    process_one_blind_file,
)


INPUT_DIR = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase57_full_corpus_inventory_audit"
JSON_REPORT = REPORT_DIR / "phase57_full_corpus_inventory_audit_report.json"
MD_REPORT = REPORT_DIR / "phase57_full_corpus_inventory_audit_report.md"
OPERATOR_SUMMARY = REPORT_DIR / "phase57_full_corpus_inventory_audit_operator_summary.md"
CLUSTERS_JSON = REPORT_DIR / "phase57_full_corpus_problem_clusters.json"
CLUSTERS_MD = REPORT_DIR / "phase57_full_corpus_problem_clusters.md"
PRIVATE_MAPPING = REPORT_DIR / "local_filename_mapping_PRIVATE.json"
CHECKPOINT = REPORT_DIR / "phase57_safe_progress_checkpoint.json"
SUPPORTED_EXTENSIONS_DISPLAY = ["PDF", "TXT", "RTF", "TIF", "TIFF", "PNG", "JPG", "JPEG", "BMP", "WEBP"]


def ensure_full_corpus_input(input_dir: Path = INPUT_DIR) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / ".gitkeep").touch(exist_ok=True)


def discover_corpus_files(input_dir: Path = INPUT_DIR, *, recursive: bool = True) -> list[Path]:
    ensure_full_corpus_input(input_dir)
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.name == ".gitkeep" or path.name.startswith("."):
            continue
        files.append(path)
    return sorted(files, key=lambda item: safe_relative_path(item, input_dir).lower())


def supported_corpus_files(input_dir: Path = INPUT_DIR, *, recursive: bool = True) -> list[Path]:
    return [path for path in discover_corpus_files(input_dir, recursive=recursive) if path.suffix.lower() in SUPPORTED_EXTENSIONS]


def run_inventory_audit(
    *,
    input_dir: Path = INPUT_DIR,
    report_dir: Path = REPORT_DIR,
    pipeline: ExecutionPipeline | None = None,
    recursive: bool = True,
) -> dict[str, Any]:
    force_local_only_runtime()
    ensure_full_corpus_input(input_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Phase 57A: enumerate every regular file and folder first, no filter.
    fs_inventory = enumerate_filesystem_inventory(input_dir)
    all_filesystem_files: list[Path] = list(fs_inventory["files"])

    # Split into ignored-system entries (counted but not processed) and the
    # processable subset (which preserves Phase 57's existing semantics for
    # `total_discovered`, `results[0]`, etc.).
    processable: list[Path] = []
    ignored_paths: list[Path] = []
    for path in all_filesystem_files:
        if is_ignored_system_file(path):
            ignored_paths.append(path)
        else:
            processable.append(path)
    processable.sort(
        key=lambda item: safe_relative_path(item, input_dir).lower()
    )
    ignored_paths.sort(
        key=lambda item: safe_relative_path(item, input_dir).lower()
    )

    run_id = str(uuid4())
    active_pipeline = pipeline or ExecutionPipeline()
    if hasattr(active_pipeline, "router"):
        active_pipeline.router.gemini_quota_blocked = True

    results: list[dict[str, Any]] = []
    inventory_entries: list[dict[str, Any]] = []
    private_mapping: dict[str, dict[str, Any]] = {}
    for index, source_path in enumerate(processable, start=1):
        safe_id = f"corpus_file_{index:06d}"
        filename_hash = hash_filename(source_path.name)
        try:
            content_hash = hash_file_content(source_path)
        except OSError as exc:
            # Treat unreadable supported/unsupported files as inaccessible.
            private_mapping[safe_id] = {
                "original_filename": source_path.name,
                "original_relative_path": safe_relative_path(source_path, input_dir),
                "filename_hash": filename_hash,
                "content_hash": None,
                "file_size_bytes": None,
                "accounting_category": "inaccessible_file",
            }
            item = inaccessible_file_result(source_path, safe_id, filename_hash, None, exc)
            results.append(item)
            inventory_entries.append(item)
            write_safe_checkpoint(report_dir, run_id, results)
            continue

        try:
            file_size = source_path.stat().st_size
        except OSError:
            file_size = None
        private_mapping[safe_id] = {
            "original_filename": source_path.name,
            "original_relative_path": safe_relative_path(source_path, input_dir),
            "filename_hash": filename_hash,
            "content_hash": content_hash,
            "file_size_bytes": file_size,
        }
        print(f"Processing {index}/{len(processable)}: {safe_id}")
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            item = unsupported_file_result(source_path, safe_id, filename_hash, content_hash)
        else:
            try:
                item = process_one_blind_file(active_pipeline, source_path, file_id=safe_id, run_id=run_id)
            except Exception as exc:
                item = processing_error_result(source_path, safe_id, filename_hash, content_hash, exc)
            item["safe_file_id"] = safe_id
            item["processed"] = item.get("status") != "error"
            item["actual_extractor"] = item.get("actual_extractor") or item.get("selected_extractor")
            item["extension"] = source_path.suffix.lower()
            item["error_category"] = "processing_error" if item.get("status") == "error" else None
            if source_path.suffix.lower() == ".pdf":
                item = apply_pdf_inventory_metadata(source_path, item)
        item["accounting_category"] = assign_accounting_category(item)
        results.append(item)
        inventory_entries.append(item)
        write_safe_checkpoint(report_dir, run_id, results)
        private_mapping[safe_id]["accounting_category"] = item["accounting_category"]

    # Phase 57A: include ignored system files in inventory_entries (and the
    # private mapping) but not in `results` so the existing test contract for
    # `results[0].safe_file_id == corpus_file_000001` is preserved.
    for offset, source_path in enumerate(ignored_paths, start=1):
        safe_id = f"corpus_system_{offset:06d}"
        filename_hash = hash_filename(source_path.name)
        try:
            content_hash = hash_file_content(source_path)
        except OSError:
            content_hash = None
        private_mapping[safe_id] = {
            "original_filename": source_path.name,
            "original_relative_path": safe_relative_path(source_path, input_dir),
            "filename_hash": filename_hash,
            "content_hash": content_hash,
            "file_size_bytes": _safe_size_or_none(source_path),
            "accounting_category": "ignored_system_file",
        }
        inventory_entries.append(
            ignored_system_file_result(source_path, safe_id, filename_hash, content_hash)
        )

    public_report_names = [
        JSON_REPORT.name,
        MD_REPORT.name,
        OPERATOR_SUMMARY.name,
        CLUSTERS_JSON.name,
        CLUSTERS_MD.name,
        CHECKPOINT.name,
    ]
    phi_artifacts = phi_artifact_tracking_status()
    distributions = build_distributions(results)
    clusters = build_problem_clusters(results)
    counts = count_inventory_results(results)
    filesystem_reconciliation = compute_filesystem_reconciliation(fs_inventory, inventory_entries)
    extension_distribution = build_extension_distribution(inventory_entries)
    unsupported_extension_distribution = build_unsupported_extension_distribution(inventory_entries)
    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase57 Full Corpus Local Inventory Audit",
        "run_id": run_id,
        "input_folder": "full_corpus_input",
        "recursive": recursive,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "total_discovered": len(processable),
        "total_supported": sum(1 for item in results if item.get("file_type") != "unsupported"),
        "total_processed": counts["processed"],
        "unsupported_count": counts["unsupported"],
        "accepted": counts["accepted"],
        "review": counts["review"],
        "review_ocr_quality": counts["review_ocr_quality"],
        "empty": counts["empty"],
        "errors": counts["errors"],
        "external_api_used": any(item.get("external_api_used") for item in results),
        "local_only_forced": True,
        "raw_phi_logged_in_public_reports": False,
        "report_pdf_artifacts_tracked": not phi_artifacts["passed"],
        "phi_artifact_check": phi_artifacts,
        "distribution_by_extension": distributions["extension"],
        "distribution_by_file_type": distributions["file_type"],
        "distribution_by_status": distributions["status"],
        "distribution_by_document_class": distributions["document_class"],
        "distribution_by_ocr_status": distributions["ocr_status"],
        "distribution_by_language_hint": distributions["language_hint"],
        "top_reason_code_clusters": distributions["reason_codes"],
        "files_requiring_attention": [
            item["safe_file_id"] for item in results if item.get("status") not in {"accepted"}
        ],
        "problem_clusters": clusters,
        "recommendations": next_action_recommendations(clusters, counts),
        "results": results,
        "filesystem_reconciliation": filesystem_reconciliation,
        "extension_distribution": extension_distribution,
        "unsupported_extension_distribution": unsupported_extension_distribution,
        "filesystem_inventory_entries": inventory_entries,
    }
    report["conclusion"] = conclusion_for(report)

    write_public_reports(report_dir, report, clusters)
    write_json(report_dir / PRIVATE_MAPPING.name, {"run_id": run_id, "files": private_mapping})
    if public_reports_contain_private_names(report_dir, private_mapping):
        report["raw_phi_logged_in_public_reports"] = True
        report["conclusion"] = conclusion_for(report)
        write_public_reports(report_dir, report, clusters)
    return report


def unsupported_file_result(source_path: Path, safe_id: str, filename_hash: str, content_hash: str) -> dict[str, Any]:
    return {
        "safe_file_id": safe_id,
        "file_id": safe_id,
        "original_filename_redacted": "[REDACTED]",
        "filename_hash": filename_hash,
        "content_hash": content_hash,
        "extension": source_path.suffix.lower(),
        "file_extension": source_path.suffix.lower(),
        "file_type": "unsupported",
        "file_size_bytes": source_path.stat().st_size,
        "processed": False,
        "status": "error",
        "outcome": None,
        "selected_extractor": None,
        "actual_extractor": None,
        "confidence": None,
        "validation_status": None,
        "document_classification": None,
        "document_type": None,
        "language_hint": None,
        "ocr_status": None,
        "image_extension": None,
        "ocr_engine": None,
        "classification_reason_codes": ["unsupported_format"],
        "review_reason_codes": ["unsupported_format"],
        "reason_codes": ["unsupported_format"],
        "external_api_used": False,
        "error_category": "unsupported_format",
        "error": "unsupported_format",
    }


def processing_error_result(
    source_path: Path,
    safe_id: str,
    filename_hash: str,
    content_hash: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "safe_file_id": safe_id,
        "file_id": safe_id,
        "original_filename_redacted": "[REDACTED]",
        "filename_hash": filename_hash,
        "content_hash": content_hash,
        "extension": source_path.suffix.lower(),
        "file_extension": source_path.suffix.lower(),
        "file_type": "image" if source_path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"} else ("pdf" if source_path.suffix.lower() == ".pdf" else "txt"),
        "file_size_bytes": source_path.stat().st_size,
        "processed": False,
        "status": "error",
        "outcome": None,
        "selected_extractor": None,
        "actual_extractor": None,
        "confidence": None,
        "validation_status": None,
        "document_classification": None,
        "document_type": None,
        "language_hint": None,
        "ocr_status": None,
        "image_extension": source_path.suffix.lower(),
        "ocr_engine": None,
        "classification_reason_codes": ["processing_error"],
        "review_reason_codes": ["processing_error"],
        "reason_codes": ["processing_error"],
        "external_api_used": False,
        "error_category": "processing_error",
        "error": "processing_error",
    }


def apply_pdf_inventory_metadata(source_path: Path, item: dict[str, Any]) -> dict[str, Any]:
    metadata = inspect_pdf_inventory_metadata(source_path)
    item["page_count"] = metadata["page_count"]
    item["pdf_embedded_files_detected"] = metadata["embedded_files_detected"]
    item["pdf_embedded_file_count"] = metadata["embedded_file_count"]
    item["possible_multi_document_pdf"] = metadata["possible_multi_document_pdf"]
    item["pdf_document_class_signals"] = metadata["document_class_signals"]
    reason_codes = list(item.get("reason_codes") or item.get("classification_reason_codes") or [])
    if metadata["embedded_files_detected"]:
        reason_codes.append("pdf_portfolio_or_embedded_files_detected")
    if metadata["possible_multi_document_pdf"]:
        reason_codes.append("possible_multi_document_pdf")
    reason_codes = sorted(set(reason_codes))
    item["reason_codes"] = reason_codes
    item["classification_reason_codes"] = reason_codes
    item["review_reason_codes"] = sorted(
        set(list(item.get("review_reason_codes") or []) + [code for code in reason_codes if "pdf" in code or "review" in code])
    )
    if metadata["embedded_files_detected"] or metadata["possible_multi_document_pdf"]:
        item["status"] = "review"
        item["outcome"] = item.get("outcome") or "queued_for_review"
        item["validation_status"] = item.get("validation_status") or "needs_review"
        item["processed"] = True
        item["error_category"] = None
        item["error"] = None
    return item


def inspect_pdf_inventory_metadata(source_path: Path) -> dict[str, Any]:
    metadata = {
        "page_count": None,
        "embedded_files_detected": False,
        "embedded_file_count": 0,
        "possible_multi_document_pdf": False,
        "document_class_signals": [],
    }
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(source_path))
        pages = list(reader.pages)
        metadata["page_count"] = len(pages)
        embedded_count = embedded_file_count_from_reader(reader)
        metadata["embedded_file_count"] = embedded_count
        metadata["embedded_files_detected"] = embedded_count > 0
        class_signals: set[str] = set()
        for page in pages[:20]:
            try:
                class_signals.update(detect_document_class_signals(page.extract_text() or ""))
            except Exception:
                continue
        metadata["document_class_signals"] = sorted(class_signals)
        metadata["possible_multi_document_pdf"] = len(class_signals) >= 2
    except Exception:
        pass
    return metadata


def embedded_file_count_from_reader(reader: Any) -> int:
    count = 0
    try:
        attachments = getattr(reader, "attachments", None)
        if isinstance(attachments, dict):
            count += len(attachments)
    except Exception:
        pass
    try:
        catalog = resolve_pdf_object(reader.trailer.get("/Root", {}))
        names = catalog.get("/Names") if hasattr(catalog, "get") else None
        names = resolve_pdf_object(names)
        embedded = names.get("/EmbeddedFiles") if hasattr(names, "get") else None
        embedded = resolve_pdf_object(embedded)
        embedded_names = embedded.get("/Names") if hasattr(embedded, "get") else None
        if embedded_names:
            count = max(count, len(embedded_names) // 2)
    except Exception:
        pass
    return count


def resolve_pdf_object(value: Any) -> Any:
    try:
        return value.get_object() if hasattr(value, "get_object") else value
    except Exception:
        return value


def detect_document_class_signals(text: str) -> set[str]:
    lowered = text.lower()
    signals: set[str] = set()
    if any(token in lowered for token in ["glucose", "cbc", "wbc", "hemoglobin", "reference range", "lipid panel"]):
        signals.add("lab_report")
    if any(token in lowered for token in ["ecg", "ekg", "12-lead", "ventricular rate", "qt interval"]):
        signals.add("ecg")
    if any(token in lowered for token in ["rx", "prescription", "sig:", "dispense", "refills"]):
        signals.add("prescription")
    if any(token in lowered for token in ["pcr", "culture", "microbiology", "organism", "antibiotic susceptibility"]):
        signals.add("microbiology_pcr")
    if any(token in lowered for token in ["x-ray", "mri", "ct scan", "ultrasound", "impression:"]):
        signals.add("imaging")
    if any(token in lowered for token in ["assessment", "plan", "chief complaint", "history of present illness"]):
        signals.add("visit_note")
    if any(token in lowered for token in ["insurance", "policy", "claim", "member id"]):
        signals.add("insurance_admin")
    return signals


def write_safe_checkpoint(report_dir: Path, run_id: str, results: list[dict[str, Any]]) -> None:
    checkpoint = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "processed_records": len(results),
        "safe_file_ids": [item.get("safe_file_id") or item.get("file_id") for item in results],
    }
    write_json(report_dir / CHECKPOINT.name, checkpoint)


def build_distributions(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    reason_counter: Counter[str] = Counter()
    for item in results:
        reason_counter.update(str(code) for code in item.get("reason_codes") or item.get("classification_reason_codes") or [])
    return {
        "extension": dict(Counter(str(item.get("extension") or item.get("file_extension") or "unknown") for item in results)),
        "file_type": dict(Counter(str(item.get("file_type") or "unknown") for item in results)),
        "status": dict(Counter(str(item.get("status") or "unknown") for item in results)),
        "document_class": dict(
            Counter(str(item.get("document_classification") or item.get("document_type") or "unknown") for item in results)
        ),
        "ocr_status": dict(Counter(str(item.get("ocr_status") or item.get("ocr_quality_band") or "unknown") for item in results)),
        "language_hint": dict(Counter(str(item.get("language_hint") or "unknown") for item in results)),
        "reason_codes": dict(reason_counter.most_common(20)),
    }


def build_problem_clusters(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for item in results:
        safe_id = str(item.get("safe_file_id") or item.get("file_id"))
        status = str(item.get("status") or "")
        file_type = str(item.get("file_type") or "")
        ocr_status = str(item.get("ocr_status") or item.get("ocr_quality_band") or "")
        document_type = str(item.get("document_type") or item.get("document_classification") or "").lower()
        language_hint = str(item.get("language_hint") or "").lower()
        reason_codes = {str(code) for code in item.get("reason_codes") or item.get("classification_reason_codes") or []}
        if item.get("file_type") == "unsupported" or "unsupported_format" in reason_codes:
            clusters["unsupported_format"].append(safe_id)
        if file_type == "image" and ocr_status in {"poor_ocr", "empty", "weak"}:
            clusters["image_ocr_low_quality"].append(safe_id)
        if file_type == "pdf" and ocr_status in {"poor_ocr", "empty", "weak"}:
            clusters["pdf_ocr_low_quality"].append(safe_id)
        if item.get("empty_extraction_flag") or status == "empty":
            clusters["empty_extraction"].append(safe_id)
        if any("low_confidence" in code for code in reason_codes):
            clusters["rules_based_low_confidence"].append(safe_id)
        if any("lab" in code or "table" in code for code in reason_codes):
            clusters["possible_lab_table_failure"].append(safe_id)
        if "ecg" in document_type:
            clusters["possible_ecg_class"].append(safe_id)
        if "prescription" in document_type:
            clusters["possible_prescription_class"].append(safe_id)
        if "microbiology" in document_type or "pcr" in document_type:
            clusters["possible_microbiology_pcr_class"].append(safe_id)
        if "ru" in language_hint or "cyrillic" in document_type or "russian" in document_type:
            clusters["possible_russian_cyrillic_class"].append(safe_id)
    known_ids = {safe_id for values in clusters.values() for safe_id in values}
    for item in results:
        safe_id = str(item.get("safe_file_id") or item.get("file_id"))
        if safe_id not in known_ids and item.get("status") != "accepted":
            clusters["unknown_other"].append(safe_id)
    for cluster_name in [
        "unsupported_format",
        "image_ocr_low_quality",
        "pdf_ocr_low_quality",
        "empty_extraction",
        "rules_based_low_confidence",
        "possible_lab_table_failure",
        "possible_ecg_class",
        "possible_prescription_class",
        "possible_microbiology_pcr_class",
        "possible_russian_cyrillic_class",
        "unknown_other",
    ]:
        clusters.setdefault(cluster_name, [])
    return {key: sorted(set(value)) for key, value in sorted(clusters.items())}


def count_inventory_results(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "processed": sum(1 for item in results if item.get("file_type") != "unsupported"),
        "unsupported": sum(1 for item in results if item.get("file_type") == "unsupported"),
        "accepted": sum(1 for item in results if item.get("status") == "accepted"),
        "review": sum(1 for item in results if item.get("status") in {"review", "review_ocr_quality", "empty"}),
        "review_ocr_quality": sum(1 for item in results if item.get("status") == "review_ocr_quality"),
        "empty": sum(1 for item in results if item.get("empty_extraction_flag") or item.get("status") == "empty"),
        "errors": sum(1 for item in results if item.get("status") == "error"),
    }


def next_action_recommendations(clusters: dict[str, list[str]], counts: dict[str, int]) -> list[str]:
    recommendations = [
        "Do not tune parsers directly from full corpus.",
        "Pick one problem class for the next phase.",
        "Create a small development subset for that class.",
        "Run regression after each fix.",
    ]
    if clusters.get("image_ocr_low_quality"):
        recommendations.append("Review image OCR quality as a separate class before parser changes.")
    if clusters.get("possible_lab_table_failure"):
        recommendations.append("Use a small lab-table subset if lab structure loss dominates.")
    if counts.get("errors"):
        recommendations.append("Inspect safe error categories before expanding automation.")
    return recommendations


def conclusion_for(report: dict[str, Any]) -> str:
    if report["external_api_used"] or report["raw_phi_logged_in_public_reports"] or report["report_pdf_artifacts_tracked"]:
        return "BLOCKED_PRIVACY_RISK"
    reconciliation = report.get("filesystem_reconciliation") or {}
    if reconciliation and not reconciliation.get("reconciliation_passed", True):
        return "BLOCKED_RECONCILIATION_GAP"
    if report["total_discovered"] == 0:
        return "no_input_files"
    if report["errors"] > 0 and report["unsupported_count"] != report["errors"]:
        return "BLOCKED_ERRORS"
    if report["accepted"] / max(report["total_supported"], 1) >= 0.8 and report["errors"] == 0:
        return "PASS_SAFETY_ACCEPTABLE_AUTOMATION"
    if report["total_supported"] > 0:
        return "PASS_SAFETY_WEAK_AUTOMATION"
    return "PASS_INVENTORY_ONLY"


def write_public_reports(report_dir: Path, report: dict[str, Any], clusters: dict[str, list[str]]) -> None:
    write_json(report_dir / JSON_REPORT.name, report)
    write_json(report_dir / CLUSTERS_JSON.name, clusters)
    (report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    (report_dir / OPERATOR_SUMMARY.name).write_text(render_operator_summary(report), encoding="utf-8")
    (report_dir / CLUSTERS_MD.name).write_text(render_clusters_markdown(clusters), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    reconciliation = report.get("filesystem_reconciliation") or {}
    lines = [
        "# Phase57 Full Corpus Inventory Audit",
        "",
        f"- Timestamp: `{report['timestamp']}`",
        f"- Total discovered: `{report['total_discovered']}`",
        f"- Total supported: `{report['total_supported']}`",
        f"- Total processed: `{report['total_processed']}`",
        f"- Unsupported: `{report['unsupported_count']}`",
        f"- Accepted: `{report['accepted']}`",
        f"- Review: `{report['review']}`",
        f"- Review OCR quality: `{report['review_ocr_quality']}`",
        f"- Empty: `{report['empty']}`",
        f"- Errors: `{report['errors']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Local-only forced: `{report['local_only_forced']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        f"- Report PDF/image artifacts tracked: `{report['report_pdf_artifacts_tracked']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Filesystem Reconciliation (Phase 57A)",
        "",
        f"- total_filesystem_files: `{reconciliation.get('total_filesystem_files', 0)}`",
        f"- total_filesystem_folders: `{reconciliation.get('total_filesystem_folders', 0)}`",
        f"- total_filesystem_bytes: `{reconciliation.get('total_filesystem_bytes', 0)}`",
        f"- total_supported_files: `{reconciliation.get('total_supported_files', 0)}`",
        f"- total_supported_processed: `{reconciliation.get('total_supported_processed', 0)}`",
        f"- total_unsupported_extension: `{reconciliation.get('total_unsupported_extension', 0)}`",
        f"- total_ignored_system_files: `{reconciliation.get('total_ignored_system_files', 0)}`",
        f"- total_skipped_policy: `{reconciliation.get('total_skipped_policy', 0)}`",
        f"- total_processing_errors: `{reconciliation.get('total_processing_errors', 0)}`",
        f"- total_inaccessible_files: `{reconciliation.get('total_inaccessible_files', 0)}`",
        f"- total_unknown_unclassified: `{reconciliation.get('total_unknown_unclassified', 0)}`",
        f"- accounted_total: `{reconciliation.get('accounted_total', 0)}`",
        f"- unexplained_count: `{reconciliation.get('unexplained_count', 0)}`",
        f"- reconciliation_passed: `{reconciliation.get('reconciliation_passed', False)}`",
        "",
        "## Extension Distribution",
        "",
        f"- extension_distribution: `{json.dumps(report.get('extension_distribution', {}), sort_keys=True)}`",
        f"- unsupported_extension_distribution: `{json.dumps(report.get('unsupported_extension_distribution', {}), sort_keys=True)}`",
        "",
        "## Distributions",
        "",
        f"- By extension: `{json.dumps(report['distribution_by_extension'], sort_keys=True)}`",
        f"- By file type: `{json.dumps(report['distribution_by_file_type'], sort_keys=True)}`",
        f"- By status: `{json.dumps(report['distribution_by_status'], sort_keys=True)}`",
        f"- By document class/type: `{json.dumps(report['distribution_by_document_class'], sort_keys=True)}`",
        f"- By OCR status: `{json.dumps(report['distribution_by_ocr_status'], sort_keys=True)}`",
        f"- By language hint: `{json.dumps(report['distribution_by_language_hint'], sort_keys=True)}`",
        "",
        "## Reason-Code Clusters",
        "",
    ]
    if report["top_reason_code_clusters"]:
        lines.extend(f"- `{code}`: `{count}`" for code, count in report["top_reason_code_clusters"].items())
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Files Requiring Attention",
            "",
        ]
    )
    if report["files_requiring_attention"]:
        lines.extend(f"- `{safe_id}`" for safe_id in report["files_requiring_attention"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Per-File Inventory",
            "",
            "| Safe File ID | Filename Hash | Content Hash | Extension | Type | Size | Pages | Status | Extractor | Confidence | OCR Status | PDF Flags | Reason Codes | Error Category |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for item in report["results"]:
        lines.append(
            " | ".join(
                [
                    f"| `{item.get('safe_file_id') or item.get('file_id')}`",
                    f"`{item.get('filename_hash')}`",
                    f"`{item.get('content_hash')}`",
                    f"`{item.get('extension') or item.get('file_extension')}`",
                    f"`{item.get('file_type')}`",
                    f"`{item.get('file_size_bytes')}`",
                    f"`{item.get('page_count') or ''}`",
                    f"`{item.get('status')}`",
                    f"`{item.get('selected_extractor')}`",
                    "" if item.get("confidence") is None else f"`{item.get('confidence')}`",
                    f"`{item.get('ocr_status')}`",
                    f"`{pdf_flags_for(item)}`",
                    f"`{', '.join(item.get('reason_codes') or item.get('classification_reason_codes') or [])}`",
                    f"`{item.get('error_category') or ''}` |",
                ]
            )
        )
    if not report["results"]:
        lines.append("")
        lines.append("No files were found in `full_corpus_input/`.")
    return "\n".join(lines) + "\n"


def render_operator_summary(report: dict[str, Any]) -> str:
    reconciliation = report.get("filesystem_reconciliation") or {}
    lines = [
        "# Phase57 Full Corpus Operator Summary",
        "",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Total discovered: `{report['total_discovered']}`",
        f"- Total supported: `{report['total_supported']}`",
        f"- Accepted: `{report['accepted']}`",
        f"- Review: `{report['review']}`",
        f"- Review OCR quality: `{report['review_ocr_quality']}`",
        f"- Empty: `{report['empty']}`",
        f"- Errors: `{report['errors']}`",
        f"- External API used: `{report['external_api_used']}`",
        f"- Raw PHI logged in public reports: `{report['raw_phi_logged_in_public_reports']}`",
        "",
        "## Filesystem Reconciliation",
        "",
        f"- total_filesystem_files: `{reconciliation.get('total_filesystem_files', 0)}`",
        f"- total_filesystem_folders: `{reconciliation.get('total_filesystem_folders', 0)}`",
        f"- accounted_total: `{reconciliation.get('accounted_total', 0)}`",
        f"- unexplained_count: `{reconciliation.get('unexplained_count', 0)}`",
        f"- reconciliation_passed: `{reconciliation.get('reconciliation_passed', False)}`",
        "",
        "## Problem Clusters",
        "",
    ]
    for cluster, safe_ids in report["problem_clusters"].items():
        lines.append(f"- `{cluster}`: `{len(safe_ids)}`")
    lines.extend(["", "## Next Action", ""])
    lines.extend(f"- {recommendation}" for recommendation in report["recommendations"])
    return "\n".join(lines) + "\n"


def render_clusters_markdown(clusters: dict[str, list[str]]) -> str:
    lines = ["# Phase57 Full Corpus Problem Clusters", ""]
    for cluster, safe_ids in clusters.items():
        lines.append(f"## {cluster}")
        if safe_ids:
            lines.extend(f"- `{safe_id}`" for safe_id in safe_ids)
        else:
            lines.append("- None.")
        lines.append("")
    return "\n".join(lines)


def pdf_flags_for(item: dict[str, Any]) -> str:
    flags: list[str] = []
    if item.get("pdf_embedded_files_detected"):
        flags.append("embedded_files")
    if item.get("possible_multi_document_pdf"):
        flags.append("possible_multi_document")
    return ", ".join(flags)


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return "[OUTSIDE_INPUT_ROOT]"


def public_reports_contain_private_names(report_dir: Path, private_mapping: dict[str, dict[str, Any]]) -> bool:
    private_values: list[str] = []
    for entry in private_mapping.values():
        for key in ("original_filename", "original_relative_path"):
            value = str(entry.get(key) or "")
            if value and value != "[OUTSIDE_INPUT_ROOT]":
                private_values.append(value)
    public_paths = [report_dir / name for name in [JSON_REPORT.name, MD_REPORT.name, OPERATOR_SUMMARY.name, CLUSTERS_JSON.name, CLUSTERS_MD.name]]
    public_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in public_paths if path.exists())
    return any(value and value in public_text for value in private_values)


def main() -> int:
    report = run_inventory_audit()
    reconciliation = report.get("filesystem_reconciliation") or {}
    print("MedAI Phase57 full corpus inventory audit complete.")
    print(f"total_discovered: {report['total_discovered']}")
    print(f"total_supported: {report['total_supported']}")
    print(f"total_processed: {report['total_processed']}")
    print(f"accepted: {report['accepted']}")
    print(f"review: {report['review']}")
    print(f"review_ocr_quality: {report['review_ocr_quality']}")
    print(f"empty: {report['empty']}")
    print(f"errors: {report['errors']}")
    print(f"external_api_used: {report['external_api_used']}")
    print(f"raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"total_filesystem_files: {reconciliation.get('total_filesystem_files', 0)}")
    print(f"total_filesystem_folders: {reconciliation.get('total_filesystem_folders', 0)}")
    print(f"accounted_total: {reconciliation.get('accounted_total', 0)}")
    print(f"unexplained_count: {reconciliation.get('unexplained_count', 0)}")
    print(f"reconciliation_passed: {reconciliation.get('reconciliation_passed', False)}")
    print(f"json_report: {JSON_REPORT}")
    print(f"operator_summary: {OPERATOR_SUMMARY}")
    return 1 if report["conclusion"] in {
        "BLOCKED_ERRORS",
        "BLOCKED_PRIVACY_RISK",
        "BLOCKED_RECONCILIATION_GAP",
    } else 0


if __name__ == "__main__":
    raise SystemExit(main())
