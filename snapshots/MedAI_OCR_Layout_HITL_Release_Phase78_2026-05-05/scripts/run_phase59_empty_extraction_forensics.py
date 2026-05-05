"""Phase 59 — Empty Extraction Forensic Subset Audit.

Reads the Phase 57A full corpus inventory report and, for a deterministic
stratified subset of files where extraction returned zero medical entities,
runs non-destructive local diagnostics to bucket each file by likely root
cause. Touches no extraction logic, OCR routing, classifier code, threshold,
or safety gate.

Privacy:
  - Public output uses safe_file_id only.
  - No raw filenames, raw paths, extracted text, or OCR text are emitted.
  - The local private mapping is read locally to resolve safe_file_id ->
    on-disk path, but is never echoed.
  - The Phase 49 / 57 PHI artifact checks remain valid.

Inputs:
  reports/phase57_full_corpus_inventory_audit/phase57_full_corpus_inventory_audit_report.json
  reports/phase57_full_corpus_inventory_audit/local_filename_mapping_PRIVATE.json

Outputs:
  reports/phase59_empty_extraction_forensics/
    phase59_empty_extraction_forensics_report.json
    phase59_empty_extraction_forensics_report.md
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force local-only as a defence in depth.
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

PHASE57_REPORT = (
    ROOT
    / "reports"
    / "phase57_full_corpus_inventory_audit"
    / "phase57_full_corpus_inventory_audit_report.json"
)
PHASE57_PRIVATE_MAPPING = (
    ROOT
    / "reports"
    / "phase57_full_corpus_inventory_audit"
    / "local_filename_mapping_PRIVATE.json"
)
DEFAULT_INPUT_DIR = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase59_empty_extraction_forensics"
JSON_REPORT = REPORT_DIR / "phase59_empty_extraction_forensics_report.json"
MD_REPORT = REPORT_DIR / "phase59_empty_extraction_forensics_report.md"

DEFAULT_SUBSET_SIZE = 30
DEFAULT_RANDOM_SEED = 20260503  # deterministic stratification

ROOT_CAUSE_BUCKETS: tuple[str, ...] = (
    "blank_or_near_blank",
    "image_only_pdf_needs_ocr",
    "ocr_ran_but_low_text",
    "pdf_text_extraction_gap",
    "embedded_or_portfolio_pdf",
    "unsupported_or_malformed_structure",
    "likely_non_medical_or_admin",
    "pipeline_bug_suspected",
    "unknown_needs_manual_review",
)


# ---------------------------------------------------------------------------
# Selection: pick the empty_extraction subset
# ---------------------------------------------------------------------------


def is_empty_extraction(item: dict[str, Any]) -> bool:
    """An item is considered an empty extraction if any of:
       - explicit empty_extraction_flag
       - status == 'empty'
       - entity_count == 0 (and not an unsupported_extension which Phase 57A
         already accounts for separately)
    """
    if not isinstance(item, dict):
        return False
    if item.get("file_type") == "unsupported":
        return False
    if item.get("accounting_category") == "ignored_system_file":
        return False
    if bool(item.get("empty_extraction_flag")):
        return True
    status = str(item.get("status") or "")
    if status == "empty":
        return True
    if int(item.get("entity_count") or 0) == 0:
        return True
    return False


def select_stratified_subset(
    empty_items: list[dict[str, Any]],
    *,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Stratify by file_type, sample deterministically by seed."""
    if not empty_items or subset_size <= 0:
        return []
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in empty_items:
        ft = str(item.get("file_type") or "unknown")
        by_type.setdefault(ft, []).append(item)
    # Sort each bucket by safe_file_id for stable sampling
    for ft, items in by_type.items():
        items.sort(key=lambda it: str(it.get("safe_file_id") or it.get("file_id") or ""))

    total = len(empty_items)
    # Allocate slots: at least 1 to each non-empty bucket; remainder
    # proportional. Cap at subset_size.
    allocations: dict[str, int] = {ft: 0 for ft in by_type}
    remaining = subset_size
    # Floor allocation by proportion
    for ft, items in by_type.items():
        share = max(1, int(round(len(items) / total * subset_size)))
        share = min(share, len(items))
        allocations[ft] = share
    overflow = sum(allocations.values()) - subset_size
    if overflow > 0:
        # Trim from the largest bucket first, but keep at least 1 per bucket
        for ft in sorted(allocations, key=lambda k: -allocations[k]):
            while overflow > 0 and allocations[ft] > 1:
                allocations[ft] -= 1
                overflow -= 1
            if overflow == 0:
                break
    elif overflow < 0:
        # Distribute remaining slots to largest buckets
        for ft in sorted(by_type, key=lambda k: -len(by_type[k])):
            cap = len(by_type[ft])
            while overflow < 0 and allocations[ft] < cap:
                allocations[ft] += 1
                overflow += 1
            if overflow == 0:
                break

    subset: list[dict[str, Any]] = []
    for ft, slot in allocations.items():
        items = by_type[ft]
        if slot >= len(items):
            chosen = list(items)
        else:
            chosen = rng.sample(items, slot)
        subset.extend(chosen)
    subset.sort(key=lambda it: str(it.get("safe_file_id") or it.get("file_id") or ""))
    return subset


# ---------------------------------------------------------------------------
# Local diagnostics — non-destructive, no text content emitted.
# ---------------------------------------------------------------------------


def _bucket_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    if size_bytes < 1024:
        return "lt_1KB"
    if size_bytes < 10 * 1024:
        return "1_10KB"
    if size_bytes < 100 * 1024:
        return "10_100KB"
    if size_bytes < 1024 * 1024:
        return "100KB_1MB"
    if size_bytes < 10 * 1024 * 1024:
        return "1_10MB"
    return "gt_10MB"


def _bucket_text_length(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length == 0:
        return "zero"
    if length < 50:
        return "1_50"
    if length < 200:
        return "50_200"
    if length < 1000:
        return "200_1000"
    if length < 5000:
        return "1000_5000"
    return "gt_5000"


def diagnose_pdf(path: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "page_count": None,
        "has_embedded_text": False,
        "extracted_text_length_bucket": "unknown",
        "ocr_attempted": False,
        "image_object_observed": False,
        "embedded_files_detected": False,
        "structure_error": None,
    }
    try:
        import fitz  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        diagnostics["structure_error"] = f"fitz_import_failed:{exc.__class__.__name__}"
        return diagnostics
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        diagnostics["structure_error"] = f"open_failed:{exc.__class__.__name__}"
        return diagnostics
    try:
        diagnostics["page_count"] = len(doc)
        total_text_len = 0
        any_image = False
        # Inspect up to first 8 pages to bound runtime.
        for page in doc[: min(8, len(doc))]:
            try:
                text = page.get_text() or ""
                total_text_len += len(text.strip())
                # Image objects on the page
                images = page.get_images(full=False)
                if images:
                    any_image = True
            except Exception:
                continue
        diagnostics["has_embedded_text"] = total_text_len >= 50
        diagnostics["extracted_text_length_bucket"] = _bucket_text_length(total_text_len)
        diagnostics["image_object_observed"] = any_image
        try:
            embedded = doc.embfile_count() if hasattr(doc, "embfile_count") else 0
            diagnostics["embedded_files_detected"] = bool(embedded)
        except Exception:
            diagnostics["embedded_files_detected"] = False
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return diagnostics


def diagnose_image(path: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "frame_count": None,
        "width": None,
        "height": None,
        "structure_error": None,
    }
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        diagnostics["structure_error"] = f"pil_import_failed:{exc.__class__.__name__}"
        return diagnostics
    try:
        with Image.open(path) as im:
            diagnostics["width"], diagnostics["height"] = im.size
            n_frames = getattr(im, "n_frames", 1)
            diagnostics["frame_count"] = int(n_frames)
    except Exception as exc:  # noqa: BLE001
        diagnostics["structure_error"] = f"open_failed:{exc.__class__.__name__}"
    return diagnostics


def diagnose_text(path: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "extracted_text_length_bucket": "unknown",
        "structure_error": None,
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        diagnostics["extracted_text_length_bucket"] = _bucket_text_length(len(text.strip()))
    except Exception as exc:  # noqa: BLE001
        diagnostics["structure_error"] = f"read_failed:{exc.__class__.__name__}"
    return diagnostics


def assign_root_cause(
    file_type: str,
    diagnostics: dict[str, Any],
    *,
    file_size_bytes: int | None,
    ocr_status: str | None,
    document_type: str | None,
) -> str:
    size_bucket = _bucket_size(file_size_bytes)
    structure_error = diagnostics.get("structure_error")
    text_bucket = str(diagnostics.get("extracted_text_length_bucket") or "unknown")
    page_count = diagnostics.get("page_count")
    has_embedded_text = bool(diagnostics.get("has_embedded_text"))
    image_object_observed = bool(diagnostics.get("image_object_observed"))
    embedded_files = bool(diagnostics.get("embedded_files_detected"))

    if structure_error and "open_failed" in str(structure_error):
        return "unsupported_or_malformed_structure"
    if size_bucket == "lt_1KB":
        return "blank_or_near_blank"
    if file_type == "pdf":
        if embedded_files:
            return "embedded_or_portfolio_pdf"
        if (page_count or 0) == 0:
            return "blank_or_near_blank"
        if not has_embedded_text and image_object_observed:
            # Scanned/image-only PDF where OCR layer either didn't run or
            # didn't produce text.
            if ocr_status in {"poor_ocr", "empty"}:
                return "ocr_ran_but_low_text"
            return "image_only_pdf_needs_ocr"
        if not has_embedded_text and not image_object_observed:
            return "pipeline_bug_suspected"
        # Has embedded text. Why did extraction return nothing?
        if ocr_status in {"poor_ocr", "empty"}:
            return "ocr_ran_but_low_text"
        if text_bucket in {"zero", "1_50"}:
            return "blank_or_near_blank"
        if text_bucket in {"50_200"}:
            return "likely_non_medical_or_admin"
        return "pdf_text_extraction_gap"
    if file_type == "image":
        if structure_error:
            return "unsupported_or_malformed_structure"
        if ocr_status in {"poor_ocr", "empty"}:
            return "ocr_ran_but_low_text"
        return "image_only_pdf_needs_ocr"
    if file_type == "txt":
        if text_bucket in {"zero", "1_50"}:
            return "blank_or_near_blank"
        if text_bucket == "50_200":
            return "likely_non_medical_or_admin"
        return "pipeline_bug_suspected"
    return "unknown_needs_manual_review"


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def load_phase57(report_path: Path | None = None) -> dict[str, Any]:
    path = report_path or PHASE57_REPORT
    if not path.exists():
        raise FileNotFoundError(f"Phase 57A report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_private_mapping(mapping_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = mapping_path or PHASE57_PRIVATE_MAPPING
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else {}
    return files if isinstance(files, dict) else {}


def resolve_local_path(
    safe_id: str,
    private_mapping: dict[str, dict[str, Any]],
    input_dir: Path,
) -> Path | None:
    entry = private_mapping.get(safe_id) if isinstance(private_mapping, dict) else None
    if not isinstance(entry, dict):
        return None
    rel = str(entry.get("original_relative_path") or "")
    if not rel or rel == "[OUTSIDE_INPUT_ROOT]":
        return None
    candidate = input_dir / rel
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def run_forensics(
    *,
    phase57_report_path: Path | None = None,
    private_mapping_path: Path | None = None,
    input_dir: Path | None = None,
    report_dir: Path | None = None,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    target_report_dir = report_dir or REPORT_DIR
    target_report_dir.mkdir(parents=True, exist_ok=True)
    phase57 = load_phase57(phase57_report_path)
    private_mapping = load_private_mapping(private_mapping_path)
    input_dir_resolved = input_dir or DEFAULT_INPUT_DIR

    empty_items = [item for item in (phase57.get("results") or []) if is_empty_extraction(item)]
    subset = select_stratified_subset(empty_items, subset_size=subset_size, seed=seed)
    forensic_entries: list[dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter({b: 0 for b in ROOT_CAUSE_BUCKETS})

    for item in subset:
        safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
        file_type = str(item.get("file_type") or "unknown")
        extension = str(item.get("extension") or item.get("file_extension") or "").lower()
        size_bytes = item.get("file_size_bytes")
        ocr_status = item.get("ocr_status") or item.get("ocr_quality_band")
        document_type = item.get("document_type") or item.get("document_classification")
        local_path = resolve_local_path(safe_id, private_mapping, input_dir_resolved)
        if local_path is None:
            diagnostics = {"structure_error": "local_path_not_resolved"}
        elif file_type == "pdf":
            diagnostics = diagnose_pdf(local_path)
            diagnostics["ocr_attempted"] = ocr_status not in {None, "unknown"}
        elif file_type == "image":
            diagnostics = diagnose_image(local_path)
            diagnostics["ocr_attempted"] = ocr_status not in {None, "unknown"}
        elif file_type == "txt":
            diagnostics = diagnose_text(local_path)
        else:
            diagnostics = {"structure_error": f"unsupported_diagnose_type:{file_type}"}

        bucket = assign_root_cause(
            file_type,
            diagnostics,
            file_size_bytes=size_bytes,
            ocr_status=ocr_status,
            document_type=document_type,
        )
        bucket_counter[bucket] += 1

        forensic_entries.append({
            "safe_file_id": safe_id,
            "file_type": file_type,
            "extension": extension,
            "file_size_bytes": size_bytes,
            "size_bucket": _bucket_size(size_bytes),
            "ocr_status_phase57": ocr_status,
            "document_type_phase57": document_type,
            "page_count": diagnostics.get("page_count"),
            "frame_count": diagnostics.get("frame_count"),
            "has_embedded_text": diagnostics.get("has_embedded_text"),
            "extracted_text_length_bucket": diagnostics.get("extracted_text_length_bucket"),
            "image_object_observed": diagnostics.get("image_object_observed"),
            "embedded_files_detected": diagnostics.get("embedded_files_detected"),
            "ocr_attempted": diagnostics.get("ocr_attempted"),
            "structure_error": diagnostics.get("structure_error"),
            "root_cause_bucket": bucket,
        })

    bucket_distribution = dict(sorted(bucket_counter.items(), key=lambda kv: (-kv[1], kv[0])))
    total_subset = len(forensic_entries)
    bucket_percent = {
        name: round((count / total_subset) if total_subset else 0.0, 4)
        for name, count in bucket_distribution.items()
    }
    dominant_bucket = next(iter(bucket_distribution)) if bucket_distribution and total_subset > 0 else None
    recommended_phase60 = recommended_phase60_for(dominant_bucket, bucket_distribution)
    privacy_passed = _privacy_self_check(forensic_entries)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 59 Empty Extraction Forensic Subset Audit",
        "source_phase57_report": str(PHASE57_REPORT.relative_to(ROOT)),
        "subset_seed": seed,
        "subset_size_requested": subset_size,
        "subset_size_actual": total_subset,
        "empty_extraction_population": len(empty_items),
        "stratification": dict(Counter(str(item.get("file_type") or "unknown") for item in subset)),
        "root_cause_bucket_counts": bucket_distribution,
        "root_cause_bucket_percent": bucket_percent,
        "dominant_root_cause_bucket": dominant_bucket,
        "recommended_phase60_target": recommended_phase60,
        "subset": forensic_entries,
        "privacy_safety": {
            "uses_safe_ids_only": privacy_passed,
            "raw_filenames_present_in_output": False,
            "raw_paths_present_in_output": False,
            "extracted_text_present_in_output": False,
            "ocr_text_present_in_output": False,
            "phi_present_in_output": False,
            "external_api_used": False,
        },
    }
    if total_subset == 0:
        report["conclusion"] = "no_empty_extraction_files"
    else:
        report["conclusion"] = "forensic_subset_audited"

    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    (target_report_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def recommended_phase60_for(dominant: str | None, distribution: dict[str, int]) -> dict[str, Any]:
    if not dominant or sum(distribution.values()) == 0:
        return {
            "title": "no_action_required",
            "reason": "No empty_extraction files surfaced in the subset.",
        }
    if dominant == "blank_or_near_blank":
        return {
            "title": "phase60_blank_file_handling_diagnostic",
            "reason": (
                "Most empty extractions are genuinely blank/near-blank files. "
                "Phase 60 should add safe-handling and reporting (no auto-accept) "
                "rather than parser changes."
            ),
        }
    if dominant == "image_only_pdf_needs_ocr":
        return {
            "title": "phase60_image_only_pdf_ocr_followup",
            "reason": (
                "Most empty extractions come from image-only / scanned PDFs. "
                "Phase 60 should evaluate whether the existing OCR fallback is "
                "engaged for these files; do not retune extraction yet."
            ),
        }
    if dominant == "ocr_ran_but_low_text":
        return {
            "title": "phase60_ocr_quality_diagnostic",
            "reason": (
                "OCR ran but produced too little text. Phase 60 should focus on "
                "OCR engine selection / DPI tuning for the affected subset, not "
                "downstream extraction."
            ),
        }
    if dominant == "pdf_text_extraction_gap":
        return {
            "title": "phase60_text_extraction_gap_diagnostic",
            "reason": (
                "Text is present in the PDF but downstream extraction returns "
                "nothing. Phase 60 should run a vocabulary / entity-coverage "
                "diagnostic against a subset of these files BEFORE any parser "
                "tuning."
            ),
        }
    if dominant == "embedded_or_portfolio_pdf":
        return {
            "title": "phase60_pdf_portfolio_diagnostic",
            "reason": (
                "Most empty extractions are PDF portfolios with embedded files. "
                "Phase 60 should add operator-visible reporting only; embedded "
                "content extraction is out of scope without explicit consent."
            ),
        }
    if dominant == "unsupported_or_malformed_structure":
        return {
            "title": "phase60_malformed_pdf_diagnostic",
            "reason": (
                "Files fail to open or have malformed structure. Phase 60 should "
                "add safe error categories and operator-visible reporting."
            ),
        }
    if dominant == "likely_non_medical_or_admin":
        return {
            "title": "phase60_non_medical_document_class_diagnostic",
            "reason": (
                "Many empty extractions are likely non-medical / administrative. "
                "Phase 60 should evaluate a document-class prefilter, not a "
                "vocabulary expansion."
            ),
        }
    if dominant == "pipeline_bug_suspected":
        return {
            "title": "phase60_pipeline_bug_investigation",
            "reason": (
                "Several files have neither text nor image content but are still "
                "marked supported. This points to a pipeline bug, not a parser "
                "gap. Phase 60 should be an investigation, not a fix."
            ),
        }
    return {
        "title": "phase60_manual_review_subset",
        "reason": (
            "Dominant bucket is unknown_needs_manual_review. Phase 60 should "
            "surface these files to an operator and not change extraction code."
        ),
    }


# ---------------------------------------------------------------------------
# Privacy self-check
# ---------------------------------------------------------------------------

_FORBIDDEN_KEY_NAMES = (
    "original_filename",
    "original_relative_path",
    "filename",
    "filepath",
    "file_path",
    "absolute_path",
    "extracted_text",
    "ocr_text",
    "raw_text",
    "selected_text",
    "page_text",
    "text_preview",
)

_RAW_FILENAME_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_-]{2,}\.(pdf|docx|rtf|jpg|jpeg|png|tif|tiff|bmp|webp|mp3|ogg|msg|xml|txt)\b",
    re.IGNORECASE,
)


def _privacy_self_check(entries: list[dict[str, Any]]) -> bool:
    """Best-effort self-check that no forbidden keys are present and no
    obvious raw-filename pattern leaked into the entries.
    """
    serialized = json.dumps(entries, default=str)
    for forbidden in _FORBIDDEN_KEY_NAMES:
        if f'"{forbidden}"' in serialized:
            return False
    for match in _RAW_FILENAME_RE.finditer(serialized):
        stem = match.group(0).split(".")[0].lower()
        if stem.startswith("phase") or stem.startswith("corpus_"):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 59 Empty Extraction Forensic Subset Audit",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Source Phase 57A report: `{report['source_phase57_report']}`",
        f"- Subset seed: `{report['subset_seed']}`",
        f"- Subset size requested: `{report['subset_size_requested']}`",
        f"- Subset size actual: `{report['subset_size_actual']}`",
        f"- Empty extraction population: `{report['empty_extraction_population']}`",
        f"- Stratification (file_type → count): `{report['stratification']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Root Cause Bucket Distribution",
        "",
        "| Bucket | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    for bucket, count in report["root_cause_bucket_counts"].items():
        pct = report["root_cause_bucket_percent"].get(bucket, 0.0)
        lines.append(f"| `{bucket}` | {count} | {pct * 100:.2f}% |")
    rec = report.get("recommended_phase60_target") or {}
    lines += [
        "",
        f"- Dominant root cause bucket: `{report.get('dominant_root_cause_bucket')}`",
        "",
        "## Recommended Phase 60 Target",
        "",
        f"- Title: `{rec.get('title')}`",
        "",
        f"_{rec.get('reason')}_",
        "",
        "## Forensic Subset Entries",
        "",
        "| safe_file_id | file_type | extension | size_bucket | OCR | doc_type | pages | text_len | embedded? | bucket |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for entry in report["subset"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{entry['safe_file_id']}`",
                str(entry.get("file_type") or ""),
                str(entry.get("extension") or ""),
                str(entry.get("size_bucket") or ""),
                str(entry.get("ocr_status_phase57") or ""),
                str(entry.get("document_type_phase57") or ""),
                str(entry.get("page_count") if entry.get("page_count") is not None else ""),
                str(entry.get("extracted_text_length_bucket") or ""),
                "yes" if entry.get("has_embedded_text") else "no",
                f"`{entry['root_cause_bucket']}`",
            ])
            + " |"
        )
    lines += [
        "",
        "## Privacy Safety",
        "",
        f"- uses_safe_ids_only: `{report['privacy_safety']['uses_safe_ids_only']}`",
        f"- raw_filenames_present_in_output: `{report['privacy_safety']['raw_filenames_present_in_output']}`",
        f"- raw_paths_present_in_output: `{report['privacy_safety']['raw_paths_present_in_output']}`",
        f"- extracted_text_present_in_output: `{report['privacy_safety']['extracted_text_present_in_output']}`",
        f"- ocr_text_present_in_output: `{report['privacy_safety']['ocr_text_present_in_output']}`",
        f"- phi_present_in_output: `{report['privacy_safety']['phi_present_in_output']}`",
        f"- external_api_used: `{report['privacy_safety']['external_api_used']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_forensics()
    print("MedAI Phase 59 empty extraction forensic subset audit complete.")
    print(f"empty_extraction_population: {report['empty_extraction_population']}")
    print(f"subset_size_actual: {report['subset_size_actual']}")
    print(f"stratification: {report['stratification']}")
    print(f"dominant_root_cause_bucket: {report['dominant_root_cause_bucket']}")
    print(f"root_cause_bucket_counts: {report['root_cause_bucket_counts']}")
    rec = report.get("recommended_phase60_target") or {}
    print(f"recommended_phase60_target: {rec.get('title')}")
    print(f"privacy_safety.uses_safe_ids_only: {report['privacy_safety']['uses_safe_ids_only']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
