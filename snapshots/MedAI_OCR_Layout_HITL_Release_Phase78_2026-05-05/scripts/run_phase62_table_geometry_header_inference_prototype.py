"""Phase 62 — Table Geometry Header Inference Prototype Diagnostic.

For files Phase 61 marked recommended_strategy=table_geometry_header_inference,
evaluate the strength of detectable column geometry as a proxy for whether a
header-inference prototype could recover structured lab data.

Touches no extraction logic, OCR routing, classifier code, threshold, or safety
gate. Production extractors are not modified, and the default recommendation is
production_extractor_should_change_yet=False.

Privacy:
  - Public output uses safe_file_id only.
  - No raw text, OCR text, table rows, inferred labels, raw filenames, or raw
    paths are emitted.
  - All measurements are categorical buckets or booleans derived locally.
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

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")


PHASE57_PRIVATE_MAPPING = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit"
    / "local_filename_mapping_PRIVATE.json"
)
PHASE61_REPORT = (
    ROOT / "reports" / "phase61_header_label_inference_diagnostic"
    / "phase61_header_label_inference_diagnostic_report.json"
)
DEFAULT_INPUT_DIR = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase62_table_geometry_header_inference_prototype"
JSON_REPORT = REPORT_DIR / "phase62_table_geometry_header_inference_prototype_report.json"
MD_REPORT = REPORT_DIR / "phase62_table_geometry_header_inference_prototype_report.md"

DEFAULT_SUBSET_SIZE = 20
DEFAULT_RANDOM_SEED = 20260503


# ---------------------------------------------------------------------------
# Regex patterns — horizontal whitespace only, never cross-line.
# ---------------------------------------------------------------------------

_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"\S+")
_REPEATED_NUMERIC_COLUMN_RE = re.compile(
    # Strictly horizontal whitespace (not \s) to prevent cross-line matching.
    r"^[ \t]*\d+(?:\.\d+)?(?:[ \t]+\d+(?:\.\d+)?){2,}[ \t]*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


def _bucket_count(n: int) -> str:
    if n == 0:
        return "zero"
    if n <= 2:
        return "very_low_1_2"
    if n <= 5:
        return "low_3_5"
    if n <= 15:
        return "medium_6_15"
    if n <= 50:
        return "high_16_50"
    return "very_high_gt_50"


def _bucket_ratio(r: float) -> str:
    if r < 0.05:
        return "very_low_lt_5pct"
    if r < 0.15:
        return "low_5_15pct"
    if r < 0.30:
        return "medium_15_30pct"
    if r < 0.50:
        return "high_30_50pct"
    return "very_high_gt_50pct"


def _bucket_text_length(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length == 0:
        return "zero"
    if length < 200:
        return "1_200"
    if length < 1000:
        return "200_1000"
    if length < 5000:
        return "1000_5000"
    if length < 20000:
        return "5000_20000"
    return "gt_20000"


def _bucket_pages(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n == 0:
        return "zero"
    if n == 1:
        return "1"
    if n <= 3:
        return "2_3"
    if n <= 10:
        return "4_10"
    return "gt_10"


# ---------------------------------------------------------------------------
# Geometry measurement — local only, never emits text
# ---------------------------------------------------------------------------


def measure_geometry_signals(text: str) -> dict[str, Any]:
    """Detect numeric column alignment geometry. Never emits raw text."""
    if not text:
        return _empty_geometry()

    lines = text.splitlines()
    total_lines = max(1, len(lines))

    # --- Repeated numeric column pattern (horizontal whitespace only) ---
    repeated_col_lines = len(_REPEATED_NUMERIC_COLUMN_RE.findall(text))

    # --- Column boundary clustering ---
    # Bucket each numeric-token character offset into 4-char tolerance bands.
    # Track how many distinct column buckets each line contributes.
    per_line_col_buckets: list[frozenset[int]] = []
    for line in lines:
        buckets: set[int] = set()
        for m in _NUMERIC_TOKEN_RE.finditer(line):
            buckets.add(m.start() // 4)
        if len(buckets) >= 2:
            per_line_col_buckets.append(frozenset(buckets))

    # Distinct column bucket positions across all qualifying lines
    all_col_buckets: set[int] = set()
    for fb in per_line_col_buckets:
        all_col_buckets.update(fb)
    distinct_column_count = len(all_col_buckets)

    # Pairs of lines sharing >= 2 column bucket positions (cap at 10 for speed)
    aligned_pair_count = 0
    for i in range(len(per_line_col_buckets)):
        for j in range(i + 1, len(per_line_col_buckets)):
            if len(per_line_col_buckets[i] & per_line_col_buckets[j]) >= 2:
                aligned_pair_count += 1
                if aligned_pair_count >= 10:
                    break
        if aligned_pair_count >= 10:
            break

    # Alignment consistency: fraction of qualifying lines whose col buckets
    # overlap with the most common col-bucket set by >= 2 positions.
    alignment_consistency = 0.0
    if per_line_col_buckets:
        # Build a single consensus col-bucket set from most-common positions.
        bucket_freq: Counter[int] = Counter()
        for fb in per_line_col_buckets:
            bucket_freq.update(fb)
        consensus_buckets = frozenset(
            b for b, cnt in bucket_freq.items()
            if cnt >= max(2, len(per_line_col_buckets) // 3)
        )
        if consensus_buckets:
            matching = sum(
                1 for fb in per_line_col_buckets
                if len(fb & consensus_buckets) >= 2
            )
            alignment_consistency = matching / len(per_line_col_buckets)

    # --- Table block detection ---
    # A table block is a maximal consecutive run of lines where >= 50% of
    # non-empty tokens are numeric.
    table_blocks: list[int] = []   # depths (row counts) of detected blocks
    block_depth = 0
    for line in lines:
        tokens = _TOKEN_RE.findall(line)
        if not tokens:
            if block_depth > 0:
                table_blocks.append(block_depth)
                block_depth = 0
            continue
        numeric_count = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.fullmatch(t))
        if numeric_count / len(tokens) >= 0.5:
            block_depth += 1
        else:
            if block_depth > 0:
                table_blocks.append(block_depth)
                block_depth = 0
    if block_depth > 0:
        table_blocks.append(block_depth)

    table_block_count = len(table_blocks)
    max_block_depth = max(table_blocks, default=0)
    numeric_heavy_line_ratio = sum(table_blocks) / total_lines

    # --- Derived boolean signals ---
    column_alignment_detected = distinct_column_count >= 3 and aligned_pair_count >= 3
    deep_table_block_present = max_block_depth >= 4
    multi_block_structure = table_block_count >= 2

    # geometry_signal_strength
    strong_signals = sum([
        column_alignment_detected,
        deep_table_block_present,
        repeated_col_lines >= 3,
    ])
    weak_signals = sum([
        distinct_column_count >= 2,
        aligned_pair_count >= 1,
        table_block_count >= 1,
        repeated_col_lines >= 1,
        multi_block_structure,
    ])

    if strong_signals >= 2:
        geometry_signal_strength = "high"
    elif strong_signals == 1 or weak_signals >= 3:
        geometry_signal_strength = "medium"
    elif weak_signals >= 1:
        geometry_signal_strength = "low"
    else:
        geometry_signal_strength = "none"

    # recoverable_table_candidate: geometry strong enough to attempt prototype
    recoverable_table_candidate = (
        geometry_signal_strength in ("high", "medium")
        and distinct_column_count >= 2
        and max_block_depth >= 3
    )

    if geometry_signal_strength == "high" and recoverable_table_candidate:
        recovery_confidence_band = "high"
    elif geometry_signal_strength == "medium" and recoverable_table_candidate:
        recovery_confidence_band = "medium"
    else:
        recovery_confidence_band = "low"

    if recoverable_table_candidate and recovery_confidence_band in ("high", "medium"):
        safe_next_action = "prototype_candidate"
    elif geometry_signal_strength in ("medium", "low"):
        safe_next_action = "manual_review_boundary"
    else:
        safe_next_action = "insufficient_geometry"

    return {
        "text_length_bucket": _bucket_text_length(len(text.strip())),
        "numeric_heavy_line_ratio_bucket": _bucket_ratio(numeric_heavy_line_ratio),
        "distinct_column_count_bucket": _bucket_count(distinct_column_count),
        "aligned_pair_count_bucket": _bucket_count(aligned_pair_count),
        "table_block_count_bucket": _bucket_count(table_block_count),
        "max_block_depth_bucket": _bucket_count(max_block_depth),
        "repeated_numeric_column_line_count_bucket": _bucket_count(repeated_col_lines),
        "column_alignment_detected": column_alignment_detected,
        "deep_table_block_present": deep_table_block_present,
        "multi_block_structure": multi_block_structure,
        "geometry_signal_strength": geometry_signal_strength,
        "recoverable_table_candidate": recoverable_table_candidate,
        "recovery_confidence_band": recovery_confidence_band,
        "safe_next_action": safe_next_action,
        "_raw_counts": {
            "distinct_column_count": distinct_column_count,
            "aligned_pair_count": aligned_pair_count,
            "table_block_count": table_block_count,
            "max_block_depth": max_block_depth,
            "repeated_col_lines": repeated_col_lines,
            "numeric_heavy_line_ratio": numeric_heavy_line_ratio,
            "alignment_consistency": alignment_consistency,
        },
    }


def _empty_geometry() -> dict[str, Any]:
    return {
        "text_length_bucket": _bucket_text_length(0),
        "numeric_heavy_line_ratio_bucket": _bucket_ratio(0.0),
        "distinct_column_count_bucket": _bucket_count(0),
        "aligned_pair_count_bucket": _bucket_count(0),
        "table_block_count_bucket": _bucket_count(0),
        "max_block_depth_bucket": _bucket_count(0),
        "repeated_numeric_column_line_count_bucket": _bucket_count(0),
        "column_alignment_detected": False,
        "deep_table_block_present": False,
        "multi_block_structure": False,
        "geometry_signal_strength": "none",
        "recoverable_table_candidate": False,
        "recovery_confidence_band": "low",
        "safe_next_action": "insufficient_geometry",
        "_raw_counts": {},
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_phase61(path: Path | None = None) -> dict[str, Any]:
    p = path or PHASE61_REPORT
    if not p.exists():
        return {"subset": []}
    return json.loads(p.read_text(encoding="utf-8"))


def load_private_mapping(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or PHASE57_PRIVATE_MAPPING
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else {}
    return files if isinstance(files, dict) else {}


def resolve_local_path(
    safe_id: str,
    private_mapping: dict[str, dict[str, Any]],
    input_dir: Path,
) -> Path | None:
    entry = private_mapping.get(safe_id)
    if not isinstance(entry, dict):
        return None
    rel = str(entry.get("original_relative_path") or "")
    if not rel or rel == "[OUTSIDE_INPUT_ROOT]":
        return None
    candidate = input_dir / rel
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def extract_pdf_text_local(path: Path, *, page_limit: int = 12) -> tuple[str, int | None]:
    try:
        import fitz  # type: ignore[import-untyped]
    except Exception:
        return "", None
    try:
        doc = fitz.open(str(path))
    except Exception:
        return "", None
    try:
        pages = list(doc)[:page_limit]
        chunks: list[str] = []
        for page in pages:
            try:
                chunks.append(page.get_text() or "")
            except Exception:
                continue
        return "\n".join(chunks), len(doc)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def select_subset(
    phase61_subset: list[dict[str, Any]],
    *,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Pick deterministic subset of files with table_geometry_header_inference."""
    candidates = [
        item for item in phase61_subset
        if str(item.get("recommended_strategy")) == "table_geometry_header_inference"
    ]
    candidates.sort(key=lambda it: str(it.get("safe_file_id") or ""))
    if not candidates:
        return []
    if len(candidates) <= subset_size:
        return candidates
    rng = random.Random(seed)
    chosen = rng.sample(candidates, subset_size)
    chosen.sort(key=lambda it: str(it.get("safe_file_id") or ""))
    return chosen


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def diagnose_file(item: dict[str, Any], local_path: Path | None) -> dict[str, Any]:
    safe_id = str(item.get("safe_file_id") or "")
    file_type = str(item.get("file_type") or "unknown")
    extension = str(item.get("extension") or "").lower()

    text = ""
    page_count: int | None = None
    structure_error: str | None = None
    if local_path is None:
        structure_error = "local_path_not_resolved"
    elif file_type == "pdf":
        text, page_count = extract_pdf_text_local(local_path)
        if not text:
            structure_error = "pdf_text_unreadable"
    else:
        structure_error = f"non_pdf_skipped:{file_type}"

    signals = measure_geometry_signals(text)
    public_signals = {k: v for k, v in signals.items() if not k.startswith("_")}

    return {
        "safe_file_id": safe_id,
        "file_type": file_type,
        "extension": extension,
        "page_count_bucket": _bucket_pages(page_count),
        "structure_error": structure_error,
        **public_signals,
    }


def aggregate_distribution(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for e in entries:
        counter[str(e.get(key) or "unknown")] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def aggregate_boolean(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for e in entries:
        counter["true" if bool(e.get(key)) else "false"] += 1
    return dict(counter)


def recommended_phase63(
    recoverable_rate: float,
    dominant_confidence: str,
) -> dict[str, Any]:
    if recoverable_rate >= 0.60 and dominant_confidence in ("high", "medium"):
        return {
            "title": "phase63_isolated_geometry_extraction_sandbox",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Majority of files show recoverable geometry with high/medium confidence. "
                "Phase 63 should prototype an ISOLATED geometry-only extraction sandbox "
                "measuring whether inferred column boundaries help — no production extractor change."
            ),
            "recoverable_rate": _bucket_ratio(recoverable_rate),
            "dominant_confidence": dominant_confidence,
        }
    return {
        "title": "no_code_change_manual_review_boundary",
        "production_extractor_should_change_yet": False,
        "reason": (
            "Insufficient recoverable-geometry signal to justify a prototype. "
            "Keep manual review boundary; do not change production extractor."
        ),
        "recoverable_rate": _bucket_ratio(recoverable_rate),
        "dominant_confidence": dominant_confidence,
    }


def run_diagnostic(
    *,
    phase61_report_path: Path | None = None,
    private_mapping_path: Path | None = None,
    input_dir: Path | None = None,
    report_dir: Path | None = None,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    target_report_dir = report_dir or REPORT_DIR
    target_report_dir.mkdir(parents=True, exist_ok=True)

    phase61 = load_phase61(phase61_report_path)
    private_mapping = load_private_mapping(private_mapping_path)
    input_dir_resolved = input_dir or DEFAULT_INPUT_DIR

    phase61_subset = phase61.get("subset") or []
    geometry_population_count = sum(
        1 for item in phase61_subset
        if str(item.get("recommended_strategy")) == "table_geometry_header_inference"
    )
    selected = select_subset(phase61_subset, subset_size=subset_size, seed=seed)
    entries: list[dict[str, Any]] = []
    for item in selected:
        safe_id = str(item.get("safe_file_id") or "")
        local_path = resolve_local_path(safe_id, private_mapping, input_dir_resolved)
        entries.append(diagnose_file(item, local_path))

    boolean_keys = (
        "column_alignment_detected",
        "deep_table_block_present",
        "multi_block_structure",
        "recoverable_table_candidate",
    )
    boolean_distributions = {k: aggregate_boolean(entries, k) for k in boolean_keys}
    strength_distribution = aggregate_distribution(entries, "geometry_signal_strength")
    confidence_distribution = aggregate_distribution(entries, "recovery_confidence_band")
    action_distribution = aggregate_distribution(entries, "safe_next_action")

    recoverable_count = sum(1 for e in entries if bool(e.get("recoverable_table_candidate")))
    recoverable_rate = recoverable_count / len(entries) if entries else 0.0
    dominant_confidence = next(iter(confidence_distribution), "low")

    recommendation = recommended_phase63(recoverable_rate, dominant_confidence)
    privacy_passed = _privacy_self_check(entries)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 62 Table Geometry Header Inference Prototype Diagnostic",
        "source_phase61_report": str(PHASE61_REPORT.relative_to(ROOT)),
        "subset_seed": seed,
        "subset_size_requested": subset_size,
        "subset_size_actual": len(entries),
        "geometry_inference_population": geometry_population_count,
        "recoverable_table_candidate_count": recoverable_count,
        "recoverable_table_candidate_rate_bucket": _bucket_ratio(recoverable_rate),
        "boolean_distributions": boolean_distributions,
        "geometry_signal_strength_distribution": strength_distribution,
        "recovery_confidence_band_distribution": confidence_distribution,
        "safe_next_action_distribution": action_distribution,
        "dominant_geometry_signal_strength": next(iter(strength_distribution), None),
        "dominant_recovery_confidence": dominant_confidence,
        "recommended_phase63_target": recommendation,
        "production_extractor_should_change_yet": recommendation.get(
            "production_extractor_should_change_yet", False
        ),
        "subset": entries,
        "privacy_safety": {
            "uses_safe_ids_only": privacy_passed,
            "raw_filenames_present_in_output": False,
            "raw_paths_present_in_output": False,
            "extracted_text_present_in_output": False,
            "ocr_text_present_in_output": False,
            "table_rows_or_inferred_labels_in_output": False,
            "phi_present_in_output": False,
            "external_api_used": False,
        },
        "conclusion": (
            "no_geometry_inference_files" if not entries
            else "table_geometry_prototype_assessed"
        ),
    }

    (target_report_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_report_dir / MD_REPORT.name).write_text(render_markdown(report), encoding="utf-8")
    return report


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
    "snippet",
    "matched_label",
    "matched_token",
    "row_text",
    "header_text",
    "inferred_label",
    "inferred_header",
)

_RAW_FILENAME_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_-]{2,}\.(pdf|docx|rtf|jpg|jpeg|png|tif|tiff|bmp|webp|mp3|ogg|msg|xml|txt)\b",
    re.IGNORECASE,
)


def _privacy_self_check(entries: list[dict[str, Any]]) -> bool:
    serialized = json.dumps(entries, default=str)
    for forbidden in _FORBIDDEN_KEY_NAMES:
        if f'"{forbidden}"' in serialized:
            return False
    for match in _RAW_FILENAME_RE.finditer(serialized):
        stem = match.group(0).split(".")[0].lower()
        if stem.startswith("phase") or stem.startswith("corpus_"):
            continue
        return False
    for entry in entries:
        for key, value in entry.items():
            if isinstance(value, str) and len(value) > 120:
                return False
    return True


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    rec = report.get("recommended_phase63_target") or {}
    bool_dists = report.get("boolean_distributions") or {}
    lines = [
        "# Phase 62 Table Geometry Header Inference Prototype Diagnostic",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Source Phase 61 report: `{report['source_phase61_report']}`",
        f"- Subset seed: `{report['subset_seed']}`",
        f"- Subset size requested: `{report['subset_size_requested']}`",
        f"- Subset size actual: `{report['subset_size_actual']}`",
        f"- geometry_inference_population: `{report['geometry_inference_population']}`",
        f"- recoverable_table_candidate_count: `{report['recoverable_table_candidate_count']}`",
        f"- recoverable_table_candidate_rate_bucket: `{report['recoverable_table_candidate_rate_bucket']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Geometry Signal Strength Distribution",
        "",
        "| Strength | Count |",
        "| --- | ---: |",
    ]
    for strength, count in report["geometry_signal_strength_distribution"].items():
        lines.append(f"| `{strength}` | {count} |")

    lines += [
        "",
        "## Recovery Confidence Band Distribution",
        "",
        "| Band | Count |",
        "| --- | ---: |",
    ]
    for band, count in report["recovery_confidence_band_distribution"].items():
        lines.append(f"| `{band}` | {count} |")

    lines += [
        "",
        "## Safe Next Action Distribution",
        "",
        "| Action | Count |",
        "| --- | ---: |",
    ]
    for action, count in report["safe_next_action_distribution"].items():
        lines.append(f"| `{action}` | {count} |")

    lines += [
        "",
        "## Boolean Indicator Distributions (true/false counts)",
        "",
        "| Indicator | True | False |",
        "| --- | ---: | ---: |",
    ]
    for indicator, dist in bool_dists.items():
        lines.append(f"| `{indicator}` | {dist.get('true', 0)} | {dist.get('false', 0)} |")

    lines += [
        "",
        "## Recommended Phase 63 Target",
        "",
        f"- Title: `{rec.get('title')}`",
        f"- production_extractor_should_change_yet: `{rec.get('production_extractor_should_change_yet')}`",
        f"- recoverable_rate: `{rec.get('recoverable_rate')}`",
        f"- dominant_confidence: `{rec.get('dominant_confidence')}`",
        "",
        f"_{rec.get('reason')}_",
        "",
        "## Forensic Subset Entries",
        "",
        "| safe_file_id | pages | text_len | num_line_ratio | col_count | aligned_pairs | blocks | max_depth | repeated_cols | col_align | deep_block | multi_block | signal | recoverable | confidence | action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["subset"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{entry['safe_file_id']}`",
                str(entry.get("page_count_bucket") or ""),
                str(entry.get("text_length_bucket") or ""),
                str(entry.get("numeric_heavy_line_ratio_bucket") or ""),
                str(entry.get("distinct_column_count_bucket") or ""),
                str(entry.get("aligned_pair_count_bucket") or ""),
                str(entry.get("table_block_count_bucket") or ""),
                str(entry.get("max_block_depth_bucket") or ""),
                str(entry.get("repeated_numeric_column_line_count_bucket") or ""),
                "yes" if entry.get("column_alignment_detected") else "no",
                "yes" if entry.get("deep_table_block_present") else "no",
                "yes" if entry.get("multi_block_structure") else "no",
                str(entry.get("geometry_signal_strength") or ""),
                "yes" if entry.get("recoverable_table_candidate") else "no",
                str(entry.get("recovery_confidence_band") or ""),
                str(entry.get("safe_next_action") or ""),
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
        f"- table_rows_or_inferred_labels_in_output: `{report['privacy_safety']['table_rows_or_inferred_labels_in_output']}`",
        f"- phi_present_in_output: `{report['privacy_safety']['phi_present_in_output']}`",
        f"- external_api_used: `{report['privacy_safety']['external_api_used']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_diagnostic()
    rec = report.get("recommended_phase63_target") or {}
    print("MedAI Phase 62 table geometry header inference prototype diagnostic complete.")
    print(f"subset_size_actual: {report['subset_size_actual']}")
    print(f"geometry_inference_population: {report['geometry_inference_population']}")
    print(f"recoverable_table_candidate_count: {report['recoverable_table_candidate_count']}")
    print(f"recoverable_table_candidate_rate_bucket: {report['recoverable_table_candidate_rate_bucket']}")
    print(f"geometry_signal_strength_distribution: {report['geometry_signal_strength_distribution']}")
    print(f"recovery_confidence_band_distribution: {report['recovery_confidence_band_distribution']}")
    print(f"dominant_geometry_signal_strength: {report.get('dominant_geometry_signal_strength')}")
    print(f"recommended_phase63_target: {rec.get('title')}")
    print(f"production_extractor_should_change_yet: {rec.get('production_extractor_should_change_yet')}")
    print(f"privacy_safety.uses_safe_ids_only: {report['privacy_safety']['uses_safe_ids_only']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
