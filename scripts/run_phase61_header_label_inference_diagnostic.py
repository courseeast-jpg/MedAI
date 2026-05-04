"""Phase 61 — Header/Label Inference Diagnostic for Numeric Table Gaps.

For files Phase 60 marked likely_gap_type=numeric_table_without_labels,
evaluate whether headers/labels could plausibly be inferred (by neighbor
lines, table geometry, generic lab units, or a multilingual label map).
Emits per-file boolean indicators and a recommended Phase 62 strategy.

Touches no extraction logic, OCR routing, classifier code, threshold,
or safety gate. Production extractors are not modified, and the default
recommendation is production_extractor_should_change_yet=False.

Privacy:
  - Public output uses safe_file_id only.
  - No raw text, OCR text, table rows, document-derived labels, raw
    filenames, or raw paths are emitted.
  - Built-in vocab categories appear as category names, not strings
    derived from documents.
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
PHASE60_REPORT = (
    ROOT / "reports" / "phase60_text_extraction_gap_diagnostic"
    / "phase60_text_extraction_gap_diagnostic_report.json"
)
DEFAULT_INPUT_DIR = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase61_header_label_inference_diagnostic"
JSON_REPORT = REPORT_DIR / "phase61_header_label_inference_diagnostic_report.json"
MD_REPORT = REPORT_DIR / "phase61_header_label_inference_diagnostic_report.md"

DEFAULT_SUBSET_SIZE = 40
DEFAULT_RANDOM_SEED = 20260503


# ---------------------------------------------------------------------------
# Internal vocab categories — predetermined, not document-derived.
# ---------------------------------------------------------------------------

_GENERIC_ENGLISH_LAB_LABELS = (
    "glucose", "creatinine", "cholesterol", "hemoglobin", "hematocrit",
    "platelet", "platelets", "sodium", "potassium", "calcium", "albumin",
    "bilirubin", "wbc", "rbc", "ldl", "hdl", "triglycerides", "tsh",
    "hba1c", "alt", "ast", "urea", "bun", "chloride",
)

_GENERIC_CYRILLIC_LAB_INDICATORS = (
    # Unicode Cyrillic substrings only.
    "глюкоз", "креатинин", "холестерин", "гемоглобин", "гематокрит",
    "тромбоцит", "натрий", "калий", "кальций", "альбумин", "билирубин",
    "лейкоцит", "эритроцит", "лпвп", "лпнп", "тригли", "анализ",
    "норма", "результат", "показатель", "мг", "ммоль", "мкмоль", "ед",
)

_MEDICATION_DOSE_INDICATORS = (
    "tablet", "capsule", "take ", "daily", "sig:", "rx", " mg ", " mcg ",
    "tab", "cap", "po ", "qd", "bid", "tid", "qid", "prn",
)

_IMAGING_SECTION_INDICATORS = (
    "impression:", "findings:", "comparison:", "technique:", "indication:",
)

_UNIT_PATTERN_RE = re.compile(
    r"\b(?:mg\s*/\s*d[lL]|mmol\s*/\s*[lL]|g\s*/\s*d[lL]|x10E[36]\s*/\s*uL|"
    r"IU\s*/\s*L|U\s*/\s*L|ng\s*/\s*mL|/hpf|CFU\s*/\s*mL|ммоль\s*/\s*л|"
    r"мг\s*/\s*дл|г\s*/\s*л)\b",
    re.IGNORECASE,
)
_REFERENCE_RANGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b"
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"\S+")
_TABLE_LIKE_LINE_RE = re.compile(r"(?:\s{2,}\S+){2,}")
_REPEATED_NUMERIC_COLUMN_RE = re.compile(
    # Whitespace strictly horizontal (spaces/tabs) so the pattern doesn't
    # span newlines under re.MULTILINE.
    r"^[ \t]*\d+(?:\.\d+)?(?:[ \t]+\d+(?:\.\d+)?){2,}[ \t]*$",
    re.MULTILINE,
)
_SHORT_ALPHA_LINE_RE = re.compile(r"^\s*[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё ,.\-]{1,40}\s*$")
_CYRILLIC_CHAR_RE = re.compile(r"[Ѐ-ӿ]")


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
# Local diagnostic measurement
# ---------------------------------------------------------------------------


def _count_hits(haystack_lower: str, needles: tuple[str, ...]) -> int:
    return sum(haystack_lower.count(needle) for needle in needles)


def measure_table_signals(text: str) -> dict[str, Any]:
    """Compute table/header/label inference signals locally. Never emits text."""
    if not text:
        return _empty_signals()

    lower = text.lower()
    lines = text.splitlines()
    total_lines = max(1, len(lines))
    tokens = _TOKEN_RE.findall(text)
    numeric_tokens = _NUMERIC_TOKEN_RE.findall(text)
    numeric_density = (len(numeric_tokens) / len(tokens)) if tokens else 0.0

    table_like_lines = sum(1 for line in lines if _TABLE_LIKE_LINE_RE.search(line))
    repeated_numeric_columns = len(_REPEATED_NUMERIC_COLUMN_RE.findall(text))
    short_lines = sum(1 for line in lines if 0 < len(line.strip()) <= 30)
    short_line_ratio = short_lines / total_lines

    unit_count = len(_UNIT_PATTERN_RE.findall(text))
    range_count = len(_REFERENCE_RANGE_RE.findall(text))

    eng_label_hits = _count_hits(lower, _GENERIC_ENGLISH_LAB_LABELS)
    cyr_label_hits = _count_hits(text, _GENERIC_CYRILLIC_LAB_INDICATORS)
    medication_hits = _count_hits(lower, _MEDICATION_DOSE_INDICATORS)
    imaging_hits = _count_hits(lower, _IMAGING_SECTION_INDICATORS)
    cyrillic_chars = len(_CYRILLIC_CHAR_RE.findall(text))
    visible = sum(1 for c in text if not c.isspace())
    cyrillic_ratio = (cyrillic_chars / visible) if visible else 0.0

    # Neighbor-line pattern: a short alpha-only line followed within 1-2
    # lines by a numeric-heavy line.
    neighbor_pairs = 0
    for i in range(len(lines) - 1):
        cur = lines[i].strip()
        if not cur or not _SHORT_ALPHA_LINE_RE.match(cur):
            continue
        for j in range(i + 1, min(i + 3, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            nxt_tokens = _TOKEN_RE.findall(nxt)
            if not nxt_tokens:
                continue
            nxt_numeric = sum(1 for t in nxt_tokens if _NUMERIC_TOKEN_RE.fullmatch(t))
            if nxt_numeric / len(nxt_tokens) >= 0.5:
                neighbor_pairs += 1
                break

    # Geometry pattern: at least 2 lines whose numeric tokens appear at
    # similar character offsets within a tolerance band.
    column_offset_hits = 0
    column_offsets: list[set[int]] = []
    for line in lines:
        offsets: set[int] = set()
        for m in _NUMERIC_TOKEN_RE.finditer(line):
            offsets.add(m.start() // 4)  # 4-char tolerance bucket
        if len(offsets) >= 2:
            column_offsets.append(offsets)
    if len(column_offsets) >= 2:
        # Count pairs of lines sharing >=2 column buckets.
        for i in range(len(column_offsets)):
            for j in range(i + 1, len(column_offsets)):
                if len(column_offsets[i] & column_offsets[j]) >= 2:
                    column_offset_hits += 1
                    if column_offset_hits >= 5:
                        break
            if column_offset_hits >= 5:
                break

    # Header indicators
    likely_table_present = (
        table_like_lines >= 3 or repeated_numeric_columns >= 2 or column_offset_hits >= 2
    )
    likely_header_missing = likely_table_present and eng_label_hits == 0 and cyr_label_hits == 0
    likely_header_fragmented = (
        likely_table_present
        and short_line_ratio >= 0.30
        and (eng_label_hits + cyr_label_hits) >= 1
        and (eng_label_hits + cyr_label_hits) < 3
    )
    likely_non_english_labels = cyrillic_ratio >= 0.10 or cyr_label_hits >= 1
    likely_cyrillic_or_mixed_script_labels = (
        cyrillic_ratio >= 0.10 and cyr_label_hits >= 1
    )
    likely_units_without_analyte_names = (
        unit_count >= 2 and eng_label_hits == 0 and cyr_label_hits == 0
    )
    likely_analyte_names_without_units = (
        (eng_label_hits + cyr_label_hits) >= 2 and unit_count == 0
    )

    # Inferability flags
    inferable_by_neighbor_lines = (
        neighbor_pairs >= 2 and likely_table_present
    )
    inferable_by_table_geometry = (
        column_offset_hits >= 3 and likely_table_present
    )
    inferable_by_generic_lab_units = (
        unit_count >= 1 and likely_table_present
    )
    inferable_by_multilingual_label_map = (
        likely_cyrillic_or_mixed_script_labels and likely_table_present
    )

    return {
        "text_length_bucket": _bucket_text_length(len(text.strip())),
        "numeric_density_bucket": _bucket_ratio(numeric_density),
        "table_like_line_count_bucket": _bucket_count(table_like_lines),
        "short_line_ratio_bucket": _bucket_ratio(short_line_ratio),
        "unit_pattern_count_bucket": _bucket_count(unit_count),
        "reference_range_pattern_count_bucket": _bucket_count(range_count),
        "repeated_numeric_column_pattern": repeated_numeric_columns >= 2,
        "likely_table_present": likely_table_present,
        "likely_header_missing": likely_header_missing,
        "likely_header_fragmented": likely_header_fragmented,
        "likely_non_english_labels": likely_non_english_labels,
        "likely_cyrillic_or_mixed_script_labels": likely_cyrillic_or_mixed_script_labels,
        "likely_units_without_analyte_names": likely_units_without_analyte_names,
        "likely_analyte_names_without_units": likely_analyte_names_without_units,
        "inferable_by_neighbor_lines": inferable_by_neighbor_lines,
        "inferable_by_table_geometry": inferable_by_table_geometry,
        "inferable_by_generic_lab_units": inferable_by_generic_lab_units,
        "inferable_by_multilingual_label_map": inferable_by_multilingual_label_map,
        "_raw_counts": {
            "table_like_lines": table_like_lines,
            "repeated_numeric_columns": repeated_numeric_columns,
            "short_line_ratio": short_line_ratio,
            "unit_count": unit_count,
            "range_count": range_count,
            "eng_label_hits": eng_label_hits,
            "cyr_label_hits": cyr_label_hits,
            "medication_hits": medication_hits,
            "imaging_hits": imaging_hits,
            "cyrillic_ratio": cyrillic_ratio,
            "neighbor_pairs": neighbor_pairs,
            "column_offset_hits": column_offset_hits,
            "numeric_density": numeric_density,
        },
    }


def _empty_signals() -> dict[str, Any]:
    return {
        "text_length_bucket": _bucket_text_length(0),
        "numeric_density_bucket": _bucket_ratio(0.0),
        "table_like_line_count_bucket": _bucket_count(0),
        "short_line_ratio_bucket": _bucket_ratio(0.0),
        "unit_pattern_count_bucket": _bucket_count(0),
        "reference_range_pattern_count_bucket": _bucket_count(0),
        "repeated_numeric_column_pattern": False,
        "likely_table_present": False,
        "likely_header_missing": False,
        "likely_header_fragmented": False,
        "likely_non_english_labels": False,
        "likely_cyrillic_or_mixed_script_labels": False,
        "likely_units_without_analyte_names": False,
        "likely_analyte_names_without_units": False,
        "inferable_by_neighbor_lines": False,
        "inferable_by_table_geometry": False,
        "inferable_by_generic_lab_units": False,
        "inferable_by_multilingual_label_map": False,
        "_raw_counts": {},
    }


def assign_strategy(signals: dict[str, Any]) -> tuple[str, str]:
    """Return (recommended_strategy, confidence_band)."""
    if not signals.get("likely_table_present"):
        return ("manual_review_boundary", "low")
    counts = signals.get("_raw_counts") or {}
    inferable_options: list[tuple[str, int]] = []
    if signals.get("inferable_by_multilingual_label_map"):
        inferable_options.append((
            "multilingual_label_map_diagnostic",
            int(counts.get("cyr_label_hits", 0)) + int(counts.get("cyrillic_ratio", 0) * 100),
        ))
    if signals.get("inferable_by_neighbor_lines"):
        inferable_options.append((
            "neighbor_line_header_inference",
            int(counts.get("neighbor_pairs", 0)),
        ))
    if signals.get("inferable_by_table_geometry"):
        inferable_options.append((
            "table_geometry_header_inference",
            int(counts.get("column_offset_hits", 0)),
        ))
    if signals.get("inferable_by_generic_lab_units"):
        inferable_options.append((
            "generic_lab_unit_inference",
            int(counts.get("unit_count", 0)),
        ))
    if not inferable_options:
        return ("no_inference_possible", "medium" if signals.get("likely_table_present") else "low")
    inferable_options.sort(key=lambda kv: -kv[1])
    strategy, top_score = inferable_options[0]
    if top_score >= 5:
        confidence = "high"
    elif top_score >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    return (strategy, confidence)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_phase60(path: Path | None = None) -> dict[str, Any]:
    p = path or PHASE60_REPORT
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
    phase60_subset: list[dict[str, Any]],
    *,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Pick deterministic subset of files marked numeric_table_without_labels."""
    candidates = [
        item for item in phase60_subset
        if str(item.get("likely_gap_type")) == "numeric_table_without_labels"
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

    signals = measure_table_signals(text)
    strategy, confidence = assign_strategy(signals)

    public_signals = {k: v for k, v in signals.items() if not k.startswith("_")}
    return {
        "safe_file_id": safe_id,
        "file_type": file_type,
        "extension": extension,
        "page_count_bucket": _bucket_pages(page_count),
        "structure_error": structure_error,
        **public_signals,
        "recommended_strategy": strategy,
        "confidence_band": confidence,
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


def recommended_phase62(strategy_distribution: dict[str, int]) -> dict[str, Any]:
    if not strategy_distribution:
        return {
            "title": "no_inference_diagnostic_required",
            "production_extractor_should_change_yet": False,
            "reason": "No subset entries surfaced.",
        }
    dominant = next(iter(strategy_distribution))
    titles = {
        "neighbor_line_header_inference": (
            "phase62_neighbor_line_header_inference_prototype",
            "Most files have short alpha-only lines preceding numeric blocks. "
            "Phase 62 should evaluate a NARROW neighbor-line header inference "
            "diagnostic prototype only — no production extractor change.",
        ),
        "table_geometry_header_inference": (
            "phase62_table_geometry_header_inference_prototype",
            "Most files have aligned numeric columns without recognised "
            "headers. Phase 62 should evaluate a NARROW geometry diagnostic "
            "prototype only — no production extractor change.",
        ),
        "multilingual_label_map_diagnostic": (
            "phase62_multilingual_lab_label_map_diagnostic",
            "Most files have Cyrillic / non-English labels. Phase 62 should "
            "build a label-map diagnostic for Russian lab terminology — no "
            "production extractor change.",
        ),
        "generic_lab_unit_inference": (
            "phase62_generic_lab_unit_inference_diagnostic",
            "Most files have generic lab units but missing/fragmented "
            "labels. Phase 62 should evaluate unit-anchored inference as a "
            "diagnostic — no production extractor change.",
        ),
        "manual_review_boundary": (
            "no_code_change_manual_review_boundary",
            "No reliable inference path. Keep manual review boundary; do not "
            "change production extractor.",
        ),
        "no_inference_possible": (
            "no_code_change_manual_review_boundary",
            "No reliable inference signal across the subset. Keep manual "
            "review boundary; do not change production extractor.",
        ),
    }
    title, reason = titles.get(
        dominant,
        ("no_code_change_manual_review_boundary", "Default safe recommendation."),
    )
    return {
        "title": title,
        "production_extractor_should_change_yet": False,
        "reason": reason,
        "dominant_strategy": dominant,
    }


def run_diagnostic(
    *,
    phase60_report_path: Path | None = None,
    private_mapping_path: Path | None = None,
    input_dir: Path | None = None,
    report_dir: Path | None = None,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    target_report_dir = report_dir or REPORT_DIR
    target_report_dir.mkdir(parents=True, exist_ok=True)

    phase60 = load_phase60(phase60_report_path)
    private_mapping = load_private_mapping(private_mapping_path)
    input_dir_resolved = input_dir or DEFAULT_INPUT_DIR

    phase60_subset = phase60.get("subset") or []
    population_count = sum(
        1 for item in phase60_subset
        if str(item.get("likely_gap_type")) == "numeric_table_without_labels"
    )
    selected = select_subset(phase60_subset, subset_size=subset_size, seed=seed)
    entries: list[dict[str, Any]] = []
    for item in selected:
        safe_id = str(item.get("safe_file_id") or "")
        local_path = resolve_local_path(safe_id, private_mapping, input_dir_resolved)
        entries.append(diagnose_file(item, local_path))

    boolean_keys = (
        "likely_table_present",
        "likely_header_missing",
        "likely_header_fragmented",
        "likely_non_english_labels",
        "likely_cyrillic_or_mixed_script_labels",
        "likely_units_without_analyte_names",
        "likely_analyte_names_without_units",
        "inferable_by_neighbor_lines",
        "inferable_by_table_geometry",
        "inferable_by_generic_lab_units",
        "inferable_by_multilingual_label_map",
        "repeated_numeric_column_pattern",
    )
    boolean_distributions = {k: aggregate_boolean(entries, k) for k in boolean_keys}
    strategy_distribution = aggregate_distribution(entries, "recommended_strategy")
    confidence_distribution = aggregate_distribution(entries, "confidence_band")
    recommendation = recommended_phase62(strategy_distribution)
    privacy_passed = _privacy_self_check(entries)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 61 Header/Label Inference Diagnostic for Numeric Table Gaps",
        "source_phase60_report": str(PHASE60_REPORT.relative_to(ROOT)),
        "subset_seed": seed,
        "subset_size_requested": subset_size,
        "subset_size_actual": len(entries),
        "numeric_table_without_labels_population": population_count,
        "boolean_distributions": boolean_distributions,
        "recommended_strategy_distribution": strategy_distribution,
        "confidence_band_distribution": confidence_distribution,
        "dominant_recommended_strategy": next(iter(strategy_distribution), None),
        "recommended_phase62_target": recommendation,
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
            "table_rows_or_labels_in_output": False,
            "phi_present_in_output": False,
            "external_api_used": False,
        },
        "conclusion": (
            "no_numeric_table_gap_files" if not entries else "header_label_inference_diagnosed"
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
    rec = report.get("recommended_phase62_target") or {}
    bool_dists = report.get("boolean_distributions") or {}
    lines = [
        "# Phase 61 Header/Label Inference Diagnostic for Numeric Table Gaps",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Source Phase 60 report: `{report['source_phase60_report']}`",
        f"- Subset seed: `{report['subset_seed']}`",
        f"- Subset size requested: `{report['subset_size_requested']}`",
        f"- Subset size actual: `{report['subset_size_actual']}`",
        f"- numeric_table_without_labels population: `{report['numeric_table_without_labels_population']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Recommended Strategy Distribution",
        "",
        "| Strategy | Count |",
        "| --- | ---: |",
    ]
    for strategy, count in report["recommended_strategy_distribution"].items():
        lines.append(f"| `{strategy}` | {count} |")
    lines += [
        "",
        f"- Dominant recommended strategy: `{report.get('dominant_recommended_strategy')}`",
        "",
        "## Confidence Band Distribution",
        "",
        "| Band | Count |",
        "| --- | ---: |",
    ]
    for band, count in report["confidence_band_distribution"].items():
        lines.append(f"| `{band}` | {count} |")

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
        "## Recommended Phase 62 Target",
        "",
        f"- Title: `{rec.get('title')}`",
        f"- production_extractor_should_change_yet: `{rec.get('production_extractor_should_change_yet')}`",
        f"- dominant_strategy: `{rec.get('dominant_strategy')}`",
        "",
        f"_{rec.get('reason')}_",
        "",
        "## Forensic Subset Entries",
        "",
        "| safe_file_id | pages | text_len | numeric_density | unit_patterns | strategy | confidence | table | header_missing | non_english | inferable_neighbor | inferable_geom | inferable_units | inferable_multilingual |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["subset"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{entry['safe_file_id']}`",
                str(entry.get("page_count_bucket") or ""),
                str(entry.get("text_length_bucket") or ""),
                str(entry.get("numeric_density_bucket") or ""),
                str(entry.get("unit_pattern_count_bucket") or ""),
                f"`{entry.get('recommended_strategy')}`",
                str(entry.get("confidence_band") or ""),
                "yes" if entry.get("likely_table_present") else "no",
                "yes" if entry.get("likely_header_missing") else "no",
                "yes" if entry.get("likely_non_english_labels") else "no",
                "yes" if entry.get("inferable_by_neighbor_lines") else "no",
                "yes" if entry.get("inferable_by_table_geometry") else "no",
                "yes" if entry.get("inferable_by_generic_lab_units") else "no",
                "yes" if entry.get("inferable_by_multilingual_label_map") else "no",
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
        f"- table_rows_or_labels_in_output: `{report['privacy_safety']['table_rows_or_labels_in_output']}`",
        f"- phi_present_in_output: `{report['privacy_safety']['phi_present_in_output']}`",
        f"- external_api_used: `{report['privacy_safety']['external_api_used']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_diagnostic()
    rec = report.get("recommended_phase62_target") or {}
    print("MedAI Phase 61 header/label inference diagnostic complete.")
    print(f"subset_size_actual: {report['subset_size_actual']}")
    print(f"numeric_table_without_labels_population: {report['numeric_table_without_labels_population']}")
    print(f"recommended_strategy_distribution: {report['recommended_strategy_distribution']}")
    print(f"confidence_band_distribution: {report['confidence_band_distribution']}")
    print(f"dominant_recommended_strategy: {report.get('dominant_recommended_strategy')}")
    print(f"recommended_phase62_target: {rec.get('title')}")
    print(f"production_extractor_should_change_yet: {rec.get('production_extractor_should_change_yet')}")
    print(f"privacy_safety.uses_safe_ids_only: {report['privacy_safety']['uses_safe_ids_only']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
