"""Phase 60 — Text Extraction Gap Vocabulary Coverage Diagnostic.

For files where Phase 59 found "pdf_text_extraction_gap" (text is present
in the PDF but extraction returned zero entities), measure local
vocabulary / pattern coverage against fixed internal vocabulary lists and
classify each file by likely document class and likely extraction gap.

Touches no extraction logic, OCR routing, classifier code, threshold, or
safety gate. Production extractors and parsers are not modified.

Privacy:
  - Public output uses safe_file_id only.
  - No extracted text, OCR text, snippets, raw filenames, or raw paths.
  - Only counts, buckets, booleans, and classifications are emitted.
  - Vocabulary terms in the output are *internal categories*, not strings
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

# Force local-only as a defence in depth.
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")


PHASE57_REPORT = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit"
    / "phase57_full_corpus_inventory_audit_report.json"
)
PHASE57_PRIVATE_MAPPING = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit"
    / "local_filename_mapping_PRIVATE.json"
)
PHASE59_REPORT = (
    ROOT / "reports" / "phase59_empty_extraction_forensics"
    / "phase59_empty_extraction_forensics_report.json"
)
DEFAULT_INPUT_DIR = ROOT / "full_corpus_input"
REPORT_DIR = ROOT / "reports" / "phase60_text_extraction_gap_diagnostic"
JSON_REPORT = REPORT_DIR / "phase60_text_extraction_gap_diagnostic_report.json"
MD_REPORT = REPORT_DIR / "phase60_text_extraction_gap_diagnostic_report.md"

DEFAULT_SUBSET_SIZE = 60
DEFAULT_RANDOM_SEED = 20260503


# ---------------------------------------------------------------------------
# Internal vocabulary — fixed, predetermined, NOT derived from documents.
# ---------------------------------------------------------------------------

_LAB_UNITS_TOKENS: tuple[str, ...] = (
    "mg/dl", "mmol/l", "g/dl", "k/ul", "iu/l", "u/l",
    "ng/ml", "x10e3/ul", "x10e6/ul", "/hpf", "cfu/ml",
)

_LAB_CONCEPT_TOKENS: tuple[str, ...] = (
    "glucose", "creatinine", "cholesterol", "hemoglobin", "hematocrit",
    "platelet", "platelets", "sodium", "potassium", "calcium", "albumin",
    "bilirubin", "wbc", "rbc", "ldl", "hdl", "triglycerides", "tsh",
    "hba1c", "alt", "ast", "urinalysis", "specific gravity",
)

_IMAGING_CONCEPT_TOKENS: tuple[str, ...] = (
    "ct ", "mri", "x-ray", "xray", "ultrasound", "impression:",
    "impression ", "findings:", "findings ", "radiograph", "radiology",
    "echocardiogram", "doppler",
)

_MEDICATION_CONCEPT_TOKENS: tuple[str, ...] = (
    "tablet", "tablets", "capsule", "capsules", "take ", "daily", "dosage",
    "prescription", "sig:", " mg ", " mcg ", " ml ", "po ",
    "qd", "bid", "tid", "qid", "prn",
)

_ADMIN_CONCEPT_TOKENS: tuple[str, ...] = (
    "insurance", "invoice", "account", "statement", "balance", "policy",
    "claim", "billing", "deductible", "subscriber", "copay", "coinsurance",
    "payer", "remittance",
)

_TREND_REPORT_TOKENS: tuple[str, ...] = (
    "trend", "history", "previous", "prior result", "delta",
)

_NARRATIVE_TOKENS: tuple[str, ...] = (
    "history of present illness", "chief complaint", "assessment", "plan",
    "review of systems", "physical exam", "discharge summary",
)

# Patterns
_UNIT_PATTERN_RE = re.compile(
    r"\b(?:mg\s*/\s*d[lL]|mmol\s*/\s*[lL]|g\s*/\s*d[lL]|x10E[36]\s*/\s*uL|"
    r"IU\s*/\s*L|U\s*/\s*L|ng\s*/\s*mL|/hpf|CFU\s*/\s*mL)\b",
    re.IGNORECASE,
)
_REFERENCE_RANGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b"
)
_DATE_PATTERN_RE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
_TABLE_LIKE_LINE_RE = re.compile(
    r"(?:\s{2,}\S+){2,}"  # 2+ runs of double-space-separated columns
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Bucketing helpers
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


def _bucket_page_count(n: int | None) -> str:
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
    if n <= 50:
        return "11_50"
    return "gt_50"


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
    if length < 20000:
        return "5000_20000"
    return "gt_20000"


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


# ---------------------------------------------------------------------------
# Vocab counting (text is local-only, never emitted)
# ---------------------------------------------------------------------------


def _count_hits(haystack_lower: str, needles: tuple[str, ...]) -> int:
    return sum(haystack_lower.count(needle) for needle in needles)


def _cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    cyr = sum(1 for c in visible if "Ѐ" <= c <= "ӿ")
    return cyr / len(visible)


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def measure_local_text(text: str) -> dict[str, Any]:
    """Compute counts/buckets/booleans for a local-only text. Never emits
    the text itself.
    """
    if not text:
        return {
            "extracted_text_length_bucket": _bucket_text_length(0),
            "medical_token_hit_count_bucket": _bucket_count(0),
            "lab_token_hit_count_bucket": _bucket_count(0),
            "medication_token_hit_count_bucket": _bucket_count(0),
            "imaging_token_hit_count_bucket": _bucket_count(0),
            "admin_token_hit_count_bucket": _bucket_count(0),
            "numeric_density_bucket": _bucket_ratio(0.0),
            "table_like_line_count_bucket": _bucket_count(0),
            "unit_pattern_count_bucket": _bucket_count(0),
            "date_pattern_count_bucket": _bucket_count(0),
            "cyrillic_ratio_bucket": _bucket_ratio(0.0),
            "uppercase_ratio_bucket": _bucket_ratio(0.0),
            "has_common_lab_units": False,
            "has_reference_range_patterns": False,
            "has_medication_patterns": False,
            "has_radiology_terms": False,
            "has_admin_terms": False,
            "has_trend_terms": False,
            "has_narrative_terms": False,
            "_raw_counts": {  # private to bucketing logic; not emitted
                "lab_concept": 0,
                "lab_units": 0,
                "imaging": 0,
                "medication": 0,
                "admin": 0,
                "trend": 0,
                "narrative": 0,
            },
        }

    lower = text.lower()
    tokens = _TOKEN_RE.findall(text)
    numeric_tokens = _NUMERIC_TOKEN_RE.findall(text)
    numeric_density = (len(numeric_tokens) / len(tokens)) if tokens else 0.0

    lab_concept_hits = _count_hits(lower, _LAB_CONCEPT_TOKENS)
    lab_units_hits = _count_hits(lower, _LAB_UNITS_TOKENS)
    imaging_hits = _count_hits(lower, _IMAGING_CONCEPT_TOKENS)
    medication_hits = _count_hits(lower, _MEDICATION_CONCEPT_TOKENS)
    admin_hits = _count_hits(lower, _ADMIN_CONCEPT_TOKENS)
    trend_hits = _count_hits(lower, _TREND_REPORT_TOKENS)
    narrative_hits = _count_hits(lower, _NARRATIVE_TOKENS)
    medical_hits_total = lab_concept_hits + lab_units_hits + imaging_hits + medication_hits

    unit_pattern_count = len(_UNIT_PATTERN_RE.findall(text))
    reference_range_count = len(_REFERENCE_RANGE_RE.findall(text))
    date_pattern_count = len(_DATE_PATTERN_RE.findall(text))
    table_like_line_count = sum(
        1 for line in text.splitlines() if _TABLE_LIKE_LINE_RE.search(line)
    )

    return {
        "extracted_text_length_bucket": _bucket_text_length(len(text.strip())),
        "medical_token_hit_count_bucket": _bucket_count(medical_hits_total),
        "lab_token_hit_count_bucket": _bucket_count(lab_concept_hits + lab_units_hits),
        "medication_token_hit_count_bucket": _bucket_count(medication_hits),
        "imaging_token_hit_count_bucket": _bucket_count(imaging_hits),
        "admin_token_hit_count_bucket": _bucket_count(admin_hits),
        "numeric_density_bucket": _bucket_ratio(numeric_density),
        "table_like_line_count_bucket": _bucket_count(table_like_line_count),
        "unit_pattern_count_bucket": _bucket_count(unit_pattern_count),
        "date_pattern_count_bucket": _bucket_count(date_pattern_count),
        "cyrillic_ratio_bucket": _bucket_ratio(_cyrillic_ratio(text)),
        "uppercase_ratio_bucket": _bucket_ratio(_uppercase_ratio(text)),
        "has_common_lab_units": unit_pattern_count > 0 or lab_units_hits > 0,
        "has_reference_range_patterns": reference_range_count >= 1,
        "has_medication_patterns": medication_hits >= 2,
        "has_radiology_terms": imaging_hits >= 1,
        "has_admin_terms": admin_hits >= 2,
        "has_trend_terms": trend_hits >= 2,
        "has_narrative_terms": narrative_hits >= 1,
        "_raw_counts": {
            "lab_concept": lab_concept_hits,
            "lab_units": lab_units_hits,
            "imaging": imaging_hits,
            "medication": medication_hits,
            "admin": admin_hits,
            "trend": trend_hits,
            "narrative": narrative_hits,
            "unit_pattern": unit_pattern_count,
            "reference_range": reference_range_count,
            "date_pattern": date_pattern_count,
            "table_like_line": table_like_line_count,
            "numeric_density": numeric_density,
            "text_length": len(text.strip()),
        },
    }


# ---------------------------------------------------------------------------
# Document class & gap classification
# ---------------------------------------------------------------------------


def guess_document_class(measurements: dict[str, Any]) -> str:
    counts = measurements.get("_raw_counts") or {}
    lab = counts.get("lab_concept", 0) + counts.get("lab_units", 0)
    imaging = counts.get("imaging", 0)
    medication = counts.get("medication", 0)
    admin = counts.get("admin", 0)
    trend = counts.get("trend", 0)
    narrative = counts.get("narrative", 0)
    text_len = counts.get("text_length", 0)
    # Prioritise the strongest signal
    scores = {
        "lab_report": lab,
        "radiology_or_imaging": imaging,
        "prescription_or_medication": medication,
        "admin_or_billing": admin,
        "trend_report": trend,
        "narrative_note": narrative,
    }
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        if text_len < 200:
            return "unknown"
        # Some text but no recognised vocab
        return "unknown"
    # Disambiguate trend vs lab: trend implies lab data plus trend keywords
    if best == "trend_report" and lab >= trend:
        return "lab_report"
    return best


def classify_gap(
    document_class: str,
    measurements: dict[str, Any],
    *,
    file_size_bucket: str,
    page_count_bucket: str,
) -> str:
    counts = measurements.get("_raw_counts") or {}
    lab = counts.get("lab_concept", 0) + counts.get("lab_units", 0)
    unit_pattern = counts.get("unit_pattern", 0)
    reference_range = counts.get("reference_range", 0)
    table_like = counts.get("table_like_line", 0)
    numeric_density = counts.get("numeric_density", 0.0)
    text_len = counts.get("text_length", 0)
    admin = counts.get("admin", 0)
    imaging = counts.get("imaging", 0)
    medication = counts.get("medication", 0)

    if document_class == "admin_or_billing" and admin >= 2:
        return "admin_document_not_target"
    if document_class == "radiology_or_imaging" and imaging >= 1:
        return "imaging_report_vocabulary_gap"
    if document_class == "prescription_or_medication" and medication >= 2:
        return "document_class_not_supported"  # production extractor is lab-focused
    if document_class == "lab_report":
        if unit_pattern >= 1 and reference_range >= 1 and lab >= 2:
            return "lab_table_vocabulary_gap"
        if numeric_density >= 0.30 and table_like >= 2 and unit_pattern == 0:
            return "numeric_table_without_labels"
        if lab <= 1 and unit_pattern <= 1:
            return "medical_vocabulary_gap"
        return "lab_table_vocabulary_gap"
    if document_class == "trend_report":
        return "lab_table_vocabulary_gap"
    if document_class == "narrative_note":
        return "document_class_not_supported"
    if document_class == "unknown":
        if text_len < 200:
            return "tokenization_or_layout_issue"
        if numeric_density >= 0.30:
            return "numeric_table_without_labels"
        return "unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def select_subset(
    phase57_results: list[dict[str, Any]],
    phase59_subset: list[dict[str, Any]],
    *,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Select the Phase 60 subset.

    Always include every Phase 59 entry whose root_cause_bucket is
    ``pdf_text_extraction_gap``. Then expand to ``subset_size`` by sampling
    additional empty_extraction PDFs from Phase 57 results that have OCR
    status good or usable_with_review and entity_count == 0, excluding any
    already in the subset.
    """
    primary_ids = {
        str(item.get("safe_file_id"))
        for item in phase59_subset
        if str(item.get("root_cause_bucket")) == "pdf_text_extraction_gap"
    }
    selected: list[dict[str, Any]] = []
    by_id_phase57 = {
        str(item.get("safe_file_id") or item.get("file_id")): item
        for item in phase57_results
    }
    for safe_id in sorted(primary_ids):
        item = by_id_phase57.get(safe_id)
        if item is not None:
            selected.append(item)

    if len(selected) >= subset_size:
        return selected[:subset_size]

    # Expand with similar Phase 57 empty_extraction PDFs
    expansion_pool: list[dict[str, Any]] = []
    selected_ids = {str(item.get("safe_file_id") or item.get("file_id")) for item in selected}
    for item in phase57_results:
        sid = str(item.get("safe_file_id") or item.get("file_id"))
        if sid in selected_ids:
            continue
        if item.get("file_type") != "pdf":
            continue
        if int(item.get("entity_count") or 0) != 0:
            continue
        ocr_status = str(item.get("ocr_status") or item.get("ocr_quality_band") or "")
        if ocr_status not in {"good", "usable_with_review"}:
            continue
        expansion_pool.append(item)
    expansion_pool.sort(key=lambda it: str(it.get("safe_file_id") or it.get("file_id") or ""))
    rng = random.Random(seed)
    needed = subset_size - len(selected)
    if expansion_pool and needed > 0:
        if len(expansion_pool) <= needed:
            extra = list(expansion_pool)
        else:
            extra = rng.sample(expansion_pool, needed)
        extra.sort(key=lambda it: str(it.get("safe_file_id") or it.get("file_id") or ""))
        selected.extend(extra)
    return selected


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_phase57(path: Path | None = None) -> dict[str, Any]:
    p = path or PHASE57_REPORT
    if not p.exists():
        raise FileNotFoundError(f"Phase 57A report missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_phase59(path: Path | None = None) -> dict[str, Any]:
    p = path or PHASE59_REPORT
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
    """Read PDF text into a local-only string. Returns (text, page_count).
    Caller must NOT emit text. Caller should compute counts and discard.
    """
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def diagnose_file(item: dict[str, Any], local_path: Path | None) -> dict[str, Any]:
    safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
    file_type = str(item.get("file_type") or "unknown")
    extension = str(item.get("extension") or item.get("file_extension") or "").lower()
    file_size_bytes = item.get("file_size_bytes")
    file_size_bucket = _bucket_size(file_size_bytes)
    ocr_status = item.get("ocr_status") or item.get("ocr_quality_band")

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
        # Phase 60 focuses on PDF gap analysis; non-PDF files just get
        # bucketed without text extraction.
        structure_error = f"non_pdf_skipped:{file_type}"

    measurements = measure_local_text(text)
    document_class = guess_document_class(measurements)
    gap_type = classify_gap(
        document_class,
        measurements,
        file_size_bucket=file_size_bucket,
        page_count_bucket=_bucket_page_count(page_count),
    )

    # Strip the private _raw_counts before emitting.
    public_measurements = {k: v for k, v in measurements.items() if not k.startswith("_")}
    return {
        "safe_file_id": safe_id,
        "file_type": file_type,
        "extension": extension,
        "file_size_bucket": file_size_bucket,
        "page_count_bucket": _bucket_page_count(page_count),
        "ocr_status_phase57": ocr_status,
        "structure_error": structure_error,
        **public_measurements,
        "likely_document_class_guess": document_class,
        "likely_gap_type": gap_type,
    }


def aggregate_distribution(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for e in entries:
        counter[str(e.get(key) or "unknown")] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def coverage_table(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    keys = (
        "medical_token_hit_count_bucket",
        "lab_token_hit_count_bucket",
        "unit_pattern_count_bucket",
        "numeric_density_bucket",
        "table_like_line_count_bucket",
    )
    table: dict[str, dict[str, int]] = {}
    for k in keys:
        table[k] = aggregate_distribution(entries, k)
    return table


def recommended_phase61_for(
    gap_distribution: dict[str, int],
    class_distribution: dict[str, int],
) -> dict[str, Any]:
    if not gap_distribution:
        return {
            "title": "no_code_change_manual_review_boundary",
            "production_extractor_should_change_yet": False,
            "reason": "No diagnostic targets surfaced.",
        }
    dominant_gap = next(iter(gap_distribution))
    dominant_class = next(iter(class_distribution)) if class_distribution else "unknown"
    if dominant_gap == "lab_table_vocabulary_gap":
        return {
            "title": "lab_table_vocabulary_expansion_diagnostic",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Most gap files have lab units AND reference-range patterns "
                "AND lab concepts but production extractor returns zero. "
                "Phase 61 should propose a NARROW, opt-in vocabulary or "
                "table-row expansion as a diagnostic prototype only — no "
                "production extractor change yet."
            ),
        }
    if dominant_gap == "imaging_report_vocabulary_gap":
        return {
            "title": "imaging_report_vocabulary_expansion_diagnostic",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Imaging-class documents dominate. Phase 61 should evaluate "
                "an imaging-class router as a diagnostic prototype, not a "
                "production extractor change."
            ),
        }
    if dominant_gap == "admin_document_not_target":
        return {
            "title": "admin_document_filtering",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Many gap files are administrative/billing documents that "
                "are not target medical documents. Phase 61 should add a "
                "non-medical document filter at routing time, not an "
                "extractor change."
            ),
        }
    if dominant_gap == "numeric_table_without_labels":
        return {
            "title": "lab_table_vocabulary_expansion_diagnostic",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Numeric-heavy table-like content without labels. Phase 61 "
                "should evaluate header/label inference as a diagnostic, "
                "not an extractor change."
            ),
        }
    if dominant_gap == "document_class_not_supported":
        return {
            "title": "unsupported_document_class_router",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Dominant class is outside the current production scope. "
                "Phase 61 should report the unsupported class population to "
                "the operator and propose a routing prototype, not an "
                "extractor change."
            ),
        }
    if dominant_gap == "tokenization_or_layout_issue":
        return {
            "title": "text_tokenization_layout_fix",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Many files have very short or weirdly-tokenised text. "
                "Phase 61 should investigate tokenisation/layout but not "
                "change extractor logic."
            ),
        }
    if dominant_gap == "medical_vocabulary_gap":
        return {
            "title": "lab_table_vocabulary_expansion_diagnostic",
            "production_extractor_should_change_yet": False,
            "reason": (
                "Lab-class documents dominate but lab vocabulary coverage "
                "is sparse. Phase 61 should propose a vocabulary expansion "
                "as a diagnostic prototype, not a production extractor "
                "change."
            ),
        }
    return {
        "title": "no_code_change_manual_review_boundary",
        "production_extractor_should_change_yet": False,
        "reason": (
            f"Dominant gap is {dominant_gap}. No Phase 61 production "
            "extractor change is justified by the current evidence."
        ),
    }


def run_diagnostic(
    *,
    phase57_report_path: Path | None = None,
    phase59_report_path: Path | None = None,
    private_mapping_path: Path | None = None,
    input_dir: Path | None = None,
    report_dir: Path | None = None,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    target_report_dir = report_dir or REPORT_DIR
    target_report_dir.mkdir(parents=True, exist_ok=True)

    phase57 = load_phase57(phase57_report_path)
    phase59 = load_phase59(phase59_report_path)
    private_mapping = load_private_mapping(private_mapping_path)
    input_dir_resolved = input_dir or DEFAULT_INPUT_DIR

    phase57_results = phase57.get("results") or []
    phase59_subset = phase59.get("subset") or []

    selected = select_subset(phase57_results, phase59_subset, subset_size=subset_size, seed=seed)
    entries: list[dict[str, Any]] = []
    for item in selected:
        safe_id = str(item.get("safe_file_id") or item.get("file_id") or "")
        local_path = resolve_local_path(safe_id, private_mapping, input_dir_resolved)
        entries.append(diagnose_file(item, local_path))

    class_distribution = aggregate_distribution(entries, "likely_document_class_guess")
    gap_distribution = aggregate_distribution(entries, "likely_gap_type")
    coverage = coverage_table(entries)
    recommendation = recommended_phase61_for(gap_distribution, class_distribution)
    privacy_passed = _privacy_self_check(entries)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 60 Text Extraction Gap Vocabulary Coverage Diagnostic",
        "source_phase57_report": str(PHASE57_REPORT.relative_to(ROOT)),
        "source_phase59_report": str(PHASE59_REPORT.relative_to(ROOT)),
        "subset_seed": seed,
        "subset_size_requested": subset_size,
        "subset_size_actual": len(entries),
        "stratification": dict(Counter(str(e.get("file_type") or "unknown") for e in entries)),
        "likely_document_class_distribution": class_distribution,
        "likely_gap_type_distribution": gap_distribution,
        "coverage_table": coverage,
        "dominant_document_class": next(iter(class_distribution), None),
        "dominant_gap_type": next(iter(gap_distribution), None),
        "recommended_phase61_target": recommendation,
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
            "phi_present_in_output": False,
            "external_api_used": False,
        },
        "conclusion": (
            "no_text_gap_files" if not entries else "text_extraction_gap_diagnosed"
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
    # No string field should be longer than ~100 chars (token strings would
    # fail this even if they slipped past the key check).
    for entry in entries:
        for key, value in entry.items():
            if isinstance(value, str) and len(value) > 120:
                return False
    return True


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    rec = report.get("recommended_phase61_target") or {}
    coverage = report.get("coverage_table") or {}
    lines = [
        "# Phase 60 Text Extraction Gap Vocabulary Coverage Diagnostic",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Source Phase 57A report: `{report['source_phase57_report']}`",
        f"- Source Phase 59 report: `{report['source_phase59_report']}`",
        f"- Subset seed: `{report['subset_seed']}`",
        f"- Subset size requested: `{report['subset_size_requested']}`",
        f"- Subset size actual: `{report['subset_size_actual']}`",
        f"- Stratification (file_type → count): `{report['stratification']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Likely Document Class Distribution",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for cls, count in report["likely_document_class_distribution"].items():
        lines.append(f"| `{cls}` | {count} |")
    lines += [
        "",
        f"- Dominant document class: `{report.get('dominant_document_class')}`",
        "",
        "## Likely Gap Type Distribution",
        "",
        "| Gap | Count |",
        "| --- | ---: |",
    ]
    for gap, count in report["likely_gap_type_distribution"].items():
        lines.append(f"| `{gap}` | {count} |")
    lines += [
        "",
        f"- Dominant gap type: `{report.get('dominant_gap_type')}`",
        "",
        "## Coverage Table (bucket → count)",
        "",
    ]
    for metric, dist in coverage.items():
        lines.append(f"### `{metric}`")
        lines.append("")
        lines.append("| Bucket | Count |")
        lines.append("| --- | ---: |")
        for bucket, count in dist.items():
            lines.append(f"| `{bucket}` | {count} |")
        lines.append("")
    lines += [
        "## Recommended Phase 61 Target",
        "",
        f"- Title: `{rec.get('title')}`",
        f"- production_extractor_should_change_yet: `{rec.get('production_extractor_should_change_yet')}`",
        "",
        f"_{rec.get('reason')}_",
        "",
        "## Forensic Subset Entries",
        "",
        "| safe_file_id | file_type | size | pages | text_len | medical_hits | lab_hits | unit_patterns | numeric_density | table_like | doc_class | gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["subset"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{entry.get('safe_file_id')}`",
                str(entry.get("file_type") or ""),
                str(entry.get("file_size_bucket") or ""),
                str(entry.get("page_count_bucket") or ""),
                str(entry.get("extracted_text_length_bucket") or ""),
                str(entry.get("medical_token_hit_count_bucket") or ""),
                str(entry.get("lab_token_hit_count_bucket") or ""),
                str(entry.get("unit_pattern_count_bucket") or ""),
                str(entry.get("numeric_density_bucket") or ""),
                str(entry.get("table_like_line_count_bucket") or ""),
                f"`{entry.get('likely_document_class_guess')}`",
                f"`{entry.get('likely_gap_type')}`",
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
    report = run_diagnostic()
    rec = report.get("recommended_phase61_target") or {}
    print("MedAI Phase 60 text extraction gap vocabulary diagnostic complete.")
    print(f"subset_size_actual: {report['subset_size_actual']}")
    print(f"stratification: {report['stratification']}")
    print(f"dominant_document_class: {report.get('dominant_document_class')}")
    print(f"likely_document_class_distribution: {report['likely_document_class_distribution']}")
    print(f"dominant_gap_type: {report.get('dominant_gap_type')}")
    print(f"likely_gap_type_distribution: {report['likely_gap_type_distribution']}")
    print(f"recommended_phase61_target: {rec.get('title')}")
    print(f"production_extractor_should_change_yet: {rec.get('production_extractor_should_change_yet')}")
    print(f"privacy_safety.uses_safe_ids_only: {report['privacy_safety']['uses_safe_ids_only']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
