"""Deterministic, local-only extracted medical fact adapter.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01.

This module is intentionally conservative:

* fewer correct facts is better than unsafe facts;
* deterministic regex + line-tokenization only;
* no external API, no model dependency;
* no raw line text is emitted in public-safe output;
* every extracted fact is review-bound by default;
* every extracted fact has ``auto_accept_allowed=False``.

The adapter is structured as a pure-Python helper that the pipeline
calls *after* the existing extractor returns entities. It does not
replace the existing extractor. It supplements it only when the
document family is lab-style and the existing entity list is empty or
sparse.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

PARSER_NAME = "deterministic_lab_line_adapter"
PARSER_VERSION = "01"
EXTRACTION_METHOD = "rules_based"

#: A new entity emitted by this adapter carries this provenance tag so the
#: pipeline can attribute it cleanly without re-parsing.
PROVENANCE_TAG = "extracted_medical_facts_adapter"

#: Conservative confidence assigned to deterministic facts. Below
#: EXTRACTION_ACCEPT_THRESHOLD on purpose so facts land in review.
CONSERVATIVE_CONFIDENCE = 0.55

#: Public-safe preview cap. Even if many facts are extracted, the public
#: report only carries the first ``MAX_PREVIEW_FACTS`` redacted entries.
MAX_PREVIEW_FACTS = 12

#: Document family / type strings that mean "lab-style" for this adapter.
LAB_STYLE_FAMILY_LABELS = {
    "Lab result",
    "lab result",
    "lab_result",
    "lab_report",
    "Lab report",
    "laboratory_result",
    "Laboratory result",
}

LAB_STYLE_DOCUMENT_TYPE_VALUES = {
    "lab_report",
    "lab_result",
    "laboratory_report",
    "laboratory_result",
    "Lab result",
    "Lab report",
}


# ---------------------------------------------------------------------------
# Date / patient-ID guards
# ---------------------------------------------------------------------------

#: Patterns we explicitly refuse to parse as lab values (false-positive guard).
#: Tight on purpose so lab reference ranges like 4.0-11.0 are not misread as
#: dates. We require either a 4-digit year (anchored at start or end) or an
#: explicit month name.
_DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"         # 2024-05-21 / 2024/05/21
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4}"        # 21-05-2024 / 21/05/2024
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

#: Lines that start with date-of-collection / collected-on / specimen-date
#: language. Conservative: only matches obvious headers, not lab rows.
_DATE_LABEL_PATTERN = re.compile(
    r"^\s*(?:date(?:\s+of)?(?:\s+(?:birth|collection|service|report|test|sample))?|collected(?:\s+on)?|specimen\s+date|reported\s+on)\b",
    re.IGNORECASE,
)

_PATIENT_ID_PATTERN = re.compile(
    r"\b(?:MRN|Patient\s*ID|Pt\s*ID|ID\s*No\.?|Chart\s*No\.?)\s*[:#]?\s*[A-Z0-9-]+\b",
    re.IGNORECASE,
)

_DOB_PATTERN = re.compile(
    r"\b(?:DOB|Date\s+of\s+Birth|D\.O\.B\.?)\s*[:#]?\s*\S+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Lab-line regexes (English Latin + Cyrillic / Russian)
# ---------------------------------------------------------------------------

#: English / Latin "test : value unit (reference ref) [flag]"
_LAB_LINE_EN = re.compile(
    r"^"
    r"(?P<name>[A-Za-z][A-Za-z0-9 ()/.-]{1,60}?)"            # test name
    r"\s*[:=]\s*"
    r"(?P<value>-?\d+(?:[.,]\d+)?|positive|negative|trace|detected|not\s+detected)"
    r"(?:\s*(?P<unit>%|mg/dL|mg/L|g/dL|g/L|mmol/L|mEq/L|"
    r"x10E3/uL|x10E6/uL|x10E6/mL|x10E9/L|x10E12/L|/hpf|/lpf|"
    r"CFU/mL|IU/L|U/L|mU/L|ng/mL|ng/dL|pg/mL|µg/L|ug/L))?"
    r"(?:\s*\(\s*(?:ref\.?|reference)\s*[:\s]?\s*(?P<reference>[^)]+?)\s*\))?"
    r"(?:\s*\[(?P<flag>H|L|HIGH|LOW|HH|LL|CRITICAL|ABNORMAL|NORMAL)\])?"
    r"\s*$",
    re.IGNORECASE,
)

#: Cyrillic / Russian "анализ : значение единица"
_LAB_LINE_RU = re.compile(
    r"^"
    r"(?P<name>[А-яЁёІіЇї][А-яЁёІіЇїA-Za-z 0-9()/.-]{1,60}?)"
    r"\s*[:=]\s*"
    r"(?P<value>-?\d+(?:[.,]\d+)?|положительн\w*|отрицательн\w*|обнаружено|не\s+обнаружено|выявлено|не\s+выявлено)"
    r"(?:\s*(?P<unit>%|мг/дл|г/л|г/дл|ммоль/л|мкмоль/л|нмоль/л|мкг/л|нг/мл|мкЕд/мл|ЕД/л|Ед/л|МЕ/л|"
    r"x10E3/мкл|x10E6/мкл|x10E9/л|x10E12/л|/мкл|/мл|/л|КОЕ/мл|"
    r"mg/dL|g/dL|mmol/L|U/L|IU/L))?"
    r"(?:\s*\(\s*(?:реф\.?|референс|норма)\s*[:\s]?\s*(?P<reference>[^)]+?)\s*\))?"
    r"\s*$",
    re.IGNORECASE | re.UNICODE,
)

#: Tab/multi-space-separated row format:
#:    "Glucose   5.4    mmol/L    3.9-5.5    [normal]"
_LAB_ROW_GENERIC = re.compile(
    r"^"
    r"(?P<name>[A-Za-zА-яЁё][A-Za-z0-9 А-яЁё()/.,-]{1,60}?)"
    r"(?:\s{2,}|\t+)"
    r"(?P<value>-?\d+(?:[.,]\d+)?|positive|negative|trace|detected|not\s+detected|"
    r"положительн\w*|отрицательн\w*|обнаружено|не\s+обнаружено)"
    r"(?:(?:\s{2,}|\t+)(?P<unit>[A-Za-zА-я%/.µμ]+(?:/[A-Za-zА-я%]+)?))?"
    r"(?:(?:\s{2,}|\t+)(?P<reference>[<>]?\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?)?))?"
    r"(?:(?:\s{2,}|\t+)\[?(?P<flag>[HL]|HIGH|LOW|HH|LL|normal|abnormal|H|L)\]?)?"
    r"\s*$",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_lab_style_document(metadata: dict[str, Any] | None) -> bool:
    """Return True if metadata indicates a lab-style document family.

    Conservative: only returns True when the family or document_type
    fields explicitly carry a recognized lab label. Defaults to False.
    """
    if not isinstance(metadata, dict):
        return False
    document_type = str(metadata.get("document_type") or "").strip()
    if document_type in LAB_STYLE_DOCUMENT_TYPE_VALUES:
        return True
    family_diagnostic = metadata.get("document_family_classification_diagnostic")
    if isinstance(family_diagnostic, dict):
        candidate = str(family_diagnostic.get("candidate_family") or "").strip()
        if candidate in LAB_STYLE_FAMILY_LABELS:
            return True
    return False


def _hash_for_line(line: str) -> str:
    """Return a stable short hash for a source line.

    Never returns the source line itself. Used as a non-identifying
    provenance handle in public-safe output.
    """
    digest = hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest()
    return digest[:12]


def _looks_like_date_or_id_line(line: str) -> bool:
    """Refuse to parse lines that look like date/ID/DOB/header lines."""
    if _DATE_LABEL_PATTERN.match(line):
        return True
    if _PATIENT_ID_PATTERN.search(line):
        return True
    if _DOB_PATTERN.search(line):
        return True
    if _DATE_PATTERN.search(line):
        return True
    return False


def _normalize_value_text(value: str) -> str:
    value = value.strip()
    return value.replace(",", ".") if re.match(r"^-?\d+,\d+$", value) else value


def _clean_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    cleaned = unit.strip().strip(".,;:")
    return cleaned or None


def _clean_optional(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip().strip(".,;:")
    return cleaned or None


def _safe_test_name(name: str) -> str:
    """Trim, collapse whitespace, drop trailing punctuation."""
    name = re.sub(r"\s+", " ", name).strip().strip(":=").strip()
    return name


def _line_iter(text: str) -> Iterable[str]:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _build_lab_entity(
    *,
    line: str,
    test_name: str,
    value: str,
    unit: str | None,
    reference_range: str | None,
    flag: str | None,
    language_hint: str,
) -> dict[str, Any]:
    """Build a structured lab-observation entity.

    The entity is shaped so it can be merged into
    ``extractor_result["entities"]`` and converted to MKBRecord by the
    existing ``ExecutionPipeline._entities_to_records`` flow.

    Public-safe contract:

    * ``text`` is the test name only — no raw source line.
    * ``source_line_hash`` is the truncated SHA-256 of the source line;
      the original line is NOT included.
    * the entity is review-bound and not auto-accept eligible.
    """
    structured: dict[str, Any] = {
        "test_name": test_name,
        "value": value,
        "source_line_hash": _hash_for_line(line),
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "extraction_method": EXTRACTION_METHOD,
        "requires_human_review": True,
        "auto_accept_allowed": False,
        "language_hint": language_hint,
        "provenance": PROVENANCE_TAG,
    }
    if unit:
        structured["unit"] = unit
    if reference_range:
        structured["reference_range"] = reference_range
    if flag:
        structured["flag"] = flag
    return {
        "type": "test_result",
        "text": test_name,
        "structured": structured,
        "confidence": CONSERVATIVE_CONFIDENCE,
        "tags": ["factual_extraction", "requires_source_comparison"],
    }


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    cyr = sum(1 for c in letters if "CYRILLIC" in unicodedata.name(c, ""))
    ratio = cyr / len(letters)
    if ratio >= 0.30:
        return "ru"
    if ratio >= 0.10:
        return "mixed"
    return "en"


def normalize_extracted_fact(entity: dict[str, Any]) -> dict[str, Any]:
    """Normalize an arbitrary fact dict into the adapter's safe shape.

    Defensive: callers may already have entities from other extractors;
    this helper enforces the public-safe contract on a single entity.
    """
    if not isinstance(entity, dict):
        return {}
    structured = dict(entity.get("structured") or {})
    structured.setdefault("requires_human_review", True)
    structured.setdefault("auto_accept_allowed", False)
    structured.setdefault("parser_name", PARSER_NAME)
    structured.setdefault("parser_version", PARSER_VERSION)
    structured.setdefault("extraction_method", structured.get("extraction_method", EXTRACTION_METHOD))
    structured.setdefault("provenance", PROVENANCE_TAG)
    structured.pop("raw_line", None)
    structured.pop("source_line", None)
    return {
        "type": str(entity.get("type") or "test_result"),
        "text": str(entity.get("text") or "").strip(),
        "structured": structured,
        "confidence": float(entity.get("confidence", CONSERVATIVE_CONFIDENCE)),
        "tags": list(entity.get("tags") or ["factual_extraction", "requires_source_comparison"]),
    }


def extract_lab_observation_entities(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract conservative lab-observation entities from raw text.

    Deterministic. Local-only. Synthetic-safe.

    Returns a list of entity dicts ready to be merged into the pipeline's
    existing ``extracted["entities"]`` list. Each entity has the
    public-safe shape produced by ``_build_lab_entity``.
    """
    if not text:
        return []
    language_hint = _detect_language(text)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for raw_line in _line_iter(text):
        if len(raw_line) > 300:
            continue
        if _looks_like_date_or_id_line(raw_line):
            continue
        match = _LAB_LINE_EN.match(raw_line) or _LAB_LINE_RU.match(raw_line) or _LAB_ROW_GENERIC.match(raw_line)
        if not match:
            continue
        name = _safe_test_name(match.group("name") or "")
        value_raw = (match.group("value") or "").strip()
        if not name or not value_raw or len(name) < 2 or len(name) > 80:
            continue
        unit = _clean_unit(match.groupdict().get("unit"))
        reference_range = _clean_optional(match.groupdict().get("reference"))
        flag = _clean_optional(match.groupdict().get("flag"))
        value_text = _normalize_value_text(value_raw)
        dedup_key = (name.lower(), value_text.lower(), (unit or "").lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(
            _build_lab_entity(
                line=raw_line,
                test_name=name,
                value=value_text,
                unit=unit,
                reference_range=reference_range,
                flag=flag,
                language_hint=language_hint,
            )
        )
    return out


def merge_facts_into_entities(
    existing_entities: list[dict[str, Any]] | None,
    new_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Defensively merge adapter facts into the pipeline's entity list.

    * existing entities are preserved verbatim;
    * adapter facts are appended only when there is no entity with the
      same ``(type, normalized_text, normalized_value)`` triple already;
    * adapter facts always carry the provenance tag in their structured
      payload.
    """
    merged: list[dict[str, Any]] = []
    existing = list(existing_entities or [])
    merged.extend(existing)

    def _key(entity: dict[str, Any]) -> tuple[str, str, str]:
        structured = entity.get("structured") or {}
        return (
            str(entity.get("type") or "").lower(),
            str(entity.get("text") or "").strip().lower(),
            str(structured.get("value") or "").strip().lower(),
        )

    seen = {_key(e) for e in existing if isinstance(e, dict)}
    for fact in new_facts:
        normalized = normalize_extracted_fact(fact)
        key = _key(normalized)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def summarize_extracted_facts_for_public_report(
    entities: list[dict[str, Any]] | None,
    *,
    max_preview: int = MAX_PREVIEW_FACTS,
) -> dict[str, Any]:
    """Build a public-safe summary of extracted facts.

    Contract:

    * counts are integers;
    * preview entries do NOT include raw source line;
    * preview entries do NOT include filenames or paths;
    * preview entries carry only: ``type``, ``test_name``, ``value``,
      ``unit``, ``reference_range``, ``flag``, ``confidence``,
      ``language_hint``, ``parser_name``, ``requires_review``,
      ``auto_accept_allowed``, ``source_line_hash`` (12-char).

    For interop with the existing spaCy extractor, top-level keys
    ``value`` / ``unit`` / ``status`` / ``reference_range`` / ``flag``
    are accepted as fallback when no ``structured`` payload is present.
    Review-bound default is enforced regardless of upstream shape.
    """
    entities = entities or []
    fact_count = 0
    types_seen: dict[str, int] = {}
    preview: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type")) != "test_result":
            structured = entity.get("structured") or {}
            if structured.get("provenance") != PROVENANCE_TAG:
                continue
        fact_count += 1
        fact_type = str(entity.get("type") or "test_result")
        types_seen[fact_type] = types_seen.get(fact_type, 0) + 1
        if len(preview) >= max_preview:
            continue
        structured = entity.get("structured") or {}
        # Defensive fallback: existing spaCy entities carry value/unit at
        # top level (not inside a structured dict). Read top-level keys
        # only when the structured payload is missing them.
        value = structured.get("value") or entity.get("value") or ""
        unit = structured.get("unit") or entity.get("unit") or ""
        reference_range = structured.get("reference_range") or entity.get("reference_range") or ""
        flag = structured.get("flag") or entity.get("status") or entity.get("flag") or ""
        preview.append(
            {
                "type": fact_type,
                "test_name": str(entity.get("text") or structured.get("test_name") or ""),
                "value": str(value),
                "unit": str(unit) or None,
                "reference_range": str(reference_range) or None,
                "flag": str(flag) or None,
                "confidence": float(entity.get("confidence", CONSERVATIVE_CONFIDENCE)),
                "language_hint": str(structured.get("language_hint") or "unknown"),
                "parser_name": str(structured.get("parser_name") or entity.get("parser_name") or "unknown"),
                # Review-bound default holds regardless of upstream extractor.
                "requires_review": bool(structured.get("requires_human_review", True)),
                "auto_accept_allowed": bool(structured.get("auto_accept_allowed", False)),
                "source_line_hash": str(structured.get("source_line_hash") or ""),
            }
        )
    return {
        "extracted_medical_fact_count": fact_count,
        "extracted_medical_fact_types": sorted(types_seen.keys()),
        "extracted_medical_fact_type_counts": types_seen,
        "extracted_medical_facts_preview_safe": preview,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "extraction_method": EXTRACTION_METHOD,
        "auto_accept_allowed_default": False,
        "review_required_default": True,
    }


def facts_for_ui(extractor_result: dict[str, Any] | None) -> dict[str, Any]:
    """Return a render-friendly dict for the Run & Review card.

    Streamlit-free. Returns a dict the UI can iterate. When no facts
    are present, returns a placeholder dict whose ``rows`` list is
    empty and whose ``message`` field tells the operator that
    classification succeeded only.
    """
    extractor_result = extractor_result or {}
    preview = extractor_result.get("extracted_medical_facts_preview_safe") or []
    if not isinstance(preview, list):
        preview = []
    rows: list[dict[str, Any]] = []
    for entry in preview:
        if not isinstance(entry, dict):
            continue
        review_status = "review-required" if entry.get("requires_review", True) else "ready"
        mkb_status = "pending_validation_review" if review_status == "review-required" else "candidate"
        rows.append(
            {
                "type": str(entry.get("type") or "test_result"),
                "test_name": str(entry.get("test_name") or ""),
                "value": str(entry.get("value") or ""),
                "unit": str(entry.get("unit") or "") or "—",
                "reference_range": str(entry.get("reference_range") or "") or "—",
                "flag": str(entry.get("flag") or "") or "—",
                "confidence": f"{float(entry.get('confidence', CONSERVATIVE_CONFIDENCE)):.2f}",
                "review_status": review_status,
                "mkb_status": mkb_status,
            }
        )
    counts = {
        "structured_facts_extracted": int(extractor_result.get("extracted_medical_fact_count", len(rows))),
        "written_to_mkb": int(extractor_result.get("extraction_to_mkb_written_count", 0)),
        "needs_review": int(extractor_result.get("extraction_to_mkb_review_count", 0)),
    }
    if not rows:
        message = "No structured medical facts extracted. Classification succeeded only."
    else:
        message = (
            "Structured factual observations may be present below. They are not "
            "clinically interpreted and require source comparison before use."
        )
    return {
        "columns": [
            "Type",
            "Test / observation",
            "Value",
            "Unit",
            "Reference range",
            "Flag",
            "Confidence",
            "Review status",
            "MKB status",
        ],
        "rows": rows,
        "row_count": len(rows),
        "counts": counts,
        "message": message,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "auto_accept_allowed": False,
        "review_required": True,
    }


__all__ = [
    "PARSER_NAME",
    "PARSER_VERSION",
    "EXTRACTION_METHOD",
    "PROVENANCE_TAG",
    "CONSERVATIVE_CONFIDENCE",
    "MAX_PREVIEW_FACTS",
    "LAB_STYLE_FAMILY_LABELS",
    "LAB_STYLE_DOCUMENT_TYPE_VALUES",
    "is_lab_style_document",
    "normalize_extracted_fact",
    "extract_lab_observation_entities",
    "merge_facts_into_entities",
    "summarize_extracted_facts_for_public_report",
    "facts_for_ui",
]
