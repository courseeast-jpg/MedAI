"""CKA-TERM-01 — streaming parsers for synthetic / small terminology fixtures.

Real licensed data is NOT shipped or downloaded by this module. Each
parser:

- streams line-by-line (no full-file reads where avoidable)
- caps the row count via `max_rows`
- normalizes terms deterministically
- emits TerminologyConcept rows with `synthetic` set per the
  caller's `force_synthetic` flag
- never invents codes — unmatched lines are skipped
- never copies clinical interpretations, license text, or PHI

Parsers also accept an in-memory `text` argument (used by tests) so
that synthetic fixtures don't have to be on disk.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologySystem,
    normalize_query,
)


_DEFAULT_MAX_ROWS = 1000


@dataclass
class ParseResult:
    """Public-report-safe parser outcome."""

    system: TerminologySystem
    rows_seen: int = 0
    concepts: List[TerminologyConcept] = field(default_factory=list)
    skipped_rows: int = 0

    def safe_public_summary(self) -> dict:
        return {
            "system": self.system.value,
            "rows_seen": self.rows_seen,
            "concepts_emitted": len(self.concepts),
            "skipped_rows": self.skipped_rows,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_lines(
    path: Optional[str],
    text: Optional[str],
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Yield lines from a path or in-memory text. Caller supplies one."""
    if text is not None:
        for line in text.splitlines():
            yield line
        return
    if not path:
        return
    p = Path(path)
    if not p.exists() or not p.is_file():
        return
    with open(p, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            yield line.rstrip("\r\n")


# ---------------------------------------------------------------------------
# UMLS MRCONSO.RRF
# ---------------------------------------------------------------------------
#
# The full UMLS MRCONSO row has 18 pipe-separated fields. The minimum we
# care about for code/display extraction is:
#   col 0  CUI           (used as concept code)
#   col 1  LAT           (language; we accept ENG only)
#   col 14 STR           (display string)
#
# Additional fields are tolerated but ignored.


def parse_umls_mrconso(
    *,
    path: Optional[str] = None,
    text: Optional[str] = None,
    max_rows: int = _DEFAULT_MAX_ROWS,
    force_synthetic: bool = True,
    source_safe_id: Optional[str] = None,
    version: Optional[str] = "synthetic-test-1",
) -> ParseResult:
    result = ParseResult(system=TerminologySystem.UMLS)

    by_cui: dict = {}    # cui -> (display, [synonyms])
    rows = 0
    for line in _open_lines(path, text):
        if rows >= max_rows:
            break
        if not line:
            continue
        rows += 1
        fields = line.split("|")
        if len(fields) < 15:
            result.skipped_rows += 1
            continue
        cui = fields[0].strip()
        lat = fields[1].strip().upper() if len(fields) > 1 else "ENG"
        try:
            display = fields[14].strip()
        except IndexError:
            result.skipped_rows += 1
            continue
        if not cui or not display or lat not in ("ENG", ""):
            result.skipped_rows += 1
            continue

        if cui not in by_cui:
            by_cui[cui] = (display, [])
        else:
            existing_display, syns = by_cui[cui]
            if display != existing_display and display not in syns:
                syns.append(display)
                by_cui[cui] = (existing_display, syns)

    for cui, (display, synonyms) in by_cui.items():
        result.concepts.append(
            TerminologyConcept(
                concept_id=f"term_concept_umls_{cui}",
                system=TerminologySystem.UMLS,
                code=cui,
                display=display,
                synonyms=synonyms,
                version=version,
                source_safe_id=source_safe_id,
                active=True,
                synthetic=bool(force_synthetic),
            )
        )

    result.rows_seen = rows
    return result


# ---------------------------------------------------------------------------
# SNOMED CT RF2 — concept + description pair
# ---------------------------------------------------------------------------
#
# Concept file (sct2_Concept_*.txt) — tab-separated:
#   id, effectiveTime, active, moduleId, definitionStatusId
# Description file (sct2_Description_*.txt) — tab-separated:
#   id, effectiveTime, active, moduleId, conceptId, languageCode,
#   typeId, term, caseSignificanceId


def parse_snomed_concept_description(
    *,
    concept_text: Optional[str] = None,
    description_text: Optional[str] = None,
    concept_path: Optional[str] = None,
    description_path: Optional[str] = None,
    max_rows: int = _DEFAULT_MAX_ROWS,
    force_synthetic: bool = True,
    source_safe_id: Optional[str] = None,
    version: Optional[str] = "synthetic-test-1",
) -> ParseResult:
    result = ParseResult(system=TerminologySystem.SNOMED_CT)

    # Active concept ids.
    active_ids: set = set()
    rows = 0
    for line in _open_lines(concept_path, concept_text):
        if rows >= max_rows:
            break
        if not line:
            continue
        if line.lower().startswith("id\t"):
            continue    # header
        rows += 1
        parts = line.split("\t")
        if len(parts) < 5:
            result.skipped_rows += 1
            continue
        cid = parts[0].strip()
        active = parts[2].strip()
        if not cid or active != "1":
            continue
        active_ids.add(cid)

    # Map concept id -> (preferred display, synonyms).
    by_id: dict = {}
    rows_d = 0
    for line in _open_lines(description_path, description_text):
        if rows_d >= max_rows:
            break
        if not line:
            continue
        if line.lower().startswith("id\t"):
            continue
        rows_d += 1
        parts = line.split("\t")
        if len(parts) < 8:
            result.skipped_rows += 1
            continue
        active = parts[2].strip()
        if active != "1":
            continue
        concept_id = parts[4].strip()
        lang = parts[5].strip().lower() if len(parts) > 5 else "en"
        type_id = parts[6].strip()
        term = parts[7].strip()
        if not concept_id or not term:
            continue
        if concept_id not in active_ids:
            continue
        if lang not in ("en", "en-us", ""):
            continue

        # type_id "900000000000003001" = FSN; anything else treated as synonym.
        if concept_id not in by_id:
            by_id[concept_id] = (term, [])
        else:
            existing_term, syns = by_id[concept_id]
            if term != existing_term and term not in syns:
                syns.append(term)
                by_id[concept_id] = (existing_term, syns)

    for cid, (display, synonyms) in by_id.items():
        result.concepts.append(
            TerminologyConcept(
                concept_id=f"term_concept_snomed_{cid}",
                system=TerminologySystem.SNOMED_CT,
                code=cid,
                display=display,
                synonyms=synonyms,
                version=version,
                source_safe_id=source_safe_id,
                active=True,
                synthetic=bool(force_synthetic),
            )
        )
    result.rows_seen = rows + rows_d
    return result


# ---------------------------------------------------------------------------
# RxNorm RXNCONSO.RRF
# ---------------------------------------------------------------------------
#
# RxNorm RXNCONSO has a similar layout to UMLS MRCONSO. We use:
#   col 0  RXCUI
#   col 1  LAT (English filter)
#   col 14 STR (display)


def parse_rxnorm_rxnconso(
    *,
    path: Optional[str] = None,
    text: Optional[str] = None,
    max_rows: int = _DEFAULT_MAX_ROWS,
    force_synthetic: bool = True,
    source_safe_id: Optional[str] = None,
    version: Optional[str] = "synthetic-test-1",
) -> ParseResult:
    # Reuse the UMLS path with the system relabeled.
    sub = parse_umls_mrconso(
        path=path, text=text, max_rows=max_rows,
        force_synthetic=force_synthetic, source_safe_id=source_safe_id,
        version=version,
    )
    relabelled = ParseResult(
        system=TerminologySystem.RXNORM,
        rows_seen=sub.rows_seen,
        skipped_rows=sub.skipped_rows,
    )
    for c in sub.concepts:
        relabelled.concepts.append(
            TerminologyConcept(
                concept_id=f"term_concept_rxnorm_{c.code}",
                system=TerminologySystem.RXNORM,
                code=c.code,
                display=c.display,
                synonyms=list(c.synonyms),
                version=c.version,
                source_safe_id=c.source_safe_id,
                active=c.active,
                synthetic=c.synthetic,
            )
        )
    return relabelled


# ---------------------------------------------------------------------------
# LOINC CSV
# ---------------------------------------------------------------------------
#
# LOINC CSV header includes LOINC_NUM, COMPONENT, ..., LONG_COMMON_NAME.


def parse_loinc_csv(
    *,
    path: Optional[str] = None,
    text: Optional[str] = None,
    max_rows: int = _DEFAULT_MAX_ROWS,
    force_synthetic: bool = True,
    source_safe_id: Optional[str] = None,
    version: Optional[str] = "synthetic-test-1",
) -> ParseResult:
    result = ParseResult(system=TerminologySystem.LOINC)

    if text is not None:
        stream = io.StringIO(text)
    elif path:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return result
        stream = open(p, "r", encoding="utf-8", errors="replace", newline="")
    else:
        return result

    try:
        reader = csv.DictReader(stream)
        rows = 0
        for row in reader:
            if rows >= max_rows:
                break
            rows += 1
            code = (row.get("LOINC_NUM") or row.get("loinc_num") or "").strip()
            display = (row.get("LONG_COMMON_NAME")
                       or row.get("long_common_name")
                       or row.get("COMPONENT")
                       or row.get("component")
                       or "").strip()
            if not code or not display:
                result.skipped_rows += 1
                continue
            result.concepts.append(
                TerminologyConcept(
                    concept_id=f"term_concept_loinc_{code}",
                    system=TerminologySystem.LOINC,
                    code=code,
                    display=display,
                    synonyms=[],
                    version=version,
                    source_safe_id=source_safe_id,
                    active=True,
                    synthetic=bool(force_synthetic),
                )
            )
        result.rows_seen = rows
    finally:
        if path:
            try:
                stream.close()
            except Exception:    # noqa: BLE001
                pass

    return result
