"""CKA-TERM-01A — filename-only terminology classifier.

Classifies a filename (NOT contents) into one of:
- TerminologySystem.LOINC
- TerminologySystem.RXNORM
- TerminologySystem.UMLS
- TerminologySystem.SNOMED_CT
- None for "unknown"

Hard rules:
- File contents are NEVER read.
- Raw paths are NEVER returned in public summaries — only safe labels
  and counts.
- ZIPs are classified but NOT extracted (extraction is the caller's
  responsibility, with explicit zip-slip protection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clinical_knowledge.terminology.models import TerminologySystem


# Filename heuristics. Matching is case-insensitive substring.
# Order matters: more-specific patterns first to avoid mis-classification.
_PATTERNS: List[Tuple[TerminologySystem, str]] = [
    # LOINC
    (TerminologySystem.LOINC, "loinctable.csv"),
    (TerminologySystem.LOINC, "loinc.csv"),
    (TerminologySystem.LOINC, "loinc_"),
    (TerminologySystem.LOINC, "loinctable"),
    # RxNorm
    (TerminologySystem.RXNORM, "rxnconso.rrf"),
    (TerminologySystem.RXNORM, "rxnrel.rrf"),
    (TerminologySystem.RXNORM, "rxnsat.rrf"),
    (TerminologySystem.RXNORM, "rxnorm_"),
    # UMLS
    (TerminologySystem.UMLS, "mrconso.rrf"),
    (TerminologySystem.UMLS, "mrsty.rrf"),
    (TerminologySystem.UMLS, "mrrel.rrf"),
    (TerminologySystem.UMLS, "umls"),
    (TerminologySystem.UMLS, "mmsys"),
    # SNOMED CT
    (TerminologySystem.SNOMED_CT, "sct2_concept"),
    (TerminologySystem.SNOMED_CT, "sct2_description"),
    (TerminologySystem.SNOMED_CT, "sct2_relationship"),
    (TerminologySystem.SNOMED_CT, "snomedct"),
]


# ZIP heuristics — only the system label, never extraction here.
_ZIP_PATTERNS: List[Tuple[TerminologySystem, str]] = [
    (TerminologySystem.LOINC, "loinc"),
    (TerminologySystem.RXNORM, "rxnorm"),
    (TerminologySystem.UMLS, "umls"),
    (TerminologySystem.UMLS, "mmsys"),
    (TerminologySystem.SNOMED_CT, "snomedct"),
    (TerminologySystem.SNOMED_CT, "snomed-ct"),
    (TerminologySystem.SNOMED_CT, "snomed_ct"),
]


@dataclass
class FileClassification:
    """Public-report-safe classification for one filename."""

    system: Optional[TerminologySystem]
    is_zip: bool
    name_length: int
    safe_kind: str   # one of: "rrf", "csv", "txt", "zip", "unknown"

    def safe_public_summary(self) -> dict:
        return {
            "system": (self.system.value if self.system else None),
            "is_zip": self.is_zip,
            "name_length": self.name_length,
            "safe_kind": self.safe_kind,
        }


def _classify_extension(name: str) -> str:
    n = name.lower()
    if n.endswith(".rrf"):
        return "rrf"
    if n.endswith(".csv"):
        return "csv"
    if n.endswith(".txt"):
        return "txt"
    if n.endswith(".zip"):
        return "zip"
    return "unknown"


def classify_filename(name: str) -> FileClassification:
    """Classify a single filename. Never opens the file.

    Returns FileClassification with `system=None` for unknown names.
    """
    if not isinstance(name, str) or not name:
        return FileClassification(
            system=None, is_zip=False, name_length=0, safe_kind="unknown",
        )
    base = Path(name).name
    n = base.lower()
    is_zip = n.endswith(".zip")
    safe_kind = _classify_extension(base)

    if is_zip:
        for system, frag in _ZIP_PATTERNS:
            if frag in n:
                return FileClassification(
                    system=system, is_zip=True,
                    name_length=len(base), safe_kind="zip",
                )
        return FileClassification(
            system=None, is_zip=True, name_length=len(base), safe_kind="zip",
        )

    for system, frag in _PATTERNS:
        if frag in n:
            return FileClassification(
                system=system, is_zip=False,
                name_length=len(base), safe_kind=safe_kind,
            )
    return FileClassification(
        system=None, is_zip=False, name_length=len(base), safe_kind=safe_kind,
    )


@dataclass
class ClassificationSummary:
    """Public-report-safe summary across many filenames."""

    counts_by_system: Dict[str, int] = field(default_factory=dict)
    zip_counts_by_system: Dict[str, int] = field(default_factory=dict)
    unknown_count: int = 0
    total: int = 0

    def safe_public_summary(self) -> dict:
        return {
            "counts_by_system": dict(self.counts_by_system),
            "zip_counts_by_system": dict(self.zip_counts_by_system),
            "unknown_count": self.unknown_count,
            "total": self.total,
        }


def classify_filenames(names: List[str]) -> ClassificationSummary:
    """Aggregate classify_filename across a list of names.

    Returns counts only — never raw filenames or paths.
    """
    summary = ClassificationSummary()
    for n in names:
        c = classify_filename(n)
        summary.total += 1
        if c.system is None:
            summary.unknown_count += 1
            continue
        if c.is_zip:
            summary.zip_counts_by_system[c.system.value] = (
                summary.zip_counts_by_system.get(c.system.value, 0) + 1
            )
        else:
            summary.counts_by_system[c.system.value] = (
                summary.counts_by_system.get(c.system.value, 0) + 1
            )
    return summary
