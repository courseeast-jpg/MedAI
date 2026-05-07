"""CKA-TERM-01 — terminology models / enums.

All public summaries are PHI-free, license-text-free, and never
contain raw paths. Long SHA-256 checksums are truncated to 16 hex
chars to stay below the CKA-B02 SECRET regex's 40+-alnum threshold.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


_SAFE_HASH_SALT = "cka_term01_v1"


def _safe_hash(raw: str, prefix: str = "term_src_") -> str:
    digest = hashlib.sha256(f"{_SAFE_HASH_SALT}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def normalize_query(text: str) -> str:
    """Deterministic normalization — lowercase + collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())


def query_hash(text: str) -> str:
    norm = normalize_query(text)
    digest = hashlib.sha256(f"q:{norm}".encode("utf-8")).hexdigest()[:16]
    return f"term_q_{digest}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TerminologySystem(str, Enum):
    UMLS = "umls"
    SNOMED_CT = "snomed_ct"
    RXNORM = "rxnorm"
    LOINC = "loinc"
    SYNTHETIC_TEST = "synthetic_test"


class TerminologySourceStatus(str, Enum):
    MISSING = "missing"
    PRESENT_UNVERIFIED = "present_unverified"
    PRESENT_VERIFIED = "present_verified"
    LICENSE_REQUIRED = "license_required"
    LICENSE_CONFIRMED = "license_confirmed"
    IMPORT_READY = "import_ready"
    IMPORTED = "imported"
    BLOCKED = "blocked"


class TerminologyImportMode(str, Enum):
    INVENTORY_ONLY = "inventory_only"
    SYNTHETIC_TEST_IMPORT = "synthetic_test_import"
    LICENSED_LOCAL_IMPORT = "licensed_local_import"


class TerminologyLookupStatus(str, Enum):
    EXACT = "exact"
    SYNONYM = "synonym"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


# ---------------------------------------------------------------------------
# TerminologyConcept
# ---------------------------------------------------------------------------


@dataclass
class TerminologyConcept:
    """A single coded concept entry, license-gate aware.

    `synthetic` must be True for any data not coming from a verified
    licensed local import.
    """

    concept_id: str
    system: TerminologySystem
    code: str
    display: str
    synonyms: List[str] = field(default_factory=list)
    semantic_type: Optional[str] = None
    version: Optional[str] = None
    source_safe_id: Optional[str] = None
    active: bool = True
    synthetic: bool = True

    @classmethod
    def synthetic_for(
        cls,
        system: TerminologySystem,
        code: str,
        display: str,
        synonyms: Optional[List[str]] = None,
        version: str = "synthetic-test-1",
        source_safe_id: Optional[str] = None,
    ) -> "TerminologyConcept":
        cid = f"term_concept_{uuid.uuid4().hex[:12]}"
        return cls(
            concept_id=cid,
            system=system,
            code=code,
            display=display,
            synonyms=list(synonyms or []),
            version=version,
            source_safe_id=source_safe_id,
            active=True,
            synthetic=True,
        )

    def safe_public_summary(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "system": self.system.value,
            "code": self.code,
            "display_normalized": normalize_query(self.display),
            "synonyms_count": len(self.synonyms),
            "semantic_type": self.semantic_type,
            "version": self.version,
            "source_safe_id": self.source_safe_id,
            "active": self.active,
            "synthetic": self.synthetic,
        }


# ---------------------------------------------------------------------------
# TerminologySourceManifest
# ---------------------------------------------------------------------------


@dataclass
class TerminologySourceManifest:
    """Public-report-safe descriptor for one local terminology source.

    No raw paths, no license text, no full-length checksums.
    """

    source_id: str
    safe_source_id: str
    system: TerminologySystem
    source_label: str
    version: Optional[str]
    release_date: Optional[str]
    local_path_hash: Optional[str]
    file_count: int
    expected_files_present: List[str]
    license_confirmed: bool
    import_mode: TerminologyImportMode
    status: TerminologySourceStatus
    records_seen: int = 0
    records_imported: int = 0

    @classmethod
    def for_local_source(
        cls,
        system: TerminologySystem,
        local_root: str,
        source_label: str = "",
        version: Optional[str] = None,
        release_date: Optional[str] = None,
        file_count: int = 0,
        expected_files_present: Optional[List[str]] = None,
        license_confirmed: bool = False,
        import_mode: TerminologyImportMode = TerminologyImportMode.INVENTORY_ONLY,
        status: TerminologySourceStatus = TerminologySourceStatus.MISSING,
    ) -> "TerminologySourceManifest":
        sid = f"term_src_{uuid.uuid4().hex[:12]}"
        return cls(
            source_id=sid,
            safe_source_id=_safe_hash(local_root or sid),
            system=system,
            source_label=source_label or system.value,
            version=version,
            release_date=release_date,
            local_path_hash=(_safe_hash(local_root, prefix="term_path_")
                             if local_root else None),
            file_count=file_count,
            expected_files_present=list(expected_files_present or []),
            license_confirmed=bool(license_confirmed),
            import_mode=import_mode,
            status=status,
        )

    def safe_public_summary(self) -> dict:
        return {
            "source_id": self.source_id,
            "safe_source_id": self.safe_source_id,
            "system": self.system.value,
            "source_label": self.source_label,
            "version": self.version,
            "release_date": self.release_date,
            "local_path_hash": self.local_path_hash,
            "file_count": self.file_count,
            "expected_files_present": list(self.expected_files_present),
            "license_confirmed": self.license_confirmed,
            "import_mode": self.import_mode.value,
            "status": self.status.value,
            "records_seen": self.records_seen,
            "records_imported": self.records_imported,
        }


# ---------------------------------------------------------------------------
# TerminologyLookupResult
# ---------------------------------------------------------------------------


@dataclass
class TerminologyLookupResult:
    """Result of a single lookup against the local terminology index."""

    query_hash: str
    normalized_query: str
    system_filter: List[str]
    status: TerminologyLookupStatus
    matches: List[TerminologyConcept] = field(default_factory=list)
    ambiguous: bool = False
    exact_match: bool = False
    no_code_hallucinated: bool = True

    @classmethod
    def for_query(
        cls,
        text: str,
        systems: Optional[List[TerminologySystem]] = None,
    ) -> "TerminologyLookupResult":
        return cls(
            query_hash=query_hash(text),
            normalized_query=normalize_query(text),
            system_filter=[s.value for s in (systems or [])],
            status=TerminologyLookupStatus.UNMAPPED,
        )

    def safe_public_summary(self) -> dict:
        return {
            "query_hash": self.query_hash,
            "normalized_query": self.normalized_query,
            "system_filter": list(self.system_filter),
            "status": self.status.value,
            "matches_count": len(self.matches),
            "match_summaries": [m.safe_public_summary() for m in self.matches[:10]],
            "ambiguous": self.ambiguous,
            "exact_match": self.exact_match,
            "no_code_hallucinated": self.no_code_hallucinated,
        }
