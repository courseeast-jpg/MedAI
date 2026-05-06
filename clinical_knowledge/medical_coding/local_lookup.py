"""Local lookup terminology source for CKA-B07.

Loads a local JSON file at construction time. No network calls.
No bundled real licensed terminology data.
Test entries must have synthetic=true.
Safe default when file absent or invalid: status=source_unavailable (no crash).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinical_knowledge.medical_coding.models import (
    CodingResult,
    CodingStatus,
    CodingSystem,
    MedicalCode,
    TerminologySourceStatus,
)
from clinical_knowledge.medical_coding.terminology_source import TerminologySource

_SALT = "medai_cka_b07_lookup_v1"
_ALLOWED_SYSTEMS = {s.value for s in CodingSystem}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _safe_path_hash(path: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:path:{path}".encode()).hexdigest()[:16]
    return f"cka_lkp_{digest}"


def load_local_lookup(path: str) -> List[Dict[str, Any]]:
    """Load entries from a local JSON lookup file.

    Expected format:
        {
          "entries": [
            {
              "normalized_text": "synthetic condition local",
              "fact_type": "diagnosis",
              "system": "snomed_ct",
              "code": "SYNTHETIC-SNOMED-001",
              "display": "Synthetic condition local",
              "version": "test-only",
              "synthetic": true
            }
          ]
        }

    Returns list of valid entries. Invalid / non-synthetic entries silently skipped.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        return []

    valid: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        norm = entry.get("normalized_text", "")
        code = entry.get("code", "")
        system = entry.get("system", "")
        if not norm or not code or not system:
            continue
        if system not in _ALLOWED_SYSTEMS:
            continue
        synthetic = bool(entry.get("synthetic", False))
        if not synthetic:
            # Non-synthetic entries from unverified sources rejected
            continue
        confidence = entry.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        valid.append({
            "normalized_text": _normalize(norm),
            "fact_type": entry.get("fact_type", ""),
            "system": system,
            "code": code,
            "display": entry.get("display", code),
            "version": entry.get("version", ""),
            "synthetic": True,
            "preferred": bool(entry.get("preferred", False)),
            "confidence": confidence,
        })
    return valid


class LocalLookupTerminologySource(TerminologySource):
    """Local JSON file-backed terminology source.

    If the file is absent or invalid, status=source_unavailable (no crash).
    All loaded entries must be synthetic=True.
    The file path is never exposed in public reports — only its hash.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._safe_path_hash = _safe_path_hash(path)
        self._file_exists = Path(path).exists()
        try:
            self._entries = load_local_lookup(path)
        except Exception:
            self._entries = []

    @property
    def name(self) -> str:
        # Safe name: hash only, never raw path
        return f"local_lookup:{self._safe_path_hash}"

    def status(self) -> TerminologySourceStatus:
        if not self._file_exists or not self._entries:
            return TerminologySourceStatus.UNAVAILABLE
        return TerminologySourceStatus.LOCAL_LOOKUP_ONLY

    def lookup(
        self,
        normalized_text: str,
        fact_type: Optional[str] = None,
        specialty: Optional[str] = None,
    ) -> CodingResult:
        if not self._file_exists or not self._entries:
            return CodingResult(
                candidate_safe_id="",
                status=CodingStatus.SOURCE_UNAVAILABLE,
                codes=[],
                preferred_code=None,
                ambiguity_count=0,
                confidence=0.0,
                terminology_source_status=TerminologySourceStatus.UNAVAILABLE,
                explanation="Local lookup source unavailable.",
                no_code_hallucinated=True,
            )

        key = _normalize(normalized_text)
        matches = [e for e in self._entries if e["normalized_text"] == key]

        if not matches:
            return CodingResult(
                candidate_safe_id="",
                status=CodingStatus.UNMAPPED,
                codes=[],
                preferred_code=None,
                ambiguity_count=0,
                confidence=0.0,
                terminology_source_status=TerminologySourceStatus.LOCAL_LOOKUP_ONLY,
                explanation="Local lookup: no entry for normalized text.",
                no_code_hallucinated=True,
            )

        codes: List[MedicalCode] = []
        for m in matches:
            try:
                sys_enum = CodingSystem(m["system"])
            except ValueError:
                sys_enum = CodingSystem.UNKNOWN
            codes.append(MedicalCode(
                system=sys_enum,
                code=m["code"],
                display=m["display"],
                version=m["version"],
                source=f"local_lookup:{self._safe_path_hash}",  # path hash only
                synthetic=True,
                confidence=m["confidence"],
            ))

        preferred_matches = [m for m in matches if m.get("preferred")]
        if len(preferred_matches) == 1:
            idx = matches.index(preferred_matches[0])
            preferred = codes[idx]
            status = CodingStatus.CODED
            explanation = "Local lookup: preferred entry matched."
        elif len(codes) == 1:
            preferred = codes[0]
            status = CodingStatus.CODED
            explanation = "Local lookup: single match found."
        else:
            preferred = None
            status = CodingStatus.AMBIGUOUS
            explanation = "Local lookup: ambiguous — multiple entries, no preferred flag."

        return CodingResult(
            candidate_safe_id="",
            status=status,
            codes=codes,
            preferred_code=preferred,
            ambiguity_count=len(codes) if status == CodingStatus.AMBIGUOUS else 0,
            confidence=codes[0].confidence if codes else 0.0,
            terminology_source_status=TerminologySourceStatus.LOCAL_LOOKUP_ONLY,
            explanation=explanation,
            no_code_hallucinated=True,
        )
