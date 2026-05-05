"""Safe ID helpers for CKA — deterministic hashing, never exposes raw paths/names.

All public report payloads must use safe IDs, not raw filenames or patient refs.
"""
from __future__ import annotations

import hashlib
import uuid


_HASH_PREFIX = "cka_"
_SALT = "medai_cka_b01_safe_id_v1"


def hash_source_ref(raw_ref: str) -> str:
    """Return a stable hex digest for a raw source reference (filename, path, etc.).

    The raw_ref is never stored or logged in public reports.
    """
    digest = hashlib.sha256(f"{_SALT}:{raw_ref}".encode()).hexdigest()[:16]
    return f"{_HASH_PREFIX}src_{digest}"


def make_safe_record_id(record_id: str) -> str:
    """Derive a safe public ID from an internal record_id."""
    digest = hashlib.sha256(f"{_SALT}:rec:{record_id}".encode()).hexdigest()[:16]
    return f"{_HASH_PREFIX}rec_{digest}"


def new_record_id() -> str:
    """Generate a fresh unique internal record ID (UUID4)."""
    return str(uuid.uuid4())


def new_event_id() -> str:
    """Generate a fresh unique ledger event ID."""
    return str(uuid.uuid4())


_PHI_PATTERNS = (
    # Common PHI-like patterns
    "patient", "dob", "date_of_birth", "ssn", "mrn", "medical_record",
    "first_name", "last_name", "full_name", "phone", "address",
    "date of birth", "social security",
)


def contains_phi_like(text: str) -> bool:
    """Heuristic check: does a string contain patterns that resemble PHI field names?"""
    lower = text.lower()
    return any(p in lower for p in _PHI_PATTERNS)
