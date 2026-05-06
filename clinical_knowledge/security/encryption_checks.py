"""CKA-SEC-01 — encryption verification helpers.

These helpers exercise an already-created encrypted database. They
never write the encryption key to disk or to a public report.

Synthetic test strings are intentionally distinctive ASCII so that the
plaintext-absence check cannot be fooled by partial matches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from clinical_knowledge.security.sqlcipher_provider import (
    SQLCipherProviderStatus,
    detect_sqlcipher_provider,
)


# Synthetic strings used by the validation script and tests.
# They are NOT real PHI / NOT real medical data.
SYNTHETIC_FORBIDDEN_STRINGS: tuple = (
    "SYNTHETIC_PRIVATE_NAME_ALPHA",
    "SYNTHETIC_MRN_0001",
    "SYNTHETIC_MEDICAL_NOTE_ALPHA",
)


def verify_cipher_version(connection) -> Optional[str]:
    """Return the SQLCipher cipher_version reported by an open connection.

    Returns None if the pragma is unsupported or returns NULL.
    """
    if connection is None:
        return None
    try:
        cur = connection.execute("PRAGMA cipher_version;")
        row = cur.fetchone()
        if row is None:
            return None
        value = row[0] if isinstance(row, (tuple, list)) else row
        if value is None or str(value).strip() == "":
            return None
        return str(value)
    except Exception:    # noqa: BLE001
        return None


def verify_wrong_key_fails(
    db_path: str,
    wrong_key: str,
    *,
    provider_status: Optional[SQLCipherProviderStatus] = None,
) -> bool:
    """Return True if attempting to open db_path with wrong_key fails to read.

    "Fails to read" means either:
    - opening + selecting raises an exception, OR
    - the cursor returns no rows because the pages cannot be decrypted.

    Returns False if the wrong key actually decrypted the database (i.e.
    encryption did not work) — this is a security failure.

    If no provider is available, this helper returns False since
    encryption cannot be verified; callers should check the provider
    status separately and mark such cases skipped, NOT passed.
    """
    status = provider_status if provider_status is not None else detect_sqlcipher_provider()
    if not status.available or status.module is None:
        return False

    if not isinstance(wrong_key, str) or wrong_key == "":
        # An empty wrong-key is meaningless for this check.
        return False

    con = None
    try:
        con = status.module.connect(db_path)    # type: ignore[union-attr]
        con.execute(f"PRAGMA key = '{wrong_key.replace(chr(39), chr(39)*2)}';")
        # Touch the schema with the wrong key — under SQLCipher this raises.
        cur = con.execute(
            "SELECT count(*) FROM sec01_synthetic_records;"
        )
        rows = cur.fetchall()
        # If we got here without raising, the wrong key decrypted the db.
        # That is a SECURITY FAILURE — return False.
        if rows is None:
            return True
        # An empty count(*) result is impossible without successful decryption.
        return False
    except Exception:    # noqa: BLE001 — any error means the wrong key correctly failed
        return True
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:    # noqa: BLE001
                pass


def verify_plaintext_absent(
    db_path: str,
    forbidden_strings: Iterable[str] = SYNTHETIC_FORBIDDEN_STRINGS,
) -> bool:
    """Scan the raw database file bytes for any forbidden synthetic string.

    Returns True if NONE of the forbidden strings appear in the raw
    bytes (i.e. the file is encrypted as expected).
    Returns False if any forbidden string is visible in plaintext.

    Operates on file bytes only — does not open a database connection
    and never receives or holds the encryption key.
    """
    p = Path(db_path)
    if not p.exists() or not p.is_file():
        # File missing: cannot verify; treat as not-verified (False).
        return False

    raw = p.read_bytes()
    for needle in forbidden_strings:
        if not isinstance(needle, str) or needle == "":
            continue
        if needle.encode("utf-8") in raw:
            return False
        if needle.encode("utf-16-le") in raw:
            return False
    return True
