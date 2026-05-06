"""CKA-SEC-01 — SQLCipher provider detection.

Tries providers in safe order:
  1. sqlcipher3 (if importable)
  2. pysqlcipher3 (if importable)
  3. stdlib sqlite3 ONLY if `PRAGMA cipher_version;` returns a value
     (i.e. only if the linked SQLite has SQLCipher built in).

Detection never:
- installs packages
- raises on missing modules
- writes the cipher key anywhere
- exposes local paths in its public summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


_KNOWN_SQLCIPHER_MODULES = ("sqlcipher3", "pysqlcipher3.dbapi2", "pysqlcipher3")


@dataclass
class SQLCipherProviderStatus:
    """Result of detect_sqlcipher_provider(). Public-report safe."""

    provider_name: Optional[str] = None
    available: bool = False
    cipher_version: Optional[str] = None
    import_error_safe: Optional[str] = None    # type-name only, no path/value
    notes: str = ""

    # Internal-only handle to the importable module (NOT in public summary).
    module: Optional[object] = field(default=None, repr=False)

    def safe_public_summary(self) -> dict:
        """Public-report-safe representation.

        Excludes the resolved module object and any path-like data.
        """
        return {
            "provider_name": self.provider_name,
            "available": bool(self.available),
            "cipher_version": self.cipher_version,
            "import_error_safe": self.import_error_safe,
            "notes": self.notes,
        }


def _try_import(module_name: str) -> tuple[Optional[object], Optional[str]]:
    """Import a module without raising; return (module, error-name-or-None)."""
    try:
        mod = __import__(module_name, fromlist=["*"])
        return mod, None
    except Exception as exc:    # noqa: BLE001 — defensive, no provider should crash detection
        return None, type(exc).__name__


def _probe_cipher_version(module: object) -> Optional[str]:
    """Open an in-memory connection and read PRAGMA cipher_version.

    Never logs the cipher key. Returns None if the pragma is unsupported
    or returns NULL (which means the module was not linked against
    SQLCipher).
    """
    connect = getattr(module, "connect", None)
    if connect is None:
        return None
    con = None
    try:
        con = connect(":memory:")
        cur = con.execute("PRAGMA cipher_version;")
        row = cur.fetchone()
        if row is None:
            return None
        value = row[0] if isinstance(row, (tuple, list)) else row
        if value is None or str(value).strip() == "":
            return None
        return str(value)
    except Exception:    # noqa: BLE001
        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:    # noqa: BLE001
                pass


def detect_sqlcipher_provider() -> SQLCipherProviderStatus:
    """Detect a SQLCipher-capable Python provider in safe order."""
    # 1. Dedicated SQLCipher providers.
    for name in _KNOWN_SQLCIPHER_MODULES:
        mod, err = _try_import(name)
        if mod is None:
            continue
        version = _probe_cipher_version(mod)
        if version:
            return SQLCipherProviderStatus(
                provider_name=name,
                available=True,
                cipher_version=version,
                import_error_safe=None,
                notes="dedicated_sqlcipher_provider",
                module=mod,
            )
        # Imported but cipher pragma did not respond: do NOT claim available.
        return SQLCipherProviderStatus(
            provider_name=name,
            available=False,
            cipher_version=None,
            import_error_safe=None,
            notes="imported_but_no_cipher_version",
            module=mod,
        )

    # 2. Stdlib sqlite3 — only acceptable if SQLCipher is linked in.
    import sqlite3 as _sqlite3
    version = _probe_cipher_version(_sqlite3)
    if version:
        return SQLCipherProviderStatus(
            provider_name="sqlite3",
            available=True,
            cipher_version=version,
            import_error_safe=None,
            notes="stdlib_sqlite3_with_sqlcipher_linkage",
            module=_sqlite3,
        )

    # 3. No SQLCipher available.
    return SQLCipherProviderStatus(
        provider_name=None,
        available=False,
        cipher_version=None,
        import_error_safe="ModuleNotFoundError",
        notes="no_sqlcipher_provider_present",
        module=None,
    )
