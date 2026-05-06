"""CKA-SEC-02 — migration plan object + read-only inventory.

This module produces:

- `inventory_candidate_db_files()` — read-only scan over a small set of
  *known-safe* project directories. Does NOT scan the entire drive,
  does NOT open files, returns only safe hashes.
- `MigrationPlan` — structured plan object with all SEC-02 invariants
  hard-coded to safe values (rehearsal_only, real_data_migrated=False).
- `MigrationRehearsalResult` — populated by migration_rehearsal.py

No real migration is performed in this module.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Safe constants
# ---------------------------------------------------------------------------

# Project-safe scan roots, expressed as paths relative to the repo root.
# We do NOT scan the whole drive; we do NOT scan the user's home dir.
_DEFAULT_SCAN_DIRS: tuple = (
    "data",
    "clinical_knowledge",
    "reports",
)

# Db-shaped suffixes we treat as candidate stores. We DO NOT open them.
_DB_SUFFIXES: tuple = (".db", ".sqlite", ".sqlite3", ".cipher", ".cka")

_HASH_SALT = "cka_sec02_inventory_v1"


def _safe_hash(raw: str) -> str:
    """Return a safe public hash for any path-like string. Never reveals the path."""
    digest = hashlib.sha256(f"{_HASH_SALT}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"cka_db_{digest}"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class InventoryResult:
    """Public-report-safe inventory result. NEVER contains raw paths."""

    candidate_db_count: int = 0
    candidate_db_safe_hashes: List[str] = field(default_factory=list)
    likely_main_store_found: bool = False
    real_main_store_touched: bool = False
    raw_paths_written_to_public_report: bool = False
    scan_dirs_relative: List[str] = field(default_factory=list)

    def safe_public_summary(self) -> dict:
        return {
            "candidate_db_count": self.candidate_db_count,
            "candidate_db_safe_hashes": list(self.candidate_db_safe_hashes),
            "likely_main_store_found": self.likely_main_store_found,
            "real_main_store_touched": self.real_main_store_touched,
            "raw_paths_written_to_public_report": self.raw_paths_written_to_public_report,
            "scan_dirs_relative": list(self.scan_dirs_relative),
        }


def inventory_candidate_db_files(
    repo_root: Optional[Path] = None,
    scan_dirs: Optional[Iterable[str]] = None,
) -> InventoryResult:
    """Read-only scan for candidate CKA/MKB DB files.

    - Only scans `_DEFAULT_SCAN_DIRS` (or the override provided).
    - Only collects files; does NOT open or modify any DB.
    - Returns SAFE HASHES; never returns or logs raw paths.
    - Does NOT mark any file as the main store unless it is named
      exactly like the project's known main MKB store basename.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    dirs = list(scan_dirs) if scan_dirs else list(_DEFAULT_SCAN_DIRS)

    safe_hashes: List[str] = []
    likely_main = False
    for rel in dirs:
        d = (repo_root / rel)
        if not d.exists() or not d.is_dir():
            continue
        # Use rglob bounded to the scan root only.
        for p in sorted(d.rglob("*")):
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            suffix = p.suffix.lower()
            if suffix not in _DB_SUFFIXES:
                continue
            # Skip anything in temp / venv / cache to avoid noise.
            parts_lower = {part.lower() for part in p.parts}
            if any(s in parts_lower for s in ("__pycache__", ".venv", "venv", ".git", ".tox")):
                continue
            # Hash the relative path so the report carries no raw path.
            try:
                rel_str = p.relative_to(repo_root).as_posix()
            except ValueError:
                rel_str = p.name
            safe_hashes.append(_safe_hash(rel_str))
            # The "main" store would canonically be a top-level mkb*.db / cka*.db
            # file directly under data/. We use a name-only heuristic so we
            # never have to OPEN the file to classify it.
            name_lower = p.name.lower()
            if name_lower.startswith(("mkb", "cka")) and rel_str.startswith("data/"):
                likely_main = True

    return InventoryResult(
        candidate_db_count=len(safe_hashes),
        candidate_db_safe_hashes=safe_hashes,
        likely_main_store_found=likely_main,
        real_main_store_touched=False,
        raw_paths_written_to_public_report=False,
        scan_dirs_relative=dirs,
    )


# ---------------------------------------------------------------------------
# MigrationPlan
# ---------------------------------------------------------------------------

@dataclass
class MigrationPlan:
    """Structured SEC-02 migration plan. All real-migration flags are
    hard-coded to False. Construction with any True value raises."""

    plan_id: str
    created_at: str
    source_store_safe_id: str
    target_store_safe_id: str
    sqlcipher_provider: Optional[str]
    provider_version: Optional[str]
    migration_mode: str = "rehearsal_only"
    real_migration_approved: bool = False
    backup_required: bool = True
    rollback_required: bool = True
    operator_approval_required: bool = True
    main_store_migration_performed: bool = False
    real_data_migrated: bool = False

    def __post_init__(self) -> None:
        if self.migration_mode != "rehearsal_only":
            raise ValueError("migration_mode_must_be_rehearsal_only_in_sec02")
        if self.real_migration_approved is True:
            raise ValueError("real_migration_approved_not_allowed_in_sec02")
        if self.main_store_migration_performed is True:
            raise ValueError("main_store_migration_performed_not_allowed_in_sec02")
        if self.real_data_migrated is True:
            raise ValueError("real_data_migrated_not_allowed_in_sec02")

    @classmethod
    def for_rehearsal(
        cls,
        source_store_safe_id: str,
        target_store_safe_id: str,
        sqlcipher_provider: Optional[str],
        provider_version: Optional[str],
    ) -> "MigrationPlan":
        return cls(
            plan_id=f"cka_sec02_plan_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_store_safe_id=source_store_safe_id,
            target_store_safe_id=target_store_safe_id,
            sqlcipher_provider=sqlcipher_provider,
            provider_version=provider_version,
        )

    def safe_public_summary(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "source_store_safe_id": self.source_store_safe_id,
            "target_store_safe_id": self.target_store_safe_id,
            "sqlcipher_provider": self.sqlcipher_provider,
            "provider_version": self.provider_version,
            "migration_mode": self.migration_mode,
            "real_migration_approved": self.real_migration_approved,
            "backup_required": self.backup_required,
            "rollback_required": self.rollback_required,
            "operator_approval_required": self.operator_approval_required,
            "main_store_migration_performed": self.main_store_migration_performed,
            "real_data_migrated": self.real_data_migrated,
        }


@dataclass
class MigrationRehearsalResult:
    """Public-report-safe result of the synthetic migration rehearsal.

    Carries booleans / counts only — never raw paths or keys.
    """

    rehearsal_performed: bool = False
    synthetic_source_created: bool = False
    encrypted_target_created: bool = False
    records_copied: int = 0
    correct_key_read_passed: bool = False
    wrong_key_failed: bool = False
    plaintext_absence_verified: bool = False
    source_unchanged: bool = False
    temp_files_staged: bool = False

    # Internal: source SHA-256 of the synthetic plaintext DB *before*
    # rehearsal, used to prove source_unchanged. NOT in safe_public_summary.
    _source_pre_hash: Optional[str] = field(default=None, repr=False)

    def safe_public_summary(self) -> dict:
        return {
            "rehearsal_performed": self.rehearsal_performed,
            "synthetic_source_created": self.synthetic_source_created,
            "encrypted_target_created": self.encrypted_target_created,
            "records_copied": self.records_copied,
            "correct_key_read_passed": self.correct_key_read_passed,
            "wrong_key_failed": self.wrong_key_failed,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "source_unchanged": self.source_unchanged,
            "temp_files_staged": self.temp_files_staged,
        }
