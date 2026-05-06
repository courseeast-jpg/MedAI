"""CKA-SEC-02 — backup and rollback policy.

SEC-02 produces *policy* only. It does NOT touch real data, perform
real backups, or perform real restores. The policy fields below define
what SEC-03 (real migration execution) MUST satisfy before any
encrypted-write touches the production MKB store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class BackupRollbackPolicy:
    """Public-report-safe backup + rollback policy state."""

    # Required behaviors for SEC-03 real migration:
    timestamped_backup_required: bool = True
    checksum_before_migration_required: bool = True
    checksum_after_backup_required: bool = True
    restore_test_on_backup_copy_required: bool = True
    rollback_instructions_required: bool = True
    no_deletion_of_original_until_verified: bool = True
    migration_lock_file_required: bool = True

    # SEC-02 dry-run state — these MUST be False in this block:
    real_backup_performed: bool = False
    real_restore_performed: bool = False
    real_db_deleted: bool = False

    # Required artifacts:
    rollback_steps: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rollback_steps:
            self.rollback_steps = [
                "1. Stop all MedAI processes that may write to the main MKB store.",
                "2. Confirm the migration lock file exists and matches this run.",
                "3. Restore the timestamped backup file to the original DB path.",
                "4. Recompute checksum and verify against pre-migration checksum.",
                "5. Restart MedAI in safe-mode and run preflight + B11 release validation.",
                "6. If preflight passes, clear the migration lock; if not, contact responsible authority.",
                "7. Document the rollback in the operator-review log; do NOT delete the encrypted DB.",
            ]

    def safe_public_summary(self) -> dict:
        return {
            "timestamped_backup_required": self.timestamped_backup_required,
            "checksum_before_migration_required": self.checksum_before_migration_required,
            "checksum_after_backup_required": self.checksum_after_backup_required,
            "restore_test_on_backup_copy_required": self.restore_test_on_backup_copy_required,
            "rollback_instructions_required": self.rollback_instructions_required,
            "no_deletion_of_original_until_verified": self.no_deletion_of_original_until_verified,
            "migration_lock_file_required": self.migration_lock_file_required,
            "real_backup_performed": self.real_backup_performed,
            "real_restore_performed": self.real_restore_performed,
            "real_db_deleted": self.real_db_deleted,
            "rollback_steps_count": len(self.rollback_steps),
        }


def get_backup_rollback_policy() -> BackupRollbackPolicy:
    return BackupRollbackPolicy()


def backup_policy_ready(policy: BackupRollbackPolicy = None) -> bool:
    p = policy if policy is not None else get_backup_rollback_policy()
    return all([
        p.timestamped_backup_required,
        p.checksum_before_migration_required,
        p.checksum_after_backup_required,
        not p.real_backup_performed,
    ])


def rollback_policy_ready(policy: BackupRollbackPolicy = None) -> bool:
    p = policy if policy is not None else get_backup_rollback_policy()
    return all([
        p.rollback_instructions_required,
        p.no_deletion_of_original_until_verified,
        p.migration_lock_file_required,
        len(p.rollback_steps) >= 5,
        not p.real_restore_performed,
        not p.real_db_deleted,
    ])
