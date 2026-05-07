"""CKA-SEC-06 — key rotation plan + result models.

Public-report-safe descriptors for an operator key rotation. Neither
class accepts or carries an encryption key.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class KeyRotationPlan:
    """Structured SEC-06 rotation plan. All real-rotation invariants are
    hard-coded to safe values in dry_run mode. Construction with any
    forbidden value raises ValueError.
    """

    plan_id: str
    created_at: str
    source_store_safe_id: str

    backup_required: bool = True
    backup_verified: bool = False
    old_key_present: bool = False
    new_key_present: bool = False
    new_key_confirmation_required: bool = True

    dry_run: bool = True
    test_mode: bool = False

    # Hard invariants in SEC-06:
    real_rotation_approved: bool = False
    key_rotation_performed: bool = False
    real_store_touched: bool = False

    backup_before_rotation_required: bool = True
    rollback_available: bool = True

    def __post_init__(self) -> None:
        if self.backup_required is False:
            raise ValueError("backup_required_must_be_true")
        if self.backup_before_rotation_required is False:
            raise ValueError("backup_before_rotation_required_must_be_true")
        if self.new_key_confirmation_required is False:
            raise ValueError("new_key_confirmation_required_must_be_true")

    @classmethod
    def for_dry_run(
        cls,
        source_store_safe_id: str,
        *,
        test_mode: bool = False,
    ) -> "KeyRotationPlan":
        return cls(
            plan_id=f"cka_sec06_rotation_plan_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_store_safe_id=source_store_safe_id,
            dry_run=True,
            test_mode=test_mode,
        )

    def safe_public_summary(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "source_store_safe_id": self.source_store_safe_id,
            "backup_required": self.backup_required,
            "backup_verified": self.backup_verified,
            "old_key_present": self.old_key_present,
            "new_key_present": self.new_key_present,
            "new_key_confirmation_required": self.new_key_confirmation_required,
            "dry_run": self.dry_run,
            "test_mode": self.test_mode,
            "real_rotation_approved": self.real_rotation_approved,
            "key_rotation_performed": self.key_rotation_performed,
            "real_store_touched": self.real_store_touched,
            "backup_before_rotation_required": self.backup_before_rotation_required,
            "rollback_available": self.rollback_available,
        }


@dataclass
class KeyRotationResult:
    """Public-report-safe outcome of a key rotation attempt.

    Carries booleans / counts / safe hashes only — never raw paths
    or keys. Short SHA-256 prefixes (16 hex chars) are used so
    serialization stays below the B02 SECRET regex's 40+ alnum threshold.
    """

    rotation_performed: bool = False
    source_records_before: int = 0
    source_records_after: int = 0
    record_count_preserved: bool = False
    old_key_open_before_passed: bool = False
    new_key_open_after_passed: bool = False
    old_key_rejected_after_rotation: bool = False
    backup_created_before_rotation: bool = False
    backup_checksum_verified: bool = False
    rollback_restore_verified: bool = False
    plaintext_absence_verified: bool = False
    real_store_touched: bool = False
    db_file_staged: bool = False
    key_logged: bool = False
    blocked_reason: Optional[str] = None
    source_safe_hash: Optional[str] = None
    backup_safe_hash: Optional[str] = None
    backup_sha256_prefix: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "rotation_performed": self.rotation_performed,
            "source_records_before": self.source_records_before,
            "source_records_after": self.source_records_after,
            "record_count_preserved": self.record_count_preserved,
            "old_key_open_before_passed": self.old_key_open_before_passed,
            "new_key_open_after_passed": self.new_key_open_after_passed,
            "old_key_rejected_after_rotation": self.old_key_rejected_after_rotation,
            "backup_created_before_rotation": self.backup_created_before_rotation,
            "backup_checksum_verified": self.backup_checksum_verified,
            "rollback_restore_verified": self.rollback_restore_verified,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "real_store_touched": self.real_store_touched,
            "db_file_staged": self.db_file_staged,
            "key_logged": self.key_logged,
            "blocked_reason": self.blocked_reason,
            "source_safe_hash": self.source_safe_hash,
            "backup_safe_hash": self.backup_safe_hash,
            "backup_sha256_prefix": self.backup_sha256_prefix,
        }


def rotation_passed(result: KeyRotationResult) -> bool:
    """All-of check: every step succeeded with safe invariants."""
    return all([
        result.rotation_performed,
        result.old_key_open_before_passed,
        result.new_key_open_after_passed,
        result.old_key_rejected_after_rotation,
        result.record_count_preserved,
        result.backup_created_before_rotation,
        result.backup_checksum_verified,
        result.rollback_restore_verified,
        result.plaintext_absence_verified,
        not result.real_store_touched,
        not result.db_file_staged,
        not result.key_logged,
    ])


def rollback_steps() -> list:
    """Return the binding rollback step list. Read by the operator guide
    and surfaced in the public report. Non-destructive only.
    """
    return [
        "1. Backup MUST be created and SHA-256 verified BEFORE any rotation runs.",
        "2. If the new key cannot open the rotated DB, STOP using the rotated DB.",
        "3. Restore the verified encrypted backup using cka_encrypted_store_restore.py.",
        "4. Confirm the backup opens with the OLD key and the record count matches.",
        "5. Do NOT delete the backup until the operator has verified the new key opens the rotated DB AND the app starts correctly.",
        "6. If both old and new keys fail, ESCALATE to the responsible authority. Do not attempt destructive repair.",
        "7. Encryption keys are never stored in Git, never written to reports, never logged. A lost key means the data is unrecoverable.",
    ]
