"""CKA-SEC-01 — security package: SQLCipher encryption readiness.

This package is a *readiness adapter*. It does NOT replace the main
CKA store. It does NOT migrate real data. It does NOT activate
encryption on the production MKB.

Public surface:
- detect_sqlcipher_provider()
- SQLCipherProviderStatus
- EncryptedCKAStore
- verify_wrong_key_fails()
- verify_plaintext_absent()
- verify_cipher_version()
"""
from clinical_knowledge.security.sqlcipher_provider import (
    SQLCipherProviderStatus,
    detect_sqlcipher_provider,
)
from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.encryption_checks import (
    verify_cipher_version,
    verify_plaintext_absent,
    verify_wrong_key_fails,
)
from clinical_knowledge.security.key_policy import (
    KeyPolicyError,
    KeyPolicyStatus,
    get_key_policy_status,
    operator_approval_checklist,
    policy_ready as key_policy_ready,
    validate_operator_key,
)
from clinical_knowledge.security.rollback_plan import (
    BackupRollbackPolicy,
    backup_policy_ready,
    get_backup_rollback_policy,
    rollback_policy_ready,
)
from clinical_knowledge.security.migration_plan import (
    InventoryResult,
    MigrationPlan,
    MigrationRehearsalResult,
    inventory_candidate_db_files,
)
from clinical_knowledge.security.migration_rehearsal import (
    rehearse_synthetic_migration,
    rehearsal_passed,
)

__all__ = [
    # SEC-01
    "SQLCipherProviderStatus",
    "detect_sqlcipher_provider",
    "EncryptedCKAStore",
    "EncryptedStoreError",
    "verify_cipher_version",
    "verify_plaintext_absent",
    "verify_wrong_key_fails",
    # SEC-02 — key policy
    "KeyPolicyError",
    "KeyPolicyStatus",
    "get_key_policy_status",
    "operator_approval_checklist",
    "key_policy_ready",
    "validate_operator_key",
    # SEC-02 — backup / rollback
    "BackupRollbackPolicy",
    "backup_policy_ready",
    "get_backup_rollback_policy",
    "rollback_policy_ready",
    # SEC-02 — migration plan / inventory / rehearsal
    "InventoryResult",
    "MigrationPlan",
    "MigrationRehearsalResult",
    "inventory_candidate_db_files",
    "rehearse_synthetic_migration",
    "rehearsal_passed",
]
