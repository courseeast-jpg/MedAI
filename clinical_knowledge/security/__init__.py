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

__all__ = [
    "SQLCipherProviderStatus",
    "detect_sqlcipher_provider",
    "EncryptedCKAStore",
    "EncryptedStoreError",
    "verify_cipher_version",
    "verify_plaintext_absent",
    "verify_wrong_key_fails",
]
