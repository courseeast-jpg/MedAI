# MEDAI-DB-REPAIR-02 SQLCipher Row Factory Compatibility

This block validates that MKB row handling works with sqlite3 and SQLCipher backends.

The fix uses a backend-neutral dictionary row factory and keeps aggregate count queries row-type agnostic.
