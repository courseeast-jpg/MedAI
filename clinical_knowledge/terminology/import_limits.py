"""CKA-TERM-01B safe terminology import limits.

These limits are for planning only. Real imports remain disabled by default
and are not implemented in TERM-01B.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminologyImportLimits:
    max_rows_per_file_default: int = 250_000
    max_rows_per_system_default: int = 1_000_000
    max_file_size_mb_default: int = 512
    chunk_size: int = 10_000
    checkpoint_interval_rows: int = 50_000
    require_license_ack_for_real_import: bool = True
    allow_synthetic_test_import: bool = True
    allow_real_import: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "max_rows_per_file_default": self.max_rows_per_file_default,
            "max_rows_per_system_default": self.max_rows_per_system_default,
            "max_file_size_mb_default": self.max_file_size_mb_default,
            "chunk_size": self.chunk_size,
            "checkpoint_interval_rows": self.checkpoint_interval_rows,
            "require_license_ack_for_real_import": self.require_license_ack_for_real_import,
            "allow_synthetic_test_import": self.allow_synthetic_test_import,
            "allow_real_import": self.allow_real_import,
        }


def build_import_limits(
    *,
    max_rows_per_file: int | None = None,
    max_rows_per_system: int | None = None,
    max_file_size_mb: int | None = None,
    chunk_size: int | None = None,
    checkpoint_interval_rows: int | None = None,
    allow_real_import: bool = False,
) -> TerminologyImportLimits:
    defaults = TerminologyImportLimits()
    return TerminologyImportLimits(
        max_rows_per_file_default=max_rows_per_file or defaults.max_rows_per_file_default,
        max_rows_per_system_default=max_rows_per_system or defaults.max_rows_per_system_default,
        max_file_size_mb_default=max_file_size_mb or defaults.max_file_size_mb_default,
        chunk_size=chunk_size or defaults.chunk_size,
        checkpoint_interval_rows=checkpoint_interval_rows or defaults.checkpoint_interval_rows,
        require_license_ack_for_real_import=True,
        allow_synthetic_test_import=True,
        allow_real_import=bool(allow_real_import),
    )
