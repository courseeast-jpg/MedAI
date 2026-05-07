"""CKA-TERM-01 — convenience builders for TerminologySourceManifest.

Thin wrapper around `models.TerminologySourceManifest` used by the
inventory and validation scripts. Kept as a separate module per the
SEC-06/SEC-07 file-structure conventions.
"""
from __future__ import annotations

from typing import List, Optional

from clinical_knowledge.terminology.models import (
    TerminologyImportMode,
    TerminologySourceManifest,
    TerminologySourceStatus,
    TerminologySystem,
)


def build_missing_manifest(
    system: TerminologySystem,
    local_root: str,
) -> TerminologySourceManifest:
    return TerminologySourceManifest.for_local_source(
        system=system,
        local_root=local_root,
        source_label=f"{system.value}_local_root",
        file_count=0,
        expected_files_present=[],
        license_confirmed=False,
        import_mode=TerminologyImportMode.INVENTORY_ONLY,
        status=TerminologySourceStatus.MISSING,
    )


def build_present_manifest(
    system: TerminologySystem,
    local_root: str,
    file_count: int,
    expected_files_present: List[str],
    license_confirmed: bool,
    import_mode: TerminologyImportMode,
    status: TerminologySourceStatus,
    version: Optional[str] = None,
    release_date: Optional[str] = None,
) -> TerminologySourceManifest:
    return TerminologySourceManifest.for_local_source(
        system=system,
        local_root=local_root,
        source_label=f"{system.value}_local_root",
        version=version,
        release_date=release_date,
        file_count=file_count,
        expected_files_present=expected_files_present,
        license_confirmed=license_confirmed,
        import_mode=import_mode,
        status=status,
    )
