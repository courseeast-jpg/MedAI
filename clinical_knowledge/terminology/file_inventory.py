"""CKA-TERM-01 — bounded local terminology-data inventory.

Scans only the operator's local `terminology_data/<system>/` directories
(or a path supplied by the caller). Never scans the whole drive,
never opens file contents during inventory, returns only safe hashes
and counts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from clinical_knowledge.terminology.license_gate import (
    license_acknowledged_for,
)
from clinical_knowledge.terminology.models import (
    TerminologyImportMode,
    TerminologySourceManifest,
    TerminologySourceStatus,
    TerminologySystem,
)
from clinical_knowledge.terminology.source_manifest import (
    build_missing_manifest,
    build_present_manifest,
)


_DEFAULT_TERMINOLOGY_ROOT = "terminology_data"


# Heuristic file-name fragments per system. Inventory looks at file NAMES
# only; never opens the file content.
_EXPECTED_FILES: Dict[TerminologySystem, List[str]] = {
    TerminologySystem.UMLS: [
        "MRCONSO.RRF",
        "MRSTY.RRF",
    ],
    TerminologySystem.SNOMED_CT: [
        "sct2_Concept",
        "sct2_Description",
    ],
    TerminologySystem.RXNORM: [
        "RXNCONSO.RRF",
    ],
    TerminologySystem.LOINC: [
        "Loinc.csv",
        "LoincTable.csv",
    ],
}

_OPTIONAL_FILES: Dict[TerminologySystem, List[str]] = {
    TerminologySystem.UMLS: ["MRREL.RRF"],
    TerminologySystem.SNOMED_CT: ["sct2_Relationship"],
    TerminologySystem.RXNORM: ["RXNREL.RRF"],
    TerminologySystem.LOINC: [],
}


@dataclass
class InventoryReport:
    """Public-report-safe inventory result."""

    terminology_root_present: bool = False
    terminology_root_safe_hash: Optional[str] = None
    sources: List[TerminologySourceManifest] = field(default_factory=list)
    files_seen_total: int = 0
    no_real_data_committed: bool = True
    raw_paths_written_to_public_report: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "terminology_root_present": self.terminology_root_present,
            "terminology_root_safe_hash": self.terminology_root_safe_hash,
            "files_seen_total": self.files_seen_total,
            "no_real_data_committed": self.no_real_data_committed,
            "raw_paths_written_to_public_report": self.raw_paths_written_to_public_report,
            "sources": [m.safe_public_summary() for m in self.sources],
            "sources_count": len(self.sources),
        }


def _safe_root_hash(p: str) -> str:
    digest = hashlib.sha256(f"cka_term01_root:{p}".encode("utf-8")).hexdigest()[:16]
    return f"term_root_{digest}"


def _system_root(repo_root: Path, system: TerminologySystem) -> Path:
    return repo_root / _DEFAULT_TERMINOLOGY_ROOT / system.value


def _filenames_present(dir_path: Path) -> List[str]:
    """List file names only (no full paths) inside `dir_path`."""
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    names: List[str] = []
    try:
        for child in sorted(dir_path.iterdir()):
            if child.is_file():
                names.append(child.name)
    except OSError:
        return []
    return names


def _which_expected_present(
    names: List[str],
    expected_fragments: List[str],
) -> List[str]:
    """Return the expected fragments that match at least one filename.

    Matching is a case-insensitive substring check on the filename.
    Fragment names like 'sct2_Concept' may appear within longer
    SNOMED file names (e.g. 'sct2_Concept_Snapshot_INT_20240101.txt').
    """
    found: List[str] = []
    lower_names = [n.lower() for n in names]
    for frag in expected_fragments:
        f = frag.lower()
        if any(f in nm for nm in lower_names):
            found.append(frag)
    return found


def inventory_terminology_data_dir(
    repo_root: Optional[Path] = None,
    *,
    license_test_mode: bool = False,
    license_env: Optional[dict] = None,
) -> InventoryReport:
    """Scan the bounded `terminology_data/` tree and produce an
    InventoryReport. Never reads file contents; never returns raw paths.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    root = repo_root / _DEFAULT_TERMINOLOGY_ROOT
    report = InventoryReport(
        terminology_root_present=root.exists(),
        terminology_root_safe_hash=_safe_root_hash(str(root)),
    )

    total_seen = 0

    for system in (
        TerminologySystem.UMLS,
        TerminologySystem.SNOMED_CT,
        TerminologySystem.RXNORM,
        TerminologySystem.LOINC,
    ):
        sys_root = _system_root(repo_root, system)
        names = _filenames_present(sys_root)
        # Any file that is not the LICENSE_ACK_PRIVATE.json sibling counts.
        names = [n for n in names if n != "LICENSE_ACK_PRIVATE.json"]
        total_seen += len(names)

        if not names:
            report.sources.append(build_missing_manifest(system, str(sys_root)))
            continue

        expected = _EXPECTED_FILES.get(system, [])
        found = _which_expected_present(names, expected)

        license_ok = license_acknowledged_for(
            system,
            env=license_env,
            test_mode=license_test_mode,
        )

        # Determine status:
        if not found:
            status = TerminologySourceStatus.PRESENT_UNVERIFIED
            mode = TerminologyImportMode.INVENTORY_ONLY
        elif not license_ok:
            status = TerminologySourceStatus.LICENSE_REQUIRED
            mode = TerminologyImportMode.INVENTORY_ONLY
        else:
            status = TerminologySourceStatus.IMPORT_READY
            mode = TerminologyImportMode.LICENSED_LOCAL_IMPORT

        manifest = build_present_manifest(
            system=system,
            local_root=str(sys_root),
            file_count=len(names),
            expected_files_present=found,
            license_confirmed=license_ok,
            import_mode=mode,
            status=status,
        )
        report.sources.append(manifest)

    report.files_seen_total = total_seen
    return report
