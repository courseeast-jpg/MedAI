"""CKA-TERM-01A — operator intake automation helpers.

Composes:
- folder preparation under `terminology_data/`
- license-ack TEMPLATE creation (never the real ack file)
- bounded local file scan (off by default, opt-in path)
- safe copy / extract helpers, with zip-slip protection
- readiness summary (combines inventory + license-gate state)

Hard rules:
- NEVER downloads anything.
- NEVER creates a real LICENSE_ACK_PRIVATE.json.
- NEVER stages files under terminology_data/.
- NEVER copies files outside terminology_data/<system>/.
- Zip extraction (when explicitly requested) refuses any entry whose
  resolved path escapes terminology_data/<system>/ (zip-slip
  protection).
- Public summaries carry counts and safe labels only — never raw
  paths.
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clinical_knowledge.terminology.ack_template import (
    TemplateWriteResult,
    write_ack_template,
)
from clinical_knowledge.terminology.file_classifier import (
    ClassificationSummary,
    FileClassification,
    classify_filename,
    classify_filenames,
)
from clinical_knowledge.terminology.file_inventory import (
    InventoryReport,
    inventory_terminology_data_dir,
)
from clinical_knowledge.terminology.license_gate import (
    license_acknowledged_for,
)
from clinical_knowledge.terminology.models import (
    TerminologyImportMode,
    TerminologySourceStatus,
    TerminologySystem,
)


_DEFAULT_TERMINOLOGY_ROOT = "terminology_data"
_SYSTEM_SUBDIRS: Tuple[TerminologySystem, ...] = (
    TerminologySystem.LOINC,
    TerminologySystem.RXNORM,
    TerminologySystem.UMLS,
    TerminologySystem.SNOMED_CT,
)


def _safe_root_hash(p: Path) -> str:
    digest = hashlib.sha256(
        f"cka_term01a_root:{p}".encode("utf-8")
    ).hexdigest()[:16]
    return f"term_root_{digest}"


# ---------------------------------------------------------------------------
# Folder preparation + ack template
# ---------------------------------------------------------------------------


@dataclass
class FolderPreparationResult:
    """Public-report-safe folder-preparation outcome."""

    root_present: bool = False
    root_safe_hash: Optional[str] = None
    subdirs_created: List[str] = field(default_factory=list)
    subdirs_already_present: List[str] = field(default_factory=list)
    template: TemplateWriteResult = field(default_factory=TemplateWriteResult)

    def safe_public_summary(self) -> dict:
        return {
            "root_present": self.root_present,
            "root_safe_hash": self.root_safe_hash,
            "subdirs_created": list(self.subdirs_created),
            "subdirs_already_present": list(self.subdirs_already_present),
            "template": self.template.safe_public_summary(),
        }


def prepare_intake_folders(
    repo_root: Optional[Path] = None,
    *,
    write_template: bool = True,
    overwrite_template: bool = False,
) -> FolderPreparationResult:
    """Create the gitignored `terminology_data/` tree if missing and
    drop a license-ack TEMPLATE next to it. Never creates a real ack.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    root = repo_root / _DEFAULT_TERMINOLOGY_ROOT

    result = FolderPreparationResult(
        root_present=False,
        root_safe_hash=_safe_root_hash(root),
    )

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    result.root_present = root.exists()

    for system in _SYSTEM_SUBDIRS:
        sub = root / system.value
        if sub.exists():
            result.subdirs_already_present.append(system.value)
        else:
            sub.mkdir(parents=True, exist_ok=True)
            result.subdirs_created.append(system.value)

    if write_template:
        result.template = write_ack_template(
            root, overwrite_template=overwrite_template,
        )
    return result


# ---------------------------------------------------------------------------
# Optional bounded local scan (off by default)
# ---------------------------------------------------------------------------


@dataclass
class LocalScanResult:
    """Public-report-safe optional-scan outcome."""

    scanned: bool = False
    scan_root_safe_hash: Optional[str] = None
    files_seen: int = 0
    summary: Optional[ClassificationSummary] = None

    def safe_public_summary(self) -> dict:
        s: Dict[str, object] = {
            "scanned": self.scanned,
            "scan_root_safe_hash": self.scan_root_safe_hash,
            "files_seen": self.files_seen,
        }
        if self.summary is not None:
            s["summary"] = self.summary.safe_public_summary()
        else:
            s["summary"] = {
                "counts_by_system": {}, "zip_counts_by_system": {},
                "unknown_count": 0, "total": 0,
            }
        return s


def optional_local_scan(
    scan_dir: Optional[Path] = None,
    *,
    enabled: bool = False,
    recurse: bool = False,
    max_files: int = 5000,
) -> LocalScanResult:
    """Scan a user-provided folder for terminology-shaped filenames.

    Off by default (`enabled=False`). When enabled, only the immediate
    directory is scanned unless `recurse=True`. File contents are never
    read; only filenames. Returns counts/labels only.
    """
    result = LocalScanResult()
    if not enabled or scan_dir is None:
        return result
    p = Path(scan_dir)
    if not p.exists() or not p.is_dir():
        return result

    result.scanned = True
    result.scan_root_safe_hash = _safe_root_hash(p)

    names: List[str] = []
    if recurse:
        for child in p.rglob("*"):
            if not child.is_file():
                continue
            names.append(child.name)
            if len(names) >= max_files:
                break
    else:
        try:
            for child in p.iterdir():
                if child.is_file():
                    names.append(child.name)
                    if len(names) >= max_files:
                        break
        except OSError:
            pass

    result.files_seen = len(names)
    result.summary = classify_filenames(names)
    return result


# ---------------------------------------------------------------------------
# Safe copy / extract helpers
# ---------------------------------------------------------------------------


@dataclass
class CopyResult:
    """Public-report-safe copy outcome."""

    copy_approved: bool = False
    files_copied: int = 0
    files_skipped_unknown: int = 0
    files_skipped_outside_terminology_data: int = 0

    def safe_public_summary(self) -> dict:
        return {
            "copy_approved": self.copy_approved,
            "files_copied": self.files_copied,
            "files_skipped_unknown": self.files_skipped_unknown,
            "files_skipped_outside_terminology_data": self.files_skipped_outside_terminology_data,
        }


def _terminology_root(repo_root: Path) -> Path:
    return (repo_root / _DEFAULT_TERMINOLOGY_ROOT).resolve()


def _is_under(child: Path, parent: Path) -> bool:
    """True if `child` resolves under `parent` (post-resolve)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def copy_classified_files(
    source_files: List[Path],
    *,
    repo_root: Optional[Path] = None,
    copy_approved: bool = False,
) -> CopyResult:
    """Copy known terminology-shaped files into terminology_data/<system>/.

    Refuses unless `copy_approved=True`. Refuses to write anywhere
    outside the resolved `terminology_data/` tree. ZIPs are classified
    but NOT extracted by this helper — see `safe_extract_zip`.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    root = _terminology_root(repo_root)

    result = CopyResult(copy_approved=bool(copy_approved))
    if not copy_approved:
        return result

    for src in source_files:
        src = Path(src)
        if not src.exists() or not src.is_file():
            continue
        c = classify_filename(src.name)
        if c.system is None:
            result.files_skipped_unknown += 1
            continue
        # Choose destination under terminology_data/<system>/.
        dest_dir = root / c.system.value
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        if not _is_under(dest, root):
            result.files_skipped_outside_terminology_data += 1
            continue
        try:
            shutil.copyfile(str(src), str(dest))
            result.files_copied += 1
        except OSError:
            result.files_skipped_unknown += 1
    return result


@dataclass
class ExtractResult:
    """Public-report-safe extraction outcome."""

    extract_approved: bool = False
    archives_attempted: int = 0
    archives_extracted: int = 0
    entries_extracted: int = 0
    entries_blocked_zip_slip: int = 0
    entries_blocked_outside_terminology_data: int = 0
    archives_blocked_unknown_system: int = 0

    def safe_public_summary(self) -> dict:
        return {
            "extract_approved": self.extract_approved,
            "archives_attempted": self.archives_attempted,
            "archives_extracted": self.archives_extracted,
            "entries_extracted": self.entries_extracted,
            "entries_blocked_zip_slip": self.entries_blocked_zip_slip,
            "entries_blocked_outside_terminology_data": self.entries_blocked_outside_terminology_data,
            "archives_blocked_unknown_system": self.archives_blocked_unknown_system,
        }


def safe_extract_zip(
    archives: List[Path],
    *,
    repo_root: Optional[Path] = None,
    extract_approved: bool = False,
) -> ExtractResult:
    """Extract recognized ZIP archives into terminology_data/<system>/.

    Defaults to a no-op (`extract_approved=False`). Each entry's
    resolved destination is checked to be under the system subdir;
    any entry that would escape (zip-slip) is counted as blocked
    and skipped. Refuses to extract archives whose system cannot be
    classified.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    root = _terminology_root(repo_root)

    result = ExtractResult(extract_approved=bool(extract_approved))
    if not extract_approved:
        return result

    for arc_path in archives:
        arc = Path(arc_path)
        if not arc.exists() or not arc.is_file():
            continue
        result.archives_attempted += 1
        c = classify_filename(arc.name)
        if c.system is None or not c.is_zip:
            result.archives_blocked_unknown_system += 1
            continue

        dest_root = (root / c.system.value).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        archive_extracted_any = False
        try:
            with zipfile.ZipFile(arc, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    raw_name = member.filename
                    # Reject absolute paths, drive letters, and parent
                    # traversal in the ZIP entry name.
                    if raw_name.startswith("/") or raw_name.startswith("\\"):
                        result.entries_blocked_zip_slip += 1
                        continue
                    if ".." in Path(raw_name).parts:
                        result.entries_blocked_zip_slip += 1
                        continue
                    if len(raw_name) >= 2 and raw_name[1] == ":":
                        result.entries_blocked_zip_slip += 1
                        continue

                    candidate = (dest_root / raw_name).resolve()
                    try:
                        candidate.relative_to(dest_root)
                    except ValueError:
                        result.entries_blocked_outside_terminology_data += 1
                        continue

                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(candidate, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    result.entries_extracted += 1
                    archive_extracted_any = True
        except (zipfile.BadZipFile, OSError):
            continue

        if archive_extracted_any:
            result.archives_extracted += 1
    return result


# ---------------------------------------------------------------------------
# Readiness checker
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    """Public-report-safe readiness summary."""

    systems_present: List[str] = field(default_factory=list)
    systems_acknowledged: List[str] = field(default_factory=list)
    systems_import_ready: List[str] = field(default_factory=list)
    systems_license_required: List[str] = field(default_factory=list)
    systems_missing: List[str] = field(default_factory=list)
    pending_acknowledgments: List[str] = field(default_factory=list)

    def safe_public_summary(self) -> dict:
        return {
            "systems_present": list(self.systems_present),
            "systems_acknowledged": list(self.systems_acknowledged),
            "systems_import_ready": list(self.systems_import_ready),
            "systems_license_required": list(self.systems_license_required),
            "systems_missing": list(self.systems_missing),
            "pending_acknowledgments": list(self.pending_acknowledgments),
        }


def compute_readiness(
    repo_root: Optional[Path] = None,
    *,
    license_test_mode: bool = False,
    license_env: Optional[dict] = None,
) -> ReadinessReport:
    """Combine the TERM-01 inventory with the license-gate state.

    Pure read; never writes or modifies any file.
    """
    inv: InventoryReport = inventory_terminology_data_dir(
        repo_root=repo_root,
        license_test_mode=license_test_mode,
        license_env=license_env,
    )
    rep = ReadinessReport()
    for manifest in inv.sources:
        sysv = manifest.system.value
        if manifest.status == TerminologySourceStatus.MISSING:
            rep.systems_missing.append(sysv)
            continue

        rep.systems_present.append(sysv)
        if license_acknowledged_for(
            manifest.system,
            env=license_env, test_mode=license_test_mode,
        ):
            rep.systems_acknowledged.append(sysv)
        else:
            rep.pending_acknowledgments.append(sysv)

        if manifest.status == TerminologySourceStatus.LICENSE_REQUIRED:
            rep.systems_license_required.append(sysv)
        elif manifest.status == TerminologySourceStatus.IMPORT_READY:
            rep.systems_import_ready.append(sysv)
    return rep
