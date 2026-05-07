"""CKA-TERM-01F TERM-02 preflight gate.

The gate is read-only. It checks whether a local terminology_data tree is
ready for a future TERM-02 import, but it never imports data and never
downloads terminology files.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from clinical_knowledge.terminology.file_inventory import (
    _EXPECTED_FILES,
    inventory_terminology_data_dir,
)
from clinical_knowledge.terminology.intake_automation import compute_readiness
from clinical_knowledge.terminology.models import TerminologySystem


ACK_FILENAME = "LICENSE_ACK_PRIVATE.json"
SUPPORTED_SYSTEMS: tuple[TerminologySystem, ...] = (
    TerminologySystem.UMLS,
    TerminologySystem.SNOMED_CT,
    TerminologySystem.RXNORM,
    TerminologySystem.LOINC,
)


@dataclass(frozen=True)
class Term02PreflightResult:
    allowed: bool
    reason_codes: list[str] = field(default_factory=list)
    systems_with_files: list[str] = field(default_factory=list)
    systems_import_ready: list[str] = field(default_factory=list)
    systems_pending_acknowledgment: list[str] = field(default_factory=list)
    terminology_root_present: bool = False
    supported_subfolders_present: bool = False
    license_ack_private_present: bool = False
    operator_acknowledged: bool = False
    terminology_files_staged: bool = False
    data_terminology_staged: bool = False
    external_api_used: bool = False
    real_import_performed: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "systems_with_files": list(self.systems_with_files),
            "systems_import_ready": list(self.systems_import_ready),
            "systems_pending_acknowledgment": list(self.systems_pending_acknowledgment),
            "terminology_root_present": self.terminology_root_present,
            "supported_subfolders_present": self.supported_subfolders_present,
            "license_ack_private_present": self.license_ack_private_present,
            "operator_acknowledged": self.operator_acknowledged,
            "terminology_files_staged": self.terminology_files_staged,
            "data_terminology_staged": self.data_terminology_staged,
            "external_api_used": self.external_api_used,
            "real_import_performed": self.real_import_performed,
        }


def run_term02_preflight_gate(
    *,
    repo_root: Path | None = None,
    terminology_root: Path | None = None,
) -> Term02PreflightResult:
    """Check whether TERM-02 may start.

    This function may inspect file names and safe ack fields only. It never
    reads terminology file contents.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    repo_root = Path(repo_root)
    terminology_root = Path(terminology_root) if terminology_root is not None else repo_root / "terminology_data"
    terminology_root = terminology_root.resolve()
    effective_repo_root = terminology_root.parent

    reason_codes: list[str] = []
    root_present = terminology_root.exists() and terminology_root.is_dir()
    if not root_present:
        reason_codes.append("terminology_data_missing")

    subfolders_present = root_present and all((terminology_root / s.value).is_dir() for s in SUPPORTED_SYSTEMS)
    systems_with_files = _systems_with_expected_files(terminology_root) if root_present else []
    if not systems_with_files:
        reason_codes.append("no_supported_files_present")

    ack_path = terminology_root / ACK_FILENAME
    ack_present = ack_path.exists() and ack_path.is_file()
    ack = _read_ack_summary(ack_path)
    if not ack_present:
        reason_codes.append("license_ack_private_missing")
    if ack_present and not ack["operator_acknowledged"]:
        reason_codes.append("license_ack_not_confirmed")

    acknowledged = set(ack["acknowledged_systems"])
    pending_ack = sorted(system for system in systems_with_files if system not in acknowledged)
    if pending_ack:
        reason_codes.append("systems_pending_acknowledgment")

    inventory = inventory_terminology_data_dir(repo_root=effective_repo_root)
    readiness = compute_readiness(repo_root=effective_repo_root)
    # The existing readiness checker uses the repository default ack location.
    # This gate also enforces the explicit ack file under terminology_root so
    # synthetic temp rehearsals and future TERM-02 calls remain location-bound.
    readiness_ready = set(readiness.safe_public_summary().get("systems_import_ready", []))
    systems_import_ready = sorted(system for system in systems_with_files if system in acknowledged and system in readiness_ready)
    if not systems_import_ready:
        # If the explicit ack is valid but the legacy readiness helper did not
        # resolve this temp root's ack, still allow the gate to reason from the
        # bounded inventory + explicit ack for temp rehearsal only.
        inventory_ready = {
            m.system.value
            for m in inventory.sources
            if m.system.value in systems_with_files and m.system.value in acknowledged
        }
        systems_import_ready = sorted(inventory_ready)
    if not systems_import_ready:
        reason_codes.append("no_system_import_ready")

    staged = _git_staged_paths(repo_root)
    terminology_staged = _any_staged_under(staged, "terminology_data/")
    data_staged = _any_staged_under(staged, "data/terminology/")
    if terminology_staged:
        reason_codes.append("terminology_files_staged")
    if data_staged:
        reason_codes.append("data_terminology_staged")

    allowed = not reason_codes
    return Term02PreflightResult(
        allowed=allowed,
        reason_codes=sorted(set(reason_codes)),
        systems_with_files=systems_with_files,
        systems_import_ready=systems_import_ready,
        systems_pending_acknowledgment=pending_ack,
        terminology_root_present=root_present,
        supported_subfolders_present=subfolders_present,
        license_ack_private_present=ack_present,
        operator_acknowledged=bool(ack["operator_acknowledged"]),
        terminology_files_staged=terminology_staged,
        data_terminology_staged=data_staged,
    )


def _systems_with_expected_files(terminology_root: Path) -> list[str]:
    systems: list[str] = []
    for system in SUPPORTED_SYSTEMS:
        root = terminology_root / system.value
        names = [p.name.lower() for p in root.iterdir() if p.is_file()] if root.exists() else []
        expected = [frag.lower() for frag in _EXPECTED_FILES[system]]
        if any(any(frag in name for name in names) for frag in expected):
            systems.append(system.value)
    return sorted(systems)


def _read_ack_summary(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {"operator_acknowledged": False, "acknowledged_systems": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"operator_acknowledged": False, "acknowledged_systems": []}
    systems = data.get("acknowledged_systems") if isinstance(data, dict) else []
    if not isinstance(systems, list):
        systems = []
    return {
        "operator_acknowledged": bool(isinstance(data, dict) and data.get("operator_acknowledged") is True),
        "acknowledged_systems": sorted({str(s).lower() for s in systems}),
    }


def _git_staged_paths(repo_root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _any_staged_under(paths: Iterable[str], prefix: str) -> bool:
    prefix = prefix.replace("\\", "/")
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for path in paths)
