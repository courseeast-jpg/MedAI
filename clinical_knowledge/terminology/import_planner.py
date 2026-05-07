"""CKA-TERM-01B terminology import dry-run planner.

The planner consumes TERM-01 inventory/readiness summaries and produces a
safe capacity plan. It does not import data, does not read licensed file
contents, and does not expose raw paths.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from clinical_knowledge.terminology.import_limits import TerminologyImportLimits


SYSTEMS = ("umls", "snomed_ct", "rxnorm", "loinc")


@dataclass
class TerminologyImportPlan:
    systems_seen: list[str] = field(default_factory=list)
    systems_import_ready: list[str] = field(default_factory=list)
    systems_blocked_license: list[str] = field(default_factory=list)
    systems_missing: list[str] = field(default_factory=list)
    estimated_files: int = 0
    estimated_rows_safe: int = 0
    estimated_chunks: int = 0
    row_caps_applied: dict[str, bool] = field(default_factory=dict)
    import_allowed: bool = False
    dry_run: bool = True
    real_files_imported: bool = False
    plan_id: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = f"term_plan_{uuid.uuid4().hex[:12]}"

    def safe_public_summary(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "systems_seen": list(self.systems_seen),
            "systems_import_ready": list(self.systems_import_ready),
            "systems_blocked_license": list(self.systems_blocked_license),
            "systems_missing": list(self.systems_missing),
            "estimated_files": self.estimated_files,
            "estimated_rows_safe": self.estimated_rows_safe,
            "estimated_chunks": self.estimated_chunks,
            "row_caps_applied": dict(self.row_caps_applied),
            "import_allowed": self.import_allowed,
            "dry_run": self.dry_run,
            "real_files_imported": self.real_files_imported,
        }


def plan_terminology_import(
    inventory: Any,
    license_status: Any,
    limits: TerminologyImportLimits,
    *,
    dry_run: bool = True,
) -> TerminologyImportPlan:
    sources = _source_summaries(inventory)
    license_summary = _license_summary(license_status)
    seen: list[str] = []
    ready: list[str] = []
    blocked: list[str] = []
    missing: list[str] = []
    estimated_files = 0
    estimated_rows = 0
    estimated_chunks = 0
    row_caps: dict[str, bool] = {}

    by_system = {str(src.get("system")): src for src in sources if src.get("system")}
    for system in SYSTEMS:
        src = by_system.get(system)
        file_count = int((src or {}).get("file_count") or 0)
        license_confirmed = _license_confirmed(system, src, license_summary)
        if src is None or file_count <= 0 or (src.get("status") == "missing"):
            missing.append(system)
            row_caps[system] = False
            continue
        seen.append(system)
        estimated_files += file_count
        raw_estimate = file_count * max(1, limits.max_rows_per_file_default)
        system_rows = min(raw_estimate, max(1, limits.max_rows_per_system_default))
        estimated_rows += system_rows
        estimated_chunks += math.ceil(system_rows / max(1, limits.chunk_size))
        row_caps[system] = raw_estimate > system_rows
        if license_confirmed:
            ready.append(system)
        else:
            blocked.append(system)

    import_allowed = (
        bool(ready)
        and not dry_run
        and limits.allow_real_import
        and not blocked
    )
    return TerminologyImportPlan(
        systems_seen=seen,
        systems_import_ready=ready,
        systems_blocked_license=blocked,
        systems_missing=missing,
        estimated_files=estimated_files,
        estimated_rows_safe=estimated_rows,
        estimated_chunks=estimated_chunks,
        row_caps_applied=row_caps,
        import_allowed=import_allowed,
        dry_run=bool(dry_run),
        real_files_imported=False,
    )


def _source_summaries(inventory: Any) -> list[dict[str, Any]]:
    if hasattr(inventory, "safe_public_summary"):
        inventory = inventory.safe_public_summary()
    if isinstance(inventory, dict):
        sources = inventory.get("sources") or []
        return [dict(src) for src in sources if isinstance(src, dict)]
    return []


def _license_summary(license_status: Any) -> dict[str, Any]:
    if hasattr(license_status, "safe_public_summary"):
        license_status = license_status.safe_public_summary()
    if isinstance(license_status, dict):
        return license_status
    return {}


def _license_confirmed(system: str, src: dict[str, Any] | None, license_summary: dict[str, Any]) -> bool:
    if src and src.get("license_confirmed") is True:
        return True
    acknowledged = {str(value) for value in license_summary.get("systems_acknowledged") or []}
    ready = {str(value) for value in license_summary.get("systems_import_ready") or []}
    return system in acknowledged or system in ready
