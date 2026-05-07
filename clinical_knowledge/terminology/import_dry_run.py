"""CKA-TERM-01B dry-run orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clinical_knowledge.terminology.file_inventory import inventory_terminology_data_dir
from clinical_knowledge.terminology.import_limits import TerminologyImportLimits
from clinical_knowledge.terminology.import_planner import TerminologyImportPlan, plan_terminology_import
from clinical_knowledge.terminology.intake_automation import compute_readiness


def run_terminology_import_dry_run(
    *,
    repo_root: Path | None = None,
    limits: TerminologyImportLimits | None = None,
    license_test_mode: bool = False,
    license_env: dict | None = None,
) -> dict[str, Any]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    active_limits = limits or TerminologyImportLimits()
    inventory = inventory_terminology_data_dir(
        repo_root=repo_root,
        license_test_mode=license_test_mode,
        license_env=license_env,
    )
    readiness = compute_readiness(
        repo_root=repo_root,
        license_test_mode=license_test_mode,
        license_env=license_env,
    )
    plan: TerminologyImportPlan = plan_terminology_import(
        inventory,
        readiness,
        active_limits,
        dry_run=True,
    )
    return {
        "inventory": inventory.safe_public_summary(),
        "readiness": readiness.safe_public_summary(),
        "limits": active_limits.safe_public_summary(),
        "plan": plan.safe_public_summary(),
        "real_files_imported": False,
        "production_index_created": False,
        "external_api_used": False,
    }
