"""CKA-B10: System preflight checker.

Verifies all CKA blocks B01-B09 are importable and operationally sound
before any feature is exposed in production.

Safety invariants checked:
- EXTERNAL_APIS_ENABLED must be False (CKAConfig)
- ConnectorRegistry must reject allow_external=True
- Enrichment allow_active_write must raise ValueError when True
- HITL release freeze document must be present
- Operator UI snapshot must not read private files
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums / status
# ---------------------------------------------------------------------------


class PreflightStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


_STATUS_RANK = {
    PreflightStatus.PASS: 0,
    PreflightStatus.WARN: 1,
    PreflightStatus.FAIL: 2,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PreflightCheck:
    name: str
    status: PreflightStatus
    detail: str
    block_id: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "block_id": self.block_id,
            "detail_ok": self.status == PreflightStatus.PASS,
        }


@dataclass
class CKAPreflightReport:
    checks: List[PreflightCheck] = field(default_factory=list)
    overall_status: PreflightStatus = PreflightStatus.PASS
    hitl_freeze_confirmed: bool = False
    external_api_blocked: bool = True
    safe_mode_active: bool = True

    @property
    def passed(self) -> bool:
        return self.overall_status == PreflightStatus.PASS

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == PreflightStatus.PASS)

    @property
    def checks_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == PreflightStatus.FAIL)

    @property
    def checks_warned(self) -> int:
        return sum(1 for c in self.checks if c.status == PreflightStatus.WARN)

    def safe_public_summary(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "checks_total": len(self.checks),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warned": self.checks_warned,
            "hitl_freeze_confirmed": self.hitl_freeze_confirmed,
            "external_api_blocked": self.external_api_blocked,
            "safe_mode_active": self.safe_mode_active,
            "check_summaries": [c.safe_public_summary() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pass(name: str, detail: str, block_id: Optional[str] = None) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.PASS, detail=detail, block_id=block_id)


def _warn(name: str, detail: str, block_id: Optional[str] = None) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.WARN, detail=detail, block_id=block_id)


def _fail(name: str, detail: str, block_id: Optional[str] = None) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.FAIL, detail=detail, block_id=block_id)


def _check_import(name: str, module: str, block_id: Optional[str]) -> PreflightCheck:
    try:
        importlib.import_module(module)
        return _pass(name, f"Module '{module}' imported successfully.", block_id=block_id)
    except Exception as exc:
        return _fail(name, f"Module '{module}' import failed: {type(exc).__name__}", block_id=block_id)


# ---------------------------------------------------------------------------
# Per-block checks
# ---------------------------------------------------------------------------


def _check_b01_mkb_foundation() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b01_import_models", "clinical_knowledge.models", "CKA-B01"))
    checks.append(_check_import("b01_import_store", "clinical_knowledge.store", "CKA-B01"))
    try:
        from clinical_knowledge.store import MKBStore
        MKBStore(":memory:")
        checks.append(_pass("b01_store_init", "MKBStore instantiated in-memory.", "CKA-B01"))
    except Exception as exc:
        checks.append(_fail("b01_store_init", f"MKBStore init failed: {type(exc).__name__}", "CKA-B01"))
    return checks


def _check_b02_privacy_boundary() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b02_import_sanitizer", "clinical_knowledge.privacy.sanitizer", "CKA-B02"))
    checks.append(_check_import("b02_import_outbound", "clinical_knowledge.privacy.outbound_audit", "CKA-B02"))
    try:
        from clinical_knowledge.privacy.patterns import ALWAYS_BLOCK_CATEGORIES
        if "SECRET" in ALWAYS_BLOCK_CATEGORIES:
            checks.append(_pass("b02_secret_blocked", "SECRET in ALWAYS_BLOCK_CATEGORIES confirmed.", "CKA-B02"))
        else:
            checks.append(_fail("b02_secret_blocked", "SECRET not found in ALWAYS_BLOCK_CATEGORIES.", "CKA-B02"))
    except Exception as exc:
        checks.append(_warn("b02_secret_blocked", f"Could not verify SECRET block: {type(exc).__name__}", "CKA-B02"))
    return checks


def _check_b03_decision_engine() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b03_import_engine", "clinical_knowledge.decision_engine.engine", "CKA-B03"))
    checks.append(_check_import("b03_import_safe_mode", "clinical_knowledge.decision_engine.safe_mode", "CKA-B03"))
    return checks


def _check_b04_truth_resolution() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b04_import_engine", "clinical_knowledge.truth_resolution.engine", "CKA-B04"))
    checks.append(_check_import("b04_import_quarantine", "clinical_knowledge.truth_resolution.quarantine", "CKA-B04"))
    return checks


def _check_b05_medication_safety() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b05_import_models", "clinical_knowledge.medication_safety.models", "CKA-B05"))
    checks.append(_check_import("b05_import_ddi", "clinical_knowledge.medication_safety.ddi_stub", "CKA-B05"))
    try:
        from clinical_knowledge.medication_safety.write_gate import MedicationWriteGateResult
        # Confirm the gate result type is importable (module-level smoke check)
        checks.append(_pass("b05_write_gate_import", "MedicationWriteGateResult importable.", "CKA-B05"))
    except Exception as exc:
        checks.append(_warn("b05_write_gate_import", f"Write gate import check skipped: {type(exc).__name__}", "CKA-B05"))
    return checks


def _check_b06_enrichment() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b06_import_models", "clinical_knowledge.enrichment.models", "CKA-B06"))
    checks.append(_check_import("b06_import_integration", "clinical_knowledge.enrichment.integration", "CKA-B06"))
    try:
        from clinical_knowledge.consensus.integration import consensus_facts_to_enrichment_candidates
        try:
            consensus_facts_to_enrichment_candidates([], allow_active_write=True)
            checks.append(_fail(
                "b06_active_write_blocked",
                "allow_active_write=True did NOT raise ValueError.",
                "CKA-B06",
            ))
        except ValueError:
            checks.append(_pass(
                "b06_active_write_blocked",
                "allow_active_write=True correctly raises ValueError.",
                "CKA-B06",
            ))
    except Exception as exc:
        checks.append(_warn(
            "b06_active_write_blocked",
            f"allow_active_write guard check skipped: {type(exc).__name__}",
            "CKA-B06",
        ))
    return checks


def _check_b07_medical_coding() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b07_import_models", "clinical_knowledge.medical_coding.models", "CKA-B07"))
    checks.append(_check_import("b07_import_mapper", "clinical_knowledge.medical_coding.synthetic_mapper", "CKA-B07"))
    return checks


def _check_b08_consensus() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b08_import_registry", "clinical_knowledge.connectors.registry", "CKA-B08"))
    checks.append(_check_import("b08_import_consensus", "clinical_knowledge.consensus.engine", "CKA-B08"))
    try:
        from clinical_knowledge.connectors.registry import ConnectorRegistry, ConnectorRegistryError
        from clinical_knowledge.connectors.models import (
            ConnectorSpec, ConnectorKind, ConnectorCapability,
        )
        registry = ConnectorRegistry()
        bad_spec = ConnectorSpec(
            name="bad_external",
            kind=ConnectorKind.DXGPT_STUB,
            enabled=True,
            allow_external=True,
            synthetic_only=True,
            capabilities=[ConnectorCapability.DIAGNOSIS_SUPPORT],
        )
        try:
            registry.register(bad_spec)
            checks.append(_fail(
                "b08_registry_rejects_external",
                "Registry allowed allow_external=True — unsafe!",
                "CKA-B08",
            ))
        except ConnectorRegistryError:
            checks.append(_pass(
                "b08_registry_rejects_external",
                "Registry correctly rejects allow_external=True.",
                "CKA-B08",
            ))
    except Exception as exc:
        checks.append(_warn(
            "b08_registry_rejects_external",
            f"Registry external-spec check skipped: {type(exc).__name__}",
            "CKA-B08",
        ))
    return checks


def _check_b09_operator_ui() -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    checks.append(_check_import("b09_import_viewer", "app.clinical_knowledge_safety_viewer", "CKA-B09"))
    try:
        from app.clinical_knowledge_safety_viewer import load_cka_safety_snapshot
        snapshot = load_cka_safety_snapshot()
        if snapshot.get("private_files_read") is False:
            checks.append(_pass(
                "b09_snapshot_no_private",
                "Snapshot confirms private_files_read=False.",
                "CKA-B09",
            ))
        else:
            checks.append(_fail(
                "b09_snapshot_no_private",
                "Snapshot reports private_files_read=True — data leak!",
                "CKA-B09",
            ))
    except Exception as exc:
        checks.append(_warn(
            "b09_snapshot_no_private",
            f"Snapshot privacy check skipped: {type(exc).__name__}",
            "CKA-B09",
        ))
    return checks


# ---------------------------------------------------------------------------
# Cross-cutting safety invariants
# ---------------------------------------------------------------------------


def _check_external_api_blocked() -> PreflightCheck:
    try:
        from clinical_knowledge.config import DEFAULT_CONFIG
        if not DEFAULT_CONFIG.EXTERNAL_APIS_ENABLED:
            return _pass("external_api_blocked", "EXTERNAL_APIS_ENABLED=False confirmed.")
        return _fail("external_api_blocked", "EXTERNAL_APIS_ENABLED=True — unsafe configuration!")
    except Exception as exc:
        return _warn("external_api_blocked", f"Could not read CKAConfig: {type(exc).__name__}")


def _check_hitl_freeze(repo_root: Optional[Path] = None) -> PreflightCheck:
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    freeze_path = repo_root / "MEDAI_OCR_LAYOUT_HITL_RELEASE_FREEZE.md"
    if freeze_path.exists():
        return _pass("hitl_freeze_present", f"HITL freeze document found: {freeze_path.name}")
    return _warn("hitl_freeze_present", "HITL freeze document not found — verify freeze status.")


def _check_config_safe_defaults() -> PreflightCheck:
    try:
        from clinical_knowledge.config import DEFAULT_CONFIG
        unsafe = []
        if DEFAULT_CONFIG.ENABLE_GRAPH:
            unsafe.append("ENABLE_GRAPH=True")
        if DEFAULT_CONFIG.ENABLE_LOCAL_LLM:
            unsafe.append("ENABLE_LOCAL_LLM=True")
        if DEFAULT_CONFIG.ENRICH_PROMOTE:
            unsafe.append("ENRICH_PROMOTE=True")
        if unsafe:
            return _warn("config_safe_defaults", f"Unsafe flags: {', '.join(unsafe)}")
        return _pass("config_safe_defaults", "ENABLE_GRAPH, ENABLE_LOCAL_LLM, ENRICH_PROMOTE all False.")
    except Exception as exc:
        return _warn("config_safe_defaults", f"Config check skipped: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Main preflight runner
# ---------------------------------------------------------------------------


def run_cka_preflight(
    safe_mode: bool = True,
    repo_root: Optional[Path] = None,
) -> CKAPreflightReport:
    """Run all CKA system preflight checks. Returns a CKAPreflightReport.

    Does NOT call any external API.
    Does NOT write to any record store.
    Does NOT perform any clinical inference.
    """
    checks: List[PreflightCheck] = []

    checks.extend(_check_b01_mkb_foundation())
    checks.extend(_check_b02_privacy_boundary())
    checks.extend(_check_b03_decision_engine())
    checks.extend(_check_b04_truth_resolution())
    checks.extend(_check_b05_medication_safety())
    checks.extend(_check_b06_enrichment())
    checks.extend(_check_b07_medical_coding())
    checks.extend(_check_b08_consensus())
    checks.extend(_check_b09_operator_ui())

    # Cross-cutting safety invariants
    checks.append(_check_external_api_blocked())
    checks.append(_check_hitl_freeze(repo_root=repo_root))
    checks.append(_check_config_safe_defaults())

    # Determine overall status
    worst = PreflightStatus.PASS
    for c in checks:
        if _STATUS_RANK[c.status] > _STATUS_RANK[worst]:
            worst = c.status

    hitl_confirmed = any(
        c.name == "hitl_freeze_present" and c.status == PreflightStatus.PASS
        for c in checks
    )
    external_blocked = any(
        c.name == "external_api_blocked" and c.status == PreflightStatus.PASS
        for c in checks
    )

    return CKAPreflightReport(
        checks=checks,
        overall_status=worst,
        hitl_freeze_confirmed=hitl_confirmed,
        external_api_blocked=external_blocked,
        safe_mode_active=safe_mode,
    )
