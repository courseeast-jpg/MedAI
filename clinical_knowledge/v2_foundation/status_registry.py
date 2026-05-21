"""Default-off public-safe V2 capability status registry.

The registry is a metadata-only helper. It does not read files, environment
variables, databases, network resources, or runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class V2CapabilityStatus(str, Enum):
    """Controlled status values for V2 capabilities."""

    FROZEN = "frozen"
    PARKED = "parked"
    BLOCKED = "blocked"
    SPEC_ONLY = "spec_only"
    TYPING_ONLY = "typing_only"
    VALIDATION_ONLY = "validation_only"
    PLANNED_DEFAULT_OFF = "planned_default_off"
    IMPLEMENTED_DEFAULT_OFF = "implemented_default_off"
    NOT_STARTED = "not_started"


class V2CapabilityCategory(str, Enum):
    """Controlled category values for V2 capabilities."""

    FOUNDATION = "foundation"
    RUNTIME_CONTRACTS = "runtime_contracts"
    VALIDATION_HARNESS = "validation_harness"
    UI_SHELL = "ui_shell"
    DATA_INFRA = "data_infra"
    EXTRACTION_OCR = "extraction_ocr"
    PACKAGING = "packaging"
    TERMINOLOGY_PRIVATE_ADAPTER = "terminology_private_adapter"
    CUE_EXPANSION = "cue_expansion"
    CLINICAL_DECISION_LOGIC = "clinical_decision_logic"


@dataclass(frozen=True)
class V2CapabilityEntry:
    """Public-safe registry entry for one V2 capability."""

    capability_id: str
    capability_name: str
    category: V2CapabilityCategory
    status: V2CapabilityStatus
    default_enabled: bool
    runtime_wired: bool
    ui_wired: bool
    requires_operator_approval: bool
    requires_privacy_review: bool
    requires_safety_review: bool
    blocked_reason: str | None
    source_spec_block: str
    source_anchor: str
    public_report_safe: bool


def _entry(
    capability_id: str,
    capability_name: str,
    category: V2CapabilityCategory,
    status: V2CapabilityStatus,
    source_spec_block: str,
    source_anchor: str,
    *,
    blocked_reason: str | None = None,
    requires_privacy_review: bool = True,
    requires_safety_review: bool = True,
) -> V2CapabilityEntry:
    return V2CapabilityEntry(
        capability_id=capability_id,
        capability_name=capability_name,
        category=category,
        status=status,
        default_enabled=False,
        runtime_wired=False,
        ui_wired=False,
        requires_operator_approval=True,
        requires_privacy_review=requires_privacy_review,
        requires_safety_review=requires_safety_review,
        blocked_reason=blocked_reason,
        source_spec_block=source_spec_block,
        source_anchor=source_anchor,
        public_report_safe=True,
    )


_REGISTRY: Tuple[V2CapabilityEntry, ...] = (
    _entry(
        "v1_local_operator_release",
        "V1 local operator release",
        V2CapabilityCategory.FOUNDATION,
        V2CapabilityStatus.FROZEN,
        "V1-LOCAL-OPERATOR-RELEASE",
        "7ef8ffd",
    ),
    _entry(
        "v2_architecture_spec",
        "V2 architecture spec",
        V2CapabilityCategory.FOUNDATION,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-ARCHITECTURE-SPEC-01",
        "551af98",
    ),
    _entry(
        "v2_foundation_spec",
        "V2 foundation spec",
        V2CapabilityCategory.FOUNDATION,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-FOUNDATION-SPEC-02",
        "8b53d82",
    ),
    _entry(
        "v2_runtime_contracts",
        "V2 runtime contracts",
        V2CapabilityCategory.RUNTIME_CONTRACTS,
        V2CapabilityStatus.TYPING_ONLY,
        "MEDAI-V2-RUNTIME-CONTRACTS-01",
        "e6e33dd",
    ),
    _entry(
        "v2_validation_harness",
        "V2 validation harness",
        V2CapabilityCategory.VALIDATION_HARNESS,
        V2CapabilityStatus.VALIDATION_ONLY,
        "MEDAI-V2-VALIDATION-HARNESS-01",
        "73af6f6",
    ),
    _entry(
        "v2_ui_shell",
        "V2 UI shell",
        V2CapabilityCategory.UI_SHELL,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-UI-SHELL-SPEC-01",
        "745a980",
    ),
    _entry(
        "v2_data_infra",
        "V2 data infra",
        V2CapabilityCategory.DATA_INFRA,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-DATA-INFRA-SPEC-01",
        "c4df477",
    ),
    _entry(
        "v2_extraction_ocr",
        "V2 extraction and OCR",
        V2CapabilityCategory.EXTRACTION_OCR,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-EXTRACTION-SPEC-01",
        "d9ac47e194bb2a230145fb621aed2abbf9a80896",
    ),
    _entry(
        "v2_packaging",
        "V2 packaging",
        V2CapabilityCategory.PACKAGING,
        V2CapabilityStatus.SPEC_ONLY,
        "MEDAI-V2-PACKAGING-SPEC-01",
        "37d056a0b36df0226d18e483f72ac4cb9d87a049",
    ),
    _entry(
        "v2_foundation_status_registry",
        "V2 foundation status registry",
        V2CapabilityCategory.FOUNDATION,
        V2CapabilityStatus.IMPLEMENTED_DEFAULT_OFF,
        "MEDAI-V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01",
        "pending_current_commit",
    ),
    _entry(
        "terminology_private_adapter",
        "Terminology private adapter",
        V2CapabilityCategory.TERMINOLOGY_PRIVATE_ADAPTER,
        V2CapabilityStatus.BLOCKED,
        "MEDAI-CKA-TERM-INTEGRATION-PARK-02",
        "b9b19ad",
        blocked_reason="operator_license_verification_incomplete",
    ),
    _entry(
        "cue_expansion",
        "Cue expansion",
        V2CapabilityCategory.CUE_EXPANSION,
        V2CapabilityStatus.BLOCKED,
        "MEDAI-V2-FOUNDATION-SPEC-02",
        "8b53d82",
        blocked_reason="explicitly_not_recommended",
    ),
    _entry(
        "clinical_decision_logic_expansion",
        "Clinical decision logic expansion",
        V2CapabilityCategory.CLINICAL_DECISION_LOGIC,
        V2CapabilityStatus.BLOCKED,
        "MEDAI-V2-FOUNDATION-SPEC-02",
        "8b53d82",
        blocked_reason="requires_separate_clinical_safety_spec",
    ),
)


def get_v2_capability_registry() -> tuple[V2CapabilityEntry, ...]:
    """Return all V2 capability entries as an immutable tuple."""

    return tuple(_REGISTRY)


def get_v2_capability_by_id(capability_id: str) -> V2CapabilityEntry | None:
    """Return one V2 capability entry by id, or None when absent."""

    for entry in _REGISTRY:
        if entry.capability_id == capability_id:
            return entry
    return None


def list_blocked_v2_capabilities() -> tuple[V2CapabilityEntry, ...]:
    """Return blocked V2 capability entries."""

    return tuple(entry for entry in _REGISTRY if entry.status is V2CapabilityStatus.BLOCKED)


def list_default_enabled_v2_capabilities() -> tuple[V2CapabilityEntry, ...]:
    """Return entries that are default-enabled."""

    return tuple(entry for entry in _REGISTRY if entry.default_enabled)


def _count_by_status() -> Dict[str, int]:
    counts = {status.value: 0 for status in V2CapabilityStatus}
    for entry in _REGISTRY:
        counts[entry.status.value] += 1
    return counts


def _count_by_category() -> Dict[str, int]:
    counts = {category.value: 0 for category in V2CapabilityCategory}
    for entry in _REGISTRY:
        counts[entry.category.value] += 1
    return counts


def summarize_v2_capability_registry() -> dict[str, object]:
    """Return public-safe aggregate registry status only."""

    default_enabled_count = len(list_default_enabled_v2_capabilities())
    runtime_wired_count = sum(1 for entry in _REGISTRY if entry.runtime_wired)
    ui_wired_count = sum(1 for entry in _REGISTRY if entry.ui_wired)
    blocked = list_blocked_v2_capabilities()
    public_report_safe_count = sum(1 for entry in _REGISTRY if entry.public_report_safe)
    cue = get_v2_capability_by_id("cue_expansion")
    terminology = get_v2_capability_by_id("terminology_private_adapter")
    clinical = get_v2_capability_by_id("clinical_decision_logic_expansion")
    return {
        "total_count": len(_REGISTRY),
        "counts_by_status": _count_by_status(),
        "counts_by_category": _count_by_category(),
        "blocked_count": len(blocked),
        "default_enabled_count": default_enabled_count,
        "runtime_wired_count": runtime_wired_count,
        "ui_wired_count": ui_wired_count,
        "public_report_safe_count": public_report_safe_count,
        "all_default_off": default_enabled_count == 0,
        "no_runtime_wiring": runtime_wired_count == 0,
        "no_ui_wiring": ui_wired_count == 0,
        "cue_expansion_blocked": cue is not None and cue.status is V2CapabilityStatus.BLOCKED,
        "terminology_private_adapter_blocked": (
            terminology is not None and terminology.status is V2CapabilityStatus.BLOCKED
        ),
        "clinical_decision_logic_expansion_blocked": (
            clinical is not None and clinical.status is V2CapabilityStatus.BLOCKED
        ),
    }

