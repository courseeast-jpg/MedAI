"""MEDAI-V2-VALIDATION-HARNESS-01 — Typing-only V2 validation catalog.

Standard-library-only module. Defines:

* the V1 five-validation health-check catalog (preserved, not new
  V2 behavior),
* the V2 validation matrix categories A–H,
* dataclasses describing expected results,
* small pure helpers for catalog lookup.

This module:

* uses **only** the standard library (``dataclasses``, ``enum``,
  ``typing``);
* has **no runtime side effects on import** — no IO, no environment
  reads, no DB, no Streamlit, no project runtime imports, no external
  packages;
* defines **no concrete adapter implementations** and **no concrete
  validation execution**. Concrete validation execution remains the
  responsibility of the existing V1 validation scripts under
  ``scripts/run_*_validation.py``.

The V1 five-validation health-check set is preserved by reference,
not re-implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple


# ── V1 five-validation health-check catalog (preserved) ───────────────────


@dataclass(frozen=True)
class V1HealthCheck:
    """A single preserved V1 validation health-check entry."""

    name: str  # controlled-vocabulary token, e.g. "cka_final_mvp_release"
    script_relpath: str  # relative path under scripts/
    expected_conclusion_token: str  # e.g. "cka_mvp_release_package_ready"
    expected_external_api_used: bool  # always False
    expected_extra_invariants_token: Tuple[str, ...] = field(default_factory=tuple)


V1_HEALTH_CHECKS: Tuple[V1HealthCheck, ...] = (
    V1HealthCheck(
        name="cka_final_mvp_release",
        script_relpath="scripts/run_cka_final_mvp_release_validation.py",
        expected_conclusion_token="cka_mvp_release_package_ready",
        expected_external_api_used=False,
        expected_extra_invariants_token=("expected_tests_total_693",),
    ),
    V1HealthCheck(
        name="b07_term01_opt_in_integration",
        script_relpath="scripts/run_b07_term01_opt_in_integration_validation.py",
        expected_conclusion_token="b07_term01_opt_in_integration_ready",
        expected_external_api_used=False,
        expected_extra_invariants_token=("expected_cases_failed_zero",),
    ),
    V1HealthCheck(
        name="medai_route_fix01",
        script_relpath="scripts/run_medai_route_fix01_validation.py",
        expected_conclusion_token="medai_route_fix01_ready",
        expected_external_api_used=False,
        expected_extra_invariants_token=("expected_passed_true",),
    ),
    V1HealthCheck(
        name="medai_ui_ops_panel",
        script_relpath="scripts/run_medai_ui_ops_panel_validation.py",
        expected_conclusion_token="medai_ui_ops_panel_ready",
        expected_external_api_used=False,
    ),
    V1HealthCheck(
        name="medai_ui_boot_fix_startup_resilience",
        script_relpath="scripts/run_medai_ui_boot_fix_validation.py",
        expected_conclusion_token="medai_ui_boot_fix_startup_resilience_ready",
        expected_external_api_used=False,
    ),
)


# ── V2 validation matrix categories ───────────────────────────────────────


class V2ValidationMatrixCategory(Enum):
    A_CONTRACT_IMPORT_AND_SIDE_EFFECT_SAFETY = "A_contract_import_and_side_effect_safety"
    B_CONTRACT_INVENTORY_CONFORMANCE = "B_contract_inventory_conformance"
    C_SAFETY_PROFILE_CONFORMANCE = "C_safety_profile_conformance"
    D_TERMINOLOGY_AGGREGATE_ONLY_CONFORMANCE = "D_terminology_aggregate_only_conformance"
    E_REVIEW_HITL_CONFORMANCE = "E_review_hitl_conformance"
    F_REPORTS_PRIVACY_CONFORMANCE = "F_reports_privacy_conformance"
    G_RUNTIME_NON_MODIFICATION_CONFORMANCE = "G_runtime_non_modification_conformance"
    H_PARKING_FREEZE_PRESERVATION_CONFORMANCE = "H_parking_freeze_preservation_conformance"


@dataclass(frozen=True)
class V2ValidationMatrixEntry:
    """A single category in the V2 validation matrix."""

    category: V2ValidationMatrixCategory
    description_token: str  # controlled-vocabulary short description
    invariants_summary: Tuple[str, ...]
    requires_runtime_change: bool = False  # always False for this harness


V2_VALIDATION_MATRIX: Tuple[V2ValidationMatrixEntry, ...] = (
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.A_CONTRACT_IMPORT_AND_SIDE_EFFECT_SAFETY,
        description_token="contract_import_silent_and_side_effect_free",
        invariants_summary=(
            "runtime_contracts_imports_with_no_stdout_or_stderr",
            "standard_library_only_imports",
            "no_streamlit_import",
            "no_network_or_http_imports",
            "no_db_or_runtime_or_terminology_imports",
            "no_side_effects_at_import_time",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.B_CONTRACT_INVENTORY_CONFORMANCE,
        description_token="contract_inventory_complete_and_protocol_shaped",
        invariants_summary=(
            "ten_boundaries_present",
            "thirty_two_contract_names_resolve_from_module",
            "required_protocol_classes_are_protocols",
            "no_concrete_adapter_implementations",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.C_SAFETY_PROFILE_CONFORMANCE,
        description_token="default_safety_profile_carries_foundation_invariants",
        invariants_summary=(
            "local_only_true",
            "review_bound_true",
            "external_api_blocked_true",
            "auto_accept_allowed_false",
            "terminology_lookup_aggregate_only_true",
            "private_adapter_implemented_false",
            "cue_expansion_recommended_false",
            "clinical_decision_expansion_false",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.D_TERMINOLOGY_AGGREGATE_ONLY_CONFORMANCE,
        description_token="terminology_summary_aggregate_only_no_row_content",
        invariants_summary=(
            "match_family_only_controlled_vocab",
            "terminology_system_family_only_controlled_vocab",
            "matches_count_is_aggregate",
            "licensed_row_content_included_false",
            "public_report_safe_true",
            "no_code_display_synonym_definition_concept_field",
            "no_raw_text_or_ocr_text_field",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.E_REVIEW_HITL_CONFORMANCE,
        description_token="review_items_review_bound_by_default",
        invariants_summary=(
            "review_required_true_by_default",
            "auto_accept_allowed_false_by_default",
            "disposition_pending_review_by_default",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.F_REPORTS_PRIVACY_CONFORMANCE,
        description_token="public_reports_pass_privacy_check",
        invariants_summary=(
            "no_phi_in_public_reports",
            "no_raw_ocr_or_document_text_in_public_reports",
            "no_raw_filenames_in_public_reports",
            "no_private_filesystem_paths_in_public_reports",
            "no_secrets_in_public_reports",
            "no_licensed_terminology_row_content_in_public_reports",
            "no_license_acknowledgement_contents_in_public_reports",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.G_RUNTIME_NON_MODIFICATION_CONFORMANCE,
        description_token="runtime_files_unchanged_no_external_api",
        invariants_summary=(
            "app_main_unchanged",
            "launchers_unchanged",
            "startup_preflight_unchanged",
            "config_unchanged",
            "no_runtime_wiring_added",
            "no_external_api_used_for_runtime",
            "no_runtime_db_accessed",
        ),
    ),
    V2ValidationMatrixEntry(
        category=V2ValidationMatrixCategory.H_PARKING_FREEZE_PRESERVATION_CONFORMANCE,
        description_token="parking_and_freeze_tags_preserved",
        invariants_summary=(
            "v1_frozen_release_at_7ef8ffd_preserved",
            "park_20_through_23_tag_pairs_unchanged",
            "park_24_through_26_anchor_commits_unchanged",
            "term_helper_wiring_park_01_tag_pair_unchanged",
            "license_gate_park_02_tag_pair_unchanged",
            "no_new_tags_created_by_this_block",
        ),
    ),
)


# ── Pure catalog helpers ──────────────────────────────────────────────────


def v1_health_check_names() -> Tuple[str, ...]:
    return tuple(hc.name for hc in V1_HEALTH_CHECKS)


def v1_health_check_by_name(name: str) -> Optional[V1HealthCheck]:
    for hc in V1_HEALTH_CHECKS:
        if hc.name == name:
            return hc
    return None


def v2_validation_matrix_categories() -> Tuple[str, ...]:
    return tuple(c.value for c in V2ValidationMatrixCategory)


def v2_validation_matrix_summary() -> Mapping[str, Tuple[str, ...]]:
    """Pure summary map: category value -> tuple of invariants."""
    return {
        entry.category.value: tuple(entry.invariants_summary)
        for entry in V2_VALIDATION_MATRIX
    }


__all__ = (
    "V1HealthCheck",
    "V1_HEALTH_CHECKS",
    "V2ValidationMatrixCategory",
    "V2ValidationMatrixEntry",
    "V2_VALIDATION_MATRIX",
    "v1_health_check_names",
    "v1_health_check_by_name",
    "v2_validation_matrix_categories",
    "v2_validation_matrix_summary",
)
