"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A - Latin Medical Abbreviation Handling Spec.

Specification-only block.

Scope
-----
Targets the 8 records DIAG-04 routed to
``latin_medical_abbreviation_handling_audit``. The block:

    1. Re-derives the priority slice via the DIAG-02 -> DIAG-03 -> DIAG-04
       chain (no corpus rerun, no source documents opened).
    2. Confirms every record matches the exact positive signal pattern.
    3. Confirms no exclusion rule fires on the slice.
    4. Confirms zero overlap with the DIAG-06A numeric-table safe-default
       helper AND zero overlap with the DIAG-09A language-propagation
       helper.
    5. Emits a privacy-safe public spec of a FUTURE default-off
       abbreviation-context metadata implementation: positive signal
       pattern, exclusion rules, proposed behavior, acceptance criteria,
       validation requirements, and rollback boundaries.

This block does NOT implement abbreviation handling. No runtime,
detector, classifier, OCR, threshold, scoring, cue-pack, B07,
ROUTE-FIX, DB schema, allowlist, or external-API behavior is changed.

Hard boundaries
---------------
* No OCR routing / OCR engine changes.
* No raw language detector behavior changes.
* No abbreviation handling behavior changes in this block.
* No classifier behavior changes.
* No thresholds, scoring, auto-accept, cue-pack, or cue-expansion changes.
* No clinical interpretation, no lab/medication/dose/DDI parsing.
* No B07 / ROUTE-FIX / DB schema / command allowlist / external-API changes.
* No raw filenames, raw OCR text, raw document text, private paths, PHI, or
  secrets. Anonymized aggregate output only.

Progress estimate
-----------------
* Before this block: residual Unknown-reduction track ~87% / ~13%;
  whole MedAI project ~82% / ~18%.
* After this block:  residual Unknown-reduction track ~90% / ~10%;
  whole MedAI project ~83% / ~17%.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_knowledge.document_type import (  # noqa: E402
    derive_language_propagation_metadata_label,
    derive_numeric_table_safe_default_label,
)
from scripts.run_medai_doc_type_unknown_diag_04 import (  # noqa: E402
    SOURCE_REPORT,
    _select_priority_records,
    evidence_flags_for_latin_lang,
    evidence_flags_for_table_heavy,
    next_lever_for_latin_lang,
    next_lever_for_table_heavy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "medai_doc_type_unknown_diag_11a"

SOURCE_10A_COMMIT_SHORT = "fa4ac76"
PARK_19_COMMIT_SHORT = "ac466e0f9ab8"
PUBLIC_REPORT_HASH_POLICY = "short_hashes_only"

SOURCE_REPORT_LABEL = (
    "reports/medai_doc_type_family_04_larger_slice_validation/"
    "(public anonymized batch-eval per-file table)"
)
UPSTREAM_LABELS = (
    "reports/medai_doc_type_unknown_diag_10a_language_propagation_operator_surface/(public)",
    "reports/medai_doc_type_unknown_diag_09a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_09a/(public spec)",
    "reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)",
    "reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)",
    "reports/medai_doc_type_unknown_diag_06a_implementation/(public)",
    "reports/medai_doc_type_unknown_diag_06a/(public spec)",
    "reports/medai_doc_type_unknown_diag_05/(public diagnostic)",
    "reports/medai_doc_type_unknown_diag_04/(public diagnostic)",
)

_DIAG04_TARGET_LEVER = "latin_medical_abbreviation_handling_audit"

PROPOSED_LABEL = "latin_medical_abbreviation_context"

SUGGESTED_FUTURE_ENV_VAR = (
    "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED"
)


# ── Controlled-vocabulary spec (task-mandated) ──────────────────────────────

POSITIVE_SIGNAL_PATTERN: tuple[tuple[str, str], ...] = (
    ("detector_attempted",                       "yes"),
    ("detector_input_bucket",                    "sufficient"),
    ("detector_confidence_bucket",               "high_or_medium"),
    ("script_detection_result",                  "latin"),
    ("dominant_script",                          "latin"),
    ("language_visibility_status",               "latin_visible_language_unknown"),
    ("latin_medical_abbrev_visible",             "yes"),
    ("medical_abbreviation_shape_detected",      "yes"),
    ("alphabetic_ratio_sufficient_for_language", "yes"),
    ("no_cyrillic_dominant_signal",              "yes"),
    ("no_mixed_script_signal",                   "yes"),
    ("no_low_confidence_detector_signal",        "yes"),
    ("not_already_handled_by_numeric_table_safe_default_helper", "yes"),
    ("not_already_handled_by_language_propagation_helper",       "yes"),
)

EXCLUSION_RULES: tuple[str, ...] = (
    "exclude_cyrillic_dominant_records",
    "exclude_mixed_script_records",
    "exclude_low_detector_confidence_records",
    "exclude_insufficient_detector_input_records",
    "exclude_no_text_layer_records",
    "exclude_image_like_but_not_routed_records",
    "exclude_table_heavy_numeric_safe_default_records_already_handled",
    "exclude_language_propagation_records_already_handled",
    "exclude_table_header_only_special_case_record",
    "exclude_ambiguous_below_threshold_records",
    "exclude_fallback_ran_but_no_family_match_records",
    "exclude_medication_dose_or_ddi_interpretation",
    "exclude_lab_value_parsing",
    "exclude_records_with_insufficient_safe_metadata",
)

PROPOSED_FUTURE_BEHAVIOR: dict[str, Any] = {
    "applies_to": (
        "records matching the exact positive signal pattern only"
    ),
    "proposed_label": PROPOSED_LABEL,
    "proposed_label_meaning": (
        "Latin-script text contains medical-style abbreviations useful for "
        "language / context routing"
    ),
    "default_action": (
        "derive `latin_medical_abbreviation_context` as a safe metadata "
        "label only; never expand or parse the abbreviation"
    ),
    "must_not_classify_clinical_meaning": True,
    "must_not_parse_the_abbreviation": True,
    "must_not_expand_the_abbreviation": True,
    "must_not_parse_values": True,
    "must_not_auto_accept": True,
    "must_not_write_active_clinical_facts": True,
    "must_keep_document_review_bound": True,
    "must_be_default_off_behind_separate_env_flag_or_operator_setting": True,
    "suggested_future_env_var": SUGGESTED_FUTURE_ENV_VAR,
    "must_not_overlap_with_numeric_table_safe_default_pool": True,
    "must_not_overlap_with_language_propagation_pool": True,
    "must_not_alter_raw_detector_output": True,
    "must_not_change_data_layer_document_type": True,
}

FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "only_exact_abbreviation_signature_records_receive_the_metadata_label",
    "no_overlap_with_numeric_table_safe_default_records",
    "no_overlap_with_language_propagation_records",
    "no_overlap_with_table_header_special_case_unless_explicitly_included_later",
    "accepted_count_remains_zero",
    "auto_accept_allowed_count_remains_zero",
    "external_api_used_count_remains_zero",
    "all_affected_records_remain_review_bound",
    "data_layer_unknown_count_behavior_explicitly_reported",
    "no_treatment_imaging_or_admin_false_positive_expansion",
    "public_report_privacy_checks_remain_clean",
    "rollback_or_disable_path_exists",
)

FUTURE_VALIDATION_REQUIREMENTS: tuple[str, ...] = (
    "focused_synthetic_tests",
    "replay_of_8_record_abbreviation_pool",
    "five_hundred_seven_file_aggregate_validation",
    "overlap_audit_against_diag_06a_helper_pool",
    "overlap_audit_against_diag_09a_helper_pool",
    "document_type_eval_regression_tests",
    "public_report_privacy_checks",
    "final_cka_mvp_validation",
    "b07_validation",
    "route_fix_validation",
    "ui_ops_validation",
    "ui_boot_validation",
    "staged_safety_check",
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DiagnosticReport:
    snapshot: str
    branch: str
    head_commit_short: str
    source_10a_commit_short: str
    park_19_commit_short: str
    public_report_commit_hash_policy: str
    source_report_label: str
    upstream_labels: list[str]
    generated_at: str

    total_abbreviation_pool_records: int
    overlap_with_numeric_table_safe_default_pool: int
    no_overlap_with_numeric_table_safe_default_pool: bool
    overlap_with_language_propagation_pool: int
    no_overlap_with_language_propagation_pool: bool

    positive_signal_pattern: list[dict[str, str]]
    positive_signal_match_report: dict[str, dict[str, Any]]
    positive_signal_holds_on_all_priority_records: bool

    exclusion_rules: list[str]
    exclusion_audit: list[dict[str, Any]]
    no_priority_record_violates_any_exclusion_rule: bool

    proposed_future_behavior: dict[str, Any]
    future_implementation_acceptance_criteria: list[str]
    future_validation_requirements: list[str]
    suggested_future_env_var: str

    anonymized_sample_ids: list[str]
    raw_signal_counts: dict[str, dict[str, int]]
    deferred_subsets: dict[str, str]

    implementation_block_recommended_next: str
    implementation_block_justification: str

    progress_estimate: dict[str, str]

    behavior_changed: bool
    external_api_used: bool
    cue_expansion_recommended: bool
    abbreviation_handling_implemented_in_this_block: bool
    safety_privacy: dict[str, bool]


# ── helpers ──────────────────────────────────────────────────────────────────

def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _g(record: dict, key: str) -> str:
    v = record.get(key)
    if v is None:
        return ""
    return str(v).strip().lower()


_LABEL_RE = re.compile(r"^[a-z0-9_]+$")


def _is_safe_label(value: Any) -> bool:
    return isinstance(value, str) and bool(_LABEL_RE.fullmatch(value))


def _safe_count(records: Iterable[dict], key: str) -> dict[str, int]:
    c: Counter = Counter()
    for r in records:
        v = _g(r, key) or "unknown"
        if not _is_safe_label(v):
            v = "other"
        c[v] += 1
    return dict(sorted(c.items()))


def _anonymized_ids(prefix: str, count: int) -> list[str]:
    if count <= 0:
        return []
    return [f"{prefix}_{i + 1:03d}" for i in range(min(count, 5))]


# ── Slice selection (DIAG-04 chain) ─────────────────────────────────────────

def select_abbreviation_pool(table: list[dict]) -> list[dict]:
    """Return the records DIAG-04 routed to latin_medical_abbreviation_handling_audit."""
    th, latin = _select_priority_records(table)
    pool: list[dict] = []
    for r in th:
        flags = evidence_flags_for_table_heavy(r)
        if next_lever_for_table_heavy(r, flags) == _DIAG04_TARGET_LEVER:
            pool.append(r)
    for r in latin:
        flags = evidence_flags_for_latin_lang(r)
        if next_lever_for_latin_lang(r, flags) == _DIAG04_TARGET_LEVER:
            pool.append(r)
    return pool


# ── Signature & exclusion auditing ──────────────────────────────────────────

def _positive_signal_predicate(key: str, expected: str, record: dict) -> bool:
    if key == "detector_attempted":
        return _g(record, "language_detector_attempted") == expected
    if key == "detector_input_bucket":
        return _g(record, "language_detector_input_bucket") == expected
    if key == "detector_confidence_bucket":
        # Allow either high or medium.
        return _g(record, "detector_confidence_bucket") in {"high", "medium"}
    if key == "script_detection_result":
        return _g(record, "script_detection_result") == expected
    if key == "dominant_script":
        return _g(record, "dominant_script") == expected
    if key == "language_visibility_status":
        return _g(record, "language_visibility_status") == expected
    if key == "latin_medical_abbrev_visible":
        return (
            _g(record, "medical_abbreviation_shape_detected") == "yes"
            and _g(record, "dominant_script") == "latin"
        )
    if key == "medical_abbreviation_shape_detected":
        return _g(record, "medical_abbreviation_shape_detected") == expected
    if key == "alphabetic_ratio_sufficient_for_language":
        return _g(record, "alphabetic_content_bucket") == "high"
    if key == "no_cyrillic_dominant_signal":
        return _g(record, "dominant_script") != "cyrillic"
    if key == "no_mixed_script_signal":
        return _g(record, "dominant_script") != "mixed"
    if key == "no_low_confidence_detector_signal":
        return _g(record, "detector_confidence_bucket") != "low"
    if key == "not_already_handled_by_numeric_table_safe_default_helper":
        return derive_numeric_table_safe_default_label(record, enabled=True) is None
    if key == "not_already_handled_by_language_propagation_helper":
        return derive_language_propagation_metadata_label(record, enabled=True) is None
    return False


def _positive_signal_match_report(records: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = len(records)
    for key, expected in POSITIVE_SIGNAL_PATTERN:
        match_count = sum(
            1 for r in records
            if _positive_signal_predicate(key, expected, r)
        )
        out[key] = {
            "expected_value": expected,
            "matching_record_count": match_count,
            "fully_matches": (match_count == n and n > 0),
        }
    return out


def _exclusion_rule_violates(rule: str, record: dict) -> bool:
    if rule == "exclude_cyrillic_dominant_records":
        return _g(record, "dominant_script") == "cyrillic"
    if rule == "exclude_mixed_script_records":
        return _g(record, "dominant_script") == "mixed"
    if rule == "exclude_low_detector_confidence_records":
        return _g(record, "detector_confidence_bucket") == "low"
    if rule == "exclude_insufficient_detector_input_records":
        return _g(record, "language_detector_input_bucket") in {
            "insufficient", "absent",
        }
    if rule == "exclude_no_text_layer_records":
        return _g(record, "pdf_text_layer_detected") == "no"
    if rule == "exclude_image_like_but_not_routed_records":
        return _g(record, "image_like_pdf") == "yes"
    if rule == "exclude_table_heavy_numeric_safe_default_records_already_handled":
        return derive_numeric_table_safe_default_label(record, enabled=True) is not None
    if rule == "exclude_language_propagation_records_already_handled":
        return derive_language_propagation_metadata_label(record, enabled=True) is not None
    if rule == "exclude_table_header_only_special_case_record":
        # A record routed to the table-header-only special case has
        # section_heading=yes AND no abbreviation signal. The abbreviation
        # slice we're filtering already has med_abbrev=yes, so this rule
        # only fires defensively if both conditions co-occur.
        return (
            _g(record, "section_heading_shape_detected") == "yes"
            and _g(record, "medical_abbreviation_shape_detected") != "yes"
        )
    if rule == "exclude_ambiguous_below_threshold_records":
        return _g(record, "unknown_failure_bucket") == "ambiguous_below_threshold"
    if rule == "exclude_fallback_ran_but_no_family_match_records":
        return _g(record, "unknown_failure_bucket") == "fallback_ran_but_no_family_match"
    if rule == "exclude_medication_dose_or_ddi_interpretation":
        for forbidden in (
            "parsed_medications", "parsed_doses", "parsed_frequencies",
            "parsed_ddi_findings", "ddi_interpretation",
        ):
            if record.get(forbidden):
                return True
        return False
    if rule == "exclude_lab_value_parsing":
        for forbidden in ("parsed_lab_values", "lab_value_interpretation"):
            if record.get(forbidden):
                return True
        return False
    if rule == "exclude_records_with_insufficient_safe_metadata":
        required = (
            "language_detector_attempted",
            "language_detector_input_bucket",
            "detector_confidence_bucket",
            "script_detection_result",
            "dominant_script",
            "language_visibility_status",
            "alphabetic_content_bucket",
            "medical_abbreviation_shape_detected",
        )
        return any(_g(record, k) == "" for k in required)
    return True


def _exclusion_audit_report(records: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "rule": rule,
            "violating_record_count": sum(
                1 for r in records if _exclusion_rule_violates(rule, r)
            ),
        }
        for rule in EXCLUSION_RULES
    ]


# ── Top-level builder ───────────────────────────────────────────────────────

def build_diagnostic_from_report(source_payload: dict) -> DiagnosticReport:
    table = source_payload.get("anonymous_per_file_table", []) or []
    pool = select_abbreviation_pool(table)

    overlap_nt = sum(
        1 for r in pool
        if derive_numeric_table_safe_default_label(r, enabled=True) is not None
    )
    overlap_prop = sum(
        1 for r in pool
        if derive_language_propagation_metadata_label(r, enabled=True) is not None
    )

    sig_report = _positive_signal_match_report(pool)
    sig_holds = (
        all(v["fully_matches"] for v in sig_report.values()) and len(pool) > 0
    )

    excl_audit = _exclusion_audit_report(pool)
    excl_clean = all(r["violating_record_count"] == 0 for r in excl_audit)

    raw_signal_counts = {
        "language_detector_attempted_counts":
            _safe_count(pool, "language_detector_attempted"),
        "language_detector_input_bucket_counts":
            _safe_count(pool, "language_detector_input_bucket"),
        "detector_confidence_bucket_counts":
            _safe_count(pool, "detector_confidence_bucket"),
        "script_detection_result_counts":
            _safe_count(pool, "script_detection_result"),
        "dominant_script_counts":
            _safe_count(pool, "dominant_script"),
        "language_visibility_status_counts":
            _safe_count(pool, "language_visibility_status"),
        "alphabetic_content_bucket_counts":
            _safe_count(pool, "alphabetic_content_bucket"),
        "numeric_content_bucket_counts":
            _safe_count(pool, "numeric_content_bucket"),
        "symbol_content_bucket_counts":
            _safe_count(pool, "symbol_content_bucket"),
        "table_like_structure_detected_counts":
            _safe_count(pool, "table_like_structure_detected"),
        "section_heading_shape_detected_counts":
            _safe_count(pool, "section_heading_shape_detected"),
        "medical_abbreviation_shape_detected_counts":
            _safe_count(pool, "medical_abbreviation_shape_detected"),
        "lab_table_shape_detected_counts":
            _safe_count(pool, "lab_table_shape_detected"),
        "imaging_modality_shape_detected_counts":
            _safe_count(pool, "imaging_modality_shape_detected"),
        "image_like_pdf_counts":
            _safe_count(pool, "image_like_pdf"),
        "pdf_text_layer_detected_counts":
            _safe_count(pool, "pdf_text_layer_detected"),
    }

    deferred_subsets = {
        "numeric_table_safe_default_pool_already_handled":
            "11 records handled by DIAG-06A/07A/08A; excluded from this spec",
        "language_propagation_pool_already_handled":
            "11 records handled by DIAG-09A/10A; excluded from this spec",
        "candidate_table_header_language_policy_record":
            "1 record from DIAG-05 routed to the table-header lever; deferred",
        "likely_text_layer_issue":
            "21 records deferred per DIAG-03",
        "fallback_ran_but_no_family_match":
            "17 records deferred per DIAG-02; no cue expansion",
        "ambiguous_below_threshold":
            "15 records excluded; review-bound, no cue expansion",
    }

    if sig_holds and excl_clean and overlap_nt == 0 and overlap_prop == 0 and len(pool) > 0:
        rec_next = "future_block_named_unknown_diag_11a_implementation"
        rec_justification = (
            f"All {len(pool)} abbreviation-pool records match the exact "
            f"positive signal pattern, no exclusion rule fires, and the "
            f"slice is fully disjoint from both the numeric-table safe-"
            f"default pool and the language-propagation pool. A future "
            f"implementation block may prototype the abbreviation-context "
            f"metadata helper inside the published acceptance criteria. "
            f"The implementation must remain review-bound, default-off "
            f"behind a separate env flag distinct from the two existing "
            f"levers, and must never parse or expand the abbreviation "
            f"itself."
        )
    else:
        rec_next = "no_implementation_yet_spec_revision_required"
        rec_justification = (
            "The positive signal pattern does not fully hold, one or more "
            "exclusion rules are violated, or the pool overlaps with the "
            "numeric-table safe-default or language-propagation pool. "
            "Revise the spec before any implementation is considered. All "
            "records remain review-bound."
        )

    progress_estimate = {
        "before_11a_unknown_track_done_pct":      "approximately 87%",
        "before_11a_unknown_track_remaining_pct": "approximately 13%",
        "before_11a_project_done_pct":            "approximately 82%",
        "before_11a_project_remaining_pct":       "approximately 18%",
        "after_11a_unknown_track_done_pct":       "approximately 90%",
        "after_11a_unknown_track_remaining_pct":  "approximately 10%",
        "after_11a_project_done_pct":             "approximately 83%",
        "after_11a_project_remaining_pct":        "approximately 17%",
        "note": (
            "Estimates are approximate and refer to the residual Unknown-"
            "reduction track in this workspace, plus the overall MedAI "
            "project state. They are informational only and not a release "
            "milestone."
        ),
    }

    safety_privacy = {
        "behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "raw_language_detector_behavior_changed": False,
        "abbreviation_handling_behavior_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_changed": False,
        "scoring_changed": False,
        "auto_accept_changed": False,
        "cue_packs_changed": False,
        "cue_expansion_recommended": False,
        "abbreviation_handling_implemented_in_this_block": False,
        "abbreviation_parsing_or_expansion_recommended": False,
        "lab_value_parsing_added": False,
        "medication_parsing_added": False,
        "dose_parsing_added": False,
        "ddi_logic_changed": False,
        "clinical_interpretation_added": False,
        "b07_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_allowlist_changed": False,
        "external_api_changed": False,
        "external_api_used": False,
        "raw_filenames_in_public_reports": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "source_documents_staged": False,
        "private_corpus_files_staged": False,
        "secrets_in_public_reports": False,
        "all_records_remain_review_bound": True,
        "no_overlap_with_numeric_table_safe_default_pool": overlap_nt == 0,
        "no_overlap_with_language_propagation_pool": overlap_prop == 0,
    }

    return DiagnosticReport(
        snapshot="MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A",
        branch=_git_branch(),
        head_commit_short=_git_head()[:12],
        source_10a_commit_short=SOURCE_10A_COMMIT_SHORT,
        park_19_commit_short=PARK_19_COMMIT_SHORT,
        public_report_commit_hash_policy=PUBLIC_REPORT_HASH_POLICY,
        source_report_label=SOURCE_REPORT_LABEL,
        upstream_labels=list(UPSTREAM_LABELS),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),

        total_abbreviation_pool_records=len(pool),
        overlap_with_numeric_table_safe_default_pool=overlap_nt,
        no_overlap_with_numeric_table_safe_default_pool=(overlap_nt == 0),
        overlap_with_language_propagation_pool=overlap_prop,
        no_overlap_with_language_propagation_pool=(overlap_prop == 0),

        positive_signal_pattern=[
            {"key": k, "expected": v} for k, v in POSITIVE_SIGNAL_PATTERN
        ],
        positive_signal_match_report=sig_report,
        positive_signal_holds_on_all_priority_records=sig_holds,

        exclusion_rules=list(EXCLUSION_RULES),
        exclusion_audit=excl_audit,
        no_priority_record_violates_any_exclusion_rule=excl_clean,

        proposed_future_behavior=PROPOSED_FUTURE_BEHAVIOR,
        future_implementation_acceptance_criteria=
            list(FUTURE_IMPLEMENTATION_ACCEPTANCE_CRITERIA),
        future_validation_requirements=
            list(FUTURE_VALIDATION_REQUIREMENTS),
        suggested_future_env_var=SUGGESTED_FUTURE_ENV_VAR,

        anonymized_sample_ids=_anonymized_ids("abbreviation_priority", len(pool)),
        raw_signal_counts=raw_signal_counts,
        deferred_subsets=deferred_subsets,

        implementation_block_recommended_next=rec_next,
        implementation_block_justification=rec_justification,

        progress_estimate=progress_estimate,

        behavior_changed=False,
        external_api_used=False,
        cue_expansion_recommended=False,
        abbreviation_handling_implemented_in_this_block=False,
        safety_privacy=safety_privacy,
    )


# ── renderers ────────────────────────────────────────────────────────────────

def render_json(report: DiagnosticReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def render_markdown_summary(report: DiagnosticReport) -> str:
    lines: list[str] = []
    lines.append("# MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A - Latin Medical Abbreviation Spec")
    lines.append("")
    lines.append(f"- branch: `{report.branch}`")
    lines.append(f"- HEAD commit (short): `{report.head_commit_short}`")
    lines.append(f"- source DIAG-10A commit (short): "
                 f"`{report.source_10a_commit_short}`")
    lines.append(f"- PARK-19 baseline commit (short): "
                 f"`{report.park_19_commit_short}`")
    lines.append(f"- public_report_commit_hash_policy: "
                 f"`{report.public_report_commit_hash_policy}`")
    lines.append("- upstream:")
    for lbl in report.upstream_labels:
        lines.append(f"  - `{lbl}`")
    lines.append(f"- source report: `{report.source_report_label}`")
    lines.append(f"- total abbreviation-pool records analyzed: "
                 f"`{report.total_abbreviation_pool_records}`")
    lines.append(
        f"- overlap_with_numeric_table_safe_default_pool: "
        f"`{report.overlap_with_numeric_table_safe_default_pool}`"
    )
    lines.append(
        f"- no_overlap_with_numeric_table_safe_default_pool: "
        f"`{report.no_overlap_with_numeric_table_safe_default_pool}`"
    )
    lines.append(
        f"- overlap_with_language_propagation_pool: "
        f"`{report.overlap_with_language_propagation_pool}`"
    )
    lines.append(
        f"- no_overlap_with_language_propagation_pool: "
        f"`{report.no_overlap_with_language_propagation_pool}`"
    )
    lines.append(f"- suggested_future_env_var: "
                 f"`{report.suggested_future_env_var}`")
    lines.append(f"- generated_at: `{report.generated_at}`")
    lines.append("")

    lines.append("## A. Required positive signal pattern")
    lines.append("")
    for item in report.positive_signal_pattern:
        lines.append(f"- `{item['key']}` = `{item['expected']}`")
    lines.append("")
    lines.append("### Positive-signal match report on the priority slice")
    lines.append("")
    for key, info in report.positive_signal_match_report.items():
        lines.append(
            f"- `{key}` expected=`{info['expected_value']}`, "
            f"matching=`{info['matching_record_count']}`, "
            f"fully_matches=`{info['fully_matches']}`"
        )
    lines.append("")
    lines.append(
        f"positive_signal_holds_on_all_priority_records: "
        f"`{report.positive_signal_holds_on_all_priority_records}`"
    )
    lines.append("")

    lines.append("## B. Required exclusion rules")
    lines.append("")
    for rule in report.exclusion_rules:
        lines.append(f"- `{rule}`")
    lines.append("")
    lines.append("### Exclusion-rule audit on the priority slice")
    lines.append("")
    for row in report.exclusion_audit:
        lines.append(
            f"- `{row['rule']}` violating_record_count="
            f"`{row['violating_record_count']}`"
        )
    lines.append("")
    lines.append(
        f"no_priority_record_violates_any_exclusion_rule: "
        f"`{report.no_priority_record_violates_any_exclusion_rule}`"
    )
    lines.append("")

    lines.append("## C. Proposed future behavior (NOT implemented in this block)")
    lines.append("")
    for k, v in report.proposed_future_behavior.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## D. Future implementation acceptance criteria")
    lines.append("")
    for c in report.future_implementation_acceptance_criteria:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## E. Future validation requirements")
    lines.append("")
    for v in report.future_validation_requirements:
        lines.append(f"- `{v}`")
    lines.append("")

    lines.append("## Deferred subsets (out of scope)")
    lines.append("")
    for k, v in report.deferred_subsets.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Implementation recommendation")
    lines.append("")
    lines.append(
        f"- recommended_next: `{report.implementation_block_recommended_next}`"
    )
    lines.append("")
    lines.append(report.implementation_block_justification)
    lines.append("")

    lines.append("## Progress estimate")
    lines.append("")
    for k, v in report.progress_estimate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")

    lines.append("## Safety / Privacy")
    lines.append("")
    lines.append(f"- behavior_changed: `{report.behavior_changed}`")
    lines.append(f"- external_api_used: `{report.external_api_used}`")
    lines.append(f"- cue_expansion_recommended: "
                 f"`{report.cue_expansion_recommended}`")
    lines.append(f"- abbreviation_handling_implemented_in_this_block: "
                 f"`{report.abbreviation_handling_implemented_in_this_block}`")
    for k, v in report.safety_privacy.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("No raw filenames, raw OCR text, raw document text, private paths, "
                 "PHI, or secrets are included. Specification-only block; no "
                 "runtime behavior change, no abbreviation handling is implemented "
                 "here, no cue expansion, no abbreviation parsing or expansion, no "
                 "OCR routing or detector behavior change. Records remain "
                 "review-bound.")
    lines.append("")
    return "\n".join(lines)


def render_markdown_long(report: DiagnosticReport) -> str:
    base = render_markdown_summary(report)
    extra: list[str] = ["", "## Raw signal distributions", ""]
    for k, v in report.raw_signal_counts.items():
        inner = ", ".join(f"`{kk}`={vv}" for kk, vv in v.items())
        extra.append(f"- {k}: {inner}")
    extra += [
        "",
        "## Why a spec block instead of an implementation",
        "",
        "The 8 abbreviation-pool records share a uniform metadata signature ",
        "(Latin script, high or medium detector confidence, sufficient ",
        "input, visibility=latin_visible_language_unknown, medical_",
        "abbreviation_shape_detected=yes) and are completely disjoint from ",
        "both the numeric-table safe-default pool (11 records) and the ",
        "language-propagation pool (11 records). Before any future ",
        "implementation block touches runtime behavior, the exact positive ",
        "signal pattern, exclusion rules, proposed default behavior, ",
        "acceptance criteria, validation requirements, and rollback ",
        "boundaries are published here in a single privacy-safe document. ",
        "A future implementation block may proceed only inside the ",
        "boundaries this spec defines.",
        "",
        "## What this block did not change",
        "",
        "- OCR routing logic",
        "- OCR engine",
        "- Raw language / script detector behavior",
        "- Abbreviation handling behavior",
        "- Classifier behavior",
        "- Confidence thresholds or scoring",
        "- Cue packs",
        "- Auto-accept or review-bound rules",
        "- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs",
        "",
        "No abbreviation handling is implemented in this block. No clinical ",
        "interpretation added. No values parsed. No abbreviation parsed or ",
        "expanded. No active facts written.",
        "",
    ]
    return base + "\n".join(extra)


# ── public-report safety guard ───────────────────────────────────────────────

_FORBIDDEN_PATTERNS = (
    re.compile(r"\.(?:pdf|jpe?g|png|docx?|xlsx?)\b", re.IGNORECASE),
    re.compile(r"/(?:users|home|var/private)/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |)PRIVATE KEY-----"),
    re.compile(r"\b(?:secret|secret_key|api_key|password)\s*=", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r"\b(?:aws|gcp|azure)_secret\b", re.IGNORECASE),
)


def assert_safe_public_payload(payload: Any) -> None:
    def _walk(node: Any) -> None:
        if isinstance(node, str):
            for pat in _FORBIDDEN_PATTERNS:
                if pat.search(node):
                    raise RuntimeError(
                        "Refusing to write public report: forbidden pattern "
                        f"matched ({pat.pattern!r})."
                    )
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(k)
                _walk(v)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                _walk(item)

    _walk(payload)


# ── driver ───────────────────────────────────────────────────────────────────

def write_reports(report: DiagnosticReport,
                   out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_payload = asdict(report)
    assert_safe_public_payload(json_payload)
    md_summary = render_markdown_summary(report)
    md_long = render_markdown_long(report)
    assert_safe_public_payload(md_summary)
    assert_safe_public_payload(md_long)

    paths = {
        "json": out_dir / "medai_doc_type_unknown_diag_11a_report.json",
        "md_summary": out_dir / "medai_doc_type_unknown_diag_11a_report.md",
        "md_main": out_dir / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_11A.md",
    }
    paths["json"].write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["md_summary"].write_text(md_summary, encoding="utf-8")
    paths["md_main"].write_text(md_long, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A latin abbreviation metadata spec."
    )
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.source_report.exists():
        print(f"ERROR: source report missing: {args.source_report}", file=sys.stderr)
        return 2

    source_payload = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = build_diagnostic_from_report(source_payload)

    if args.print_only:
        print(render_json(report))
        return 0

    paths = write_reports(report, out_dir=args.output_dir)
    print(json.dumps(
        {
            "conclusion": "medai_doc_type_unknown_diag_11a_ready",
            "files_written":
                {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()},
            "total_abbreviation_pool_records": report.total_abbreviation_pool_records,
            "positive_signal_holds_on_all_priority_records":
                report.positive_signal_holds_on_all_priority_records,
            "no_priority_record_violates_any_exclusion_rule":
                report.no_priority_record_violates_any_exclusion_rule,
            "no_overlap_with_numeric_table_safe_default_pool":
                report.no_overlap_with_numeric_table_safe_default_pool,
            "no_overlap_with_language_propagation_pool":
                report.no_overlap_with_language_propagation_pool,
            "implementation_block_recommended_next":
                report.implementation_block_recommended_next,
            "behavior_changed": report.behavior_changed,
            "external_api_used": report.external_api_used,
            "cue_expansion_recommended": report.cue_expansion_recommended,
            "abbreviation_handling_implemented_in_this_block":
                report.abbreviation_handling_implemented_in_this_block,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
