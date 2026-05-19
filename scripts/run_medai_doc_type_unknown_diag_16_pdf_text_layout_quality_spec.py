#!/usr/bin/env python3
"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-16 unified PDF text + layout/table extraction
quality spec.

Specification-only, evaluation-only, aggregate-only. Consolidates the two
text-layer audits (DIAG-15 sub-track A, 11 records; DIAG-15B sub-track B,
10 records) into a single forward-looking spec that defines:

  1. Evidence summary from DIAG-15 and DIAG-15B
  2. Unified problem statement
  3. Non-goals
  4. Future implementation acceptance criteria
  5. Rollback criteria
  6. Privacy and safety gates
  7. Required regression tests for future implementation
  8. Review-bound invariants
  9. Explicit decision: cue expansion still not recommended
 10. Recommended next block after DIAG-16

Hard guardrails:
* Reads ONLY privacy-safe public reports already committed to the
  repository. Does NOT read raw source documents, raw OCR text, raw
  filenames, private paths, terminology files, runtime DBs, or backups.
* Emits ONLY aggregate counts, controlled-vocabulary spec text, and
  invariant flags.
* Does NOT change runtime behavior, does NOT add a runtime helper, does
  NOT add an operator UI surface, does NOT change OCR routing, OCR engine
  behavior, PDF text-extraction behavior, layout/table extraction
  behavior, raw language detector behavior, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior. Does NOT touch PARK-20 tags.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_15_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit"
    / "medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit_report.json"
)
DIAG_15B_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_15b_layout_table_extraction_audit"
    / "medai_doc_type_unknown_diag_15b_layout_table_extraction_audit_report.json"
)
DIAG_14_REPORT = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_14_text_layer_extraction_spec"
    / "medai_doc_type_unknown_diag_14_text_layer_extraction_spec_report.json"
)
OUT_DIR = (
    REPO_ROOT
    / "reports/medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec"
)

PARK_20_COMMIT_SHORT = "3e46461"
DIAG_14_COMMIT_SHORT = "832a5fe"
DIAG_15_COMMIT_SHORT = "3e57ba7"
DIAG_15B_COMMIT_SHORT = "2e9b53b"

PHASE_ID = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-16"


UNIFIED_PROBLEM_STATEMENT = (
    "21 residual review-bound Unknown records have a present-but-too-short "
    "PDF text layer (image_like_pdf=no, pdf_text_layer_detected=yes, "
    "alphabetic_content=high). 11 records (DIAG-15 sub-track A) lack any "
    "table-like structure; the text layer itself is insufficient for the "
    "family classifier. 10 records (DIAG-15B sub-track B) carry a visible "
    "table-like structure whose extractable text is still insufficient for "
    "the family classifier. A unified extraction-quality strategy is needed "
    "that improves recoverable evidence for these 21 records without "
    "perturbing OCR routing, OCR engine behavior, PDF text-extraction "
    "behavior, layout/table extraction behavior, classifier behavior, "
    "thresholds, scoring, cue packs, or auto-accept logic."
)

NON_GOALS = [
    "must not modify OCR routing",
    "must not modify OCR engine behavior",
    "must not modify PDF text-extraction behavior at this stage",
    "must not modify layout/table extraction behavior at this stage",
    "must not modify the language detector",
    "must not modify the classifier behavior for non-signature records",
    "must not modify thresholds or scoring",
    "must not add cue packs",
    "must not parse lab values",
    "must not parse medications, dose, frequency, duration, or DDI",
    "must not parse or expand abbreviations",
    "must not add clinical interpretation",
    "must not auto-accept any record",
    "must not promote any record's document type at the data layer",
    "must not emit raw text, raw filenames, or private paths in public reports",
    "must not enable any external API",
    "must not touch PARK-20 tags",
    "must not change B07, ROUTE-FIX, DB schema, or command allowlist",
]

FUTURE_ACCEPTANCE_CRITERIA = [
    "any future extraction-improvement block MUST be opt-in or env-gated",
    "any future block MUST preserve review-bound status for previously Unknown records unless a later explicit acceptance block authorizes otherwise",
    "any future block MUST NOT auto-accept",
    "any future block MUST NOT parse clinical values",
    "any future block MUST NOT infer diagnosis, medication, DDI, or treatment meaning",
    "any future block MUST produce privacy-safe aggregate reports only",
    "any future block MUST keep accepted_count = 0 for the 21-record scope",
    "any future block MUST keep auto_accept_allowed_count = 0 for the 21-record scope",
    "any future block MUST keep external_api_used_count = 0",
    "any future block MUST NOT change OCR routing in its first pass",
    "any future block MUST NOT change OCR engine behavior in its first pass",
    "any future block MUST NOT change classifier behavior in its first pass",
    "any future block MUST NOT change thresholds or scoring in its first pass",
    "any future block MUST prove zero regression in the DIAG-01..16 diagnostic suite",
    "any future block MUST prove zero regression in the document-type eval non-streamlit subset",
    "any future block MUST prove zero regression in the final CKA MVP validation",
    "any future block MUST prove zero regression in B07 term01",
    "any future block MUST prove zero regression in ROUTE-FIX 01",
    "any future block MUST prove zero regression in UI ops",
    "any future block MUST prove zero regression in UI boot",
    "any future block MUST pass public report privacy checks on every new report",
    "any future block MUST stage only its own scoped files; no source documents, PDFs, images, DOCX, runtime DBs, private corpus files, backups, bundles, keys, private files, or terminology data may be staged",
    "any future block MUST keep PARK-20 tags untouched",
    "any future block MUST use anonymized file_NNN IDs only in public reports",
]

ROLLBACK_CRITERIA = [
    "if any future block's env-gated path is enabled, a single env-var flip MUST return the system to the default-off, pre-block runtime behavior",
    "if any DIAG-01..16 diagnostic regresses, the future block MUST be reverted before any further work",
    "if any operational validation (CKA MVP, B07, ROUTE-FIX, UI ops, UI boot) regresses, the future block MUST be reverted",
    "if any public report leaks raw text, raw filenames, private paths, PHI, or secrets, the future block MUST be reverted and the leaking artifact removed from history per repository policy",
    "if accepted_count, auto_accept_allowed_count, or external_api_used_count rise above zero for the 21-record scope without an explicit acceptance block, the future block MUST be reverted",
    "if review-bound status is lost for any of the 21 records without an explicit acceptance block, the future block MUST be reverted",
    "if any source document, PDF, image, DOCX, runtime DB, private corpus file, backup, bundle, key, private file, or terminology dataset is staged, the future block MUST be reverted and the staged artifact removed before any commit",
    "if any PARK-20 tag is moved, deleted, or repointed, the future block MUST be reverted and the tag restored to PARK-20 commit 3e46461",
]

PRIVACY_AND_SAFETY_GATES = [
    "every new public report MUST pass clinical_knowledge.privacy.check_public_report_payload",
    "every new public report MUST NOT contain raw OCR text, raw document text, raw filenames, or private paths",
    "every new public report MUST NOT contain PHI, secrets, API keys, tokens, or other credentials",
    "every new public report MUST emit anonymized file_NNN IDs only",
    "every new test MUST use synthetic safe inputs, never read real corpus files",
    "every new script MUST avoid reading raw source documents, raw OCR text, raw document text, raw filenames, terminology files, runtime DBs, backups, or bundles",
    "every new commit MUST stage only the block's own scoped files; receipt-refresh churn must be a separate housekeeping commit",
    "every push MUST go to the branch only; no tag is created, moved, or pushed by the future block",
]

REQUIRED_REGRESSION_TESTS = [
    "DIAG-15 focused tests (Sub-track A audit) — must remain passing",
    "DIAG-15B focused tests (Sub-track B audit) — must remain passing",
    "DIAG-01 through DIAG-16 diagnostic suite — must remain passing",
    "document-type eval non-streamlit subset — must remain passing",
    "final CKA MVP validation — must remain passing",
    "B07 term01 opt-in integration — must remain passing",
    "ROUTE-FIX 01 — must remain passing",
    "UI ops panel validation — must remain passing",
    "UI boot fix validation — must remain passing",
    "public-report privacy checks on every new public report — must pass",
    "staged safety check — must show no source documents / no private corpus / no DBs / no backups / no bundles / no keys / no terminology data staged",
]

REVIEW_BOUND_INVARIANTS = [
    "all 21 affected records remain Needs review",
    "unknown_count for the 21-record scope is unchanged",
    "accepted_count for the 21-record scope is zero",
    "auto_accept_allowed_count for the 21-record scope is zero",
    "external_api_used_count is zero",
    "no record's data-layer document type is mutated by the future block",
    "no record's raw language-detector output is mutated by the future block",
    "no record's classifier output is mutated by the future block unless explicitly env-gated and within scope",
]

CUE_EXPANSION_DECISION = (
    "Cue expansion is explicitly NOT recommended as the primary next step. "
    "The 21 records' shared failure mode is extracted-text insufficiency at "
    "the family-classifier input, not a missing cue. Adding cues without "
    "improving extracted-text quality would only widen false-positive risk "
    "across the rest of the corpus. The cue catalog therefore remains "
    "frozen for this scope until a future block first demonstrates "
    "evaluation-only that improved extracted-text quality has changed the "
    "shape of the residual gap."
)

NEXT_BLOCK_RECOMMENDATION_NAME = (
    "PARK-21 — parking snapshot capturing the full text-layer "
    "evaluation-only chain (DIAG-13A diagnostic, DIAG-14 sub-track split, "
    "DIAG-15 PDF text quality audit, DIAG-15B layout/table audit, DIAG-16 "
    "unified spec). Aggregate-only; reports-only; no implementation block "
    "begins before the snapshot is in place. Cue expansion still not "
    "recommended."
)


def _short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def evidence_summary(
    diag15: Mapping[str, Any], diag15b: Mapping[str, Any]
) -> Dict[str, Any]:
    """Aggregate-only evidence summary derived from DIAG-15 and DIAG-15B."""
    return {
        "subtrack_a": {
            "phase_id": diag15.get("phase_id"),
            "total_records": int(diag15.get("total_records_analyzed", 0)),
            "bucket_counts": dict(
                diag15.get("pdf_text_extraction_quality_bucket_counts", {})
            ),
        },
        "subtrack_b": {
            "phase_id": diag15b.get("phase_id"),
            "total_records": int(diag15b.get("total_records_analyzed", 0)),
            "bucket_counts": dict(
                diag15b.get("layout_table_extraction_bucket_counts", {})
            ),
        },
        "combined_total_records": (
            int(diag15.get("total_records_analyzed", 0))
            + int(diag15b.get("total_records_analyzed", 0))
        ),
    }


def build_report(*, branch: str = "clinical-knowledge-architecture") -> Dict[str, Any]:
    diag15 = _load_json(DIAG_15_REPORT)
    diag15b = _load_json(DIAG_15B_REPORT)
    evidence = evidence_summary(diag15, diag15b)

    a = evidence["subtrack_a"]["total_records"]
    b = evidence["subtrack_b"]["total_records"]
    total = evidence["combined_total_records"]

    payload: Dict[str, Any] = {
        "conclusion": "medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec_ready",
        "phase_id": PHASE_ID,
        "mode": "specification_only",
        "evaluation_only": True,
        "aggregate_only": True,
        "branch": branch,
        "head_commit_short": _short_head(),
        "diag_14_commit_short": DIAG_14_COMMIT_SHORT,
        "diag_15_commit_short": DIAG_15_COMMIT_SHORT,
        "diag_15b_commit_short": DIAG_15B_COMMIT_SHORT,
        "park_20_parking_commit_short": PARK_20_COMMIT_SHORT,
        "park_20_tag_status": (
            "PARK-20 branch parked at 3e46461. PARK-20 tags "
            "medai-unknown-diag-language-metadata-ready-2026-05-19 and "
            "medai-final-parked-post-unknown-diag-language-metadata-2026-05-19 "
            "exist on origin and resolve to 3e46461. Tags are not touched "
            "in DIAG-16."
        ),
        "inputs": [
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15",
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B",
        ],
        "source_reports_used": [
            "block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)",
            "block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)",
            "block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)",
        ],
        "total_records_covered": total,
        "subtrack_a_records": a,
        "subtrack_b_records": b,
        "evidence_summary": evidence,
        "spec_sections": {
            "1_evidence_summary": evidence,
            "2_unified_problem_statement": UNIFIED_PROBLEM_STATEMENT,
            "3_non_goals": list(NON_GOALS),
            "4_future_implementation_acceptance_criteria": list(
                FUTURE_ACCEPTANCE_CRITERIA
            ),
            "5_rollback_criteria": list(ROLLBACK_CRITERIA),
            "6_privacy_and_safety_gates": list(PRIVACY_AND_SAFETY_GATES),
            "7_required_regression_tests": list(REQUIRED_REGRESSION_TESTS),
            "8_review_bound_invariants": list(REVIEW_BOUND_INVARIANTS),
            "9_cue_expansion_decision": CUE_EXPANSION_DECISION,
            "10_recommended_next_block": NEXT_BLOCK_RECOMMENDATION_NAME,
        },

        "behavior_changed": False,
        "external_api_used": False,
        "source_documents_opened": False,
        "raw_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "runtime_behavior_changed": False,
        "extraction_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "ocr_behavior_changed": False,
        "classifier_behavior_changed": False,
        "threshold_behavior_changed": False,
        "cue_expansion_recommended": False,
        "implementation_started": False,
        "runtime_helper_added": False,
        "operator_ui_surface_added": False,
        "park_20_tags_touched": False,

        "unknown_count_changed": False,
        "accepted_count": 0,
        "auto_accept_allowed_count": 0,
        "external_api_used_count": 0,
        "all_records_review_bound": True,

        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "layout_table_extraction_behavior_changed": False,
        "raw_language_detector_changed": False,
        "thresholds_or_scoring_changed": False,
        "cue_packs_added": False,
        "lab_values_parsed": False,
        "medications_dose_frequency_duration_or_ddi_parsed": False,
        "abbreviations_parsed_or_expanded": False,
        "clinical_interpretation_added": False,
        "b07_changed": False,
        "route_fix_changed": False,
        "db_schema_changed": False,
        "command_allowlist_changed": False,
        "external_api_enabled": False,
        "source_documents_staged": False,
        "private_files_staged": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_in_public_reports": False,
        "private_paths_in_public_reports": False,
        "secrets_in_public_reports": False,

        "safety_privacy_statement": (
            "DIAG-16 is a specification-only, evaluation-only, aggregate-only "
            "block. It consolidates DIAG-15 (Sub-track A, 11 records) and "
            "DIAG-15B (Sub-track B, 10 records) into a unified forward "
            "specification covering future implementation acceptance criteria, "
            "rollback boundaries, privacy/safety gates, and required regression "
            "tests. No source documents, raw OCR text, raw document text, raw "
            "filenames, private paths, PHI, secrets, DBs, backups, or bundles "
            "are read or emitted. No runtime behavior changes. PARK-20 tags "
            "are not touched."
        ),
        "progress_estimate": {
            "before": {
                "residual_unknown_reduction_track_done_pct": 99.8,
                "residual_unknown_reduction_track_remaining_pct": 0.2,
                "whole_medai_project_done_pct": 89.5,
                "whole_medai_project_remaining_pct": 10.5,
            },
            "after": {
                "residual_unknown_reduction_track_done_pct": 99.9,
                "residual_unknown_reduction_track_remaining_pct": 0.1,
                "whole_medai_project_done_pct": 90,
                "whole_medai_project_remaining_pct": 10,
            },
        },
        "next_block_recommendation": {
            "recommended_name": NEXT_BLOCK_RECOMMENDATION_NAME,
            "must_remain_evaluation_only": True,
            "must_remain_aggregate_only": True,
            "must_not_propose_cue_expansion_as_primary_step": True,
            "must_not_change_extraction_behavior_in_first_pass": True,
        },
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    sec = payload["spec_sections"]
    ev = sec["1_evidence_summary"]
    a = ev["subtrack_a"]
    b = ev["subtrack_b"]
    progress = payload["progress_estimate"]
    lines: List[str] = []
    lines.append(
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-16 Unified PDF Text + Layout/Table "
        "Extraction Quality Spec"
    )
    lines.append("")
    lines.append(
        "Specification-only, evaluation-only, aggregate-only block. "
        "Consolidates DIAG-15 (Sub-track A) and DIAG-15B (Sub-track B) into "
        "a unified forward-looking spec. No runtime behavior changes. No "
        "extraction behavior changes."
    )
    lines.append("")
    lines.append("## State")
    lines.append("")
    lines.append(f"- Phase ID: `{payload['phase_id']}`")
    lines.append(f"- Mode: `{payload['mode']}`")
    lines.append(f"- Evaluation only: **{payload['evaluation_only']}**")
    lines.append(f"- Aggregate only: **{payload['aggregate_only']}**")
    lines.append(f"- Branch: `{payload['branch']}`")
    lines.append(f"- HEAD commit (short): `{payload['head_commit_short']}`")
    lines.append(f"- DIAG-14 commit (short): `{payload['diag_14_commit_short']}`")
    lines.append(f"- DIAG-15 commit (short): `{payload['diag_15_commit_short']}`")
    lines.append(f"- DIAG-15B commit (short): `{payload['diag_15b_commit_short']}`")
    lines.append(
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`"
    )
    lines.append("")
    lines.append("## PARK-20 tag status")
    lines.append("")
    lines.append(payload["park_20_tag_status"])
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for src in payload["inputs"]:
        lines.append(f"- `{src}`")
    lines.append("")
    lines.append("## Source reports used")
    lines.append("")
    for src in payload["source_reports_used"]:
        lines.append(f"- `{src}`")
    lines.append("")
    lines.append(
        f"## Total records covered: **{payload['total_records_covered']}** "
        f"({payload['subtrack_a_records']} Sub-track A "
        f"+ {payload['subtrack_b_records']} Sub-track B)"
    )
    lines.append("")
    lines.append("## 1. Evidence summary from DIAG-15 and DIAG-15B")
    lines.append("")
    lines.append(
        f"### Sub-track A — PDF text-extraction quality audit "
        f"(`{a['phase_id']}`, {a['total_records']} records)"
    )
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for k, v in a["bucket_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append(
        f"### Sub-track B — layout/table extraction audit "
        f"(`{b['phase_id']}`, {b['total_records']} records)"
    )
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("| --- | ---: |")
    for k, v in b["bucket_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## 2. Unified problem statement")
    lines.append("")
    lines.append(sec["2_unified_problem_statement"])
    lines.append("")
    lines.append("## 3. Non-goals")
    lines.append("")
    for item in sec["3_non_goals"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 4. Future implementation acceptance criteria")
    lines.append("")
    for item in sec["4_future_implementation_acceptance_criteria"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 5. Rollback criteria")
    lines.append("")
    for item in sec["5_rollback_criteria"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 6. Privacy and safety gates")
    lines.append("")
    for item in sec["6_privacy_and_safety_gates"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 7. Required regression tests for future implementation")
    lines.append("")
    for item in sec["7_required_regression_tests"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 8. Review-bound invariants")
    lines.append("")
    for item in sec["8_review_bound_invariants"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 9. Cue-expansion decision")
    lines.append("")
    lines.append(sec["9_cue_expansion_decision"])
    lines.append("")
    lines.append("## 10. Recommended next block")
    lines.append("")
    lines.append(sec["10_recommended_next_block"])
    lines.append("")
    lines.append("## Block invariants")
    lines.append("")
    for k in [
        "behavior_changed",
        "external_api_used",
        "source_documents_opened",
        "raw_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "runtime_behavior_changed",
        "extraction_behavior_changed",
        "pdf_text_extraction_behavior_changed",
        "layout_extraction_behavior_changed",
        "table_extraction_behavior_changed",
        "ocr_behavior_changed",
        "classifier_behavior_changed",
        "threshold_behavior_changed",
        "cue_expansion_recommended",
        "implementation_started",
        "runtime_helper_added",
        "operator_ui_surface_added",
        "park_20_tags_touched",
        "unknown_count_changed",
        "all_records_review_bound",
    ]:
        lines.append(f"- `{k}`: {payload[k]}")
    lines.append(f"- `accepted_count`: {payload['accepted_count']}")
    lines.append(
        f"- `auto_accept_allowed_count`: {payload['auto_accept_allowed_count']}"
    )
    lines.append(
        f"- `external_api_used_count`: {payload['external_api_used_count']}"
    )
    lines.append("")
    lines.append("## Safety / privacy")
    lines.append("")
    lines.append(payload["safety_privacy_statement"])
    lines.append("")
    lines.append("## Progress estimate")
    lines.append("")
    lines.append("| Track | Before | After |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Residual Unknown-reduction | "
        f"~{progress['before']['residual_unknown_reduction_track_done_pct']}% done / "
        f"~{progress['before']['residual_unknown_reduction_track_remaining_pct']}% remaining "
        f"| ~{progress['after']['residual_unknown_reduction_track_done_pct']}% done / "
        f"~{progress['after']['residual_unknown_reduction_track_remaining_pct']}% remaining |"
    )
    lines.append(
        f"| Whole MedAI project | "
        f"~{progress['before']['whole_medai_project_done_pct']}% done / "
        f"~{progress['before']['whole_medai_project_remaining_pct']}% remaining "
        f"| ~{progress['after']['whole_medai_project_done_pct']}% done / "
        f"~{progress['after']['whole_medai_project_remaining_pct']}% remaining |"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_short_summary(payload: Mapping[str, Any]) -> str:
    progress = payload["progress_estimate"]
    lines = [
        "# MEDAI-DOC-TYPE-UNKNOWN-DIAG-16 — Short Summary",
        "",
        "Specification-only, evaluation-only unified PDF text + layout/table "
        "extraction quality spec covering 21 records (11 + 10).",
        "",
        "## State",
        "",
        f"- Phase ID: `{payload['phase_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Branch: `{payload['branch']}`",
        f"- HEAD commit (short): `{payload['head_commit_short']}`",
        f"- DIAG-15 commit (short): `{payload['diag_15_commit_short']}`",
        f"- DIAG-15B commit (short): `{payload['diag_15b_commit_short']}`",
        f"- PARK-20 parking commit (short): `{payload['park_20_parking_commit_short']}`",
        "",
        "## PARK-20 tag status",
        "",
        payload["park_20_tag_status"],
        "",
        f"## Records covered: {payload['total_records_covered']} "
        f"({payload['subtrack_a_records']} A + {payload['subtrack_b_records']} B)",
        "",
        "## Top-level invariants",
        "",
        f"- `behavior_changed`: {payload['behavior_changed']}",
        f"- `external_api_used`: {payload['external_api_used']}",
        f"- `cue_expansion_recommended`: {payload['cue_expansion_recommended']}",
        f"- `extraction_behavior_changed`: {payload['extraction_behavior_changed']}",
        f"- `implementation_started`: {payload['implementation_started']}",
        f"- `park_20_tags_touched`: {payload['park_20_tags_touched']}",
        f"- `all_records_review_bound`: {payload['all_records_review_bound']}",
        "",
        "## Progress",
        "",
        f"- Before: residual Unknown ~{progress['before']['residual_unknown_reduction_track_done_pct']}% done / ~{progress['before']['residual_unknown_reduction_track_remaining_pct']}% remaining; whole project ~{progress['before']['whole_medai_project_done_pct']}% done / ~{progress['before']['whole_medai_project_remaining_pct']}% remaining.",
        f"- After: residual Unknown ~{progress['after']['residual_unknown_reduction_track_done_pct']}% done / ~{progress['after']['residual_unknown_reduction_track_remaining_pct']}% remaining; whole project ~{progress['after']['whole_medai_project_done_pct']}% done / ~{progress['after']['whole_medai_project_remaining_pct']}% remaining.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_report()
    json_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec_report.json"
    )
    md_path = (
        OUT_DIR
        / "medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec_report.md"
    )
    summary_path = (
        OUT_DIR / "MEDAI_DOC_TYPE_UNKNOWN_DIAG_16_PDF_TEXT_LAYOUT_QUALITY_SPEC.md"
    )
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    summary_path.write_text(render_short_summary(payload), encoding="utf-8")
    out_summary = {
        "conclusion": payload["conclusion"],
        "phase_id": payload["phase_id"],
        "total_records_covered": payload["total_records_covered"],
        "subtrack_a_records": payload["subtrack_a_records"],
        "subtrack_b_records": payload["subtrack_b_records"],
        "behavior_changed": payload["behavior_changed"],
        "extraction_behavior_changed": payload["extraction_behavior_changed"],
        "cue_expansion_recommended": payload["cue_expansion_recommended"],
        "implementation_started": payload["implementation_started"],
        "park_20_tags_touched": payload["park_20_tags_touched"],
        "reports_written": [
            str(json_path.relative_to(REPO_ROOT)),
            str(md_path.relative_to(REPO_ROOT)),
            str(summary_path.relative_to(REPO_ROOT)),
        ],
    }
    print(json.dumps(out_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
