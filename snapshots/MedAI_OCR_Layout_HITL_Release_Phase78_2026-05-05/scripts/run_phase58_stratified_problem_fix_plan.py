"""Phase 58 — Stratified Problem-Class Fix Plan.

Reads the Phase 57A full corpus inventory report and emits a prioritized,
PHI-safe engineering plan grouped by problem class. Pure analysis. Touches
no extraction, OCR, or classifier code.

Inputs:
  reports/phase57_full_corpus_inventory_audit/phase57_full_corpus_inventory_audit_report.json
  reports/phase57_full_corpus_inventory_audit/phase57_full_corpus_problem_clusters.json

Outputs:
  reports/phase58_stratified_problem_fix_plan/phase58_stratified_problem_fix_plan.json
  reports/phase58_stratified_problem_fix_plan/phase58_stratified_problem_fix_plan.md

Privacy guarantees: the script reads safe_file_id values only; raw filenames
or relative paths in the source report would already have failed Phase 57's
PHI-leak check. The Phase 58 outputs contain class-level counts and
safe_file_id samples only.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE57_REPORT = (
    ROOT
    / "reports"
    / "phase57_full_corpus_inventory_audit"
    / "phase57_full_corpus_inventory_audit_report.json"
)
PHASE57_CLUSTERS = (
    ROOT
    / "reports"
    / "phase57_full_corpus_inventory_audit"
    / "phase57_full_corpus_problem_clusters.json"
)
REPORT_DIR = ROOT / "reports" / "phase58_stratified_problem_fix_plan"
JSON_REPORT = REPORT_DIR / "phase58_stratified_problem_fix_plan.json"
MD_REPORT = REPORT_DIR / "phase58_stratified_problem_fix_plan.md"


# ---------------------------------------------------------------------------
# Class definitions (deterministic order)
# ---------------------------------------------------------------------------

PROBLEM_CLASS_ORDER: tuple[str, ...] = (
    "unsupported_extension",
    "empty_extraction",
    "review_ocr_quality",
    "rules_based_low_confidence",
    "image_ocr_low_quality",
    "pdf_ocr_low_quality",
    "possible_multi_document_pdf",
    "pdf_portfolio_or_embedded_files_detected",
    "possible_lab_table_failure",
    "possible_ecg_class",
    "possible_prescription_class",
    "possible_microbiology_pcr_class",
    "possible_russian_cyrillic_class",
    "unknown_other",
)


# ---------------------------------------------------------------------------
# Plan: deterministic safety/opportunity/difficulty per class.
#
# These values are engineering judgments, NOT signals derived from the
# corpus. Encoding them here makes the plan reproducible and reviewable.
# ---------------------------------------------------------------------------

CLASS_PROFILES: dict[str, dict[str, Any]] = {
    "unsupported_extension": {
        "safety_risk": "low",
        "automation_opportunity": "low",
        "implementation_difficulty": "low",
        "recommended_action_kind": "narrow_format_support",
        "rationale": (
            "Adding deterministic .docx and .rtf text extraction is well-scoped, "
            "well-understood, and produces no PHI exposure beyond what already "
            "applies to .txt. .mp3/.ogg/.msg are out of scope (audio/email)."
        ),
    },
    "empty_extraction": {
        "safety_risk": "medium",
        "automation_opportunity": "high_if_narrowly_scoped",
        "implementation_difficulty": "high",
        "recommended_action_kind": "forensic_subset_phase",
        "rationale": (
            "Largest class but root cause varies (lab table loss, vocabulary "
            "gap, sparse rules). Acting before forensic stratification would "
            "risk weakening safety gates. Run a Phase 42-style forensics on a "
            "small random subset first."
        ),
    },
    "review_ocr_quality": {
        "safety_risk": "high",
        "automation_opportunity": "low",
        "implementation_difficulty": "medium",
        "recommended_action_kind": "defer",
        "rationale": (
            "review_ocr_quality is by design a safety gate against bad OCR. "
            "Phase 45 already reconciled the false-positive prescription case. "
            "Touching the gate further requires class-specific OCR evidence, "
            "not blanket reclassification."
        ),
    },
    "rules_based_low_confidence": {
        "safety_risk": "high",
        "automation_opportunity": "medium",
        "implementation_difficulty": "high",
        "recommended_action_kind": "defer",
        "rationale": (
            "Lowering confidence thresholds or expanding rules globally would "
            "weaken the accept-vs-review boundary. Any change must be "
            "subset-driven, not corpus-driven."
        ),
    },
    "image_ocr_low_quality": {
        "safety_risk": "medium",
        "automation_opportunity": "low",
        "implementation_difficulty": "medium",
        "recommended_action_kind": "small_followup",
        "rationale": (
            "Tiny class. Phase 56 image OCR support already exists. A short "
            "diagnostic on the 5 affected files can confirm whether DPI / "
            "preprocessing is the lever, but ROI is low until volume grows."
        ),
    },
    "pdf_ocr_low_quality": {
        "safety_risk": "medium",
        "automation_opportunity": "low",
        "implementation_difficulty": "medium",
        "recommended_action_kind": "small_followup",
        "rationale": (
            "Tiny class. Likely scanned PDFs where OCR fallback is already "
            "active. A targeted diagnostic on these files can confirm whether "
            "the existing PDF OCR pipeline is doing the right thing."
        ),
    },
    "possible_multi_document_pdf": {
        "safety_risk": "high",
        "automation_opportunity": "medium",
        "implementation_difficulty": "high",
        "recommended_action_kind": "diagnostic_only_phase",
        "rationale": (
            "Splitting multi-document PDFs is not safe to do automatically "
            "without operator review. Start with a detection-only diagnostic "
            "phase that reports candidate split boundaries via safe_file_id."
        ),
    },
    "pdf_portfolio_or_embedded_files_detected": {
        "safety_risk": "high",
        "automation_opportunity": "low",
        "implementation_difficulty": "high",
        "recommended_action_kind": "diagnostic_only_phase",
        "rationale": (
            "PDF portfolios contain embedded files that may be PHI by separate "
            "consent. Do not extract embedded content automatically. Surface "
            "the count and let the operator decide."
        ),
    },
    "possible_lab_table_failure": {
        "safety_risk": "high",
        "automation_opportunity": "medium",
        "implementation_difficulty": "high",
        "recommended_action_kind": "defer",
        "rationale": (
            "Phase 40 / 41 already extended lab-row parsing twice. The "
            "current cluster is too broad (any reason code containing 'lab' "
            "or 'table'). Further expansion requires a stratified per-format "
            "subset, not a corpus-wide rewrite."
        ),
    },
    "possible_ecg_class": {
        "safety_risk": "medium",
        "automation_opportunity": "low",
        "implementation_difficulty": "medium",
        "recommended_action_kind": "defer",
        "rationale": (
            "Zero detected on this corpus. Build only when a class subset "
            "exists to validate against."
        ),
    },
    "possible_prescription_class": {
        "safety_risk": "low",
        "automation_opportunity": "low",
        "implementation_difficulty": "low",
        "recommended_action_kind": "defer",
        "rationale": (
            "Phase 43/45 already routes prescriptions correctly when detected. "
            "Zero detected on this corpus's ENGLISH classifier surface. The "
            "Cyrillic prescription path was validated on the holdout set."
        ),
    },
    "possible_microbiology_pcr_class": {
        "safety_risk": "low",
        "automation_opportunity": "low",
        "implementation_difficulty": "low",
        "recommended_action_kind": "defer",
        "rationale": (
            "Phase 43 already routes microbiology/PCR. Zero detected on this "
            "corpus's English surface."
        ),
    },
    "possible_russian_cyrillic_class": {
        "safety_risk": "medium",
        "automation_opportunity": "medium",
        "implementation_difficulty": "medium",
        "recommended_action_kind": "investigate_language_hint_signal",
        "rationale": (
            "Zero Cyrillic class flags but language_hint=unknown for the "
            "entire corpus. This is an upstream signal problem (language "
            "hint not propagating), NOT evidence of zero Cyrillic content. "
            "Worth investigating but not the highest-ROI starting point."
        ),
    },
    "unknown_other": {
        "safety_risk": "medium",
        "automation_opportunity": "low",
        "implementation_difficulty": "low",
        "recommended_action_kind": "defer",
        "rationale": (
            "Already zero on this corpus thanks to Phase 57A reconciliation. "
            "Re-evaluate only if a future inventory surfaces members."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_phase57_inputs(
    *,
    report_path: Path | None = None,
    clusters_path: Path | None = None,
) -> dict[str, Any]:
    report_path = report_path or PHASE57_REPORT
    clusters_path = clusters_path or PHASE57_CLUSTERS
    if not report_path.exists():
        raise FileNotFoundError(f"Phase 57A report not found: {report_path}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"Phase 57A clusters file not found: {clusters_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    return {"report": report, "clusters": clusters}


def class_count_for(report: dict[str, Any], clusters: dict[str, list[str]], class_name: str) -> int:
    """Resolve the volume for a problem class from the Phase 57A inputs."""
    if class_name == "review_ocr_quality":
        return int(report.get("review_ocr_quality") or 0)
    if class_name == "unsupported_extension":
        return int(report.get("unsupported_count") or 0)
    if class_name == "empty_extraction":
        return int(report.get("empty") or 0)
    if class_name == "unknown_other":
        # Phase 57 stores this cluster as 'unknown_other'.
        return len(clusters.get("unknown_other") or [])
    if class_name == "possible_multi_document_pdf":
        return _count_results_with_flag(report, "possible_multi_document_pdf")
    if class_name == "pdf_portfolio_or_embedded_files_detected":
        return _count_results_with_flag(report, "pdf_embedded_files_detected")
    return len(clusters.get(class_name) or [])


def _count_results_with_flag(report: dict[str, Any], flag_name: str) -> int:
    return sum(
        1
        for item in (report.get("results") or [])
        if isinstance(item, dict) and bool(item.get(flag_name))
    )


def safe_id_sample(clusters: dict[str, list[str]], class_name: str, *, limit: int = 5) -> list[str]:
    members = clusters.get(class_name) or []
    return list(sorted(members))[:limit]


def class_percent_of_corpus(count: int, total_supported: int) -> float:
    if total_supported <= 0:
        return 0.0
    return round(count / total_supported, 4)


def derive_priority_score(profile: dict[str, Any], count: int, total_supported: int) -> dict[str, Any]:
    """Combine profile and count into a deterministic priority score.

    Score formula (transparent and reproducible):
      base   = volume_weight * count
      multiplier = opportunity_weight / difficulty_weight
      action = recommended_action_kind
    """
    safety_weight = {"low": 1.0, "medium": 0.85, "high": 0.6}.get(profile["safety_risk"], 0.7)
    opportunity_weight = {
        "low": 1.0,
        "medium": 1.5,
        "high_if_narrowly_scoped": 1.7,
        "high": 2.0,
    }.get(profile["automation_opportunity"], 1.0)
    difficulty_weight = {"low": 1.0, "medium": 1.6, "high": 2.6}.get(
        profile["implementation_difficulty"], 1.6
    )
    action = profile["recommended_action_kind"]
    # Defer actions get a heavy penalty so the queue surfaces actionable
    # classes first.
    action_multiplier = 0.2 if action == "defer" else 1.0
    volume_normalized = count / max(total_supported, 1)
    score = round(
        100.0
        * volume_normalized
        * safety_weight
        * opportunity_weight
        / difficulty_weight
        * action_multiplier,
        4,
    )
    return {
        "priority_score": score,
        "safety_weight": safety_weight,
        "opportunity_weight": opportunity_weight,
        "difficulty_weight": difficulty_weight,
        "action_multiplier": action_multiplier,
    }


def build_class_entry(
    class_name: str,
    *,
    count: int,
    total_supported: int,
    clusters: dict[str, list[str]],
) -> dict[str, Any]:
    profile = CLASS_PROFILES[class_name]
    priority = derive_priority_score(profile, count, total_supported)
    return {
        "class_name": class_name,
        "file_count": count,
        "percent_of_supported_corpus": class_percent_of_corpus(count, total_supported),
        "safety_risk": profile["safety_risk"],
        "automation_opportunity": profile["automation_opportunity"],
        "implementation_difficulty": profile["implementation_difficulty"],
        "recommended_action_kind": profile["recommended_action_kind"],
        "rationale": profile["rationale"],
        "priority_score": priority["priority_score"],
        "scoring_inputs": priority,
        "safe_id_sample": safe_id_sample(clusters, class_name),
    }


def build_prioritized_queue(class_entries: list[dict[str, Any]]) -> dict[str, Any]:
    actionable = [
        entry for entry in class_entries
        if entry["recommended_action_kind"] != "defer" and entry["file_count"] > 0
    ]
    actionable_sorted = sorted(
        actionable,
        key=lambda e: (-e["priority_score"], e["class_name"]),
    )
    queue: dict[str, Any] = {}
    slot_names = ["phase59_candidate", "phase60_candidate", "phase61_candidate"]
    for slot, entry in zip(slot_names, actionable_sorted):
        queue[slot] = {
            "class_name": entry["class_name"],
            "file_count": entry["file_count"],
            "recommended_action_kind": entry["recommended_action_kind"],
            "priority_score": entry["priority_score"],
            "rationale": entry["rationale"],
            "safe_id_sample": entry["safe_id_sample"],
        }
    for unfilled in slot_names[len(actionable_sorted):]:
        queue[unfilled] = None
    return queue


def build_explicit_decision(
    class_entries: list[dict[str, Any]],
    queue: dict[str, Any],
) -> dict[str, Any]:
    fix_first = queue.get("phase59_candidate")
    if not fix_first:
        return {
            "fix_first_class": None,
            "why_fix_first": (
                "No actionable problem class is currently scored above zero. "
                "The corpus is either pristine, fully deferred, or the inputs "
                "need re-investigation."
            ),
            "do_not_fix_yet": [
                entry["class_name"]
                for entry in class_entries
                if entry["recommended_action_kind"] == "defer"
            ],
        }
    do_not_fix = sorted({
        entry["class_name"]
        for entry in class_entries
        if entry["recommended_action_kind"] == "defer"
        and entry["file_count"] > 0
    })
    why = (
        f"{fix_first['class_name']} ranks highest under the deterministic "
        f"score (priority={fix_first['priority_score']}). It is the most "
        f"actionable class given the safety/opportunity/difficulty profile "
        f"encoded in CLASS_PROFILES. Other large classes (empty_extraction, "
        f"rules_based_low_confidence, possible_lab_table_failure) are deferred "
        f"because acting on them corpus-wide would risk weakening existing "
        f"safety gates without subset-level evidence."
    )
    return {
        "fix_first_class": fix_first["class_name"],
        "fix_first_action_kind": fix_first["recommended_action_kind"],
        "fix_first_file_count": fix_first["file_count"],
        "fix_first_priority_score": fix_first["priority_score"],
        "why_fix_first": why,
        "do_not_fix_yet": do_not_fix,
    }


def build_plan(phase57_inputs: dict[str, Any]) -> dict[str, Any]:
    report = phase57_inputs["report"]
    clusters = phase57_inputs["clusters"]
    reconciliation = report.get("filesystem_reconciliation") or {}
    total_supported = int(
        reconciliation.get("total_supported_processed")
        or report.get("total_supported")
        or 0
    )
    corpus_totals = {
        "total_filesystem_files": int(reconciliation.get("total_filesystem_files") or 0),
        "total_filesystem_folders": int(reconciliation.get("total_filesystem_folders") or 0),
        "total_supported_processed": total_supported,
        "total_unsupported_extension": int(reconciliation.get("total_unsupported_extension") or 0),
        "total_ignored_system_files": int(reconciliation.get("total_ignored_system_files") or 0),
        "total_processing_errors": int(reconciliation.get("total_processing_errors") or 0),
        "total_inaccessible_files": int(reconciliation.get("total_inaccessible_files") or 0),
        "accepted": int(report.get("accepted") or 0),
        "review": int(report.get("review") or 0),
        "review_ocr_quality": int(report.get("review_ocr_quality") or 0),
        "empty": int(report.get("empty") or 0),
        "errors": int(report.get("errors") or 0),
        "reconciliation_passed": bool(reconciliation.get("reconciliation_passed", False)),
    }
    class_entries: list[dict[str, Any]] = []
    for name in PROBLEM_CLASS_ORDER:
        count = class_count_for(report, clusters, name)
        class_entries.append(
            build_class_entry(name, count=count, total_supported=total_supported, clusters=clusters)
        )
    queue = build_prioritized_queue(class_entries)
    decision = build_explicit_decision(class_entries, queue)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 58 Stratified Problem-Class Fix Plan",
        "source_phase57_report": str(PHASE57_REPORT.relative_to(ROOT)),
        "source_phase57_clusters": str(PHASE57_CLUSTERS.relative_to(ROOT)),
        "corpus_totals": corpus_totals,
        "problem_classes": class_entries,
        "prioritized_fix_queue": queue,
        "explicit_decision": decision,
        "privacy_safety": {
            "uses_safe_ids_only": True,
            "raw_filenames_present_in_output": False,
            "raw_paths_present_in_output": False,
            "extracted_text_present_in_output": False,
            "phi_present_in_output": False,
        },
    }


def write_plan(plan: dict[str, Any], *, report_dir: Path | None = None) -> dict[str, Path]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / JSON_REPORT.name
    md_path = target_dir / MD_REPORT.name
    json_path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def render_markdown(plan: dict[str, Any]) -> str:
    totals = plan["corpus_totals"]
    decision = plan["explicit_decision"]
    queue = plan["prioritized_fix_queue"]
    lines = [
        "# Phase 58 Stratified Problem-Class Fix Plan",
        "",
        f"- Generated at: `{plan['generated_at']}`",
        f"- Source Phase 57A report: `{plan['source_phase57_report']}`",
        f"- Source Phase 57A clusters: `{plan['source_phase57_clusters']}`",
        "",
        "## Corpus Totals (Phase 57A)",
        "",
        f"- total_filesystem_files: `{totals['total_filesystem_files']}`",
        f"- total_filesystem_folders: `{totals['total_filesystem_folders']}`",
        f"- total_supported_processed: `{totals['total_supported_processed']}`",
        f"- total_unsupported_extension: `{totals['total_unsupported_extension']}`",
        f"- total_ignored_system_files: `{totals['total_ignored_system_files']}`",
        f"- total_processing_errors: `{totals['total_processing_errors']}`",
        f"- total_inaccessible_files: `{totals['total_inaccessible_files']}`",
        f"- accepted: `{totals['accepted']}`",
        f"- review: `{totals['review']}`",
        f"- review_ocr_quality: `{totals['review_ocr_quality']}`",
        f"- empty: `{totals['empty']}`",
        f"- errors: `{totals['errors']}`",
        f"- reconciliation_passed: `{totals['reconciliation_passed']}`",
        "",
        "## Explicit Decision",
        "",
        f"- **fix_first_class:** `{decision.get('fix_first_class')}`",
        f"- fix_first_action_kind: `{decision.get('fix_first_action_kind')}`",
        f"- fix_first_file_count: `{decision.get('fix_first_file_count')}`",
        f"- fix_first_priority_score: `{decision.get('fix_first_priority_score')}`",
        "",
        f"**Why fix this first:** {decision.get('why_fix_first')}",
        "",
        f"**Do not fix yet (deferred classes with non-zero volume):** "
        f"`{decision.get('do_not_fix_yet') or []}`",
        "",
        "## Prioritized Fix Queue",
        "",
    ]
    for slot in ("phase59_candidate", "phase60_candidate", "phase61_candidate"):
        entry = queue.get(slot)
        if entry is None:
            lines.append(f"- **{slot}**: _(no actionable class)_")
            continue
        lines += [
            f"### {slot}: `{entry['class_name']}`",
            "",
            f"- recommended_action_kind: `{entry['recommended_action_kind']}`",
            f"- file_count: `{entry['file_count']}`",
            f"- priority_score: `{entry['priority_score']}`",
            f"- safe_id_sample (up to 5): `{entry['safe_id_sample']}`",
            "",
            f"_{entry['rationale']}_",
            "",
        ]
    lines += [
        "## Per-Class Breakdown",
        "",
        "| Class | Count | % of supported | Safety | Opportunity | Difficulty | Action | Score |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    for entry in plan["problem_classes"]:
        lines.append(
            "| "
            + " | ".join([
                f"`{entry['class_name']}`",
                str(entry["file_count"]),
                f"{entry['percent_of_supported_corpus'] * 100:.2f}%",
                entry["safety_risk"],
                entry["automation_opportunity"],
                entry["implementation_difficulty"],
                entry["recommended_action_kind"],
                str(entry["priority_score"]),
            ])
            + " |"
        )
    lines += [
        "",
        "## Class Rationales",
        "",
    ]
    for entry in plan["problem_classes"]:
        lines += [
            f"### `{entry['class_name']}`",
            "",
            f"- file_count: `{entry['file_count']}`",
            f"- percent_of_supported_corpus: `{entry['percent_of_supported_corpus']}`",
            f"- safety_risk: `{entry['safety_risk']}`",
            f"- automation_opportunity: `{entry['automation_opportunity']}`",
            f"- implementation_difficulty: `{entry['implementation_difficulty']}`",
            f"- recommended_action_kind: `{entry['recommended_action_kind']}`",
            f"- priority_score: `{entry['priority_score']}`",
            f"- safe_id_sample: `{entry['safe_id_sample']}`",
            "",
            entry["rationale"],
            "",
        ]
    lines += [
        "## Privacy Safety",
        "",
        f"- uses_safe_ids_only: `{plan['privacy_safety']['uses_safe_ids_only']}`",
        f"- raw_filenames_present_in_output: `{plan['privacy_safety']['raw_filenames_present_in_output']}`",
        f"- raw_paths_present_in_output: `{plan['privacy_safety']['raw_paths_present_in_output']}`",
        f"- extracted_text_present_in_output: `{plan['privacy_safety']['extracted_text_present_in_output']}`",
        f"- phi_present_in_output: `{plan['privacy_safety']['phi_present_in_output']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    inputs = load_phase57_inputs()
    plan = build_plan(inputs)
    paths = write_plan(plan)
    decision = plan["explicit_decision"]
    print("MedAI Phase 58 stratified problem-class fix plan complete.")
    print(f"fix_first_class: {decision.get('fix_first_class')}")
    print(f"fix_first_action_kind: {decision.get('fix_first_action_kind')}")
    print(f"fix_first_file_count: {decision.get('fix_first_file_count')}")
    print(f"fix_first_priority_score: {decision.get('fix_first_priority_score')}")
    print(f"do_not_fix_yet: {decision.get('do_not_fix_yet')}")
    print(f"json_report: {paths['json']}")
    print(f"markdown_report: {paths['md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
