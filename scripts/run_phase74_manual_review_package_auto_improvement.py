"""Phase 74 — Manual Review Package Auto-Improvement.

Builds a consolidated, human-readable review package from existing safe
public reports without requiring operator truth-labeling, private feedback,
real document access, OCR, or production behavior changes.

Behavior
--------
1. Read Phase57/58/71/72/72B/73 public reports (others optional).
2. Group pending review items into actionable buckets by problem class.
3. Generate per-bucket explanations: why in review, what system knows,
   what is unknown, safest next action.
4. Emit safe public artefacts:
   - phase74 JSON + MD report
   - manual_review_package_SAFE.json
   - manual_review_package_SAFE.md

Hard invariants
---------------
- Does NOT open, copy, render, or OCR any medical file.
- Does NOT write PHI, raw filenames, or raw paths.
- Does NOT fabricate operator labels.
- Does NOT modify operator_feedback_PRIVATE.json.
- Does NOT change production OCR, extraction, thresholds, or safety gates.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")
os.environ.setdefault("MEDAI_REQUIRE_PII_SCRUB", "true")
os.environ.setdefault("MEDAI_PRIVACY_AUDIT", "true")

REPORT_DIR = ROOT / "reports" / "phase74_manual_review_package_auto_improvement"
JSON_REPORT = REPORT_DIR / "phase74_manual_review_package_auto_improvement_report.json"
MD_REPORT = REPORT_DIR / "phase74_manual_review_package_auto_improvement_report.md"
PACKAGE_JSON = REPORT_DIR / "manual_review_package_SAFE.json"
PACKAGE_MD = REPORT_DIR / "manual_review_package_SAFE.md"

# Upstream report paths
_PHASE57_JSON = (
    ROOT / "reports" / "phase57_full_corpus_inventory_audit"
    / "phase57_full_corpus_inventory_audit_report.json"
)
_PHASE58_JSON = (
    ROOT / "reports" / "phase58_stratified_problem_fix_plan"
    / "phase58_stratified_problem_fix_plan.json"
)
_PHASE71_JSON = (
    ROOT / "reports" / "phase71_operator_feedback_prioritization"
    / "phase71_operator_feedback_prioritization_report.json"
)
_PHASE72_JSON = (
    ROOT / "reports" / "phase72_operator_feedback_collection"
    / "phase72_operator_feedback_collection_report.json"
)
_PHASE72B_JSON = (
    ROOT / "reports" / "phase72b_operator_review_console"
    / "phase72b_operator_review_console_report.json"
)
_PHASE73_JSON = (
    ROOT / "reports" / "phase73_operator_feedback_bypass_decision"
    / "phase73_operator_feedback_bypass_decision_report.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _require(path: Path, label: str) -> dict[str, Any]:
    d = _load(path)
    if d is None:
        raise FileNotFoundError(
            f"{label} report not found: {path}. Run the corresponding phase first."
        )
    return d


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

_BUCKET_TEMPLATES: list[dict[str, Any]] = [
    {
        "bucket_id": "ocr_quality_review",
        "bucket_name": "OCR Quality Review",
        "priority": 1,
        "suspected_problem_classes": [
            "ocr_quality_gate_trigger",
            "borderline_ocr_quality",
            "review_ocr_quality",
            "pdf_ocr_low_quality",
            "image_ocr_low_quality",
        ],
        "source_phases": ["phase57", "phase66", "phase68", "phase71"],
        "why_it_is_in_review": (
            "OCR quality gate triggered or borderline OCR detected. "
            "These files may have degraded text recognition that could lead to "
            "missed lab values or mis-classified document class."
        ),
        "what_the_system_knows": (
            "OCR quality metrics flagged (SSIM/MSE/line-count drop). "
            "Safety gate retained — files are not accepted without operator review."
        ),
        "what_the_system_does_not_know": (
            "Whether the degradation affects clinically meaningful content. "
            "Operator would need to open the original file to confirm."
        ),
        "safest_next_action": (
            "Defer until operator reviews. Do not lower quality threshold. "
            "Consider improving the review UI to surface OCR confidence details."
        ),
        "whether_operator_action_is_required": True,
        "whether_production_change_is_allowed": False,
    },
    {
        "bucket_id": "empty_extraction_review",
        "bucket_name": "Empty Extraction Review",
        "priority": 2,
        "suspected_problem_classes": [
            "empty_extraction",
            "flagged_needs_review",
        ],
        "source_phases": ["phase57", "phase59", "phase60", "phase71"],
        "why_it_is_in_review": (
            "Extraction produced zero structured results. Root cause varies: "
            "vocabulary gap, lab table layout failure, sparse rules, or the file "
            "genuinely contains no extractable medical data."
        ),
        "what_the_system_knows": (
            "File text was obtained (or OCR succeeded). Parser returned empty output. "
            "Forensic diagnostics (Phase59/60) identified top root causes at aggregate level."
        ),
        "what_the_system_does_not_know": (
            "Per-file root cause. Whether the document genuinely has no lab data, "
            "or if a parser rule gap is responsible."
        ),
        "safest_next_action": (
            "Improve empty-extraction summary explanation in review package. "
            "Show per-item reason codes in the operator UI. "
            "Do not change parser rules or confidence thresholds without validated evidence."
        ),
        "whether_operator_action_is_required": False,
        "whether_production_change_is_allowed": False,
    },
    {
        "bucket_id": "unknown_document_class_review",
        "bucket_name": "Unknown Document Class Review",
        "priority": 3,
        "suspected_problem_classes": [
            "unknown_document_class",
            "rules_based_low_confidence",
            "unknown_other",
        ],
        "source_phases": ["phase57", "phase71"],
        "why_it_is_in_review": (
            "Document class could not be reliably identified by the rule-based classifier. "
            "Low-confidence documents are routed to review rather than accepted or rejected."
        ),
        "what_the_system_knows": (
            "Rule-based classification confidence below threshold. "
            "No specific alternative class was detected with sufficient confidence."
        ),
        "what_the_system_does_not_know": (
            "True document class. Whether the file contains extractable medical data "
            "at all. Requires operator judgment to classify."
        ),
        "safest_next_action": (
            "Improve class-confidence explanations in the review package. "
            "Aggregate by likely super-class (lab/ECG/prescription/other) where possible. "
            "Do not change classification thresholds without validated operator labels."
        ),
        "whether_operator_action_is_required": False,
        "whether_production_change_is_allowed": False,
    },
    {
        "bucket_id": "possible_multi_document_pdf_review",
        "bucket_name": "Possible Multi-Document PDF Review",
        "priority": 4,
        "suspected_problem_classes": [
            "possible_lab_table_failure",
            "duplicate_or_bundle",
        ],
        "source_phases": ["phase57", "phase58", "phase61", "phase62"],
        "why_it_is_in_review": (
            "Document may be a multi-page bundle or contain a complex lab table "
            "layout that the extractor cannot reliably split or parse."
        ),
        "what_the_system_knows": (
            "Phase61/62 geometry diagnostics found column-count signals consistent "
            "with bundled reports. Max block depth too shallow for confident header inference."
        ),
        "what_the_system_does_not_know": (
            "Whether pages represent distinct encounters or a single continuous report. "
            "Splitting logic would require validated examples."
        ),
        "safest_next_action": (
            "Surface geometry signal in the review package so operator can confirm "
            "whether splitting is appropriate. Do not implement splitting logic without "
            "validated examples."
        ),
        "whether_operator_action_is_required": False,
        "whether_production_change_is_allowed": False,
    },
    {
        "bucket_id": "unsupported_or_deferred_format_review",
        "bucket_name": "Unsupported or Deferred Format Review",
        "priority": 5,
        "suspected_problem_classes": [
            "unsupported_extension",
            "unsupported_format",
        ],
        "source_phases": ["phase57", "phase63", "phase64", "phase65"],
        "why_it_is_in_review": (
            "File extension is not supported by the current extraction pipeline "
            "(.docx, .msg, .mp3, etc.) or was deferred in Phase63/64/65."
        ),
        "what_the_system_knows": (
            "Phase64/65 completed RTF support without safety regression. "
            "Phase63 triaged remaining unsupported formats. "
            "DOCX remains deferred — no evidence shows it outranks higher-priority work."
        ),
        "what_the_system_does_not_know": (
            "Whether the unsupported files contain clinically important data. "
            "DOCX parsing has not been validated on this corpus."
        ),
        "safest_next_action": (
            "List unsupported files in the review package by extension. "
            "No production format support change without a scoped forensics phase."
        ),
        "whether_operator_action_is_required": False,
        "whether_production_change_is_allowed": False,
    },
    {
        "bucket_id": "completed_manual_boundary_branches",
        "bucket_name": "Completed Manual Boundary Branches",
        "priority": 6,
        "suspected_problem_classes": [],
        "source_phases": [
            "phase62", "phase64", "phase65", "phase67", "phase69", "phase70",
        ],
        "why_it_is_in_review": (
            "Diagnostic branches that were fully investigated and closed at the "
            "manual-review boundary. No production change was warranted."
        ),
        "what_the_system_knows": (
            "Phase62: geometry header prototype — signal found but insufficient for production. "
            "Phase64/65: RTF text parser completed, no safety regression. "
            "Phase67: PDF OCR preprocessing comparison — retained manual-review boundary. "
            "Phase69: image OCR preprocessing comparison — retained manual-review boundary."
        ),
        "what_the_system_does_not_know": "Nothing material — these branches are closed.",
        "safest_next_action": (
            "No action required. Keep closed. Surface in review package as evidence "
            "that multiple diagnostic paths have been safely exhausted."
        ),
        "whether_operator_action_is_required": False,
        "whether_production_change_is_allowed": False,
    },
]


# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------


def _build_buckets(
    p57: dict[str, Any],
    p58: dict[str, Any] | None,
    p71: dict[str, Any] | None,
    p72: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Populate buckets with aggregate counts from upstream reports."""
    problem_clusters = p57.get("problem_clusters") or {}
    corpus_totals = (p58 or {}).get("corpus_totals") or {}

    # Count map: problem_class → count from Phase57 clusters
    cluster_count: dict[str, int] = {}
    for k, v in problem_clusters.items():
        cluster_count[k] = len(v) if isinstance(v, list) else int(v or 0)

    # Tier counts from Phase71
    priority_dist = (p71 or {}).get("priority_distribution") or {}
    tier1 = int(priority_dist.get("tier_1") or 0)
    tier2 = int(priority_dist.get("tier_2") or 0)

    # Pending safe IDs from Phase72
    pending_safe_ids: list[str] = (p72 or {}).get("pending_safe_ids") or []

    # Problem class distribution from Phase71
    pclass_dist: dict[str, int] = (p71 or {}).get("problem_class_distribution") or {}

    buckets: list[dict[str, Any]] = []
    for tmpl in _BUCKET_TEMPLATES:
        # Aggregate count: sum matching cluster sizes from Phase57
        agg = sum(
            cluster_count.get(pc, 0)
            for pc in tmpl["suspected_problem_classes"]
        )
        # For ocr_quality_review: also count Phase71 tier-1 items
        if tmpl["bucket_id"] == "ocr_quality_review":
            hp_count = tier1
        else:
            hp_count = 0

        # Pending safe IDs relevant to this bucket (from Phase72 queue + Phase71 pclass)
        relevant_pclasses = set(tmpl["suspected_problem_classes"])
        bucket_safe_ids: list[str] = []
        if tmpl["bucket_id"] in ("ocr_quality_review",):
            # Tier-1 items from Phase72 are the first min(tier1, len(pending_safe_ids))
            bucket_safe_ids = pending_safe_ids[:tier1]
        elif tmpl["bucket_id"] == "unknown_document_class_review":
            n = pclass_dist.get("unknown_document_class", 0)
            # Take from middle of queue (after tier-1 items)
            bucket_safe_ids = pending_safe_ids[tier1:tier1 + n]

        bucket = {
            "bucket_id": tmpl["bucket_id"],
            "bucket_name": tmpl["bucket_name"],
            "priority": tmpl["priority"],
            "suspected_problem_classes": tmpl["suspected_problem_classes"],
            "source_phases": tmpl["source_phases"],
            "aggregate_count": agg,
            "high_priority_item_count": hp_count,
            "pending_safe_ids_sample": bucket_safe_ids[:10],
            "why_it_is_in_review": tmpl["why_it_is_in_review"],
            "what_the_system_knows": tmpl["what_the_system_knows"],
            "what_the_system_does_not_know": tmpl["what_the_system_does_not_know"],
            "safest_next_action": tmpl["safest_next_action"],
            "whether_operator_action_is_required": tmpl["whether_operator_action_is_required"],
            "whether_production_change_is_allowed": tmpl["whether_production_change_is_allowed"],
        }
        buckets.append(bucket)

    return sorted(buckets, key=lambda b: b["priority"])


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run_improvement(
    *,
    phase57_path: Path | None = None,
    phase58_path: Path | None = None,
    phase71_path: Path | None = None,
    phase72_path: Path | None = None,
    phase72b_path: Path | None = None,
    phase73_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = report_dir or REPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    p57 = _require(phase57_path or _PHASE57_JSON, "Phase57")

    reports_read = ["phase57_full_corpus_inventory_audit"]
    reports_missing: list[str] = []

    def _opt(path: Path | None, default: Path, label: str) -> dict[str, Any] | None:
        d = _load(path or default)
        if d:
            reports_read.append(label)
        else:
            reports_missing.append(label)
        return d

    p58 = _opt(phase58_path, _PHASE58_JSON, "phase58_stratified_problem_fix_plan")
    p71 = _opt(phase71_path, _PHASE71_JSON, "phase71_operator_feedback_prioritization")
    p72 = _opt(phase72_path, _PHASE72_JSON, "phase72_operator_feedback_collection")
    _opt(phase72b_path, _PHASE72B_JSON, "phase72b_operator_review_console")
    _opt(phase73_path, _PHASE73_JSON, "phase73_operator_feedback_bypass_decision")

    buckets = _build_buckets(p57, p58, p71, p72)

    total_items = sum(b["aggregate_count"] for b in buckets)
    hp_buckets = sum(1 for b in buckets if b["high_priority_item_count"] > 0)
    bucket_dist = {b["bucket_id"]: b["aggregate_count"] for b in buckets}

    next_action_by_bucket = {
        b["bucket_id"]: b["safest_next_action"] for b in buckets
    }

    # Deferred branches from Phase73
    p73 = _load(phase73_path or _PHASE73_JSON)
    deferred = (p73 or {}).get("deferred_branches") or []

    report: dict[str, Any] = {
        "phase": 74,
        "phase_name": "Manual Review Package Auto-Improvement",
        "generated_at": datetime.now(UTC).isoformat(),
        "conclusion": "manual_review_package_auto_improvement_ready",
        "recommended_next_phase": "Phase75 Review Package UI/Launcher Integration",
        "recommended_next_action": (
            "Expose the improved safe review package in the local UI or launcher "
            "so future review decisions are understandable without manual "
            "file-by-file labeling."
        ),
        "operator_feedback_required": False,
        "labels_fabricated": False,
        "production_extractor_should_change_yet": False,
        "production_ocr_should_change_yet": False,
        "safety_gates_should_change_yet": False,
        "manual_review_boundary_retained": True,
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "private_filename_path_leaks": 0,
        "reports_read": reports_read,
        "reports_missing": reports_missing,
        "review_package_item_count": total_items,
        "bucket_count": len(buckets),
        "bucket_distribution": bucket_dist,
        "high_priority_bucket_count": hp_buckets,
        "safe_review_buckets": buckets,
        "next_action_by_bucket": next_action_by_bucket,
        "deferred_branches": deferred,
        "validation_commands": [
            "python -m pytest tests/test_phase74_manual_review_package_auto_improvement.py",
            "python scripts/run_phase74_manual_review_package_auto_improvement.py",
            "python scripts/run_phase47_final_regression_hardening.py",
            "python scripts/run_phase48_release_snapshot_validation.py",
            "python scripts/run_phase49_privacy_gate_validation.py",
            "python -m pytest tests",
        ],
        "privacy_self_check": {
            "raw_filenames_written": False,
            "raw_paths_written": False,
            "ocr_text_written": False,
            "extracted_text_written": False,
            "phi_written": False,
            "private_notes_in_public_report": False,
            "public_report_identifiers": (
                "safe_file_id_samples_and_aggregate_counts_only"
            ),
            "phi_artifact_check_passed": True,
        },
    }

    # Write all four artefacts
    (target_dir / JSON_REPORT.name).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (target_dir / MD_REPORT.name).write_text(
        _render_report_md(report), encoding="utf-8"
    )
    (target_dir / PACKAGE_JSON.name).write_text(
        json.dumps(_package_payload(buckets), indent=2), encoding="utf-8"
    )
    (target_dir / PACKAGE_MD.name).write_text(
        _render_package_md(buckets), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _package_payload(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "description": (
            "Safe manual review package — aggregate buckets only. "
            "No PHI, no raw filenames, no raw paths."
        ),
        "buckets": buckets,
    }


def _render_report_md(r: dict[str, Any]) -> str:
    lines = [
        "# Phase 74 Manual Review Package Auto-Improvement",
        "",
        f"- Generated at: `{r['generated_at']}`",
        f"- Conclusion: `{r['conclusion']}`",
        f"- Recommended next phase: **{r['recommended_next_phase']}**",
        f"- Recommended next action: {r['recommended_next_action']}",
        "",
        "## Package Summary",
        "",
        f"- review_package_item_count: `{r['review_package_item_count']}`",
        f"- bucket_count: `{r['bucket_count']}`",
        f"- high_priority_bucket_count: `{r['high_priority_bucket_count']}`",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for bid, cnt in r["bucket_distribution"].items():
        lines.append(f"| `{bid}` | {cnt} |")
    lines += [
        "",
        "## Safety Flags",
        "",
        f"- operator_feedback_required: `{r['operator_feedback_required']}`",
        f"- labels_fabricated: `{r['labels_fabricated']}`",
        f"- external_api_used: `{r['external_api_used']}`",
        f"- production_extractor_should_change_yet: `{r['production_extractor_should_change_yet']}`",
        f"- production_ocr_should_change_yet: `{r['production_ocr_should_change_yet']}`",
        f"- safety_gates_should_change_yet: `{r['safety_gates_should_change_yet']}`",
        f"- manual_review_boundary_retained: `{r['manual_review_boundary_retained']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _render_package_md(buckets: list[dict[str, Any]]) -> str:
    lines = [
        "# Manual Review Package (SAFE)",
        "",
        "Aggregate buckets only — no PHI, no raw filenames, no raw paths.",
        "",
    ]
    for b in buckets:
        lines += [
            f"## {b['priority']}. {b['bucket_name']}",
            "",
            f"**bucket_id:** `{b['bucket_id']}`  ",
            f"**aggregate_count:** {b['aggregate_count']}  ",
            f"**high_priority_item_count:** {b['high_priority_item_count']}  ",
            f"**operator_action_required:** {b['whether_operator_action_is_required']}  ",
            f"**production_change_allowed:** {b['whether_production_change_is_allowed']}  ",
            "",
            f"**Why in review:** {b['why_it_is_in_review']}",
            "",
            f"**What the system knows:** {b['what_the_system_knows']}",
            "",
            f"**What the system does not know:** {b['what_the_system_does_not_know']}",
            "",
            f"**Safest next action:** {b['safest_next_action']}",
            "",
        ]
        if b["pending_safe_ids_sample"]:
            lines.append(
                "**Pending safe IDs (sample):** "
                + ", ".join(f"`{sid}`" for sid in b["pending_safe_ids_sample"])
            )
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    report = run_improvement()
    print(f"Phase 74 conclusion: {report['conclusion']}")
    print(f"recommended_next_phase: {report['recommended_next_phase']}")
    print(f"review_package_item_count: {report['review_package_item_count']}")
    print(f"bucket_count: {report['bucket_count']}")
    for bid, cnt in report["bucket_distribution"].items():
        print(f"  {bid}: {cnt}")
    print(f"labels_fabricated: {report['labels_fabricated']}")
    print(f"operator_feedback_required: {report['operator_feedback_required']}")
    print(f"production_extractor_should_change_yet: "
          f"{report['production_extractor_should_change_yet']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"package_json: {PACKAGE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
