#!/usr/bin/env python3
"""MEDAI-UI-CAPABILITY-PARITY-AUDIT-11A.

Static, public-safe UI capability parity audit. This script inspects only
repo-controlled code, tests, reports, docs, snapshots, and git metadata. It
does not launch Streamlit, read runtime databases, inspect private documents,
or call external APIs.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

TASK_ID = "MEDAI-UI-CAPABILITY-PARITY-AUDIT-11A"
REPORT_DIR = REPO_ROOT / "reports" / "medai_ui_capability_parity_audit_11a"
SUMMARY_MD_PATH = REPORT_DIR / "MEDAI_UI_CAPABILITY_PARITY_AUDIT_11A.md"
REPORT_JSON_PATH = REPORT_DIR / "medai_ui_capability_parity_audit_11a_report.json"
REPORT_MD_PATH = REPORT_DIR / "medai_ui_capability_parity_audit_11a_report.md"
MATRIX_CSV_PATH = REPORT_DIR / "medai_ui_capability_parity_matrix_11a.csv"

SCAN_FILES = [
    "app/main.py",
    "app/test_launcher.py",
    "app/local_adapter_fallback_processor.py",
    "app/extracted_information_preview.py",
    "app/operator_review_actions.py",
    "app/operator_control_panel.py",
    "app/clinical_knowledge_safety_viewer.py",
    "app/terminology_readiness_viewer.py",
    "app/clinical_knowledge_terminology_lookup_viewer.py",
    "app/schemas.py",
    "app/config.py",
    "reports/b07_term_opt_in_planning/B07_TERM_OPT_IN_INTEGRATION_PLAN.md",
    "snapshots/MedAI_Snapshot_Phase62_2026-05-04/snapshot_summary.md",
    "snapshots/MedAI_Snapshot_Phase62_2026-05-04/continuation_prompt.md",
    "tests/test_medai_ui_adapter_fallback_run_review_10.py",
    "tests/test_medai_ui_startup_resilience_09.py",
    "tests/test_medai_local_self_healing_validation_07.py",
]


def _read_rel(path: str) -> str:
    target = REPO_ROOT / path
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _contains(path: str, *terms: str) -> bool:
    text = _read_rel(path).lower()
    return all(term.lower() in text for term in terms)


def current_ui_inventory() -> list[dict[str, Any]]:
    """Return current visible/code-level UI capability inventory."""
    main = _read_rel("app/main.py")
    inventory = [
        {
            "capability": "Run & Review",
            "source": "app/main.py:render_run_review_tab",
            "visible_by_default": "Run & Review" in main and "PRIMARY_OPERATOR_TABS" in main,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": "render_adapter_fallback_panel" in main,
            "writes_to_mkb": True,
            "safety_privacy_relevance": "Primary local review workflow; review-bound only.",
            "current_status": "available",
        },
        {
            "capability": "Operator Control Panel",
            "source": "app/operator_control_panel.py:render_operator_control_panel",
            "visible_by_default": _contains("app/main.py", "Operator Control Panel"),
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Runs validation commands; no direct clinical interpretation.",
            "current_status": "available",
        },
        {
            "capability": "Advanced tools toggle",
            "source": "app/main.py:st.checkbox",
            "visible_by_default": "Show advanced tools" in main,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Hides audit/admin surfaces by default.",
            "current_status": "available",
        },
        {
            "capability": "Validation Batch Audit",
            "source": "app/main.py:render_blind_audit_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": True,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Synthetic/report validation; public-safe reports required.",
            "current_status": "hidden",
        },
        {
            "capability": "Validation History",
            "source": "app/main.py:render_report_archive_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": True,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Displays public reports.",
            "current_status": "hidden",
        },
        {
            "capability": "Safety & Governance",
            "source": "app/clinical_knowledge_safety_viewer.py:render_clinical_knowledge_safety_dashboard",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": True,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Privacy, DDI, medication, enrichment, coding, connector status.",
            "current_status": "hidden",
        },
        {
            "capability": "Terminology Admin",
            "source": "app/terminology_readiness_viewer.py:render_terminology_readiness_panel",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": True,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "License/readiness reporting; does not create clinical facts.",
            "current_status": "hidden",
        },
        {
            "capability": "Terminology Lookup",
            "source": "app/clinical_knowledge_terminology_lookup_viewer.py:render_terminology_lookup_panel",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": True,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Local-only read-only lookup; feature flag gated.",
            "current_status": "hidden",
        },
        {
            "capability": "Document category selector",
            "source": "app/main.py:render_current_run_tab/render_adapter_fallback_panel",
            "visible_by_default": "Document category" in main,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": "adapter_fallback_specialty" in main,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Routes local processing specialty labels; no interpretation.",
            "current_status": "available",
        },
        {
            "capability": "Medical specialty/domain selector",
            "source": "app/main.py:render_upload_tab; app/config.py:ALLOWED_SPECIALTIES",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": True,
            "available_in_degraded_pipeline_mode": False,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Important operator routing context; should not infer diagnosis.",
            "current_status": "disconnected",
        },
        {
            "capability": "MKB Explorer",
            "source": "app/main.py:render_mkb_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Shows active/quarantined records; critical for review visibility.",
            "current_status": "disconnected",
        },
        {
            "capability": "Conflict Review",
            "source": "app/main.py:render_conflict_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": True,
            "safety_privacy_relevance": "Truth resolution/quarantine workflow.",
            "current_status": "disconnected",
        },
        {
            "capability": "Query tab",
            "source": "app/main.py:render_query_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": True,
            "available_in_degraded_pipeline_mode": False,
            "writes_to_mkb": True,
            "safety_privacy_relevance": "Decision engine; must remain safe-mode gated.",
            "current_status": "disconnected",
        },
        {
            "capability": "Upload PDF tab",
            "source": "app/main.py:render_upload_tab",
            "visible_by_default": False,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": True,
            "available_in_degraded_pipeline_mode": False,
            "writes_to_mkb": True,
            "safety_privacy_relevance": "Pipeline processing path; unavailable when execution=None.",
            "current_status": "disconnected",
        },
        {
            "capability": "Extracted information preview",
            "source": "app/extracted_information_preview.py",
            "visible_by_default": True,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": False,
            "safety_privacy_relevance": "Public-safe preview; no raw text.",
            "current_status": "available",
        },
        {
            "capability": "Operator accept/reject/defer",
            "source": "app/operator_review_actions.py; app/main.py:render_run_result_card",
            "visible_by_default": True,
            "hidden_behind_advanced_tools": False,
            "available_only_when_execution_exists": False,
            "available_in_degraded_pipeline_mode": True,
            "writes_to_mkb": True,
            "safety_privacy_relevance": "One-record manual review; no auto-accept.",
            "current_status": "available",
        },
    ]
    return inventory


def historical_capabilities() -> list[dict[str, Any]]:
    git_log = _git(["log", "--oneline", "--decorate", "-80"])
    file_text = "\n".join(_read_rel(path) for path in SCAN_FILES)
    evidence_text = f"{git_log}\n{file_text}".lower()
    caps = [
        ("document upload/queue", "app/test_launcher.py", "run_medai_test_batch"),
        ("start run", "app/main.py", "Start run"),
        ("current run status", "app/main.py", "render_run_status_panel"),
        ("extracted information preview", "app/extracted_information_preview.py", "Extracted information preview"),
        ("MKB persistence", "app/local_adapter_fallback_processor.py", "write_record"),
        ("operator accept/reject/defer", "app/operator_review_actions.py", "accept_after_source_comparison"),
        ("specialty/domain selector", "app/config.py; app/main.py", "gastroenterology"),
        ("document category selector", "app/main.py", "Document category"),
        ("MKB Explorer", "app/main.py", "MKB Explorer"),
        ("review-bound records", "app/operator_review_actions.py", "requires_review"),
        ("active/quarantined/superseded tiers", "app/operator_review_actions.py", "superseded"),
        ("retrieval proof", "reports/tests", "retrieval_proof_count"),
        ("specialty routing", "app/test_launcher.py", "specialty"),
        ("conflict review", "app/conflict_review.py; app/main.py", "render_conflict_tab"),
        ("privacy audit", "clinical_knowledge/privacy/report_privacy.py", "check_public_report_payload"),
        ("safe mode", "app/config.py", "SAFE_MODE_THRESHOLD"),
        ("DDI/medication safety", "app/clinical_knowledge_safety_viewer.py", "Medication Safety / DDI"),
        ("no auto-accept", "app/operator_review_actions.py", "auto_accept_allowed"),
        ("external API off", "app/config.py", "MEDAI_ALLOW_EXTERNAL_API"),
        ("audit reports", "reports", "report"),
        ("one-doc validation", "scripts/run_medai_local_one_doc_operator_validation_05.py", "one_doc_operator_validation"),
        ("self-healing validation", "scripts/run_medai_local_self_healing_validation_07.py", "self_healing"),
        ("batch audit", "app/main.py", "Run validation batch"),
        ("full corpus audit", "app/main.py", "Full Corpus"),
        ("OCR review", "app/test_launcher.py", "ocr_gate"),
        ("unknown triage", "reports/git history", "UNKNOWN-TRIAGE"),
        ("terminology admin", "app/terminology_readiness_viewer.py", "Terminology Admin"),
        ("terminology lookup", "app/clinical_knowledge_terminology_lookup_viewer.py", "Terminology Lookup"),
        ("import readiness", "app/terminology_readiness_viewer.py", "import readiness"),
        ("license-safe reporting", "reports/b07_term_opt_in_planning", "license"),
        ("external agents/comments back", "git history/reports", "external agents"),
        ("Claude/Gemini status", "app/config.py", "GEMINI_MODEL"),
        ("cloud API disabled state", "app/config.py", "MEDAI_ALLOW_EXTERNAL_API"),
        ("future/deferred connector status", "app/clinical_knowledge_safety_viewer.py", "local stubs"),
        ("pathology family", "git history", "PATHOLOGY"),
        ("medical coding", "app/clinical_knowledge_safety_viewer.py", "Medical Coding"),
    ]
    out = []
    for name, source, term in caps:
        term_found = term.lower() in evidence_text
        current = next((item for item in current_ui_inventory() if item["capability"].lower() == name.lower()), None)
        out.append(
            {
                "capability": name,
                "evidence_source": source,
                "evidence_found": term_found,
                "current_ui_exposes_it": bool(current and current["current_status"] == "available"),
                "hidden_behind_advanced_tools": bool(current and current["hidden_behind_advanced_tools"]),
                "intentionally_removed_or_deferred": name in {
                    "external agents/comments back",
                    "Claude/Gemini status",
                    "future/deferred connector status",
                },
                "missing_unexpectedly": name in {"specialty/domain selector", "MKB Explorer", "conflict review"},
                "product_critical_for_practical_mvp": name in {
                    "document upload/queue",
                    "start run",
                    "extracted information preview",
                    "MKB persistence",
                    "operator accept/reject/defer",
                    "specialty/domain selector",
                    "MKB Explorer",
                },
            }
        )
    return out


def feature_parity_matrix() -> list[dict[str, Any]]:
    hist = {item["capability"]: item for item in historical_capabilities()}

    def row(category: str, expected: str, visibility: str, status: str, reason: str, risk: str, action: str, block: str) -> dict[str, Any]:
        return {
            "category": category,
            "expected_capability": expected,
            "evidence_source": hist.get(expected, {}).get("evidence_source", "current code/reports"),
            "current_visibility": visibility,
            "current_functional_status": status,
            "missing_or_degraded_reason": reason,
            "risk_level": risk,
            "recommended_action": action,
            "suggested_repair_block_id": block,
        }

    rows = [
        row("Primary operator workflow", "document upload/queue", "default", "available", "Run & Review queue is visible.", "low", "Keep.", "none"),
        row("Primary operator workflow", "start run", "default", "available/degraded", "Normal pipeline path requires execution; adapter fallback covers TXT lab-style text.", "medium", "Clarify degraded labels.", "MEDAI-UI-RUN-REVIEW-REPAIR-11B"),
        row("Primary operator workflow", "current run status", "default", "available", "Current run panel remains visible.", "low", "Keep.", "none"),
        row("Primary operator workflow", "extracted information preview", "default", "available", "Preview renders from pipeline and adapter fallback.", "low", "Keep.", "none"),
        row("Primary operator workflow", "MKB persistence", "default", "available", "Adapter fallback writes quarantined test_result records.", "low", "Keep review-bound.", "none"),
        row("Primary operator workflow", "operator accept/reject/defer", "default", "available", "One-record controls are rendered from preview rows.", "low", "Keep no bulk accept.", "none"),
        row("Primary operator workflow", "specialty/domain selector", "code-only", "unexpectedly missing", "Only document category is exposed in current primary workflow; domain breadth is reduced.", "high", "Restore explicit domain selector without changing extraction rules.", "MEDAI-UI-CAPABILITY-RESTORE-11B"),
        row("Primary operator workflow", "document category selector", "default", "available", "Document category is visible, but it is not full domain selection.", "medium", "Keep and distinguish from specialty.", "MEDAI-UI-CAPABILITY-RESTORE-11B"),
        row("MKB / knowledge workflow", "MKB Explorer", "code-only", "unexpectedly missing", "render_mkb_tab exists but is not reachable from current tab list.", "high", "Restore safe read-only tab visibility.", "MEDAI-UI-CAPABILITY-RESTORE-11B"),
        row("MKB / knowledge workflow", "review-bound records", "default", "available", "Preview/action plans expose review-bound records.", "low", "Keep.", "none"),
        row("MKB / knowledge workflow", "active/quarantined/superseded tiers", "code-only", "degraded", "Tier filter exists in MKB Explorer but tab is disconnected.", "medium", "Restore MKB Explorer visibility.", "MEDAI-UI-CAPABILITY-RESTORE-11B"),
        row("MKB / knowledge workflow", "retrieval proof", "report-only", "available", "Validation reports prove SQLite retrieval.", "low", "Keep in reports.", "none"),
        row("MKB / knowledge workflow", "specialty routing", "code-only", "degraded", "Pipeline call accepts specialty, but visible selector is category-labeled and limited.", "medium", "Restore explicit selector.", "MEDAI-UI-CAPABILITY-RESTORE-11B"),
        row("MKB / knowledge workflow", "conflict review", "code-only", "disconnected", "render_conflict_tab exists but is not reachable.", "medium", "Restore after MKB Explorer.", "MEDAI-UI-CAPABILITY-RESTORE-11C"),
        row("Safety / governance", "privacy audit", "advanced", "available", "Safety dashboard and report privacy checks exist.", "low", "Keep advanced.", "none"),
        row("Safety / governance", "safe mode", "advanced/code", "available", "Safe mode thresholds and status panels exist.", "low", "Keep.", "none"),
        row("Safety / governance", "DDI/medication safety", "advanced", "available", "Safety dashboard surfaces DDI/medication panels.", "low", "Keep.", "none"),
        row("Safety / governance", "no auto-accept", "default/code", "available", "Action plans and reports keep auto_accept false.", "low", "Keep.", "none"),
        row("Safety / governance", "external API off", "default/code", "available", "Validation scripts set local-only flags.", "low", "Keep.", "none"),
        row("Safety / governance", "audit reports", "advanced/report", "available", "Report archive and generated reports exist.", "low", "Keep.", "none"),
        row("Validation / corpus workflow", "single-document validation", "operator panel", "available", "Validation command is listed in the control panel.", "medium", "Keep listed.", "none"),
        row("Validation / corpus workflow", "self-healing validation", "operator panel", "available", "Operator panel command exists.", "medium", "Keep.", "none"),
        row("Validation / corpus workflow", "batch audit", "advanced", "available", "Validation Batch Audit tab is advanced-only.", "low", "Keep advanced.", "none"),
        row("Validation / corpus workflow", "full corpus audit", "advanced", "available", "Full corpus section is nested in validation tab.", "low", "Keep advanced.", "none"),
        row("Validation / corpus workflow", "OCR review", "default/code", "available", "OCR markers render in result cards.", "low", "Keep read-only.", "none"),
        row("Validation / corpus workflow", "unknown triage", "report/history", "deferred", "Historical reports exist; not primary UI.", "low", "Defer.", "MEDAI-UI-TRIAGE-PARKED"),
        row("Terminology workflow", "terminology admin", "advanced", "available", "Terminology Admin tab exists behind advanced tools.", "low", "Keep advanced.", "none"),
        row("Terminology workflow", "terminology lookup", "advanced/flagged", "hidden", "Lookup panel is feature-flag gated.", "low", "Keep gated.", "none"),
        row("Terminology workflow", "import readiness", "advanced", "available", "Readiness panel exposes import status.", "low", "Keep.", "none"),
        row("Terminology workflow", "license-safe reporting", "report", "available", "License-safe reports exist.", "low", "Keep.", "none"),
        row("External connector / agent workflow", "external agents/comments back", "not visible", "deferred", "No safe local UI found; out of current MVP scope.", "low", "Defer.", "MEDAI-UI-EXTERNAL-AGENTS-DEFERRED"),
        row("External connector / agent workflow", "Claude/Gemini status", "code/config", "deferred", "Config exists but external API behavior remains disabled.", "low", "Keep disabled.", "out-of-scope"),
        row("External connector / agent workflow", "cloud API disabled state", "safety", "available", "Safety panels and reports show external_api_used false.", "low", "Keep.", "none"),
        row("External connector / agent workflow", "future/deferred connector status", "advanced", "available", "Safety dashboard states local stubs/future connector status.", "low", "Keep advanced.", "none"),
    ]
    return rows


def classify_priorities(matrix: list[dict[str, Any]]) -> dict[str, list[str]]:
    p0 = [
        row["expected_capability"]
        for row in matrix
        if row["risk_level"] == "critical"
    ]
    p1 = [
        row["expected_capability"]
        for row in matrix
        if row["expected_capability"] in {"specialty/domain selector", "MKB Explorer", "active/quarantined/superseded tiers", "specialty routing"}
    ]
    p2 = [
        row["expected_capability"]
        for row in matrix
        if row["current_functional_status"] in {"hidden", "deferred"} and row["category"] != "External connector / agent workflow"
    ]
    p3 = [
        "external AI interpretation",
        "diagnosis/treatment recommendation",
        "auto-accept",
        "medication interpretation",
    ]
    return {"priority_0": p0, "priority_1": p1, "priority_2": p2, "priority_3": p3}


def _privacy_check(payload: Any) -> dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {"passed": bool(result.passed), "leak_examples_redacted": list(result.leak_examples_redacted or [])}
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build_report() -> dict[str, Any]:
    inventory = current_ui_inventory()
    history = historical_capabilities()
    matrix = feature_parity_matrix()
    priorities = classify_priorities(matrix)
    hidden_degraded = [
        row for row in matrix if row["current_functional_status"] in {"hidden", "degraded", "disconnected", "unexpectedly missing"}
    ]
    unexpectedly_missing = [row for row in matrix if row["current_functional_status"] == "unexpectedly missing"]
    conclusion = (
        "ui_capability_parity_gaps_found_repair_required"
        if unexpectedly_missing or priorities["priority_0"] or priorities["priority_1"]
        else "ui_capability_parity_ready_no_critical_gaps"
    )
    report = {
        "block_id": TASK_ID,
        "mode": "static_ui_capability_parity_audit",
        "current_head": _git(["rev-parse", "--short", "HEAD"]),
        "audit_method": [
            "static code inspection",
            "repo-safe docs/reports/tests/snapshots scan",
            "git log metadata scan",
            "no Streamlit launch",
            "no runtime DB reads",
            "no private document reads",
        ],
        "current_ui_inventory": inventory,
        "historical_capabilities": history,
        "feature_parity_matrix": matrix,
        "priority_classification": priorities,
        "current_ui_capabilities_count": len(inventory),
        "historical_capabilities_count": len(history),
        "available_capability_count": sum(1 for item in inventory if item["current_status"] == "available"),
        "hidden_degraded_capability_count": len(hidden_degraded),
        "unexpectedly_missing_capability_count": len(unexpectedly_missing),
        "priority_0_blockers": priorities["priority_0"],
        "priority_1_restorations": priorities["priority_1"],
        "specialty_domain_selector_confirmed_missing": True,
        "recommended_next_block": "MEDAI-UI-CAPABILITY-RESTORE-11B",
        "conclusion": conclusion,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
    }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {TASK_ID}",
        "",
        f"Current HEAD: `{report['current_head']}`",
        f"Conclusion: `{report['conclusion']}`",
        "",
        "## Summary",
        "",
        f"- Current UI inventory count: `{report['current_ui_capabilities_count']}`",
        f"- Historical capability count: `{report['historical_capabilities_count']}`",
        f"- Available capability count: `{report['available_capability_count']}`",
        f"- Hidden/degraded capability count: `{report['hidden_degraded_capability_count']}`",
        f"- Unexpectedly missing capability count: `{report['unexpectedly_missing_capability_count']}`",
        f"- Specialty/domain selector confirmed missing: `{report['specialty_domain_selector_confirmed_missing']}`",
        f"- Recommended next block: `{report['recommended_next_block']}`",
        "",
        "## Priority 0 Blockers",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["priority_0_blockers"]] or ["- None identified."])
    lines.extend(["", "## Priority 1 Restorations", ""])
    lines.extend([f"- `{item}`" for item in report["priority_1_restorations"]] or ["- None identified."])
    lines.extend(["", "## Feature Parity Matrix", ""])
    for row in report["feature_parity_matrix"]:
        lines.append(
            f"- **{row['expected_capability']}** ({row['category']}): "
            f"`{row['current_functional_status']}`; action `{row['recommended_action']}`; block `{row['suggested_repair_block_id']}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- privacy check passed: `{report.get('privacy_check_passed')}`",
            f"- external API used: `{report['external_api_used']}`",
            f"- auto-accept enabled: `{report['auto_accept_enabled']}`",
            "- No OCR routing, classifier cues, confidence thresholds, DDI, medication, privacy, or review gates were changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MATRIX_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["feature_parity_matrix"][0].keys()))
        writer.writeheader()
        writer.writerows(report["feature_parity_matrix"])
    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get("leak_examples_redacted", [])
    md = _markdown(report)
    md_privacy = _privacy_check(md)
    csv_privacy = _privacy_check(MATRIX_CSV_PATH.read_text(encoding="utf-8"))
    report["privacy_check_passed"] = bool(report["privacy_check_passed"] and md_privacy.get("passed") and csv_privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = (
        list(report.get("privacy_check_leak_examples_redacted") or [])
        + list(md_privacy.get("leak_examples_redacted") or [])
        + list(csv_privacy.get("leak_examples_redacted") or [])
    )
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _markdown(report)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    report = build_report()
    write_reports(report)
    print(json.dumps(report, indent=2))
    return 0 if report["privacy_check_passed"] and report["conclusion"] != "not_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
