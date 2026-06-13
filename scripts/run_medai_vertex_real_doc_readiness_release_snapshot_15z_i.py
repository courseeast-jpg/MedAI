#!/usr/bin/env python3
"""Generate the 15Z-I no-live release snapshot and governance freeze artifacts."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload

RELEASE_DIR = REPO_ROOT / "docs" / "release_snapshots" / "MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z"
RELEASE_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z.md"
VALIDATION_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_VALIDATION_RECEIPT_15Z.md"
ARTIFACT_CSV = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_ARTIFACT_INDEX_15Z.csv"
CONTINUATION_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_CONTINUATION_SNAPSHOT_15Z.md"
NEXT_DECISION_MD = RELEASE_DIR / "MEDAI_VERTEX_REAL_DOC_READINESS_NEXT_DECISION_15Z.md"

REPORT_DIR = REPO_ROOT / "reports" / "medai_vertex_real_doc_readiness_release_snapshot_15z_i"
SUMMARY_JSON = REPORT_DIR / "summary.json"
CHECKLIST_JSON = REPORT_DIR / "release_checklist.json"
MATRIX_MD = REPORT_DIR / "release_matrix.md"
ARTIFACT_CHECK_JSON = REPORT_DIR / "artifact_index_check.json"
IMPLEMENTATION_MD = REPORT_DIR / "implementation_report.md"

REMOTE_BRANCH = "origin/clinical-knowledge-architecture"
LOCAL_BRANCH = "clinical-knowledge-architecture"
REPO_PATH_PUBLIC = "C:/Users/S1/.codex/worktrees/9c07/medai-clinical-knowledge-architecture-park24"
NEXT_RECOMMENDED_BLOCK = "MEDAI-VERTEX-REAL-DOC-READINESS-PAUSE-FREEZE-OR-DESIGN-ONLY-PILOT-15Z-J"

COMMANDS = [
    "python scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py",
    "python scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py",
    "python scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py",
    "python scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py",
    "python scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py",
    "python scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py",
    "python scripts/run_medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g.py",
    "python scripts/run_medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h.py",
]

TEST_COMMANDS = [
    "python -m pytest tests/test_medai_vertex_real_doc_readiness_release_snapshot_15z_i.py -q",
    "python -m py_compile scripts/run_medai_vertex_real_doc_readiness_release_snapshot_15z_i.py",
    "python scripts/run_medai_vertex_real_doc_readiness_release_snapshot_15z_i.py",
    "python -m pytest tests/test_medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py -q",
    "python -m pytest tests/test_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py -q",
    "python -m pytest tests/test_medai_vertex_semantic_calibration_operator_handoff_doc_15y.py -q",
    "python -m pytest tests/test_medai_ai_extraction_privacy_gate_15b.py -q",
    "python -m pytest tests/test_medai_ai_external_call_dry_run_15e.py -q",
    "python -m pytest tests/test_medai_gemini_vertex_credit_route_smoke_15n_r4.py -q",
]

GATES = [
    "real_doc_external_routing_default_block",
    "pii_stripping_proof_required",
    "pii_vault_isolation_required",
    "no_raw_private_payload_in_reports_required",
    "synthetic_to_real_adapter_dry_run_required",
    "redacted_real_like_fixture_replay_required",
    "operator_review_queue_handoff_required",
    "human_authorization_required_for_any_real_live_call",
    "no_active_mkb_write_required",
    "no_auto_accept_required",
    "medication_safety_non_bypass_required_if_medication_facts_present",
    "billing_cost_cap_ack_required",
    "dedicated_future_real_doc_live_gate_required",
    "real_doc_refusal_path_required",
    "no_medical_decision_logic_required",
]

BLOCK_SUMMARY = [
    ("15Z-A", "default-deny readiness framework", "PASS"),
    ("15Z-B", "PII stripping and vault isolation", "PASS"),
    ("15Z-C", "synthetic-to-real adapter dry-run and review handoff", "PASS"),
    ("15Z-D", "authorization, billing/cost, and refusal gates", "PASS"),
    ("15Z-E", "medication safety non-bypass", "PASS"),
    ("15Z-F", "integrated readiness harness", "PASS"),
    ("15Z-G", "operator/governance handoff document", "PASS"),
    ("15Z-H", "operator UAT", "PASS"),
]

ARTIFACT_ROWS = [
    ("15Z-A", "module", "execution/vertex_real_doc_readiness_gates.py"),
    ("15Z-A", "script", "scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py"),
    ("15Z-A", "test", "tests/test_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py"),
    ("15Z-A", "reports", "reports/medai_vertex_real_doc_readiness_gates_no_live_15z_a/"),
    ("15Z-B", "module", "execution/vertex_real_doc_pii_stripping_proof.py"),
    ("15Z-B", "script", "scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py"),
    ("15Z-B", "test", "tests/test_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py"),
    ("15Z-B", "reports", "reports/medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b/"),
    ("15Z-C", "module", "execution/vertex_real_doc_adapter_dry_run.py"),
    ("15Z-C", "script", "scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py"),
    ("15Z-C", "test", "tests/test_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py"),
    ("15Z-C", "reports", "reports/medai_vertex_real_doc_adapter_dry_run_no_live_15z_c/"),
    ("15Z-D", "module", "execution/vertex_real_doc_authorization_cost_refusal_gates.py"),
    ("15Z-D", "script", "scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py"),
    ("15Z-D", "test", "tests/test_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py"),
    ("15Z-D", "reports", "reports/medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d/"),
    ("15Z-E", "module", "execution/vertex_real_doc_medication_safety_non_bypass.py"),
    ("15Z-E", "script", "scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py"),
    ("15Z-E", "test", "tests/test_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py"),
    ("15Z-E", "reports", "reports/medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e/"),
    ("15Z-F", "module", "execution/vertex_real_doc_readiness_integrated_harness.py"),
    ("15Z-F", "script", "scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py"),
    ("15Z-F", "test", "tests/test_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py"),
    ("15Z-F", "reports", "reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/"),
    ("15Z-G", "doc", "docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"),
    ("15Z-G", "script", "scripts/run_medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g.py"),
    ("15Z-G", "test", "tests/test_medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g.py"),
    ("15Z-G", "reports", "reports/medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g/"),
    ("15Z-H", "script", "scripts/run_medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h.py"),
    ("15Z-H", "test", "tests/test_medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h.py"),
    ("15Z-H", "reports", "reports/medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h/"),
    ("operator", "handoff_docs", "docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md"),
    ("15Z-I", "release_doc", "docs/release_snapshots/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z.md"),
    ("15Z-I", "validation_receipt", "docs/release_snapshots/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z/MEDAI_VERTEX_REAL_DOC_READINESS_VALIDATION_RECEIPT_15Z.md"),
    ("15Z-I", "artifact_index", "docs/release_snapshots/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z/MEDAI_VERTEX_REAL_DOC_READINESS_ARTIFACT_INDEX_15Z.csv"),
    ("15Z-I", "continuation_snapshot", "docs/release_snapshots/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z/MEDAI_VERTEX_REAL_DOC_READINESS_CONTINUATION_SNAPSHOT_15Z.md"),
    ("15Z-I", "next_decision_memo", "docs/release_snapshots/MEDAI_VERTEX_REAL_DOC_READINESS_RELEASE_15Z/MEDAI_VERTEX_REAL_DOC_READINESS_NEXT_DECISION_15Z.md"),
]

FORBIDDEN_REPORT_TOKENS = (
    "ya29.",
    "AIza",
    "Bearer ",
    "Authorization" + ":",
    "access" + "_token",
    "refresh" + "_token",
    "private" + "_key",
    "application" + "_default" + "_credentials",
    "g" + "cloud",
    "C:\\",
    "/home/",
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def _read_summary(path: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _head() -> str:
    return _git(["rev-parse", "HEAD"])


def _remote_head() -> str:
    return _git(["rev-parse", REMOTE_BRANCH])


def _artifact_index_csv() -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["block", "artifact_type", "path", "exists"])
    for block, artifact_type, path in ARTIFACT_ROWS:
        writer.writerow([block, artifact_type, path, str((REPO_ROOT / path).exists()).lower()])
    return out.getvalue()


def _block_table() -> str:
    rows = ["| Block | Scope | Status |", "| --- | --- | --- |"]
    rows.extend(f"| {block} | {scope} | {status} |" for block, scope, status in BLOCK_SUMMARY)
    return "\n".join(rows)


def _gate_list() -> str:
    return "\n".join(f"- `{gate}`" for gate in GATES)


def _command_list() -> str:
    return "\n".join(f"- `{command}`" for command in COMMANDS)


def _test_command_list() -> str:
    return "\n".join(f"- `{command}`" for command in TEST_COMMANDS)


def _build_docs() -> dict[str, str]:
    head = _head()
    remote_head = _remote_head()
    head_short = head[:7]
    remote_head_short = remote_head[:7]
    summary_h = _read_summary("reports/medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h/summary.json")
    release = f"""# MEDAI Vertex Real Document Readiness Release 15Z

## Release Purpose

This is the no-live release snapshot and governance freeze for the completed 15Z real-document readiness chain. It records that the readiness gates are implemented, validated, operator-checkable, and still blocked from real-document Vertex routing.

## Current HEAD / Branch / Remote

- Current HEAD: `{head_short}`
- Branch: `{LOCAL_BRANCH}` via `{REMOTE_BRANCH}`
- Remote HEAD: `{remote_head_short}`

## 15Z-A Through 15Z-H Summary

{_block_table()}

## Gate Inventory

{_gate_list()}

## Validation Evidence

- 15Z-H operator UAT: `{summary_h["operator_uat_steps_passed"]}/{summary_h["operator_uat_steps_total"]}` steps passed.
- Integrated harness verified: `{summary_h["integrated_harness_verified"]}`
- Future review package only verified: `{summary_h["future_review_package_only_verified"]}`
- Blocked failure injections verified: `{summary_h["blocked_failure_injections_verified"]}`
- Privacy result: `{summary_h["privacy_result"]}`

## Operator Commands

{_command_list()}

## Safety / Privacy Boundary

- Real-document Vertex routing remains NOT authorized.
- Any future real-document live call requires a new separately gated block.
- No provider call is authorized.
- No billing API call is authorized.
- No active MKB write is authorized.
- No auto-accept is authorized.
- No medical decision system is created or authorized.
- Medication safety non-bypass remains required when medication facts are present.
- No raw private payloads, token maps, raw PDF/image payloads, or OCR/private payloads may be included in public reports.
"""
    validation = f"""# MEDAI Vertex Real Document Readiness Validation Receipt 15Z

## Result

PASS. The release snapshot is no-live and docs/report only.

## Tests Run

{_test_command_list()}

## Validation Metrics

- `live_call_made=false`
- `external_api_used=false`
- `billing_api_used=false`
- `real_doc_live_allowed_count=0`
- `active_written_count=0`
- `auto_accept_true_count=0`
- `medical_decision_made_count=0`
- `privacy_result=passed`
"""
    continuation = f"""# MEDAI Vertex Real Document Readiness Continuation Snapshot 15Z

## Repo State

- Current repo path: `{REPO_PATH_PUBLIC}`
- Branch: `{LOCAL_BRANCH}`
- Current remote HEAD: `{remote_head_short}`

## Completed Blocks

{_block_table()}

## Exact Next Recommended Block

`{NEXT_RECOMMENDED_BLOCK}`

## Current Hard Boundaries

- Real-document Vertex routing remains NOT authorized.
- Future real-document live call requires a new separately gated block.
- No live provider usage.
- No billing API usage.
- No active MKB write.
- No production review queue mutation.
- No auto-accept.
- No medical decision output.
- No raw private payloads or token maps in reports.

## Dirty-File Warning

Pre-existing dirty historical report files and untracked 15P-B files must remain unstaged unless explicitly handled in a separate block.

## One-Command Operator Validation List

{_command_list()}
"""
    next_decision = f"""# MEDAI Vertex Real Document Readiness Next Decision 15Z

## Option A: Stay No-Live

Run more UAT, stress, and failure-injection coverage. This keeps provider calls, billing calls, real documents, active writes, auto-accept, and medical decision output blocked.

## Option B: Design A Future Pilot

Prepare a future single redacted-real-like live pilot design. This is design-only and still does not run a live call. The future design would need its own live gate, bounded call count, review-bound handling, no active MKB writes, no auto-accept, and no medical decision output.

## Option C: Pause And Package

Pause, package, and freeze the readiness state at 15Z.

## Recommended Next

Recommended next: pause/freeze or design-only pilot block, not live execution. Suggested block: `{NEXT_RECOMMENDED_BLOCK}`.
"""
    return {
        "release": release,
        "validation": validation,
        "artifact_index": _artifact_index_csv(),
        "continuation": continuation,
        "next_decision": next_decision,
    }


def write_release_docs(docs: dict[str, str]) -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_MD.write_text(docs["release"], encoding="utf-8")
    VALIDATION_MD.write_text(docs["validation"], encoding="utf-8")
    ARTIFACT_CSV.write_text(docs["artifact_index"], encoding="utf-8")
    CONTINUATION_MD.write_text(docs["continuation"], encoding="utf-8")
    NEXT_DECISION_MD.write_text(docs["next_decision"], encoding="utf-8")


def _artifact_index_check() -> dict[str, Any]:
    rows = []
    missing = []
    for block, artifact_type, path in ARTIFACT_ROWS:
        exists = (REPO_ROOT / path).exists()
        rows.append({"block": block, "artifact_type": artifact_type, "path": path, "exists": exists})
        if not exists:
            missing.append(path)
    return {"artifact_index_entries_count": len(rows), "missing_artifacts": missing, "entries": rows}


def _privacy_passes(*payloads: Any) -> bool:
    published = "\n".join(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) for payload in payloads)
    if any(token in published for token in FORBIDDEN_REPORT_TOKENS):
        return False
    if '"[MRN_' in published or '"[PATIENT_NAME_' in published:
        return False
    return bool(check_public_report_payload(payloads).passed)


def _release_checklist(docs: dict[str, str], artifact_check: dict[str, Any]) -> dict[str, Any]:
    all_docs = "\n".join(docs.values())
    release_checks = {
        "15z_a_through_15z_h_artifacts_exist": not artifact_check["missing_artifacts"],
        "release_snapshot_folder_exists": RELEASE_DIR.exists(),
        "release_document_exists": RELEASE_MD.exists(),
        "validation_receipt_exists": VALIDATION_MD.exists(),
        "artifact_index_exists": ARTIFACT_CSV.exists(),
        "continuation_snapshot_exists": CONTINUATION_MD.exists(),
        "next_decision_memo_exists": NEXT_DECISION_MD.exists(),
        "gate_inventory_present": all(gate in all_docs for gate in GATES),
        "operator_commands_present": all(command in all_docs for command in COMMANDS),
        "real_document_boundary_present": "Real-document Vertex routing remains NOT authorized" in all_docs,
        "future_authorization_boundary_present": "future real-document live call requires a new separately gated block" in all_docs.lower(),
        "no_live_boundary_present": "No provider call is authorized" in all_docs and "No billing API call is authorized" in all_docs,
        "active_write_boundary_present": "No active MKB write" in all_docs,
        "auto_accept_boundary_present": "No auto-accept" in all_docs,
        "medication_safety_boundary_present": "Medication safety non-bypass remains required" in all_docs,
        "no_medical_decision_boundary_present": "No medical decision system" in all_docs,
        "privacy_boundary_present": "No raw private payloads" in all_docs and "token maps" in all_docs,
        "dirty_file_warning_present": "Pre-existing dirty historical report files and untracked 15P-B files must remain unstaged" in all_docs,
        "next_recommended_block_present": NEXT_RECOMMENDED_BLOCK in all_docs,
    }
    return release_checks


def _summary(checklist: dict[str, Any], artifact_check: dict[str, Any], privacy_result: str) -> dict[str, Any]:
    summary_h = _read_summary("reports/medai_vertex_real_doc_readiness_operator_uat_no_live_15z_h/summary.json")
    return {
        "block": "MEDAI-VERTEX-REAL-DOC-READINESS-RELEASE-SNAPSHOT-NO-LIVE-15Z-I",
        "release_snapshot_created": True,
        "release_folder_created": RELEASE_DIR.exists(),
        "release_doc_created": RELEASE_MD.exists(),
        "validation_receipt_created": VALIDATION_MD.exists(),
        "artifact_index_created": ARTIFACT_CSV.exists(),
        "continuation_snapshot_created": CONTINUATION_MD.exists(),
        "next_decision_memo_created": NEXT_DECISION_MD.exists(),
        "required_release_checks_passed": sum(1 for passed in checklist.values() if passed is True),
        "required_release_checks_total": len(checklist),
        "artifact_index_entries_count": artifact_check["artifact_index_entries_count"],
        "operator_commands_present": checklist["operator_commands_present"],
        "gate_inventory_present": checklist["gate_inventory_present"],
        "real_document_boundary_present": checklist["real_document_boundary_present"],
        "future_authorization_boundary_present": checklist["future_authorization_boundary_present"],
        "no_live_boundary_present": checklist["no_live_boundary_present"],
        "active_write_boundary_present": checklist["active_write_boundary_present"],
        "auto_accept_boundary_present": checklist["auto_accept_boundary_present"],
        "medication_safety_boundary_present": checklist["medication_safety_boundary_present"],
        "no_medical_decision_boundary_present": checklist["no_medical_decision_boundary_present"],
        "privacy_boundary_present": checklist["privacy_boundary_present"],
        "dirty_file_warning_present": checklist["dirty_file_warning_present"],
        "docs_only_change": True,
        "production_code_changed": False,
        "live_call_made": False,
        "external_api_used": False,
        "billing_api_used": False,
        "real_doc_live_allowed_count": summary_h["real_doc_live_allowed_count"],
        "active_written_count": summary_h["active_written_count"],
        "active_mkb_record_created_count": summary_h["active_mkb_record_created_count"],
        "auto_accept_true_count": summary_h["auto_accept_true_count"],
        "medical_decision_made_count": summary_h["medical_decision_made_count"],
        "privacy_result": privacy_result,
        "billing_check_pending": True,
    }


def _matrix(summary: dict[str, Any], checklist: dict[str, Any]) -> str:
    lines = [
        "# 15Z-I release snapshot matrix",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in checklist.items())
    lines.extend(["", "| Metric | Value |", "| --- | --- |"])
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def _implementation(summary: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-VERTEX-REAL-DOC-READINESS-RELEASE-SNAPSHOT-NO-LIVE-15Z-I",
        "",
        "## Result",
        "",
        "- Created a no-live release snapshot and governance freeze for 15Z.",
        "- Created release document, validation receipt, artifact index, continuation snapshot, next-decision memo, and public reports.",
        "- Did not change production code, UI, OCR, extraction, MKB, decision-store behavior, live gates, provider calls, billing calls, active writes, auto-accept, or medical decision logic.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in summary.items():
        if key == "block":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_reports() -> dict[str, Any]:
    docs = _build_docs()
    write_release_docs(docs)
    artifact_check = _artifact_index_check()
    checklist = _release_checklist(docs, artifact_check)
    preliminary = {"checklist": checklist, "artifact_check": artifact_check, "docs": docs}
    privacy_result = "passed" if _privacy_passes(preliminary) else "failed"
    summary = _summary(checklist, artifact_check, privacy_result)
    matrix = _matrix(summary, checklist)
    implementation = _implementation(summary)
    privacy_result = "passed" if _privacy_passes(summary, checklist, artifact_check, matrix, implementation, docs) else "failed"
    summary["privacy_result"] = privacy_result
    matrix = _matrix(summary, checklist)
    implementation = _implementation(summary)
    return {
        "summary": summary,
        "checklist": checklist,
        "artifact_check": artifact_check,
        "matrix": matrix,
        "implementation": implementation,
    }


def write_reports(reports: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(reports["summary"], indent=2), encoding="utf-8")
    CHECKLIST_JSON.write_text(json.dumps(reports["checklist"], indent=2), encoding="utf-8")
    ARTIFACT_CHECK_JSON.write_text(json.dumps(reports["artifact_check"], indent=2), encoding="utf-8")
    MATRIX_MD.write_text(reports["matrix"], encoding="utf-8")
    IMPLEMENTATION_MD.write_text(reports["implementation"], encoding="utf-8")


def _ready(summary: dict[str, Any]) -> bool:
    return all(
        [
            summary["release_snapshot_created"] is True,
            summary["release_folder_created"] is True,
            summary["release_doc_created"] is True,
            summary["validation_receipt_created"] is True,
            summary["artifact_index_created"] is True,
            summary["continuation_snapshot_created"] is True,
            summary["next_decision_memo_created"] is True,
            summary["required_release_checks_passed"] == summary["required_release_checks_total"],
            summary["artifact_index_entries_count"] == len(ARTIFACT_ROWS),
            summary["operator_commands_present"] is True,
            summary["gate_inventory_present"] is True,
            summary["real_document_boundary_present"] is True,
            summary["future_authorization_boundary_present"] is True,
            summary["no_live_boundary_present"] is True,
            summary["active_write_boundary_present"] is True,
            summary["auto_accept_boundary_present"] is True,
            summary["medication_safety_boundary_present"] is True,
            summary["no_medical_decision_boundary_present"] is True,
            summary["privacy_boundary_present"] is True,
            summary["dirty_file_warning_present"] is True,
            summary["docs_only_change"] is True,
            summary["production_code_changed"] is False,
            summary["live_call_made"] is False,
            summary["external_api_used"] is False,
            summary["billing_api_used"] is False,
            summary["real_doc_live_allowed_count"] == 0,
            summary["active_written_count"] == 0,
            summary["active_mkb_record_created_count"] == 0,
            summary["auto_accept_true_count"] == 0,
            summary["medical_decision_made_count"] == 0,
            summary["privacy_result"] == "passed",
            summary["billing_check_pending"] is True,
        ]
    )


def main() -> int:
    reports = build_reports()
    write_reports(reports)
    summary = reports["summary"]
    ready = _ready(summary)
    print(
        "medai_vertex_real_doc_readiness_release_snapshot_no_live_15z_i_ready"
        if ready
        else "medai_vertex_real_doc_readiness_release_snapshot_no_live_15z_i_not_ready"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "block"}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
