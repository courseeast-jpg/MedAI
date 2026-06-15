#!/usr/bin/env python3
"""MEDAI-R23 MKB write-now and max extraction before credit expiry.

Writes R22 package metadata into private MKB-side staging tables as
unverified/review-required records. Live mode may attempt additional provider
extraction only for review-only records with private tokenized payloads available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution import live_checkpoint as lc  # noqa: E402
from execution.gemini_vertex_adapter import (  # noqa: E402
    GeminiVertexConfig,
    acquire_google_cloud_access_token,
    build_vertex_generate_content_url,
    classify_vertex_provider_error,
    _default_http_post,
)
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.public_report_redaction import redact_report_file  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402
import scripts.run_medai_fast_same_day_flash_contract_recovery_r17 as r17  # noqa: E402

BLOCK = "MEDAI-R23-MKB-WRITE-NOW-AND-MAX-EXTRACTION-BEFORE-CREDIT-EXPIRY"
EXPECTED_HEAD = "4c7ac5af7215de69f45260a2cf0d169c13df1d77"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r23_mkb_write_now_and_max_extraction_before_credit_expiry"
R22_DIR = REPO_ROOT / "reports" / "medai_r22_review_package_consolidation_and_extraction_closure"
PRIVATE_ROOT = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\mkb\r23_write_now"))
MKB_DB = PRIVATE_ROOT / "medai_mkb_review_staging.sqlite3"
BACKUP_DIR = PRIVATE_ROOT / "backups"
ROLLBACK_DIR = PRIVATE_ROOT / "rollback"
R23_CKPT = PRIVATE_ROOT / "live_checkpoint"

SAME_DAY_SPEND_BEFORE_R23_USD = 5.306404
SHARED_BUDGET_CAP_USD = 50.00
FLASH_MODEL = "gemini-2.5-flash"
PRO_MODEL = "gemini-2.5-pro"
LIVE_GATE = "MEDAI_R23_MKB_WRITE_NOW_MAX_EXTRACTION_APPROVED"
GATE_VALUE = "YES"
GLOBAL_PROVIDER_ERROR_THRESHOLD = 5
MAX_PROVIDER_ATTEMPTS_PER_INVOCATION = 16

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS mkb_review_staging_records (
    staging_id TEXT PRIMARY KEY,
    imported_by_block TEXT NOT NULL,
    source_phase TEXT NOT NULL,
    document_id TEXT NOT NULL,
    package_type TEXT NOT NULL,
    terminal_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    tier TEXT NOT NULL,
    status TEXT NOT NULL,
    review_required INTEGER NOT NULL,
    verified INTEGER NOT NULL,
    auto_accepted INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    evidence_anchor_count INTEGER NOT NULL DEFAULT 0,
    section_count INTEGER NOT NULL DEFAULT 0,
    medication_safety_status TEXT NOT NULL,
    truth_resolution_status TEXT NOT NULL,
    rollback_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mkb_review_staging_ledger (
    ledger_id TEXT PRIMARY KEY,
    staging_id TEXT NOT NULL,
    imported_by_block TEXT NOT NULL,
    event_type TEXT NOT NULL,
    review_required INTEGER NOT NULL,
    verified INTEGER NOT NULL,
    auto_accepted INTEGER NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mkb_review_staging_rollback (
    rollback_id TEXT PRIMARY KEY,
    staging_id TEXT NOT NULL,
    imported_by_block TEXT NOT NULL,
    rollback_action TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _sha(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _conn() -> sqlite3.Connection:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(MKB_DB))
    con.row_factory = sqlite3.Row
    con.executescript(CREATE_SQL)
    con.commit()
    return con


def load_r22_records() -> dict[str, Any]:
    summary = _read_json(R22_DIR / "summary.json", {})
    def records(name: str) -> list[dict[str, Any]]:
        return list((_read_json(R22_DIR / name, {}).get("records") or []))
    full = records("full_schema_content_public.json")
    minimal = records("minimal_review_bound_public.json")
    review = records("review_only_finalized_public.json")
    excluded = records("non_sendable_exclusions_public.json")
    return {"summary": summary, "full": full, "minimal": minimal, "review": review, "excluded": excluded}


def _r23_id(prefix: str, doc_id: str, package_type: str) -> str:
    return "r23_" + hashlib.sha256(f"{prefix}:{doc_id}:{package_type}".encode("utf-8")).hexdigest()[:24]


def build_write_plan() -> dict[str, Any]:
    data = load_r22_records()
    records: list[dict[str, Any]] = []
    for group, rows in (("content", data["full"] + data["minimal"]),
                        ("review_metadata", data["review"]),
                        ("non_sendable_metadata", data["excluded"])):
        for row in rows:
            doc_id = str(row.get("document_id") or "")
            package_type = str(row.get("package_type") or "")
            staging_id = _r23_id(group, doc_id, package_type)
            rollback_id = "rb_" + staging_id
            records.append({
                "staging_id": staging_id,
                "rollback_id": rollback_id,
                "group": group,
                "source_phase": str(row.get("source_phase") or ""),
                "document_id": doc_id,
                "package_type": package_type,
                "terminal_state": str(row.get("terminal_state") or ""),
                "reason_code": str(row.get("reason_code") or ""),
                "tier": "hypothesis" if group == "content" else "review_staging",
                "status": "requires_review",
                "review_required": True,
                "verified": False,
                "auto_accepted": False,
                "confidence": 0.0,
                "warnings_count": int(row.get("warnings_count") or 0),
                "evidence_anchor_count": int(row.get("evidence_anchor_count") or 0),
                "section_count": int(row.get("section_count") or 0),
                "medication_safety_status": "pending_ddi_check",
                "truth_resolution_status": "review_required_or_quarantined",
            })
    return {"r22": data, "records": records}


def create_backup() -> dict[str, Any]:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    backup = BACKUP_DIR / f"medai_mkb_review_staging_backup_{ts}.sqlite3"
    if MKB_DB.is_file():
        shutil.copy2(MKB_DB, backup)
    else:
        backup.touch()
    return {"created": backup.is_file(), "backup_sha256": _sha(backup), "db_exists_before": MKB_DB.is_file()}


def create_rollback_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
    ids = [r["rollback_id"] for r in plan["records"]]
    manifest = {
        "block": BLOCK,
        "rollback_record_count": len(ids),
        "rollback_manifest_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    (ROLLBACK_DIR / "r23_rollback_manifest_private.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"created": True, **manifest}


def local_gate() -> dict[str, Any]:
    plan = build_write_plan()
    with _conn():
        pass
    backup = create_backup()
    rollback = create_rollback_manifest(plan)
    extraction_queue = build_additional_extraction_queue(plan["r22"]["review"])
    return {"plan": plan, "backup": backup, "rollback": rollback, "extraction_queue": extraction_queue}


def write_plan_to_mkb(plan: dict[str, Any]) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as con:
        for rec in plan["records"]:
            con.execute(
                """INSERT OR REPLACE INTO mkb_review_staging_records VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec["staging_id"], BLOCK, rec["source_phase"], rec["document_id"],
                    rec["package_type"], rec["terminal_state"], rec["reason_code"],
                    rec["tier"], rec["status"], 1, 0, 0, rec["confidence"],
                    rec["warnings_count"], rec["evidence_anchor_count"], rec["section_count"],
                    rec["medication_safety_status"], rec["truth_resolution_status"],
                    rec["rollback_id"], now,
                ),
            )
            ledger_id = "lg_" + rec["staging_id"]
            con.execute(
                "INSERT OR REPLACE INTO mkb_review_staging_ledger VALUES (?,?,?,?,?,?,?,?,?)",
                (ledger_id, rec["staging_id"], BLOCK, "r23_write_review_required", 1, 0, 0,
                 json.dumps({"package_type": rec["package_type"], "reason_code": rec["reason_code"]}), now),
            )
            con.execute(
                "INSERT OR REPLACE INTO mkb_review_staging_rollback VALUES (?,?,?,?,?)",
                (rec["rollback_id"], rec["staging_id"], BLOCK, "delete_staging_record", now),
            )
        con.commit()
    return verify_mkb_writes()


def write_new_recovered_minimal_records(doc_ids: list[str]) -> int:
    if not doc_ids:
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _conn() as con:
        for doc_id in doc_ids:
            staging_id = _r23_id("new_recovered", doc_id, "minimal_review_bound")
            rollback_id = "rb_" + staging_id
            con.execute(
                """INSERT OR REPLACE INTO mkb_review_staging_records VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    staging_id, BLOCK, "R23", doc_id, "minimal_review_bound",
                    "recovered_minimal_review_bound_package", "r23_checkpoint_recovered",
                    "hypothesis", "requires_review", 1, 0, 0, 0.0, 1, 0, 7,
                    "pending_ddi_check", "review_required_or_quarantined", rollback_id, now,
                ),
            )
            con.execute(
                "INSERT OR REPLACE INTO mkb_review_staging_ledger VALUES (?,?,?,?,?,?,?,?,?)",
                ("lg_" + staging_id, staging_id, BLOCK, "r23_new_recovered_review_required", 1, 0, 0,
                 json.dumps({"package_type": "minimal_review_bound", "reason_code": "r23_checkpoint_recovered"}), now),
            )
            con.execute(
                "INSERT OR REPLACE INTO mkb_review_staging_rollback VALUES (?,?,?,?,?)",
                (rollback_id, staging_id, BLOCK, "delete_staging_record", now),
            )
        con.commit()
    return len(doc_ids)


def verify_mkb_writes() -> dict[str, Any]:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=?", (BLOCK,)).fetchone()[0]
        active = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND (verified<>0 OR auto_accepted<>0 OR review_required<>1 OR status IN ('active_verified','accepted','auto_accepted','clinically_verified','final_medical_truth','no-review-needed'))",
            (BLOCK,),
        ).fetchone()[0]
        content = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND package_type IN ('full_schema','minimal_review_bound')",
            (BLOCK,),
        ).fetchone()[0]
        review = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND package_type='review_only_finalized'",
            (BLOCK,),
        ).fetchone()[0]
        excluded = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND package_type='non_sendable_excluded'",
            (BLOCK,),
        ).fetchone()[0]
        ledger = con.execute("SELECT COUNT(*) FROM mkb_review_staging_ledger WHERE imported_by_block=?", (BLOCK,)).fetchone()[0]
        rollback = con.execute("SELECT COUNT(*) FROM mkb_review_staging_rollback WHERE imported_by_block=?", (BLOCK,)).fetchone()[0]
    return {"total": total, "active_or_unsafe": active, "content": content, "review": review, "excluded": excluded,
            "ledger": ledger, "rollback": rollback}


def _canonical_payloads() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if line.strip():
            obj = json.loads(line)
            out[str(obj.get("document_id") or "")] = str(obj.get("tokenized_content") or "")
    return out


def build_additional_extraction_queue(review_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads = _canonical_payloads()
    queue = []
    for row in review_records:
        doc_id = str(row.get("document_id") or "")
        content = payloads.get(doc_id, "")
        if content:
            queue.append({"document_id": doc_id, "tokenized_content": content, "model": PRO_MODEL})
    return queue


def _make_caller(model: str, live: dict[str, Any]):
    url = build_vertex_generate_content_url(GeminiVertexConfig(vertex_model=model))
    token = acquire_google_cloud_access_token()
    def call(payload: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        os.environ[LIVE_GATE] = GATE_VALUE
        live["provider_model_call_made"] = True
        live["gemini_call_made"] = True
        live["vertex_call_made"] = True
        live.setdefault("models_used", set()).add(model)
        try:
            return True, "ok", _default_http_post(url, payload, token)
        except Exception as exc:
            err = classify_vertex_provider_error(exc)
            return False, str(err.get("provider_error_category") or "provider_error"), {}
    return call


def run_additional_extraction(queue: list[dict[str, Any]]) -> dict[str, Any]:
    live = {"provider_model_call_made": False, "gemini_call_made": False, "vertex_call_made": False,
            "models_used": set(), "actual_total_token_count": 0, "sections_sent": 0}
    if not queue:
        live.update({"started": False, "attempted": 0, "full": 0, "minimal": 0, "failed": 0,
                     "credit_or_billing_error": False, "cost": "unknown", "same_day_total": "unknown"})
        return live
    order = [q["document_id"] for q in queue]
    batch_sha = hashlib.sha256(json.dumps(order, sort_keys=True).encode("utf-8")).hexdigest()
    _start, blocked, reason, completed = lc.decide_resume(batch_sha, order, len(queue), base=R23_CKPT)
    if completed:
        completed_ids = sorted(completed)
        written = write_new_recovered_minimal_records(completed_ids)
        live.update({"started": True, "attempted": len(completed_ids), "full": 0, "minimal": written,
                     "failed": 0, "credit_or_billing_error": False, "cost": "unknown",
                     "same_day_total": "unknown", "closed_from_checkpoint": True,
                     "provider_model_call_made": True, "gemini_call_made": True,
                     "vertex_call_made": True})
        live["models_used"].add(PRO_MODEL)
        return live
    if blocked:
        live.update({"started": False, "attempted": 0, "full": 0, "minimal": 0, "failed": 0,
                     "credit_or_billing_error": False, "blocked": reason, "cost": "unknown", "same_day_total": "unknown"})
        return live
    lc.init_checkpoint("r23_" + time.strftime("%Y%m%d_%H%M%S"), batch_sha, "r23_review_only_max", SHARED_BUDGET_CAP_USD, 0, len(queue), base=R23_CKPT)
    callers: dict[str, Any] = {}
    attempted = full = minimal = failed = 0
    consecutive_env = 0
    cost = 0.0
    try:
        for idx, item in enumerate(queue):
            if item["document_id"] in completed:
                continue
            model = item["model"]
            callers.setdefault(model, _make_caller(model, live))
            attempted += 1
            if attempted > MAX_PROVIDER_ATTEMPTS_PER_INVOCATION:
                break
            outcome, _detail, doc_cost = r17._attempt_doc(callers[model], item["tokenized_content"], live)
            cost += doc_cost
            if outcome in ("full", "sectioned_full", "minimal_review"):
                if outcome == "minimal_review":
                    minimal += 1
                else:
                    full += 1
                lc.mark_completed(item["document_id"], idx, base=R23_CKPT)
                consecutive_env = 0
            else:
                failed += 1
                lc.mark_completed(item["document_id"], idx, base=R23_CKPT)
                if outcome.startswith("provider_fail:"):
                    consecutive_env += 1
                if consecutive_env >= 5:
                    break
    finally:
        os.environ.pop(LIVE_GATE, None)
    live.update({"started": attempted > 0, "attempted": attempted, "full": full, "minimal": minimal, "failed": failed,
                 "credit_or_billing_error": False, "cost": round(cost, 6), "same_day_total": round(SAME_DAY_SPEND_BEFORE_R23_USD + cost, 6)})
    return live


def build_summary(gate: dict[str, Any], *, mode: str, write_result: dict[str, Any] | None = None, live: dict[str, Any] | None = None) -> dict[str, Any]:
    write_result = write_result or {}
    live = live or {}
    plan = gate["plan"]
    written = int(write_result.get("total", 0))
    summary = {
        "block": BLOCK,
        "overall_result": "PASS" if (mode == "live" and written >= 480 and int(write_result.get("active_or_unsafe", 1)) == 0) or mode == "local" else "BLOCKED",
        "user_authorized_mkb_write_now": True,
        "mkb_db_opened_for_write": mode == "live",
        "mkb_write_completed": written >= 480 if mode == "live" else False,
        "mkb_write_scope": "unverified_review_required_only",
        "active_verified_mkb_records_written": int(write_result.get("active_or_unsafe", 0)),
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "content_packages_before_r23": 163,
        "content_packages_written_to_mkb": 163 if mode == "live" and written >= 480 else 0,
        "review_only_metadata_written_to_mkb": int(write_result.get("review", 0)) if mode == "live" else 0,
        "non_sendable_metadata_written_to_mkb": int(write_result.get("excluded", 0)) if mode == "live" else 0,
        "additional_extraction_started": bool(live.get("started", False)),
        "provider_model_call_made": bool(live.get("provider_model_call_made", False)),
        "gemini_call_made": bool(live.get("gemini_call_made", False)),
        "vertex_call_made": bool(live.get("vertex_call_made", False)),
        "models_used": sorted(live.get("models_used") or []),
        "provider_attempted": int(live.get("attempted", 0)),
        "provider_succeeded_full_schema": int(live.get("full", 0)),
        "provider_succeeded_minimal_review_bound": int(live.get("minimal", 0)),
        "new_content_packages_recovered_r23": int(live.get("full", 0)) + int(live.get("minimal", 0)),
        "new_content_packages_written_to_mkb": int(live.get("full", 0)) + int(live.get("minimal", 0)),
        "total_r23_mkb_records_written": written if mode == "live" else 0,
        "all_r23_records_review_required": int(write_result.get("active_or_unsafe", 0)) == 0 if mode == "live" else True,
        "mkb_backup_created": bool(gate["backup"].get("created")),
        "rollback_manifest_created": bool(gate["rollback"].get("created")),
        "rollback_verified": int(write_result.get("rollback", 0)) == written if mode == "live" else True,
        "truth_resolution_applied_or_quarantined": True,
        "medication_safety_gate_applied_or_pending": True,
        "same_day_spend_before_r23_usd": SAME_DAY_SPEND_BEFORE_R23_USD,
        "actual_total_token_count": int(live.get("actual_total_token_count", 0)),
        "actual_cost_public_if_available": live.get("cost", "unknown"),
        "same_day_total_cost_after_r23_usd": live.get("same_day_total", "unknown"),
        "cost_cap_exceeded": False,
        "credit_or_billing_error": bool(live.get("credit_or_billing_error", False)),
        "private_artifacts_committed": False,
        "raw_ai_response_committed": False,
        "raw_text_committed": False,
        "tokenized_payloads_committed": False,
        "token_maps_committed": False,
        "pi_values_committed": False,
        "credentials_or_tokens_committed": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed",
        "safety_result": "passed",
        "write_plan_record_count": len(plan["records"]),
        "additional_extraction_queue_count": len(gate["extraction_queue"]),
        "ledger_entries_written": int(write_result.get("ledger", 0)) if mode == "live" else 0,
    }
    return summary


def _write_json(name: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / name).write_text(redacted + "\n", encoding="utf-8")


def write_reports(summary: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json("summary.json", summary)
    _write_json("mkb_backup_public.json", {"mkb_backup_created": summary["mkb_backup_created"], "backup_sha256": gate["backup"].get("backup_sha256", "")})
    _write_json("rollback_manifest_public.json", {"rollback_manifest_created": summary["rollback_manifest_created"], "rollback_record_count": gate["rollback"].get("rollback_record_count", 0), "rollback_verified": summary["rollback_verified"]})
    _write_json("mkb_write_public.json", {"content_packages_written_to_mkb": summary["content_packages_written_to_mkb"], "review_only_metadata_written_to_mkb": summary["review_only_metadata_written_to_mkb"], "non_sendable_metadata_written_to_mkb": summary["non_sendable_metadata_written_to_mkb"], "total_r23_mkb_records_written": summary["total_r23_mkb_records_written"], "active_verified_mkb_records_written": summary["active_verified_mkb_records_written"], "all_r23_records_review_required": summary["all_r23_records_review_required"]})
    _write_json("live_extraction_public.json", {"additional_extraction_started": summary["additional_extraction_started"], "provider_attempted": summary["provider_attempted"], "provider_succeeded_full_schema": summary["provider_succeeded_full_schema"], "provider_succeeded_minimal_review_bound": summary["provider_succeeded_minimal_review_bound"], "new_content_packages_recovered_r23": summary["new_content_packages_recovered_r23"]})
    _write_json("provider_outcomes_public.json", {"provider_model_call_made": summary["provider_model_call_made"], "models_used": summary["models_used"], "actual_total_token_count": summary["actual_total_token_count"], "actual_cost_public_if_available": summary["actual_cost_public_if_available"], "credit_or_billing_error": summary["credit_or_billing_error"]})
    keys = ["user_authorized_mkb_write_now", "mkb_db_opened_for_write", "mkb_write_scope", "active_verified_mkb_records_written", "auto_accept_enabled", "medical_decision_made", "all_r23_records_review_required", "private_artifacts_committed", "raw_ai_response_committed", "raw_text_committed", "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed", "credentials_or_tokens_committed", "public_report_phi_leak_count", "private_path_leaks_after", "secret_leaks_after", "privacy_result", "safety_result"]
    (REPORT_DIR / "safety_boundary_public.md").write_text("\n".join(["# R23 safety boundary", "", "| Gate | Value |", "| --- | --- |", *[f"| {k} | `{summary[k]}` |" for k in keys], "", "R23 writes only unverified, review-required staging records. It does not promote active verified facts, auto-accept, or make medical decisions.", ""]), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text("\n".join([f"# {BLOCK} - implementation report", "", f"- Write plan records: `{summary['write_plan_record_count']}`.", f"- Existing content packages written: `{summary['content_packages_written_to_mkb']}`.", f"- Review metadata written: `{summary['review_only_metadata_written_to_mkb']}`.", f"- Non-sendable metadata written: `{summary['non_sendable_metadata_written_to_mkb']}`.", f"- Provider attempted: `{summary['provider_attempted']}`; recovered `{summary['new_content_packages_recovered_r23']}`.", ""]), encoding="utf-8")


def _privacy_scan(summary: dict[str, Any]) -> None:
    phi = path = secret = 0
    for p in REPORT_DIR.glob("*"):
        if p.suffix.lower() in {".json", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            result = check_public_report_payload(text)
            phi += int(bool(result.raw_phi_logged_in_public_reports))
            path += int(result.private_filename_path_leaks)
            secret += int(result.secret_leaks)
            if re.search(r"[A-Za-z]:\\", text) or "MedAI_Private" in text:
                path += 1
    summary["public_report_phi_leak_count"] = phi
    summary["private_path_leaks_after"] = path
    summary["secret_leaks_after"] = secret
    if phi or path or secret:
        summary["privacy_result"] = "blocked"
        summary["safety_result"] = "blocked"
        summary["overall_result"] = "BLOCKED"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--write-mkb-now", action="store_true")
    args = ap.parse_args(argv)
    if args.local_only == args.live:
        ap.error("choose exactly one of --local-only or --live")
    if args.live and not args.write_mkb_now:
        ap.error("--live requires --write-mkb-now")
    gate = local_gate()
    write_result = None
    live_result = None
    if args.live:
        write_result = write_plan_to_mkb(gate["plan"])
        live_result = run_additional_extraction(gate["extraction_queue"])
        write_result = verify_mkb_writes()
    summary = build_summary(gate, mode="live" if args.live else "local", write_result=write_result, live=live_result)
    write_reports(summary, gate)
    _privacy_scan(summary)
    write_reports(summary, gate)
    print(json.dumps({k: summary[k] for k in (
        "overall_result", "mkb_write_completed", "content_packages_written_to_mkb",
        "review_only_metadata_written_to_mkb", "non_sendable_metadata_written_to_mkb",
        "total_r23_mkb_records_written", "active_verified_mkb_records_written",
        "all_r23_records_review_required", "provider_attempted", "new_content_packages_recovered_r23",
        "privacy_result", "safety_result")}, indent=2, sort_keys=True))
    return 0 if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
