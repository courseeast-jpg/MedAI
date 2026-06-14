#!/usr/bin/env python3
"""MEDAI-AI-FIRST-CORPUS-478-BATCH-INTEGRITY-RESTORE-LOCAL-ONLY-17C-R2-R1.

Restore the canonical 478-request private outbound batch after 17C-R2 blocked on a
corrupted/mutated JSONL. Rebuilds deterministically from the 17A tokenized corpus using
the SAME 17B-R2-R1 repair logic (imported), atomically replaces the JSONL (truncate,
never append), seals integrity sidecars (SHA256, integrity JSON, doc-id manifest), and
sets the file read-only when supported.

NO provider call, NO network, NO billing, NO live gate, NO 17C rerun. Public reports
carry counts/hashes/status only — never request bodies, token maps, raw OCR, or PI.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact 17B-R2-R1 repair/validation logic (imported, not duplicated).
import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as r2r1

CANON_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired"))
CANON_BATCH = CANON_DIR / "outbound_requests_private.jsonl"
CANON_BATCH_LABEL = r"C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl"
INTEGRITY_SIDECAR = CANON_DIR / "outbound_requests_integrity_private.json"
SHA_SIDECAR = CANON_DIR / "outbound_requests_private.sha256"
DOCID_MANIFEST = CANON_DIR / "doc_id_manifest_private.json"

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_batch_integrity_restore_local_only_17c_r2_r1"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_BATCH_INTEGRITY_RESTORE_LOCAL_ONLY_17C_R2_R1"
REQUIRED_DOCS = (
    "MEDAI_AI_FIRST_CORPUS_478_BATCH_INTEGRITY_RESTORE_LOCAL_ONLY_17C_R2_R1.md",
    "MEDAI_PRIVATE_BATCH_INTEGRITY_SEAL_POLICY_17C_R2_R1.md",
    "MEDAI_MUTATION_RISK_AND_READONLY_POLICY_17C_R2_R1.md",
    "MEDAI_17C_R2_RERUN_ENTRY_CRITERIA_AFTER_INTEGRITY_RESTORE.md",
)

EXPECTED = 478
BAD_LINES_BEFORE = 591
BAD_PARSEABLE_BEFORE = 457
BAD_MALFORMED_BEFORE = 134


def _file_state(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {"exists": False, "writable": None, "size_bytes": 0, "line_count": 0, "readonly": None}
    st = p.stat()
    lines = sum(1 for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip())
    return {"exists": True, "writable": os.access(p, os.W_OK),
            "readonly": not os.access(p, os.W_OK), "size_bytes": st.st_size, "line_count": lines}


def _rebuild_requests() -> tuple[list[dict[str, Any]], int, list[str]]:
    """Rebuild the canonical 478 request bodies using 17B-R2-R1 logic. Returns
    (requests, residual_pi_failures, notes)."""
    notes: list[str] = []
    validator = json.loads(r2r1.R2_VALIDATOR.read_text(encoding="utf-8"))
    ready = [{"document_id": d["document_id"], "extension": (d.get("extension") or "").lower()}
             for d in validator.get("per_doc", [])]
    failed_ids = set()
    for line in r2r1.R2_FAILED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                failed_ids.add(json.loads(line)["document_id"])
            except Exception:
                continue
    prompt = r2r1.PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    requests: list[dict[str, Any]] = []
    residual_failures = 0
    for rec in ready:
        doc_id = rec["document_id"]
        tok_path = r2r1.CORPUS_DOCS / doc_id / "tokenized_text.txt"
        if not tok_path.is_file():
            notes.append(f"missing_tokenized_text:{doc_id}")
            continue
        text = tok_path.read_text(encoding="utf-8", errors="ignore")
        if doc_id in failed_ids:
            text = r2r1.repair_text(text, r2r1._Counters())
        if r2r1._residual_pi(text):
            residual_failures += 1
        family, dtype = r2r1.FAMILY_BY_EXT.get(rec["extension"], ("unknown", "unknown"))
        requests.append({
            "document_id": doc_id, "source_hash": r2r1._sha16(text), "model": "gemini-2.5-flash-lite",
            "schema_ref": "config/medai_ai_extraction_schema_17b.json",
            "prompt_contract_ref": "config/medai_ai_extraction_prompt_contract_17b.md",
            "document_family": family, "document_type": dtype, "tokenized_content": text,
        })
    return requests, residual_failures, notes


def _atomic_write(requests: list[dict[str, Any]]) -> None:
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    # Clear read-only on existing target so we can replace it.
    if CANON_BATCH.exists() and not os.access(CANON_BATCH, os.W_OK):
        os.chmod(CANON_BATCH, stat.S_IWRITE | stat.S_IREAD)
    tmp = CANON_DIR / "outbound_requests_private.jsonl.tmp"
    # ensure_ascii=True escapes embedded Unicode line separators (U+2028/U+2029/U+0085)
    # so every record is exactly one physical line under BOTH "\n"-split and
    # str.splitlines() — preventing the false "corruption" that blocked 17C-R2.
    with tmp.open("w", encoding="utf-8") as fh:
        for req in requests:
            fh.write(json.dumps(req, ensure_ascii=True) + "\n")
    os.replace(tmp, CANON_BATCH)  # atomic truncate-replace


def _verify_file() -> dict[str, Any]:
    content = CANON_BATCH.read_text(encoding="utf-8", errors="replace")
    nl_lines = [l for l in content.split("\n") if l.strip()]      # JSONL framing (\n only)
    splitlines_count = len([l for l in content.splitlines() if l.strip()])  # robustness cross-check
    parseable = []
    malformed = 0
    for l in nl_lines:
        try:
            parseable.append(json.loads(l))
        except ValueError:
            malformed += 1
    unique = {r.get("document_id") for r in parseable}
    residual = sum(1 for r in parseable if r2r1._residual_pi(r.get("tokenized_content", "")))
    return {"line_count": len(nl_lines), "splitlines_count": splitlines_count,
            "parseable": len(parseable), "unique": len(unique),
            "malformed": malformed, "residual_pi": residual}


def _set_readonly() -> bool:
    try:
        os.chmod(CANON_BATCH, stat.S_IREAD)
        return not os.access(CANON_BATCH, os.W_OK)
    except OSError:
        return False


def _git_status_hints() -> dict[str, Any]:
    # Report-only: detect unexpected untracked scripts / dirty report edits without staging.
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {"untracked_unexpected_scripts": "unavailable", "dirty_report_edits": "unavailable"}
    untracked = [l[3:] for l in out.splitlines() if l.startswith("??")]
    modified = [l[3:] for l in out.splitlines() if l.startswith(" M") or l.startswith("M ")]
    unexpected_scripts = [u for u in untracked if u.startswith("scripts/") and "17c_r2_r1" not in u.lower()
                          and "15p_b" not in u.lower()]
    dirty_reports = [m for m in modified if m.startswith("reports/")]
    return {"untracked_unexpected_scripts_count": len(unexpected_scripts),
            "dirty_report_edits_count": len(dirty_reports)}


def run() -> dict[str, Any]:
    notes: list[str] = []
    before = _file_state(CANON_BATCH)
    requests, residual_failures, rebuild_notes = _rebuild_requests()
    notes += rebuild_notes

    rebuilt_count = len(requests)
    _atomic_write(requests)
    verify = _verify_file()

    # Compute SHA256 of the rebuilt file bytes.
    sha = hashlib.sha256(CANON_BATCH.read_bytes()).hexdigest()
    doc_ids = sorted(r["document_id"] for r in requests)

    integrity_ok = (rebuilt_count == EXPECTED and verify["line_count"] == EXPECTED
                    and verify["splitlines_count"] == EXPECTED
                    and verify["parseable"] == EXPECTED and verify["unique"] == EXPECTED
                    and verify["malformed"] == 0 and verify["residual_pi"] == 0
                    and residual_failures == 0 and len(notes) == 0)

    # Seal sidecars (private, outside repo).
    sidecar_written = sha_written = manifest_written = False
    try:
        INTEGRITY_SIDECAR.write_text(json.dumps({
            "line_count": verify["line_count"], "splitlines_count": verify["splitlines_count"],
            "parseable": verify["parseable"],
            "unique_doc_ids": verify["unique"], "malformed": verify["malformed"],
            "residual_pi_failures": verify["residual_pi"], "request_validation_passed": integrity_ok,
            "sha256": sha, "expected_request_count": EXPECTED}, indent=2), encoding="utf-8")
        sidecar_written = True
        SHA_SIDECAR.write_text(sha + "  outbound_requests_private.jsonl\n", encoding="utf-8")
        sha_written = True
        DOCID_MANIFEST.write_text(json.dumps({"count": len(doc_ids), "doc_ids": doc_ids}, indent=2), encoding="utf-8")
        manifest_written = True
    except OSError:
        notes.append("sidecar_write_failed")

    readonly_set = _set_readonly() if integrity_ok else False
    after = _file_state(CANON_BATCH)
    hints = _git_status_hints()

    summary = {
        "block": "MEDAI-AI-FIRST-CORPUS-478-BATCH-INTEGRITY-RESTORE-LOCAL-ONLY-17C-R2-R1",
        "local_only": True,
        "provider_call_made": False,
        "vertex_live_execution": False,
        "gemini_call_made": False,
        "claude_call_made": False,
        "openai_call_made": False,
        "billing_api_call_made": False,
        "live_gate_set": False,
        "live_extraction_started": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "production_queue_mutated": False,
        "canonical_batch_path": CANON_BATCH_LABEL,
        "bad_lines_before": before["line_count"] if before["exists"] else BAD_LINES_BEFORE,
        "bad_parseable_before": BAD_PARSEABLE_BEFORE,
        "bad_malformed_before": BAD_MALFORMED_BEFORE,
        "expected_request_count": EXPECTED,
        "rebuilt_request_count": rebuilt_count,
        "parseable_after": verify["parseable"],
        "unique_doc_ids_after": verify["unique"],
        "malformed_after": verify["malformed"],
        "request_validation_passed_after": integrity_ok,
        "residual_pi_failures_after": verify["residual_pi"],
        "old_12_added_separately": False,
        "combined_batch_count": EXPECTED,
        "private_integrity_sidecar_written": sidecar_written,
        "sha256_sidecar_written": sha_written,
        "doc_id_manifest_written": manifest_written,
        "file_set_readonly_after_restore": readonly_set,
        "file_writable_before": before["writable"],
        "file_size_before": before["line_count"] and before["size_bytes"],
        "file_size_after": after["size_bytes"],
        "file_writable_after": after["writable"],
        "sha256_prefix": "sha256:" + sha[:16],
        "mutation_hint_untracked_unexpected_scripts": hints.get("untracked_unexpected_scripts_count", "unavailable"),
        "mutation_hint_dirty_report_edits": hints.get("dirty_report_edits_count", "unavailable"),
        "private_outbound_requests_written_outside_repo": True,
        "private_outbound_requests_committed": False,
        "tokenized_payloads_written_to_repo": False,
        "raw_ocr_written_to_repo": False,
        "token_maps_written_to_repo": False,
        "private_identifier_values_written_to_repo": False,
        "public_report_phi_leak_count": 0,
        "future_17c_r2_live_not_started": True,
        "privacy_result": "passed" if integrity_ok else "blocked",
        "safety_result": "passed",
    }
    return summary, notes


def _integrity_matrix(s: dict[str, Any]) -> str:
    keys = ["bad_lines_before", "bad_parseable_before", "bad_malformed_before", "expected_request_count",
            "rebuilt_request_count", "parseable_after", "unique_doc_ids_after", "malformed_after",
            "request_validation_passed_after", "residual_pi_failures_after",
            "private_integrity_sidecar_written", "sha256_sidecar_written", "doc_id_manifest_written",
            "file_set_readonly_after_restore", "sha256_prefix", "privacy_result", "safety_result"]
    return "\n".join(["# 17C-R2-R1 integrity restore matrix", "", "| Field | Value |", "| --- | --- |",
                      *[f"| {k} | `{s[k]}` |" for k in keys], "",
                      "Counts/hashes/status only. The canonical JSONL and sidecars stay private,",
                      "outside the repo, and are never committed.", ""])


def _integrity_public(s: dict[str, Any]) -> dict[str, Any]:
    return {"block": s["block"], "expected_request_count": s["expected_request_count"],
            "rebuilt_request_count": s["rebuilt_request_count"], "parseable_after": s["parseable_after"],
            "unique_doc_ids_after": s["unique_doc_ids_after"], "malformed_after": s["malformed_after"],
            "request_validation_passed_after": s["request_validation_passed_after"],
            "residual_pi_failures_after": s["residual_pi_failures_after"],
            "sha256_prefix": s["sha256_prefix"], "file_set_readonly_after_restore": s["file_set_readonly_after_restore"],
            "note": "Hashes/counts only; the full SHA256 lives in the private sidecar."}


def _mutation_md(s: dict[str, Any]) -> str:
    return "\n".join(["# Mutation risk (report-only)", "",
                      f"- file_writable_before: `{s['file_writable_before']}`",
                      f"- file_writable_after: `{s['file_writable_after']}`",
                      f"- file_set_readonly_after_restore: `{s['file_set_readonly_after_restore']}`",
                      f"- file_size_after_bytes: `{s['file_size_after']}`",
                      f"- untracked_unexpected_scripts: `{s['mutation_hint_untracked_unexpected_scripts']}`",
                      f"- dirty_report_edits: `{s['mutation_hint_dirty_report_edits']}`", "",
                      "An external writer outside this session has been mutating files under",
                      "`MedAI_Private` and editing committed reports. This block does not attribute",
                      "the writer to a specific process, kills no process, and removes no file. The",
                      "rebuilt JSONL is sealed (SHA256 sidecar) and set read-only when supported;",
                      "a future live runner must verify the SHA256 before any provider call.", ""])


def _rerun_gate_md(s: dict[str, Any]) -> str:
    return "\n".join(["# 17C-R2 rerun entry gate (after integrity restore)", "",
                      f"- request_validation_passed_after: `{s['request_validation_passed_after']}`",
                      f"- rebuilt_request_count: `{s['rebuilt_request_count']}` / `{s['expected_request_count']}`",
                      f"- malformed_after: `{s['malformed_after']}` | residual_pi_failures_after: `{s['residual_pi_failures_after']}`", "",
                      "17C-R2 live extraction is NOT re-run here. Before any rerun, verify the canonical",
                      "JSONL SHA256 matches the sealed sidecar and the counts match, confirm the Vertex",
                      "credential preflight PASSES, and keep the run chunked/capped with stop-on-first-failure.", ""])


def _implementation(s: dict[str, Any], notes: list[str]) -> str:
    lines = [f"# {s['block']}", "",
             f"## Result: **{'PASS' if s['privacy_result'] == 'passed' else 'BLOCKED'}** (safety: {s['safety_result']})",
             "", "## Metrics", ""]
    for k in ("bad_lines_before", "bad_parseable_before", "bad_malformed_before", "expected_request_count",
              "rebuilt_request_count", "parseable_after", "unique_doc_ids_after", "malformed_after",
              "request_validation_passed_after", "residual_pi_failures_after", "old_12_added_separately",
              "combined_batch_count", "private_integrity_sidecar_written", "sha256_sidecar_written",
              "doc_id_manifest_written", "file_set_readonly_after_restore", "provider_call_made",
              "gemini_call_made", "claude_call_made", "openai_call_made", "billing_api_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "private_outbound_requests_committed", "tokenized_payloads_written_to_repo",
              "raw_ocr_written_to_repo", "token_maps_written_to_repo",
              "private_identifier_values_written_to_repo", "public_report_phi_leak_count",
              "future_17c_r2_live_not_started", "privacy_result", "safety_result"):
        lines.append(f"- {k}: `{s[k]}`")
    if notes:
        lines += ["", "## Notes", ""] + [f"- `{n}`" for n in notes]
    nxt = ("rerun the Vertex credential preflight; if credentials PASS, rerun 17C-R2 live extraction"
           if s["privacy_result"] == "passed" else "fix integrity recovery before any live run")
    lines += ["", "## Recommended next (no live run started)", "", f"- {nxt}.", ""]
    return "\n".join(lines)


def main() -> int:
    summary, notes = run()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    integrity_md = _integrity_matrix(summary)
    integrity_pub = _integrity_public(summary)
    mutation_md = _mutation_md(summary)
    rerun_md = _rerun_gate_md(summary)
    impl_md = _implementation(summary, notes)

    # Defense-in-depth: no request body / tokenized content leaked into public blobs.
    public_blob = "\n".join([json.dumps(summary), integrity_md, json.dumps(integrity_pub),
                             mutation_md, rerun_md, impl_md])
    leak = 0
    try:
        if CANON_BATCH.is_file():
            for rl in CANON_BATCH.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    c = json.loads(rl).get("tokenized_content", "")
                except ValueError:
                    c = ""
                for ln in {l.strip() for l in c.splitlines() if len(l.strip()) >= 14}:
                    if ln in public_blob:
                        leak += 1
                        break
                if leak:
                    break
    except OSError:
        pass
    summary["public_report_phi_leak_count"] = leak
    if leak:
        summary["safety_result"] = "blocked"

    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "implementation_report.md").write_text(_implementation(summary, notes), encoding="utf-8")
    (REPORT_DIR / "integrity_restore_matrix.md").write_text(integrity_md, encoding="utf-8")
    (REPORT_DIR / "private_batch_integrity_public.json").write_text(json.dumps(integrity_pub, indent=2), encoding="utf-8")
    (REPORT_DIR / "mutation_risk_public.md").write_text(mutation_md, encoding="utf-8")
    (REPORT_DIR / "rerun_17c_r2_entry_gate.md").write_text(rerun_md, encoding="utf-8")

    docs_ok = all((DOC_DIR / d).exists() for d in REQUIRED_DOCS)
    ok = (summary["safety_result"] == "passed" and summary["public_report_phi_leak_count"] == 0
          and summary["provider_call_made"] is False and summary["request_validation_passed_after"] is True
          and summary["future_17c_r2_live_not_started"] is True and docs_ok)
    print("medai_ai_first_corpus_478_batch_integrity_restore_local_only_17c_r2_r1_"
          + ("pass" if ok and summary["privacy_result"] == "passed" else "blocked"))
    print(json.dumps({k: summary[k] for k in (
        "rebuilt_request_count", "parseable_after", "unique_doc_ids_after", "malformed_after",
        "request_validation_passed_after", "residual_pi_failures_after", "sha256_sidecar_written",
        "private_integrity_sidecar_written", "doc_id_manifest_written", "file_set_readonly_after_restore",
        "public_report_phi_leak_count", "provider_call_made", "gemini_call_made",
        "future_17c_r2_live_not_started", "privacy_result", "safety_result")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
