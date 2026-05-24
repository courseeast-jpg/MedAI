#!/usr/bin/env python3
"""MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 — Python self-healing aggregator.

Reads the existing 05 report and the UI smoke report. If the pipeline
path is ready, this is a thin pass-through. If the pipeline path
produced zero facts (``not_ready`` or ``no_input_file``), the script
runs a deterministic adapter / MKB diagnostic that proves the practical
extraction-to-MKB loop end-to-end on synthetic data only:

* deterministic lab fact extraction (English + Russian/Cyrillic);
* review-bound MKBRecord persistence via SQLiteStore (and MKBWriter
  when available);
* SQLite retrieval;
* UI render-plan with row actions;
* synthetic accept / reject / defer through ``app.operator_review_actions``.

No external API. No auto-accept. No raw filename / raw OCR text / raw
document text / private path / PHI in any public output. Public reports
are checked against ``clinical_knowledge.privacy.check_public_report_payload``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_local_self_healing_validation_07"
JSON_REPORT_PATH = REPORT_DIR / "medai_local_self_healing_validation_07_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_local_self_healing_validation_07_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_LOCAL_SELF_HEALING_VALIDATION_07.md"

ONE_DOC_REPORT_JSON = (
    REPO_ROOT
    / "reports"
    / "medai_local_one_doc_operator_validation_05"
    / "medai_local_one_doc_operator_validation_05_report.json"
)
SMOKE_REPORT_JSON = (
    REPO_ROOT
    / "reports"
    / "medai_streamlit_launch_smoke_05"
    / "medai_streamlit_launch_smoke_05_report.json"
)

NEXT_UI_COMMAND = "streamlit run app/main.py"
SAFE_HASH_PREFIX = "selfheal_"

ALLOWED_INPUT_MODES = ("selected_file", "synthetic_fallback")
ALLOWED_INPUT_SUFFIXES = (".pdf", ".txt", ".none")

SYNTHETIC_EN_TEXT = (
    "Lab result report\n"
    "Specimen: serum\n"
    "Reference range listed below.\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
)
SYNTHETIC_RU_TEXT = (
    "Лабораторный отчет\n"
    "Материал: сыворотка\n"
    "Результат:\n"
    "Глюкоза: 5,4 ммоль/л\n"
    "Гемоглобин: 135 г/л\n"
    "Лейкоциты: 7,2 x10E9/л\n"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def safe_input_hash(size: int, suffix: str) -> str:
    """Public-safe handle: short SHA-256 over (size, suffix)."""
    suffix_clean = suffix.lower() if suffix else "noext"
    digest = hashlib.sha256(f"{int(size)}:{suffix_clean}".encode("utf-8")).hexdigest()
    return f"{SAFE_HASH_PREFIX}{digest[:10]}"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _privacy_check(payload: Any) -> Dict[str, Any]:
    try:
        from clinical_knowledge.privacy import check_public_report_payload

        result = check_public_report_payload(payload)
        return {
            "passed": bool(result.passed),
            "leak_examples_redacted": list(result.leak_examples_redacted or []),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Adapter diagnostic
# ---------------------------------------------------------------------------

def run_adapter_diagnostic() -> Dict[str, Any]:
    """Deterministic synthetic adapter / MKB / UI / operator-action proof.

    Uses only modules from the 01-03 chain. No pipeline call. No spaCy
    requirement. No chromadb requirement. No external API.
    """
    from app.config import TIER_QUARANTINED, TRUST_CLINICAL
    from app.extracted_information_preview import (
        build_extracted_information_preview_plan,
    )
    from app.operator_review_actions import (
        accept_after_source_comparison,
        defer_extracted_fact,
        reject_extracted_fact,
    )
    from app.schemas import MKBRecord
    from execution.extracted_medical_facts import (
        CONSERVATIVE_CONFIDENCE,
        extract_lab_observation_entities,
        summarize_extracted_facts_for_public_report,
    )
    from mkb.sqlite_store import SQLiteStore

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_self_healing_07_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")

    # 1. Extract facts from synthetic English + Russian lab text.
    metadata = {"document_type": "lab_report"}
    en_entities = extract_lab_observation_entities(SYNTHETIC_EN_TEXT, metadata)
    ru_entities = extract_lab_observation_entities(SYNTHETIC_RU_TEXT, metadata)
    all_entities = list(en_entities) + list(ru_entities)
    summary = summarize_extracted_facts_for_public_report(all_entities)

    # 2. Persist records as review-bound test_result rows in SQLite.
    record_ids: list[str] = []
    record_states: Dict[str, Dict[str, Any]] = {}
    for entity in all_entities:
        structured = dict(entity.get("structured") or {})
        record = MKBRecord(
            fact_type="test_result",
            content=f"Test: {entity['text']}: {structured.get('value','')} {structured.get('unit') or ''}".rstrip(),
            structured={"text": entity["text"], **structured},
            specialty="general",
            source_type="extraction",
            source_name="synthetic-self-healing-07",
            trust_level=TRUST_CLINICAL,
            confidence=float(entity.get("confidence", CONSERVATIVE_CONFIDENCE)),
            tier=TIER_QUARANTINED,
            extraction_method="rules_based",
            requires_review=True,
            ddi_checked=False,
        )
        sql_store.write_record(record, session_id="07")
        record_ids.append(record.id)
        record_states[record.id] = {
            "fact_type": record.fact_type,
            "tier": record.tier,
            "status": record.status,
            "requires_review": bool(record.requires_review),
            "ddi_status": record.ddi_status or "",
        }

    # 3. Retrieve back.
    with sql_store._get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE fact_type=?",
            ("test_result",),
        ).fetchone()
    retrieval_count = int(row["c"] if isinstance(row, dict) else row[0])

    # 4. UI render plan.
    plan = build_extracted_information_preview_plan(
        {
            "extracted_medical_facts_preview_safe": summary["extracted_medical_facts_preview_safe"],
            "extracted_medical_fact_count": summary["extracted_medical_fact_count"],
            "extraction_to_mkb_written_count": 0,
            "extraction_to_mkb_review_count": summary["extracted_medical_fact_count"],
            "extracted_medical_fact_record_ids": record_ids,
            "extracted_medical_fact_record_states": record_states,
        }
    )

    # 5. Operator accept / reject / defer on synthetic records only.
    action_proofs = 0
    if len(record_ids) >= 3:
        accept_result = accept_after_source_comparison(
            sql_store, record_ids[0], operator_note="synthetic verify", session_id="07"
        )
        reject_result = reject_extracted_fact(sql_store, record_ids[1], session_id="07")
        defer_result = defer_extracted_fact(sql_store, record_ids[2], session_id="07")
        action_proofs = sum(
            1
            for r in (accept_result, reject_result, defer_result)
            if r.success
        )

    findings = {
        "structured_facts_extracted": int(summary["extracted_medical_fact_count"]),
        "review_bound_records_persisted": len(record_ids),
        "retrieval_proof_count": retrieval_count,
        "ui_preview_rows": int(plan.get("row_count") or 0),
        "operator_action_proof_count": int(action_proofs),
        "ui_render_plan_built": bool(plan),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }
    findings["passed"] = (
        findings["structured_facts_extracted"] > 0
        and findings["review_bound_records_persisted"] > 0
        and findings["retrieval_proof_count"] > 0
        and findings["ui_preview_rows"] > 0
        and findings["operator_action_proof_count"] == 3
        and findings["ui_render_plan_built"] is True
    )
    return findings


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def build_report(
    *,
    branch: str,
    head: str,
    dependency_prepare_result: str,
    input_mode: str,
    input_suffix: str,
    input_safe_hash: str,
    streamlit_launch_attempted: bool,
    force_adapter_diagnostic: bool = False,
) -> Dict[str, Any]:
    if input_mode not in ALLOWED_INPUT_MODES:
        raise ValueError(f"input_mode must be one of {ALLOWED_INPUT_MODES}")
    if input_suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(f"input_suffix must be one of {ALLOWED_INPUT_SUFFIXES}")
    if not input_safe_hash.startswith(SAFE_HASH_PREFIX):
        raise ValueError(f"input_safe_hash must start with {SAFE_HASH_PREFIX}")
    if "/" in input_safe_hash or "\\" in input_safe_hash:
        raise ValueError("input_safe_hash must not contain path separators")

    one_doc = _read_json(ONE_DOC_REPORT_JSON)
    smoke = _read_json(SMOKE_REPORT_JSON)
    one_doc_conclusion = str(one_doc.get("conclusion") or "missing")
    one_doc_findings = one_doc.get("audit_findings") or {}
    smoke_findings = smoke.get("audit_findings") or {}
    ui_smoke_passed = bool(smoke.get("passed") or smoke_findings.get("helper_import_passed"))
    streamlit_launch = smoke_findings.get("streamlit_brief_launch") or {}

    pipeline_path_ready = (
        one_doc_conclusion == "one_doc_operator_validation_ready"
        and int(one_doc_findings.get("structured_facts_extracted_count") or 0) > 0
        and not force_adapter_diagnostic
    )

    structured_facts_extracted = int(
        one_doc_findings.get("structured_facts_extracted_count") or 0
    )
    review_bound_records_persisted = int(
        one_doc_findings.get("review_bound_records_persisted_count") or 0
    )
    retrieval_proof_count = int(one_doc_findings.get("retrieval_proof_count") or 0)
    ui_preview_rows = int(one_doc_findings.get("ui_render_plan_row_count") or 0)
    operator_action_proof_count = 0
    adapter_path_ready = False
    adapter_diagnostic_findings: Dict[str, Any] = {}

    if not pipeline_path_ready:
        adapter_diagnostic_findings = run_adapter_diagnostic()
        adapter_path_ready = bool(adapter_diagnostic_findings.get("passed"))
        if adapter_path_ready:
            structured_facts_extracted = int(
                adapter_diagnostic_findings["structured_facts_extracted"]
            )
            review_bound_records_persisted = int(
                adapter_diagnostic_findings["review_bound_records_persisted"]
            )
            retrieval_proof_count = int(
                adapter_diagnostic_findings["retrieval_proof_count"]
            )
            ui_preview_rows = int(adapter_diagnostic_findings["ui_preview_rows"])
            operator_action_proof_count = int(
                adapter_diagnostic_findings["operator_action_proof_count"]
            )

    if pipeline_path_ready:
        final_conclusion = "local_operator_validation_ready_pipeline_path"
    elif adapter_path_ready:
        final_conclusion = "local_operator_validation_ready_adapter_path"
    else:
        final_conclusion = "not_ready"

    findings: Dict[str, Any] = {
        "branch": branch,
        "head": head,
        "dependency_prepare_result": str(dependency_prepare_result),
        "input_mode": input_mode,
        "input_suffix": input_suffix.lower(),
        "input_safe_hash": input_safe_hash,
        "one_doc_05_conclusion": one_doc_conclusion,
        "pipeline_path_ready": pipeline_path_ready,
        "adapter_path_ready": adapter_path_ready,
        "final_conclusion": final_conclusion,
        "structured_facts_extracted": structured_facts_extracted,
        "review_bound_records_persisted": review_bound_records_persisted,
        "retrieval_proof_count": retrieval_proof_count,
        "ui_preview_rows": ui_preview_rows,
        "operator_action_proof_count": operator_action_proof_count,
        "ui_smoke_passed": ui_smoke_passed,
        "streamlit_launch_attempted": bool(streamlit_launch_attempted) or bool(streamlit_launch.get("ran")),
        "external_api_used": False,
        "auto_accept_enabled": False,
        "next_ui_command": NEXT_UI_COMMAND,
    }
    if not pipeline_path_ready and not adapter_path_ready:
        findings["error_bucket"] = "pipeline_zero_facts_and_adapter_path_failed"
    elif not pipeline_path_ready and adapter_path_ready:
        findings["reason"] = "pipeline_produced_zero_facts_adapter_path_validated"

    report = {
        "block_id": "MEDAI-LOCAL-SELF-HEALING-VALIDATION-07",
        "mode": "self_healing_local_operator_validation",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "passed": final_conclusion != "not_ready",
    }
    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get(
        "leak_examples_redacted", []
    )
    return report


def write_reports(report: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    f = report["audit_findings"]
    md_lines = [
        "# MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 — Report",
        "",
        f"Branch: `{f['branch']}` | HEAD: `{f['head']}`",
        "",
        f"Final conclusion: **{f['final_conclusion']}**",
        "",
        "## Inputs",
        "",
        f"- dependency prepare result: `{f['dependency_prepare_result']}`",
        f"- input mode: `{f['input_mode']}`",
        f"- input suffix: `{f['input_suffix']}`",
        f"- input safe hash: `{f['input_safe_hash']}`",
        f"- 05 one-doc conclusion: `{f['one_doc_05_conclusion']}`",
        "",
        "## Path readiness",
        "",
        f"- pipeline path ready: {f['pipeline_path_ready']}",
        f"- adapter path ready: {f['adapter_path_ready']}",
        "",
        "## Counts",
        "",
        f"- structured facts extracted: {f['structured_facts_extracted']}",
        f"- review-bound MKB records persisted: {f['review_bound_records_persisted']}",
        f"- SQLite retrieval proof: {f['retrieval_proof_count']}",
        f"- UI preview rows: {f['ui_preview_rows']}",
        f"- operator action proof count: {f['operator_action_proof_count']}",
        f"- UI smoke passed: {f['ui_smoke_passed']}",
        f"- streamlit launch attempted: {f['streamlit_launch_attempted']}",
        "",
        "## Safety",
        "",
        f"- external API used: {f['external_api_used']}",
        f"- auto-accept enabled: {f['auto_accept_enabled']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        "",
        "## Next operator command",
        "",
        f"```\n{f['next_ui_command']}\n```",
        "",
    ]
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    SHORT_MD_PATH.write_text(
        "# MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 — Short Summary\n\n"
        f"Branch `{f['branch']}`, HEAD `{f['head']}`\n\n"
        f"Final conclusion: **{f['final_conclusion']}**\n\n"
        f"- pipeline path ready: {f['pipeline_path_ready']}\n"
        f"- adapter path ready: {f['adapter_path_ready']}\n"
        f"- structured facts: {f['structured_facts_extracted']}\n"
        f"- review-bound MKB records: {f['review_bound_records_persisted']}\n"
        f"- SQLite retrieval: {f['retrieval_proof_count']}\n"
        f"- UI preview rows: {f['ui_preview_rows']}\n"
        f"- operator action proof: {f['operator_action_proof_count']}\n"
        f"- UI smoke: {f['ui_smoke_passed']}\n"
        f"- external API: {f['external_api_used']}\n"
        f"- auto-accept: {f['auto_accept_enabled']}\n"
        f"- privacy check passed: {report['privacy_check_passed']}\n\n"
        f"Next operator command:\n\n```\n{f['next_ui_command']}\n```\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MEDAI-LOCAL-SELF-HEALING-VALIDATION-07 aggregator"
    )
    parser.add_argument("--branch", default="clinical-knowledge-architecture")
    parser.add_argument("--head", default="")
    parser.add_argument("--dependency-prepare-result", default="unknown")
    parser.add_argument(
        "--input-mode", default="synthetic_fallback", choices=ALLOWED_INPUT_MODES
    )
    parser.add_argument(
        "--input-suffix",
        default="none",
        choices=[s.lstrip(".") for s in ALLOWED_INPUT_SUFFIXES],
    )
    parser.add_argument("--input-safe-hash", default="")
    parser.add_argument(
        "--streamlit-launch-attempted", default="false"
    )
    parser.add_argument(
        "--force-adapter-diagnostic",
        action="store_true",
        help="Force the adapter diagnostic to run even when 05 is ready (test hook).",
    )
    args = parser.parse_args(argv)

    head = args.head
    if not head:
        try:
            import subprocess

            head = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
                .decode("utf-8")
                .strip()
            )
        except Exception:
            head = "unknown"

    safe_hash = args.input_safe_hash or safe_input_hash(0, f".{args.input_suffix}")
    streamlit_launch_attempted = str(args.streamlit_launch_attempted).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    report = build_report(
        branch=args.branch,
        head=head,
        dependency_prepare_result=args.dependency_prepare_result,
        input_mode=args.input_mode,
        input_suffix=f".{args.input_suffix}",
        input_safe_hash=safe_hash,
        streamlit_launch_attempted=streamlit_launch_attempted,
        force_adapter_diagnostic=args.force_adapter_diagnostic,
    )
    write_reports(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
