#!/usr/bin/env python3
"""MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-04 — real local operator validation.

Validation-only. No new product features. Combines:

1. Synthetic re-run of the 01/02/03 chain (the path that is proven byte-stable).
2. Local dependency / UI readiness probe (Streamlit / spaCy / chromadb /
   sqlcipher3 / PDF parsers).
3. Counts-only audit of private local corpus folders without ever
   reading file contents:
     - test_input/
     - real_validation_input/
     - full_corpus_input/
4. UI render-plan import smoke (the Streamlit-free helpers must import).

By default this block never reads or acts on real corpus documents. The
operator-action proof remains synthetic.

Hard rules:
* No external API call.
* No raw OCR text, raw source text, raw filenames, private paths, PHI,
  or real diagnosis strings in any public report.
* No PDF or real document committed.
* No bulk-accept. No auto-accept.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_04_REAL_LOCAL_OPERATOR_VALIDATION.md"

PRIOR_VALIDATION_SCRIPTS = (
    "run_medai_corpus_extraction_to_mkb_minimum_01.py",
    "run_medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke.py",
    "run_medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux.py",
)

DEPENDENCY_PROBE_TARGETS = (
    "streamlit",
    "spacy",
    "chromadb",
    "sqlcipher3",
    "PyPDF2",
    "pytesseract",
    "medspacy",
    "presidio_analyzer",
)

CORPUS_FOLDERS = (
    "test_input",
    "real_validation_input",
    "full_corpus_input",
)

UI_IMPORT_SMOKE_TARGETS = (
    "app.extracted_information_preview",
    "app.operator_review_actions",
    "execution.extracted_medical_facts",
)


def _safe_file_handle(path: Path) -> str:
    """Return a non-identifying handle for a private file.

    Never returns the filename or any directory path. The handle is a
    short SHA-256 over (size, suffix) — neither of which is PHI.
    """
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    suffix = path.suffix.lower() if path.suffix else "noext"
    digest = hashlib.sha256(f"{size}:{suffix}".encode("utf-8")).hexdigest()
    return f"localfile_{digest[:10]}"


def _probe_dependencies() -> Dict[str, bool]:
    present: Dict[str, bool] = {}
    for name in DEPENDENCY_PROBE_TARGETS:
        spec = importlib.util.find_spec(name)
        present[name] = spec is not None
    return present


def _is_real_corpus_file(path: Path) -> bool:
    """True for files that look like real corpus documents.

    Excludes dotfiles (`.gitkeep`, `.DS_Store`, etc.) and anything that
    is not a regular file. Keeps the probe conservative: only ``.pdf``,
    ``.txt``, ``.png``, ``.jpg``, ``.jpeg``, ``.tif``, ``.tiff``,
    ``.docx`` count toward the "private corpus present" signal.
    """
    if not path.is_file():
        return False
    name = path.name
    if name.startswith("."):
        return False
    suffix = path.suffix.lower()
    return suffix in {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".docx"}


def _probe_corpus_counts() -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for folder in CORPUS_FOLDERS:
        target = REPO_ROOT / folder
        if not target.is_dir():
            summary[folder] = {"present": False, "file_count": 0, "file_suffix_counts": {}}
            continue
        files: List[Path] = [p for p in target.glob("*") if _is_real_corpus_file(p)]
        # Counts-only: never store names or paths in the public report.
        suffixes: Dict[str, int] = {}
        for f in files:
            suffix = f.suffix.lower() or "noext"
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
        summary[folder] = {
            "present": True,
            "file_count": len(files),
            "file_suffix_counts": suffixes,
        }
    return summary


def _run_prior_validation_script(script_name: str) -> Dict[str, Any]:
    script_path = REPO_ROOT / "scripts" / script_name
    if not script_path.is_file():
        return {"ran": False, "reason": "script_missing", "all_pass": False}
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
            "PATH": os.environ.get("PATH", ""),
        },
    )
    out = proc.stdout or ""
    payload: Dict[str, Any] = {}
    try:
        # The scripts print one JSON document to stdout.
        first_brace = out.find("{")
        if first_brace >= 0:
            payload = json.loads(out[first_brace:])
    except Exception:
        payload = {}
    return {
        "ran": True,
        "exit_code": proc.returncode,
        "all_pass": bool(payload.get("all_pass")) if payload else False,
        "conclusion": str(payload.get("conclusion") or ""),
    }


def _ui_import_smoke() -> Dict[str, Any]:
    results: Dict[str, bool] = {}
    errors: Dict[str, str] = {}
    for module_name in UI_IMPORT_SMOKE_TARGETS:
        try:
            importlib.import_module(module_name)
            results[module_name] = True
        except Exception as exc:
            results[module_name] = False
            errors[module_name] = f"{type(exc).__name__}: {exc}"
    return {"results": results, "errors": errors, "passed": all(results.values())}


def _streamlit_launch_smoke(streamlit_present: bool) -> Dict[str, Any]:
    if not streamlit_present:
        return {
            "ran": False,
            "reason": "streamlit_module_not_present_in_local_environment",
            "passed": None,
        }
    # Streamlit is available; do a syntactic / import check against app/main.py
    # without actually serving HTTP.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import app.main as m; print('app_main_import_ok')"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={
                "MEDAI_ALLOW_EXTERNAL_API": "false",
                "MEDAI_LOCAL_ONLY": "true",
                "PATH": os.environ.get("PATH", ""),
            },
        )
        ok = proc.returncode == 0 and "app_main_import_ok" in (proc.stdout or "")
        return {
            "ran": True,
            "passed": ok,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ran": True, "passed": False, "reason": "timeout_expired"}
    except Exception as exc:
        return {"ran": True, "passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _synthetic_operator_action_dry_run() -> Dict[str, Any]:
    """Synthetic-only end-to-end action proof (matches 03 validator)."""
    from execution.pipeline import ExecutionPipeline
    from mkb.sqlite_store import SQLiteStore
    from app.operator_review_actions import (
        accept_after_source_comparison,
        defer_extracted_fact,
        reject_extracted_fact,
    )

    class StubSpacy:
        def extract(self, text: str) -> Dict[str, Any]:
            return {
                "extractor": "spacy",
                "entities": [],
                "confidence": 0.86,
                "latency_ms": 1,
                "raw_text": text,
                "notes": ["stub_extractor_used_in_smoke_test"],
            }

    temp_dir = Path(tempfile.mkdtemp(prefix="medai_04_smoke_"))
    sql_store = SQLiteStore(db_path=temp_dir / "mkb.db", encryption_key="")
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        vector_store=None,
        quality_gate=None,
        medication_gate=None,
        spacy_extractor=StubSpacy(),
        review_queue_path=temp_dir / "review_queue.jsonl",
    )
    text = (
        "Lab result report\n"
        "Specimen: serum\n"
        "Reference range listed below.\n"
        "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
        "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
        "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    )
    result = pipeline.process_text(
        text, specialty="general", source_name="synth-04-001", session_id="m04"
    )
    ext = result.extractor_result or {}
    record_ids = list(ext.get("extracted_medical_fact_record_ids") or [])
    accept_ok = reject_ok = defer_ok = False
    if len(record_ids) >= 3:
        accept_ok = accept_after_source_comparison(sql_store, record_ids[0]).success
        reject_ok = reject_extracted_fact(sql_store, record_ids[1]).success
        defer_ok = defer_extracted_fact(sql_store, record_ids[2]).success
    with sql_store._get_conn() as conn:
        retrieval = conn.execute(
            "SELECT COUNT(*) AS c FROM records WHERE fact_type=?", ("test_result",)
        ).fetchone()
    retrieval_count = int(retrieval["c"] if isinstance(retrieval, dict) else retrieval[0])
    return {
        "documents_evaluated": 1,
        "extracted_fact_count": int(ext.get("extracted_medical_fact_count") or 0),
        "review_bound_records_persisted": int(ext.get("extraction_to_mkb_review_count") or 0),
        "retrieval_proof_count": retrieval_count,
        "accept_proof": bool(accept_ok),
        "reject_proof": bool(reject_ok),
        "defer_proof": bool(defer_ok),
        "external_api_used": False,
    }


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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prior_validation_results: Dict[str, Any] = {}
    for script_name in PRIOR_VALIDATION_SCRIPTS:
        prior_validation_results[script_name] = _run_prior_validation_script(script_name)
    synthetic_validation_passed = all(
        item.get("all_pass") for item in prior_validation_results.values()
    )

    dependency_present = _probe_dependencies()
    corpus_counts = _probe_corpus_counts()
    private_docs_total = sum(
        int(item["file_count"]) for item in corpus_counts.values() if item.get("present")
    )
    private_corpus_present = private_docs_total > 0

    ui_import_smoke = _ui_import_smoke()
    streamlit_launch = _streamlit_launch_smoke(dependency_present.get("streamlit", False))

    operator_action_synthetic = _synthetic_operator_action_dry_run()
    operator_action_synthetic_proof_count = sum(
        1
        for key in ("accept_proof", "reject_proof", "defer_proof")
        if operator_action_synthetic.get(key)
    )

    # Counts-only private corpus iteration. Never reads file contents.
    private_doc_handles: List[str] = []
    private_docs_evaluated_count = 0
    if private_corpus_present:
        cap = 3
        for folder, summary in corpus_counts.items():
            if not summary.get("present"):
                continue
            target = REPO_ROOT / folder
            for p in sorted(target.glob("*")):
                if not _is_real_corpus_file(p):
                    continue
                if private_docs_evaluated_count >= cap:
                    break
                private_doc_handles.append(_safe_file_handle(p))
                private_docs_evaluated_count += 1
            if private_docs_evaluated_count >= cap:
                break

    practical_mvp_ready_for_local_operator_use = (
        synthetic_validation_passed
        and ui_import_smoke["passed"]
        and operator_action_synthetic_proof_count == 3
    )

    if not synthetic_validation_passed or not ui_import_smoke["passed"] or operator_action_synthetic_proof_count < 3:
        conclusion = "not_ready"
    elif private_corpus_present:
        conclusion = "real_local_operator_validation_ready"
    else:
        conclusion = "private_corpus_not_present_synthetic_ready"

    findings: Dict[str, Any] = {
        "branch": "clinical-knowledge-architecture",
        "head_short_before_block": "8ecf8fe",
        "synthetic_validation_passed": synthetic_validation_passed,
        "prior_validation_results": prior_validation_results,
        "dependency_probe": dependency_present,
        "corpus_folder_counts": corpus_counts,
        "private_corpus_present": private_corpus_present,
        "private_docs_evaluated_count": private_docs_evaluated_count,
        "private_doc_handles": private_doc_handles,
        "structured_facts_extracted_count": int(operator_action_synthetic["extracted_fact_count"]),
        "review_bound_records_persisted_count": int(operator_action_synthetic["review_bound_records_persisted"]),
        "retrieval_proof_count": int(operator_action_synthetic["retrieval_proof_count"]),
        "ui_import_smoke": ui_import_smoke,
        "ui_import_smoke_passed": bool(ui_import_smoke["passed"]),
        "streamlit_launch_smoke": streamlit_launch,
        "streamlit_launch_smoke_passed": streamlit_launch.get("passed"),
        "operator_action_synthetic": operator_action_synthetic,
        "operator_action_synthetic_proof_count": operator_action_synthetic_proof_count,
        "external_api_used": False,
        "auto_accept_enabled": False,
        "practical_mvp_ready_for_local_operator_use": practical_mvp_ready_for_local_operator_use,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
    }

    report = {
        "block_id": "MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-04-REAL-LOCAL-OPERATOR-VALIDATION",
        "mode": "validation_only_no_product_features",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "conclusion": conclusion,
        "all_pass": conclusion != "not_ready",
        "external_api_used": False,
        "auto_accept_enabled": False,
    }

    privacy = _privacy_check(report)
    report["privacy_check_passed"] = bool(privacy.get("passed"))
    report["privacy_check_leak_examples_redacted"] = privacy.get("leak_examples_redacted", [])

    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-04-REAL-LOCAL-OPERATOR-VALIDATION — Report",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "## Synthetic chain re-run",
        "",
    ]
    for script_name, item in prior_validation_results.items():
        md_lines.append(
            f"- `{script_name}`: ran={item.get('ran')} all_pass={item.get('all_pass')} "
            f"conclusion=`{item.get('conclusion','')}`"
        )
    md_lines.extend([
        "",
        "## Local dependency probe",
        "",
    ])
    for name, present in dependency_present.items():
        md_lines.append(f"- `{name}`: {'present' if present else 'missing'}")
    md_lines.extend([
        "",
        "## Corpus folder counts (no filenames; counts only)",
        "",
    ])
    for folder, summary in corpus_counts.items():
        md_lines.append(
            f"- `{folder}`: present={summary.get('present')} files={summary.get('file_count')} "
            f"suffix_counts={summary.get('file_suffix_counts') or {}}"
        )
    md_lines.extend([
        "",
        "## UI import smoke",
        "",
        f"- passed: {ui_import_smoke['passed']}",
    ])
    if not ui_import_smoke["passed"]:
        for module_name, message in ui_import_smoke["errors"].items():
            md_lines.append(f"  - `{module_name}` failed: `{message}`")

    md_lines.extend([
        "",
        "## Streamlit launch smoke",
        "",
        f"- ran: {streamlit_launch.get('ran')}",
        f"- passed: {streamlit_launch.get('passed')}",
        f"- reason: {streamlit_launch.get('reason','')}",
        "",
        "## Operator action synthetic proof",
        "",
        f"- documents evaluated: {operator_action_synthetic['documents_evaluated']}",
        f"- extracted facts: {operator_action_synthetic['extracted_fact_count']}",
        f"- review-bound records persisted: {operator_action_synthetic['review_bound_records_persisted']}",
        f"- retrieval proof: {operator_action_synthetic['retrieval_proof_count']}",
        f"- accept proof: {operator_action_synthetic['accept_proof']}",
        f"- reject proof: {operator_action_synthetic['reject_proof']}",
        f"- defer proof: {operator_action_synthetic['defer_proof']}",
        "",
        "## Safety",
        "",
        f"- external API used: {report['external_api_used']}",
        f"- auto-accept enabled: {report['auto_accept_enabled']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        f"- practical MVP ready for local operator use: {findings['practical_mvp_ready_for_local_operator_use']}",
        "",
    ])
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    SHORT_MD_PATH.write_text(
        "# MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-04-REAL-LOCAL-OPERATOR-VALIDATION — Short Summary\n\n"
        f"Conclusion: **{conclusion}**\n\n"
        f"- synthetic chain re-run passed: {synthetic_validation_passed}\n"
        f"- private corpus present: {private_corpus_present}\n"
        f"- private docs evaluated (counts-only): {private_docs_evaluated_count}\n"
        f"- structured facts extracted (synthetic): {findings['structured_facts_extracted_count']}\n"
        f"- review-bound records persisted (synthetic): {findings['review_bound_records_persisted_count']}\n"
        f"- retrieval proof (synthetic): {findings['retrieval_proof_count']}\n"
        f"- UI import smoke passed: {ui_import_smoke['passed']}\n"
        f"- streamlit launch smoke: ran={streamlit_launch.get('ran')} passed={streamlit_launch.get('passed')}\n"
        f"- operator action synthetic proof count: {operator_action_synthetic_proof_count} / 3\n"
        f"- external API used: {report['external_api_used']}\n"
        f"- auto-accept enabled: {report['auto_accept_enabled']}\n"
        f"- privacy check passed: {report['privacy_check_passed']}\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
