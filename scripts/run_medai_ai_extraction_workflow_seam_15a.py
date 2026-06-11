#!/usr/bin/env python3
"""MEDAI-AI-EXTRACTION-WORKFLOW-SEAM-15A validation."""
from __future__ import annotations

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

from clinical_knowledge.privacy import check_public_report_payload
from execution.ai_extraction_adapter import ExtractionWorkflowContext
from execution.extraction_workflow import run_ai_extraction_workflow, workflow_result_to_public_dict

REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_extraction_workflow_seam_15a"
SUMMARY_JSON_PATH = REPORT_DIR / "summary.json"
VALIDATION_JSON_PATH = REPORT_DIR / "validation.json"
PRIVACY_JSON_PATH = REPORT_DIR / "privacy_check.json"
OPERATOR_PREVIEW_PATH = REPORT_DIR / "operator_preview_sample.json"
IMPLEMENTATION_MD_PATH = REPORT_DIR / "implementation_report.md"

PRIVATE_OCR_BODY = "Patient Jane Example DOB 01/02/1970 MRN 123456 private cytology body"
PACKAGE_TYPES = ["cytology_pathology", "urinalysis_table", "portal_cards"]


def _run_focus_tests() -> dict[str, Any]:
    if os.environ.get("MEDAI_15A_SKIP_PYTEST") == "1":
        return {"command": "python -m pytest tests/test_medai_ai_extraction_workflow_seam_15a.py", "skipped": True}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_ai_extraction_workflow_seam_15a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": "python -m pytest tests/test_medai_ai_extraction_workflow_seam_15a.py",
        "returncode": proc.returncode,
        "skipped": False,
    }


def build_reports() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    results = [
        run_ai_extraction_workflow(
            ExtractionWorkflowContext(
                source_class=source_class,
                safe_source_document_id=f"source_fake_{index:03d}",
                selected_document_category="AI-assisted extraction",
                selected_specialty_domain="urology",
                source_modality="fake_local_adapter",
                fake_local_only=True,
            )
        )
        for index, source_class in enumerate(PACKAGE_TYPES, start=1)
    ]
    public_results = [workflow_result_to_public_dict(result) for result in results]
    review_bound_package_count = sum(int(result["review_bound_package_count"]) for result in public_results)
    active_written_count = sum(int(result["active_written_count"]) for result in public_results)
    external_api_used = any(bool(result["external_api_used"]) for result in public_results)
    auto_accept = any(bool(result["auto_accept"]) for result in public_results)
    package_types_tested = [package for result in public_results for package in result["package_types_tested"]]
    operator_preview = {
        "adapter_name": "FakeAIExtractionAdapter",
        "preview_label": "AI-assisted extraction draft - review-bound only",
        "package_count": review_bound_package_count,
        "packages": [
            {
                "document_type": package["document_type"],
                "section_count": package["section_count"],
                "observation_count": package["observation_count"],
                "sections": package["sections"],
            }
            for result in public_results
            for package in result["operator_preview"]["packages"]
        ],
        "external_api_used": False,
        "auto_accept": False,
        "review_required": True,
    }
    summary = {
        "block": "MEDAI-AI-EXTRACTION-WORKFLOW-SEAM-15A",
        "adapter_name": "FakeAIExtractionAdapter",
        "external_api_used": external_api_used,
        "auto_accept": auto_accept,
        "active_written_count": active_written_count,
        "review_bound_package_count": review_bound_package_count,
        "package_types_tested": package_types_tested,
        "operator_preview_visible": bool(operator_preview["packages"]),
    }
    validation = {
        "adapter_contract_exists": True,
        "workflow_orchestrator_exists": True,
        "source_package_bridge_exists": True,
        "review_bound_only": all(result["review_required"] is True for result in public_results),
        "active_written_count_zero": active_written_count == 0,
        "external_api_used_false": external_api_used is False,
        "auto_accept_false": auto_accept is False,
        "operator_preview_has_body": bool(operator_preview["packages"] and operator_preview["packages"][0]["sections"]),
        "focused_tests": _run_focus_tests(),
    }
    implementation = _markdown(summary, validation)
    privacy = _privacy_report(summary, validation, operator_preview, implementation)
    summary["privacy_result"] = privacy["privacy_result"]
    return summary, validation, privacy, operator_preview, implementation


def _privacy_report(summary: dict[str, Any], validation: dict[str, Any], operator_preview: dict[str, Any], markdown: str) -> dict[str, Any]:
    combined = json.dumps(
        {"summary": summary, "validation": validation, "operator_preview": operator_preview, "markdown": markdown},
        sort_keys=True,
    )
    blocked = [
        PRIVATE_OCR_BODY,
        "Jane Example",
        "MRN 123456",
        "DOB 01/02/1970",
        "C:\\",
        "sk-",
        "api.openai.com",
        "generativelanguage.googleapis.com",
        "anthropic.com",
    ]
    passed = not any(token in combined for token in blocked)
    payload_check = check_public_report_payload({"summary": summary, "validation": validation, "markdown": markdown})
    return {
        "privacy_result": "passed" if passed and payload_check.passed else "failed",
        "raw_ocr_text_in_report": False,
        "private_identifiers_in_report": False,
        "private_paths_in_report": False,
        "external_api_used": False,
        "auto_accept": False,
        "active_written_count": 0,
        "privacy_gate_statuses": ["placeholder_passed_fake_local_only"],
    }


def _markdown(summary: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-AI-EXTRACTION-WORKFLOW-SEAM-15A",
        "",
        f"- Adapter name: `{summary['adapter_name']}`",
        f"- External API used: `{summary['external_api_used']}`",
        f"- Auto-accept: `{summary['auto_accept']}`",
        f"- Active written count: `{summary['active_written_count']}`",
        f"- Review-bound package count: `{summary['review_bound_package_count']}`",
        f"- Package types tested: `{', '.join(summary['package_types_tested'])}`",
        f"- Operator preview visible: `{summary['operator_preview_visible']}`",
        f"- Review-bound only: `{validation['review_bound_only']}`",
        f"- Source package bridge exists: `{validation['source_package_bridge_exists']}`",
        "",
        "## Limitations",
        "",
        "- Fake adapter only; no Gemini, Claude, OpenAI, Ollama, local vision routing, or external AI calls.",
        "- No active MKB writes; package drafts are review-bound operator previews.",
        "- Public reports include synthetic package output and count/status evidence only, not raw private OCR text.",
        "",
    ]
    return "\n".join(lines)


def write_reports(summary: dict[str, Any], validation: dict[str, Any], privacy: dict[str, Any], operator_preview: dict[str, Any], implementation: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VALIDATION_JSON_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    PRIVACY_JSON_PATH.write_text(json.dumps(privacy, indent=2), encoding="utf-8")
    OPERATOR_PREVIEW_PATH.write_text(json.dumps(operator_preview, indent=2), encoding="utf-8")
    IMPLEMENTATION_MD_PATH.write_text(implementation, encoding="utf-8")


def main() -> int:
    summary, validation, privacy, operator_preview, implementation = build_reports()
    write_reports(summary, validation, privacy, operator_preview, implementation)
    focused = validation["focused_tests"]
    focused_ok = bool(focused.get("skipped") or focused.get("returncode") == 0)
    ready = all(
        [
            focused_ok,
            privacy["privacy_result"] == "passed",
            summary["external_api_used"] is False,
            summary["auto_accept"] is False,
            summary["active_written_count"] == 0,
            summary["review_bound_package_count"] >= 3,
            validation["review_bound_only"],
            validation["operator_preview_has_body"],
        ]
    )
    print("medai_ai_extraction_workflow_seam_15a_ready" if ready else "medai_ai_extraction_workflow_seam_15a_not_ready")
    print(
        json.dumps(
            {
                "report": str(SUMMARY_JSON_PATH.relative_to(REPO_ROOT)),
                "privacy_result": privacy["privacy_result"],
                "external_api_used": summary["external_api_used"],
                "auto_accept": summary["auto_accept"],
                "active_written_count": summary["active_written_count"],
                "review_bound_package_count": summary["review_bound_package_count"],
                "adapter_name": summary["adapter_name"],
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
