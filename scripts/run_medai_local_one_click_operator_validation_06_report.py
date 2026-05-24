#!/usr/bin/env python3
"""MEDAI-LOCAL-ONE-CLICK-OPERATOR-VALIDATION-06 — report aggregator.

Streamlit-free, deterministic, local-only. The PowerShell orchestrator
invokes this helper after it has run the existing 05 validation and UI
smoke scripts. This helper:

* aggregates the 05 outputs into 3 public-safe 06 reports;
* never reads or echoes any raw filename, raw file path, raw OCR text,
  raw document text, PHI, or selected-file metadata beyond suffix and
  short stable hash;
* runs the same ``clinical_knowledge.privacy.check_public_report_payload``
  used by every other public report in this chain.

This helper is also the unit-testable layer for the one-click flow,
because PowerShell file-picker UI cannot be exercised in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["MEDAI_ALLOW_EXTERNAL_API"] = "false"
os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")

REPORT_DIR = REPO_ROOT / "reports" / "medai_local_one_click_operator_validation_06"
JSON_REPORT_PATH = REPORT_DIR / "medai_local_one_click_operator_validation_06_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_local_one_click_operator_validation_06_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_LOCAL_ONE_CLICK_OPERATOR_VALIDATION_06.md"

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
SYNTHETIC_INPUT_SUFFIX = ".txt"
SAFE_HASH_PREFIX = "oneclick_"

ALLOWED_INPUT_MODES = ("selected_file", "synthetic_fallback")
ALLOWED_INPUT_SUFFIXES = (".pdf", ".txt")


def safe_input_hash(size: int, suffix: str) -> str:
    """Public-safe handle: short SHA-256 over (size, suffix).

    Never includes the filename or any directory path.
    """
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


def build_report(
    *,
    branch: str,
    head: str,
    dependency_prepare_exit_code: int,
    selected_input_mode: str,
    input_suffix: str,
    input_safe_hash: str,
    one_doc_validation_exit_code: int,
    streamlit_smoke_exit_code: int,
) -> Dict[str, Any]:
    """Build the canonical 06 report dict from sanitized inputs.

    The PowerShell orchestrator passes only sanitized facts:

    * ``selected_input_mode`` — exactly one of ``selected_file`` or
      ``synthetic_fallback``;
    * ``input_suffix`` — only ``.pdf`` or ``.txt``;
    * ``input_safe_hash`` — already-computed safe handle from
      :func:`safe_input_hash`; the helper re-validates the prefix.

    Anything else is rejected as a privacy violation.
    """
    if selected_input_mode not in ALLOWED_INPUT_MODES:
        raise ValueError(f"selected_input_mode must be one of {ALLOWED_INPUT_MODES}")
    if input_suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
        raise ValueError(f"input_suffix must be one of {ALLOWED_INPUT_SUFFIXES}")
    if not input_safe_hash.startswith(SAFE_HASH_PREFIX):
        raise ValueError(f"input_safe_hash must start with {SAFE_HASH_PREFIX}")
    if "/" in input_safe_hash or "\\" in input_safe_hash:
        raise ValueError("input_safe_hash must not contain path separators")

    one_doc_payload = _read_json(ONE_DOC_REPORT_JSON)
    smoke_payload = _read_json(SMOKE_REPORT_JSON)
    one_doc_conclusion = str(one_doc_payload.get("conclusion") or "missing")
    smoke_findings = (smoke_payload.get("audit_findings") or {}) if smoke_payload else {}
    smoke_helper_passed = bool(smoke_findings.get("helper_import_passed"))
    smoke_overall_passed = bool(smoke_payload.get("passed"))

    ui_smoke_result = "passed" if smoke_overall_passed else (
        "helper_only_passed" if smoke_helper_passed else "not_passed"
    )

    one_doc_findings = (one_doc_payload.get("audit_findings") or {}) if one_doc_payload else {}
    external_api_used = bool(
        one_doc_payload.get("external_api_used")
        or smoke_payload.get("external_api_used")
        or one_doc_findings.get("external_api_used")
        or smoke_findings.get("external_api_used")
    )
    auto_accept_enabled = bool(
        one_doc_payload.get("auto_accept_enabled")
        or smoke_payload.get("auto_accept_enabled")
    )

    overall_pass = (
        dependency_prepare_exit_code == 0
        and one_doc_validation_exit_code == 0
        and streamlit_smoke_exit_code == 0
        and one_doc_conclusion in {"one_doc_operator_validation_ready", "no_input_file"}
        and not external_api_used
        and not auto_accept_enabled
    )

    findings: Dict[str, Any] = {
        "branch": branch,
        "head": head,
        "dependency_prepare_exit_code": int(dependency_prepare_exit_code),
        "selected_input_mode": selected_input_mode,
        "input_suffix": input_suffix.lower(),
        "input_safe_hash": input_safe_hash,
        "one_doc_validation_exit_code": int(one_doc_validation_exit_code),
        "one_doc_validation_conclusion": one_doc_conclusion,
        "streamlit_smoke_exit_code": int(streamlit_smoke_exit_code),
        "ui_smoke_result": ui_smoke_result,
        "external_api_used": external_api_used,
        "auto_accept_enabled": auto_accept_enabled,
        "next_ui_command": NEXT_UI_COMMAND,
    }
    report = {
        "block_id": "MEDAI-LOCAL-ONE-CLICK-OPERATOR-VALIDATION-06",
        "mode": "one_click_operator_validation_aggregation",
        "branch_expected": "clinical-knowledge-architecture",
        "audit_findings": findings,
        "external_api_used": external_api_used,
        "auto_accept_enabled": auto_accept_enabled,
        "real_pdf_committed": False,
        "real_screenshot_committed": False,
        "raw_text_printed": False,
        "raw_ocr_text_printed": False,
        "raw_filenames_printed": False,
        "private_paths_printed": False,
        "phi_printed": False,
        "passed": bool(overall_pass),
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
    findings = report["audit_findings"]
    md_lines = [
        "# MEDAI-LOCAL-ONE-CLICK-OPERATOR-VALIDATION-06 — Report",
        "",
        f"Branch: `{findings['branch']}` | HEAD: `{findings['head']}`",
        "",
        "## Pipeline",
        "",
        f"- dependency prepare exit code: {findings['dependency_prepare_exit_code']}",
        f"- selected input mode: `{findings['selected_input_mode']}`",
        f"- input suffix: `{findings['input_suffix']}`",
        f"- input safe hash: `{findings['input_safe_hash']}`",
        f"- one-doc validation exit code: {findings['one_doc_validation_exit_code']}",
        f"- one-doc validation conclusion: `{findings['one_doc_validation_conclusion']}`",
        f"- streamlit smoke exit code: {findings['streamlit_smoke_exit_code']}",
        f"- UI smoke result: `{findings['ui_smoke_result']}`",
        "",
        "## Safety",
        "",
        f"- external API used: {findings['external_api_used']}",
        f"- auto-accept enabled: {findings['auto_accept_enabled']}",
        f"- privacy check passed: {report['privacy_check_passed']}",
        "",
        "## Next operator command",
        "",
        f"```\n{findings['next_ui_command']}\n```",
        "",
    ]
    MD_REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    SHORT_MD_PATH.write_text(
        "# MEDAI-LOCAL-ONE-CLICK-OPERATOR-VALIDATION-06 — Short Summary\n\n"
        f"Branch `{findings['branch']}`, HEAD `{findings['head']}`\n\n"
        f"- input mode: `{findings['selected_input_mode']}`\n"
        f"- input suffix: `{findings['input_suffix']}`\n"
        f"- one-doc conclusion: `{findings['one_doc_validation_conclusion']}`\n"
        f"- UI smoke: `{findings['ui_smoke_result']}`\n"
        f"- external API used: {findings['external_api_used']}\n"
        f"- auto-accept enabled: {findings['auto_accept_enabled']}\n"
        f"- privacy check passed: {report['privacy_check_passed']}\n\n"
        f"Next operator command:\n\n```\n{findings['next_ui_command']}\n```\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the 05 outputs into the one-click 06 report"
    )
    parser.add_argument("--branch", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument(
        "--dependency-prepare-exit-code", required=True, type=int
    )
    parser.add_argument(
        "--selected-input-mode", required=True, choices=ALLOWED_INPUT_MODES
    )
    parser.add_argument(
        "--input-suffix", required=True, choices=[s.lstrip(".") for s in ALLOWED_INPUT_SUFFIXES]
    )
    parser.add_argument("--input-safe-hash", required=True)
    parser.add_argument(
        "--one-doc-validation-exit-code", required=True, type=int
    )
    parser.add_argument(
        "--streamlit-smoke-exit-code", required=True, type=int
    )
    args = parser.parse_args(argv)

    report = build_report(
        branch=args.branch,
        head=args.head,
        dependency_prepare_exit_code=args.dependency_prepare_exit_code,
        selected_input_mode=args.selected_input_mode,
        input_suffix=f".{args.input_suffix}",
        input_safe_hash=args.input_safe_hash,
        one_doc_validation_exit_code=args.one_doc_validation_exit_code,
        streamlit_smoke_exit_code=args.streamlit_smoke_exit_code,
    )
    write_reports(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
