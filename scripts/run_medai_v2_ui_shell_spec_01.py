#!/usr/bin/env python3
"""MEDAI-V2-UI-SHELL-SPEC-01 — Reports-only audit.

Read-only. Confirms:

* the three V2-UI-SHELL-SPEC-01 report files exist;
* prior V2 dependency report directories exist (architecture /
  foundation / runtime contracts / validation harness);
* the typing-only V2 contract + validation-harness modules still
  import silently;
* the JSON report carries the canonical UI invariants set to the
  expected values;
* every report passes
  ``clinical_knowledge.privacy.check_public_report_payload``;
* no Streamlit code or ``app/main.py`` mention of this block exists
  (i.e. no UI implementation has been wired by this block).

No runtime code is modified. No private files are opened. No
licensed terminology rows are read. No tags are touched.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_UI_SHELL_SPEC_01.md",
    REPORT_DIR / "medai_v2_ui_shell_spec_01_report.json",
    REPORT_DIR / "medai_v2_ui_shell_spec_01_report.md",
)

REQUIRED_PRIOR_V2_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
)

REQUIRED_FALSE_BOOLEANS = (
    "runtime_behavior_changed",
    "app_main_changed",
    "streamlit_code_changed",
    "ui_changed",
    "launcher_changed",
    "startup_config_changed",
    "extraction_changed",
    "ocr_changed",
    "classifier_changed",
    "threshold_scoring_changed",
    "cue_pack_changed",
    "clinical_behavior_changed",
    "ddi_behavior_changed",
    "terminology_behavior_changed",
    "private_adapter_implemented",
    "concrete_adapters_implemented",
    "runtime_wiring_added",
    "external_api_used",
    "private_data_accessed",
    "licensed_rows_read",
    "licensed_rows_exposed",
    "private_license_ack_read",
    "private_config_read",
    "runtime_db_accessed",
    "source_documents_opened",
    "raw_text_printed",
    "raw_filenames_printed",
    "raw_filenames_exposed",
    "private_paths_printed",
    "private_paths_exposed",
    "secrets_printed",
    "tags_touched",
    "cue_expansion_recommended",
    "auto_accept_allowed_default",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "requires_preceding_spec",
    "preceding_architecture_spec_present",
    "preceding_foundation_spec_present",
    "preceding_runtime_contracts_present",
    "preceding_validation_harness_present",
    "ui_shell_spec_created",
    "screen_inventory_created",
    "future_ui_implementation_gates_defined",
    "ui_spec_only",
    "read_only_shell_spec",
    "no_ui_actions_added",
    "no_callbacks_added",
    "no_session_state_logic_added",
    "review_bound_default",
    "local_only_default",
    "external_api_blocked_default",
    "v1_release_preserved",
)


def _silent_import(module_dotted: str) -> Dict[str, Any]:
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        sys.modules.pop(module_dotted, None)
        __import__(module_dotted)
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
        return {
            "module": module_dotted,
            "imported": True,
            "stdout": out,
            "stderr": err,
            "silent": (out == "" and err == ""),
            "error": None,
        }
    except Exception as exc:
        return {
            "module": module_dotted,
            "imported": False,
            "stdout": sys.stdout.getvalue(),
            "stderr": sys.stderr.getvalue(),
            "silent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _app_main_does_not_mention_this_block() -> bool:
    p = REPO_ROOT / "app" / "main.py"
    if not p.is_file():
        return True
    src = p.read_text(encoding="utf-8")
    return ("MEDAI-V2-UI-SHELL-SPEC-01" not in src) and (
        "v2_ui_shell_spec_01" not in src
    )


def _audit() -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "report_files_present": True,
        "missing_report_files": [],
        "prior_v2_report_dirs_present": True,
        "missing_prior_v2_report_dirs": [],
        "runtime_contracts_import": {},
        "validation_harness_import": {},
        "json_required_false_booleans_match": True,
        "json_required_false_booleans_mismatches": [],
        "json_required_true_booleans_match": True,
        "json_required_true_booleans_mismatches": [],
        "next_recommended_block_correct": False,
        "shell_count_correct": False,
        "shell_count_actual": 0,
        "app_main_does_not_mention_this_block": _app_main_does_not_mention_this_block(),
        "public_report_privacy_passed": False,
        "public_report_privacy_failures": [],
    }

    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            findings["report_files_present"] = False
            findings["missing_report_files"].append(str(p.relative_to(REPO_ROOT)))

    for d in REQUIRED_PRIOR_V2_REPORT_DIRS:
        if not d.is_dir():
            findings["prior_v2_report_dirs_present"] = False
            findings["missing_prior_v2_report_dirs"].append(str(d.relative_to(REPO_ROOT)))

    findings["runtime_contracts_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.runtime_contracts"
    )
    findings["validation_harness_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.validation_harness"
    )

    json_path = REPORT_DIR / "medai_v2_ui_shell_spec_01_report.json"
    if not json_path.is_file():
        return findings

    payload: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    for key in REQUIRED_FALSE_BOOLEANS:
        if payload.get(key) is not False:
            findings["json_required_false_booleans_match"] = False
            findings["json_required_false_booleans_mismatches"].append(
                {"key": key, "value": payload.get(key)}
            )

    for key in REQUIRED_TRUE_BOOLEANS:
        if payload.get(key) is not True:
            findings["json_required_true_booleans_match"] = False
            findings["json_required_true_booleans_mismatches"].append(
                {"key": key, "value": payload.get(key)}
            )

    findings["next_recommended_block_correct"] = (
        payload.get("next_recommended_block") == "V2-DATA-INFRA-SPEC-01"
    )

    shells = payload.get("section_d_proposed_v2_ui_shell_map")
    if isinstance(shells, list):
        findings["shell_count_actual"] = len(shells)
        findings["shell_count_correct"] = len(shells) == 8

    # Privacy check on the three reports
    from clinical_knowledge.privacy import check_public_report_payload  # noqa: WPS433

    privacy_pass = True
    privacy_failures: List[str] = []
    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            continue
        try:
            content: Any = p.read_text(encoding="utf-8")
            if p.suffix == ".json":
                content = json.loads(content)
        except Exception as exc:
            privacy_pass = False
            privacy_failures.append(f"{p.name}: read/parse failure ({exc})")
            continue
        result = check_public_report_payload(content)
        if not result.passed:
            privacy_pass = False
            privacy_failures.append(p.name)
    findings["public_report_privacy_passed"] = privacy_pass
    findings["public_report_privacy_failures"] = privacy_failures

    return findings


def _all_clean(findings: Dict[str, Any]) -> bool:
    return all(
        (
            findings["report_files_present"],
            findings["prior_v2_report_dirs_present"],
            findings["runtime_contracts_import"].get("imported") is True,
            findings["runtime_contracts_import"].get("silent") is True,
            findings["validation_harness_import"].get("imported") is True,
            findings["validation_harness_import"].get("silent") is True,
            findings["json_required_false_booleans_match"],
            findings["json_required_true_booleans_match"],
            findings["next_recommended_block_correct"],
            findings["shell_count_correct"],
            findings["app_main_does_not_mention_this_block"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-V2-UI-SHELL-SPEC-01",
        "mode": "reports_only_ui_shell_spec_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "V2-DATA-INFRA-SPEC-01",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
