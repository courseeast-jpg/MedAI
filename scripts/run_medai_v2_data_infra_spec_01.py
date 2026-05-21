#!/usr/bin/env python3
"""MEDAI-V2-DATA-INFRA-SPEC-01 — Reports-only audit.

Read-only. Confirms:

* the three V2-DATA-INFRA-SPEC-01 report files exist;
* prior V2 dependency report directories exist
  (architecture / foundation / runtime contracts /
  validation harness / UI shell);
* the V2 typing-only modules (runtime_contracts, validation_harness)
  still import silently;
* the JSON report carries the canonical data-infra invariants set to
  the expected values;
* the report describes 8 conceptual stores;
* the report enumerates 10 future implementation gates A-J;
* the report enumerates 9 migration gates;
* every report passes
  ``clinical_knowledge.privacy.check_public_report_payload``;
* no app/main.py / launcher / persistence / Streamlit / V2-contract
  mention of this block exists.

No runtime code is modified. No private files are opened. No
licensed terminology rows are read. No tags are touched. No DB row is
read.
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_data_infra_spec_01"

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_DATA_INFRA_SPEC_01.md",
    REPORT_DIR / "medai_v2_data_infra_spec_01_report.json",
    REPORT_DIR / "medai_v2_data_infra_spec_01_report.md",
)

REQUIRED_PRIOR_V2_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
    REPO_ROOT / "reports" / "medai_v2_validation_harness_01",
    REPO_ROOT / "reports" / "medai_v2_ui_shell_spec_01",
)

REQUIRED_FALSE_BOOLEANS = (
    "runtime_db_accessed",
    "schema_changed",
    "migration_created",
    "migration_executed",
    "persistence_code_changed",
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
    "source_documents_opened",
    "raw_text_printed",
    "raw_filenames_printed",
    "private_paths_printed",
    "secrets_printed",
    "tags_touched",
    "cue_expansion_recommended",
)

REQUIRED_TRUE_BOOLEANS = (
    "reports_only",
    "requires_preceding_spec",
    "preceding_architecture_spec_present",
    "preceding_foundation_spec_present",
    "preceding_runtime_contracts_present",
    "preceding_validation_harness_present",
    "preceding_ui_shell_spec_present",
    "data_infra_spec_created",
    "conceptual_store_map_created",
    "ledger_audit_separation_defined",
    "rollback_doctrine_defined",
    "migration_gate_doctrine_defined",
    "public_report_data_doctrine_defined",
    "future_data_implementation_gates_defined",
    "v1_validation_healthcheck_catalog_preserved",
    "runtime_db_row_blind",
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


def _file_does_not_mention_this_block(p: Path) -> bool:
    if not p.is_file():
        return True
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return True
    return ("MEDAI-V2-DATA-INFRA-SPEC-01" not in src) and (
        "v2_data_infra_spec_01" not in src
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
        "store_count_correct": False,
        "store_count_actual": 0,
        "future_gate_count_correct": False,
        "future_gate_count_actual": 0,
        "migration_gate_count_correct": False,
        "migration_gate_count_actual": 0,
        "runtime_files_do_not_mention_this_block": True,
        "runtime_files_with_mention": [],
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
            findings["missing_prior_v2_report_dirs"].append(
                str(d.relative_to(REPO_ROOT))
            )

    findings["runtime_contracts_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.runtime_contracts"
    )
    findings["validation_harness_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.validation_harness"
    )

    runtime_check_files = (
        REPO_ROOT / "app" / "main.py",
        REPO_ROOT / "app" / "startup_preflight.py",
        REPO_ROOT / "app" / "config.py",
        REPO_ROOT / "Start_MedAI_UI.bat",
        REPO_ROOT / "Start_MedAI_UI_Silent.vbs",
        REPO_ROOT / "Start_MedAI_Test_UI.bat",
        REPO_ROOT / "Start_MedAI_UI_Encrypted.bat",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py",
        REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "__init__.py",
    )
    for p in runtime_check_files:
        if not _file_does_not_mention_this_block(p):
            findings["runtime_files_do_not_mention_this_block"] = False
            findings["runtime_files_with_mention"].append(str(p.relative_to(REPO_ROOT)))

    json_path = REPORT_DIR / "medai_v2_data_infra_spec_01_report.json"
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
        payload.get("next_recommended_block")
        == "V2-EXTRACTION-SPEC-01_OR_V2-ROADMAP-02"
    )

    stores = payload.get("section_d_conceptual_persistence_store_map")
    if isinstance(stores, list):
        findings["store_count_actual"] = len(stores)
        findings["store_count_correct"] = len(stores) == 8

    gates = payload.get("section_l_future_implementation_gates")
    if isinstance(gates, list):
        findings["future_gate_count_actual"] = len(gates)
        findings["future_gate_count_correct"] = len(gates) == 10

    mgates = payload.get("section_g_migration_gate_doctrine")
    if isinstance(mgates, list):
        findings["migration_gate_count_actual"] = len(mgates)
        findings["migration_gate_count_correct"] = len(mgates) == 9

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
            findings["store_count_correct"],
            findings["future_gate_count_correct"],
            findings["migration_gate_count_correct"],
            findings["runtime_files_do_not_mention_this_block"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-V2-DATA-INFRA-SPEC-01",
        "mode": "reports_only_data_infra_spec_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "V2-EXTRACTION-SPEC-01_OR_V2-ROADMAP-02",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
