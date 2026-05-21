#!/usr/bin/env python3
"""MEDAI-V2-VALIDATION-HARNESS-01 — Reports-only audit.

Read-only. Confirms:

* the three V2-VALIDATION-HARNESS-01 report files exist;
* the typing-only validation_harness module exists and imports
  silently (no stdout/stderr);
* the validation_harness module uses only standard-library imports;
* the runtime_contracts module still imports silently;
* the V1 health-check catalog enumerates the five canonical entries;
* the V2 validation matrix exposes all 8 categories;
* every report passes ``clinical_knowledge.privacy.check_public_report_payload``;
* the V2-FOUNDATION-SPEC-02 / V2-ARCHITECTURE-SPEC-01 / V2-RUNTIME-
  CONTRACTS-01 report directories exist (the preceding-spec chain).

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

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_validation_harness_01"
HARNESS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "validation_harness.py"
)
CONTRACTS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py"
)

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_VALIDATION_HARNESS_01.md",
    REPORT_DIR / "medai_v2_validation_harness_01_report.json",
    REPORT_DIR / "medai_v2_validation_harness_01_report.md",
)

REQUIRED_PRIOR_REPORT_DIRS = (
    REPO_ROOT / "reports" / "medai_v2_architecture_spec_01",
    REPO_ROOT / "reports" / "medai_v2_foundation_spec_02",
    REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01",
)

ALLOWED_STDLIB_TOP_LEVEL_MODULES = {
    "dataclasses",
    "datetime",
    "enum",
    "typing",
    "__future__",
}

FORBIDDEN_IMPORT_PREFIXES = (
    "streamlit",
    "requests",
    "httpx",
    "http.",
    "urllib",
    "socket",
    "sqlite3",
    "sqlcipher3",
    "anthropic",
    "google.generativeai",
    "openai",
    "boto3",
    "psycopg",
)


def _module_uses_stdlib_only(path: Path) -> List[Dict[str, Any]]:
    """Return a list of violations: non-stdlib or forbidden imports."""
    violations: List[Dict[str, Any]] = []
    src = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if stripped.startswith("from "):
            module_name = stripped.split(None, 2)[1]
        else:
            module_name = stripped.split(None, 2)[1].split(",")[0]
        top_level = module_name.split(".", 1)[0]
        if top_level not in ALLOWED_STDLIB_TOP_LEVEL_MODULES:
            violations.append(
                {"file": path.name, "lineno": lineno, "module": module_name, "kind": "non_stdlib"}
            )
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            if module_name == prefix or module_name.startswith(prefix + "."):
                violations.append(
                    {"file": path.name, "lineno": lineno, "module": module_name, "kind": "forbidden_prefix", "prefix": prefix}
                )
    return violations


def _silent_import(module_path: str) -> Dict[str, Any]:
    """Import a module under captured stdio. Returns import status dict."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        sys.modules.pop(module_path, None)
        __import__(module_path)
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
        return {
            "module": module_path,
            "imported": True,
            "stdout": out,
            "stderr": err,
            "silent": (out == "" and err == ""),
            "error": None,
        }
    except Exception as exc:
        return {
            "module": module_path,
            "imported": False,
            "stdout": sys.stdout.getvalue(),
            "stderr": sys.stderr.getvalue(),
            "silent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _audit() -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "report_files_present": True,
        "missing_report_files": [],
        "prior_v2_report_dirs_present": True,
        "missing_prior_v2_report_dirs": [],
        "harness_module_present": HARNESS_MODULE_PATH.is_file(),
        "harness_standard_library_only_violations": [],
        "contracts_standard_library_only_violations": [],
        "harness_import": {},
        "contracts_import": {},
        "v1_health_check_count": 0,
        "v1_health_check_names": [],
        "v2_validation_matrix_category_count": 0,
        "v2_validation_matrix_categories": [],
        "public_report_privacy_passed": False,
        "public_report_privacy_failures": [],
    }

    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            findings["report_files_present"] = False
            findings["missing_report_files"].append(str(p.relative_to(REPO_ROOT)))

    for d in REQUIRED_PRIOR_REPORT_DIRS:
        if not d.is_dir():
            findings["prior_v2_report_dirs_present"] = False
            findings["missing_prior_v2_report_dirs"].append(str(d.relative_to(REPO_ROOT)))

    if HARNESS_MODULE_PATH.is_file():
        findings["harness_standard_library_only_violations"] = (
            _module_uses_stdlib_only(HARNESS_MODULE_PATH)
        )
    if CONTRACTS_MODULE_PATH.is_file():
        findings["contracts_standard_library_only_violations"] = (
            _module_uses_stdlib_only(CONTRACTS_MODULE_PATH)
        )

    findings["harness_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.validation_harness"
    )
    findings["contracts_import"] = _silent_import(
        "clinical_knowledge.v2_contracts.runtime_contracts"
    )

    if findings["harness_import"]["imported"]:
        from clinical_knowledge.v2_contracts import validation_harness as vh  # noqa: WPS433

        findings["v1_health_check_count"] = len(vh.V1_HEALTH_CHECKS)
        findings["v1_health_check_names"] = list(vh.v1_health_check_names())
        findings["v2_validation_matrix_category_count"] = len(vh.V2_VALIDATION_MATRIX)
        findings["v2_validation_matrix_categories"] = list(
            vh.v2_validation_matrix_categories()
        )

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
            findings["harness_module_present"],
            findings["harness_standard_library_only_violations"] == [],
            findings["contracts_standard_library_only_violations"] == [],
            findings["harness_import"].get("imported") is True,
            findings["harness_import"].get("silent") is True,
            findings["contracts_import"].get("imported") is True,
            findings["contracts_import"].get("silent") is True,
            findings["v1_health_check_count"] == 5,
            findings["v2_validation_matrix_category_count"] == 8,
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-V2-VALIDATION-HARNESS-01",
        "mode": "validation_harness_only_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "V2-UI-SHELL-SPEC-01_OR_V2-DATA-INFRA-SPEC-01",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
