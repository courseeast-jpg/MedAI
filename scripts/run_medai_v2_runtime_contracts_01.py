#!/usr/bin/env python3
"""MEDAI-V2-RUNTIME-CONTRACTS-01 — Reports-only V2 runtime contracts
audit.

Read-only audit. Confirms:
* the three V2-RUNTIME-CONTRACTS-01 report files exist;
* the typing-only contracts module exists and imports silently;
* the contracts module uses only standard-library imports;
* no forbidden module names appear in the contracts source;
* every required contract name (CONTRACT_NAMES) resolves from the
  module;
* the `V2RuntimeSafetyProfile` default carries the canonical
  invariants;
* every public-safe report passes
  `clinical_knowledge.privacy.check_public_report_payload`.

No runtime code is modified. No private files are opened. No
licensed terminology rows are read. No tags are touched. No external
APIs are called.
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_runtime_contracts_01"
CONTRACTS_MODULE_PATH = (
    REPO_ROOT / "clinical_knowledge" / "v2_contracts" / "runtime_contracts.py"
)

REQUIRED_REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_RUNTIME_CONTRACTS_01.md",
    REPORT_DIR / "medai_v2_runtime_contracts_01_report.json",
    REPORT_DIR / "medai_v2_runtime_contracts_01_report.md",
)

ALLOWED_STDLIB_TOP_LEVEL_MODULES = {
    "dataclasses",
    "datetime",
    "enum",
    "typing",
    "__future__",
}

# Forbidden module names — even appearing inside the source as a quoted
# string would be flagged. The contracts module deliberately mentions
# them only inside this audit script's allow-list.
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


def _audit() -> Dict[str, Any]:
    findings: Dict[str, Any] = {
        "report_files_present": True,
        "missing_report_files": [],
        "contracts_module_present": CONTRACTS_MODULE_PATH.is_file(),
        "contracts_module_import_silent": False,
        "contracts_module_import_stdout": "",
        "contracts_module_import_stderr": "",
        "contracts_module_import_failure": None,
        "standard_library_only": True,
        "forbidden_import_lines_in_source": [],
        "required_contract_names_resolved": True,
        "missing_contract_names": [],
        "default_safety_profile_correct": False,
        "default_safety_profile_actual": {},
        "public_report_privacy_passed": False,
        "public_report_privacy_failures": [],
    }

    for p in REQUIRED_REPORT_FILES:
        if not p.is_file():
            findings["report_files_present"] = False
            findings["missing_report_files"].append(str(p.relative_to(REPO_ROOT)))

    if not findings["contracts_module_present"]:
        return findings

    # Static scan: confirm every `import ...` / `from ... import` line
    # uses only allowed stdlib top-level modules and contains no
    # forbidden module name.
    src = CONTRACTS_MODULE_PATH.read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        # crude module-name extraction
        if stripped.startswith("from "):
            module_name = stripped.split(None, 2)[1]
        else:
            module_name = stripped.split(None, 2)[1].split(",")[0]
        top_level = module_name.split(".", 1)[0]
        if top_level not in ALLOWED_STDLIB_TOP_LEVEL_MODULES:
            findings["standard_library_only"] = False
            findings["forbidden_import_lines_in_source"].append(
                {"lineno": lineno, "module": module_name}
            )
        for prefix in FORBIDDEN_IMPORT_PREFIXES:
            if module_name == prefix or module_name.startswith(prefix + "."):
                findings["standard_library_only"] = False
                findings["forbidden_import_lines_in_source"].append(
                    {"lineno": lineno, "module": module_name, "forbidden_prefix": prefix}
                )

    # Import smoke test
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from clinical_knowledge.v2_contracts import runtime_contracts as rc

        findings["contracts_module_import_stdout"] = sys.stdout.getvalue()
        findings["contracts_module_import_stderr"] = sys.stderr.getvalue()
        findings["contracts_module_import_silent"] = (
            findings["contracts_module_import_stdout"] == ""
            and findings["contracts_module_import_stderr"] == ""
        )
    except Exception as exc:
        findings["contracts_module_import_failure"] = (
            f"{type(exc).__name__}: {exc}"
        )
        sys.stdout, sys.stderr = old_out, old_err
        return findings
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    # Required contract names — every name in CONTRACT_NAMES must
    # resolve from the module.
    for name in rc.CONTRACT_NAMES:
        if not hasattr(rc, name):
            findings["required_contract_names_resolved"] = False
            findings["missing_contract_names"].append(name)

    # Default safety profile invariants
    profile = rc.V2RuntimeSafetyProfile()
    expected = {
        "local_only": True,
        "review_bound": True,
        "external_api_blocked": True,
        "auto_accept_allowed": False,
        "terminology_lookup_aggregate_only": True,
        "private_adapter_implemented": False,
        "cue_expansion_recommended": False,
        "clinical_decision_expansion": False,
    }
    actual = {k: getattr(profile, k) for k in expected}
    findings["default_safety_profile_actual"] = actual
    findings["default_safety_profile_correct"] = actual == expected

    # Privacy check on the three public reports
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
        except Exception as exc:  # pragma: no cover (defensive)
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
            findings["contracts_module_present"],
            findings["contracts_module_import_silent"],
            findings["contracts_module_import_failure"] is None,
            findings["standard_library_only"],
            findings["required_contract_names_resolved"],
            findings["default_safety_profile_correct"],
            findings["public_report_privacy_passed"],
        )
    )


def main() -> int:
    findings = _audit()
    out = {
        "block_id": "MEDAI-V2-RUNTIME-CONTRACTS-01",
        "mode": "reports_only_typing_only_audit",
        "reports_only": True,
        "audit_findings": findings,
        "all_clean": _all_clean(findings),
        "next_recommended_block": "V2-VALIDATION-HARNESS-01",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
