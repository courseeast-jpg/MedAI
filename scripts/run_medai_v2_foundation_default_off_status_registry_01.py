#!/usr/bin/env python3
"""MEDAI-V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01 audit."""
from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "reports" / "medai_v2_foundation_default_off_status_registry_01"
REPORT_FILES = (
    REPORT_DIR / "MEDAI_V2_FOUNDATION_DEFAULT_OFF_STATUS_REGISTRY_01.md",
    REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.json",
    REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.md",
)
REGISTRY_MODULE = REPO_ROOT / "clinical_knowledge" / "v2_foundation" / "status_registry.py"
ALLOWED_IMPORT_ROOTS = {"__future__", "dataclasses", "enum", "typing"}
FORBIDDEN_RUNTIME_FILES = (
    REPO_ROOT / "app" / "main.py",
    REPO_ROOT / "app" / "startup_preflight.py",
    REPO_ROOT / "app" / "config.py",
)
FORBIDDEN_RUNTIME_DIRS = (
    REPO_ROOT / "app",
    REPO_ROOT / "launchers",
    REPO_ROOT / "scripts",
    REPO_ROOT / "clinical_knowledge" / "extraction",
    REPO_ROOT / "clinical_knowledge" / "extractors",
    REPO_ROOT / "clinical_knowledge" / "ocr",
    REPO_ROOT / "clinical_knowledge" / "terminology",
    REPO_ROOT / "clinical_knowledge" / "classification",
    REPO_ROOT / "clinical_knowledge" / "classifier",
    REPO_ROOT / "clinical_knowledge" / "persistence",
    REPO_ROOT / "clinical_knowledge" / "db",
)


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _files_mentioning_registry() -> list[str]:
    needles = ("v2_foundation.status_registry", "status_registry")
    hits: list[str] = []
    for root in FORBIDDEN_RUNTIME_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path == REGISTRY_MODULE or path.name == "run_medai_v2_foundation_default_off_status_registry_01.py":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(needle in text for needle in needles):
                hits.append(str(path.relative_to(REPO_ROOT)))
    for path in FORBIDDEN_RUNTIME_FILES:
        if path.exists() and str(path.relative_to(REPO_ROOT)) not in hits:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(needle in text for needle in needles):
                hits.append(str(path.relative_to(REPO_ROOT)))
    return sorted(set(hits))


def audit() -> Dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        module = importlib.import_module("clinical_knowledge.v2_foundation.status_registry")
    registry = module.get_v2_capability_registry()
    summary = module.summarize_v2_capability_registry()
    import_roots = _import_roots(REGISTRY_MODULE)
    forbidden_mentions = _files_mentioning_registry()
    report_files_present = all(path.is_file() for path in REPORT_FILES)
    payload = {}
    if (REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.json").is_file():
        payload = json.loads(
            (REPORT_DIR / "medai_v2_foundation_default_off_status_registry_01_report.json").read_text(
                encoding="utf-8"
            )
        )
    findings: Dict[str, Any] = {
        "registry_imported": True,
        "stdout_on_import": bool(stdout.getvalue()),
        "stderr_on_import": bool(stderr.getvalue()),
        "standard_library_only": import_roots <= ALLOWED_IMPORT_ROOTS,
        "import_roots": sorted(import_roots),
        "entry_count": len(registry),
        "default_enabled_count": summary["default_enabled_count"],
        "runtime_wired_count": summary["runtime_wired_count"],
        "ui_wired_count": summary["ui_wired_count"],
        "blocked_count": summary["blocked_count"],
        "all_default_off": summary["all_default_off"],
        "no_runtime_wiring": summary["no_runtime_wiring"],
        "no_ui_wiring": summary["no_ui_wiring"],
        "terminology_private_adapter_blocked": summary["terminology_private_adapter_blocked"],
        "cue_expansion_blocked": summary["cue_expansion_blocked"],
        "clinical_decision_logic_expansion_blocked": summary[
            "clinical_decision_logic_expansion_blocked"
        ],
        "forbidden_runtime_mentions": forbidden_mentions,
        "report_files_present": report_files_present,
        "json_registry_entry_count": payload.get("registry_entry_count"),
        "json_recommended_next_block": payload.get("recommended_next_block"),
    }
    findings["passed"] = (
        findings["registry_imported"]
        and not findings["stdout_on_import"]
        and not findings["stderr_on_import"]
        and findings["standard_library_only"]
        and findings["entry_count"] == 13
        and findings["default_enabled_count"] == 0
        and findings["runtime_wired_count"] == 0
        and findings["ui_wired_count"] == 0
        and findings["blocked_count"] == 3
        and findings["all_default_off"]
        and findings["no_runtime_wiring"]
        and findings["no_ui_wiring"]
        and findings["terminology_private_adapter_blocked"]
        and findings["cue_expansion_blocked"]
        and findings["clinical_decision_logic_expansion_blocked"]
        and not findings["forbidden_runtime_mentions"]
        and findings["report_files_present"]
        and findings["json_registry_entry_count"] == 13
        and findings["json_recommended_next_block"] == "V2-ROADMAP-03"
    )
    return findings


def main() -> int:
    findings = audit()
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0 if findings["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

