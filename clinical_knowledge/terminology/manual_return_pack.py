"""CKA-TERM-01F manual return pack helpers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate


@dataclass(frozen=True)
class ManualReturnPackResult:
    preparation_checked: bool
    readiness_checked: bool
    dry_run_checked: bool
    term02_preflight_checked: bool
    term02_preflight_allowed: bool
    term02_preflight_reason_codes: list[str]
    real_import_performed: bool = False
    external_api_used: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "preparation_checked": self.preparation_checked,
            "readiness_checked": self.readiness_checked,
            "dry_run_checked": self.dry_run_checked,
            "term02_preflight_checked": self.term02_preflight_checked,
            "term02_preflight_allowed": self.term02_preflight_allowed,
            "term02_preflight_reason_codes": list(self.term02_preflight_reason_codes),
            "real_import_performed": self.real_import_performed,
            "external_api_used": self.external_api_used,
        }


def run_manual_return_pack(
    *,
    repo_root: Path | None = None,
    run_prepare: bool = False,
    run_readiness: bool = True,
    run_dry_run: bool = True,
    run_preflight: bool = True,
) -> ManualReturnPackResult:
    """Run one-command safe readiness checks.

    `run_prepare` defaults off because it may create the gitignored local
    terminology_data folder/template. It never downloads or imports.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    prep_checked = _run_script(repo_root, "scripts/cka_terminology_prepare_intake.py") if run_prepare else False
    readiness_checked = _run_script(repo_root, "scripts/cka_terminology_check_ready.py") if run_readiness else False
    dry_run_checked = _run_script(repo_root, "scripts/cka_terminology_import_dry_run.py") if run_dry_run else False
    preflight = run_term02_preflight_gate(repo_root=repo_root) if run_preflight else None
    return ManualReturnPackResult(
        preparation_checked=prep_checked,
        readiness_checked=readiness_checked,
        dry_run_checked=dry_run_checked,
        term02_preflight_checked=preflight is not None,
        term02_preflight_allowed=bool(preflight and preflight.allowed),
        term02_preflight_reason_codes=(preflight.reason_codes if preflight else []),
    )


def build_manual_return_guide_text() -> str:
    return (
        "# CKA-TERM-01F Manual Return Guide\n\n"
        "TERM-02 has not started. Real terminology import remains blocked until the operator manually supplies licensed files and a private acknowledgment.\n\n"
        "## Manual Files To Obtain\n\n"
        "- UMLS: licensed local files containing MRCONSO.RRF and MRSTY.RRF.\n"
        "- SNOMED CT: licensed local release files containing sct2_Concept and sct2_Description files.\n"
        "- RxNorm: licensed local files containing RXNCONSO.RRF.\n"
        "- LOINC: licensed local files containing Loinc.csv or LoincTable.csv.\n\n"
        "## Where To Place Files\n\n"
        "Place files under the local gitignored `terminology_data/` folder using these subfolders: `umls/`, `snomed_ct/`, `rxnorm/`, and `loinc/`.\n\n"
        "## Private License Acknowledgment\n\n"
        "Use `terminology_data/LICENSE_ACK_PRIVATE.json` locally. This file must stay private and must not be committed. Use only the safe fields `operator_acknowledged` and `acknowledged_systems`. Keep vendor terms out of reports and public files.\n\n"
        "Example structure:\n\n"
        "```json\n"
        "{\n"
        "  \"operator_acknowledged\": true,\n"
        "  \"acknowledged_systems\": [\"umls\", \"snomed_ct\", \"rxnorm\", \"loinc\"]\n"
        "}\n"
        "```\n\n"
        "## Before TERM-02\n\n"
        "Run `python scripts/cka_term02_preflight_gate.py`. TERM-02 cannot start until the preflight gate passes.\n\n"
        "No external APIs or downloads are used by the readiness checks. No clinical advice is generated.\n"
    )


def build_term02_preflight_checklist_text() -> str:
    return (
        "# CKA-TERM-02 Preflight Checklist\n\n"
        "- `terminology_data/` exists locally.\n"
        "- Supported subfolders exist: `umls/`, `snomed_ct/`, `rxnorm/`, `loinc/`.\n"
        "- Expected files are present for at least one licensed system.\n"
        "- `terminology_data/LICENSE_ACK_PRIVATE.json` exists locally.\n"
        "- `operator_acknowledged` is true.\n"
        "- `acknowledged_systems` covers every system with files present.\n"
        "- Readiness checker reports at least one import-ready system.\n"
        "- No `terminology_data/` files are staged.\n"
        "- No `data/terminology/` files are staged.\n"
        "- TERM-02 has not started before the gate passes.\n"
    )


def _run_script(repo_root: Path, script: str) -> bool:
    proc = subprocess.run(
        ["python", script],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0
