"""CKA-TERM-01F synthetic intake rehearsal.

Creates tiny synthetic terminology ZIPs in temporary directories only and
rehearses classification, safe extraction, test ack, readiness, dry-run
planning, and TERM-02 preflight.
"""
from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from clinical_knowledge.terminology.file_classifier import classify_filename
from clinical_knowledge.terminology.import_dry_run import run_terminology_import_dry_run
from clinical_knowledge.terminology.intake_automation import (
    compute_readiness,
    prepare_intake_folders,
    safe_extract_zip,
)
from clinical_knowledge.terminology.term02_preflight_gate import run_term02_preflight_gate


@dataclass(frozen=True)
class SyntheticIntakeRehearsalResult:
    classification_passed: bool
    safe_entries_extracted: bool
    zip_slip_protection_verified: bool
    readiness_passed: bool
    dry_run_passed: bool
    term02_preflight_passed: bool
    repo_terminology_data_created: bool
    external_api_used: bool = False
    real_import_performed: bool = False

    def safe_public_summary(self) -> dict:
        return {
            "classification_passed": self.classification_passed,
            "safe_entries_extracted": self.safe_entries_extracted,
            "zip_slip_protection_verified": self.zip_slip_protection_verified,
            "readiness_passed": self.readiness_passed,
            "dry_run_passed": self.dry_run_passed,
            "term02_preflight_passed": self.term02_preflight_passed,
            "repo_terminology_data_created": self.repo_terminology_data_created,
            "external_api_used": self.external_api_used,
            "real_import_performed": self.real_import_performed,
        }


def run_synthetic_intake_rehearsal(*, repo_root: Path | None = None) -> SyntheticIntakeRehearsalResult:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    real_repo_term_root_existed = (repo_root / "terminology_data").exists()

    with tempfile.TemporaryDirectory(prefix="medai_term01f_") as tmp:
        tmp_root = Path(tmp)
        prepare_intake_folders(repo_root=tmp_root, write_template=False)
        zip_dir = tmp_root / "zips"
        zip_dir.mkdir()
        archives = _create_synthetic_archives(zip_dir)
        classification_passed = all(classify_filename(p.name).system is not None for p in archives)
        extract = safe_extract_zip(archives, repo_root=tmp_root, extract_approved=True)
        _write_test_ack(tmp_root / "terminology_data" / "LICENSE_ACK_PRIVATE.json")
        test_env = {"CKA_TERM01_TEST_LICENSE_ACK": "1"}
        readiness = compute_readiness(repo_root=tmp_root, license_test_mode=True, license_env=test_env)
        dry_run = run_terminology_import_dry_run(
            repo_root=tmp_root,
            license_test_mode=True,
            license_env=test_env,
        )
        gate = run_term02_preflight_gate(repo_root=tmp_root)

        return SyntheticIntakeRehearsalResult(
            classification_passed=classification_passed,
            safe_entries_extracted=extract.entries_extracted >= 6,
            zip_slip_protection_verified=extract.entries_blocked_zip_slip >= 12,
            readiness_passed=bool(readiness.systems_import_ready),
            dry_run_passed=bool(dry_run["plan"]["systems_import_ready"]) and dry_run["plan"]["dry_run"] is True,
            term02_preflight_passed=gate.allowed,
            repo_terminology_data_created=(repo_root / "terminology_data").exists() and not real_repo_term_root_existed,
        )


def _create_synthetic_archives(zip_dir: Path) -> list[Path]:
    archives = {
        "Loinc_Synthetic_Test.zip": {"Loinc.csv": "LOINC_NUM,COMPONENT\nSYN-1,Example\n"},
        "RxNorm_Synthetic_Test.zip": {"RXNCONSO.RRF": "RXCUI|LAT|TS|LUI|STT|SUI|ISPREF|RXAUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|\n"},
        "UMLS_Synthetic_Test.zip": {
            "MRCONSO.RRF": "C0000001|ENG|||||||||||||Synthetic term|\n",
            "MRSTY.RRF": "C0000001|T000|Synthetic Type|\n",
        },
        "SnomedCT_Synthetic_Test.zip": {
            "sct2_Concept_Synthetic.txt": "id\teffectiveTime\tactive\n100\t20240101\t1\n",
            "sct2_Description_Synthetic.txt": "id\tconceptId\tterm\n200\t100\tSynthetic description\n",
        },
    }
    paths: list[Path] = []
    malicious = ["../evil.txt", "/absolute/evil.txt", "C:/evil.txt"]
    for name, files in archives.items():
        path = zip_dir / name
        with zipfile.ZipFile(path, "w") as zf:
            for member_name, content in files.items():
                zf.writestr(member_name, content)
            for member_name in malicious:
                zf.writestr(member_name, "blocked")
        paths.append(path)
    return paths


def _write_test_ack(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "operator_acknowledged": True,
                "acknowledged_systems": ["loinc", "rxnorm", "umls", "snomed_ct"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
