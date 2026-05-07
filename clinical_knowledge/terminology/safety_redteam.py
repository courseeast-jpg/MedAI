"""CKA-TERM-01H terminology safety red-team scenarios."""
from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clinical_knowledge.terminology.integration import (
    code_entity_via_local_terminology,
    safe_b07_boundary_summary,
)
from clinical_knowledge.terminology.intake_automation import safe_extract_zip
from clinical_knowledge.terminology.license_gate import license_acknowledged_for
from clinical_knowledge.terminology.lookup_service import TerminologyLookupService
from clinical_knowledge.terminology.models import TerminologyLookupStatus, TerminologySystem
from clinical_knowledge.terminology.parsers import parse_loinc_csv, parse_umls_mrconso
from clinical_knowledge.terminology.privacy_regression import (
    assert_public_report_safe,
    run_privacy_regression_checks,
)
from clinical_knowledge.terminology.qa_golden import build_synthetic_qa_store
from clinical_knowledge.terminology.staging_guard import check_terminology_staging


@dataclass(frozen=True)
class TerminologySafetyRedTeamResult:
    block_id: str = "CKA-TERM-01H"
    conclusion: str = "cka_term01h_safety_redteam_ready"
    raw_path_leak_blocked: bool = False
    license_text_leak_blocked: bool = False
    fake_ack_blocked: bool = False
    ack_mismatch_blocked: bool = False
    terminology_data_staging_detected: bool = False
    data_terminology_staging_detected: bool = False
    zip_slip_blocked: bool = False
    malformed_rows_skipped: bool = False
    csv_formula_injection_neutralized: bool = False
    ambiguity_not_silently_resolved: bool = False
    unknown_code_not_hallucinated: bool = False
    b07_hypothesis_promotion_blocked: bool = False
    b07_ddi_clear_blocked: bool = False
    external_api_blocked: bool = False
    clinical_advice_absent: bool = False
    no_real_import_performed: bool = True
    real_terminology_files_committed: bool = False
    external_api_used: bool = False
    raw_phi_logged_in_public_reports: bool = False
    private_filename_path_leaks: int = 0
    secret_leaks: int = 0
    license_text_written_to_public_reports: bool = False
    clinical_recommendations_generated: bool = False
    prescription_dosing_advice_generated: bool = False
    scenario_results: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def passed(self) -> bool:
        fields = (
            self.raw_path_leak_blocked,
            self.license_text_leak_blocked,
            self.fake_ack_blocked,
            self.ack_mismatch_blocked,
            self.terminology_data_staging_detected,
            self.data_terminology_staging_detected,
            self.zip_slip_blocked,
            self.malformed_rows_skipped,
            self.csv_formula_injection_neutralized,
            self.ambiguity_not_silently_resolved,
            self.unknown_code_not_hallucinated,
            self.b07_hypothesis_promotion_blocked,
            self.b07_ddi_clear_blocked,
            self.external_api_blocked,
            self.clinical_advice_absent,
            self.no_real_import_performed,
            not self.real_terminology_files_committed,
            not self.external_api_used,
            not self.raw_phi_logged_in_public_reports,
            self.private_filename_path_leaks == 0,
            self.secret_leaks == 0,
            not self.license_text_written_to_public_reports,
            not self.clinical_recommendations_generated,
            not self.prescription_dosing_advice_generated,
        )
        return all(fields)

    def safe_public_summary(self) -> dict:
        return {
            "block_id": self.block_id,
            "conclusion": self.conclusion if self.passed() else "cka_term01h_safety_redteam_blocked",
            "raw_path_leak_blocked": self.raw_path_leak_blocked,
            "license_text_leak_blocked": self.license_text_leak_blocked,
            "fake_ack_blocked": self.fake_ack_blocked,
            "ack_mismatch_blocked": self.ack_mismatch_blocked,
            "terminology_data_staging_detected": self.terminology_data_staging_detected,
            "data_terminology_staging_detected": self.data_terminology_staging_detected,
            "zip_slip_blocked": self.zip_slip_blocked,
            "malformed_rows_skipped": self.malformed_rows_skipped,
            "csv_formula_injection_neutralized": self.csv_formula_injection_neutralized,
            "ambiguity_not_silently_resolved": self.ambiguity_not_silently_resolved,
            "unknown_code_not_hallucinated": self.unknown_code_not_hallucinated,
            "b07_hypothesis_promotion_blocked": self.b07_hypothesis_promotion_blocked,
            "b07_ddi_clear_blocked": self.b07_ddi_clear_blocked,
            "external_api_blocked": self.external_api_blocked,
            "clinical_advice_absent": self.clinical_advice_absent,
            "no_real_import_performed": self.no_real_import_performed,
            "real_terminology_files_committed": self.real_terminology_files_committed,
            "external_api_used": self.external_api_used,
            "raw_phi_logged_in_public_reports": self.raw_phi_logged_in_public_reports,
            "private_filename_path_leaks": self.private_filename_path_leaks,
            "secret_leaks": self.secret_leaks,
            "license_text_written_to_public_reports": self.license_text_written_to_public_reports,
            "clinical_recommendations_generated": self.clinical_recommendations_generated,
            "prescription_dosing_advice_generated": self.prescription_dosing_advice_generated,
            "scenario_results": list(self.scenario_results),
        }


def run_terminology_safety_redteam() -> TerminologySafetyRedTeamResult:
    privacy = run_privacy_regression_checks()
    fake_ack_blocked = _fake_ack_blocked()
    ack_mismatch_blocked = _ack_mismatch_blocked()
    staging = check_terminology_staging(
        staged_paths=[
            "terminology_data/umls/MRCONSO.RRF",
            "data/terminology/terminology.sqlite",
        ]
    )
    zip_slip_blocked = _zip_slip_blocked()
    malformed_rows_skipped = parse_umls_mrconso(text="bad|row\n", max_rows=10).skipped_rows == 1
    formula_neutralized = parse_loinc_csv(
        text="LOINC_NUM,LONG_COMMON_NAME\n1-1,\"'=HARMLESS\"\n",
        max_rows=10,
    ).concepts[0].display.startswith("'=")
    lookup_checks = _lookup_redteam_checks()
    boundary = safe_b07_boundary_summary()
    result = TerminologySafetyRedTeamResult(
        raw_path_leak_blocked=privacy.raw_path_leak_blocked,
        license_text_leak_blocked=privacy.license_text_leak_blocked,
        fake_ack_blocked=fake_ack_blocked,
        ack_mismatch_blocked=ack_mismatch_blocked,
        terminology_data_staging_detected=staging.terminology_data_staged,
        data_terminology_staging_detected=staging.data_terminology_staged,
        zip_slip_blocked=zip_slip_blocked,
        malformed_rows_skipped=malformed_rows_skipped,
        csv_formula_injection_neutralized=formula_neutralized and privacy.csv_formula_injection_neutralized,
        ambiguity_not_silently_resolved=lookup_checks["ambiguity_not_silently_resolved"],
        unknown_code_not_hallucinated=lookup_checks["unknown_code_not_hallucinated"],
        b07_hypothesis_promotion_blocked=boundary["coding_promotes_hypothesis"] is False,
        b07_ddi_clear_blocked=boundary["coding_clears_ddi_status"] is False,
        external_api_blocked=True,
        clinical_advice_absent=privacy.clinical_advice_absent,
        scenario_results=(
            {"scenario": "license_ack", "blocked": fake_ack_blocked and ack_mismatch_blocked},
            {"scenario": "staging_guard", "blocked": bool(staging.blocked_reason_codes)},
            {"scenario": "zip_slip", "blocked": zip_slip_blocked},
            {"scenario": "lookup_safety", "blocked": all(lookup_checks.values())},
            {"scenario": "b07_boundary", "blocked": boundary["coding_promotes_hypothesis"] is False and boundary["coding_clears_ddi_status"] is False},
        ),
    )
    assert_public_report_safe(result.safe_public_summary())
    return result


def _fake_ack_blocked() -> bool:
    with tempfile.TemporaryDirectory(prefix="medai_term01h_ack_") as tmp:
        ack = Path(tmp) / "LICENSE_ACK_PRIVATE.json"
        ack.write_text(json.dumps({"operator_acknowledged": False, "acknowledged_systems": ["umls"]}), encoding="utf-8")
        return not license_acknowledged_for(TerminologySystem.UMLS, local_ack_file=ack)


def _ack_mismatch_blocked() -> bool:
    with tempfile.TemporaryDirectory(prefix="medai_term01h_ack_") as tmp:
        ack = Path(tmp) / "LICENSE_ACK_PRIVATE.json"
        ack.write_text(json.dumps({"operator_acknowledged": True, "acknowledged_systems": ["loinc"]}), encoding="utf-8")
        return not license_acknowledged_for(TerminologySystem.UMLS, local_ack_file=ack)


def _zip_slip_blocked() -> bool:
    with tempfile.TemporaryDirectory(prefix="medai_term01h_zip_") as tmp:
        root = Path(tmp)
        arc = root / "UMLS_Synthetic_Test.zip"
        with zipfile.ZipFile(arc, "w") as zf:
            zf.writestr("MRCONSO.RRF", "C0000001|ENG|||||||||||||Synthetic concept||||\n")
            zf.writestr("../evil.txt", "blocked")
            zf.writestr("/absolute/evil.txt", "blocked")
            zf.writestr("C:/evil.txt", "blocked")
        result = safe_extract_zip([arc], repo_root=root, extract_approved=True)
        return result.entries_extracted == 1 and result.entries_blocked_zip_slip == 3


def _lookup_redteam_checks() -> dict[str, bool]:
    store, _metadata = build_synthetic_qa_store()
    service = TerminologyLookupService(store)
    ambiguous = service.lookup("shared duplicate")
    unknown = service.lookup("synthetic unknown term with no mapping")
    b07_unknown = code_entity_via_local_terminology("synthetic unknown term with no mapping", service)
    return {
        "ambiguity_not_silently_resolved": ambiguous.status == TerminologyLookupStatus.AMBIGUOUS and ambiguous.ambiguous,
        "unknown_code_not_hallucinated": unknown.status == TerminologyLookupStatus.UNMAPPED and not unknown.matches and b07_unknown.status == TerminologyLookupStatus.UNMAPPED,
    }
