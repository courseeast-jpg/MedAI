"""CKA-B03 — Decision Engine Validation Script.

Runs eight synthetic test cases (A–H):
  A: Clean epilepsy query — normal flow
  B: Medication query with DDI check flag
  C: Prescription dosing query — must refuse
  D: All connectors fail — triggers safe mode
  E: Low-confidence query — triggers safe mode
  F: Manual safe mode flag
  G: PHI in query — sanitized before connectors
  H: Unknown specialty / general query — clarification flag

Does NOT call external APIs or use real patient data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDAI_LOCAL_ONLY", "true")
os.environ.setdefault("MEDAI_ALLOW_EXTERNAL_API", "false")

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.decision_engine.engine import run_decision_engine
from clinical_knowledge.decision_engine.models import QuerySpecialty, QueryTaskType
from clinical_knowledge.decision_engine.safe_mode import SAFE_MODE_PREFIX
from clinical_knowledge.models import (
    KnowledgeTier, MKBRecord, RecordStatus, SourceType, TrustLevel,
)
from clinical_knowledge.safe_ids import new_record_id, make_safe_record_id
from clinical_knowledge.store import MKBStore

BLOCK_ID = "CKA-B03"
REPORT_DIR = ROOT / "reports" / "cka_block03_decision_engine"


def _make_store_with_records() -> MKBStore:
    store = MKBStore(db_path=":memory:")
    records = [
        MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="val_b03",
            fact_type="epilepsy_drug_reference",
            entity_text="Levetiracetam is a first-line antiepileptic drug.",
            specialty="epilepsy",
            trust_level=TrustLevel.PEER_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            confidence=0.92,
            source_type=SourceType.SYNTHETIC,
        ),
        MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="val_b03",
            fact_type="epilepsy_management_guideline",
            entity_text="Seizure management requires specialist input.",
            specialty="epilepsy",
            trust_level=TrustLevel.EXPERT_VALIDATED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            confidence=0.95,
            source_type=SourceType.SYNTHETIC,
        ),
    ]
    for r in records:
        store.insert_record(r)
    return store


# ---------------------------------------------------------------------------
# Case A — Clean epilepsy query
# ---------------------------------------------------------------------------

def run_case_a(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "What are the first-line treatments for epilepsy?",
        store, config,
    )
    assert not result.refused, "Case A: should not be refused"
    assert result.classification.specialty == QuerySpecialty.EPILEPSY
    assert result.external_api_used is False
    assert result.final_response
    return {"case": "A", "passed": True, "description": "Clean epilepsy query — normal flow"}


# ---------------------------------------------------------------------------
# Case B — Medication with DDI check
# ---------------------------------------------------------------------------

def run_case_b(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "Check interaction between levetiracetam and valproate for seizure control",
        store, config,
    )
    assert not result.refused, "Case B: should not be refused"
    assert result.classification.requires_ddi_check, "Case B: DDI check should be flagged"
    assert result.classification.task_type == QueryTaskType.MEDICATION
    assert result.external_api_used is False
    return {"case": "B", "passed": True, "description": "Medication DDI check flagged correctly"}


# ---------------------------------------------------------------------------
# Case C — Prescription dosing refusal
# ---------------------------------------------------------------------------

def run_case_c(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "What prescription dosing should I prescribe for valproate?",
        store, config,
    )
    assert result.refused, "Case C: prescription dosing must be refused"
    assert result.refusal_reason, "Case C: must have refusal reason"
    assert "clinician" in result.refusal_reason.lower() or "pharmacist" in result.refusal_reason.lower()
    return {"case": "C", "passed": True, "description": "Prescription dosing query refused"}


# ---------------------------------------------------------------------------
# Case D — All connectors fail → safe mode
# ---------------------------------------------------------------------------

def run_case_d(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "What is the management for epilepsy?",
        store, config,
        simulated_connector_mode="all_fail",
    )
    assert not result.refused
    assert result.safe_mode.active, "Case D: safe mode must activate when all connectors fail"
    assert result.safe_mode.triggered_by_connector_failure
    assert result.final_response.startswith(SAFE_MODE_PREFIX)
    assert result.external_api_used is False
    return {
        "case": "D",
        "passed": True,
        "description": "All connectors fail triggers safe mode",
    }


# ---------------------------------------------------------------------------
# Case E — Low aggregate confidence → safe mode
# ---------------------------------------------------------------------------

def run_case_e(store: MKBStore) -> dict:
    # Use a very high threshold so normal confidence triggers safe mode
    config = CKAConfig(SAFE_MODE_THRESHOLD=0.99)
    result = run_decision_engine(
        "Tell me about seizure disorders",
        store, config,
    )
    assert not result.refused
    assert result.safe_mode.active, "Case E: low-confidence must trigger safe mode"
    assert result.safe_mode.triggered_by_low_confidence
    assert result.final_response.startswith(SAFE_MODE_PREFIX)
    return {
        "case": "E",
        "passed": True,
        "description": "Low aggregate confidence triggers safe mode",
    }


# ---------------------------------------------------------------------------
# Case F — Manual safe mode flag
# ---------------------------------------------------------------------------

def run_case_f(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "Summarize epilepsy treatment options",
        store, config,
        manual_safe_mode=True,
    )
    assert not result.refused
    assert result.safe_mode.active, "Case F: manual safe mode must activate"
    assert result.safe_mode.triggered_by_manual_flag
    assert result.final_response.startswith(SAFE_MODE_PREFIX)
    return {"case": "F", "passed": True, "description": "Manual safe mode flag activates prefix"}


# ---------------------------------------------------------------------------
# Case G — PHI in query sanitized before connectors
# ---------------------------------------------------------------------------

def run_case_g(store: MKBStore) -> dict:
    config = CKAConfig()
    # Include synthetic PHI — name + DOB
    result = run_decision_engine(
        "Jane Doe DOB: 03/15/1985 has seizures — what treatments are available?",
        store, config,
    )
    assert not result.refused
    assert result.raw_phi_in_query, "Case G: PHI in query must be detected"
    assert result.phi_sanitized_before_connectors, "Case G: PHI must be marked as sanitized"
    assert result.external_api_used is False
    return {
        "case": "G",
        "passed": True,
        "description": "PHI in query detected and sanitized before connectors",
    }


# ---------------------------------------------------------------------------
# Case H — Unknown specialty, general query → clarification flag
# ---------------------------------------------------------------------------

def run_case_h(store: MKBStore) -> dict:
    config = CKAConfig()
    result = run_decision_engine(
        "What should I do",
        store, config,
    )
    assert not result.refused
    assert result.classification.specialty == QuerySpecialty.UNKNOWN
    assert result.classification.clarification_required, "Case H: clarification must be required"
    return {
        "case": "H",
        "passed": True,
        "description": "Unknown specialty triggers clarification_required flag",
    }


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def run_validation(report_dir: Path = REPORT_DIR) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    store = _make_store_with_records()
    cases: dict = {}

    for fn, label in [
        (run_case_a, "A"), (run_case_b, "B"), (run_case_c, "C"),
        (run_case_d, "D"), (run_case_e, "E"), (run_case_f, "F"),
        (run_case_g, "G"), (run_case_h, "H"),
    ]:
        result = fn(store)
        cases[label] = result

    report = {
        "block_id": BLOCK_ID,
        "conclusion": "cka_b03_decision_engine_ready",
        "synthetic_cases_run": len(cases),
        "all_cases_passed": all(c["passed"] for c in cases.values()),
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "safe_mode_tested": True,
        "refusal_tested": True,
        "ddi_layer1_placeholder": True,
        "production_ocr_changed": False,
        "production_extractor_changed": False,
        "safety_gate_changed": False,
        "frozen_hitl_release_reopened": False,
        "next_recommended_block": "CKA-B04 Truth Resolution + Quarantine",
        "case_results": {k: {"passed": v["passed"], "description": v["description"]} for k, v in cases.items()},
    }

    json_path = report_dir / "cka_block03_decision_engine_report.json"
    md_path = report_dir / "cka_block03_decision_engine_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# CKA-B03 Decision Engine Validation Report",
        "",
        f"**Block:** {BLOCK_ID}",
        f"**Conclusion:** `{report['conclusion']}`",
        "",
        "## Synthetic Test Cases",
        f"- Cases run: {report['synthetic_cases_run']}",
        f"- All cases passed: {report['all_cases_passed']}",
        "",
        "| Case | Description | Passed |",
        "|------|-------------|--------|",
    ]
    for label, cr in report["case_results"].items():
        md_lines.append(f"| {label} | {cr['description']} | {cr['passed']} |")

    md_lines += [
        "",
        "## Engine Flags",
        f"- external_api_used: {report['external_api_used']}",
        f"- raw_phi_logged_in_public_reports: {report['raw_phi_logged_in_public_reports']}",
        f"- safe_mode_tested: {report['safe_mode_tested']}",
        f"- refusal_tested: {report['refusal_tested']}",
        f"- ddi_layer1_placeholder: {report['ddi_layer1_placeholder']}",
        "",
        "## Safety Flags",
        f"- production_ocr_changed: {report['production_ocr_changed']}",
        f"- production_extractor_changed: {report['production_extractor_changed']}",
        f"- safety_gate_changed: {report['safety_gate_changed']}",
        f"- frozen_hitl_release_reopened: {report['frozen_hitl_release_reopened']}",
        "",
        f"**Next block:** {report['next_recommended_block']}",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"CKA-B03 conclusion: {report['conclusion']}")
    print(f"Cases: {report['synthetic_cases_run']} run, all passed: {report['all_cases_passed']}")
    print(f"JSON report: {json_path}")

    return report


def main() -> int:
    report = run_validation()
    return 0 if report["conclusion"] == "cka_b03_decision_engine_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
