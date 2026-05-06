"""CKA-B07 Medical Coding / SNOMED-UMLS Interface — test suite.

All tests use synthetic data only. No real codes. No external APIs.
No UMLS/SNOMED license required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clinical_knowledge.medical_coding.integration import (
    apply_coding_result_to_record,
    code_entity,
    coding_candidate_from_mkb_record,
    write_coding_ledger_event,
)
from clinical_knowledge.medical_coding.local_lookup import (
    LocalLookupTerminologySource,
    load_local_lookup,
)
from clinical_knowledge.medical_coding.models import (
    CodingCandidate,
    CodingResult,
    CodingStatus,
    CodingSystem,
    CodingValidationResult,
    MedicalCode,
    TerminologySourceStatus,
)
from clinical_knowledge.medical_coding.synthetic_mapper import SyntheticTerminologySource
from clinical_knowledge.medical_coding.validator import validate_code
from clinical_knowledge.models import (
    DDIStatus,
    KnowledgeTier,
    LedgerEventType,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
from clinical_knowledge.store import MKBStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_src():
    return SyntheticTerminologySource()


@pytest.fixture
def store():
    return MKBStore(db_path=":memory:")


@pytest.fixture
def active_record():
    rid = new_record_id()
    return MKBRecord(
        record_id=rid,
        safe_record_id=make_safe_record_id(rid),
        session_id="b07_test",
        fact_type="diagnosis",
        entity_text="synthetic condition alpha",
        trust_level=TrustLevel.OPERATOR_REVIEWED,
        tier=KnowledgeTier.ACTIVE,
        status=RecordStatus.CONFIRMED,
        source_type=SourceType.SYNTHETIC,
        confidence=0.80,
    )


@pytest.fixture
def hypothesis_record():
    rid = new_record_id()
    return MKBRecord(
        record_id=rid,
        safe_record_id=make_safe_record_id(rid),
        session_id="b07_test",
        fact_type="diagnosis",
        entity_text="synthetic condition alpha",
        trust_level=TrustLevel.OPERATOR_REVIEWED,
        tier=KnowledgeTier.HYPOTHESIS,
        status=RecordStatus.PENDING,
        source_type=SourceType.SYNTHETIC,
        confidence=0.60,
    )


@pytest.fixture
def tmp_lookup_file(tmp_path):
    """Create a temporary synthetic local lookup JSON file."""
    data = {
        "entries": [
            {
                "normalized_text": "synthetic condition local",
                "fact_type": "diagnosis",
                "system": "snomed_ct",
                "code": "SYNTHETIC-SNOMED-001",
                "display": "Synthetic condition local",
                "version": "test-only",
                "synthetic": True,
            }
        ]
    }
    p = tmp_path / "test_lookup.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_ambiguous_file(tmp_path):
    data = {
        "entries": [
            {
                "normalized_text": "synthetic ambiguous term",
                "fact_type": "diagnosis",
                "system": "snomed_ct",
                "code": "SYNTHETIC-AMB-A",
                "display": "Ambiguous A",
                "version": "test-only",
                "synthetic": True,
            },
            {
                "normalized_text": "synthetic ambiguous term",
                "fact_type": "diagnosis",
                "system": "snomed_ct",
                "code": "SYNTHETIC-AMB-B",
                "display": "Ambiguous B",
                "version": "test-only",
                "synthetic": True,
            },
        ]
    }
    p = tmp_path / "ambiguous_lookup.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Enum / model tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_coding_system_values(self):
        assert CodingSystem.SYNTHETIC == "synthetic"
        assert CodingSystem.UMLS == "umls"
        assert CodingSystem.SNOMED_CT == "snomed_ct"
        assert CodingSystem.RXNORM == "rxnorm"
        assert CodingSystem.LOINC == "loinc"
        assert CodingSystem.UNKNOWN == "unknown"

    def test_coding_status_values(self):
        assert CodingStatus.CODED == "coded"
        assert CodingStatus.UNMAPPED == "unmapped"
        assert CodingStatus.AMBIGUOUS == "ambiguous"
        assert CodingStatus.CODING_UNAVAILABLE == "coding_unavailable"
        assert CodingStatus.INVALID_CODE == "invalid_code"
        assert CodingStatus.SOURCE_UNAVAILABLE == "source_unavailable"

    def test_terminology_source_status_values(self):
        assert TerminologySourceStatus.AVAILABLE == "available"
        assert TerminologySourceStatus.UNAVAILABLE == "unavailable"
        assert TerminologySourceStatus.STUB_ONLY == "stub_only"
        assert TerminologySourceStatus.LOCAL_LOOKUP_ONLY == "local_lookup_only"

    def test_medical_coding_ledger_type_exists(self):
        assert LedgerEventType.MEDICAL_CODING == "medical_coding"

    def test_medical_coding_not_reserved(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES
        assert LedgerEventType.MEDICAL_CODING not in _RESERVED_EVENT_TYPES


class TestMedicalCodeModel:
    def test_valid_synthetic_code(self):
        code = MedicalCode(
            system=CodingSystem.SYNTHETIC,
            code="SYN-DX-001",
            display="Synthetic Dx",
            synthetic=True,
        )
        assert code.system == CodingSystem.SYNTHETIC
        assert code.synthetic is True
        assert "safe_code_id" in code.safe_public_summary
        assert code.safe_public_summary["safe_code_id"].startswith("cka_code_")

    def test_confidence_range_rejected(self):
        with pytest.raises(ValueError):
            MedicalCode(
                system=CodingSystem.SYNTHETIC,
                code="X",
                display="X",
                synthetic=True,
                confidence=1.5,
            )

    def test_safe_public_summary_no_raw_source(self):
        code = MedicalCode(
            system=CodingSystem.SYNTHETIC,
            code="SYN-DX-001",
            display="Synthetic Dx",
            source="synthetic_stub",
            synthetic=True,
        )
        # Source is not exposed in safe_public_summary
        assert "source" not in code.safe_public_summary


# ---------------------------------------------------------------------------
# Synthetic mapper tests
# ---------------------------------------------------------------------------

class TestSyntheticMapper:
    def test_known_term_mapped(self, synth_src):
        result = synth_src.lookup("synthetic condition alpha")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code is not None
        assert result.preferred_code.code == "SYN-DX-001"
        assert result.preferred_code.system == CodingSystem.SYNTHETIC
        assert result.preferred_code.synthetic is True

    def test_known_med_term_mapped(self, synth_src):
        result = synth_src.lookup("synthetic medication beta")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.code == "SYN-RX-001"

    def test_known_lab_term_mapped(self, synth_src):
        result = synth_src.lookup("synthetic lab gamma")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.code == "SYN-LAB-001"

    def test_known_procedure_term_mapped(self, synth_src):
        result = synth_src.lookup("synthetic procedure delta")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.code == "SYN-PROC-001"

    def test_unknown_term_unmapped(self, synth_src):
        result = synth_src.lookup("synthetic unknown condition xyz")
        assert result.status == CodingStatus.UNMAPPED
        assert result.codes == []
        assert result.preferred_code is None

    def test_no_code_hallucinated_for_unknown(self, synth_src):
        result = synth_src.lookup("this entity does not exist in any table")
        assert result.no_code_hallucinated is True
        assert result.codes == []

    def test_case_insensitive_normalisation(self, synth_src):
        result = synth_src.lookup("SYNTHETIC CONDITION ALPHA")
        assert result.status == CodingStatus.CODED

    def test_status_is_stub_only(self, synth_src):
        assert synth_src.status() == TerminologySourceStatus.STUB_ONLY

    def test_all_codes_marked_synthetic(self, synth_src):
        for term in [
            "synthetic condition alpha",
            "synthetic medication beta",
            "synthetic lab gamma",
            "synthetic procedure delta",
        ]:
            result = synth_src.lookup(term)
            assert result.preferred_code.synthetic is True

    def test_no_real_clinical_assertions(self, synth_src):
        # Display strings must not claim real SNOMED/UMLS/RxNorm codes
        for term in ["synthetic condition alpha", "synthetic lab gamma"]:
            result = synth_src.lookup(term)
            display = result.preferred_code.display.lower()
            assert "test only" in display or "(test only)" in display


# ---------------------------------------------------------------------------
# Local lookup tests
# ---------------------------------------------------------------------------

class TestLocalLookup:
    def test_load_valid_file(self, tmp_lookup_file):
        entries = load_local_lookup(tmp_lookup_file)
        assert len(entries) == 1
        assert entries[0]["code"] == "SYNTHETIC-SNOMED-001"
        assert entries[0]["synthetic"] is True

    def test_missing_file_returns_empty(self):
        entries = load_local_lookup("/nonexistent/path.json")
        assert entries == []

    def test_non_synthetic_entry_skipped(self, tmp_path):
        data = {
            "entries": [
                {
                    "normalized_text": "real term",
                    "system": "snomed_ct",
                    "code": "12345678",
                    "display": "Real Term",
                    "synthetic": False,
                }
            ]
        }
        p = tmp_path / "non_synthetic.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        entries = load_local_lookup(str(p))
        assert entries == []

    def test_local_source_maps_entry(self, tmp_lookup_file):
        src = LocalLookupTerminologySource(tmp_lookup_file)
        result = src.lookup("synthetic condition local")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.system == CodingSystem.SNOMED_CT
        assert result.preferred_code.synthetic is True
        assert result.preferred_code.version == "test-only"

    def test_local_source_unmapped_for_missing_term(self, tmp_lookup_file):
        src = LocalLookupTerminologySource(tmp_lookup_file)
        result = src.lookup("synthetic unknown condition")
        assert result.status == CodingStatus.UNMAPPED
        assert result.codes == []

    def test_missing_file_source_unavailable(self):
        src = LocalLookupTerminologySource("/nonexistent/path.json")
        result = src.lookup("any term")
        assert result.status == CodingStatus.SOURCE_UNAVAILABLE
        assert src.status() == TerminologySourceStatus.UNAVAILABLE

    def test_ambiguous_returns_ambiguous(self, tmp_ambiguous_file):
        src = LocalLookupTerminologySource(tmp_ambiguous_file)
        result = src.lookup("synthetic ambiguous term")
        assert result.status == CodingStatus.AMBIGUOUS
        assert result.preferred_code is None
        assert result.ambiguity_count == 2

    def test_preferred_flag_resolves_ambiguity(self, tmp_path):
        data = {
            "entries": [
                {
                    "normalized_text": "synthetic preferred term",
                    "system": "snomed_ct",
                    "code": "SYN-PREF-A",
                    "display": "Pref A",
                    "version": "test-only",
                    "synthetic": True,
                    "preferred": True,
                },
                {
                    "normalized_text": "synthetic preferred term",
                    "system": "snomed_ct",
                    "code": "SYN-PREF-B",
                    "display": "Pref B",
                    "version": "test-only",
                    "synthetic": True,
                },
            ]
        }
        p = tmp_path / "preferred.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        src = LocalLookupTerminologySource(str(p))
        result = src.lookup("synthetic preferred term")
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.code == "SYN-PREF-A"

    def test_source_name_uses_hash_not_raw_path(self, tmp_lookup_file):
        src = LocalLookupTerminologySource(tmp_lookup_file)
        assert tmp_lookup_file not in src.name
        assert "cka_lkp_" in src.name

    def test_raw_path_not_in_lookup_result(self, tmp_lookup_file):
        src = LocalLookupTerminologySource(tmp_lookup_file)
        result = src.lookup("synthetic condition local")
        # Raw path must not appear in public summary or preferred_code source
        result_str = json.dumps(result.safe_public_summary)
        assert tmp_lookup_file not in result_str
        if result.preferred_code:
            assert tmp_lookup_file not in result.preferred_code.source


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def test_valid_synthetic_code_passes(self):
        code = MedicalCode(system=CodingSystem.SYNTHETIC, code="SYN-DX-001",
                           display="Synth", synthetic=True)
        result = validate_code(code)
        assert result.valid is True
        assert result.status == CodingStatus.CODED

    def test_empty_code_rejected(self):
        code = MedicalCode(system=CodingSystem.SYNTHETIC, code="",
                           display="Empty", synthetic=True)
        result = validate_code(code)
        assert result.valid is False
        assert result.status == CodingStatus.INVALID_CODE
        assert any("empty" in r for r in result.invalid_reasons)

    def test_unknown_system_rejected(self):
        code = MedicalCode(system=CodingSystem.UNKNOWN, code="X001",
                           display="Unknown sys", synthetic=True)
        result = validate_code(code)
        assert result.valid is False
        assert result.status == CodingStatus.INVALID_CODE

    def test_synthetic_system_synthetic_false_rejected(self):
        code = MedicalCode(system=CodingSystem.SYNTHETIC, code="SYN-999",
                           display="Bad", synthetic=False)
        result = validate_code(code)
        assert result.valid is False
        assert result.status == CodingStatus.INVALID_CODE

    def test_local_lookup_code_valid(self):
        # Code from a local_lookup source with synthetic=True is valid
        code = MedicalCode(
            system=CodingSystem.SNOMED_CT,
            code="SYNTHETIC-SNOMED-001",
            display="Synthetic SNOMED",
            source="local_lookup:cka_lkp_abc123",
            synthetic=True,
        )
        result = validate_code(code)
        assert result.valid is True

    def test_non_synthetic_from_unverified_source_rejected(self):
        code = MedicalCode(
            system=CodingSystem.SNOMED_CT,
            code="12345678",
            display="Real snomed claim",
            source="some_unknown_source",
            synthetic=False,
        )
        result = validate_code(code)
        assert result.valid is False

    def test_validation_result_safe_public_summary(self):
        code = MedicalCode(system=CodingSystem.SYNTHETIC, code="SYN-DX-001",
                           display="Synth", synthetic=True)
        result = validate_code(code)
        assert "valid" in result.safe_public_summary
        assert "patient" not in str(result.safe_public_summary).lower()


# ---------------------------------------------------------------------------
# code_entity / coding service tests
# ---------------------------------------------------------------------------

class TestCodingService:
    def test_code_entity_returns_coded(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert result.status == CodingStatus.CODED
        assert result.no_code_hallucinated is True

    def test_code_entity_unknown_returns_unmapped(self, synth_src):
        rid = new_record_id()
        rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="diagnosis",
            entity_text="this entity does not exist",
            trust_level=TrustLevel.UNVERIFIED, tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED, source_type=SourceType.SYNTHETIC,
        )
        cand = coding_candidate_from_mkb_record(rec)
        result = code_entity(cand, [synth_src])
        assert result.status == CodingStatus.UNMAPPED
        assert result.codes == []
        assert result.no_code_hallucinated is True

    def test_code_entity_all_unavailable_returns_coding_unavailable(self):
        src = LocalLookupTerminologySource("/nonexistent/path.json")
        rid = new_record_id()
        rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            trust_level=TrustLevel.UNVERIFIED, tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED, source_type=SourceType.SYNTHETIC,
        )
        cand = coding_candidate_from_mkb_record(rec)
        result = code_entity(cand, [src])
        assert result.status in (CodingStatus.CODING_UNAVAILABLE, CodingStatus.SOURCE_UNAVAILABLE)

    def test_code_entity_no_sources_returns_coding_unavailable(self, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [])
        assert result.status == CodingStatus.CODING_UNAVAILABLE

    def test_code_entity_queries_sources_in_order(self, tmp_path, synth_src):
        # Put a local lookup first that has no match; synth source should then match
        data = {"entries": []}
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        local_src = LocalLookupTerminologySource(str(p))

        rid = new_record_id()
        rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="diagnosis",
            entity_text="synthetic condition alpha",
            trust_level=TrustLevel.UNVERIFIED, tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED, source_type=SourceType.SYNTHETIC,
        )
        cand = coding_candidate_from_mkb_record(rec)
        # local_src (empty) first, then synth_src
        result = code_entity(cand, [local_src, synth_src])
        assert result.status == CodingStatus.CODED
        assert result.preferred_code.code == "SYN-DX-001"

    def test_code_entity_ambiguous_returned_immediately(self, tmp_ambiguous_file, synth_src):
        src = LocalLookupTerminologySource(tmp_ambiguous_file)
        rid = new_record_id()
        rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="diagnosis",
            entity_text="synthetic ambiguous term",
            trust_level=TrustLevel.UNVERIFIED, tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED, source_type=SourceType.SYNTHETIC,
        )
        cand = coding_candidate_from_mkb_record(rec)
        result = code_entity(cand, [src, synth_src])
        assert result.status == CodingStatus.AMBIGUOUS

    def test_no_external_api_required(self, synth_src, active_record):
        # code_entity must work with zero network access — pure local
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert result is not None  # no exception, no network dependency


# ---------------------------------------------------------------------------
# Integration: coding_candidate_from_mkb_record
# ---------------------------------------------------------------------------

class TestCodingCandidateFromRecord:
    def test_safe_candidate_id_uses_hash(self, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        assert cand.safe_candidate_id.startswith("cka_cc_")
        assert active_record.record_id not in cand.safe_candidate_id

    def test_raw_record_id_not_in_public_summary(self, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        assert active_record.record_id not in str(cand.safe_public_summary)

    def test_entity_text_normalised(self, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        assert cand.normalized_text == cand.entity_text.lower().strip()

    def test_tier_preserved_in_source_tier(self, hypothesis_record):
        cand = coding_candidate_from_mkb_record(hypothesis_record)
        assert cand.source_tier == KnowledgeTier.HYPOTHESIS.value


# ---------------------------------------------------------------------------
# Integration: apply_coding_result_to_record
# ---------------------------------------------------------------------------

class TestApplyCodingResult:
    def test_active_tier_preserved(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(active_record, result)
        assert active_record.tier == KnowledgeTier.ACTIVE

    def test_active_status_preserved(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(active_record, result)
        assert active_record.status == RecordStatus.CONFIRMED

    def test_hypothesis_tier_preserved(self, synth_src, hypothesis_record):
        cand = coding_candidate_from_mkb_record(hypothesis_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(hypothesis_record, result)
        assert hypothesis_record.tier == KnowledgeTier.HYPOTHESIS

    def test_hypothesis_not_promoted_after_coding(self, synth_src, hypothesis_record):
        cand = coding_candidate_from_mkb_record(hypothesis_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(hypothesis_record, result)
        assert hypothesis_record.tier == KnowledgeTier.HYPOTHESIS
        assert hypothesis_record.status == RecordStatus.PENDING

    def test_coding_metadata_added_to_structured(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(active_record, result)
        assert "coding" in active_record.structured
        assert active_record.structured["coding"]["coding_status"] == "coded"

    def test_ddi_status_not_cleared(self, synth_src):
        rid = new_record_id()
        med_rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="medication",
            entity_text="synthetic medication beta",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.HYPOTHESIS,
            status=RecordStatus.PENDING,
            source_type=SourceType.SYNTHETIC,
            ddi_checked=True,
            ddi_status=DDIStatus.BLOCKED,
        )
        original_ddi = med_rec.ddi_status
        cand = coding_candidate_from_mkb_record(med_rec)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(med_rec, result)
        assert med_rec.ddi_status == original_ddi

    def test_unmapped_coding_applied_cleanly(self, synth_src):
        rid = new_record_id()
        rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="diagnosis",
            entity_text="synthetic unknown condition xyz",
            trust_level=TrustLevel.UNVERIFIED, tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED, source_type=SourceType.SYNTHETIC,
        )
        cand = coding_candidate_from_mkb_record(rec)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(rec, result)
        assert rec.structured["coding"]["coding_status"] == "unmapped"
        assert rec.tier == KnowledgeTier.ACTIVE  # unchanged


# ---------------------------------------------------------------------------
# Ledger event tests
# ---------------------------------------------------------------------------

class TestLedgerEvents:
    def test_medical_coding_event_written(self, synth_src, active_record, store):
        store.insert_record(active_record)
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        write_coding_ledger_event(active_record, result, store, ["synthetic_stub"])
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.MEDICAL_CODING.value in types

    def test_coding_event_safe_public_details(self, synth_src, active_record, store):
        store.insert_record(active_record)
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        write_coding_ledger_event(active_record, result, store, ["synthetic_stub"])
        events = [e for e in store.read_ledger_events()
                  if e["event_type"] == LedgerEventType.MEDICAL_CODING.value]
        assert events
        details = events[0].get("safe_public_details", "{}")
        if isinstance(details, str):
            details = json.loads(details)
        assert "safe_record_id" in details
        assert active_record.record_id not in str(details)
        # No raw PHI, no raw path
        assert "patient" not in str(details).lower()

    def test_make_medical_coding_event_helper(self):
        from clinical_knowledge.ledger import make_medical_coding_event
        rid = new_record_id()
        evt = make_medical_coding_event(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            coding_status="coded",
            systems_attempted=["synthetic_stub"],
            preferred_code_summary={"system": "synthetic", "synthetic": True},
        )
        assert evt.event_type == LedgerEventType.MEDICAL_CODING
        assert rid not in str(evt.safe_public_details)  # safe_record_id used, not raw


# ---------------------------------------------------------------------------
# Safety boundary tests
# ---------------------------------------------------------------------------

class TestSafetyBoundaries:
    def test_no_scispacy_or_umls_required(self, synth_src, active_record):
        # Must work without any optional NLP dependency
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert result is not None

    def test_no_clinical_recommendation_in_results(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        summary_str = str(result.safe_public_summary).lower()
        assert "recommend" not in summary_str
        assert "prescri" not in summary_str

    def test_no_prescription_dosing_advice_in_explanation(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert "prescri" not in result.explanation.lower()
        assert "dosing" not in result.explanation.lower()

    def test_coding_does_not_bypass_truth_resolution(self, synth_src, active_record):
        # Coding must not modify tier or status — TR decisions are preserved
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(active_record, result)
        assert active_record.tier == KnowledgeTier.ACTIVE
        assert active_record.status == RecordStatus.CONFIRMED

    def test_coding_does_not_bypass_medication_safety_gate(self, synth_src):
        rid = new_record_id()
        med_rec = MKBRecord(
            record_id=rid, safe_record_id=make_safe_record_id(rid),
            session_id="t", fact_type="medication",
            entity_text="synthetic medication beta",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.HYPOTHESIS,
            status=RecordStatus.PENDING,
            source_type=SourceType.SYNTHETIC,
            ddi_status=DDIStatus.BLOCKED,
        )
        cand = coding_candidate_from_mkb_record(med_rec)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(med_rec, result)
        # Coding must not override ddi_status
        assert med_rec.ddi_status == DDIStatus.BLOCKED

    def test_synthetic_codes_not_claimed_as_real_snomed(self, synth_src, active_record):
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert result.preferred_code.system == CodingSystem.SYNTHETIC
        assert result.preferred_code.synthetic is True

    def test_no_real_api_calls_required(self, synth_src, active_record):
        # All coding must work offline/locally
        cand = coding_candidate_from_mkb_record(active_record)
        result = code_entity(cand, [synth_src])
        assert result.status == CodingStatus.CODED


# ---------------------------------------------------------------------------
# Enrichment integration
# ---------------------------------------------------------------------------

class TestEnrichmentIntegration:
    def test_enrichment_candidate_coding_preserves_hypothesis(self, synth_src, tmp_path):
        from clinical_knowledge.config import CKAConfig
        from clinical_knowledge.enrichment.candidate_extractor import (
            extract_enrichment_candidates_from_structured_response,
        )
        from clinical_knowledge.enrichment.enrichment_queue import EnrichmentQueue
        from clinical_knowledge.enrichment.integration import process_enrichment_candidate

        store = MKBStore(db_path=":memory:")
        queue = EnrichmentQueue()
        config = CKAConfig()
        payload = {
            "source_name": "dxgpt_stub",
            "source_kind": "ai_response",
            "specialty": "epilepsy",
            "facts": [{"fact_type": "diagnosis",
                        "entity_text": "synthetic condition alpha",
                        "confidence": 0.72, "structured": {}}],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        enrich_result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert enrich_result.written_record is not None
        written = enrich_result.written_record
        assert written.tier == KnowledgeTier.HYPOTHESIS

        cand = coding_candidate_from_mkb_record(written)
        result = code_entity(cand, [synth_src])
        apply_coding_result_to_record(written, result)

        assert written.tier == KnowledgeTier.HYPOTHESIS
        assert "coding" in written.structured


# ---------------------------------------------------------------------------
# Validation script tests
# ---------------------------------------------------------------------------

class TestValidationScript:
    def test_validation_script_succeeds(self, tmp_path):
        from scripts.run_cka_block07_medical_coding_validation import run_validation
        report = run_validation(report_dir=tmp_path)
        assert report["conclusion"] == "cka_b07_medical_coding_ready"
        assert report["all_cases_passed"] is True
        assert report["synthetic_cases_run"] == 11
        assert report["no_code_hallucinated"] is True
        assert report["unknown_entities_remain_unmapped"] is True
        assert report["real_umls_api_used"] is False
        assert report["real_snomed_download_used"] is False
        assert report["real_scispacy_linker_required"] is False
        assert report["coding_does_not_promote_hypothesis"] is True
        assert report["coding_does_not_clear_ddi_status"] is True
        assert report["external_api_used"] is False
        assert report["frozen_hitl_release_reopened"] is False

    def test_final_report_no_private_strings(self, tmp_path):
        from scripts.run_cka_block07_medical_coding_validation import run_validation
        report = run_validation(report_dir=tmp_path)
        check = check_public_report_payload(report)
        assert check.passed
        assert not check.raw_phi_logged_in_public_reports
        assert check.private_filename_path_leaks == 0
        assert check.secret_leaks == 0
