"""CKA-TERM-01D synthetic terminology QA golden cases."""
from __future__ import annotations

from dataclasses import dataclass, field

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologyLookupStatus,
    TerminologySystem,
)


@dataclass(frozen=True)
class TerminologyGoldenCase:
    case_id: str
    query: str
    expected_status: TerminologyLookupStatus
    expected_codes: tuple[str, ...] = ()
    systems: tuple[TerminologySystem, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    def safe_public_summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "expected_status": self.expected_status.value,
            "expected_codes_count": len(self.expected_codes),
            "systems": [system.value for system in self.systems],
            "tags": list(self.tags),
        }


def synthetic_golden_cases() -> list[TerminologyGoldenCase]:
    return [
        TerminologyGoldenCase(
            case_id="exact_umls_hypertension",
            query="hypertension",
            expected_status=TerminologyLookupStatus.EXACT,
            expected_codes=("UMLS001",),
            systems=(TerminologySystem.UMLS,),
            tags=("exact_match", "umls"),
        ),
        TerminologyGoldenCase(
            case_id="synonym_umls_high_bp",
            query="high blood pressure",
            expected_status=TerminologyLookupStatus.SYNONYM,
            expected_codes=("UMLS001",),
            systems=(TerminologySystem.UMLS,),
            tags=("synonym_match", "umls"),
        ),
        TerminologyGoldenCase(
            case_id="ambiguous_aspirin_cross_system",
            query="aspirin",
            expected_status=TerminologyLookupStatus.AMBIGUOUS,
            expected_codes=("SNOMED100", "RXN001"),
            tags=("ambiguous_match", "multi_system_duplicate"),
        ),
        TerminologyGoldenCase(
            case_id="unmapped_unknown_no_code",
            query="not a synthetic terminology term",
            expected_status=TerminologyLookupStatus.UNMAPPED,
            expected_codes=(),
            tags=("unmapped_term",),
        ),
        TerminologyGoldenCase(
            case_id="duplicate_term_multi_system",
            query="shared duplicate",
            expected_status=TerminologyLookupStatus.AMBIGUOUS,
            expected_codes=("UMLS777", "LOINC777"),
            tags=("multi_system_duplicate", "ambiguous_match"),
        ),
        TerminologyGoldenCase(
            case_id="malformed_code_skipped",
            query="malformed skipped",
            expected_status=TerminologyLookupStatus.UNMAPPED,
            expected_codes=(),
            tags=("malformed_code_skipped", "unmapped_term"),
        ),
        TerminologyGoldenCase(
            case_id="inactive_concept_excluded",
            query="inactive hidden",
            expected_status=TerminologyLookupStatus.UNMAPPED,
            expected_codes=(),
            tags=("inactive_concept_excluded", "unmapped_term"),
        ),
    ]


def build_synthetic_qa_store() -> tuple[LocalTerminologyStore, dict]:
    store = LocalTerminologyStore()
    source_ids = {
        TerminologySystem.UMLS: store.register_source(
            TerminologySystem.UMLS,
            safe_source_id="term_qa_src_umls",
            license_confirmed=True,
        ),
        TerminologySystem.SNOMED_CT: store.register_source(
            TerminologySystem.SNOMED_CT,
            safe_source_id="term_qa_src_snomed",
            license_confirmed=True,
        ),
        TerminologySystem.RXNORM: store.register_source(
            TerminologySystem.RXNORM,
            safe_source_id="term_qa_src_rxnorm",
            license_confirmed=True,
        ),
        TerminologySystem.LOINC: store.register_source(
            TerminologySystem.LOINC,
            safe_source_id="term_qa_src_loinc",
            license_confirmed=True,
        ),
    }
    concepts = [
        TerminologyConcept.synthetic_for(
            TerminologySystem.UMLS,
            "UMLS001",
            "hypertension",
            synonyms=["high blood pressure"],
            source_safe_id=source_ids[TerminologySystem.UMLS],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.SNOMED_CT,
            "SNOMED100",
            "aspirin",
            source_safe_id=source_ids[TerminologySystem.SNOMED_CT],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.RXNORM,
            "RXN001",
            "aspirin",
            source_safe_id=source_ids[TerminologySystem.RXNORM],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.UMLS,
            "UMLS777",
            "shared duplicate",
            source_safe_id=source_ids[TerminologySystem.UMLS],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.LOINC,
            "LOINC777",
            "shared duplicate",
            source_safe_id=source_ids[TerminologySystem.LOINC],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.LOINC,
            "LOINC001",
            "glucose synthetic lab",
            source_safe_id=source_ids[TerminologySystem.LOINC],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.RXNORM,
            "bad code",
            "malformed skipped",
            source_safe_id=source_ids[TerminologySystem.RXNORM],
        ),
        TerminologyConcept.synthetic_for(
            TerminologySystem.SNOMED_CT,
            "SNOMED999",
            "inactive hidden",
            source_safe_id=source_ids[TerminologySystem.SNOMED_CT],
        ),
    ]
    concepts[-1].active = False

    malformed_skipped = 0
    for concept in concepts:
        if not _valid_synthetic_code(concept.code):
            malformed_skipped += 1
            continue
        store.add_concepts([concept], source_id=concept.source_safe_id)

    metadata = {
        "systems_loaded": [system.value for system in source_ids],
        "concepts_loaded": len(concepts) - malformed_skipped,
        "malformed_codes_skipped": malformed_skipped,
        "inactive_concepts_loaded": 1,
        "real_terminology_imported": False,
        "external_api_used": False,
    }
    return store, metadata


def _valid_synthetic_code(code: str) -> bool:
    return bool(code) and " " not in code and code.upper() == code
