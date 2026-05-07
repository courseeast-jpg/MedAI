"""CKA-TERM-01 — local-only, license-gated terminology data layer.

This package provides:
- inventory of operator-supplied terminology files (no network)
- license-gate helpers (operator must explicitly acknowledge license)
- streaming parsers for synthetic / small UMLS / SNOMED / RxNorm / LOINC fixtures
- a small SQLite terminology index abstraction (temp by default)
- a lookup service (no code hallucination, no clinical interpretation)
- a narrow integration helper for CKA-B07 medical_coding

It does NOT download data, does NOT bypass licensing, does NOT change
clinical logic, and does NOT promote hypothesis facts.
"""
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologyImportMode,
    TerminologyLookupResult,
    TerminologyLookupStatus,
    TerminologySourceManifest,
    TerminologySourceStatus,
    TerminologySystem,
)
from clinical_knowledge.terminology.license_gate import (
    LicenseGateError,
    license_acknowledged_for,
    require_license_acknowledgment,
    verify_operator_license_acknowledgment,
)
from clinical_knowledge.terminology.file_inventory import (
    InventoryReport,
    inventory_terminology_data_dir,
)
from clinical_knowledge.terminology.parsers import (
    ParseResult,
    parse_loinc_csv,
    parse_rxnorm_rxnconso,
    parse_snomed_concept_description,
    parse_umls_mrconso,
)
from clinical_knowledge.terminology.local_store import (
    LocalTerminologyStore,
)
from clinical_knowledge.terminology.lookup_service import (
    TerminologyLookupService,
)
from clinical_knowledge.terminology.integration import (
    code_entity_via_local_terminology,
    safe_b07_boundary_summary,
)


__all__ = [
    "TerminologyConcept",
    "TerminologyImportMode",
    "TerminologyLookupResult",
    "TerminologyLookupStatus",
    "TerminologySourceManifest",
    "TerminologySourceStatus",
    "TerminologySystem",
    "LicenseGateError",
    "license_acknowledged_for",
    "require_license_acknowledgment",
    "verify_operator_license_acknowledgment",
    "InventoryReport",
    "inventory_terminology_data_dir",
    "ParseResult",
    "parse_loinc_csv",
    "parse_rxnorm_rxnconso",
    "parse_snomed_concept_description",
    "parse_umls_mrconso",
    "LocalTerminologyStore",
    "TerminologyLookupService",
    "code_entity_via_local_terminology",
    "safe_b07_boundary_summary",
]
