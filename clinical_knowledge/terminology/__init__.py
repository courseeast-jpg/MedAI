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
from clinical_knowledge.terminology.file_classifier import (
    ClassificationSummary,
    FileClassification,
    classify_filename,
    classify_filenames,
)
from clinical_knowledge.terminology.ack_template import (
    TemplateWriteResult,
    real_ack_filename,
    template_filename,
    template_payload,
    write_ack_template,
)
from clinical_knowledge.terminology.intake_automation import (
    CopyResult,
    ExtractResult,
    FolderPreparationResult,
    LocalScanResult,
    ReadinessReport,
    compute_readiness,
    copy_classified_files,
    optional_local_scan,
    prepare_intake_folders,
    safe_extract_zip,
)
from clinical_knowledge.terminology.import_limits import (
    TerminologyImportLimits,
    build_import_limits,
)
from clinical_knowledge.terminology.import_checkpoint import (
    TerminologyImportCheckpoint,
    simulate_checkpoint_resume,
)
from clinical_knowledge.terminology.import_planner import (
    TerminologyImportPlan,
    plan_terminology_import,
)
from clinical_knowledge.terminology.import_dry_run import (
    run_terminology_import_dry_run,
)
from clinical_knowledge.terminology.import_audit import (
    TerminologyImportAuditSummary,
)
from clinical_knowledge.terminology.import_transaction import (
    TerminologyImportTransaction,
)
from clinical_knowledge.terminology.import_executor import (
    RealTerminologyImportBlocked,
    TerminologyImportExecutionResult,
    TerminologyImportExecutor,
)
from clinical_knowledge.terminology.qa_golden import (
    TerminologyGoldenCase,
    build_synthetic_qa_store,
    synthetic_golden_cases,
)
from clinical_knowledge.terminology.qa_metrics import (
    TerminologyQAMetrics,
)
from clinical_knowledge.terminology.qa_runner import (
    TerminologyQACaseResult,
    TerminologyQAReport,
    run_synthetic_terminology_qa,
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
    # TERM-01A — operator intake automation
    "ClassificationSummary",
    "FileClassification",
    "classify_filename",
    "classify_filenames",
    "TemplateWriteResult",
    "real_ack_filename",
    "template_filename",
    "template_payload",
    "write_ack_template",
    "CopyResult",
    "ExtractResult",
    "FolderPreparationResult",
    "LocalScanResult",
    "ReadinessReport",
    "compute_readiness",
    "copy_classified_files",
    "optional_local_scan",
    "prepare_intake_folders",
    "safe_extract_zip",
    # TERM-01B dry-run import planner
    "TerminologyImportLimits",
    "build_import_limits",
    "TerminologyImportCheckpoint",
    "simulate_checkpoint_resume",
    "TerminologyImportPlan",
    "plan_terminology_import",
    "run_terminology_import_dry_run",
    # TERM-01C synthetic import executor scaffold
    "TerminologyImportAuditSummary",
    "TerminologyImportTransaction",
    "RealTerminologyImportBlocked",
    "TerminologyImportExecutionResult",
    "TerminologyImportExecutor",
    # TERM-01D terminology QA golden lookup harness
    "TerminologyGoldenCase",
    "build_synthetic_qa_store",
    "synthetic_golden_cases",
    "TerminologyQAMetrics",
    "TerminologyQACaseResult",
    "TerminologyQAReport",
    "run_synthetic_terminology_qa",
]
