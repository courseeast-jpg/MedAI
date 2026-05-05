"""CKA-B05 Medication Safety / DDI Dual-Layer Gate."""
from clinical_knowledge.medication_safety.ddi_stub import check_ddi_stub
from clinical_knowledge.medication_safety.evidence_modifier import apply_ddi_evidence_modifier
from clinical_knowledge.medication_safety.integration import attempt_medication_record_write
from clinical_knowledge.medication_safety.models import (
    DDICheckResult,
    DDICheckStatus,
    DDIFinding,
    DDISeverity,
    Layer1DDIScoreResult,
    MedicationSafetyAction,
    MedicationWriteGateResult,
)
from clinical_knowledge.medication_safety.write_gate import evaluate_medication_write_gate

__all__ = [
    "check_ddi_stub",
    "apply_ddi_evidence_modifier",
    "attempt_medication_record_write",
    "evaluate_medication_write_gate",
    "DDICheckResult",
    "DDICheckStatus",
    "DDIFinding",
    "DDISeverity",
    "Layer1DDIScoreResult",
    "MedicationSafetyAction",
    "MedicationWriteGateResult",
]
