"""CKA-B04 Truth Resolution Engine — deterministic conflict detection and resolution."""
from clinical_knowledge.truth_resolution.engine import apply_truth_resolution, resolve_conflict
from clinical_knowledge.truth_resolution.models import (
    ConflictPair,
    ConflictType,
    ResolutionAction,
    ResolutionRule,
    TruthResolutionResult,
)

__all__ = [
    "apply_truth_resolution",
    "resolve_conflict",
    "ConflictPair",
    "ConflictType",
    "ResolutionAction",
    "ResolutionRule",
    "TruthResolutionResult",
]
