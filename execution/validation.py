"""Phase 2 extraction validation and review classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import (
    ALLOWED_FACT_TYPES,
    CONSENSUS_ACCEPT_THRESHOLD,
    CONSENSUS_REVIEW_THRESHOLD,
    EXTRACTION_ACCEPT_THRESHOLD,
    EXTRACTION_REVIEW_THRESHOLD,
)


@dataclass(frozen=True)
class ValidationDecision:
    status: str
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def validate_extraction_result(extracted: dict[str, Any], *, extractor_route: str) -> ValidationDecision:
    """Validate Phase 2 extraction payloads and classify them for handling."""

    errors: list[dict[str, Any]] = []
    confidence = float(extracted.get("confidence", 0.0))
    actual_extractor = str(extracted.get("actual_extractor", extracted.get("extractor", "")))
    agreement_score = float(extracted.get("agreement_score", 1.0))
    disagreement_flag = bool(extracted.get("disagreement_flag", False))
    fallback_used = bool(extracted.get("fallback_used", False))

    if actual_extractor != extractor_route:
        errors.append(_issue(
            code="route_actual_mismatch",
            severity="warning" if fallback_used else "fatal",
            message="Extractor route does not match the actual extractor used."
            if not fallback_used
            else "Extractor route fell back to a different connector before validation.",
            field="actual_extractor",
            expected=extractor_route,
            actual=actual_extractor,
        ))

    _validate_entities(extracted.get("entities", []), errors)
    _validate_consensus(agreement_score, disagreement_flag, errors)

    if confidence < EXTRACTION_REVIEW_THRESHOLD:
        errors.append(_issue(
            code="confidence_below_reject_threshold",
            severity="fatal",
            message="Extraction confidence is below the rejection threshold.",
            field="confidence",
            expected=f">={EXTRACTION_REVIEW_THRESHOLD}",
            actual=confidence,
        ))
    elif confidence < EXTRACTION_ACCEPT_THRESHOLD:
        errors.append(_issue(
            code="confidence_below_accept_threshold",
            severity="warning",
            message="Extraction confidence is below the accepted threshold and requires review.",
            field="confidence",
            expected=f">={EXTRACTION_ACCEPT_THRESHOLD}",
            actual=confidence,
        ))

    if any(error["severity"] == "fatal" for error in errors):
        return ValidationDecision(status="rejected", errors=errors)
    if errors:
        return ValidationDecision(status="needs_review", errors=errors)
    return ValidationDecision(status="accepted", errors=[])


def _validate_consensus(agreement_score: float, disagreement_flag: bool, errors: list[dict[str, Any]]) -> None:
    if agreement_score < CONSENSUS_REVIEW_THRESHOLD:
        errors.append(_issue(
            code="agreement_below_reject_threshold",
            severity="fatal",
            message="Consensus agreement is below the rejection threshold.",
            field="agreement_score",
            expected=f">={CONSENSUS_REVIEW_THRESHOLD}",
            actual=agreement_score,
        ))
    elif agreement_score < CONSENSUS_ACCEPT_THRESHOLD:
        errors.append(_issue(
            code="agreement_below_accept_threshold",
            severity="warning",
            message="Consensus agreement is below the accepted threshold and requires review.",
            field="agreement_score",
            expected=f">={CONSENSUS_ACCEPT_THRESHOLD}",
            actual=agreement_score,
        ))

    if disagreement_flag:
        errors.append(_issue(
            code="consensus_disagreement_flag",
            severity="warning",
            message="Consensus merge detected disagreement across extractor outputs.",
            field="disagreement_flag",
            expected=False,
            actual=True,
        ))


def _validate_entities(entities: list[Any], errors: list[dict[str, Any]]) -> None:
    if not entities:
        errors.append(_issue(
            code="empty_extraction",
            severity="fatal",
            message="Extraction produced no entities.",
            field="entities",
            expected="non-empty list",
            actual=entities,
        ))
        return

    for index, entity in enumerate(entities):
        entity_path = f"entities[{index}]"
        if not isinstance(entity, dict):
            errors.append(_issue(
                code="entity_not_object",
                severity="fatal",
                message="Entity entries must be JSON objects.",
                field=entity_path,
                expected="object",
                actual=type(entity).__name__,
            ))
            continue

        entity_type = entity.get("type")
        if not isinstance(entity_type, str) or not entity_type.strip():
            errors.append(_issue(
                code="missing_required_field",
                severity="fatal",
                message="Entity type is required.",
                field=f"{entity_path}.type",
                expected="non-empty string",
                actual=entity_type,
            ))
        elif entity_type not in ALLOWED_FACT_TYPES:
            errors.append(_issue(
                code="invalid_entity_type",
                severity="fatal",
                message="Entity type is not supported by the pipeline.",
                field=f"{entity_path}.type",
                expected=sorted(ALLOWED_FACT_TYPES),
                actual=entity_type,
            ))

        text = entity.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(_issue(
                code="missing_required_field",
                severity="fatal",
                message="Entity text is required.",
                field=f"{entity_path}.text",
                expected="non-empty string",
                actual=text,
            ))

        if "structured" in entity and entity["structured"] is not None and not isinstance(entity["structured"], dict):
            errors.append(_issue(
                code="invalid_schema_shape",
                severity="fatal",
                message="Entity structured payload must be an object when provided.",
                field=f"{entity_path}.structured",
                expected="object",
                actual=type(entity["structured"]).__name__,
            ))


def _issue(
    *,
    code: str,
    severity: str,
    message: str,
    field: str,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "field": field,
        "expected": expected,
        "actual": actual,
    }
