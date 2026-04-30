from __future__ import annotations

from dataclasses import asdict, dataclass


HIGH_CONFIDENCE_THRESHOLD = 0.85
ACCEPTABLE_CONFIDENCE_THRESHOLD = 0.65
REVIEW_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ConfidenceCalibration:
    raw_confidence: float
    calibrated_confidence: float
    confidence_band: str
    calibration_reason: str
    extractor_route: str
    extractor_actual: str
    route_mismatch_flag: bool
    review_recommendation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_confidence(
    *,
    raw_confidence: float,
    extractor_route: str | None,
    extractor_actual: str | None,
    requested_extractor_route: str | None,
    fallback_used: bool = False,
) -> ConfidenceCalibration:
    raw_value = _normalize_confidence(raw_confidence)
    route_value = str(extractor_route or "unknown")
    actual_value = str(extractor_actual or "unknown")
    requested_value = str(requested_extractor_route or route_value)
    route_mismatch_flag = requested_value not in {"", "unknown"} and requested_value != actual_value

    reason_parts = ["raw_confidence_retained"]
    if route_mismatch_flag:
        reason_parts.append("requested_route_mismatch_observed")
    if fallback_used:
        reason_parts.append("fallback_connector_used")

    calibrated_value = round(raw_value, 3)
    confidence_band = classify_confidence_band(calibrated_value)
    review_recommendation = build_review_recommendation(
        confidence_band=confidence_band,
        route_mismatch_flag=route_mismatch_flag,
    )

    return ConfidenceCalibration(
        raw_confidence=raw_value,
        calibrated_confidence=calibrated_value,
        confidence_band=confidence_band,
        calibration_reason=",".join(reason_parts),
        extractor_route=route_value,
        extractor_actual=actual_value,
        route_mismatch_flag=route_mismatch_flag,
        review_recommendation=review_recommendation,
    )


def classify_confidence_band(confidence: float) -> str:
    value = _normalize_confidence(confidence)
    if value >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if value >= ACCEPTABLE_CONFIDENCE_THRESHOLD:
        return "acceptable"
    if value >= REVIEW_CONFIDENCE_THRESHOLD:
        return "review"
    return "reject"


def build_review_recommendation(*, confidence_band: str, route_mismatch_flag: bool) -> str:
    if confidence_band == "reject":
        return "reject_do_not_write"
    if confidence_band == "review":
        return "operator_review"
    if route_mismatch_flag:
        return "accept_with_route_audit"
    return "accept"


def _normalize_confidence(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)
