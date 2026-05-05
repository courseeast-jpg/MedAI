from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ocr_layout.ocr_candidates import OcrCandidate


@dataclass(frozen=True)
class OcrRouteDecision:
    route_decision: str
    selected_candidate: OcrCandidate
    selected_text: str
    selected_engine: str
    input_quality_score: float
    input_quality_band: str
    input_quality_warnings: list[str]
    language: str
    candidates: list[OcrCandidate]

    def to_dict(self, *, include_candidate_text: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_candidate_text:
            payload["selected_candidate"]["text"] = ""
            for candidate in payload["candidates"]:
                candidate["text"] = ""
        return payload


def route_ocr_input(candidates: list[OcrCandidate]) -> OcrRouteDecision:
    if not candidates:
        raise ValueError("At least one OCR/input candidate is required")
    selected = max(candidates, key=_candidate_rank)
    route = _route_for(selected)
    return OcrRouteDecision(
        route_decision=route,
        selected_candidate=selected,
        selected_text=selected.text,
        selected_engine=selected.engine_name,
        input_quality_score=selected.quality_score,
        input_quality_band=selected.quality_band,
        input_quality_warnings=list(selected.warnings),
        language=selected.language,
        candidates=candidates,
    )


def _candidate_rank(candidate: OcrCandidate) -> tuple[float, int, int]:
    preferred_engine = 1 if candidate.engine_name in {"existing_pdf_pipeline", "plain_text", "existing_text_source"} else 0
    return (candidate.quality_score, len(candidate.text.strip()), preferred_engine)


def _route_for(candidate: OcrCandidate) -> str:
    profile = candidate.metadata.get("document_profile") if isinstance(candidate.metadata, dict) else {}
    input_type = str(profile.get("input_type", "unknown")) if isinstance(profile, dict) else "unknown"
    if candidate.quality_band == "empty":
        return "empty"
    if candidate.quality_band == "poor_ocr":
        return "poor_ocr"
    if input_type in {"scanned_pdf", "mixed_pdf"} or "low_text_density" in candidate.warnings:
        return "scanned_or_low_text"
    return "digital_clean_text"
