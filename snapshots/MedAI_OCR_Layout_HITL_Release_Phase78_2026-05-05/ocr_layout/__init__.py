"""Non-destructive OCR/Layout profiling layer for MedAI validation."""

from ocr_layout.document_profiler import DocumentProfile, profile_document
from ocr_layout.ocr_candidates import OcrCandidate, collect_candidates
from ocr_layout.image_ocr import ImageOcrResult, SUPPORTED_IMAGE_EXTENSIONS, extract_image_text
from ocr_layout.ocr_router import OcrRouteDecision, route_ocr_input
from ocr_layout.text_quality import TextQualityAssessment, assess_text_quality

__all__ = [
    "DocumentProfile",
    "OcrCandidate",
    "OcrRouteDecision",
    "ImageOcrResult",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "TextQualityAssessment",
    "assess_text_quality",
    "collect_candidates",
    "extract_image_text",
    "profile_document",
    "route_ocr_input",
]
