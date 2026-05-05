from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ocr_layout.cyrillic_ocr import try_cyrillic_ocr_candidate
from ocr_layout.document_profiler import profile_document
from ocr_layout.ocr_capabilities import OcrCapabilityReport, get_ocr_capability_report
from ocr_layout.text_quality import assess_text_quality


@dataclass(frozen=True)
class OcrCandidate:
    engine_name: str
    text: str
    quality_score: float
    quality_band: str
    language: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_candidate(
    *,
    engine_name: str,
    text: str,
    source_path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> OcrCandidate:
    quality = assess_text_quality(text)
    profile = profile_document(source_path, text, extraction_metadata=metadata)
    warnings = sorted(set(quality.warnings + profile.warnings))
    return OcrCandidate(
        engine_name=engine_name,
        text=text or "",
        quality_score=quality.score,
        quality_band=quality.band,
        language=profile.language,
        warnings=warnings,
        metadata={
            **(metadata or {}),
            "document_profile": profile.to_dict(),
            "quality_metrics": quality.metrics,
        },
    )


def collect_candidates(path: Path | str, *, existing_text: str | None = None) -> list[OcrCandidate]:
    source_path = Path(path)
    candidates: list[OcrCandidate] = []
    if existing_text is not None:
        candidates.append(
            build_candidate(
                engine_name="existing_text_source",
                text=existing_text,
                source_path=source_path,
                metadata={"source": "provided_text"},
            )
        )
    elif source_path.suffix.lower() == ".txt":
        text = source_path.read_text(encoding="utf-8", errors="replace")
        candidates.append(
            build_candidate(
                engine_name="plain_text",
                text=text,
                source_path=source_path,
                metadata={"source": "plain_text"},
            )
        )
    elif source_path.suffix.lower() == ".pdf":
        candidates.extend(_pdf_candidates(source_path))
    else:
        candidates.append(
            build_candidate(
                engine_name="unsupported",
                text="",
                source_path=source_path,
                metadata={"source": "unsupported_extension"},
            )
        )
    return candidates


def _pdf_candidates(source_path: Path) -> list[OcrCandidate]:
    candidates: list[OcrCandidate] = []
    existing_text, existing_audit = _existing_pdf_pipeline_text(source_path)
    candidates.append(
        build_candidate(
            engine_name="existing_pdf_pipeline",
            text=existing_text,
            source_path=source_path,
            metadata={"source": "PDFPipeline.extract_text_with_audit", **existing_audit},
        )
    )
    pymupdf_text, pymupdf_metadata = _pymupdf_native_text(source_path)
    candidates.append(
        build_candidate(
            engine_name="pymupdf_native_text",
            text=pymupdf_text,
            source_path=source_path,
            metadata=pymupdf_metadata,
        )
    )

    cyrillic_candidate = _maybe_cyrillic_ocr_candidate(source_path, candidates)
    if cyrillic_candidate is not None:
        candidates.append(cyrillic_candidate)
    return candidates


def _maybe_cyrillic_ocr_candidate(
    source_path: Path,
    existing_candidates: list[OcrCandidate],
) -> OcrCandidate | None:
    """Generate a Cyrillic OCR candidate when language hints suggest it.

    Triggered when the best existing candidate text shows Cyrillic markers,
    or when the document classifier flags the source as needing
    language-aware OCR. No-ops when local OCR capability is unavailable.
    """
    if not existing_candidates:
        return None
    best = max(existing_candidates, key=lambda c: c.quality_score)
    if not _should_attempt_cyrillic_ocr(best):
        return None

    capability = get_ocr_capability_report()
    cyrillic_metadata: dict[str, Any] = {
        "trigger_reason": "language_aware_ocr_required",
        "capability": capability.to_dict(),
    }
    if not capability.tesseract_available or not capability.russian_available:
        # Emit a placeholder candidate with empty text so the report shows that
        # we wanted to try Cyrillic OCR but couldn't.
        cyrillic_metadata["cyrillic_ocr_unavailable"] = True
        cyrillic_metadata["candidate_error"] = (
            "tesseract_binary_not_found"
            if not capability.tesseract_available
            else "rus_traineddata_missing"
        )
        return build_candidate(
            engine_name="tesseract_rus_unavailable",
            text="",
            source_path=source_path,
            metadata=cyrillic_metadata,
        )

    result = try_cyrillic_ocr_candidate(source_path, capability=capability)
    cyrillic_metadata.update(result.get("metadata") or {})
    return build_candidate(
        engine_name="tesseract_rus_eng",
        text=result.get("text") or "",
        source_path=source_path,
        metadata={
            **cyrillic_metadata,
            "candidate_warnings": list(result.get("warnings") or []),
        },
    )


def _should_attempt_cyrillic_ocr(candidate: OcrCandidate) -> bool:
    """Heuristic: trigger when Cyrillic indicators appear in existing text.

    Uses two signals so we work for both proper-Cyrillic OCR (cyrillic_ratio
    high) and OCR-mangled-to-Latin homoglyph cases (document classifier
    detects pseudo-Russian markers).
    """
    metrics = (candidate.metadata or {}).get("quality_metrics") or {}
    cyrillic_ratio = float(metrics.get("cyrillic_ratio") or 0.0)
    if cyrillic_ratio >= 0.10:
        return True
    text = candidate.text or ""
    if not text.strip():
        return False
    try:
        from document_classification import classify_document
    except Exception:  # noqa: BLE001
        return False
    classification = classify_document(text)
    return bool(classification.should_recommend_language_aware_ocr)


def _existing_pdf_pipeline_text(source_path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from extraction.extractor import Extractor
        from extraction.pii_stripper import PIIStripper
        from ingestion.pdf_pipeline import PDFPipeline

        pipeline = PDFPipeline(Extractor(), PIIStripper())
        text, audit = pipeline.extract_text_with_audit(source_path)
        return text, dict(audit)
    except Exception as exc:  # noqa: BLE001 - candidate failures are reportable, not fatal.
        return "", {"candidate_error": str(exc), "text_quality_status": "empty", "text_quality_score": 0.0}


def _pymupdf_native_text(source_path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import fitz

        doc = fitz.open(str(source_path))
        pages: list[str] = []
        page_audits: list[dict[str, Any]] = []
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            pages.append(f"[Page {page_number}]\n{page_text}")
            page_audits.append({
                "page": page_number,
                "native_text_length": len(page_text.strip()),
                "ocr_fallback_used": False,
            })
        doc.close()
        return "\n".join(pages), {
            "source": "pymupdf_native_text",
            "ocr_fallback_used": False,
            "ocr_engine": None,
            "page_audits": page_audits,
        }
    except Exception as exc:  # noqa: BLE001 - optional alternative may be unavailable.
        return "", {"source": "pymupdf_native_text", "candidate_error": str(exc)}
