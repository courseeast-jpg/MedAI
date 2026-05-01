from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ocr_layout.document_profiler import profile_document
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
    return candidates


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
