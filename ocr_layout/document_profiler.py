from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ocr_layout.text_quality import assess_text_quality


@dataclass(frozen=True)
class DocumentProfile:
    input_type: str
    language: str
    low_text_density: bool
    table_layout_heavy: bool
    empty_or_near_empty: bool
    page_count: int | None
    text_length: int
    warnings: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_document(path: Path | str, text: str, *, extraction_metadata: dict[str, Any] | None = None) -> DocumentProfile:
    source_path = Path(path)
    metadata = extraction_metadata if isinstance(extraction_metadata, dict) else {}
    quality = assess_text_quality(text)
    metrics = dict(quality.metrics)
    text_length = int(metrics.get("text_length", 0))
    page_count = _page_count(metadata)
    input_type = _input_type(source_path=source_path, text_length=text_length, metadata=metadata, page_count=page_count)
    low_text_density = _low_text_density(text_length=text_length, page_count=page_count)
    table_layout_heavy = bool(metrics.get("numeric_table_ratio", 0.0) >= 0.42 or _layout_markers(text) >= 4)
    empty_or_near_empty = quality.band == "empty" or text_length < 25
    warnings = list(quality.warnings)
    if low_text_density and "low_text_density" not in warnings:
        warnings.append("low_text_density")
    if table_layout_heavy and "layout_or_table_heavy" not in warnings:
        warnings.append("layout_or_table_heavy")

    return DocumentProfile(
        input_type=input_type,
        language=str(metrics.get("script", "unknown")),
        low_text_density=low_text_density,
        table_layout_heavy=table_layout_heavy,
        empty_or_near_empty=empty_or_near_empty,
        page_count=page_count,
        text_length=text_length,
        warnings=warnings,
        metrics={
            **metrics,
            "quality_score": quality.score,
            "quality_band": quality.band,
            "layout_marker_count": _layout_markers(text),
        },
    )


def _input_type(*, source_path: Path, text_length: int, metadata: dict[str, Any], page_count: int | None) -> str:
    suffix = source_path.suffix.lower()
    if suffix == ".txt":
        return "text_file"
    if suffix != ".pdf":
        return "unknown"
    ocr_fallback_used = bool(metadata.get("ocr_fallback_used", False))
    page_audits = metadata.get("page_audits") if isinstance(metadata.get("page_audits"), list) else []
    native_lengths = [int(item.get("native_text_length", 0) or 0) for item in page_audits if isinstance(item, dict)]
    native_pages = sum(1 for length in native_lengths if length >= 80)
    ocr_pages = sum(1 for item in page_audits if isinstance(item, dict) and item.get("ocr_fallback_used"))
    if native_pages and ocr_pages:
        return "mixed_pdf"
    if ocr_fallback_used or (page_count and text_length < page_count * 80):
        return "scanned_pdf"
    if text_length >= 80:
        return "digital_pdf"
    return "unknown"


def _page_count(metadata: dict[str, Any]) -> int | None:
    page_audits = metadata.get("page_audits")
    if isinstance(page_audits, list) and page_audits:
        return len(page_audits)
    value = metadata.get("page_count")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _low_text_density(*, text_length: int, page_count: int | None) -> bool:
    if text_length < 120:
        return True
    if page_count and page_count > 0:
        return (text_length / page_count) < 160
    return False


def _layout_markers(text: str) -> int:
    return sum(text.count(marker) for marker in ("|", "\t", "....", "____", "----"))
