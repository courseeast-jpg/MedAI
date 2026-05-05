"""Phase 44 — Cyrillic OCR candidate generation.

Renders PDF pages to images via PyMuPDF and runs the local Tesseract
binary with ``-l rus+eng`` to produce a Cyrillic-aware text candidate.
Failures (missing binary, missing rus traineddata, render error, timeout)
degrade silently into an empty result with structured warnings.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ocr_layout.ocr_capabilities import OcrCapabilityReport, get_ocr_capability_report


# Cap pages per file to keep batch validation runtimes bounded.
_PAGE_LIMIT_DEFAULT = 8

# Render DPI. 200 is a reasonable default — high enough for OCR accuracy on
# small text, low enough to keep the PNG sizes manageable.
_RENDER_DPI_DEFAULT = 200

# Per-page tesseract timeout
_PER_PAGE_TIMEOUT_SECONDS = 60.0


def try_cyrillic_ocr_candidate(
    source_path: Path | str,
    *,
    capability: OcrCapabilityReport | None = None,
    page_limit: int = _PAGE_LIMIT_DEFAULT,
    dpi: int = _RENDER_DPI_DEFAULT,
    languages: str = "rus+eng",
) -> dict[str, Any]:
    """Attempt a Cyrillic OCR pass on a PDF.

    Returns a dict with keys: ``text``, ``warnings``, ``metadata``.
    Never raises — all failures are reported as warnings.
    """
    path = Path(source_path)
    cap = capability if capability is not None else get_ocr_capability_report()

    metadata: dict[str, Any] = {
        "source": "cyrillic_ocr_candidate",
        "engine": "tesseract",
        "requested_languages": languages,
        "tesseract_available": cap.tesseract_available,
        "russian_available": cap.russian_available,
        "page_limit": page_limit,
        "render_dpi": dpi,
        "pages_processed": 0,
        "pages_skipped": 0,
        "ocr_attempted": False,
    }
    warnings: list[str] = []

    if not path.exists():
        warnings.append("cyrillic_ocr_source_not_found")
        return {"text": "", "warnings": warnings, "metadata": metadata}
    if path.suffix.lower() != ".pdf":
        warnings.append("cyrillic_ocr_unsupported_extension")
        return {"text": "", "warnings": warnings, "metadata": metadata}

    if not cap.tesseract_available:
        warnings.append("cyrillic_ocr_unavailable")
        warnings.append("tesseract_binary_not_found")
        return {"text": "", "warnings": warnings, "metadata": metadata}
    if not cap.russian_available:
        warnings.append("cyrillic_ocr_unavailable")
        warnings.append("rus_traineddata_missing")
        return {"text": "", "warnings": warnings, "metadata": metadata}

    try:
        import fitz  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        warnings.append("pymupdf_unavailable")
        metadata["render_error"] = f"{exc.__class__.__name__}:{exc}"
        return {"text": "", "warnings": warnings, "metadata": metadata}

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        warnings.append("pdf_open_failed")
        metadata["open_error"] = f"{exc.__class__.__name__}:{exc}"
        return {"text": "", "warnings": warnings, "metadata": metadata}

    metadata["ocr_attempted"] = True
    page_texts: list[str] = []
    per_page: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="phase44_cyrillic_") as tmp:
            tmp_dir = Path(tmp)
            pages_to_process = min(len(doc), page_limit)
            metadata["page_count_total"] = len(doc)
            metadata["page_count_to_process"] = pages_to_process
            for index in range(pages_to_process):
                page = doc[index]
                page_audit: dict[str, Any] = {"page": index + 1}
                try:
                    pix = page.get_pixmap(dpi=dpi, alpha=False)
                    image_path = tmp_dir / f"page_{index + 1:03d}.png"
                    pix.save(str(image_path))
                except Exception as exc:  # noqa: BLE001
                    page_audit["render_error"] = f"{exc.__class__.__name__}:{exc}"
                    per_page.append(page_audit)
                    metadata["pages_skipped"] += 1
                    continue

                page_text, page_warnings, page_audit_run = _run_tesseract_on_image(
                    binary_path=cap.tesseract_path or "tesseract",
                    image_path=image_path,
                    languages=languages,
                )
                page_audit.update(page_audit_run)
                if page_warnings:
                    page_audit["warnings"] = page_warnings
                    for w in page_warnings:
                        if w not in warnings:
                            warnings.append(w)
                if page_text:
                    page_texts.append(f"[Page {index + 1}]\n{page_text}")
                    metadata["pages_processed"] += 1
                else:
                    metadata["pages_skipped"] += 1
                per_page.append(page_audit)
            if len(doc) > page_limit:
                warnings.append("cyrillic_ocr_page_limit_truncated")
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass

    metadata["per_page"] = per_page
    text = "\n".join(page_texts).strip()
    if not text:
        warnings.append("cyrillic_ocr_empty_output")
    return {"text": text, "warnings": warnings, "metadata": metadata}


def _run_tesseract_on_image(
    *,
    binary_path: str,
    image_path: Path,
    languages: str,
) -> tuple[str, list[str], dict[str, Any]]:
    output_stem = image_path.with_suffix("")
    try:
        result = subprocess.run(
            [binary_path, str(image_path), str(output_stem), "-l", languages],
            capture_output=True,
            text=True,
            timeout=_PER_PAGE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", ["cyrillic_ocr_timeout"], {"timeout": True}
    except (FileNotFoundError, OSError) as exc:
        return "", ["cyrillic_ocr_invocation_error"], {"error": f"{exc.__class__.__name__}"}

    output_txt = Path(str(output_stem) + ".txt")
    if not output_txt.exists():
        warnings = ["cyrillic_ocr_output_missing"]
        if result.returncode != 0:
            warnings.append(f"tesseract_returncode_{result.returncode}")
        return "", warnings, {"returncode": result.returncode}

    try:
        text = output_txt.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return "", ["cyrillic_ocr_output_read_error"], {"read_error": str(exc)}

    audit = {
        "returncode": result.returncode,
        "output_chars": len(text),
    }
    return text, [], audit
